import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from .util import AverageMeter, accuracy


def train_distill_taylor(epoch, train_loader, module_list, criterion_list, optimizer, opt,
                         mask_list, sensitivity_list, importance_tracker, mask_updater,
                         pruning_scheduler, attacker=None):
    """基于泰勒展开的知识蒸馏训练"""

    # 设置训练模式
    for module in module_list:
        module.train()

    # 获取模型和损失函数
    model_s = module_list[0]  # 学生模型
    model_t = module_list[-1]  # 教师模型
    model_t.eval()  # 教师模型保持评估模式

    criterion_cls = criterion_list[0]  # 分类损失
    criterion_div = criterion_list[1]  # KL散度损失
    criterion_kd = criterion_list[2]  # 知识蒸馏损失

    # 初始化计量器
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 掩码更新计数器
    update_frequency = opt.mask_update_frequency if hasattr(opt, 'mask_update_frequency') else 100

    end = time.time()

    for idx, data in enumerate(train_loader):
        if attacker:
            if len(data) == 3:
                input, adv_input, target = data
            else:
                input, target = data
                adv_input = attacker.attack(model_s, input, target, mask_list, opt.kd_T_adv)
        else:
            if len(data) == 3:
                input, target, index = data
            else:
                input, target = data

        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if attacker:
                adv_input = adv_input.cuda()

        # ===================前向传播===================
        features = []
        mask_this = mask_list.copy()

        # 学生模型前向传播（带重要性跟踪）
        output_s, features = model_s(input, mask_this, features, is_feat=False,
                                     importance_tracker=importance_tracker)

        # 教师模型前向传播
        features_t = []
        with torch.no_grad():
            output_t, features_t = model_t(input, features_t, is_feat=False)

        # ===================计算损失===================
        # 分类损失
        loss_cls = criterion_cls(output_s, target)

        # KL散度损失
        loss_div = criterion_div(output_s, output_t)

        # 总损失
        loss = opt.gamma * loss_cls + opt.alpha * loss_div

        # ===================反向传播===================
        optimizer.zero_grad()
        loss.backward()

        # 更新重要性分数（每个batch都更新）
        importance_tracker.update_importance_scores()

        # 梯度裁剪（可选）
        if hasattr(opt, 'grad_clip'):
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), opt.grad_clip)

        optimizer.step()

        # ===================更新掩码（按频率）===================
        if (idx + 1) % update_frequency == 0 and pruning_scheduler.should_prune(epoch, idx):
            # 基于累积的重要性分数更新掩码
            mask_list = mask_updater.update_masks_layerwise(
                importance_tracker, pruning_scheduler, epoch,
                mask_list, opt.size_list, opt.channel_size
            )
            # 将更新后的掩码移到GPU
            mask_list = [mask.cuda() for mask in mask_list]

            # 打印剪枝统计信息
            if opt.local_rank == 0 and (idx + 1) % (update_frequency * 10) == 0:
                sparsity_per_layer = []
                for layer_idx, mask in enumerate(mask_list):
                    sparsity = 1 - mask.mean().item()
                    sparsity_per_layer.append(sparsity)
                print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}] - Layer sparsities: {sparsity_per_layer}')

        # 清除缓存的激活值和梯度
        importance_tracker.clear_cache()

        # ===================记录统计信息===================
        losses.update(loss.item(), input.size(0))
        metrics = accuracy(output_s, target, topk=(1, 5))
        top1.update(metrics[0].item(), input.size(0))
        top5.update(metrics[1].item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # ===================打印信息===================
        if idx % opt.print_freq == 0 and opt.local_rank == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    # 返回统计信息和更新后的掩码
    return top1.avg, None, losses.avg, mask_list


def validate_taylor(val_loader, model, criterion, opt, mask_list, importance_tracker=None, attacker=None):
    """验证函数（支持泰勒剪枝）"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # 切换到评估模式
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # 前向传播
            features = []
            mask_this = mask_list.copy()

            # 不需要重要性跟踪的验证
            output, features = model(input, mask_this, features, is_feat=False,
                                     importance_tracker=None)

            loss = criterion(output, target)

            # 度量准确率
            metrics = accuracy(output, target, topk=(1, 5))
            top1.update(metrics[0].item(), input.size(0))
            top5.update(metrics[1].item(), input.size(0))
            losses.update(loss.item(), input.size(0))

            # 时间度量
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, None, losses.avg, mask_list


def compute_pruning_stats(mask_list, sensitivity_list):
    """计算剪枝统计信息"""
    stats = {
        'layer_sparsity': [],
        'total_pruned': 0,
        'total_params': 0,
        'effective_sparsity': []
    }

    for layer_idx, mask in enumerate(mask_list):
        # 基础稀疏度
        sparsity = 1 - mask.mean().item()
        stats['layer_sparsity'].append(sparsity)

        # 考虑sensitivity的有效稀疏度
        effective_sparsity = 1 - (mask.mean().item() * sensitivity_list[layer_idx])
        stats['effective_sparsity'].append(effective_sparsity)

        # 统计总数
        num_params = mask.numel()
        num_pruned = int(num_params * sparsity)
        stats['total_params'] += num_params
        stats['total_pruned'] += num_pruned

    stats['overall_sparsity'] = stats['total_pruned'] / stats['total_params']

    return stats