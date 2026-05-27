# 导入基础库
from __future__ import print_function
import os


import gc
import numpy as np
# 深度学习相关库

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys

from TXTlogger import DualWriter
from utils import parse_options, get_teacher_name, load_dataset, load_teacher, AverageMeter, accuracy

# 自定义模块
from models import model_dict

from helper.util import adjust_learning_rate

# 知识蒸馏损失函数
from distiller_zoo import DistillKL

# 训练/验证循环
from helper.loops import validate

# 特征操作工具
from einops import rearrange, reduce, repeat

import time
from collections import deque

class LayerImportanceCalculator:
    """计算每一层的重要性权重，考虑深度先验（优化版本）"""

    def __init__(self, momentum=0.95, depth_bias=0.1):
        self.momentum = momentum
        self.depth_bias = depth_bias
        self.layer_gradients = {}
        self.layer_fisher = {}
        self.layer_importance = {}
        self.total_layers = 0

    @torch.no_grad()
    def update_layer_statistics(self, layer_idx, features, gradients):
        """更新层级统计信息（优化：减少CPU传输）"""
        self.total_layers = max(self.total_layers, layer_idx + 1)

        # 在GPU上计算所有统计量
        grad_norm = torch.norm(gradients.reshape(gradients.size(0), -1), p=2, dim=1).mean()
        feature_var = features.var(dim=[0, 2, 3]).mean()
        fisher_approx = (gradients ** 2).mean()

        # 批量更新统计（减少item()调用）
        if layer_idx not in self.layer_gradients:
            self.layer_gradients[layer_idx] = 0
            self.layer_fisher[layer_idx] = 0

        # 延迟转换到CPU
        grad_norm_val = grad_norm.cpu().item()
        fisher_approx_val = fisher_approx.cpu().item()

        self.layer_gradients[layer_idx] = (
                self.momentum * self.layer_gradients[layer_idx] +
                (1 - self.momentum) * grad_norm_val
        )

        self.layer_fisher[layer_idx] = (
                self.momentum * self.layer_fisher[layer_idx] +
                (1 - self.momentum) * fisher_approx_val
        )

        base_importance = (self.layer_gradients[layer_idx] *
                           np.sqrt(self.layer_fisher[layer_idx] + 1e-8))

        depth_factor = 1.0 + self.depth_bias * (layer_idx / max(1, self.total_layers - 1))

        self.layer_importance[layer_idx] = base_importance * depth_factor

    def get_layer_importance_weights(self):
        """获取归一化的层重要性权重"""
        if not self.layer_importance:
            return {}

        values = list(self.layer_importance.values())
        min_val = min(values)
        max_val = max(values)

        if max_val - min_val > 0:
            normalized_importance = {
                k: (v - min_val) / (max_val - min_val) + 0.1
                for k, v in self.layer_importance.items()
            }
        else:
            normalized_importance = {k: 1.0 for k in self.layer_importance}

        return normalized_importance


class GlobalAdaptiveTaylorImportanceCalculator:
    """全局自适应的泰勒重要性计算器（优化版本）"""

    def __init__(self, momentum=0.9, use_second_order=True, layer_calculator=None,
                 activation_history_size=10):
        self.momentum = momentum
        self.use_second_order = use_second_order
        self.layer_calculator = layer_calculator
        self.activation_history_size = activation_history_size

        # 使用tensor存储重要性，避免频繁的CPU-GPU传输
        self.importance_tensors = {}  # {layer_idx: tensor(C, H, W)}
        # 存储激活历史，用于累积考虑

        self.activation_history = {}  # {layer_idx: deque}
        self.fisher_diag = {}

    @torch.no_grad()
    def update_activation_history(self, pre_relu_activation, layer_idx):
        """更新激活历史（优化内存使用）"""
        if layer_idx not in self.activation_history:
            self.activation_history[layer_idx] = deque(maxlen=self.activation_history_size)

        # 只存储压缩后的统计信息，而不是完整的tensor
        negative_mask = (pre_relu_activation < 0).float()
        # 在batch维度平均，减少存储
        neg_prob = negative_mask.mean(dim=0)  # (C, H, W)

        self.activation_history[layer_idx].append(neg_prob)

        # 计算累积概率
        if len(self.activation_history[layer_idx]) > 0:
            # 计算历史上该位置为负的概率
            history_stack = torch.stack(list(self.activation_history[layer_idx]), dim=0)
            accumulated_mask = history_stack.mean(dim=0)
            return accumulated_mask
        else:
            return neg_prob

    def compute_position_importance(self, pre_relu_activation, post_relu_gradient, layer_idx):
        """计算每个ReLU位置的重要性（优化版本）"""
        with torch.no_grad():
            # 更新激活历史
            accumulated_negative_mask = self.update_activation_history(pre_relu_activation, layer_idx)

            # 当前的负激活掩码
            current_negative_mask = (pre_relu_activation < 0).float()

            # 结合历史信息：考虑历史上该位置为负的概率
            # 权重 = 当前负激活 * 历史负激活概率

            # 在batch维度平均
            current_neg_avg = current_negative_mask.mean(dim=0)

            # 结合历史信息
            combined_mask = current_neg_avg * accumulated_negative_mask

            # 一阶泰勒项（批量计算）
            first_order = torch.abs(post_relu_gradient * pre_relu_activation) * accumulated_negative_mask
            first_order_spatial = first_order.mean(dim=0)  # (C, H, W)

            # 二阶项
            if self.use_second_order:
                if layer_idx not in self.fisher_diag:
                    self.fisher_diag[layer_idx] = torch.zeros_like(first_order_spatial)

                grad_squared = (post_relu_gradient ** 2).mean(dim=0)
                self.fisher_diag[layer_idx] = (
                        0.99 * self.fisher_diag[layer_idx] +
                        0.01 * grad_squared
                )

                activation_squared = (pre_relu_activation ** 2).mean(dim=0)
                second_order = 0.5 * activation_squared * self.fisher_diag[layer_idx]
                # importance_spatial = first_order_spatial + 0.1 * second_order
                importance_spatial = second_order + first_order_spatial
            else:
                importance_spatial = first_order_spatial

            # 应用层权重
            if self.layer_calculator is not None:
                layer_weights = self.layer_calculator.get_layer_importance_weights()
                layer_weight = layer_weights.get(layer_idx, 1.0)
                importance_spatial = importance_spatial * layer_weight

        # 更新层级统计
        if self.layer_calculator is not None:
            features = F.relu(pre_relu_activation)
            self.layer_calculator.update_layer_statistics(
                layer_idx, features, post_relu_gradient
            )

        return importance_spatial

    @torch.no_grad()
    def update_importance_tensor(self, importance_map, layer_idx):
        """直接在GPU上更新重要性tensor"""
        if layer_idx not in self.importance_tensors:
            self.importance_tensors[layer_idx] = torch.zeros_like(importance_map)

        # EMA更新，完全在GPU上进行
        self.importance_tensors[layer_idx] = (
                self.momentum * self.importance_tensors[layer_idx] +
                (1 - self.momentum) * importance_map
        )

    def get_global_importance_masks(self, feature_shapes, keep_ratio):
        """直接生成掩码，避免CPU排序（优化版本）"""
        # 收集所有重要性值到一个大tensor
        all_importance_values = []
        layer_info = []  # 记录每个值对应的层和位置

        for layer_idx, importance_tensor in self.importance_tensors.items():
            if layer_idx in feature_shapes:
                # Flatten重要性tensor
                flat_importance = importance_tensor.flatten()
                all_importance_values.append(flat_importance)

                # 记录层信息
                layer_info.append({
                    'layer_idx': layer_idx,
                    'shape': importance_tensor.shape,
                    'start_idx': len(all_importance_values) - 1
                })

        if not all_importance_values:
            return {}

        # 在GPU上拼接所有重要性值
        global_importance = torch.cat(all_importance_values)

        # 计算需要保留的数量
        total_elements = global_importance.numel()
        num_keep = max(1, int(total_elements * keep_ratio))

        # 使用topk在GPU上找出最重要的位置
        _, top_indices = torch.topk(global_importance, num_keep)

        # 创建全局掩码
        global_mask = torch.zeros_like(global_importance)
        global_mask[top_indices] = 1.0

        # 将全局掩码分配回各层
        masks = {}
        current_idx = 0
        for info in layer_info:
            layer_idx = info['layer_idx']
            shape = info['shape']
            layer_size = shape[0] * shape[1] * shape[2]

            # 提取该层的掩码部分
            layer_mask_flat = global_mask[current_idx:current_idx + layer_size]
            # 重塑为原始形状
            masks[layer_idx] = layer_mask_flat.reshape(shape)

            current_idx += layer_size

        return masks


class GlobalProgressiveMaskGenerator:
    """全局渐进式掩码生成器（优化版本）"""

    def __init__(self, global_keep_ratio=0.5, update_interval=100,
                 warmup_epochs=5, target_epoch_ratio=0.6, total_epochs=100):
        self.global_keep_ratio = global_keep_ratio
        self.update_interval = update_interval
        self.warmup_epochs = warmup_epochs
        self.target_epoch_ratio = target_epoch_ratio
        self.total_epochs = total_epochs
        self.batch_counter = 0
        self.current_epoch = 0

        self.target_epoch = int(self.warmup_epochs +
                                (self.total_epochs - self.warmup_epochs) * self.target_epoch_ratio)

        self.layer_keep_stats = {}

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def should_update_mask(self):
        self.batch_counter += 1
        if self.current_epoch < self.warmup_epochs:
            return False
        if self.current_epoch >= self.target_epoch:
            return self.batch_counter % (self.update_interval * 2) == 0
        return self.batch_counter % self.update_interval == 0

    def get_current_global_keep_ratio(self):
        """获取当前的全局保留比例"""
        if self.current_epoch < self.warmup_epochs:
            return 1.0

        if self.current_epoch >= self.target_epoch:
            return self.global_keep_ratio

        effective_epochs = self.target_epoch - self.warmup_epochs
        progress = (self.current_epoch - self.warmup_epochs) / effective_epochs
        progress = min(1.0, max(0.0, progress))

        accelerated_progress = progress ** 1.5
        cosine_decay = 0.5 * (1 + np.cos(np.pi * accelerated_progress))
        current_ratio = self.global_keep_ratio + (1.0 - self.global_keep_ratio) * cosine_decay

        return current_ratio

    @torch.no_grad()
    def update_stats_from_masks(self, masks):
        """从掩码更新统计信息（批量计算）"""
        self.layer_keep_stats = {}

        for layer_idx, mask in masks.items():
            # 批量计算统计信息
            keep_ratio = mask.mean().item()
            kept_positions = int(mask.sum().item())
            total_positions = mask.numel()

            # 计算每个通道的保留比例
            channel_keep_ratios = mask.mean(dim=[1, 2]).cpu().numpy()

            self.layer_keep_stats[layer_idx] = {
                'keep_ratio': keep_ratio,
                'kept_positions': kept_positions,
                'total_positions': total_positions,
                'channel_keep_ratios': channel_keep_ratios
            }

    def print_layer_statistics(self):
        """打印每层的剪枝统计"""
        if not self.layer_keep_stats:
            return

        print("\n层级剪枝统计:")
        print(f"{'Layer':<8} {'Keep Ratio':<12} {'Kept/Total':<15} {'Min/Max Channel Keep':<20}")
        print("-" * 60)

        total_kept = 0
        total_positions = 0

        for layer_idx in sorted(self.layer_keep_stats.keys()):
            stats = self.layer_keep_stats[layer_idx]
            keep_ratio = stats['keep_ratio']
            kept = stats['kept_positions']
            total = stats['total_positions']
            channel_ratios = stats['channel_keep_ratios']

            min_channel_keep = channel_ratios.min()
            max_channel_keep = channel_ratios.max()

            total_kept += kept
            total_positions += total

            print(f"{layer_idx:<8} {keep_ratio:<12.3f} {kept}/{total:<15} "
                  f"{min_channel_keep:.3f}/{max_channel_keep:.3f}")

        global_actual_ratio = total_kept / total_positions if total_positions > 0 else 0
        print("-" * 60)
        print(f"全局保留比例: {global_actual_ratio:.3f} (目标: {self.get_current_global_keep_ratio():.3f})")

def forward_with_features(model, x, mask_list):
    """修改的前向传播，收集pre-relu和post-relu特征"""
    features = []

    if hasattr(model, 'forward') and 'collect_pre_post' in model.forward.__code__.co_varnames:
        _, logit, features, pre_relu_features, post_relu_features = model(
            x, mask_list, features, is_feat=True, collect_pre_post=True
        )

        pre_relu_features_indexed = [(i, feat) for i, feat in enumerate(pre_relu_features)]
        post_relu_features_indexed = []
        for i, feat in enumerate(post_relu_features):
            feat_with_grad = feat.requires_grad_(True)
            feat_with_grad.retain_grad()
            post_relu_features_indexed.append((i, feat_with_grad))

        return logit, pre_relu_features_indexed, post_relu_features_indexed
    else:
        raise ValueError("Model does not support collect_pre_post parameter")


def train_global_adaptive_taylor_distill(epoch, train_loader, module_list, criterion_list,
                                         optimizer, opt, mask_list, global_keep_ratio,
                                         importance_calculator, mask_generator):
    """使用全局自适应泰勒重要性的知识蒸馏训练（优化版本）"""

    mask_generator.set_epoch(epoch)

    model_s = module_list[0]
    model_t = module_list[-1]

    model_s.train()
    model_t.eval()

    criterion_cls = criterion_list[0]
    criterion_kd = criterion_list[1]

    losses = AverageMeter()
    cls_losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()

    feature_shapes = {}



    for idx, (input, target, _) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        batch_size = input.shape[0]

        batch_masks = []
        for mask in mask_list:
            if mask.dim() == 3:
                batch_mask = mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
            else:
                batch_mask = mask
            batch_masks.append(batch_mask)

        # 前向传播
        logit_s, pre_relu_features_s, post_relu_features_s = forward_with_features(
            model_s, input, batch_masks
        )

        with torch.no_grad():
            features_t = []
            _, logit_t, features_t = model_t(input, features_t, is_feat=True)

        # 记录特征形状
        if not feature_shapes:
            for layer_idx, (_, feat) in enumerate(post_relu_features_s):
                if feat is not None:
                    feature_shapes[layer_idx] = (feat.shape[1], feat.shape[2], feat.shape[3])

        # 计算损失
        loss_cls = criterion_cls(logit_s, target)
        loss_kd = criterion_kd(logit_s, logit_t)
        loss = opt.gamma * loss_cls + opt.alpha * loss_kd

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 批量计算重要性
        with torch.no_grad():
            for layer_idx, pre_relu_act in pre_relu_features_s:
                post_relu_feat = None
                for idx_post, feat_post in post_relu_features_s:
                    if idx_post == layer_idx:
                        post_relu_feat = feat_post
                        break

                if post_relu_feat is not None and post_relu_feat.grad is not None:
                    importance_map = importance_calculator.compute_position_importance(
                        pre_relu_act, post_relu_feat.grad, layer_idx
                    )
                    importance_calculator.update_importance_tensor(importance_map, layer_idx)

        optimizer.step()

        # 更新掩码
        if mask_generator.should_update_mask():
            current_ratio = mask_generator.get_current_global_keep_ratio()

            # 直接在GPU上生成掩码
            new_masks_dict = importance_calculator.get_global_importance_masks(
                feature_shapes, current_ratio
            )

            if new_masks_dict:
                # 更新掩码列表
                mask_list = [new_masks_dict[i] if i in new_masks_dict else mask_list[i]
                             for i in range(len(mask_list))]

                # 更新统计信息
                mask_generator.update_stats_from_masks(new_masks_dict)

                # 清空batch masks缓存，强制下次重新创建
                batch_masks_cache = None

                if idx % opt.print_freq == 0:
                    mask_generator.print_layer_statistics()

        # 更新统计
        acc1, _ = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        cls_losses.update(loss_cls.item(), input.size(0))
        kd_losses.update(loss_kd.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        # 打印进度
        if idx % opt.print_freq == 0:
            current_ratio = mask_generator.get_current_global_keep_ratio()
            print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Cls {cls_losses.val:.4f} ({cls_losses.avg:.4f})\t'
                  f'KD {kd_losses.val:.4f} ({kd_losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Global Keep: {current_ratio:.3f}')

        # 减少GPU缓存清理频率
        if idx % 100 == 0:
            torch.cuda.empty_cache()

    return top1.avg, losses.avg, mask_list\

def main():
    """主函数"""
    opt = parse_options('layer')

    # 优化数据加载
    if not hasattr(opt, 'num_workers'):
        opt.num_workers = 4  # 增加数据加载线程数

    # 启用pin memory加速数据传输
    if not hasattr(opt, 'pin_memory'):
        opt.pin_memory = True

    # 设置全局保留比例
    if hasattr(opt, 'global_keep_ratio'):
        global_keep_ratio = opt.global_keep_ratio
    else:
        global_keep_ratio = 0.5

    print(f"全局ReLU保留比例目标: {global_keep_ratio}")

    # 日志设置
    save_file = os.path.join(opt.save_folder, 'running_log.txt')
    log_file = open(save_file, 'w')
    sys.stdout = DualWriter(sys.stdout, log_file)

    # 基础设置
    opt.distributed = False
    opt.device = 'cuda:0'

    # 数据加载（确保使用优化的参数）
    train_loader, val_loader, n_data, n_cls = load_dataset(opt)

    # 模型初始化
    model_t = load_teacher(opt, opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)
    print(opt.model_s)

    if opt.pretrain_load:
        model_s.load_state_dict(torch.load(opt.path_t)['model'])

    module_list = nn.ModuleList([model_s, model_t])

    criterion_cls = nn.CrossEntropyLoss()
    criterion_kd = DistillKL(opt.kd_T)
    criterion_list = nn.ModuleList([criterion_cls, criterion_kd])

    optimizer = optim.SGD(
        model_s.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay
    )

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # 教师模型验证
    teacher_acc, teacher_acc_robust, _, _ = validate(
        val_loader, model_t, criterion_cls, opt, attacker=None
    )
    print('教师模型准确率:', teacher_acc)

    # 获取特征图尺寸
    if opt.dataset == 'imagenet':
        dummy_input = torch.randn(2, 3, 224, 224).cuda()
    elif opt.dataset == 'tiny_imagenet':
        dummy_input = torch.randn(2, 3, 64, 64).cuda()
    else:
        dummy_input = torch.randn(2, 3, 32, 32).cuda()

    features = []
    with torch.no_grad():
        _, features = model_t(dummy_input, features, is_feat=False)

    # 初始化掩码
    mask_list = []
    for idx, feature in enumerate(features):
        if(idx == 0):
            continue
        h, w = feature.shape[2:]
        c = feature.shape[1]
        mask = torch.ones(c, h, w).cuda()
        mask_list.append(mask)

    # 初始化优化后的计算器
    # layer_calculator = LayerImportanceCalculator(momentum=0.95, depth_bias=0.2)
    layer_calculator = LayerImportanceCalculator(momentum=0.95, depth_bias=opt.depth_bias)
    importance_calculator = GlobalAdaptiveTaylorImportanceCalculator(
        momentum=0.95,
        use_second_order=True,
        layer_calculator=layer_calculator,
        activation_history_size=10
    )
    mask_generator = GlobalProgressiveMaskGenerator(
        global_keep_ratio=global_keep_ratio,
        update_interval=100,
        warmup_epochs=5,
        target_epoch_ratio=0.6,
        total_epochs=opt.t1_epochs
    )

    # 训练循环
    best_acc = 0
    best_acc_at_budget = 0
    reached_budget_epoch = -1

    train_total_time = time.time()
    for epoch in range(1, opt.t1_epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)

        train_acc, train_loss, mask_list = train_global_adaptive_taylor_distill(
            epoch, train_loader, module_list, criterion_list,
            optimizer, opt, mask_list, global_keep_ratio,
            importance_calculator, mask_generator
        )

        # 验证时禁用梯度计算
        with torch.no_grad():
            test_acc, _, test_loss, _ = validate(
                val_loader, model_s, criterion_cls, opt, mask_list
            )

        current_ratio = mask_generator.get_current_global_keep_ratio()
        reached_target = abs(current_ratio - global_keep_ratio) < 0.01

        print(f'\nEpoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%')
        print(f'当前全局保留比例: {current_ratio:.3f}, 目标: {global_keep_ratio:.3f}')

        if epoch % 5 == 0 or reached_target:
            mask_generator.print_layer_statistics()

            layer_weights = layer_calculator.get_layer_importance_weights()
            if layer_weights:
                print("\n层重要性权重:")
                for layer_idx in sorted(layer_weights.keys()):
                    print(f"Layer {layer_idx}: {layer_weights[layer_idx]:.3f}")

        if reached_target and reached_budget_epoch == -1:
            reached_budget_epoch = epoch
            print(f"\n>>> 首次达到全局剪枝目标！Epoch: {epoch}")

        if reached_target and test_acc > best_acc_at_budget:
            best_acc_at_budget = test_acc
            # 只保存必要的信息
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc_at_budget': best_acc_at_budget,
                'masks': [m.cpu() for m in mask_list],  # 保存到CPU
                'reached_budget': True,
                'global_keep_ratio': global_keep_ratio
            }
            save_path = os.path.join(opt.save_folder, f'{opt.model_s}_best_at_budget.pth')
            torch.save(state, save_path)
            print(f'保存达到预算后的最佳模型，准确率: {best_acc_at_budget:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
                'masks': [m.cpu() for m in mask_list],
                'reached_budget': reached_target,
                'global_keep_ratio': global_keep_ratio
            }
            save_path = os.path.join(opt.save_folder, f'{opt.model_s}_best.pth')
            torch.save(state, save_path)
            print(f'保存最佳模型，准确率: {best_acc:.2f}%')

        # 定期清理内存
        if epoch % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f'\n训练完成！')
    final_duration = time.time() - train_total_time
    print(f'总耗时: {final_duration:.2f}秒')
    print(f'最佳准确率: {best_acc:.2f}%')
    if reached_budget_epoch > 0:
        print(f'达到剪枝预算的epoch: {reached_budget_epoch}')
        print(f'达到预算后的最佳准确率: {best_acc_at_budget:.2f}%')

    log_file.close()


if __name__ == '__main__':
    main()