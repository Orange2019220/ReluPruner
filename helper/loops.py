from __future__ import print_function, division

import sys
import time
import torch

from .util import AverageMeter, accuracy
from .context import ctx_noparamgrad_and_eval
from einops import reduce, repeat
import copy
# from torchattacks.attacks.autoattack import AutoAttack

def topkmask(feature, sensity):
    """
    生成一个二值掩码张量，保留输入特征中前k个最大值的位置。

    参数:
        feature (torch.Tensor): 输入特征张量，形状任意
        sensity (float): 指定保留元素的比例因子，范围[0,1]

    返回:
        torch.Tensor: 与输入特征形状相同的二值掩码张量（0或1）
                     保留前k个最大值位置，其余为0
    """
    # 将输入特征展平为一维张量进行阈值计算
    feature_flat = feature.view(1, -1)

    # 获取前k个最大值的阈值（k由sensity比例确定）
    # topk返回值按降序排列，取最后一个元素作为动态阈值
    value, _ = torch.topk(feature_flat, int(sensity * feature_flat.shape[1]))  # 获取前k个最大值
    value = value[-1][-1]

    # 构建二值掩码：大于阈值的位置为1，其余为0
    return (feature > value) + 0

# # 假设
# feature = torch.tensor([[1, 2, 3, 4, 5]])  # shape (1, 5)
# sensity = 0.6  # 保留60%的特征
# k = 0.6 * 5 = 3
#
# # 执行
# value, _ = torch.topk(feature, k)  # value = [[5, 4, 3]]
# value = value[-1][-1]  # value = 3
#
# # 掩码生成
# mask = (feature > value) + 0  # mask = [[0, 0, 0, 1, 1]]

    

def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        features = []
        # ===================forward=====================
        output, _ = model(input, features, is_feat=False)
        # output, _ = model(input, features, is_feat=False, mask_type='nat')

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))
        top5.update(acc5, input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_adversarial(epoch, train_loader, model, criterion_list, optimizer, opt, attacker, mask_weight=None):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    rtop1 = AverageMeter()
    rtop5 = AverageMeter()

    end = time.time()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    # if mask_weight is not None:
    #     mask_weight.restore_original_model()

    for idx, data in enumerate(train_loader):
         
        if opt.augment_mode == 'augmix':
            data_nat, data_aug = data
            input, target = data_nat
            input_aug, _ = data_aug
        else:
            input, target = data            
        data_time.update(time.time() - end)
        split_idx = input.shape[0] // 2

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if opt.augment_mode == 'augmix':
                input_aug = input_aug.float().cuda()

        features = []
        # ===================forward=====================
        input_norm = attacker.normalize(input, dataset=opt.dataset)

        # if mask_weight is not None:
        #     mask_weight.apply_mask(mask_type='nat')
        # for name, tensor in model.named_parameters():
        #     print(name, torch.sum(tensor != 0))
        # for name, tensor in mask_weight.modules[0].named_parameters():
        #     print(name, torch.sum(tensor != 0))
        if opt.use2BN:
            if opt.augment_mode == 'augmix':
                # input_nat_norm = attacker.normalize(input_nat, dataset=opt.dataset)
                output_nat, _ = model(input_norm[:split_idx], features, is_feat=False, idx2BN=0)
                output, _ = model(input_aug[:split_idx], features, is_feat=False, idx2BN=0)
            else:
                output, _ = model(input_norm[:split_idx], features, is_feat=False, idx2BN=0)
        else:
            output, _ = model(input_norm, features, is_feat=False)

        #=================== attack =====================
        # if mask_weight is not None:
        #     mask_weight.restore_original_model()
        #     mask_weight.apply_mask(mask_type='adv')
        
        with ctx_noparamgrad_and_eval(model):
            if opt.use2BN:
                input_adv = attacker.attack(model, input[split_idx:], labels=target[split_idx:], targets=None, idx2BN=1, dataset=opt.dataset)
            else:
                if opt.robust_train_mode == 'trades':
                    input_adv = attacker.attack(model, input, labels=output, targets=None, dataset=opt.dataset) #TRADES
                else:
                    input_adv = attacker.attack(model, input, labels=target, targets=None, dataset=opt.dataset)

        input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
        if opt.use2BN:
            output_adv, _ = model(input_adv_norm, features, is_feat=False, idx2BN=1)
        else:
            output_adv, _ = model(input_adv_norm, features, is_feat=False)

        if opt.use2BN:
            if opt.augment_mode == 'augmix':
                # loss = 0.5 * (criterion_cls(output_nat, target[:split_idx]) + criterion_cls(output, target[:split_idx]))/2 + 0.5 * criterion_cls(output_adv, target[split_idx:])
                loss = (criterion_cls(output_nat, target[:split_idx]) + criterion_cls(output, target[:split_idx]) + criterion_cls(output_adv, target[split_idx:]))/3
                # loss = 0.5 * criterion_cls(output, target[:split_idx]) + 0.5 * criterion_cls(output_adv, target[split_idx:])
            else:
                loss = 0.5 * criterion_cls(output, target[:split_idx]) + 0.5 * criterion_cls(output_adv, target[split_idx:])
        else:
            # TRADES Loss Function
            if opt.robust_train_mode == 'trades':
                loss = criterion_cls(output, target) + opt._lambda * criterion_div(output_adv, output)
            else:
                loss = 0.5 * criterion_cls(output, target) + 0.5 * criterion_cls(output_adv, target)

        if opt.use2BN:
            acc1, acc5 = accuracy(output, target[:split_idx], topk=(1, 5))
            racc1, racc5 = accuracy(output_adv, target[split_idx:], topk=(1, 5))
        else:                        
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            racc1, racc5 = accuracy(output_adv, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))
        top5.update(acc5, input.size(0))
        rtop1.update(racc1, input.size(0))
        rtop5.update(racc5, input.size(0))

        # if mask_weight is not None:
        #     mask_weight.restore_original_model() # restore original model before updating weights so that all weights are updated
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        if mask_weight is not None:
            #print('taking mask step')
            mask_weight.restore_original_model()
            optimizer.step()
            mask_weight.original_model = copy.deepcopy(model)
            # mask_weight.step()
        else:
            optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                #   'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'AdvAcc@1 {rtop1.val:.3f} ({rtop1.avg:.3f})\t'
                #   'AdvAcc@5 {rtop5.val:.3f} ({rtop5.avg:.3f})'
                  .format(epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, rtop1=rtop1, rtop5=rtop5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    print(' * AdvAcc@1 {rtop1.avg:.3f} AdvAcc@5 {rtop5.avg:.3f}'
          .format(rtop1=rtop1, rtop5=rtop5))
    return top1.avg, rtop1.avg, losses.avg


def train_adversarial_dualmask(epoch, train_loader, model, criterion_list, optimizer, opt, attacker, mask_weight=None):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    rtop1 = AverageMeter()
    rtop5 = AverageMeter()

    end = time.time()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    for idx, data in enumerate(train_loader):
         
        if opt.augment_mode == 'augmix':
            data_nat, data_aug = data
            input, target = data_nat
            input_aug, _ = data_aug
        else:
            input, target = data            
        data_time.update(time.time() - end)
        split_idx = input.shape[0] // 2

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if opt.augment_mode == 'augmix':
                input_aug = input_aug.float().cuda()

        features = []
        # ===================forward=====================
        input_norm = attacker.normalize(input, dataset=opt.dataset)
       
        if opt.use2BN:
            if opt.augment_mode == 'augmix':
                # input_nat_norm = attacker.normalize(input_nat, dataset=opt.dataset)
                # output_nat, _ = model(input_nat_norm[:split_idx], features, is_feat=False, idx2BN=split_idx, mask_type='nat')
                output_nat, _ = model(input_norm[:split_idx], features, is_feat=False, idx2BN=0, mask_type='nat')
                output, _ = model(input_aug[:split_idx], features, is_feat=False, idx2BN=0, mask_type='nat')
            else:
                output, _ = model(input_norm[:split_idx], features, is_feat=False, idx2BN=0, mask_type='nat')
        else:
            output, _ = model(input_norm, features, is_feat=False)

        # #=================== attack =====================
        with ctx_noparamgrad_and_eval(model):
            if opt.use2BN:
                # input_adv = attacker.attack(model, input_nat[split_idx:], labels=target[split_idx:], targets=None, idx2BN=0, mask_type='adv')
                input_adv = attacker.attack(model, input[split_idx:], labels=target[split_idx:], targets=None, idx2BN=1, mask_type='adv', dataset=opt.dataset)
            else:
                if opt.robust_train_mode == 'trades':
                    input_adv = attacker.attack(model, input, labels=output, targets=None, dataset=opt.dataset) #TRADES
                else:
                    input_adv = attacker.attack(model, input, labels=target, targets=None, dataset=opt.dataset)

        input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
        if opt.use2BN:
            # output_adv, _ = model(input_adv_norm, features, is_feat=False, idx2BN=0, mask_type='adv')
            output_adv, _ = model(input_adv_norm, features, is_feat=False, idx2BN=1, mask_type='adv')
        else:
            output_adv, _ = model(input_adv_norm, features, is_feat=False)

        if opt.use2BN:
            if opt.augment_mode == 'augmix':
                # loss = 0.5 * (criterion_cls(output_nat, target[:split_idx]) + criterion_cls(output, target[:split_idx]))/2 + 0.5 * criterion_cls(output_adv, target[split_idx:])
                loss = (criterion_cls(output_nat, target[:split_idx]) + criterion_cls(output, target[:split_idx]) + criterion_cls(output_adv, target[split_idx:]))/3
                # loss = 0.5 * criterion_cls(output, target[:split_idx]) + 0.5 * criterion_cls(output_adv, target[split_idx:])
                # loss = criterion_cls(output, target[:split_idx])
            else:
                loss = 0.5 * criterion_cls(output, target[:split_idx]) + 0.5 * criterion_cls(output_adv, target[split_idx:])
        else:
            # TRADES Loss Function
            if opt.robust_train_mode == 'trades':
                loss = criterion_cls(output, target) + opt._lambda * criterion_div(output_adv, output)
            else:
                loss = 0.5 * criterion_cls(output, target) + 0.5 * criterion_cls(output_adv, target)

        if opt.use2BN:
            acc1, acc5 = accuracy(output, target[:split_idx], topk=(1, 5))
            racc1, racc5 = accuracy(output_adv, target[split_idx:], topk=(1, 5))
            # racc1, racc5 = 0.0, 0.0
        else:                        
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            racc1, racc5 = accuracy(output_adv, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))
        top5.update(acc5, input.size(0))
        rtop1.update(racc1, input.size(0))
        rtop5.update(racc5, input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                #   'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'AdvAcc@1 {rtop1.val:.3f} ({rtop1.avg:.3f})\t'
                #   'AdvAcc@5 {rtop5.val:.3f} ({rtop5.avg:.3f})'
                  .format(epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, rtop1=rtop1, rtop5=rtop5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    print(' * AdvAcc@1 {rtop1.avg:.3f} AdvAcc@5 {rtop5.avg:.3f}'
          .format(rtop1=rtop1, rtop5=rtop5))
    return top1.avg, rtop1.avg, losses.avg


def train_comp_robust(epoch, train_loader, model, criterion_list, optimizer, opt, attacker=None):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    rtop1 = AverageMeter()
    rtop5 = AverageMeter()

    end = time.time()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    for idx, ((input, target), (input_deepaug, target_deepaug), (input_texture_debias, target_texture_debias)) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            input_deepaug = input_deepaug.cuda()
            target_deepaug = target_deepaug.cuda()
            input_texture_debias = input_texture_debias.cuda()
            target_texture_debias = target_texture_debias.cuda()
        split_idx = input.shape[0] // 2
        features = []
        # ===================forward=====================
        input_norm = input
        if attacker:
            input_norm = attacker.normalize(input, dataset=opt.dataset)
        # logits for clean imgs:
        if opt.use2BN:
            output, _ = model(input_norm[:split_idx], features, is_feat=False, idx2BN=split_idx)
            output_deepaug, _ = model(input_deepaug[:split_idx], features, is_feat=False, idx2BN=split_idx)
            output_texture_debias, _ = model(input_texture_debias[:split_idx], features, is_feat=False, idx2BN=split_idx)
        else:
            output, _ = model(input_norm, features, is_feat=False)
            output_deepaug, _ = model(input_deepaug, features, is_feat=False)
            output_texture_debias, _ = model(input_texture_debias, features, is_feat=False)

        #=================== attack =====================
        if attacker:
            with ctx_noparamgrad_and_eval(model):
                if opt.use2BN:
                    input_adv = attacker.attack(model, input[split_idx:], labels=target[split_idx:], targets=None, idx2BN=0, dataset=opt.dataset)
                else:
                    if opt.robust_train_mode == 'trades':
                        input_adv = attacker.attack(model, input, labels=output, targets=None, dataset=opt.dataset) #TRADES
                    else:
                        input_adv = attacker.attack(model, input, labels=target, targets=None, dataset=opt.dataset)

            input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
            if opt.use2BN:
                output_adv, _ = model(input_adv_norm, features, is_feat=False, idx2BN=0)
            else:
                output_adv, _ = model(input_adv_norm, features, is_feat=False)

        # TRADES Loss Function
        if opt.use2BN:
            loss = 0.5 * (criterion_cls(output, target[:split_idx]) + criterion_cls(output_deepaug, target_deepaug[:split_idx]) + criterion_cls(output_texture_debias, target_texture_debias[:split_idx]))/3 + 0.5 * criterion_cls(output_adv, target[split_idx:])
        else:
            if attacker and opt.robust_train_mode == 'trades':
                loss = (criterion_cls(output, target) + criterion_div(output_deepaug, output) + criterion_div(output_texture_debias, output) + criterion_div(output_adv, output))/4
            elif attacker and opt.robust_train_mode == 'pgd':
                loss = 0.5 * (criterion_cls(output, target) + criterion_cls(output_deepaug, target_deepaug) + criterion_cls(output_texture_debias, target_texture_debias))/3 + 0.5 * criterion_cls(output_adv, target)
            elif 'deepaug' in opt.robust_train_mode:
                loss = (criterion_cls(output, target) + criterion_cls(output_deepaug, target_deepaug) + criterion_cls(output_texture_debias, target_texture_debias))/3
            elif opt.robust_train_mode == 'deepaugwo_texdebias':
                loss = (criterion_cls(output, target) + criterion_cls(output_deepaug, target_deepaug))/2
            # loss = (criterion_cls(output, target) + criterion_div(output_deepaug, output) + criterion_div(output_texture_debias, output))/3
        if opt.use2BN:
            acc1, acc5 = accuracy(output, target[:split_idx], topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))
        top5.update(acc5, input.size(0))

        if attacker:
            if opt.use2BN:
                racc1, racc5 = accuracy(output_adv, target[split_idx:], topk=(1, 5))
            else:
                racc1, racc5 = accuracy(output_adv, target, topk=(1, 5))
            rtop1.update(racc1, input.size(0))
            rtop5.update(racc5, input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            if attacker:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    #   'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                    'AdvAcc@1 {rtop1.val:.3f} ({rtop1.avg:.3f})\t'
                    #   'AdvAcc@5 {rtop5.val:.3f} ({rtop5.avg:.3f})'
                    .format(epoch, idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, rtop1=rtop1, rtop5=rtop5))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                    # 'AdvAcc@1 {rtop1.val:.3f} ({rtop1.avg:.3f})\t'
                    #   'AdvAcc@5 {rtop5.val:.3f} ({rtop5.avg:.3f})'
                    .format(epoch, idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, rtop1=rtop1, rtop5=rtop5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    if attacker:
        print(' * AdvAcc@1 {rtop1.avg:.3f} AdvAcc@5 {rtop5.avg:.3f}'
            .format(rtop1=rtop1, rtop5=rtop5))
        return top1.avg, rtop1.avg, losses.avg
    return top1.avg, 0.0, losses.avg


# 第一阶段训练 掩码
def train_distill_stage1(epoch, train_loader, module_list, criterion_list, optimizer, opt, mask_list, sensitivity_list_final, attacker=None):
    """One epoch distillation"""

    # 模型初始化：学生模型训练模式，教师模型评估模式
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()  # 教师模型设为评估模式

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    # 损失函数初始化
    criterion_cls = criterion_list[0]  # 分类损失
    criterion_div = criterion_list[1]  # KL散度损失
    criterion_kd = criterion_list[2]  # 蒸馏损失

    # 模型引用
    model_s = module_list[0]  # 学生模型
    model_t = module_list[-1]  # 教师模型


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    rtop1 = AverageMeter()
    rtop5 = AverageMeter()

    end = time.time()

    # 所有mask list 移到cuda上
    if torch.cuda.is_available():
        mask_list = [mask.cuda() for mask in mask_list]

    mask_list_copy = copy.deepcopy(mask_list) #c, h, w
    mask_epoch = []

    # 数据加载器迭代
    for idx, data in enumerate(train_loader):
    #for idx, (input, target) in enumerate(train_loader):

        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        for mask_index, mask in enumerate(mask_list):
            mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input.shape)[0]) # b c h w

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            mask_list = [mask.cuda() for mask in mask_list]

            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()



        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True

        input_norm = input
        if attacker:
            input_norm = attacker.normalize(input, dataset=opt.dataset)

        features_s = []
        feature_eachlayer_s, logit_s, features_s = model_s(input_norm, mask_list, features_s, is_feat=True)
        mask_list = copy.deepcopy(mask_list_copy)


        with torch.no_grad():

            features_s = [f.detach() for f in features_s]

            features_t = []
            feature_diff = []
            features_t_adv = []

            feature_eachlayer_t, logit_t, features_t = model_t(input_norm, features_t, is_feat=True)
            features_t = [f.detach() for f in features_t]
            feature_eachlayer_t = [f.detach() for f in feature_eachlayer_t]


        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)


        # Feature difference calculation for mask
        with torch.no_grad():
            assert len(features_t) == len(features_s)
            # if opt.use_l2_norm:
            #     feature_diff = [(features_t[i] - features_s[i]) for i in range(len(features_s))]
            if attacker and opt.mask_calculation == 'feat_diff_nat+adv':
                print('no  attacker')
            else:

                # 计算学生模型与教师模型的特征差异---------------------------------------
                feature_diff = [torch.abs(features_t[i] - features_s[i]) for i in range(len(features_s))]
                feature_diff = [reduce(feature, 'b c h w -> c h w', 'sum') for feature in feature_diff]
            # 累积每个batch的特征差异
            if len(mask_epoch) == 0:
                mask_epoch = feature_diff  # 首次直接赋值
            else:
                mask_epoch = [mask_epoch[i] + feature_diff[i] for i in range(len(feature_diff))]



        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0      # 取这个loss
        elif opt.distill == 'hint':
            f_s = module_list[1](feature_eachlayer_s[opt.hint_layer])
            f_t = feature_eachlayer_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feature_eachlayer_s[-1]
            f_t = feature_eachlayer_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feature_eachlayer_s[-2]]
            g_t = [feature_eachlayer_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feature_eachlayer_s[-1]
            f_t = feature_eachlayer_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feature_eachlayer_s[-1]
            f_t = feature_eachlayer_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feature_eachlayer_s[-1])
            f_t = module_list[2](feature_eachlayer_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feature_eachlayer_s[-2])
            factor_t = module_list[2](feature_eachlayer_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        # loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        # RSLAD: Robust soft label adversarial distilllation loss
        if attacker and opt.robust_train_mode == 'rslad':
            # loss = (1 - opt.alpha) * loss_div + opt.alpha * loss_div_adv
            print('no  attacker')
        else:
            # --gamma=0.5 --alpha=0.5
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))
        top5.update(acc5, input.size(0))

        if attacker:
            print('no  attacker')
        # ===================backward=====================

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            if attacker:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                #   'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'AdvAcc@1 {rtop1.val:.3f} ({rtop1.avg:.3f})\t'
                #   'AdvAcc@5 {rtop5.val:.3f} ({rtop5.avg:.3f})'
                .format(epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, rtop1=rtop1, rtop5=rtop5))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    if attacker:
        print(' * AdvAcc@1 {rtop1.avg:.3f} AdvAcc@5 {rtop5.avg:.3f}'
          .format(rtop1=rtop1, rtop5=rtop5))

    assert len(sensitivity_list_final) == len(mask_epoch)

    #  =====================topkmask 生成一个二值掩码张量，保留输入特征中前k个最大值的位置。------------------------------------------------------------------------------------------
    mask_epoch = [topkmask(mask_epoch[i], sensitivity_list_final[i]) for i in range(len(mask_epoch))]

    if attacker:
        return top1.avg, rtop1.avg, losses.avg, mask_epoch
    return top1.avg, 0.0, losses.avg, mask_epoch

def validate(val_loader, model, criterion, opt, mask_list = None, attacker=None, dual_masking=False, mask_list_adv=None):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    rtop1 = AverageMeter()
    rtop5 = AverageMeter()
    # attack = PGD(model, eps=8/255, alpha=2/255, steps=20, random_start=True)

    # switch to evaluate mode
    model.eval()
    mask_list_copy = copy.deepcopy(mask_list)
    if dual_masking:
        mask_list_adv_copy = copy.deepcopy(mask_list_adv)
    # with torch.no_grad():
    end = time.time()
    for idx, (input, target) in enumerate(val_loader):

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if mask_list is not None:
                mask_list = [mask.cuda() for mask in mask_list]
            if dual_masking:
                mask_list_adv = [mask.cuda() for mask in mask_list_adv]

        features = []

        # compute output
        if attacker:
            if mask_list == None:
                input_norm = attacker.normalize(input, dataset=opt.dataset)
                output, _ = model(input_norm, features, is_feat = False)
                with ctx_noparamgrad_and_eval(model):
                    # input_adv = attack(input, labels=target)
                    input_adv = attacker.attack(model, input, labels=target, targets=None, dataset=opt.dataset)
                input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
                output_adv, _ = model(input_adv_norm, features, is_feat=False)

            else:
                for mask_index, mask in enumerate(mask_list):
                    mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input.shape)[0]) # b c h w
                input_norm = attacker.normalize(input, dataset=opt.dataset)
                output, _ = model(input_norm, mask_list, features, is_feat=False)
                mask_list = copy.deepcopy(mask_list_copy)
                if not dual_masking:
                    with ctx_noparamgrad_and_eval(model):
                        input_adv = attacker.attack(model, input, labels=target, targets=None, mask_list=mask_list, dataset=opt.dataset)
                    mask_list = copy.deepcopy(mask_list_copy)
                    for mask_index, mask in enumerate(mask_list):
                        mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input.shape)[0]) # b c h w
                    input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
                    output_adv, _ = model(input_adv_norm, mask_list, features, is_feat=False)
                    mask_list = copy.deepcopy(mask_list_copy)
                else:
                    with ctx_noparamgrad_and_eval(model):
                        input_adv = attacker.attack(model, input, labels=target, targets=None, mask_list=mask_list_adv, dataset=opt.dataset)
                    mask_list_adv = copy.deepcopy(mask_list_adv_copy)
                    for mask_index, mask in enumerate(mask_list_adv):
                        mask_list_adv[mask_index] = repeat(mask_list_adv[mask_index], 'c h w-> b c h w', b = list(input.shape)[0]) # b c h w
                    input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
                    output_adv, _ = model(input_adv_norm, mask_list_adv, features, is_feat=False)
                    mask_list_adv = copy.deepcopy(mask_list_adv_copy)
        else:
            if mask_list == None:
                output, _ = model(input, features, is_feat=False)
            else:
                for mask_index, mask in enumerate(mask_list):
                    mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input.shape)[0]) # b c h w
                output, _ = model(input, mask_list, features, is_feat=False)
                mask_list = copy.deepcopy(mask_list_copy)

        # if mask_list == None:
        #     if attacker:
        #         input_norm = attacker.normalize(input)
        #         output, _ = model(input_norm, features, is_feat = False)
        #         with ctx_noparamgrad_and_eval(model):
        #             # input_adv = attack(input, labels=target)
        #             input_adv = attacker.attack(model, input, labels=target, targets=None)
        #         input_adv_norm = attacker.normalize(input_adv)
        #         output_adv, _ = model(input_adv_norm, features, is_feat=False)

        # else:
        #     for mask_index, mask in enumerate(mask_list):
        #         mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input.shape)[0]) # b c h w
        #     output, _ = model(input, mask_list, features, is_feat=False)
        #     mask_list = copy.deepcopy(mask_list_copy)
        #     if attacker:
        #         with ctx_noparamgrad_and_eval(model):
        #             input_adv = attacker.attack(model, input, labels=target, targets=None, mask_list=mask_list)
        #         mask_list = copy.deepcopy(mask_list_copy)
        #         for mask_index, mask in enumerate(mask_list):
        #             mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input.shape)[0]) # b c h w
        #         output_adv, _ = model(input_adv, mask_list, features, is_feat=False)
        #         mask_list = copy.deepcopy(mask_list_copy)
        if attacker:
            loss = 0.5*criterion(output, target) + 0.5*criterion(output_adv, target)
        else:
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))
        top5.update(acc5, input.size(0))
        if attacker:
            racc1, racc5 = accuracy(output_adv, target, topk=(1, 5))
            rtop1.update(racc1, input.size(0))
            rtop5.update(racc5, input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            if attacker:
                print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                # 'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                'AdvAcc@1 {rtop1.val:.3f} ({rtop1.avg:.3f})\t'
                # 'AdvAcc@5 {rtop5.val:.3f} ({rtop5.avg:.3f})'
                .format(idx, len(val_loader), batch_time=batch_time,
                loss=losses, top1=top1, top5=top5, rtop1=rtop1, rtop5=rtop5))
            else:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
    if attacker:
        print(' * AdvAcc@1 {rtop1.avg:.3f} AdvAcc@5 {rtop5.avg:.3f}'.format(rtop1=rtop1, rtop5=rtop5))
    if attacker:
        return top1.avg, rtop1.avg, losses.avg, mask_list_copy
    return top1.avg, 0.0, losses.avg, mask_list_copy


#
# class DistillKL(nn.Module):
#     """Distilling the Knowledge in a Neural Network"""
#     def __init__(self, T):
#         super(DistillKL, self).__init__()
#         self.T = T
#
#     def forward(self, y_s, y_t):
#         p_s = F.log_softmax(y_s/self.T, dim=1)
#         p_t = F.softmax(y_t/self.T, dim=1)
#         loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
#         return loss

def train_distill_stage2(epoch, train_loader, module_list, criterion_list, optimizer, opt, mask_list, mask_weight=None, attacker=None):
    """One epoch distillation"""
    # set modules as train()

    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    rtop1 = AverageMeter()
    rtop5 = AverageMeter()
    end = time.time()

    mask_list_copy = copy.deepcopy(mask_list) #c, h, w  # 存储原始掩码
    mask_epoch = []    # 用于累积每个batch的特征差异

    for idx, data in enumerate(train_loader):
    #for idx, (input, target) in enumerate(train_loader):

        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        for mask_index, mask in enumerate(mask_list):
            # 将掩码扩展到batch维度 (c,h,w) -> (b,c,h,w)
            mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input.shape)[0]) # b c h w

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            mask_list = [mask.cuda() for mask in mask_list]
            
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        
        input_norm = input
        if attacker:
            input_norm = attacker.normalize(input, dataset=opt.dataset)

        features_s = []
         # 模型前向传播时应用掩码
        feature_eachlayer_s, logit_s, _ = model_s(input_norm, mask_list, features_s, is_feat=True)
        mask_list = copy.deepcopy(mask_list_copy)

        with torch.no_grad():
            features_t = []
            feature_eachlayer_t, logit_t, _ = model_t(input_norm, features_t, is_feat=True)
            feature_eachlayer_t = [f.detach() for f in feature_eachlayer_t]
            
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        
        if attacker:
            with ctx_noparamgrad_and_eval(model_s):
                if opt.robust_train_mode == 'rslad':
                    input_adv = attacker.attack(model_s, input, labels=logit_t, targets=None, mask_list=mask_list, dataset=opt.dataset)
                else:
                    input_adv = attacker.attack(model_s, input, labels=target, targets=None, mask_list=mask_list, dataset=opt.dataset)
            model_s.train()
            mask_list = copy.deepcopy(mask_list_copy)
            for mask_index, mask in enumerate(mask_list):
                mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input_adv.shape)[0]) # b c h w

            features_s_adv = []
            input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
            feature_eachlayer_s_adv, logit_s_adv, _ = model_s(input_adv_norm, mask_list, features_s_adv, is_feat=True)
            mask_list = copy.deepcopy(mask_list_copy)
            loss_cls_adv = criterion_cls(logit_s_adv, target)
            loss_div_adv = criterion_div(logit_s_adv, logit_t)
            
        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feature_eachlayer_s[opt.hint_layer])
            f_t = feature_eachlayer_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feature_eachlayer_s[-1]
            f_t = feature_eachlayer_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feature_eachlayer_s[-2]]
            g_t = [feature_eachlayer_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feature_eachlayer_s[-1]
            f_t = feature_eachlayer_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feature_eachlayer_s[-1]
            f_t = feature_eachlayer_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feature_eachlayer_s[-1])
            f_t = module_list[2](feature_eachlayer_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feature_eachlayer_s[-2])
            factor_t = module_list[2](feature_eachlayer_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        # loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        # if attacker:
        #     loss = loss_div_adv + opt.beta * loss_kd 

        # ARD loss

        # RSLAD: Robust soft label adversarial distilllation loss
        if attacker and opt.robust_train_mode == 'rslad':
            loss = (1 - opt.alpha) * loss_div + opt.alpha * loss_div_adv
        # RSLAD+Attention:
        elif attacker and opt.robust_train_mode == 'rslad+attn':
            loss = (1 - opt.alpha) * loss_div + opt.alpha * loss_div_adv + opt.beta * loss_kd 
        # RSLAD+Attention+Adv_Attention:
        elif attacker and opt.robust_train_mode == 'rslad+attn+advattn':
            with torch.no_grad():
                features_t_adv = []
                feature_eachlayer_t_adv, logit_t_adv, features_t_adv = model_t(input_adv_norm, features_t_adv, is_feat=True)
            g_s_adv = feature_eachlayer_s_adv[1:-1]
            g_t_adv = feature_eachlayer_t_adv[1:-1]
            loss_group = criterion_kd(g_s_adv, g_t_adv)
            loss_kd_adv = sum(loss_group)
            loss = (1 - opt.alpha) * loss_div + opt.alpha * loss_div_adv + opt.beta * 0.5 * loss_kd + opt.beta * 0.5 * loss_kd_adv
        # elif opt.robust_train_mode == 'augmix':
        #     # loss = loss_cls
        #     # loss = opt.gamma * loss_cls + opt.alpha * loss_div
        #     loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        else:
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))
        top5.update(acc5, input.size(0))

        if attacker:        
            racc1, racc5 = accuracy(logit_s_adv, target, topk=(1, 5))
            rtop1.update(racc1, input.size(0))
            rtop5.update(racc5, input.size(0))
        # ===================backward=====================
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        loss.backward()
        if mask_weight is not None:
            #print('taking mask step')
            mask_weight.step()
        else:
            #print('taking optimizer step')
            optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            if attacker:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                #   'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'AdvAcc@1 {rtop1.val:.3f} ({rtop1.avg:.3f})\t'
                #   'AdvAcc@5 {rtop5.val:.3f} ({rtop5.avg:.3f})'
                .format(epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, rtop1=rtop1, rtop5=rtop5))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    if attacker:
        print(' * AdvAcc@1 {rtop1.avg:.3f} AdvAcc@5 {rtop5.avg:.3f}'
          .format(rtop1=rtop1, rtop5=rtop5))

    mask_epoch = copy.deepcopy(mask_list_copy)
    if attacker:
        return top1.avg, rtop1.avg, losses.avg, mask_epoch
    return top1.avg, 0.0, losses.avg, mask_epoch


def train_distill_stage2_dual_mask(epoch, train_loader, module_list, criterion_list, optimizer, opt, mask_list, mask_weight=None, attacker=None, mask_list_adv=None):
    """One epoch distillation"""
    # set modules as train()

    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    rtop1 = AverageMeter()
    rtop5 = AverageMeter()
    end = time.time()

    mask_list_copy = copy.deepcopy(mask_list) #c, h, w
    mask_list_adv_copy = copy.deepcopy(mask_list_adv) #c, h, w
    mask_epoch = []
    mask_epoch_adv = []

    for idx, data in enumerate(train_loader):
    #for idx, (input, target) in enumerate(train_loader):

        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        for mask_index, mask in enumerate(mask_list):
            mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input.shape)[0]) # b c h w
            # mask_list_adv[mask_index] = repeat(mask_list_adv[mask_index], 'c h w-> b c h w', b = list(input.shape)[0])
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            mask_list = [mask.cuda() for mask in mask_list]
            mask_list_adv = [mask.cuda() for mask in mask_list_adv]

            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True

        input_norm = input
        if attacker:
            input_norm = attacker.normalize(input, dataset=opt.dataset)

        features_s = []
        feature_eachlayer_s, logit_s, _ = model_s(input_norm, mask_list, features_s, is_feat=True)
        mask_list = copy.deepcopy(mask_list_copy)
        with torch.no_grad():
            features_t = []
            feature_eachlayer_t, logit_t, _ = model_t(input_norm, features_t, is_feat=True)
            feature_eachlayer_t = [f.detach() for f in feature_eachlayer_t]

        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        if attacker:
            with ctx_noparamgrad_and_eval(model_s):
                if opt.robust_train_mode == 'rslad':
                    input_adv = attacker.attack(model_s, input, labels=logit_t, targets=None, mask_list=mask_list_adv, dataset=opt.dataset)
                else:
                    input_adv = attacker.attack(model_s, input, labels=target, targets=None, mask_list=mask_list_adv, dataset=opt.dataset)
            model_s.train()
            mask_list_adv = copy.deepcopy(mask_list_adv_copy)
            for mask_index, mask in enumerate(mask_list_adv):
                mask_list_adv[mask_index] = repeat(mask_list_adv[mask_index], 'c h w-> b c h w', b = list(input_adv.shape)[0]) # b c h w

            features_s_adv = []
            input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
            feature_eachlayer_s_adv, logit_s_adv, _ = model_s(input_adv_norm, mask_list_adv, features_s_adv, is_feat=True)
            mask_list_adv = copy.deepcopy(mask_list_adv_copy)
            loss_cls_adv = criterion_cls(logit_s_adv, target)
            loss_div_adv = criterion_div(logit_s_adv, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feature_eachlayer_s[opt.hint_layer])
            f_t = feature_eachlayer_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feature_eachlayer_s[-1]
            f_t = feature_eachlayer_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feature_eachlayer_s[-2]]
            g_t = [feature_eachlayer_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feature_eachlayer_s[-1]
            f_t = feature_eachlayer_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feature_eachlayer_s[-1]
            f_t = feature_eachlayer_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feature_eachlayer_s[-1])
            f_t = module_list[2](feature_eachlayer_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feature_eachlayer_s[-2])
            factor_t = module_list[2](feature_eachlayer_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        # loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        # if attacker:
        #     loss = loss_div_adv + opt.beta * loss_kd

        # ARD loss

        # RSLAD: Robust soft label adversarial distilllation loss
        if attacker and opt.robust_train_mode == 'rslad':
            loss = (1 - opt.alpha) * loss_div + opt.alpha * loss_div_adv
        # RSLAD+Attention:
        elif attacker and opt.robust_train_mode == 'rslad+attn':
            loss = (1 - opt.alpha) * loss_div + opt.alpha * loss_div_adv + opt.beta * loss_kd
        # RSLAD+Attention+Adv_Attention:
        elif attacker and opt.robust_train_mode == 'rslad+attn+advattn':
            with torch.no_grad():
                features_t_adv = []
                feature_eachlayer_t_adv, logit_t_adv, features_t_adv = model_t(input_adv_norm, features_t_adv, is_feat=True)
            g_s_adv = feature_eachlayer_s_adv[1:-1]
            g_t_adv = feature_eachlayer_t_adv[1:-1]
            loss_group = criterion_kd(g_s_adv, g_t_adv)
            loss_kd_adv = sum(loss_group)
            loss = (1 - opt.alpha) * loss_div + opt.alpha * loss_div_adv + opt.beta * 0.5 * loss_kd + opt.beta * 0.5 * loss_kd_adv
        else:
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd


        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))
        top5.update(acc5, input.size(0))

        if attacker:
            racc1, racc5 = accuracy(logit_s_adv, target, topk=(1, 5))
            rtop1.update(racc1, input.size(0))
            rtop5.update(racc5, input.size(0))
        # ===================backward=====================
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        loss.backward()
        if mask_weight is not None:
            #print('taking mask step')
            mask_weight.step()
        else:
            #print('taking optimizer step')
            optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            if attacker:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                #   'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'AdvAcc@1 {rtop1.val:.3f} ({rtop1.avg:.3f})\t'
                #   'AdvAcc@5 {rtop5.val:.3f} ({rtop5.avg:.3f})'
                .format(epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, rtop1=rtop1, rtop5=rtop5))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    if attacker:
        print(' * AdvAcc@1 {rtop1.avg:.3f} AdvAcc@5 {rtop5.avg:.3f}'
          .format(rtop1=rtop1, rtop5=rtop5))

    mask_epoch = copy.deepcopy(mask_list_copy)
    mask_epoch_adv = copy.deepcopy(mask_list_adv_copy)

    return top1.avg, rtop1.avg, losses.avg, mask_epoch, mask_epoch_adv


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt, attacker=None):
    """One epoch distillation"""
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()
    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    rtop1 = AverageMeter()
    rtop5 = AverageMeter()
    end = time.time()

    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True

        input_norm = input
        if attacker:
            input_norm = attacker.normalize(input)
        
        features_s = []
        feature_eachlayer_s, logit_s, features_s = model_s(input, features_s, is_feat=True)

        with torch.no_grad():
            features_s = [f.detach() for f in features_s]
            features_t = []
            feature_diff = []

            feature_eachlayer_t, logit_t, features_t = model_t(input, features_t, is_feat=True)
            features_t = [f.detach() for f in features_t]
            feature_eachlayer_t = [f.detach() for f in feature_eachlayer_t]

            assert len(features_t) == len(features_s)
            if opt.use_l2_norm:
                feature_diff = [(features_t[i] - features_s[i]) for i in range(len(features_s))]

            # 计算教师模型和学生模型的特征差异
            feature_diff = [torch.abs(features_t[i] - features_s[i]) for i in range(len(features_s))]
            # 按通道维度聚合
            feature_diff = [reduce(feature, 'b c h w -> c h w', 'sum') for feature in feature_diff]
            

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        if attacker:
            with ctx_noparamgrad_and_eval(model_s):
                if opt.robust_train_mode == 'rslad':
                    input_adv = attacker.attack(model_s, input, labels=logit_t, targets=None)
                else:
                    input_adv = attacker.attack(model_s, input, labels=target, targets=None)
            model_s.train()
            
            features_s_adv = []
            input_adv_norm = attacker.normalize(input_adv)
            logit_s_adv, _ = model_s(input_adv_norm, features_s_adv, is_feat=False)
            loss_cls_adv = criterion_cls(logit_s_adv, target)
            loss_div_adv = criterion_div(logit_s_adv, logit_t)
            
        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feature_eachlayer_s[opt.hint_layer])
            f_t = feature_eachlayer_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feature_eachlayer_s[-1]
            f_t = feature_eachlayer_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feature_eachlayer_s[-2]]
            g_t = [feature_eachlayer_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feature_eachlayer_s[-1]
            f_t = feature_eachlayer_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feature_eachlayer_s[-1]
            f_t = feature_eachlayer_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feature_eachlayer_s[-1])
            f_t = module_list[2](feature_eachlayer_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feature_eachlayer_s[-2])
            factor_t = module_list[2](feature_eachlayer_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        if attacker and opt.robust_train_mode == 'rslad':
            loss = (1 - opt.alpha) * loss_div + opt.alpha * loss_div_adv
        else:
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))
        top5.update(acc5, input.size(0))

        if attacker:        
            racc1, racc5 = accuracy(logit_s_adv, target, topk=(1, 5))
            rtop1.update(racc1, input.size(0))
            rtop5.update(racc5, input.size(0))

        # ===================backward=====================
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            if attacker:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                #   'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'AdvAcc@1 {rtop1.val:.3f} ({rtop1.avg:.3f})\t'
                #   'AdvAcc@5 {rtop5.val:.3f} ({rtop5.avg:.3f})'
                .format(epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, rtop1=rtop1, rtop5=rtop5))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    if attacker:
        print(' * AdvAcc@1 {rtop1.avg:.3f} AdvAcc@5 {rtop5.avg:.3f}'
          .format(rtop1=rtop1, rtop5=rtop5))
        
    if attacker:
        return top1.avg, rtop1.avg, losses.avg
    return top1.avg, 0.0, losses.avg


def validate_dualBN(val_loader, model, criterion, opt, mask_list = None, attacker=None, mask_weight=None):
    """validation"""

    val_lambdas = [0.0, 1.0] # For lambda=0.0, BN for natural images is used; lambda=1.0 BN for adv images is used
    val_accs, val_accs_adv = {}, {}
    batch_time = AverageMeter()
    for val_lambda in val_lambdas:
        val_accs[val_lambda], val_accs_adv[val_lambda] = AverageMeter(), AverageMeter()
        # losses = AverageMeter()
        # top1 = AverageMeter()
        # top5 = AverageMeter()
        # rtop1 = AverageMeter()
        # rtop5 = AverageMeter()
    # attack = PGD(model, eps=8/255, alpha=2/255, steps=20, random_start=True)

    # switch to evaluate mode
    model.eval()
    mask_list_copy = copy.deepcopy(mask_list)
    end = time.time()
    for idx, (input, target) in enumerate(val_loader):
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if mask_list is not None:
                mask_list = [mask.cuda() for mask in mask_list]
            
        features = []
        for j, val_lambda in enumerate(val_lambdas):
            # _lambda = np.expand_dims( np.repeat(val_lambda, target.size()[0]), axis=1)
            # _lambda = torch.from_numpy(_lambda).float().cuda()
            # idx2BN = int(target.size()[0]) if val_lambda==0 else 0
            idx2BN = val_lambda
            # if mask_weight is not None:
            #     mask_weight.restore_original_model()
            #     mask_weight.apply_mask(mask_type='nat')
            # compute output
            if mask_list == None:
                input_norm = attacker.normalize(input, dataset=opt.dataset)
                # output, _ = model(input_norm, features, is_feat = False, idx2BN=idx2BN, mask_type='nat')
                output, _ = model(input_norm, features, is_feat = False, idx2BN=idx2BN)
                # if mask_weight is not None:
                #     mask_weight.restore_original_model()
                #     mask_weight.apply_mask(mask_type='adv')
                with ctx_noparamgrad_and_eval(model):
                    # input_adv = attack(input, labels=target)
                    input_adv = attacker.attack(model, input, labels=target, targets=None, idx2BN=idx2BN, dataset=opt.dataset)
                    # input_adv = attacker.attack(model, input, labels=target, targets=None, idx2BN=idx2BN, mask_type='adv')
                input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
                # output_adv, _ = model(input_adv_norm, features, is_feat=False, idx2BN=idx2BN, mask_type='adv')
                output_adv, _ = model(input_adv_norm, features, is_feat=False, idx2BN=idx2BN)
            else:
                for mask_index, mask in enumerate(mask_list):
                    mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input.shape)[0]) # b c h w
                input_norm = attacker.normalize(input, dataset=opt.dataset)
                output, _ = model(input_norm, mask_list, features, is_feat=False, idx2BN=idx2BN)
                mask_list = copy.deepcopy(mask_list_copy)
                with ctx_noparamgrad_and_eval(model):
                    input_adv = attacker.attack(model, input, labels=target, targets=None, mask_list=mask_list, idx2BN=idx2BN, dataset=opt.dataset)
                mask_list = copy.deepcopy(mask_list_copy)
                for mask_index, mask in enumerate(mask_list):
                    mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input.shape)[0]) # b c h w
                input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
                output_adv, _ = model(input_adv_norm, mask_list, features, is_feat=False, idx2BN=idx2BN)
                mask_list = copy.deepcopy(mask_list_copy)
            
            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            # losses.update(loss.item(), input.size(0))
            val_accs[val_lambda].update(acc1, input.size(0))
            if attacker:        
                racc1, = accuracy(output_adv, target, topk=(1,))
                val_accs_adv[val_lambda].update(racc1, input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        #     if idx % opt.print_freq == 0:
        #         print('Test: [{0}/{1}]\t'
        #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #         'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #         # 'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
        #         'AdvAcc@1 {rtop1.val:.3f} ({rtop1.avg:.3f})\t'
        #         # 'AdvAcc@5 {rtop5.val:.3f} ({rtop5.avg:.3f})'
        #         .format(idx, len(val_loader), batch_time=batch_time,
        #         loss=losses, top1=top1, top5=top5, rtop1=rtop1, rtop5=rtop5))

        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #         .format(top1=top1, top5=top5))
        # print(' * AdvAcc@1 {rtop1.avg:.3f} AdvAcc@5 {rtop5.avg:.3f}'.format(rtop1=rtop1, rtop5=rtop5))
    return val_accs, val_accs_adv, mask_list


def validate_self_distill(val_loader, model, criterion, opt, mask_list = None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_aux = AverageMeter()
    top5_aux = AverageMeter()

    # switch to evaluate mode
    model.eval()

    #mask_list_aux = copy.deepcopy(mask_list)
    mask_list_copy = copy.deepcopy(mask_list)
    #mask_list_copy_aux = copy.deepcopy(mask_list_aux)
    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                if mask_list is not None:
                    mask_list = [mask.cuda() for mask in mask_list]
                    #mask_list_aux = [mask.cuda() for mask in mask_list_aux]
            
            features = []
            
            # compute output
            if mask_list == None:
                output, _ = model(input, features, is_feat = False)
            else:
                for mask_index, mask in enumerate(mask_list):
                    mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input.shape)[0]) # b c h w

                #for mask_index, mask in enumerate(mask_list_aux):
                #    mask_list_aux[mask_index] = repeat(mask_list_aux[mask_index], 'c h w-> b c h w', b = list(input.shape)[0]) # b c h w

                output, output_aux, _ = model(input, mask_list, features, is_feat=False)
                mask_list = copy.deepcopy(mask_list_copy)
                #mask_list_aux = copy.deepcopy(mask_list_copy_aux)
            
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1_aux, acc5_aux = accuracy(output_aux, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1, input.size(0))
            top5.update(acc5, input.size(0))
            top1_aux.update(acc1_aux, input.size(0))
            top5_aux.update(acc5_aux, input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'AuxClf_Acc@1 {top1_aux.val:.3f} ({top1_aux.avg:.3f})\t'
                      'AuxClf_Acc@5 {top5_aux.val:.3f} ({top5_aux.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5, top1_aux=top1_aux, top5_aux=top5_aux))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} AuxClf_Acc@1 {top1_aux.avg:.3f} AuxClf_Acc@5 {top5_aux.avg:.3f}'
              .format(top1=top1, top5=top5, top1_aux=top1_aux, top5_aux=top5_aux))

    return top1.avg, top5.avg, top1_aux.avg, losses.avg, mask_list_copy



