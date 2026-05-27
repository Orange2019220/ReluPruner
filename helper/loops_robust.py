import sys
import time
import torch

from .util import AverageMeter, accuracy
from .context import ctx_noparamgrad_and_eval
from einops import reduce, repeat
import copy

def topkmask(feature, sensity):
    # import pdb; pdb.set_trace()
    feature_flat = feature.view(1, -1)
    value, _ = torch.topk(feature_flat, int(sensity * feature_flat.shape[1]))
    value = value[-1][-1]
    return (feature > value) + 0

def train_distill_stage1_3BN(epoch, train_loader, module_list, criterion_list, optimizer, opt, mask_list, sensitivity_list_final, attacker=None):
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
    
    if torch.cuda.is_available():
        mask_list = [mask.cuda() for mask in mask_list]
    mask_list_copy = copy.deepcopy(mask_list) #c, h, w
    mask_epoch = []
    for idx, data in enumerate(train_loader):
    #for idx, (input, target) in enumerate(train_loader):

        if opt.augment_mode == 'augmix':
            data_nat, data_aug = data
            input, target, index = data_nat
            input_aug, _, _ = data_aug
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        if opt.augment_mode == 'augmix':
            idx1 = input.shape[0] // 3
            idx2 = idx1 * 2
        else:
            split_idx = input.shape[0] // 2

        for mask_index, mask in enumerate(mask_list):
            if opt.augment_mode == 'augmix':
                mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = idx1) # b c h w
            else:
                mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = split_idx) # b c h w

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            mask_list = [mask.cuda() for mask in mask_list]
            
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()
            if opt.augment_mode == 'augmix':
                input_aug = input_aug.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True

        input_norm = input
        if attacker:
            input_norm = attacker.normalize(input, dataset=opt.dataset)

        features_s = []
        features_s_aug = []

        if opt.augment_mode == 'augmix':
            input_aug_norm = attacker.normalize(input_aug, dataset=opt.dataset)
            feature_eachlayer_s, logit_s, features_s = model_s(input_norm[:idx1], mask_list, features_s, is_feat=True, idx2BN=0)
            mask_list = copy.deepcopy(mask_list_copy)
            for mask_index, mask in enumerate(mask_list):
                mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = idx1) # b c h w
            feature_eachlayer_s_aug, logit_s_aug, _ = model_s(input_aug_norm[idx1:idx2], mask_list, features_s_aug, is_feat=True, idx2BN=1)
        else:
            feature_eachlayer_s, logit_s, features_s = model_s(input_norm[:split_idx], mask_list, features_s, is_feat=True, idx2BN=0)
       
        mask_list = copy.deepcopy(mask_list_copy)

        with torch.no_grad():
            features_s = [f.detach() for f in features_s]

            features_t = []
            features_t_adv = []
            features_t_aug = []
            feature_diff = []

            if opt.augment_mode == 'augmix':
                feature_eachlayer_t, logit_t, features_t = model_t(input_norm, features_t, is_feat=True, idx2BN=0)
                feature_eachlayer_t_aug, logit_t_aug, features_t_aug = model_t(input_aug_norm[idx1:idx2], features_t_aug, is_feat=True, idx2BN=1)
                features_t = [f[:idx1].detach() for f in features_t]
            else:
                feature_eachlayer_t, logit_t, features_t = model_t(input_norm, features_t, is_feat=True, idx2BN=0)
                features_t = [f[:split_idx].detach() for f in features_t]

            feature_eachlayer_t = [f.detach() for f in feature_eachlayer_t]                       
    
        # cls + kl div
        if opt.augment_mode == 'augmix':
            loss_cls = criterion_cls(logit_s, target[:idx1])
            loss_div = criterion_div(logit_s, logit_t[:idx1])
            loss_div_aug = criterion_div(logit_s_aug, logit_t[idx1:idx2]) # distill teacher o/p for nat images
            # loss_div_aug = criterion_div(logit_s_aug, logit_t_aug) # distill teacher output for augmix images
        else:
            loss_cls = criterion_cls(logit_s, target[:split_idx])
            loss_div = criterion_div(logit_s, logit_t[:split_idx])

        if attacker:
             # adversarial example computed wrt student model 
            with ctx_noparamgrad_and_eval(model_s):
                if opt.augment_mode == 'augmix':
                    if opt.robust_train_mode == 'rslad':
                        input_adv = attacker.attack(model_s, input[idx2:], labels=logit_t[idx2:], targets=None, mask_list=mask_list, idx2BN=2, dataset=opt.dataset)
                    else:
                        input_adv = attacker.attack(model_s, input[idx2:], labels=target[idx2:], targets=None, mask_list=mask_list, idx2BN=2, dataset=opt.dataset)
                else:
                    if opt.robust_train_mode == 'rslad':
                        input_adv = attacker.attack(model_s, input[split_idx:], labels=logit_t[split_idx:], targets=None, mask_list=mask_list, idx2BN=2, dataset=opt.dataset)
                    else:
                        input_adv = attacker.attack(model_s, input[split_idx:], labels=target[split_idx:], targets=None, mask_list=mask_list, idx2BN=2, dataset=opt.dataset)
            input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
            model_s.train()
            mask_list = copy.deepcopy(mask_list_copy)
            for mask_index, mask in enumerate(mask_list):
                mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input_adv.shape)[0]) # b c h w
    
            features_s_adv = []
            feature_eachlayer_s_adv, logit_s_adv, features_s_adv = model_s(input_adv_norm, mask_list, features_s_adv, is_feat=True, idx2BN=2)
            mask_list = copy.deepcopy(mask_list_copy)
            features_t_adv = []
            feature_eachlayer_t_adv, logit_t_adv, features_t_adv = model_t(input_adv_norm, features_t_adv, is_feat=True, idx2BN=2)
            
            # loss_div_adv = criterion_div(logit_s_adv, logit_t_adv) # distill teacher o/p for adv images
            if opt.augment_mode == 'augmix':
                loss_div_adv = criterion_div(logit_s_adv, logit_t[idx2:]) # distill teacher o/p for nat images
            else:
                loss_div_adv = criterion_div(logit_s_adv, logit_t[split_idx:]) # distill teacher o/p for nat images
            
        
        # # Feature difference calculation for mask 
        with torch.no_grad():
            assert len(features_t) == len(features_s)
            # if opt.use_l2_norm:
            #     feature_diff = [(features_t[i] - features_s[i]) for i in range(len(features_s))]
            feature_diff = [torch.abs(features_t[i] - features_s[i]) for i in range(len(features_s))]
            feature_diff = [reduce(feature, 'b c h w -> c h w', 'sum') for feature in feature_diff]
            if len(mask_epoch) == 0:
                mask_epoch = feature_diff
            else:
                mask_epoch = [mask_epoch[i] + feature_diff[i] for i in range(len(feature_diff))]
            


        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'attention':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        else:
            raise NotImplementedError(opt.distill)

        # loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        
        # RSLAD: Robust soft label adversarial distilllation loss
        if opt.robust_train_mode == 'rslad':
            if opt.augment_mode == 'augmix':
                loss = (loss_div + loss_div_aug + loss_div_adv)/3
            else:
                loss = 0.5 * loss_div + 0.5 * loss_div_adv
        else:
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        
        if opt.augment_mode == 'augmix':
            acc1, acc5 = accuracy(logit_s, target[:idx1], topk=(1, 5))
            racc1, racc5 = accuracy(logit_s_adv, target[idx2:], topk=(1, 5))
        else:
            acc1, acc5 = accuracy(logit_s, target[:split_idx], topk=(1, 5))
            racc1, racc5 = accuracy(logit_s_adv, target[split_idx:], topk=(1, 5))
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
    print(' * AdvAcc@1 {rtop1.avg:.3f} AdvAcc@5 {rtop5.avg:.3f}'
          .format(rtop1=rtop1, rtop5=rtop5))
        
    assert len(sensitivity_list_final) == len(mask_epoch)
    mask_epoch = [topkmask(mask_epoch[i], sensitivity_list_final[i]) for i in range(len(mask_epoch))]
    
    return top1.avg, rtop1.avg, losses.avg, mask_epoch


def train_distill_stage2_3BN(epoch, train_loader, module_list, criterion_list, optimizer, opt, mask_list, mask_weight=None, attacker=None):
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
    mask_epoch = []
    for idx, data in enumerate(train_loader):
    #for idx, (input, target) in enumerate(train_loader):

        if opt.augment_mode == 'augmix':
            data_nat, data_aug = data
            input, target, index = data_nat
            input_aug, _, _ = data_aug
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        if opt.augment_mode == 'augmix':
            idx1 = input.shape[0] // 3
            idx2 = idx1 * 2
        else:
            split_idx = input.shape[0] // 2

        for mask_index, mask in enumerate(mask_list):
            if opt.augment_mode == 'augmix':
                mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = idx1) # b c h w
            else:
                mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = split_idx) # b c h w

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            mask_list = [mask.cuda() for mask in mask_list]
            
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()
            if opt.augment_mode == 'augmix':
                input_aug = input_aug.cuda()
        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        
        input_norm = input
        if attacker:
            input_norm = attacker.normalize(input, dataset=opt.dataset)
            
        features_s = []
        features_s_aug = []

        if opt.augment_mode == 'augmix':
            input_aug_norm = attacker.normalize(input_aug, dataset=opt.dataset)
            feature_eachlayer_s, logit_s, features_s = model_s(input_norm[:idx1], mask_list, features_s, is_feat=True, idx2BN=0)
            mask_list = copy.deepcopy(mask_list_copy)
            for mask_index, mask in enumerate(mask_list):
                mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = idx1) # b c h w
            feature_eachlayer_s_aug, logit_s_aug, _ = model_s(input_aug_norm[idx1:idx2], mask_list, features_s_aug, is_feat=True, idx2BN=1)
            # feature_eachlayer_s_aug, logit_s_aug, _ = model_s(input_aug_norm[:split_idx], mask_list, features_s_aug, is_feat=True, idx2BN=0)
        else:
            feature_eachlayer_s, logit_s, _ = model_s(input_norm[:split_idx], mask_list, features_s, is_feat=True, idx2BN=0)
        
        mask_list = copy.deepcopy(mask_list_copy)

        with torch.no_grad():
            features_t = []
            features_t_adv = []
            features_t_aug = []

            if opt.augment_mode == 'augmix':
                feature_eachlayer_t, logit_t, _ = model_t(input_norm, features_t, is_feat=True, idx2BN=0)
                feature_eachlayer_t_aug, logit_t_aug, features_t_aug = model_t(input_aug_norm[idx1:idx2], features_t_aug, is_feat=True, idx2BN=1)
                # feature_eachlayer_t_aug, logit_t_aug, features_t_aug = model_t(input_aug_norm[:split_idx], features_t_aug, is_feat=True, idx2BN=0)
                feature_eachlayer_t1 = [f[:idx1].detach() for f in feature_eachlayer_t]
                # feature_eachlayer_t = [f[:split_idx].detach() for f in feature_eachlayer_t]
            else:
                feature_eachlayer_t, logit_t, _ = model_t(input_norm, features_t, is_feat=True, idx2BN=0)
                feature_eachlayer_t = [f[:split_idx].detach() for f in feature_eachlayer_t]

        if opt.augment_mode == 'augmix':
            loss_cls = criterion_cls(logit_s, target[:idx1])
            loss_cls_aug = criterion_cls(logit_s_aug, target[idx1:idx2])
            loss_div = criterion_div(logit_s, logit_t[:idx1])
            loss_div_aug = criterion_div(logit_s_aug, logit_t[idx1:idx2]) # distill teacher o/p for nat images
            # loss_div_aug = criterion_div(logit_s_aug, logit_t[:split_idx]) # distill teacher o/p for nat images
            # loss_div_aug = criterion_div(logit_s_aug, logit_t_aug) # distill teacher o/p for augmix images
        else:
            loss_cls = criterion_cls(logit_s, target[:split_idx])
            loss_div = criterion_div(logit_s, logit_t[:split_idx])
        
        if attacker:
            with ctx_noparamgrad_and_eval(model_s):
                if opt.augment_mode == 'augmix':
                    if 'rslad' in opt.robust_train_mode :
                        input_adv = attacker.attack(model_s, input[idx2:], labels=logit_t[idx2:], targets=None, mask_list=mask_list, idx2BN=2, dataset=opt.dataset)
                        # Exp-1
                        # input_adv = attacker.attack(model_s, input[split_idx:], labels=logit_t[split_idx:], targets=None, mask_list=mask_list, idx2BN=2)
                    else:
                        input_adv = attacker.attack(model_s, input[idx2:], labels=target[idx2:], targets=None, mask_list=mask_list, idx2BN=2, dataset=opt.dataset)
                else:
                    if 'rslad' in opt.robust_train_mode :
                        input_adv = attacker.attack(model_s, input[split_idx:], labels=logit_t[split_idx:], targets=None, mask_list=mask_list, idx2BN=2, dataset=opt.dataset)
                    else:
                        input_adv = attacker.attack(model_s, input[split_idx:], labels=target[split_idx:], targets=None, mask_list=mask_list, idx2BN=2, dataset=opt.dataset)
            model_s.train()
            mask_list = copy.deepcopy(mask_list_copy)
            for mask_index, mask in enumerate(mask_list):
                mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input_adv.shape)[0]) # b c h w

            features_s_adv = []
            input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
            feature_eachlayer_s_adv, logit_s_adv, _ = model_s(input_adv_norm, mask_list, features_s_adv, is_feat=True, idx2BN=2)
            mask_list = copy.deepcopy(mask_list_copy)
            features_t_adv = []
            feature_eachlayer_t_adv, logit_t_adv, features_t_adv = model_t(input_adv_norm, features_t_adv, is_feat=True, idx2BN=2)
            
            loss_cls_adv = criterion_cls(logit_s_adv, target[idx2:])
            # loss_div_adv = criterion_div(logit_s_adv, logit_t_adv) # distill teacher o/p for adv images
            # distill teacher o/p for nat images
            if opt.augment_mode == 'augmix':
                loss_div_adv = criterion_div(logit_s_adv, logit_t[idx2:])
                # loss_div_adv = criterion_div(logit_s_adv, logit_t[split_idx:])
            else:
                loss_div_adv = criterion_div(logit_s_adv, logit_t[split_idx:])
            
        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'attention':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t1[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
            # attention for augmix images
            # g_t = [f[idx1:idx2].detach() for f in feature_eachlayer_t][1:-1]
            # g_s_aug = feature_eachlayer_s_aug[1:-1]
            # loss_group_aug = criterion_kd(g_s_aug, g_t)
            # loss_kd = (sum(loss_group) + sum(loss_group_aug))/2

        else:
            raise NotImplementedError(opt.distill)

        # loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        # if attacker:
        #     loss = loss_div_adv + opt.beta * loss_kd 

        # ARD loss
        if opt.augment_mode == 'augmix':
            # RSLAD: Robust soft label adversarial distilllation loss
            if attacker and opt.robust_train_mode == 'rslad':
                loss = (loss_div + loss_div_aug + loss_div_adv)/3
                # CE + RSLAD
                # loss = (loss_div + loss_div_aug + loss_div_adv)/3 + (loss_cls + loss_cls_aug + loss_cls_adv)/3
                # Exp-1
                # loss = 0.5 * loss_div_aug + 0.5 * loss_div_adv

            # RSLAD+Attention:
            elif attacker and opt.robust_train_mode == 'rslad+attn':
                loss = (loss_div + loss_div_aug + loss_div_adv)/3 + opt.beta * loss_kd
                # Exp-1
                # loss = 0.5 * loss_div_aug + 0.5 * loss_div_adv + opt.beta * loss_kd
        else:
            # RSLAD: Robust soft label adversarial distilllation loss
            if attacker and opt.robust_train_mode == 'rslad':
                loss = (1 - opt.alpha) * loss_div + opt.alpha * loss_div_adv
            # RSLAD+Attention:
            elif attacker and opt.robust_train_mode == 'rslad+attn':
                loss = (1 - opt.alpha) * loss_div + opt.alpha * loss_div_adv + opt.beta * loss_kd 
            else:
                loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        # RSLAD+Attention+Adv_Attention:
        # elif attacker and opt.robust_train_mode == 'rslad+attn+advattn':
        #     with torch.no_grad():
        #         features_t_adv = []
        #         feature_eachlayer_t_adv, logit_t_adv, features_t_adv = model_t(input_adv_norm, features_t_adv, is_feat=True)
        #     g_s_adv = feature_eachlayer_s_adv[1:-1]
        #     g_t_adv = feature_eachlayer_t_adv[1:-1]
        #     loss_group = criterion_kd(g_s_adv, g_t_adv)
        #     loss_kd_adv = sum(loss_group)
        #     loss = (1 - opt.alpha) * loss_div + opt.alpha * loss_div_adv + opt.beta * 0.5 * loss_kd + opt.beta * 0.5 * loss_kd_adv
        # elif opt.robust_train_mode == 'augmix':
        #     # loss = loss_cls
        #     # loss = opt.gamma * loss_cls + opt.alpha * loss_div
        #     loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        
        if opt.augment_mode == 'augmix':
            acc1, acc5 = accuracy(logit_s, target[:idx1], topk=(1, 5))
            racc1, racc5 = accuracy(logit_s_adv, target[idx2:], topk=(1, 5))
        else:
            acc1, acc5 = accuracy(logit_s, target[:split_idx], topk=(1, 5))
            racc1, racc5 = accuracy(logit_s_adv, target[split_idx:], topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))
        top5.update(acc5, input.size(0))
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
    
    return top1.avg, rtop1.avg, losses.avg, mask_epoch


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

    # with torch.no_grad():
    end = time.time()
    for idx, (input, target) in enumerate(val_loader):

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if mask_list is not None:
                mask_list = [mask.cuda() for mask in mask_list]
            
        features = []

        # compute output
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


def train_adversarial_3BN(epoch, train_loader, model, criterion_list, optimizer, opt, attacker):
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
        idx1 = input.shape[0] // 3
        idx2 = idx1 * 2
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if opt.augment_mode == 'augmix':
                input_aug = input_aug.float().cuda()

        features = []
        # ===================forward=====================
        input_norm = attacker.normalize(input, dataset=opt.dataset)

        if opt.use3BN:
            if opt.augment_mode == 'augmix':
                # input_nat_norm = attacker.normalize(input_nat, dataset=opt.dataset)
                output_nat, _ = model(input_norm[:idx1], features, is_feat=False, idx2BN=0)
            output, _ = model(input_aug[idx1:idx2], features, is_feat=False, idx2BN=1)
        else:
            output, _ = model(input_norm, features, is_feat=False)

        #=================== attack =====================
        with ctx_noparamgrad_and_eval(model):
            if opt.use3BN:
                input_adv = attacker.attack(model, input[idx2:], labels=target[idx2:], targets=None, idx2BN=2, dataset=opt.dataset)
            else:
                if opt.robust_train_mode == 'trades':
                    input_adv = attacker.attack(model, input, labels=output, targets=None, dataset=opt.dataset) #TRADES
                else:
                    input_adv = attacker.attack(model, input, labels=target, targets=None, dataset=opt.dataset)

        input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
        if opt.use3BN:
            output_adv, _ = model(input_adv_norm, features, is_feat=False, idx2BN=2)
        else:
            output_adv, _ = model(input_adv_norm, features, is_feat=False)

        if opt.use3BN:
            # if opt.augment_mode == 'augmix':
            #     loss = 0.5 * (criterion_cls(output_nat, target) + criterion_cls(output, target))/2 + 0.5 * criterion_cls(output_adv, target)
            # else:
            loss = (criterion_cls(output_nat, target[:idx1]) + criterion_cls(output, target[idx1:idx2]) + criterion_cls(output_adv, target[idx2:]))/3
        else:
            # TRADES Loss Function
            if opt.robust_train_mode == 'trades':
                loss = criterion_cls(output, target) + opt._lambda * criterion_div(output_adv, output)
            else:
                loss = 0.5 * criterion_cls(output, target) + 0.5 * criterion_cls(output_adv, target)

        
        if opt.use3BN:
            acc1, acc5 = accuracy(output, target[:idx1], topk=(1, 5))
            racc1, racc5 = accuracy(output_adv, target[idx2:], topk=(1, 5))
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


def train_comp_robust_3BN(epoch, train_loader, model, criterion_list, optimizer, opt, attacker=None):
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
        if opt.use3BN:
            output, _ = model(input_norm, features, is_feat=False, idx2BN=0)
            output_deepaug, _ = model(input_deepaug, features, is_feat=False, idx2BN=1)
            output_texture_debias, _ = model(input_texture_debias, features, is_feat=False, idx2BN=1)
        else:
            output, _ = model(input_norm, features, is_feat=False)
            output_deepaug, _ = model(input_deepaug, features, is_feat=False)
            output_texture_debias, _ = model(input_texture_debias, features, is_feat=False)

        #=================== attack =====================
        if attacker:
            with ctx_noparamgrad_and_eval(model):
                if opt.use3BN:
                    input_adv = attacker.attack(model, input, labels=target, targets=None, idx2BN=2, dataset=opt.dataset)
                else:
                    if opt.robust_train_mode == 'trades':
                        input_adv = attacker.attack(model, input, labels=output, targets=None, dataset=opt.dataset) #TRADES
                    else:
                        input_adv = attacker.attack(model, input, labels=target, targets=None, dataset=opt.dataset)

            input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
            if opt.use3BN:
                output_adv, _ = model(input_adv_norm, features, is_feat=False, idx2BN=2)
            else:
                output_adv, _ = model(input_adv_norm, features, is_feat=False)

        # TRADES Loss Function
        if opt.use3BN:
            loss = 0.5 * (criterion_cls(output, target) + criterion_cls(output_deepaug, target_deepaug) + criterion_cls(output_texture_debias, target_texture_debias))/3 + 0.5 * criterion_cls(output_adv, target)
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
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))
        top5.update(acc5, input.size(0))

        if attacker:
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

def validate_3BN(val_loader, model, criterion, opt, mask_list = None, attacker=None, dual_masking=False, mask_list_adv=None):
    """validation"""

    val_lambdas = [0.0, 2.0] # For lambda=0.0, BN for natural images is used; lambda=1.0 BN for adv images is used
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
            idx2BN = val_lambda
            # compute output
            if mask_list == None:
                input_norm = attacker.normalize(input, dataset=opt.dataset)
                output, _ = model(input_norm, features, is_feat = False, idx2BN=idx2BN)
                with ctx_noparamgrad_and_eval(model):
                    # input_adv = attack(input, labels=target)
                    input_adv = attacker.attack(model, input, labels=target, targets=None, idx2BN=idx2BN, dataset=opt.dataset)
                input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
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


def train_distill_stage1_2BN(epoch, train_loader, module_list, criterion_list, optimizer, opt, mask_list, sensitivity_list_final, attacker=None):
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
    
    if torch.cuda.is_available():
        mask_list = [mask.cuda() for mask in mask_list]
    mask_list_copy = copy.deepcopy(mask_list) #c, h, w
    mask_epoch = []
    for idx, data in enumerate(train_loader):
    #for idx, (input, target) in enumerate(train_loader):

        if opt.augment_mode == 'augmix':
            data_nat, data_aug = data
            input, target, index = data_nat
            input_aug, _, _ = data_aug
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        split_idx = input.shape[0] // 2

        for mask_index, mask in enumerate(mask_list):
            mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = split_idx) # b c h w

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            mask_list = [mask.cuda() for mask in mask_list]
            
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()
            if opt.augment_mode == 'augmix':
                input_aug = input_aug.float().cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True

        input_norm = input
        if attacker:
            input_norm = attacker.normalize(input, dataset=opt.dataset)

        features_s = []
        features_s_aug = []

        feature_eachlayer_s, logit_s, features_s = model_s(input_norm[:split_idx], mask_list, features_s, is_feat=True, idx2BN=0)
        if opt.augment_mode == 'augmix':
            # input_aug_norm = attacker.normalize(input_aug, dataset=opt.dataset)
            mask_list = copy.deepcopy(mask_list_copy)
            for mask_index, mask in enumerate(mask_list):
                mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = split_idx) # b c h w
            feature_eachlayer_s_aug, logit_s_aug, _ = model_s(input_aug[:split_idx], mask_list, features_s_aug, is_feat=True, idx2BN=0)
       
        mask_list = copy.deepcopy(mask_list_copy)

        with torch.no_grad():
            features_s = [f.detach() for f in features_s]

            features_t = []
            features_t_adv = []
            features_t_aug = []
            feature_diff = []

            if opt.augment_mode == 'augmix':
                feature_eachlayer_t_aug, logit_t_aug, features_t_aug = model_t(input_aug[:split_idx], features_t_aug, is_feat=True, idx2BN=0)
            feature_eachlayer_t, logit_t, features_t = model_t(input_norm, features_t, is_feat=True, idx2BN=0)
            features_t = [f[:split_idx].detach() for f in features_t]
            feature_eachlayer_t = [f.detach() for f in feature_eachlayer_t]                       
    
        # cls + kl div
        loss_cls = criterion_cls(logit_s, target[:split_idx])
        loss_div = criterion_div(logit_s, logit_t[:split_idx])
        if opt.augment_mode == 'augmix':
            loss_cls_aug = criterion_cls(logit_s_aug, target[:split_idx])
            loss_div_aug = criterion_div(logit_s_aug, logit_t[:split_idx]) # distill teacher o/p for nat images
            # loss_div_aug = criterion_div(logit_s_aug, logit_t_aug) # distill teacher output for augmix images
            
        if attacker:
             # adversarial example computed wrt student model 
            with ctx_noparamgrad_and_eval(model_s):
                if 'rslad' in opt.robust_train_mode:
                    input_adv = attacker.attack(model_s, input[split_idx:], labels=logit_t[split_idx:], targets=None, mask_list=mask_list, idx2BN=1, dataset=opt.dataset)
                else:
                    input_adv = attacker.attack(model_s, input[split_idx:], labels=target[split_idx:], targets=None, mask_list=mask_list, idx2BN=1, dataset=opt.dataset)
            input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
            model_s.train()
            mask_list = copy.deepcopy(mask_list_copy)
            for mask_index, mask in enumerate(mask_list):
                mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input_adv.shape)[0]) # b c h w
    
            features_s_adv = []
            feature_eachlayer_s_adv, logit_s_adv, features_s_adv = model_s(input_adv_norm, mask_list, features_s_adv, is_feat=True, idx2BN=1)
            mask_list = copy.deepcopy(mask_list_copy)
            # features_t_adv = []
            # feature_eachlayer_t_adv, logit_t_adv, features_t_adv = model_t(input_adv_norm, features_t_adv, is_feat=True, idx2BN=1)
            
            # loss_div_adv = criterion_div(logit_s_adv, logit_t_adv) # distill teacher o/p for adv images
            loss_div_adv = criterion_div(logit_s_adv, logit_t[split_idx:]) # distill teacher o/p for nat images
            loss_cls_adv = criterion_cls(logit_s_adv, target[split_idx:])
        
        # # Feature difference calculation for mask 
        with torch.no_grad():
            assert len(features_t) == len(features_s)
            # if opt.use_l2_norm:
            #     feature_diff = [(features_t[i] - features_s[i]) for i in range(len(features_s))]
            feature_diff = [torch.abs(features_t[i] - features_s[i]) for i in range(len(features_s))]
            feature_diff = [reduce(feature, 'b c h w -> c h w', 'sum') for feature in feature_diff]
            if len(mask_epoch) == 0:
                mask_epoch = feature_diff
            else:
                mask_epoch = [mask_epoch[i] + feature_diff[i] for i in range(len(feature_diff))]
            


        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'attention':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        else:
            raise NotImplementedError(opt.distill)

        # loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        
        # RSLAD: Robust soft label adversarial distilllation loss
        if opt.robust_train_mode == 'rslad':
            if opt.augment_mode == 'augmix':
                loss = (loss_div + loss_div_aug + loss_div_adv)/3
            else:
                loss = 0.5 * loss_div + 0.5 * loss_div_adv
        elif opt.robust_train_mode == 'rslad+ce':
            # CE+RSLAD
            loss = opt.gamma * (loss_cls + loss_cls_aug + loss_cls_adv)/3 + opt.alpha * (loss_div + loss_div_aug + loss_div_adv)/3 
        else:
            # KL Divergence loss
            # loss = (loss_div + loss_div_aug + loss_div_adv)/3
            # ARD
            # loss = (loss_cls + loss_cls_aug + loss_div_adv)/3
            # PGDAT
            loss = (loss_cls + loss_cls_aug + loss_cls_adv)/3 
            # loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
            # loss = opt.gamma * (loss_cls + loss_cls_aug + loss_cls_adv)/3 + opt.alpha * (loss_div + loss_div_aug + loss_div_adv)/3 + opt.beta * loss_kd
        
        acc1, acc5 = accuracy(logit_s, target[:split_idx], topk=(1, 5))
        racc1, racc5 = accuracy(logit_s_adv, target[split_idx:], topk=(1, 5))
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
    print(' * AdvAcc@1 {rtop1.avg:.3f} AdvAcc@5 {rtop5.avg:.3f}'
          .format(rtop1=rtop1, rtop5=rtop5))
        
    assert len(sensitivity_list_final) == len(mask_epoch)
    mask_epoch = [topkmask(mask_epoch[i], sensitivity_list_final[i]) for i in range(len(mask_epoch))]
    
    return top1.avg, rtop1.avg, losses.avg, mask_epoch


def train_distill_stage2_2BN(epoch, train_loader, module_list, criterion_list, optimizer, opt, mask_list, mask_weight=None, attacker=None):
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
    mask_epoch = []
    for idx, data in enumerate(train_loader):
    #for idx, (input, target) in enumerate(train_loader):

        if opt.augment_mode == 'augmix':
            data_nat, data_aug = data
            input, target, index = data_nat
            input_aug, _, _ = data_aug
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        split_idx = input.shape[0] // 2

        for mask_index, mask in enumerate(mask_list):
            mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = split_idx) # b c h w

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            mask_list = [mask.cuda() for mask in mask_list]
            
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()
            if opt.augment_mode == 'augmix':
                input_aug = input_aug.float().cuda()
        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        
        input_norm = input
        if attacker:
            input_norm = attacker.normalize(input, dataset=opt.dataset)
            
        features_s = []
        features_s_aug = []

        feature_eachlayer_s, logit_s, features_s = model_s(input_norm[:split_idx], mask_list, features_s, is_feat=True, idx2BN=0)
        if opt.augment_mode == 'augmix':
            # input_aug_norm = attacker.normalize(input_aug, dataset=opt.dataset)
            mask_list = copy.deepcopy(mask_list_copy)
            for mask_index, mask in enumerate(mask_list):
                mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = split_idx) # b c h w
            feature_eachlayer_s_aug, logit_s_aug, _ = model_s(input_aug[:split_idx], mask_list, features_s_aug, is_feat=True, idx2BN=0)
                
        mask_list = copy.deepcopy(mask_list_copy)

        with torch.no_grad():
            features_t = []
            features_t_adv = []
            features_t_aug = []

            if opt.augment_mode == 'augmix':
                feature_eachlayer_t_aug, logit_t_aug, features_t_aug = model_t(input_aug[:split_idx], features_t_aug, is_feat=True, idx2BN=0)
                feature_eachlayer_t_aug = [f[:split_idx].detach() for f in feature_eachlayer_t_aug]
            feature_eachlayer_t, logit_t, _ = model_t(input_norm, features_t, is_feat=True, idx2BN=0)
            feature_eachlayer_t = [f[:split_idx].detach() for f in feature_eachlayer_t]

        if opt.augment_mode == 'augmix':
            loss_cls_aug = criterion_cls(logit_s_aug, target[:split_idx])
            loss_div_aug = criterion_div(logit_s_aug, logit_t[:split_idx]) # distill teacher o/p for nat images
            # loss_div_aug = criterion_div(logit_s_aug, logit_t_aug) # distill teacher o/p for augmix images
        loss_cls = criterion_cls(logit_s, target[:split_idx])
        loss_div = criterion_div(logit_s, logit_t[:split_idx])
    
        if attacker:
            with ctx_noparamgrad_and_eval(model_s):
                if 'rslad' in opt.robust_train_mode :
                    input_adv = attacker.attack(model_s, input[split_idx:], labels=logit_t[split_idx:], targets=None, mask_list=mask_list, idx2BN=1, dataset=opt.dataset)
                else:
                    input_adv = attacker.attack(model_s, input[split_idx:], labels=target[split_idx:], targets=None, mask_list=mask_list, idx2BN=1, dataset=opt.dataset)
            model_s.train()
            mask_list = copy.deepcopy(mask_list_copy)
            for mask_index, mask in enumerate(mask_list):
                mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input_adv.shape)[0]) # b c h w

            features_s_adv = []
            input_adv_norm = attacker.normalize(input_adv, dataset=opt.dataset)
            feature_eachlayer_s_adv, logit_s_adv, _ = model_s(input_adv_norm, mask_list, features_s_adv, is_feat=True, idx2BN=1)
            mask_list = copy.deepcopy(mask_list_copy)
            features_t_adv = []
            # adv attn
            with torch.no_grad():
                feature_eachlayer_t_adv, logit_t_adv, features_t_adv = model_t(input_adv_norm, features_t_adv, is_feat=True, idx2BN=1)
            
            loss_cls_adv = criterion_cls(logit_s_adv, target[split_idx:])
            # loss_div_adv = criterion_div(logit_s_adv, logit_t_adv) # distill teacher o/p for adv images
            loss_div_adv = criterion_div(logit_s_adv, logit_t[split_idx:]) # distill teacher o/p for nat images
            
        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'attention':
            g_s = feature_eachlayer_s[1:-1]
            g_t = feature_eachlayer_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
            # attention for augmix images
            g_s_aug = feature_eachlayer_s_aug[1:-1]
            g_t_aug = feature_eachlayer_t_aug[1:-1]
            loss_group_aug = criterion_kd(g_s_aug, g_t_aug)
            loss_kd_aug = sum(loss_group_aug)
            # attention for adversarial images
            g_s_adv = feature_eachlayer_s_adv[1:-1]
            g_t_adv = feature_eachlayer_t_adv[1:-1]
            loss_group_adv = criterion_kd(g_s_adv, g_t_adv)
            loss_kd_adv = sum(loss_group_adv)
            # g_t = [f[idx1:idx2].detach() for f in feature_eachlayer_t][1:-1]
            # g_s_aug = feature_eachlayer_s_aug[1:-1]
            # loss_group_aug = criterion_kd(g_s_aug, g_t)
            # loss_kd = (sum(loss_group) + sum(loss_group_aug))/2

        else:
            raise NotImplementedError(opt.distill)

        # loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        # if attacker:
        #     loss = loss_div_adv + opt.beta * loss_kd 

        # ARD loss
        if opt.augment_mode == 'augmix':
            # RSLAD: Robust soft label adversarial distilllation loss
            if attacker and opt.robust_train_mode == 'rslad':
                loss = (loss_div + loss_div_aug + loss_div_adv)/3
                # loss = 0.5 * loss_div_aug + 0.5 * loss_div_adv
            elif attacker and opt.robust_train_mode == 'rslad+ce':
                # CE + RSLAD
                loss = opt.gamma * (loss_cls + loss_cls_aug + loss_cls_adv)/3 + opt.alpha * (loss_div + loss_div_aug + loss_div_adv)/3
            # RSLAD+Attention:
            elif attacker and opt.robust_train_mode == 'rslad+attn':
                loss = (loss_div + loss_div_aug + loss_div_adv)/3 + opt.beta * loss_kd
            elif attacker and opt.robust_train_mode == 'rslad+attn+augattn':
                # augmix attention
                loss = (loss_div + loss_div_aug + loss_div_adv)/3 + opt.beta * (loss_kd + loss_kd_aug)/2
            elif attacker and opt.robust_train_mode == 'rslad+attn+augattn+advattn':
                loss = (loss_div + loss_div_aug + loss_div_adv)/3 + opt.beta * (loss_kd + loss_kd_aug + loss_kd_adv)/3            
            elif attacker and opt.robust_train_mode == 'rslad+attn+augattn+ce':
                # CE+RSLAD
                loss = opt.gamma * (loss_cls + loss_cls_aug + loss_cls_adv)/3 + opt.alpha * (loss_div + loss_div_aug + loss_div_adv)/3 + opt.beta * (loss_kd + loss_kd_aug)/2        
            elif attacker and opt.robust_train_mode == 'rslad+attn+augattn+advattn+ce':
                # CE+RSLAD
                loss = opt.gamma * (loss_cls + loss_cls_aug + loss_cls_adv)/3 + opt.alpha * (loss_div + loss_div_aug + loss_div_adv)/3 + opt.beta * (loss_kd + loss_kd_aug + loss_kd_adv)/3       
            else:
                # KL Div loss
                # loss = (loss_div + loss_div_aug + loss_div_adv)/3
                # ARD
                # loss = (loss_cls + loss_cls_aug + loss_div_adv)/3
                # PGDAT
                loss = (loss_cls + loss_cls_aug + loss_cls_adv)/3  
        else:
            # RSLAD: Robust soft label adversarial distilllation loss
            if attacker and opt.robust_train_mode == 'rslad':
                loss = (1 - opt.alpha) * loss_div + opt.alpha * loss_div_adv
            # RSLAD+Attention:
            elif attacker and opt.robust_train_mode == 'rslad+attn':
                loss = (1 - opt.alpha) * loss_div + opt.alpha * loss_div_adv + opt.beta * loss_kd 
            else:
                loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        
        acc1, acc5 = accuracy(logit_s, target[:split_idx], topk=(1, 5))
        racc1, racc5 = accuracy(logit_s_adv, target[split_idx:], topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))
        top5.update(acc5, input.size(0))
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
    
    return top1.avg, rtop1.avg, losses.avg, mask_epoch
