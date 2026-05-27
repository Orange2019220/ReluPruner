import sys
import time
import torch

from .util import AverageMeter, accuracy
from .context import ctx_noparamgrad_and_eval
from einops import reduce, repeat
import copy

def train_distill_stage1(epoch, train_loader, module_list, criterion_list, optimizer, opt, sensitivity_list_final, attacker=None):
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
    #for idx, (input, target) in enumerate(train_loader):

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
        feature_eachlayer_s, logit_s, features_s = model_s(input_norm, features_s, is_feat=True)

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

        if attacker:
             # adversarial example computed wrt student model 
            with ctx_noparamgrad_and_eval(model_s):
                if opt.robust_train_mode == 'rslad':
                    input_adv = attacker.attack(model_s, input, labels=logit_t, targets=None, mask_list=mask_list)
                else:
                    input_adv = attacker.attack(model_s, input, labels=target, targets=None, mask_list=mask_list)
            input_adv_norm = attacker.normalize(input_adv)
            model_s.train()
            
            features_s_adv = []
            feature_eachlayer_s_adv, logit_s_adv, features_s_adv = model_s(input_adv_norm, mask_list, features_s_adv, is_feat=True)
            loss_div_adv = criterion_div(logit_s_adv, logit_t)
            
        
        # # Feature difference calculation for mask 
        # with torch.no_grad():
        #     assert len(features_t) == len(features_s)
        #     # if opt.use_l2_norm:
        #     #     feature_diff = [(features_t[i] - features_s[i]) for i in range(len(features_s))]
        #     if attacker and opt.mask_calculation == 'feat_diff_nat+adv':
        #         features_t_adv = []
        #         feature_eachlayer_t_adv, logit_t_adv, features_t_adv = model_t(input_adv_norm, features_t_adv, is_feat=True)
        #         features_t_adv = [f.detach() for f in features_t_adv]
        #         features_s_adv = [f.detach() for f in features_s_adv]

        #         # feature_eachlayer_t = [f.detach() for f in feature_eachlayer_t]
        #         split_idx = int(opt.batch_size/2)
        #         features_t_comb = [torch.cat([features_t[i][:split_idx], features_t_adv[i][split_idx:]], dim=0) for i in range(len(features_t))]
        #         features_s_comb = [torch.cat([features_s[i][:split_idx], features_s_adv[i][split_idx:]], dim=0) for i in range(len(features_s))]        
        #         feature_diff = [torch.abs(features_t_comb[i] - features_s_comb[i]) for i in range(len(features_s))]
        #         feature_diff = [reduce(feature, 'b c h w -> c h w', 'sum') for feature in feature_diff]
            
        #     elif attacker and opt.mask_calculation == 'feat_diff_adv':
        #         features_t_adv = []
        #         feature_eachlayer_t_adv, logit_t_adv, features_t_adv = model_t(input_adv_norm, features_t_adv, is_feat=True)
        #         features_t_adv = [f.detach() for f in features_t_adv]
        #         features_s_adv = [f.detach() for f in features_s_adv]
                
        #         feature_diff = [torch.abs(features_t_adv[i] - features_s_adv[i]) for i in range(len(features_s_adv))]
        #         feature_diff = [reduce(feature, 'b c h w -> c h w', 'sum') for feature in feature_diff]
            
        #     else:
        #         feature_diff = [torch.abs(features_t[i] - features_s[i]) for i in range(len(features_s))]
        #         feature_diff = [reduce(feature, 'b c h w -> c h w', 'sum') for feature in feature_diff]
        #     if len(mask_epoch) == 0:
        #         mask_epoch = feature_diff
        #     else:
        #         mask_epoch = [mask_epoch[i] + feature_diff[i] for i in range(len(feature_diff))]
            


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
        
    # assert len(sensitivity_list_final) == len(mask_epoch)
    # mask_epoch = [topkmask(mask_epoch[i], sensitivity_list_final[i]) for i in range(len(mask_epoch))]
    
    if attacker:
        return top1.avg, rtop1.avg, losses.avg
    return top1.avg, 0.0, losses.avg


def train_distill_stage2(epoch, train_loader, module_list, criterion_list, optimizer, opt, attacker=None):
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

    for idx, data in enumerate(train_loader):
    #for idx, (input, target) in enumerate(train_loader):

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
            input_norm = attacker.normalize(input, dataset=opt.dataset)

        features_s = []
        feature_eachlayer_s, logit_s, _ = model_s(input_norm, features_s, is_feat=True)

        with torch.no_grad():
            features_t = []
            feature_eachlayer_t, logit_t, _ = model_t(input_norm, features_t, is_feat=True)
            feature_eachlayer_t = [f.detach() for f in feature_eachlayer_t]
            
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        
        if attacker:
            with ctx_noparamgrad_and_eval(model_s):
                if opt.robust_train_mode == 'rslad':
                    input_adv = attacker.attack(model_s, input, labels=logit_t, targets=None, mask_list=mask_list)
                else:
                    input_adv = attacker.attack(model_s, input, labels=target, targets=None, mask_list=mask_list)
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
        return top1.avg, rtop1.avg, losses.avg, None
    return top1.avg, 0.0, losses.avg, None
