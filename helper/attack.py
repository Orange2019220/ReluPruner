from torch.autograd import Variable
import torch
import torch.nn as nn
import copy
from einops import reduce, repeat
import torch.nn.functional as F
# from torchattacks.attack.autoattack import AutoAttack

def linf_clamp(x, _min, _max):
    '''
    Inplace linf clamping on Tensor x.

    Args:
        x: Tensor. shape=(N,C,W,H)
        _min: Tensor with same shape as x.
        _max: Tensor with same shape as x.
    '''
    idx = x.data < _min
    x.data[idx] = _min[idx]
    idx = x.data > _max
    x.data[idx] = _max[idx]

    return x

class PGD():
    def __init__(self, eps=8/255, steps=7, alpha=2/255, criterion=None):
        '''
        Args:
            eps: float. noise bound.
            targeted: bool. If Ture, do targeted attack.
            PR: true if using partial relu models.
        '''
        self.eps = eps
        self.alpha = alpha
        # self.alpha = alpha if alpha else min(eps * 1.25, eps + 4/255) / steps 
        self.steps = steps
        self.criterion = criterion if criterion else nn.CrossEntropyLoss(reduction="sum") 
        self.normalization_used = None

    def normalize(self, inputs, dataset='cifar10'):
        mean = {
            'cifar10': (0.4914, 0.4822, 0.4465),
            'cifar100': (0.5071, 0.4867, 0.4408),
            'tiny_imagenet': (0.0, 0.0, 0.0),
        }

        std = {
            'cifar10': (0.2023, 0.1994, 0.2010),
            'cifar100': (0.2675, 0.2565, 0.2761),
            'tiny_imagenet': (1.0, 1.0, 1.0),
        }
        mean = mean[dataset]
        std = std[dataset]
        mean = torch.tensor(mean).reshape(1, 3, 1, 1).cuda()
        std = torch.tensor(std).reshape(1, 3, 1, 1).cuda()
        return (inputs - mean) / std
       
    def attack(self, model, x, labels=None, targets=None, mask_list=None, idx2BN=None, dataset='cifar10', mask_type=None):
        '''
        Args:
            x: Tensor. Original images. size=(N,C,W,H)
            model: nn.Module. The model to be attacked.
            gtlabels: Tensor. ground truth labels for x. size=(N,). Useful only under untargeted attack.
            targets: Tensor. target attack class for x. size=(N,). Useful only under targeted attack.

        Return:
            x_adv: Tensor. Adversarial images. size=(N,C,W,H)
        '''
        
        model.eval()

        # initialize x_adv:
        x_adv = x.clone()
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * self.eps # random initialize
        x_adv = torch.clamp(x_adv, 0, 1) # clamp to RGB range [0,1]
        x_adv = Variable(x_adv.cuda(), requires_grad=True)

        features = []
        mask_list_copy = None
        if mask_list: mask_list_copy = copy.deepcopy(mask_list)
        for t in range(self.steps):
            if mask_list: 
                for mask_index, mask in enumerate(mask_list):
                    mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(x_adv.shape)[0]) # b c h w
                if not self.normalization_used:
                    x_norm = self.normalize(x_adv, dataset=dataset)
                logits_adv, _ = model(x_norm, mask_list, features, is_feat=False, idx2BN=idx2BN)
                # logits_adv, _ = model(x_norm, mask_list, features, is_feat=False, idx2BN=idx2BN, mask_type=mask_type)
                mask_list = copy.deepcopy(mask_list_copy)
            else: 
                # logits_adv = model(x_adv)
                if not self.normalization_used:
                    x_norm = self.normalize(x_adv, dataset=dataset)
                logits_adv, _ = model(x_norm, features, is_feat=False, idx2BN=idx2BN)
                # logits_adv, _ = model(x_norm, features, is_feat=False, idx2BN=idx2BN, mask_type=mask_type)
            if targets is not None:
                loss_adv = - self.criterion(logits_adv, targets)
            else: # untargeted attack
                loss_adv = self.criterion(logits_adv, labels)
            grad_adv = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]
            x_adv.data.add_(self.alpha * torch.sign(grad_adv.data)) # gradient assend by Sign-SGD
            x_adv = linf_clamp(x_adv, _min=x-self.eps, _max=x+self.eps) # clamp to linf ball centered by x
            x_adv = torch.clamp(x_adv, 0, 1) # clamp to RGB range [0,1]
            
        return x_adv


    # def attack(self, model, data, labels, targets=None, k=20, a=0.01, random_start=True,
    #            d_min=0, d_max=1, mask_list=None):
        
    #     model.eval()
    #     # perturbed_data = copy.deepcopy(data)
    #     perturbed_data = data.clone()
                                                
    #     perturbed_data.requires_grad = True
        
    #     data_max = data + self.eps
    #     data_min = data - self.eps
    #     data_max.clamp_(d_min, d_max)
    #     data_min.clamp_(d_min, d_max)

    #     if random_start:
    #         with torch.no_grad():
    #             perturbed_data.data = data + perturbed_data.uniform_(-1*self.eps, self.eps)
    #             perturbed_data.data.clamp_(d_min, d_max)
        
    #     features = []
    #     for _ in range(k):
            
    #         output, _ = model( perturbed_data, features, is_feat=False )
    #         loss = F.cross_entropy(output, labels)
            
    #         if perturbed_data.grad is not None:
    #             perturbed_data.grad.data.zero_()
            
    #         loss.backward()
    #         data_grad = perturbed_data.grad.data
            
    #         with torch.no_grad():
    #             perturbed_data.data += a * torch.sign(data_grad)
    #             perturbed_data.data = torch.max(torch.min(perturbed_data, data_max),
    #                                             data_min)
    #     perturbed_data.requires_grad = False
        
    #     return perturbed_data

class FGSM():
    def __init__(self, eps=8/255, alpha=2/255, targeted=False, criterion=None):
        '''
        Args:
            eps: float. noise bound.
            targeted: bool. If Ture, do targeted attack.
            PR: true if using partial relu models.
        '''
        self.eps = eps 
        self.targeted = targeted
        self.alpha = alpha
        # self.alpha = alpha if alpha else min(eps * 1.25, eps + 4/255) 
        self.criterion = criterion if criterion else nn.CrossEntropyLoss(reduction="sum") 

    def normalize(self, inputs, dataset='cifar10'):
        mean = {
            'cifar10': (0.4914, 0.4822, 0.4465),
            'cifar100': (0.5071, 0.4867, 0.4408),
            'tiny_imagenet': (0.0, 0.0, 0.0),
        }

        std = {
            'cifar10': (0.2023, 0.1994, 0.2010),
            'cifar100': (0.2675, 0.2565, 0.2761),
            'tiny_imagenet': (1.0, 1.0, 1.0),
        }
        mean = mean[dataset]
        std = std[dataset]
        mean = torch.tensor(mean).reshape(1, 3, 1, 1).cuda()
        std = torch.tensor(std).reshape(1, 3, 1, 1).cuda()
        return (inputs - mean) / std
       
    def attack(self, model, x, labels, targets=None, mask_list=None, idx2BN=None, dataset='cifar10'):
        '''
        Args:
            x: Tensor. Original images. size=(N,C,W,H)
            model: nn.Module. The model to be attacked.
            gtlabels: Tensor. ground truth labels for x. size=(N,). Useful only under untargeted attack.
            targets: Tensor. target attack class for x. size=(N,). Useful only under targeted attack.

        Return:
            x_adv: Tensor. Adversarial images. size=(N,C,W,H)
        '''
        
        model.eval()

        # initialize x_adv:
        x_adv = x.clone()
        # x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * self.eps # random initialize
        # x_adv = torch.clamp(x_adv, 0, 1) # clamp to RGB range [0,1]
        # x_adv = Variable(x_adv.cuda(), requires_grad=True)
        x_adv.requires_grad = True

        mask_list_copy = None
        features = []
        if mask_list: 
            mask_list_copy = copy.deepcopy(mask_list)
            # mask_list = [mask.cuda() for mask in mask_list]
            for mask_index, mask in enumerate(mask_list):
                mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(x_adv.shape)[0]) # b c h w
            x_norm = self.normalize(x_adv, dataset=dataset)
            logits_adv, _ = model(x_norm, mask_list, features, is_feat=False, idx2BN=idx2BN)
        else: 
            x_norm = self.normalize(x_adv, dataset=dataset)
            logits_adv, _ = model(x_norm, features, is_feat=False, idx2BN=idx2BN)
        if targets is not None:
            loss_adv = - self.criterion(logits_adv, targets)
        else: # untargeted attack
            loss_adv = self.criterion(logits_adv, labels)
        grad_adv = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]
        x_adv.data.add_(self.eps * torch.sign(grad_adv.data)) # gradient assend by Sign-SGD
        # x_adv = linf_clamp(x_adv, _min=x-self.eps, _max=x+self.eps) # clamp to linf ball centered by x
        x_adv = torch.clamp(x_adv, 0, 1) # clamp to RGB range [0,1]
            
        return x_adv


# class Autoattack(AutoAttack):
#     def __init__(self, model):
#         super().__init__("AutoAttack", model)


#     def get_logits(self, inputs, labels=None, mask_list=None, idx2BN=None, *args, **kwargs):
#         if self._normalization_applied is False:
#             inputs = self.normalize(inputs)
#         # logits = self.model(inputs)
#         # return logits
#         features = []
#         if mask_list: 
#             for mask_index, mask in enumerate(mask_list):
#                 mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(inputs.shape)[0]) # b c h w
#             logits_adv, _ = self.model(inputs, mask_list, features, is_feat=False, idx2BN=idx2BN)
#         else: 
#             logits_adv, _ = self.model(inputs, features, is_feat=False, idx2BN=idx2BN)
#         return logits_adv
    
    
#     def forward(self, images, labels, mask_list=None, idx2BN=None, ):
#         """
#         Overridden.
#         """
#         self._check_inputs(images)

#         images = images.clone().detach().to(self.device)
#         labels = labels.clone().detach().to(self.device)
#         adv_images = self._autoattack(images, labels, mask_list=mask_list, idx2BN=idx2BN)

#         return adv_images