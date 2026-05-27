'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchextractor as tx
from .resnetv2 import ResNet18
from einops import rearrange, reduce, repeat
from .utils.ordered_dropops import SwitchableBatchNorm2d, OrderedConv2d, OrderedleLinear
from .utils.ordered_dropops import width_mult_list

import numpy as np
# class feature_afterrelu:
#     feature_list = []
#     mask = None



class HookTool: 
    def __init__(self):
        self.fea = None 

    def hook_fun(self, module, fea_in, fea_out):

        self.fea = fea_out

def get_feas_by_hook(model):
    fea_hooks = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)

    return fea_hooks


def maskedrelu(x, mask):


    assert x.shape == mask[0].shape
    mask_inv = torch.ones_like(mask[0]) - mask[0]
    out = x * mask[0]
    out = F.relu(out)
    out = out + x * mask_inv
    mask.pop(0)
    return out, mask

class OrderedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes_lst, out_planes_lst, stride=1):
        super(OrderedBasicBlock, self).__init__()
        self.conv1 = OrderedConv2d(in_planes_lst, out_planes_lst, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = SwitchableBatchNorm2d(out_planes_lst)
        self.conv2 = OrderedConv2d(out_planes_lst, out_planes_lst, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = SwitchableBatchNorm2d(out_planes_lst)

        self.shortcut = nn.Sequential()
        if stride != 1 or list(in_planes_lst) != list(out_planes_lst):
            self.shortcut = nn.Sequential(
                OrderedConv2d(in_planes_lst, out_planes_lst, kernel_size=1, stride=stride, bias=False),
                SwitchableBatchNorm2d(out_planes_lst),
            )

    def forward(self, input):
        x, mask, features = input
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(x))
        # mask = torch.ones_like(out)
        out, mask = maskedrelu(out, mask)
        features.append(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # mask = torch.ones_like(out)
        out, mask = maskedrelu(out, mask)
        features.append(out)
        return (out, mask, features)


class OrderedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes_lst, out_planes_lst, stride=1):
        super(OrderedBottleneck, self).__init__()
        self.conv1 = OrderedConv2d(in_planes_lst, out_planes_lst, kernel_size=1, bias=False)
        self.bn1 = SwitchableBatchNorm2d(out_planes_lst)
        self.conv2 = OrderedConv2d(out_planes_lst, out_planes_lst, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = SwitchableBatchNorm2d(out_planes_lst)
        self.conv3 = OrderedConv2d(out_planes_lst, self.expansion*out_planes_lst, kernel_size=1, bias=False)
        self.bn3 = SwitchableBatchNorm2d(self.expansion*out_planes_lst)

        self.shortcut = nn.Sequential()
        if stride != 1 or list(in_planes_lst) != list(self.expansion*out_planes_lst):
            self.shortcut = nn.Sequential(
                OrderedConv2d(in_planes_lst, self.expansion*out_planes_lst, kernel_size=1, stride=stride, bias=False),
                SwitchableBatchNorm2d(self.expansion*out_planes_lst),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out

class CustomOrderedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, zero_init_residual=False):
        super(CustomOrderedResNet, self).__init__()
        self.in_planes_list = np.array([int(64 * width_mult) for width_mult in width_mult_list])

        self.conv1 = OrderedConv2d(np.array([3 for _ in width_mult_list]), self.in_planes_list,
                                        kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = SwitchableBatchNorm2d(self.in_planes_list)
        self.layer1 = self._make_layer(block, np.array([int(64 * width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, np.array([int(128 * width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, np.array([int(256 * width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, np.array([int(512* width_mult) for width_mult in width_mult_list]), 
                                                    num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = OrderedleLinear(
            np.array([int(512* width_mult) for width_mult in width_mult_list]) * block.expansion, 
            np.array([num_classes for width_mult in width_mult_list])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, OrderedBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, OrderedBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.layer4)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], OrderedBottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], OrderedBasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3, bn4]

    def _make_layer(self, block, planes_list, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes_list, planes_list, stride))
            self.in_planes_list = planes_list * block.expansion
        return nn.Sequential(*layers)

    def forward(self,x, mask, features, is_feat):
        # out = F.relu(self.bn1(self.conv1(x)))

        out  = self.bn1(self.conv1(x))

        # if mask == None:
        #     print('===========================')
        #     print('Careful!!!! your mask now is None!')
        #     mask = torch.ones_like(out)
        out, mask = maskedrelu(out, mask)
        f0 = out
        features.append(out)
  
        out, mask, features = self.layer1((out, mask, features))
        f1 = out
        # print(layer1.)
        out, mask, features = self.layer2((out, mask, features))
        f2 = out
        out, mask, features = self.layer3((out, mask, features))
        f3 = out
        out, mask, features = self.layer4((out, mask, features))

        f4 = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.linear(out)
        if is_feat:
            return [f0, f1, f2, f3, f4, f5], out, features
            
        else:
            return out, features


def CustomODResNet18(**kwargs):
    return CustomOrderedResNet(OrderedBasicBlock, [2, 2, 2, 2], **kwargs)


def CustomODResNet34(**kwargs):
    return CustomOrderedResNet(OrderedBasicBlock, [3, 4, 6, 3], **kwargs)


def CustomODResNet50(**kwargs):
    return CustomOrderedResNet(OrderedBottleneck, [3, 4, 6, 3], **kwargs)


def CustomODResNet101(**kwargs):
    return CustomOrderedResNet(OrderedBottleneck, [3, 4, 23, 3], **kwargs)


def CustomODResNet152(**kwargs):
    return CustomOrderedResNet(OrderedBottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == '__main__':
    net_t = ResNet18(num_classes = 100)
    net = CustomODResNet18(num_classes=100)
    
    x = torch.randn(2, 3, 32, 32)

    data = torch.randn(2, 3, 32, 32)
    features = []

    net.eval()
    out_t, feature_t = net_t(data, features, is_feat = False)
    size_list = [feature.shape[2:] for feature in feature_t]
    channel_size = list([feature.shape[1] for feature in feature_t])
    # mask_list = [(torch.rand(size) > 0.75) + 0 for size in size_list]
    sensitivity = [0.01, 0.11, 0.14, 0.19, 0.21, 0.03, 0.03, 0.03, 0.03, 0.1, 0.14, 0.19, 0.34, 0.35, 0.76, 0.83, 0.85]
    mask_list = [(torch.rand(size) > (1 - sensitivity[size_idx])) + 0 for size_idx, size in enumerate(size_list)]
    # import pdb; pdb.set_trace()
    print('width_mult:1')
    net.apply(lambda m: setattr(m, 'width_mult', 1))
    for index, mask in enumerate(mask_list):
        mask_list[index] = repeat(mask_list[index], 'h w-> b c h w', c = int(channel_size[index]), b = 2)
    logit, features = net(x, mask_list, features, is_feat=False)
    print(len(features))

    # net.apply(lambda m: setattr(m, 'width_mult', 0.5))
    # for index, mask in enumerate(mask_list):
    #     mask_list[index] = repeat(mask_list[index], 'h w-> b c h w', c = int(channel_size[index] * 0.5), b = 2)
    # # mask_list = [mask[:int(mask.shape[0] * 0.5), :, :] for mask in mask_list]
    # # for mask_index, mask in enumerate(mask_list):
    # #     mask_list[mask_index] = repeat(mask_list[mask_index], 'c h w-> b c h w', b = list(input.shape)[0])
    # features = []

    # logit, features = net(x, mask_list, features, is_feat=False)
    # print(len(features))

    # feature_list = []
    # for name, module in net.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         print(name)
    #         cur_hook = HookTool()
    #         module.register_forward_hook(cur_hook.hook_fun)
    #         feature_list.append(cur_hook)
    
    # print(len(feature_list))
    # print(feature_list[1].fea)
    # for f in feats:
    #     print(f.shape, f.min().item())
    # print(len(features))

    # for m in net.get_bn_before_relu():
    #     if isinstance(m, nn.BatchNorm2d):
    #         print('pass')
    #     else:
    #         print('warning')
