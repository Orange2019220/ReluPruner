''' PyTorch implementation of ResNet taken from 
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
and used by 
Author: Anonymous.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from .utils.ordered_dropops import SwitchableBatchNorm2d, OrderedConv2d, OrderedleLinear
from .utils.ordered_dropops import width_mult_list

class OrderedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes_lst, out_planes_lst, stride=1, is_last=False):
        super(OrderedBasicBlock, self).__init__()
        self.is_last = is_last
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

        x, features = input
        out = F.relu(self.bn1(self.conv1(x)))
        features.append(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        features.append(out)
        return (out, features)


class OrderedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes_lst, out_planes_lst, stride=1, is_last=False):
        super(OrderedBottleneck, self).__init__()
        self.is_last = is_last
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


    def forward(self, input):
        x, features = input
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        return (out, features)


class OrderedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, zero_init_residual=False):
        super(OrderedResNet, self).__init__()
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


    def forward(self, x, features, is_feat=False):
        out = F.relu(self.bn1(self.conv1(x)))
        features.append(out)
        f0 = out
        out, features = self.layer1((out, features))
        f1= out
        out, features = self.layer2((out, features))
        f2 = out
        out, features = self.layer3((out, features))
        f3 = out
        out, features = self.layer4((out, features))
        f4 = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.linear(out)
        if is_feat:
            return [f0, f1, f2, f3, f4, f5], out, features
        else:
            return out, features

def ODResNet18(**kwargs):
    return OrderedResNet(OrderedBasicBlock, [2, 2, 2, 2], **kwargs)

def ODResNet34(**kwargs):
    return OrderedResNet(OrderedBasicBlock, [3, 4, 6, 3], **kwargs)

def ODResNet50(**kwargs):
    return OrderedResNet(OrderedBottleneck, [3, 4, 6, 3], **kwargs)

def ODResNet101(**kwargs):
    return OrderedResNet(OrderedBottleneck, [3, 4, 23, 3], **kwargs)

def ODResNet152(**kwargs):
    return OrderedResNet(OrderedBottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == '__main__':

    net = ODResNet18(num_classes=100)
    net.apply(lambda m: setattr(m, 'width_mult', 0.5))
    x = torch.randn(2, 3, 32, 32)
    features = []
    feat_eachlayer, logit, feature_afterrelu = net(x, features, is_feat=True)
    print([feature.shape for feature in feat_eachlayer])
    # print([feature.shape for feature in feature_afterrelu])

    # for f in feats:
    #     print(f.shape, f.min().item())
    # print(logit.shape)

    # for m in net.get_bn_before_relu():
    #     if isinstance(m, nn.BatchNorm2d):
    #         print('pass')
    #     else:
    #         print('warning')

