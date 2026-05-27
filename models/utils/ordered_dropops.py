'''
Implementation of Custom conv and linear
to support ordered dropout technique
Author: Souvik Kundu, Ph.D., USC. 
'''
import torch.nn as nn
import torch
import numpy as np
import math

#width_mult_list = [0.25, 0.50, 0.75, 1.0]
width_mult_list = [0.5, 1.0]
#width_mult_list = [0.2, 0.4, 0.6, 0.8, 1.0]
#width_mult_list = [0.125, 0.5, 1.0]

class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        assert len(num_features_list) == len(width_mult_list)
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(width_mult_list)
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)
        return y


class OrderedConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(OrderedConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        assert len(in_channels_list) == len(width_mult_list)
        assert len(out_channels_list) == len(width_mult_list)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(width_mult_list)

    def forward(self, input):
        idx = width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y

class OrderedleLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(OrderedleLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        assert len(in_features_list) == len(width_mult_list)
        assert len(out_features_list) == len(width_mult_list)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(width_mult_list)

    def forward(self, input):
        idx = width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)

