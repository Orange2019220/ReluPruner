import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from .layer_defs import *


def get_convlayer(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False, conv_type="Plain"):
    if(conv_type=="Plain"):
        l = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
    elif(conv_type=="sWS"):
        l = ScaledStdConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
    elif(conv_type=="WeightNormalized"):
        l = WN_self(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
    elif(conv_type=="WeightCentered"):
        l = WCConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)

    torch.nn.init.kaiming_normal_(l.weight, mode='fan_in', nonlinearity='relu')
    try:
        torch.nn.init.zeros_(l.bias)
    except:
        pass
    return l

def get_norm_and_activ_layer(norm_type, num_channels, n_groups):
    if(norm_type=="Plain"):
        l = [nn.Identity(), nn.ReLU(inplace=True)]
    elif(norm_type=="BatchNorm"):
        l = [BN_self(num_channels), nn.ReLU(inplace=True)]
    elif(norm_type=="LayerNorm"):
        l = [LN_self(num_channels), nn.ReLU(inplace=True)]
    elif(norm_type=="Instance Normalization"):
        l = [IN_self(num_channels), nn.ReLU(inplace=True)]
    elif(norm_type=="GroupNorm"):
        l = [GN_self(num_channels, groups=n_groups), nn.ReLU(inplace=True)]
    elif(norm_type=="Filter Response Normalization"):
        l = [FRN_self(num_channels), TLU(num_channels)]
    elif(norm_type=="Weight Normalization"):
        l = [nn.Identity(), WN_scaledReLU(inplace=True)]
    elif(norm_type=="Scaled Weight Standardization"):
        l = [nn.Identity(), nn.ReLU(inplace=True)]
    elif(norm_type=="EvoNormBO"):
        l = [nn.Identity(), EvoNormBO(num_features=num_channels)]
    elif(norm_type=="EvoNormSO"):
        l = [nn.Identity(), EvoNormSO(num_features=num_channels)]
    elif(norm_type=="Variance Normalization"):
        l = [VN_self(num_channels), nn.ReLU(inplace=True)]
    elif(norm_type=="Mean Centering"):
        l = [MC_self(num_channels), nn.ReLU(inplace=True)]
    return l
