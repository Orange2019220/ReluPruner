import math
from grpc import Channel
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.ordered_dropops import SwitchableBatchNorm2d, OrderedConv2d, OrderedleLinear
from .utils.ordered_dropops import width_mult_list
# from ordered_dropops import SwitchableBatchNorm2d, OrderedConv2d, OrderedleLinear
# from ordered_dropops import width_mult_list
import numpy as np



__all__ = ['ODwrn']


class OrderedBasicBlock(nn.Module):
    def __init__(self, in_planes_lst, out_planes_lst, stride, dropRate=0.0):
        super(OrderedBasicBlock, self).__init__()
        self.bn1 = SwitchableBatchNorm2d(in_planes_lst)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = OrderedConv2d(in_planes_lst, out_planes_lst, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = SwitchableBatchNorm2d(out_planes_lst)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = OrderedConv2d(out_planes_lst, out_planes_lst, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes_lst == out_planes_lst).sum()
        
        self.convShortcut = (not self.equalInOut) and OrderedConv2d(in_planes_lst, out_planes_lst, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, input):
        x, features = input

        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            features.append(x)
        else:
            out = self.relu1(self.bn1(x))
            features.append(out)
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        features.append(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        
        return (torch.add(x if self.equalInOut else self.convShortcut(x), out), features)


class OrderedNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes_lst, out_planes_lst, block, stride, dropRate=0.0):
        super(OrderedNetworkBlock, self).__init__()

        self.layer = self._make_layer(block, in_planes_lst, out_planes_lst, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes_lst, out_planes_lst, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            if i == 0:
                layers.append(block(in_planes_lst, out_planes_lst, stride, dropRate))
            else:
                layers.append(block(out_planes_lst, out_planes_lst, 1, dropRate))
            # layers.append(block(i == 0 and in_planes_lst or out_planes_lst, out_planes_lst, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, input):
        # x, features = input
        return self.layer(input)


class ODWideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(ODWideResNet, self).__init__()

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        nChannels_list = [np.array([int(channel * width_mult) for width_mult in width_mult_list]) for channel in nChannels]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = OrderedBasicBlock
        # 1st conv before any network block


        self.conv1 = OrderedConv2d(np.array([3 for _ in width_mult_list]), nChannels_list[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = OrderedNetworkBlock(n, nChannels_list[0], nChannels_list[1], block, 1, dropRate)
        # 2nd block
        self.block2 = OrderedNetworkBlock(n, nChannels_list[1], nChannels_list[2], block, 2, dropRate)
        # 3rd block
        self.block3 = OrderedNetworkBlock(n, nChannels_list[2], nChannels_list[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = SwitchableBatchNorm2d(nChannels_list[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = OrderedleLinear(nChannels_list[3], np.array([num_classes for width_mult in width_mult_list]))

        self.nChannels = nChannels_list[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        feat_m.append(self.block3)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block2.layer[0].bn1
        bn2 = self.block3.layer[0].bn1
        bn3 = self.bn1

        return [bn1, bn2, bn3]

    def forward(self, x, features, is_feat=False):
        out = self.conv1(x)
        f0 = out
        out, features = self.block1((out, features))
        f1 = out
        out, features = self.block2((out, features))
        f2 = out
        out, features = self.block3((out, features))
        f3 = out
        out = self.relu(self.bn1(out))
        features.append(out)
        out = F.avg_pool2d(out, 8)
        # out = out.view(-1, self.nChannels)
        out = out.view(out.size(0), -1)
        f4 = out
        out = self.fc(out)
        if is_feat:
            f1 = self.block2.layer[0].bn1(f1)
            f2 = self.block3.layer[0].bn1(f2)
            f3 = self.bn1(f3)
            return [f0, f1, f2, f3, f4], out, features
        else:
            return out, features


def ODwrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = ODWideResNet(**kwargs)
    return model


def ODwrn_40_2(**kwargs):
    model = ODWideResNet(depth=40, widen_factor=2, **kwargs)
    return model


def ODwrn_40_1(**kwargs):
    model = ODWideResNet(depth=40, widen_factor=1, **kwargs)
    return model


def ODwrn_16_2(**kwargs):
    model = ODWideResNet(depth=16, widen_factor=2, **kwargs)
    return model


def ODwrn_16_1(**kwargs):
    model = ODWideResNet(depth=16, widen_factor=1, **kwargs)
    return model

def ODwrn_22_8(**kwargs):
    model = ODWideResNet(depth=22, widen_factor=8, **kwargs)
    return model


if __name__ == '__main__':
    import torch

    net = ODwrn_22_8(num_classes=100)
    
    net.apply(lambda m: setattr(m, 'width_mult', 1))
    x = torch.randn(2, 3, 32, 32)
    features = []
    
    feat_eachlayer, logit, feature_afterrelu = net(x, features, is_feat=True)
    print([feature.shape for feature in feat_eachlayer])
    print([feature.shape for feature in feature_afterrelu])
    # for f in feats:
    #     print(f.shape, f.min().item())
    # print(logit.shape)

    # for m in net.get_bn_before_relu():
    #     if isinstance(m, SwitchableBatchNorm2d):
    #         print('pass')
    #     else:
    #         print('warning')