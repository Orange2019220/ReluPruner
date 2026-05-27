import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.DualBN import DualBN2d, TripleBN2d

"""
Original Author: Wei Yang
"""

__all__ = ['wrn']


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, use3BN=False):
        super(BasicBlock, self).__init__()
        self.use3BN = use3BN
        if self.use3BN:
            Norm2d = TripleBN2d
        else:
            Norm2d = nn.BatchNorm2d
        self.bn1 = Norm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = Norm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, input):
        x, features, idx2BN = input
        if not self.equalInOut:
            if self.use3BN:
                x = self.relu1(self.bn1(x, idx2BN))
            else:
                x = self.relu1(self.bn1(x))
            features.append(x)
        else:
            if self.use3BN:
                out = self.relu1(self.bn1(x, idx2BN))
            else:
                out = self.relu1(self.bn1(x))
            features.append(out)
        if self.use3BN:
            out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x), idx2BN))
        else:
            out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))            
        features.append(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return (torch.add(x if self.equalInOut else self.convShortcut(x), out), features, idx2BN)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, use3BN=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, use3BN=use3BN)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, use3BN=False):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, use3BN=use3BN))
        return nn.Sequential(*layers)

    def forward(self, input):
        x, features, idx2BN = input
        return self.layer((x, features, idx2BN))


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, use3BN=False):
        super(WideResNet, self).__init__()
        self.use3BN = use3BN
        if self.use3BN:
            Norm2d = TripleBN2d
        else:
            Norm2d = nn.BatchNorm2d
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, use3BN=use3BN)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, use3BN=use3BN)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, use3BN=use3BN)
        # global average pooling and classifier
        self.bn1 = Norm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

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

    def forward(self, x, features, is_feat=False, preact=False, idx2BN=None):
        out = self.conv1(x)
        f0 = out
        out, features, idx2BN = self.block1((out, features, idx2BN))
        f1 = out
        out, features, idx2BN = self.block2((out, features, idx2BN))
        f2 = out
        out, features, idx2BN = self.block3((out, features, idx2BN))
        f3 = out
        if self.use3BN:
            out = self.relu(self.bn1(out, idx2BN))
        else:
            out = self.relu(self.bn1(out))
        features.append(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        f4 = out
        out = self.fc(out)
        if is_feat:
            if preact:
                if self.use3BN:
                    f1 = self.block2.layer[0].bn1(f1, idx2BN)
                    f2 = self.block3.layer[0].bn1(f2, idx2BN)
                    f3 = self.bn1(f3, idx2BN)
                else:
                    f1 = self.block2.layer[0].bn1(f1)
                    f2 = self.block3.layer[0].bn1(f2)
                    f3 = self.bn1(f3)
            return [f0, f1, f2, f3, f4], out, features
        else:
            return out, features


def wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model


def wrn_40_2(**kwargs):
    model = WideResNet(depth=40, widen_factor=2, **kwargs)
    return model


def wrn_40_1(**kwargs):
    model = WideResNet(depth=40, widen_factor=1, **kwargs)
    return model


def wrn_16_2(**kwargs):
    model = WideResNet(depth=16, widen_factor=2, **kwargs)
    return model

def wrn_22_8(**kwargs):
    model = WideResNet(depth=22, widen_factor=8, **kwargs)
    return model


def wrn_16_1(**kwargs):
    model = WideResNet(depth=16, widen_factor=1, **kwargs)
    return model


if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)
    net = wrn_40_2(num_classes=100)
    feats, logit = net(x, is_feat=True, preact=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')
