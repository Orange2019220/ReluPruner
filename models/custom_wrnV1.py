import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.DualBN import DualBN2d

from .wrn import wrn_22_8
from einops import rearrange, reduce, repeat

"""
Original Author: Wei Yang
"""

__all__ = ['custom_wrn']



def maskedrelu(x, mask, pre_relu_features=None, post_relu_features=None):

    # print("================",x.shape)
    # print("================",mask[0].shape)
    # print("================",len(mask))

    assert x.shape == mask[0].shape

    if pre_relu_features is not None:
        pre_relu_features.append(x)


    mask_inv = torch.ones_like(mask[0]) - mask[0]
    out = x * mask[0]
    out = F.relu(out)
    out = out + x * mask_inv

    if post_relu_features is not None:
        post_relu_features.append(out)

    mask.pop(0)
    return out, mask


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, use2BN=False):
        super(BasicBlock, self).__init__()
        self.use2BN = use2BN
        if self.use2BN:
            Norm2d = DualBN2d
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
        x, mask, features, idx2BN = input
        # 支持额外的特征收集参数
        pre_relu_features = None
        post_relu_features = None
        if isinstance(features, tuple):
            features, pre_relu_features, post_relu_features = features

        if not self.equalInOut:
            # x = self.relu1(self.bn1(x))
            x, mask = maskedrelu(self.bn1(x), mask, pre_relu_features, post_relu_features)
            features.append(x)
        else:
            # out = self.relu1(self.bn1(x))
            out, mask = maskedrelu(self.bn1(x), mask, pre_relu_features, post_relu_features)
            features.append(out)
        # out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.use2BN:
            out, mask = maskedrelu(self.bn2(self.conv1(out if self.equalInOut else x), idx2BN), mask, pre_relu_features, post_relu_features)
        else:
            out, mask = maskedrelu(self.bn2(self.conv1(out if self.equalInOut else x)), mask, pre_relu_features, post_relu_features)

        features.append(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        # 恢复features格式
        if pre_relu_features is not None:
            features = (features, pre_relu_features, post_relu_features)

        return (torch.add(x if self.equalInOut else self.convShortcut(x), out), mask, features, idx2BN)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, use2BN=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, use2BN=use2BN)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, use2BN=False):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, use2BN=use2BN))
        return nn.Sequential(*layers)

    def forward(self, input):
        # x, features = input
        return self.layer(input)


class CustomWideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, use2BN=False):
        super(CustomWideResNet, self).__init__()
        self.use2BN = use2BN
        if self.use2BN:
            Norm2d = DualBN2d
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
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, use2BN=use2BN)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, use2BN=use2BN)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, use2BN=use2BN)
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

    def forward(self, x, mask, features, is_feat, idx2BN=None, collect_pre_post=False):
        """
        修改的forward函数，支持收集pre-relu和post-relu特征

        Args:
            collect_pre_post: 是否收集pre-relu和post-relu特征
        """
        # 初始化特征收集列表
        pre_relu_features = [] if collect_pre_post else None
        post_relu_features = [] if collect_pre_post else None

        # 如果需要收集特征，修改features格式
        if collect_pre_post:
            features_with_pre_post = (features, pre_relu_features, post_relu_features)
        else:
            features_with_pre_post = features

        out = self.conv1(x)
        f0 = out

        out, mask, features_updated, idx2BN = self.block1((out, mask, features_with_pre_post, idx2BN))
        if collect_pre_post:
            features = features_updated[0]
            pre_relu_features = features_updated[1]
            post_relu_features = features_updated[2]
            features_with_pre_post = (features, pre_relu_features, post_relu_features)
        f1 = out

        out, mask, features_updated, idx2BN = self.block2((out, mask, features_with_pre_post, idx2BN))
        if collect_pre_post:
            features = features_updated[0]
            pre_relu_features = features_updated[1]
            post_relu_features = features_updated[2]
            features_with_pre_post = (features, pre_relu_features, post_relu_features)
        f2 = out

        out, mask, features_updated, idx2BN = self.block3((out, mask, features_with_pre_post, idx2BN))
        if collect_pre_post:
            features = features_updated[0]
            pre_relu_features = features_updated[1]
            post_relu_features = features_updated[2]
            features_with_pre_post = (features, pre_relu_features, post_relu_features)
        f3 = out

        # out = self.relu(self.bn1(out))
        if self.use2BN:
            out, mask = maskedrelu(self.bn1(out, idx2BN), mask, pre_relu_features, post_relu_features)
        else:
            out, mask = maskedrelu(self.bn1(out), mask, pre_relu_features, post_relu_features)
        features.append(out)

        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        f4 = out
        out = self.fc(out)

        if is_feat:
            if self.use2BN:
                f1 = self.block2.layer[0].bn1(f1, idx2BN)
                f2 = self.block3.layer[0].bn1(f2, idx2BN)
                f3 = self.bn1(f3, idx2BN)
            else:
                f1 = self.block2.layer[0].bn1(f1)
                f2 = self.block3.layer[0].bn1(f2)
                f3 = self.bn1(f3)

            # if collect_pre_post:
            #     return [f0, f1, f2, f3, f4], out, features, pre_relu_features, post_relu_features
            # else:
            #     return [f0, f1, f2, f3, f4], out, features
            if collect_pre_post:
                return [f1, f2, f3, f4], out, features, pre_relu_features, post_relu_features
            else:
                return [f1, f2, f3, f4], out, features
        else:
            if collect_pre_post:
                return out, features, pre_relu_features, post_relu_features
            else:
                return out, features


def Customwrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = CustomWideResNet(**kwargs)
    return model


def Custom_wrn_40_2(**kwargs):
    model = CustomWideResNet(depth=40, widen_factor=2, **kwargs)
    return model


def Custom_wrn_40_1(**kwargs):
    model = CustomWideResNet(depth=40, widen_factor=1, **kwargs)
    return model


def Custom_wrn_16_2(**kwargs):
    model = CustomWideResNet(depth=16, widen_factor=2, **kwargs)
    return model


def Custom_wrn_16_1(**kwargs):
    model = CustomWideResNet(depth=16, widen_factor=1, **kwargs)
    return model

def Custom_wrn_22_8(**kwargs):
    model = CustomWideResNet(depth=22, widen_factor=8, **kwargs)
    return model


if __name__ == '__main__':
    import torch
    net_t = wrn_22_8(num_classes=100)
    data = torch.randn(2, 3, 32, 32)
    features = []
    net_t.eval()
    out_t, feature_t = net_t(data, features, is_feat = False)
    size_list = [feature.shape[2:] for feature in feature_t]
    channel_size = list([feature.shape[1] for feature in feature_t])
    mask_list = [(torch.rand(size) > 0.75) + 0 for size in size_list]
    for index, mask in enumerate(mask_list):
        mask_list[index] = repeat(mask_list[index], 'h w-> b c h w', c = channel_size[index], b = 2)

    x = torch.randn(2, 3, 32, 32)
    net = Custom_wrn_22_8(num_classes=100)
    features = []
    feats, logit, features = net(x, mask_list, features, is_feat=True)
    print(len(features))
    print([feature.shape for feature in features])

    # for f in feats:
    #     print(f.shape, f.min().item())
    # print(logit.shape)

    # for m in net.get_bn_before_relu():
    #     if isinstance(m, nn.BatchNorm2d):
    #         print('pass')
    #     else:
    #         print('warning')
