'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
from pyexpat import features
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils.ordered_dropops import SwitchableBatchNorm2d, OrderedConv2d, OrderedleLinear
from .utils.ordered_dropops import width_mult_list
# from ordered_dropops import SwitchableBatchNorm2d, OrderedConv2d, OrderedleLinear
# from ordered_dropops import width_mult_list
import numpy as np

from .vgg import vgg16_bn
from einops import rearrange, reduce, repeat
import torch

__all__ = [
    'CustomODVGG', 'CustomODvgg11', 'CustomODvgg11_bn', 'CustomODvgg13', 'CustomODvgg13_bn', 'CustomODvgg16', 'CustomODvgg16_bn',
    'CustomODvgg19_bn', 'CustomODvgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


def maskedrelu(x, mask):


    assert x.shape == mask[0].shape
    mask_inv = torch.ones_like(mask[0]) - mask[0]
    out = x * mask[0]
    out = F.relu(out)
    out = out + x * mask_inv
    mask.pop(0)
    return out, mask


class CustomODVGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(CustomODVGG, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = OrderedleLinear(np.array([int(512* width_mult) for width_mult in width_mult_list]), np.array([num_classes for width_mult in width_mult_list]))
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, mask, features, is_feat=False):

        h = x.shape[2]
        x = self.block0[:2](x)
        x, mask = maskedrelu(x, mask)
        features.append(x)
        x = self.block0[3:](x)
        x, mask = maskedrelu(x, mask)
        # x = F.relu(self.block0(x))
        features.append(x)
        f0 = x

        x = self.pool0(x)

        x = self.block1[:2](x)
        x, mask = maskedrelu(x, mask)
        features.append(x)
        x = self.block1[3:](x)
        x, mask = maskedrelu(x, mask)
        features.append(x)
        # x = self.block1(x)
        f1 = x

        x = self.pool1(x)

        x = self.block2[:2](x)
        x, mask = maskedrelu(x, mask)
        features.append(x)
        x =self.block2[3:5](x)
        x, mask = maskedrelu(x, mask)
        features.append(x)
        x = self.block2[6:](x)
        x, mask = maskedrelu(x, mask)
        features.append(x)
        # x = self.block2(x)
        # f2_pre = x
        # x = F.relu(x)
        # features.append(x)
        f2 = x

        x = self.pool2(x)

        x = self.block3[:2](x)
        x, mask = maskedrelu(x, mask)
        features.append(x)
        x =self.block3[3:5](x)
        x, mask = maskedrelu(x, mask)
        features.append(x)
        x = self.block3[6:](x)
        x, mask = maskedrelu(x, mask)
        features.append(x)
        # x = self.block3(x)
        # f3_pre = x
        # x = F.relu(x)
        # features.append(x)
        f3 = x

        if h == 64:
            x = self.pool3(x)

        x = self.block4[:2](x)
        x, mask = maskedrelu(x, mask)
        features.append(x)
        x =self.block4[3:5](x)
        x, mask = maskedrelu(x, mask)
        features.append(x)
        x = self.block4[6:](x)
        x, mask = maskedrelu(x, mask)
        features.append(x)
        # x = self.block4(x)
        # f4_pre = x
        # x = F.relu(x)
        # features.append(x)
        f4 = x

        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        f5 = x
        x = self.classifier(x)

        if is_feat:
            return [f0, f1, f2, f3, f4, f5], x, features
        else:
            return x, features

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):

        cfg_list = [np.array([int(value * width_mult) for width_mult in width_mult_list]) for value in cfg]
        if in_channels == 3:
            in_channels_list = np.array([3 for _ in width_mult_list])
        else:
            in_channels_list = np.array([int(in_channels * width_mult) for width_mult in width_mult_list])
        layers = []
        for v in cfg_list:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = OrderedConv2d(in_channels_list, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, SwitchableBatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels_list = v
        layers = layers[:-1]
        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]],
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
    'S': [[64], [128], [256], [512], [512]],
}


def CustomODvgg8(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CustomODVGG(cfg['S'], **kwargs)
    return model


def CustomODvgg8_bn(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CustomODVGG(cfg['S'], batch_norm=True, **kwargs)
    return model


def CustomODvgg11(**kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CustomODVGG(cfg['A'], **kwargs)
    return model


def CustomODvgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = CustomODVGG(cfg['A'], batch_norm=True, **kwargs)
    return model


def CustomODvgg13(**kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CustomODVGG(cfg['B'], **kwargs)
    return model


def CustomODvgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = CustomODVGG(cfg['B'], batch_norm=True, **kwargs)
    return model


def CustomODvgg16(**kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CustomODVGG(cfg['D'], **kwargs)
    return model


def CustomODvgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = CustomODVGG(cfg['D'], batch_norm=True, **kwargs)
    return model


def CustomODvgg19(**kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CustomODVGG(cfg['E'], **kwargs)
    return model


def CustomODvgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = CustomODVGG(cfg['E'], batch_norm=True, **kwargs)
    return model


if __name__ == '__main__':

    import torch
    net_t = vgg16_bn(num_classes = 100)
    net = CustomODvgg16_bn(num_classes=100)
    x = torch.randn(2, 3, 32, 32)

    data = torch.randn(2, 3, 32, 32)
    features = []

    net.eval()
    out_t, feature_t = net_t(data, features, is_feat = False)
    size_list = [feature.shape[2:] for feature in feature_t]
    channel_size = list([feature.shape[1] for feature in feature_t])
    mask_list = [(torch.rand(size) > 0.75) + 0 for size in size_list]
    # import pdb; pdb.set_trace()

    for index, mask in enumerate(mask_list):
        mask_list[index] = repeat(mask_list[index], 'h w-> b c h w', c = channel_size[index], b = 2)
    features = []
    logit, features = net(x, mask_list, features, is_feat=False)
    print(len(features))
