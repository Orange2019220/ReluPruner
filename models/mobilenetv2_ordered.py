"""
MobileNetV2 implementation used in
<Knowledge Distillation via Route Constrained Optimization>
"""

import torch
import torch.nn as nn
import math
# from .utils.ordered_dropops import SwitchableBatchNorm2d, OrderedConv2d, OrderedleLinear
# from .utils.ordered_dropops import width_mult_list
from ordered_dropops import SwitchableBatchNorm2d, OrderedConv2d, OrderedleLinear
from ordered_dropops import width_mult_list
import numpy as np

__all__ = ['ODmobilenetv2_T_w', 'ODmobile_half']

BN = None


def conv_bn(in_planes_lst, out_planes_lst, stride):
    return nn.Sequential(
        OrderedConv2d(in_planes_lst, out_planes_lst, 3, stride, 1, bias=False),
        SwitchableBatchNorm2d(out_planes_lst),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(in_planes_lst, out_planes_lst):
    return nn.Sequential(
        OrderedConv2d(in_planes_lst, out_planes_lst, 1, 1, 0, bias=False),
        SwitchableBatchNorm2d(out_planes_lst),
        nn.ReLU(inplace=True)
    )


class OrderedInvertedResidual(nn.Module):
    def __init__(self, in_planes_lst, out_planes_lst, stride, expand_ratio):
        super(OrderedInvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and in_planes_lst == out_planes_lst

        # self.conv = nn.Sequential(
        #     # pw
        #     OrderedConv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
        #     SwitchableBatchNorm2d(inp * expand_ratio),
        #     nn.ReLU(inplace=True),
        #     # dw
        #     OrderedConv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
        #     SwitchableBatchNorm2d(inp * expand_ratio),
        #     nn.ReLU(inplace=True),
        #     # pw-linear
        #     OrderedConv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
        #     SwitchableBatchNorm2d(oup),
        # )
        self.conv_pw = nn.Sequential(
            OrderedConv2d(in_planes_lst, in_planes_lst * expand_ratio, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(in_planes_lst * expand_ratio),
            nn.ReLU(inplace=True)
        )
        self.conv_dw = nn.Sequential(
            OrderedConv2d(in_planes_lst * expand_ratio, in_planes_lst * expand_ratio, 3, stride, 1, groups_list=in_planes_lst * expand_ratio, bias=False),
            SwitchableBatchNorm2d(in_planes_lst * expand_ratio),
            nn.ReLU(inplace=True)
        )
        self.conv_pw_linear = nn.Sequential(
            OrderedConv2d(in_planes_lst * expand_ratio, out_planes_lst, 1, 1, 0, bias=False),
            SwitchableBatchNorm2d(out_planes_lst)
        )

        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

    def forward(self, input):

        x, features = input
        t = x
        out = self.conv_pw(x)
        features.append(out)
        out = self.conv_dw(out)
        features.append(out)
        out = self.conv_pw_linear(out)
        if self.use_res_connect:
            return (t + out, features)
        else:
            return (out, features)


class ODMobileNetV2(nn.Module):
    """mobilenetV2"""
    def __init__(self, T,
                 feature_dim,
                 input_size=32,
                 self_width_mult=1.,
                 remove_avg=False):
        super(ODMobileNetV2, self).__init__()
        self.remove_avg = remove_avg

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [T, 24, 2, 1],
            [T, 32, 3, 2],
            [T, 64, 4, 2],
            [T, 96, 3, 1],
            [T, 160, 3, 2],
            [T, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = int(32 * self_width_mult)
        input_channel_list = np.array([int(32 * self_width_mult * width_mult) for width_mult in width_mult_list])

        self.conv1 = conv_bn(np.array([3 for _ in width_mult_list]), input_channel_list, 2)

        # building inverted residual blocks
        self.blocks = nn.ModuleList([])
        for t, c, n, s in self.interverted_residual_setting:
            # output_channel = int(c * self_width_mult)
            output_channel_list = np.array([int(c * self_width_mult * width_mult) for width_mult in width_mult_list])
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    OrderedInvertedResidual(input_channel_list, output_channel_list, stride, t)
                )
                input_channel_list = output_channel_list
            self.blocks.append(nn.Sequential(*layers))

        # self.last_channel = int(1280 * self_width_mult) if self_width_mult > 1.0 else 1280
        self.last_channel_list = np.array([int((int(1280 * self_width_mult) if self_width_mult > 1.0 else 1280) * width_mult) for width_mult in width_mult_list])
        self.conv2 = conv_1x1_bn(input_channel_list, self.last_channel_list)

        # building classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            OrderedleLinear(self.last_channel_list, np.array([feature_dim for width_mult in width_mult_list]))
            
        )

        H = input_size // (32//2)
        self.avgpool = nn.AvgPool2d(H, ceil_mode=True)

        self._initialize_weights()
        print(T, self_width_mult)

    def get_bn_before_relu(self):
        bn1 = self.blocks[1][-1].conv[-1]
        bn2 = self.blocks[2][-1].conv[-1]
        bn3 = self.blocks[4][-1].conv[-1]
        bn4 = self.blocks[6][-1].conv[-1]
        return [bn1, bn2, bn3, bn4]

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.blocks)
        return feat_m

    def forward(self, x, features, is_feat=False):

        out = self.conv1(x)
        features.append(out)
        f0 = out

        out, features = self.blocks[0]((out,features))
        out, features = self.blocks[1]((out,features))
        f1 = out
        out, features = self.blocks[2]((out,features))
        f2 = out
        out, features = self.blocks[3]((out,features))
        out, features = self.blocks[4]((out,features))
        f3 = out
        out, features = self.blocks[5]((out,features))
        out, features = self.blocks[6]((out,features))
        f4 = out

        out = self.conv2(out)
        features.append(out)

        if not self.remove_avg:
            out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.classifier(out)

        if is_feat:
            return [f0, f1, f2, f3, f4, f5], out, features
        else:
            return out ,features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, OrderedConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, SwitchableBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, OrderedleLinear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def ODmobilenetv2_T_w(T, W, feature_dim=100):
    model = ODMobileNetV2(T=T, feature_dim=feature_dim, self_width_mult=W)
    return model


def ODmobile_half(num_classes):
    return ODmobilenetv2_T_w(6, 0.5, num_classes)


if __name__ == '__main__':
    # x = torch.randn(2, 3, 32, 32)

    # net = mobile_half(100)

    # feats, logit = net(x, is_feat=True, preact=True)
    # for f in feats:
    #     print(f.shape, f.min().item())
    # print(logit.shape)

    # for m in net.get_bn_before_relu():
    #     if isinstance(m, SwitchableBatchNorm2d):
    #         print('pass')
    #     else:
    #         print('warning')
    net = ODmobile_half(num_classes=100)
    net.apply(lambda m: setattr(m, 'width_mult', 0.5))
    x = torch.randn(2, 3, 32, 32)
    features = []
    feat_eachlayer, logit, feature_afterrelu = net(x, features, is_feat=True)
    print([feature.shape for feature in feat_eachlayer])