"""
MobileNetV2 implementation used in
<Knowledge Distillation via Route Constrained Optimization>
"""


import torch
import torch.nn as nn
import math
from mobilenetv2 import mobile_half
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

__all__ = ['Custom_mobilenetv2_T_w', 'Custom_mobile_half']

BN = None


def maskedrelu(x, mask):


    assert x.shape == mask[0].shape
    mask_inv = torch.ones_like(mask[0]) - mask[0]
    out = x * mask[0]
    out = F.relu(out)
    out = out + x * mask_inv
    mask.pop(0)
    return out, mask


# class MaskedRelu(nn.Module):
#     def __init__(self, inplace=True):
#         super(MaskedRelu, self).__init__()
#         self.relu = nn.ReLU(inplace=inplace)
    
#     def forward(self, input):
#         x, mask = input
#         assert x.shape == mask[0].shape
#         mask_inv = torch.ones_like(mask[0]) - mask[0]
#         out = x * mask[0]
#         out = self.relu(out)
#         out = out + x * mask_inv
#         mask.pop(0)
#         return (out, mask)



def conv_bn(inp, oup, stride):

    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        # self.conv = nn.Sequential(
        #     # pw
        #     nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(inp * expand_ratio),
        #     nn.ReLU(inplace=True),
        #     # dw
        #     nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
        #     nn.BatchNorm2d(inp * expand_ratio),
        #     nn.ReLU(inplace=True),
        #     # pw-linear
        #     nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(oup),
        # )
        self.conv_pw = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio)
        )
        self.conv_dw = nn.Sequential(
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio)
        )
        self.conv_pw_linear = nn.Sequential(
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

    def forward(self, input):
        x, mask, features = input
        t = x

        out = self.conv_pw(x)
        out, mask = maskedrelu(out, mask)
        features.append(out)
        out = self.conv_dw(out)
        out, mask = maskedrelu(out, mask)
        features.append(out)
        out = self.conv_pw_linear(out)
        if self.use_res_connect:
            return (t + out, mask, features)
        else:
            return (out, mask, features)


class CustomMobileNetV2(nn.Module):
    """mobilenetV2"""
    def __init__(self, T,
                 feature_dim,
                 input_size=32,
                 width_mult=1.,
                 remove_avg=False):
        super(CustomMobileNetV2, self).__init__()
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
        input_channel = int(32 * width_mult)
        self.conv1 = conv_bn(3, input_channel, 2)

        # building inverted residual blocks
        self.blocks = nn.ModuleList([])
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
            self.blocks.append(nn.Sequential(*layers))

        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv2 = conv_1x1_bn(input_channel, self.last_channel)

        # building classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(self.last_channel, feature_dim),
        )

        H = input_size // (32//2)
        self.avgpool = nn.AvgPool2d(H, ceil_mode=True)

        self._initialize_weights()
        print(T, width_mult)

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

    def forward(self, x, mask, features, is_feat=False):

        out = self.conv1(x)
        out, mask = maskedrelu(out, mask)
        features.append(out)
        f0 = out

        out, mask, features = self.blocks[0]((out, mask, features))
        out, mask, features = self.blocks[1]((out, mask, features))
        f1 = out
        out, mask, features = self.blocks[2]((out, mask, features))
        f2 = out
        out, mask, features = self.blocks[3]((out, mask, features))
        out, mask, features = self.blocks[4]((out, mask, features))
        f3 = out
        out, mask, features = self.blocks[5]((out, mask, features))
        out, mask, features = self.blocks[6]((out, mask, features))
        f4 = out

        out = self.conv2(out)
        out, mask = maskedrelu(out, mask)
        features.append(out)
        if not self.remove_avg:
            out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.classifier(out)

        if is_feat:
            return [f0, f1, f2, f3, f4, f5], out, features
        else:
            return out, features

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


def Custom_mobilenetv2_T_w(T, W, feature_dim=100):
    model = CustomMobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
    return model


def Custom_mobile_half(num_classes):
    return Custom_mobilenetv2_T_w(6, 0.5, num_classes)


if __name__ == '__main__':
    net_t = mobile_half(num_classes = 100)
    net = Custom_mobile_half(num_classes=100)
    x = torch.randn(2, 3, 32, 32)

    data = torch.randn(2, 3, 32, 32)
    features = []

    net_t.eval()
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

