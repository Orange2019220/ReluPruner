'''
支持动态ReLU剪枝的ResNet
基于泰勒展开的重要性评估进行自适应剪枝
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchextractor as tx
from .utils.DualBN import DualBN2d
from einops import rearrange, reduce, repeat


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


def dynamic_masked_relu(x, mask, feature_collector=None):
    """
    动态掩码ReLU：支持基于重要性的动态剪枝

    Args:
        x: 输入特征
        mask: 当前层的剪枝掩码（可以是None，表示不剪枝）
        feature_collector: 用于收集特征的列表
    """
    if mask is None:
        # 没有掩码，正常的ReLU
        out = F.relu(x)
    else:
        # 应用掩码剪枝
        assert x.shape[1:] == mask.shape, f"Shape mismatch: x {x.shape}, mask {mask.shape}"

        # 扩展mask以匹配batch维度
        mask_expanded = mask.unsqueeze(0).expand_as(x)

        # 计算掩码的反向（未被剪枝的位置）
        mask_inv = 1.0 - mask_expanded

        # 对被掩码的位置应用ReLU，未被掩码的位置保持原值
        out = F.relu(x * mask_expanded) + x * mask_inv

    # 收集特征用于重要性评估
    if feature_collector is not None:
        feature_collector.append(out)

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use2BN=False):
        super(BasicBlock, self).__init__()
        self.use2BN = use2BN
        if self.use2BN:
            Norm2d = DualBN2d
        else:
            Norm2d = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = Norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = Norm2d(planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.mismatch = True
            self.conv_sc = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            self.bn_sc = Norm2d(self.expansion * planes)
        else:
            self.mismatch = False

        # 用于存储当前block的掩码
        self.relu1_mask = None
        self.relu2_mask = None

    def forward(self, input):
        x, masks, features, idx2BN = input

        # 第一个卷积层
        if self.use2BN:
            out = self.bn1(self.conv1(x), idx2BN)
        else:
            out = self.bn1(self.conv1(x))

        # 从masks列表中获取当前层的掩码（如果存在）
        current_mask = masks.pop(0) if masks else self.relu1_mask
        out = dynamic_masked_relu(out, current_mask, features)

        # 第二个卷积层
        if self.use2BN:
            out = self.bn2(self.conv2(out), idx2BN)
        else:
            out = self.bn2(self.conv2(out))

        # 残差连接
        if self.mismatch:
            if self.use2BN:
                out += self.bn_sc(self.conv_sc(x), idx2BN)
            else:
                out += self.bn_sc(self.conv_sc(x))
        else:
            out += x

        # 第二个ReLU
        current_mask = masks.pop(0) if masks else self.relu2_mask
        out = dynamic_masked_relu(out, current_mask, features)

        return (out, masks, features, idx2BN)

    def update_masks(self, relu1_mask, relu2_mask):
        """更新block的剪枝掩码"""
        self.relu1_mask = relu1_mask
        self.relu2_mask = relu2_mask


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        # 存储掩码
        self.relu1_mask = None
        self.relu2_mask = None
        self.relu3_mask = None

    def forward(self, input):
        x, masks, features, idx2BN = input

        # 1x1 conv
        out = self.bn1(self.conv1(x))
        current_mask = masks.pop(0) if masks else self.relu1_mask
        out = dynamic_masked_relu(out, current_mask, features)

        # 3x3 conv
        out = self.bn2(self.conv2(out))
        current_mask = masks.pop(0) if masks else self.relu2_mask
        out = dynamic_masked_relu(out, current_mask, features)

        # 1x1 conv
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)

        # Final ReLU
        preact = out
        current_mask = masks.pop(0) if masks else self.relu3_mask
        out = dynamic_masked_relu(out, current_mask, features)

        if self.is_last:
            return (out, masks, features, idx2BN), preact
        else:
            return (out, masks, features, idx2BN)

    def update_masks(self, relu1_mask, relu2_mask, relu3_mask):
        """更新bottleneck的剪枝掩码"""
        self.relu1_mask = relu1_mask
        self.relu2_mask = relu2_mask
        self.relu3_mask = relu3_mask


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, zero_init_residual=False, use2BN=False):
        super(ResNet, self).__init__()
        self.use2BN = use2BN
        self.num_blocks = num_blocks
        self.in_planes = 64

        if num_classes == 1000:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        if self.use2BN:
            Norm2d = DualBN2d
        else:
            Norm2d = nn.BatchNorm2d

        self.bn1 = Norm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use2BN=use2BN)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use2BN=use2BN)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use2BN=use2BN)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, use2BN=use2BN)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # 第一层的ReLU掩码
        self.relu1_mask = None

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
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
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3, bn4]

    def _make_layer(self, block, planes, num_blocks, stride, use2BN=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride, use2BN=use2BN))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, mask=None, features=None, is_feat=False, idx2BN=None):
        """
        前向传播

        Args:
            x: 输入数据
            mask: 剪枝掩码列表（可以是None或空列表，表示使用模型内部存储的掩码）
            features: 特征收集列表
            is_feat: 是否返回中间特征
            idx2BN: 双BN索引
        """
        if features is None:
            features = []

        # 如果没有提供mask，使用空列表（将使用模型内部的掩码）
        if mask is None:
            mask = []

        # 第一层卷积
        if self.use2BN:
            out = self.bn1(self.conv1(x), idx2BN)
        else:
            out = self.bn1(self.conv1(x))

        # 第一层ReLU
        current_mask = mask.pop(0) if mask else self.relu1_mask
        out = dynamic_masked_relu(out, current_mask, features)
        f0 = out

        if x.shape[-1] == 224:  # ImageNet
            out = self.maxpool(out)

        # 通过各层
        out, mask, features, idx2BN = self.layer1((out, mask, features, idx2BN))
        f1 = out

        out, mask, features, idx2BN = self.layer2((out, mask, features, idx2BN))
        f2 = out

        out, mask, features, idx2BN = self.layer3((out, mask, features, idx2BN))
        f3 = out

        out, mask, features, idx2BN = self.layer4((out, mask, features, idx2BN))
        f4 = out

        # 全局平均池化和分类器
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.linear(out)

        if is_feat:
            return [f0, f1, f2, f3, f4, f5], out, features
        else:
            return out, features

    def update_relu_masks(self, masks):
        """
        更新整个网络的ReLU剪枝掩码

        Args:
            masks: 包含所有ReLU层掩码的列表
        """
        mask_idx = 0

        # 更新第一层的掩码
        if mask_idx < len(masks):
            self.relu1_mask = masks[mask_idx]
            mask_idx += 1

        # 更新每个stage中的掩码
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if isinstance(block, BasicBlock):
                    if mask_idx + 1 < len(masks):
                        block.update_masks(masks[mask_idx], masks[mask_idx + 1])
                        mask_idx += 2
                elif isinstance(block, Bottleneck):
                    if mask_idx + 2 < len(masks):
                        block.update_masks(masks[mask_idx], masks[mask_idx + 1], masks[mask_idx + 2])
                        mask_idx += 3

    def get_relu_count(self):
        """获取网络中ReLU的总数"""
        count = 1  # conv1后的ReLU

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if isinstance(block, BasicBlock):
                    count += 2  # 每个BasicBlock有2个ReLU
                elif isinstance(block, Bottleneck):
                    count += 3  # 每个Bottleneck有3个ReLU

        return count


def CustomResNet18(**kwargs):
    """动态剪枝版本的ResNet18"""
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def CustomResNet34(**kwargs):
    """动态剪枝版本的ResNet34"""
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def CustomResNet50(**kwargs):
    """动态剪枝版本的ResNet50"""
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)