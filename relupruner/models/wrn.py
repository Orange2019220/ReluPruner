"""Wide ResNet with maskable ReLU positions."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import ActivationCursor, ModelOutput


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.droprate = drop_rate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            None
            if self.equalInOut
            else nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)
        )

    def forward(self, x: Tensor, activation: ActivationCursor) -> Tensor:
        if self.equalInOut:
            residual = x
            out = activation(self.bn1(x))
        else:
            x = activation(self.bn1(x))
            residual = self.convShortcut(x)
            out = x
        out = activation(self.bn2(self.conv1(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return residual + self.conv2(out)


class NetworkBlock(nn.Module):
    def __init__(
        self, count: int, in_planes: int, out_planes: int, stride: int, drop_rate: float
    ) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            BasicBlock(
                in_planes if i == 0 else out_planes,
                out_planes,
                stride if i == 0 else 1,
                drop_rate,
            )
            for i in range(count)
        )

    def forward(self, x: Tensor, activation: ActivationCursor) -> Tensor:
        for block in self.layer:
            x = block(x, activation)
        return x


class WideResNet(nn.Module):
    def __init__(
        self,
        depth: int,
        num_classes: int,
        widen_factor: int = 1,
        drop_rate: float = 0.0,
        prunable: bool = False,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6:
            raise ValueError("WideResNet depth must be 6n+4")
        count = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.prunable = prunable
        self.conv1 = nn.Conv2d(3, channels[0], 3, padding=1, bias=False)
        self.block1 = NetworkBlock(count, channels[0], channels[1], 1, drop_rate)
        self.block2 = NetworkBlock(count, channels[1], channels[2], 2, drop_rate)
        self.block3 = NetworkBlock(count, channels[2], channels[3], 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self.nChannels = channels[3]

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / fan))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.bias.data.zero_()

    def forward(
        self,
        x: Tensor,
        masks: list[Tensor] | tuple[Tensor, ...] | None = None,
        collect_activations: bool = False,
    ) -> ModelOutput:
        if not self.prunable and masks is not None:
            raise ValueError("teacher WideResNet does not accept ReLU masks")
        activation = ActivationCursor(masks if self.prunable else None, collect_activations)
        out = self.conv1(x)
        out = self.block1(out, activation)
        f1 = out
        out = self.block2(out, activation)
        f2 = out
        out = self.block3(out, activation)
        f3 = out
        out = activation(self.bn1(out)) if self.prunable else F.relu(self.bn1(out))
        logits = self.fc(torch.flatten(F.adaptive_avg_pool2d(out, 1), 1))
        activation.validate()
        return ModelOutput(logits, [f1, f2, f3], activation.pre, activation.post)


def wrn_22_8(num_classes: int, prunable: bool = False) -> WideResNet:
    return WideResNet(22, num_classes, widen_factor=8, prunable=prunable)
