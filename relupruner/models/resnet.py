"""CIFAR/Tiny-ImageNet ResNet used by the ReLUPruner experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import ActivationCursor, ModelOutput


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        prunable: bool = False,
    ) -> None:
        super().__init__()
        self.prunable = prunable
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        mismatch = stride != 1 or in_planes != planes
        if prunable:
            # Preserve the local customresnetV1 checkpoint names.
            self.mismatch = mismatch
            if mismatch:
                self.conv_sc = nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)
                self.bn_sc = nn.BatchNorm2d(planes)
        else:
            self.shortcut = nn.Sequential()
            if mismatch:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )

    def forward(self, x: Tensor, activation: ActivationCursor) -> Tensor:
        out = activation(self.bn1(self.conv1(x)))
        residual = self.bn_sc(self.conv_sc(x)) if self.prunable and self.mismatch else x
        if not self.prunable:
            residual = self.shortcut(x)
        out = self.bn2(self.conv2(out)) + residual
        return activation(out)


class ResNet(nn.Module):
    def __init__(
        self,
        blocks: tuple[int, int, int, int],
        num_classes: int = 10,
        prunable: bool = False,
        imagenet_stem: bool = False,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        self.prunable = prunable
        self.imagenet_stem = imagenet_stem
        self.in_planes = 64
        if imagenet_stem:
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
            self.maxpool = nn.Identity()
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, blocks[0], stride=1)
        self.layer2 = self._make_layer(128, blocks[1], stride=2)
        self.layer3 = self._make_layer(256, blocks[2], stride=2)
        self.layer4 = self._make_layer(512, blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def _make_layer(self, planes: int, count: int, stride: int) -> nn.ModuleList:
        layers = nn.ModuleList()
        for block_stride in [stride] + [1] * (count - 1):
            layers.append(BasicBlock(self.in_planes, planes, block_stride, self.prunable))
            self.in_planes = planes
        return layers

    @staticmethod
    def _run_layer(layer: nn.ModuleList, x: Tensor, activation: ActivationCursor) -> Tensor:
        for block in layer:
            x = block(x, activation)
        return x

    def forward(
        self,
        x: Tensor,
        masks: list[Tensor] | tuple[Tensor, ...] | None = None,
        collect_activations: bool = False,
    ) -> ModelOutput:
        if not self.prunable and masks is not None:
            raise ValueError("teacher ResNet does not accept ReLU masks")
        activation = ActivationCursor(masks if self.prunable else None, collect_activations)
        out = self.bn1(self.conv1(x))
        # This asymmetry is intentional and matches local customresnetV1.py.
        out = out if self.prunable else F.relu(out)
        out = self.maxpool(out)

        out = self._run_layer(self.layer1, out, activation)
        f1 = out
        out = self._run_layer(self.layer2, out, activation)
        f2 = out
        out = self._run_layer(self.layer3, out, activation)
        f3 = out
        out = self._run_layer(self.layer4, out, activation)
        f4 = out
        logits = self.linear(torch.flatten(self.avgpool(out), 1))
        activation.validate()
        return ModelOutput(logits, [f1, f2, f3, f4], activation.pre, activation.post)


def resnet18(num_classes: int, prunable: bool = False, imagenet_stem: bool = False) -> ResNet:
    return ResNet((2, 2, 2, 2), num_classes, prunable, imagenet_stem)


def resnet34(num_classes: int, prunable: bool = False, imagenet_stem: bool = False) -> ResNet:
    return ResNet((3, 4, 6, 3), num_classes, prunable, imagenet_stem)
