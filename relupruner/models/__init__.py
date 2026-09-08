from __future__ import annotations

from torch import nn

from .common import ModelOutput
from .resnet import resnet18, resnet34
from .wrn import wrn_22_8

MODEL_NAMES = ("resnet18", "resnet34", "wrn_22_8")
MODEL_DATASETS = {
    "resnet18": ("cifar10", "cifar100", "tiny_imagenet"),
    "resnet34": ("cifar10", "cifar100", "tiny_imagenet"),
    "wrn_22_8": ("cifar10", "cifar100"),
}


def validate_model_dataset(model: str, dataset: str) -> None:
    """Reject architecture/dataset pairs that are outside the supported surface."""
    if model not in MODEL_DATASETS:
        raise ValueError(f"unknown model {model!r}; choose one of {', '.join(MODEL_NAMES)}")
    if dataset not in MODEL_DATASETS[model]:
        supported = ", ".join(MODEL_DATASETS[model])
        raise ValueError(f"{model} does not support {dataset}; supported datasets: {supported}")


def create_model(
    name: str,
    num_classes: int,
    *,
    prunable: bool = False,
    imagenet_stem: bool = False,
) -> nn.Module:
    normalized = name.lower().replace("custom", "").replace("resnet", "resnet")
    aliases = {
        "resnet18": "resnet18",
        "resnet34": "resnet34",
        "_wrn_22_8": "wrn_22_8",
        "wrn_22_8": "wrn_22_8",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "resnet18":
        return resnet18(num_classes, prunable, imagenet_stem)
    if normalized == "resnet34":
        return resnet34(num_classes, prunable, imagenet_stem)
    if normalized == "wrn_22_8":
        if imagenet_stem:
            raise ValueError("wrn_22_8 supports CIFAR-size inputs only")
        return wrn_22_8(num_classes, prunable)
    raise ValueError(f"unknown model {name!r}; choose one of {', '.join(MODEL_NAMES)}")


__all__ = [
    "MODEL_DATASETS",
    "MODEL_NAMES",
    "ModelOutput",
    "create_model",
    "validate_model_dataset",
]
