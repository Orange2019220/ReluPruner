"""Mask application and progressive global-budget scheduling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, pi

import torch
from torch import Tensor
from torch.nn import functional as F


def expand_mask(mask: Tensor, activation: Tensor) -> Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    try:
        return mask.to(device=activation.device, dtype=activation.dtype).expand_as(activation)
    except RuntimeError as error:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match activation {tuple(activation.shape)}"
        ) from error


def masked_relu(activation: Tensor, mask: Tensor) -> Tensor:
    """Apply ReLU at selected positions and identity everywhere else."""
    expanded = expand_mask(mask, activation)
    return F.relu(activation * expanded) + activation * (1.0 - expanded)


def dense_masks(pre_activations: Sequence[Tensor]) -> list[Tensor]:
    return [torch.ones(tensor.shape[1:], device=tensor.device) for tensor in pre_activations]


def count_kept_relus(masks: Sequence[Tensor]) -> tuple[int, int]:
    kept = sum(int(mask.sum().item()) for mask in masks)
    total = sum(mask.numel() for mask in masks)
    return kept, total


@dataclass
class ProgressiveMaskScheduler:
    """Match the progressive schedule used by the local training script."""

    target_keep_ratio: float = 0.1
    update_interval: int = 100
    warmup_epochs: int = 5
    target_epoch_ratio: float = 0.6
    total_epochs: int = 150
    epoch: int = 0
    batch_counter: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.target_keep_ratio <= 1:
            raise ValueError("target_keep_ratio must be in (0, 1]")
        self.target_epoch = int(
            self.warmup_epochs + (self.total_epochs - self.warmup_epochs) * self.target_epoch_ratio
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def should_update(self) -> bool:
        self.batch_counter += 1
        if self.epoch < self.warmup_epochs:
            return False
        interval = self.update_interval * (2 if self.epoch >= self.target_epoch else 1)
        return self.batch_counter % interval == 0

    def keep_ratio(self) -> float:
        if self.epoch < self.warmup_epochs:
            return 1.0
        if self.epoch >= self.target_epoch:
            return self.target_keep_ratio
        effective_epochs = max(1, self.target_epoch - self.warmup_epochs)
        progress = (self.epoch - self.warmup_epochs) / effective_epochs
        accelerated = min(1.0, max(0.0, progress)) ** 1.5
        cosine_decay = 0.5 * (1.0 + cos(pi * accelerated))
        return self.target_keep_ratio + (1.0 - self.target_keep_ratio) * cosine_decay
