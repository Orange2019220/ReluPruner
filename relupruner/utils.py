from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0

    def update(self, value: float, count: int) -> None:
        self.total += value * count
        self.count += count


def accuracy(logits: Tensor, targets: Tensor) -> float:
    return logits.argmax(dim=1).eq(targets).float().mean().mul(100).item()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device
