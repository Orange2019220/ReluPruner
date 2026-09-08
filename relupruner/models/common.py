from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F

from relupruner.pruning import masked_relu


@dataclass
class ModelOutput:
    logits: Tensor
    features: list[Tensor]
    pre_activations: list[Tensor]
    post_activations: list[Tensor]


class ActivationCursor:
    """Apply masks without mutating the caller's list, and optionally collect Taylor tensors."""

    def __init__(self, masks: list[Tensor] | tuple[Tensor, ...] | None, collect: bool) -> None:
        self.masks = masks
        self.collect = collect
        self.index = 0
        self.pre: list[Tensor] = []
        self.post: list[Tensor] = []

    def __call__(self, activation: Tensor) -> Tensor:
        if self.masks is None:
            output = F.relu(activation)
        else:
            if self.index >= len(self.masks):
                raise ValueError(f"missing mask for prunable activation {self.index}")
            output = masked_relu(activation, self.masks[self.index])
        if self.collect:
            self.pre.append(activation)
            if output.requires_grad:
                output.retain_grad()
            self.post.append(output)
        self.index += 1
        return output

    def validate(self) -> None:
        if self.masks is not None and self.index != len(self.masks):
            raise ValueError(f"model used {self.index} masks, but {len(self.masks)} were provided")
