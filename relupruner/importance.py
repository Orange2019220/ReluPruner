"""Taylor importance estimators used by ReLUPruner.

The numerical choices in this module intentionally follow the local experiment
code, which is the reference implementation for this cleaned release.
"""

from __future__ import annotations

from collections import deque
from math import sqrt

import torch
from torch import Tensor
from torch.nn import functional as F


class LayerImportance:
    """Track a scalar importance weight for each prunable activation layer."""

    def __init__(self, momentum: float = 0.95, depth_bias: float = 0.2) -> None:
        self.momentum = momentum
        self.depth_bias = depth_bias
        self.gradients: dict[int, float] = {}
        self.fisher: dict[int, float] = {}
        self.importance: dict[int, float] = {}
        self.total_layers = 0

    @torch.no_grad()
    def update(self, layer_index: int, features: Tensor, gradients: Tensor) -> None:
        """Update the local-code EMA statistics for one layer.

        ``features`` is retained in the public signature for compatibility with
        the original implementation. The computed feature variance was unused
        there, so it is deliberately not calculated here.
        """
        del features
        self.total_layers = max(self.total_layers, layer_index + 1)
        grad_norm = gradients.reshape(gradients.size(0), -1).norm(p=2, dim=1).mean().item()
        fisher = gradients.square().mean().item()

        old_grad = self.gradients.get(layer_index, 0.0)
        old_fisher = self.fisher.get(layer_index, 0.0)
        self.gradients[layer_index] = self.momentum * old_grad + (1 - self.momentum) * grad_norm
        self.fisher[layer_index] = self.momentum * old_fisher + (1 - self.momentum) * fisher

        base = self.gradients[layer_index] * sqrt(self.fisher[layer_index] + 1e-8)
        depth = 1.0 + self.depth_bias * layer_index / max(1, self.total_layers - 1)
        self.importance[layer_index] = base * depth

    def weights(self) -> dict[int, float]:
        if not self.importance:
            return {}
        low, high = min(self.importance.values()), max(self.importance.values())
        if high == low:
            return {index: 1.0 for index in self.importance}
        return {
            index: (value - low) / (high - low) + 0.1 for index, value in self.importance.items()
        }


class GlobalAdaptiveTaylorImportance:
    """Estimate global position importance with first/second-order Taylor terms."""

    def __init__(
        self,
        momentum: float = 0.95,
        use_second_order: bool = True,
        layer_importance: LayerImportance | None = None,
        activation_history_size: int = 10,
    ) -> None:
        self.momentum = momentum
        self.use_second_order = use_second_order
        self.layer_importance = layer_importance
        self.activation_history_size = activation_history_size
        self.importance: dict[int, Tensor] = {}
        self.activation_history: dict[int, deque[Tensor]] = {}
        self.fisher_diagonal: dict[int, Tensor] = {}

    @torch.no_grad()
    def _negative_history(self, activation: Tensor, layer_index: int) -> Tensor:
        if layer_index not in self.activation_history:
            self.activation_history[layer_index] = deque(maxlen=self.activation_history_size)
        negative_probability = (activation < 0).float().mean(dim=0)
        self.activation_history[layer_index].append(negative_probability)
        return torch.stack(tuple(self.activation_history[layer_index]), dim=0).mean(dim=0)

    @torch.no_grad()
    def compute(self, pre_activation: Tensor, post_gradient: Tensor, layer_index: int) -> Tensor:
        negative_history = self._negative_history(pre_activation, layer_index)
        first_order = (post_gradient * pre_activation).abs() * negative_history
        value = first_order.mean(dim=0)

        if self.use_second_order:
            grad_squared = post_gradient.square().mean(dim=0)
            old_fisher = self.fisher_diagonal.get(layer_index, torch.zeros_like(value))
            fisher = 0.99 * old_fisher + 0.01 * grad_squared
            self.fisher_diagonal[layer_index] = fisher
            second_order = 0.5 * pre_activation.square().mean(dim=0) * fisher
            value = value + second_order

        if self.layer_importance is not None:
            # Preserve the original one-step lag: weight first, update statistics second.
            value = value * self.layer_importance.weights().get(layer_index, 1.0)
            self.layer_importance.update(layer_index, F.relu(pre_activation), post_gradient)
        return value

    @torch.no_grad()
    def update(self, value: Tensor, layer_index: int) -> None:
        old = self.importance.get(layer_index, torch.zeros_like(value))
        self.importance[layer_index] = self.momentum * old + (1 - self.momentum) * value

    @torch.no_grad()
    def global_masks(
        self,
        keep_ratio: float,
        *,
        keep_count: int | None = None,
    ) -> list[Tensor]:
        """Return one ``(C,H,W)`` mask per layer using a single global top-k."""
        if not 0 < keep_ratio <= 1:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        if not self.importance:
            return []

        indices = sorted(self.importance)
        flattened = [self.importance[index].flatten() for index in indices]
        global_importance = torch.cat(flattened)
        if keep_count is None:
            keep = max(1, int(global_importance.numel() * keep_ratio))
        else:
            if not 1 <= keep_count <= global_importance.numel():
                raise ValueError(
                    f"keep_count must be in [1, {global_importance.numel()}], got {keep_count}"
                )
            keep = keep_count
        top_indices = torch.topk(global_importance, keep).indices
        global_mask = torch.zeros_like(global_importance)
        global_mask[top_indices] = 1.0

        masks: list[Tensor] = []
        offset = 0
        for index in indices:
            shape = self.importance[index].shape
            size = self.importance[index].numel()
            masks.append(global_mask[offset : offset + size].reshape(shape))
            offset += size
        return masks
