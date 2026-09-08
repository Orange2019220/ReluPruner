"""Unified checkpoints with compatibility for the original experiment files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def _strip_module_prefix(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return dict(state)


def _adapt_resnet_shortcuts(
    state: Mapping[str, Tensor], expected_keys: set[str]
) -> dict[str, Tensor]:
    """Map dense ``shortcut`` keys to the local prunable ``conv_sc/bn_sc`` names."""
    adapted = dict(state)
    for stage in (2, 3, 4):
        prefix = f"layer{stage}.0."
        dense_to_prunable = {
            f"{prefix}shortcut.0.weight": f"{prefix}conv_sc.weight",
            f"{prefix}shortcut.1.weight": f"{prefix}bn_sc.weight",
            f"{prefix}shortcut.1.bias": f"{prefix}bn_sc.bias",
            f"{prefix}shortcut.1.running_mean": f"{prefix}bn_sc.running_mean",
            f"{prefix}shortcut.1.running_var": f"{prefix}bn_sc.running_var",
            f"{prefix}shortcut.1.num_batches_tracked": f"{prefix}bn_sc.num_batches_tracked",
        }
        for dense_key, prunable_key in dense_to_prunable.items():
            if prunable_key in expected_keys and dense_key in adapted:
                adapted[prunable_key] = adapted.pop(dense_key)
            elif dense_key in expected_keys and prunable_key in adapted:
                adapted[dense_key] = adapted.pop(prunable_key)
    return adapted


def read_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        # PyTorch 2.6+ defaults to weights_only=True, while old experiment files
        # may contain argparse/config objects. PyTorch 1.12 lacks this argument.
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint {path} must contain a dictionary")
    return checkpoint


def load_model(model: nn.Module, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain a model state dictionary")
    stripped = _strip_module_prefix(state)
    adapted = _adapt_resnet_shortcuts(stripped, set(model.state_dict()))
    model.load_state_dict(adapted, strict=strict)


def load_masks(checkpoint: Mapping[str, Any], device: str | torch.device) -> list[Tensor]:
    masks = checkpoint.get("masks", checkpoint.get("mask_epoch"))
    if masks is None:
        raise ValueError("checkpoint does not contain 'masks' or legacy 'mask_epoch'")
    if isinstance(masks, Mapping):
        masks = [masks[index] for index in sorted(masks)]
    return [mask.detach().to(device) for mask in masks]


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    epoch: int,
    masks: Sequence[Tensor] | None = None,
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "epoch": epoch,
        "model": model.state_dict(),
        "config": dict(config or {}),
        "metrics": dict(metrics or {}),
    }
    if masks is not None:
        payload["masks"] = [mask.detach().cpu() for mask in masks]
        payload["relu_count"] = sum(int(mask.sum().item()) for mask in masks)
    torch.save(payload, destination)
