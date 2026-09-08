"""Losses for ReLUPruner distillation and fixed-mask finetuning."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor
from torch.nn import functional as F


def distillation_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 4.0,
) -> Tensor:
    return (
        F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        )
        * temperature**2
    )


def attention_map(feature: Tensor) -> Tensor:
    attention = feature.square().mean(dim=1).flatten(1)
    return F.normalize(attention, p=2, dim=1)


def pram_loss(student_features: Sequence[Tensor], teacher_features: Sequence[Tensor]) -> Tensor:
    """Position-aware relation/attention matching used in local stage two."""
    if len(student_features) != len(teacher_features):
        raise ValueError("student and teacher must expose the same number of PRAM features")
    if not student_features:
        raise ValueError("PRAM requires at least one feature map")
    return sum(
        F.mse_loss(attention_map(student), attention_map(teacher))
        for student, teacher in zip(student_features, teacher_features)
    )
