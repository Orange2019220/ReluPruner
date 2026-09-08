"""Training and evaluation loops for the four-step ReLUPruner workflow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .importance import GlobalAdaptiveTaylorImportance
from .losses import distillation_kl, pram_loss
from .models import ModelOutput
from .pruning import ProgressiveMaskScheduler
from .utils import AverageMeter, accuracy


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    classification_loss: float = 0.0
    kd_loss: float = 0.0
    pram_loss: float = 0.0


def _move_masks(masks: Sequence[Tensor], device: torch.device) -> list[Tensor]:
    return [mask.to(device, non_blocking=True) for mask in masks]


@torch.no_grad()
def initialize_dense_masks(model: nn.Module, image_size: int, device: torch.device) -> list[Tensor]:
    sample = torch.zeros(1, 3, image_size, image_size, device=device)
    output: ModelOutput = model(sample, collect_activations=True)
    return [torch.ones(tensor.shape[1:], device=device) for tensor in output.pre_activations]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    masks: Sequence[Tensor] | None = None,
    max_batches: int | None = None,
) -> EpochMetrics:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses, accuracies = AverageMeter(), AverageMeter()
    local_masks = _move_masks(masks, device) if masks is not None else None
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        output: ModelOutput = model(images, masks=local_masks)
        loss = criterion(output.logits, targets)
        batch = images.size(0)
        losses.update(loss.item(), batch)
        accuracies.update(accuracy(output.logits, targets), batch)
    return EpochMetrics(losses.average, accuracies.average)


def train_teacher_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> EpochMetrics:
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses, accuracies = AverageMeter(), AverageMeter()
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        output: ModelOutput = model(images)
        loss = criterion(output.logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch = images.size(0)
        losses.update(loss.item(), batch)
        accuracies.update(accuracy(output.logits.detach(), targets), batch)
    return EpochMetrics(losses.average, accuracies.average, losses.average)


def train_pruning_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    masks: list[Tensor],
    importance: GlobalAdaptiveTaylorImportance,
    mask_scheduler: ProgressiveMaskScheduler,
    *,
    gamma: float = 1.0,
    alpha: float = 1.0,
    temperature: float = 4.0,
    target_relus: int | None = None,
    max_batches: int | None = None,
) -> tuple[EpochMetrics, list[Tensor]]:
    student.train()
    teacher.eval()
    criterion = nn.CrossEntropyLoss()
    losses, classifications, distillations, accuracies = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )
    masks = _move_masks(masks, device)

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        student_output: ModelOutput = student(images, masks=masks, collect_activations=True)
        with torch.no_grad():
            teacher_output: ModelOutput = teacher(images)

        classification = criterion(student_output.logits, targets)
        distillation = distillation_kl(student_output.logits, teacher_output.logits, temperature)
        loss = gamma * classification + alpha * distillation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        for index, (pre, post) in enumerate(
            zip(student_output.pre_activations, student_output.post_activations)
        ):
            if post.grad is None:
                raise RuntimeError(f"gradient was not retained for prunable activation {index}")
            value = importance.compute(pre.detach(), post.grad.detach(), index)
            importance.update(value, index)

        optimizer.step()
        if mask_scheduler.should_update():
            scheduled_ratio = mask_scheduler.keep_ratio()
            final_keep_count = (
                target_relus
                if target_relus is not None
                and scheduled_ratio == mask_scheduler.target_keep_ratio
                else None
            )
            updated = importance.global_masks(
                scheduled_ratio,
                keep_count=final_keep_count,
            )
            if updated:
                masks = updated

        batch = images.size(0)
        losses.update(loss.item(), batch)
        classifications.update(classification.item(), batch)
        distillations.update(distillation.item(), batch)
        accuracies.update(accuracy(student_output.logits.detach(), targets), batch)

    return (
        EpochMetrics(
            losses.average,
            accuracies.average,
            classifications.average,
            distillations.average,
        ),
        masks,
    )


def finetune_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    masks: Sequence[Tensor],
    *,
    gamma: float = 0.5,
    alpha: float = 0.5,
    beta: float = 1000.0,
    temperature: float = 4.0,
    max_batches: int | None = None,
) -> EpochMetrics:
    student.train()
    teacher.eval()
    criterion = nn.CrossEntropyLoss()
    masks = _move_masks(masks, device)
    losses, classifications, distillations, relations, accuracies = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        student_output: ModelOutput = student(images, masks=masks)
        with torch.no_grad():
            teacher_output: ModelOutput = teacher(images)
        classification = criterion(student_output.logits, targets)
        distillation = distillation_kl(student_output.logits, teacher_output.logits, temperature)
        relation = pram_loss(student_output.features[1:], teacher_output.features[1:])
        loss = gamma * classification + alpha * distillation + beta * relation

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch = images.size(0)
        losses.update(loss.item(), batch)
        classifications.update(classification.item(), batch)
        distillations.update(distillation.item(), batch)
        relations.update(relation.item(), batch)
        accuracies.update(accuracy(student_output.logits.detach(), targets), batch)

    return EpochMetrics(
        losses.average,
        accuracies.average,
        classifications.average,
        distillations.average,
        relations.average,
    )
