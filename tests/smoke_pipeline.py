"""One-batch smoke test for pruning, finetuning, evaluation, and checkpoint I/O."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader, TensorDataset

from relupruner.checkpoints import load_masks, load_model, read_checkpoint, save_checkpoint
from relupruner.engine import (
    evaluate,
    finetune_epoch,
    initialize_dense_masks,
    train_pruning_epoch,
)
from relupruner.importance import GlobalAdaptiveTaylorImportance, LayerImportance
from relupruner.models import create_model
from relupruner.pruning import ProgressiveMaskScheduler, count_kept_relus


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    loader = DataLoader(
        TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1])),
        batch_size=2,
    )
    teacher = create_model("resnet18", 10).to(device).eval()
    student = create_model("resnet18", 10, prunable=True).to(device)
    masks = initialize_dense_masks(student, 32, device)
    importance = GlobalAdaptiveTaylorImportance(
        layer_importance=LayerImportance(momentum=0.95, depth_bias=0.2)
    )
    scheduler = ProgressiveMaskScheduler(
        target_keep_ratio=0.5,
        update_interval=1,
        warmup_epochs=0,
        target_epoch_ratio=0.0,
        total_epochs=1,
    )
    scheduler.set_epoch(1)
    # The local scheduler doubles the update interval once the target epoch is reached.
    scheduler.batch_counter = 1
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
    pruning_metrics, masks = train_pruning_epoch(
        student, teacher, loader, optimizer, device, masks, importance, scheduler
    )
    kept, total = count_kept_relus(masks)
    assert kept == total // 2

    finetune_metrics = finetune_epoch(student, teacher, loader, optimizer, device, masks, beta=1.0)
    evaluation = evaluate(student, loader, device, masks)

    with TemporaryDirectory() as directory:
        path = f"{directory}/smoke.pt"
        save_checkpoint(path, student, epoch=1, masks=masks)
        checkpoint = read_checkpoint(path)
        reloaded = create_model("resnet18", 10, prunable=True)
        load_model(reloaded, checkpoint)
        assert len(load_masks(checkpoint, "cpu")) == 16

    print(
        f"smoke_ok pruning_loss={pruning_metrics.loss:.4f} "
        f"finetune_loss={finetune_metrics.loss:.4f} eval_loss={evaluation.loss:.4f} "
        f"relus={kept}/{total}"
    )


if __name__ == "__main__":
    main()
