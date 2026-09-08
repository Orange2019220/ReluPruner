from __future__ import annotations

import argparse
from pathlib import Path

import torch

from relupruner.data import DATASETS, create_dataloaders
from relupruner.models import MODEL_NAMES, create_model, validate_model_dataset
from relupruner.utils import resolve_device, set_seed


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=DATASETS, default="cifar10")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true", help="download CIFAR if absent")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="enable CUDA pinned-memory loading (off by default for WSL/PyTorch 1.x compatibility)",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="limit batches per training epoch for smoke tests",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="limit validation batches for smoke tests",
    )


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=0)


def add_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=MODEL_NAMES, default="resnet18")


def add_optimizer_arguments(parser: argparse.ArgumentParser, *, lr: float, epochs: int) -> None:
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--learning-rate", type=float, default=lr)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lr-milestones", default="150,180,210")
    parser.add_argument("--lr-decay", type=float, default=0.1)


def setup(args: argparse.Namespace, *, load_train: bool = True):
    set_seed(args.seed)
    device = resolve_device(args.device)
    loaders = create_dataloaders(
        args.dataset,
        args.data_root,
        args.batch_size,
        args.workers,
        args.download,
        args.pin_memory,
        load_train,
    )
    return device, loaders


def make_optimizer(model: torch.nn.Module, args: argparse.Namespace):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    milestones = [int(value) for value in args.lr_milestones.split(",") if value]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=args.lr_decay)
    return optimizer, scheduler


def make_model(args: argparse.Namespace, num_classes: int, *, prunable: bool):
    validate_model_dataset(args.model, args.dataset)
    return create_model(args.model, num_classes, prunable=prunable).to(args._device)


def config_dict(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }
