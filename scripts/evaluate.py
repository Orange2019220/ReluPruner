from __future__ import annotations

import argparse
from pathlib import Path

from relupruner.checkpoints import load_masks, load_model, read_checkpoint
from relupruner.engine import evaluate

from .common import add_data_arguments, add_model_argument, add_runtime_arguments, make_model, setup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a teacher or pruned checkpoint")
    add_data_arguments(parser)
    add_model_argument(parser)
    add_runtime_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device, (_, validation_loader, info) = setup(args, load_train=False)
    checkpoint = read_checkpoint(args.checkpoint, device)
    has_masks = "masks" in checkpoint or "mask_epoch" in checkpoint
    args._device = device
    model = make_model(args, info.num_classes, prunable=has_masks)
    load_model(model, checkpoint)
    masks = load_masks(checkpoint, device) if has_masks else None
    metrics = evaluate(
        model,
        validation_loader,
        device,
        masks,
        max_batches=args.max_val_batches,
    )
    print(f"loss={metrics.loss:.4f} accuracy={metrics.accuracy:.2f}")


if __name__ == "__main__":
    main()
