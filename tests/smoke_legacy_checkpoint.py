"""Verify that a real checkpoint from the original repository loads strictly."""

from __future__ import annotations

import argparse

import torch

from relupruner.checkpoints import load_masks, load_model, read_checkpoint
from relupruner.models import create_model
from relupruner.pruning import count_kept_relus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--prunable", action="store_true")
    args = parser.parse_args()
    checkpoint = read_checkpoint(args.checkpoint)
    model = create_model("resnet18", args.num_classes, prunable=args.prunable)
    load_model(model, checkpoint, strict=True)
    if args.prunable:
        masks = load_masks(checkpoint, "cpu")
        logits = model(torch.randn(1, 3, 32, 32), masks=masks).logits
        kept, total = count_kept_relus(masks)
        print(
            f"legacy_checkpoint_ok masks={len(masks)} relus={kept}/{total} "
            f"logits={tuple(logits.shape)}"
        )
    else:
        logits = model(torch.randn(1, 3, 32, 32)).logits
        print(f"legacy_checkpoint_ok teacher logits={tuple(logits.shape)}")


if __name__ == "__main__":
    main()
