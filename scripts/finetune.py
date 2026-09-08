from __future__ import annotations

import argparse
from pathlib import Path

from relupruner.checkpoints import load_masks, load_model, read_checkpoint, save_checkpoint
from relupruner.engine import evaluate, finetune_epoch

from .common import (
    add_data_arguments,
    add_model_argument,
    add_optimizer_arguments,
    add_runtime_arguments,
    config_dict,
    make_model,
    make_optimizer,
    setup,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finetune a fixed ReLUPruner mask")
    add_data_arguments(parser)
    add_model_argument(parser)
    add_runtime_arguments(parser)
    add_optimizer_arguments(parser, lr=0.01, epochs=240)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--stage-one", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/finetune_best.pt"))
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1000.0)
    parser.add_argument("--temperature", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device, (train_loader, validation_loader, info) = setup(args)
    args._device = device
    teacher = make_model(args, info.num_classes, prunable=False)
    load_model(teacher, read_checkpoint(args.teacher, device))
    teacher.eval()
    student = make_model(args, info.num_classes, prunable=True)
    stage_one = read_checkpoint(args.stage_one, device)
    load_model(student, stage_one)
    masks = load_masks(stage_one, device)
    optimizer, scheduler = make_optimizer(student, args)
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        train = finetune_epoch(
            student,
            teacher,
            train_loader,
            optimizer,
            device,
            masks,
            gamma=args.gamma,
            alpha=args.alpha,
            beta=args.beta,
            temperature=args.temperature,
            max_batches=args.max_train_batches,
        )
        validation = evaluate(
            student,
            validation_loader,
            device,
            masks,
            max_batches=args.max_val_batches,
        )
        scheduler.step()
        print(
            f"epoch={epoch:03d} loss={train.loss:.4f} ce={train.classification_loss:.4f} "
            f"kd={train.kd_loss:.4f} pram={train.pram_loss:.6f} "
            f"train_acc={train.accuracy:.2f} val_acc={validation.accuracy:.2f}"
        )
        if validation.accuracy > best:
            best = validation.accuracy
            save_checkpoint(
                args.output,
                student,
                epoch=epoch,
                masks=masks,
                config=config_dict(args),
                metrics={"validation_accuracy": best},
            )
    print(f"best_validation_accuracy={best:.2f} checkpoint={args.output}")


if __name__ == "__main__":
    main()
