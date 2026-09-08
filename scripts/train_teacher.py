from __future__ import annotations

import argparse

from relupruner.checkpoints import save_checkpoint
from relupruner.engine import evaluate, train_teacher_epoch

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
    parser = argparse.ArgumentParser(description="Train a dense ReLUPruner teacher")
    add_data_arguments(parser)
    add_model_argument(parser)
    add_runtime_arguments(parser)
    add_optimizer_arguments(parser, lr=0.05, epochs=240)
    parser.add_argument("--output", default="outputs/teacher_best.pt")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device, (train_loader, validation_loader, info) = setup(args)
    args._device = device
    model = make_model(args, info.num_classes, prunable=False)
    optimizer, scheduler = make_optimizer(model, args)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        train = train_teacher_epoch(model, train_loader, optimizer, device, args.max_train_batches)
        validation = evaluate(model, validation_loader, device, max_batches=args.max_val_batches)
        scheduler.step()
        print(
            f"epoch={epoch:03d} train_loss={train.loss:.4f} "
            f"train_acc={train.accuracy:.2f} val_acc={validation.accuracy:.2f}"
        )
        if validation.accuracy > best:
            best = validation.accuracy
            save_checkpoint(
                args.output,
                model,
                epoch=epoch,
                config=config_dict(args),
                metrics={"validation_accuracy": best},
            )
    print(f"best_validation_accuracy={best:.2f} checkpoint={args.output}")


if __name__ == "__main__":
    main()
