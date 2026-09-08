from __future__ import annotations

import argparse
from pathlib import Path

from relupruner.checkpoints import load_model, read_checkpoint, save_checkpoint
from relupruner.engine import evaluate, initialize_dense_masks, train_pruning_epoch
from relupruner.importance import GlobalAdaptiveTaylorImportance, LayerImportance
from relupruner.pruning import ProgressiveMaskScheduler, count_kept_relus

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
    parser = argparse.ArgumentParser(description="Train with global Taylor ReLU pruning")
    add_data_arguments(parser)
    add_model_argument(parser)
    add_runtime_arguments(parser)
    add_optimizer_arguments(parser, lr=0.05, epochs=150)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pruning"))
    parser.add_argument("--init-from-teacher", action="store_true")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--global-keep-ratio",
        type=float,
        default=None,
        help="fraction of all maskable ReLU positions to retain (default: 0.1)",
    )
    budget.add_argument(
        "--target-relus",
        type=int,
        help="exact number of maskable ReLU positions to retain",
    )
    parser.add_argument("--update-interval", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--target-epoch-ratio", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device, (train_loader, validation_loader, info) = setup(args)
    args._device = device
    teacher = make_model(args, info.num_classes, prunable=False)
    teacher_checkpoint = read_checkpoint(args.teacher, device)
    load_model(teacher, teacher_checkpoint)
    teacher.eval()
    student = make_model(args, info.num_classes, prunable=True)
    if args.init_from_teacher:
        load_model(student, teacher_checkpoint)

    masks = initialize_dense_masks(student, info.image_size, device)
    layer_importance = LayerImportance(momentum=0.95, depth_bias=0.2)
    importance = GlobalAdaptiveTaylorImportance(
        momentum=0.95,
        use_second_order=True,
        layer_importance=layer_importance,
        activation_history_size=10,
    )
    _, total_relus = count_kept_relus(masks)
    if args.target_relus is not None:
        if not 1 <= args.target_relus <= total_relus:
            raise ValueError(
                f"target_relus must be in [1, {total_relus}] for {args.model}, "
                f"got {args.target_relus}"
            )
        target_keep_ratio = args.target_relus / total_relus
    else:
        target_keep_ratio = args.global_keep_ratio or 0.1

    print(
        f"relu_budget target={args.target_relus or int(total_relus * target_keep_ratio)} "
        f"total={total_relus} keep_ratio={target_keep_ratio:.8f}"
    )
    mask_scheduler = ProgressiveMaskScheduler(
        target_keep_ratio=target_keep_ratio,
        update_interval=args.update_interval,
        warmup_epochs=args.warmup_epochs,
        target_epoch_ratio=args.target_epoch_ratio,
        total_epochs=args.epochs,
    )
    optimizer, learning_rate_scheduler = make_optimizer(student, args)
    best, best_at_budget = -1.0, -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        mask_scheduler.set_epoch(epoch)
        train, masks = train_pruning_epoch(
            student,
            teacher,
            train_loader,
            optimizer,
            device,
            masks,
            importance,
            mask_scheduler,
            gamma=args.gamma,
            alpha=args.alpha,
            temperature=args.temperature,
            target_relus=args.target_relus,
            max_batches=args.max_train_batches,
        )
        validation = evaluate(
            student,
            validation_loader,
            device,
            masks,
            max_batches=args.max_val_batches,
        )
        learning_rate_scheduler.step()
        kept, total = count_kept_relus(masks)
        print(
            f"epoch={epoch:03d} loss={train.loss:.4f} train_acc={train.accuracy:.2f} "
            f"val_acc={validation.accuracy:.2f} relus={kept}/{total} "
            f"keep={kept / total:.5f} schedule={mask_scheduler.keep_ratio():.5f}"
        )
        metrics = {"validation_accuracy": validation.accuracy, "train_accuracy": train.accuracy}
        if validation.accuracy > best:
            best = validation.accuracy
            save_checkpoint(
                args.output_dir / "best.pt",
                student,
                epoch=epoch,
                masks=masks,
                config=config_dict(args),
                metrics=metrics,
            )
        target_keep_count = (
            args.target_relus
            if args.target_relus is not None
            else max(1, int(total * target_keep_ratio))
        )
        reached_budget = kept == target_keep_count
        if reached_budget and validation.accuracy > best_at_budget:
            best_at_budget = validation.accuracy
            save_checkpoint(
                args.output_dir / "best_at_budget.pt",
                student,
                epoch=epoch,
                masks=masks,
                config=config_dict(args),
                metrics=metrics,
            )
    print(f"best_validation_accuracy={best:.2f} best_at_budget={best_at_budget:.2f}")


if __name__ == "__main__":
    main()
