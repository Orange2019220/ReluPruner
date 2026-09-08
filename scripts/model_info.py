from __future__ import annotations

import argparse

import torch

from relupruner.data import DATASETS
from relupruner.models import MODEL_NAMES, create_model, validate_model_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show a model's maskable ReLU total")
    parser.add_argument("--dataset", choices=DATASETS, default="cifar10")
    parser.add_argument("--model", choices=MODEL_NAMES, default="resnet18")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_model_dataset(args.model, args.dataset)
    info = DATASETS[args.dataset]
    model = create_model(args.model, info.num_classes, prunable=True)
    sample = torch.zeros(1, 3, info.image_size, info.image_size)
    with torch.no_grad():
        output = model(sample, collect_activations=True)
    total_relus = sum(activation[0].numel() for activation in output.pre_activations)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"dataset={args.dataset} model={args.model} "
        f"total_maskable_relus={total_relus} parameters={parameters}"
    )


if __name__ == "__main__":
    main()
