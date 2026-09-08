"""Read one real dataset batch and connect it to a prunable model."""

from __future__ import annotations

import argparse

from relupruner.data import create_dataloaders
from relupruner.models import create_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    args = parser.parse_args()
    print(f"loading {args.dataset} metadata", flush=True)
    train_loader, validation_loader, info = create_dataloaders(
        args.dataset, args.data_root, batch_size=2, workers=0
    )
    print(f"dataset ready: {len(train_loader.dataset)} training images", flush=True)
    sample_image, sample_target = train_loader.dataset[0]
    print(
        f"first sample ready: {tuple(sample_image.shape)}, target={sample_target}",
        flush=True,
    )
    images, targets = next(iter(validation_loader))
    print("first validation batch ready", flush=True)
    output = create_model("resnet18", info.num_classes, prunable=True)(
        images, collect_activations=True
    )
    print(
        "real_data_ok",
        args.dataset,
        tuple(images.shape),
        tuple(targets.shape),
        tuple(output.logits.shape),
        len(output.pre_activations),
        info.num_classes,
    )


if __name__ == "__main__":
    main()
