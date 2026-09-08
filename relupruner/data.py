"""Small, explicit dataset surface for the published experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DatasetInfo:
    num_classes: int
    image_size: int


DATASETS = {
    "cifar10": DatasetInfo(10, 32),
    "cifar100": DatasetInfo(100, 32),
    "tiny_imagenet": DatasetInfo(200, 64),
}


class TinyImageNetValidation(Dataset):
    """Read the canonical Tiny-ImageNet validation split without reorganizing it."""

    def __init__(
        self,
        validation_root: Path,
        class_to_index: dict[str, int],
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        annotations = validation_root / "val_annotations.txt"
        image_root = validation_root / "images"
        if not annotations.is_file() or not image_root.is_dir():
            raise FileNotFoundError(
                "Tiny-ImageNet validation requires val_annotations.txt and val/images under "
                f"{validation_root}"
            )

        records = [
            fields
            for line in annotations.read_text(encoding="utf-8").splitlines()
            if len(fields := line.split("\t")) >= 2
        ]
        if not records:
            raise ValueError(f"no validation samples were found in {annotations}")
        flat_layout = (image_root / records[0][0]).is_file()

        samples: list[tuple[Path, int]] = []
        for fields in records:
            if len(fields) < 2:
                continue
            filename, class_name = fields[:2]
            if class_name not in class_to_index:
                raise ValueError(f"unknown Tiny-ImageNet class {class_name!r} in {annotations}")
            image_path = (
                image_root / filename
                if flat_layout
                else image_root / class_name / filename
            )
            samples.append((image_path, class_to_index[class_name]))

        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, target


def create_dataloaders(
    name: str,
    root: str | Path,
    batch_size: int = 128,
    workers: int = 4,
    download: bool = False,
    pin_memory: bool = False,
    load_train: bool = True,
) -> tuple[DataLoader | None, DataLoader, DatasetInfo]:
    """Create train/validation loaders with transforms used by the local code."""
    try:
        from torchvision import datasets, transforms
    except ImportError as error:
        raise RuntimeError("torchvision is required to load image datasets") from error

    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose one of {', '.join(DATASETS)}")
    root = Path(root).expanduser()
    info = DATASETS[name]

    if name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        dataset_class = datasets.CIFAR10
    elif name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        dataset_class = datasets.CIFAR100
    else:
        mean, std = None, None
        dataset_class = None

    train_steps = [
        transforms.RandomCrop(info.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    validation_steps = [transforms.ToTensor()]
    if mean is not None and std is not None:
        normalize = transforms.Normalize(mean, std)
        train_steps.append(normalize)
        validation_steps.append(normalize)
    train_transform = transforms.Compose(train_steps)
    validation_transform = transforms.Compose(validation_steps)

    if dataset_class is not None:
        train_set = (
            dataset_class(
                root=root, train=True, transform=train_transform, download=download
            )
            if load_train
            else None
        )
        validation_set = dataset_class(
            root=root, train=False, transform=validation_transform, download=download
        )
    else:
        tiny_root = root / "tiny-imagenet-200" if (root / "tiny-imagenet-200").is_dir() else root
        train_root, validation_root = tiny_root / "train", tiny_root / "val"
        if not train_root.is_dir() or not validation_root.is_dir():
            raise FileNotFoundError(
                "Tiny-ImageNet must contain train/ and class-organized val/ directories under "
                f"{tiny_root}"
            )
        train_set = (
            datasets.ImageFolder(train_root, transform=train_transform) if load_train else None
        )
        class_to_index = (
            train_set.class_to_idx
            if train_set is not None
            else {
                directory.name: index
                for index, directory in enumerate(
                    sorted(path for path in train_root.iterdir() if path.is_dir())
                )
            }
        )
        if (validation_root / "val_annotations.txt").is_file():
            validation_set = TinyImageNetValidation(
                validation_root,
                class_to_index,
                validation_transform,
            )
        else:
            validation_set = datasets.ImageFolder(
                validation_root, transform=validation_transform
            )

    train_loader = (
        DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=workers > 0,
        )
        if train_set is not None
        else None
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    return train_loader, validation_loader, info
