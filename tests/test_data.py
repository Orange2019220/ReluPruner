from pathlib import Path

from PIL import Image

from relupruner.data import TinyImageNetValidation


def test_tiny_imagenet_official_validation_layout(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (8, 8), color="red").save(image_root / "sample.JPEG")
    (tmp_path / "val_annotations.txt").write_text(
        "sample.JPEG\tn00000001\t0\t0\t8\t8\n",
        encoding="utf-8",
    )

    dataset = TinyImageNetValidation(tmp_path, {"n00000001": 7})
    image, target = dataset[0]

    assert len(dataset) == 1
    assert image.mode == "RGB"
    assert target == 7


def test_tiny_imagenet_class_organized_validation_layout(tmp_path: Path):
    image_root = tmp_path / "images" / "n00000001"
    image_root.mkdir(parents=True)
    Image.new("RGB", (8, 8), color="blue").save(image_root / "sample.JPEG")
    (tmp_path / "val_annotations.txt").write_text(
        "sample.JPEG\tn00000001\t0\t0\t8\t8\n",
        encoding="utf-8",
    )

    dataset = TinyImageNetValidation(tmp_path, {"n00000001": 3})

    assert len(dataset) == 1
    assert dataset[0][1] == 3
