import pytest
import torch

from relupruner.models import create_model, validate_model_dataset
from relupruner.pruning import count_kept_relus


def test_prunable_resnet18_has_published_cifar_relu_space():
    model = create_model("resnet18", 10, prunable=True).eval()
    sample = torch.randn(1, 3, 32, 32)
    dense = model(sample, collect_activations=True)
    masks = [torch.ones(tensor.shape[1:]) for tensor in dense.pre_activations]
    kept, total = count_kept_relus(masks)
    assert len(masks) == 16
    assert kept == total == 491_520


def test_dense_masks_match_unmasked_prunable_forward():
    torch.manual_seed(3)
    model = create_model("resnet18", 10, prunable=True).eval()
    sample = torch.randn(2, 3, 32, 32)
    reference = model(sample, collect_activations=True)
    masks = [torch.ones(tensor.shape[1:]) for tensor in reference.pre_activations]
    masked = model(sample, masks=masks)
    torch.testing.assert_close(reference.logits, masked.logits)


def test_teacher_and_student_keep_local_shortcut_names():
    teacher = create_model("resnet18", 10)
    student = create_model("resnet18", 10, prunable=True)
    assert "layer2.0.shortcut.0.weight" in teacher.state_dict()
    assert "layer2.0.conv_sc.weight" in student.state_dict()


def test_wrn_22_8_mask_count_and_forward():
    model = create_model("wrn_22_8", 100, prunable=True).eval()
    output = model(torch.randn(1, 3, 32, 32), collect_activations=True)
    assert output.logits.shape == (1, 100)
    assert len(output.pre_activations) == 19


def test_wrn_22_8_rejects_tiny_imagenet():
    with pytest.raises(ValueError, match="does not support tiny_imagenet"):
        validate_model_dataset("wrn_22_8", "tiny_imagenet")
