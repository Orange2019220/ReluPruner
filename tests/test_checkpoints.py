import torch

from relupruner.checkpoints import load_masks, load_model
from relupruner.models import create_model


def test_legacy_stage_two_mask_key_is_supported():
    masks = load_masks({"mask_epoch": [torch.ones(2, 3, 3)]}, "cpu")
    assert len(masks) == 1
    assert masks[0].shape == (2, 3, 3)


def test_teacher_shortcuts_can_initialize_local_prunable_student():
    teacher = create_model("resnet18", 10)
    student = create_model("resnet18", 10, prunable=True)
    load_model(student, {"model": teacher.state_dict()})
    torch.testing.assert_close(
        student.layer2[0].conv_sc.weight,
        teacher.layer2[0].shortcut[0].weight,
    )
