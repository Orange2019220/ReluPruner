import pytest
import torch

from relupruner.pruning import ProgressiveMaskScheduler, masked_relu


def test_masked_relu_is_relu_at_one_and_identity_at_zero():
    values = torch.tensor([[[[-2.0, -1.0, 3.0]]]])
    mask = torch.tensor([[[-0.0, 1.0, 0.0]]])
    expected = torch.tensor([[[[-2.0, 0.0, 3.0]]]])
    torch.testing.assert_close(masked_relu(values, mask), expected)


def test_progressive_schedule_matches_local_endpoints():
    scheduler = ProgressiveMaskScheduler(
        target_keep_ratio=0.1,
        warmup_epochs=5,
        target_epoch_ratio=0.6,
        total_epochs=150,
    )
    scheduler.set_epoch(4)
    assert scheduler.keep_ratio() == 1.0
    scheduler.set_epoch(scheduler.target_epoch)
    assert scheduler.keep_ratio() == pytest.approx(0.1)


def test_mask_update_interval_doubles_after_target():
    scheduler = ProgressiveMaskScheduler(update_interval=2, warmup_epochs=0, total_epochs=10)
    scheduler.set_epoch(0)
    assert scheduler.should_update() is False
    assert scheduler.should_update() is True
    scheduler.set_epoch(scheduler.target_epoch)
    scheduler.batch_counter = 0
    assert [scheduler.should_update() for _ in range(4)] == [False, False, False, True]
