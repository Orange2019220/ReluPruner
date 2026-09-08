import torch

from relupruner.importance import GlobalAdaptiveTaylorImportance


def test_local_taylor_formula_is_preserved():
    pre = torch.tensor([[[[-2.0, 1.0]]], [[[-4.0, -2.0]]]])
    gradient = torch.tensor([[[[3.0, 5.0]]], [[[1.0, 2.0]]]])
    calculator = GlobalAdaptiveTaylorImportance(momentum=0.0, use_second_order=True)

    actual = calculator.compute(pre, gradient, layer_index=0)
    negative_history = (pre < 0).float().mean(dim=0)
    first = (gradient * pre).abs().mul(negative_history).mean(dim=0)
    fisher = 0.01 * gradient.square().mean(dim=0)
    second = 0.5 * pre.square().mean(dim=0) * fisher

    torch.testing.assert_close(actual, first + second)


def test_global_topk_uses_one_budget_across_layers():
    calculator = GlobalAdaptiveTaylorImportance(momentum=0.0)
    calculator.importance = {
        0: torch.tensor([[[1.0, 2.0]]]),
        1: torch.tensor([[[3.0, 4.0]]]),
    }
    masks = calculator.global_masks(keep_ratio=0.5)
    assert sum(int(mask.sum()) for mask in masks) == 2
    assert masks[0].sum() == 0
    assert masks[1].sum() == 2


def test_global_topk_accepts_an_exact_relu_count():
    calculator = GlobalAdaptiveTaylorImportance(momentum=0.0)
    calculator.importance = {
        0: torch.tensor([[[1.0, 2.0]]]),
        1: torch.tensor([[[3.0, 4.0]]]),
    }

    masks = calculator.global_masks(keep_ratio=0.75, keep_count=3)

    assert sum(int(mask.sum()) for mask in masks) == 3
