"""Loss tests: correctness, ranges, differentiability and config wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.losses import BCEDiceLoss, BCELoss, DiceLoss, JaccardLoss, build_loss
from src.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def target_mask() -> torch.Tensor:
    """Return a ``(1, 1, 8, 8)`` mask with a 4x4 foreground square."""
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, 2:6, 2:6] = 1.0
    return mask


def perfect_logits() -> torch.Tensor:
    """Return logits that threshold to the target exactly."""
    return torch.where(target_mask() > 0.5, 30.0, -30.0)


def test_dice_loss_is_near_zero_for_a_perfect_prediction() -> None:
    loss = DiceLoss()(perfect_logits(), target_mask())
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_dice_loss_is_high_for_an_inverted_prediction() -> None:
    inverted = torch.where(target_mask() > 0.5, -30.0, 30.0)
    loss = DiceLoss()(inverted, target_mask())
    assert loss.item() == pytest.approx(1.0, abs=1e-5)


def test_dice_loss_is_between_zero_and_one() -> None:
    logits = torch.zeros(2, 1, 8, 8)
    loss = DiceLoss()(logits, target_mask().expand(2, 1, 8, 8).clone())
    assert 0.0 <= loss.item() <= 1.0


def test_jaccard_loss_is_near_zero_for_a_perfect_prediction() -> None:
    assert JaccardLoss()(perfect_logits(), target_mask()).item() == pytest.approx(0.0, abs=1e-5)


def test_jaccard_loss_penalises_over_prediction() -> None:
    over = torch.full((1, 1, 8, 8), 30.0)
    assert JaccardLoss()(over, target_mask()).item() > 0.3


def test_jaccard_smoothing_keeps_empty_case_finite() -> None:
    empty_target = torch.zeros(1, 1, 8, 8)
    loss = JaccardLoss(smooth=1.0)(torch.full((1, 1, 8, 8), -30.0), empty_target)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_bce_dice_is_lower_for_the_perfect_prediction() -> None:
    loss = BCEDiceLoss()
    good = loss(perfect_logits(), target_mask())
    bad = loss(-perfect_logits(), target_mask())

    assert good.item() < bad.item()
    assert good.item() >= 0.0


def test_bce_dice_weights_are_applied() -> None:
    logits = torch.zeros(1, 1, 8, 8)
    target = target_mask()
    bce_only = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)(logits, target)
    assert bce_only.item() == pytest.approx(BCELoss()(logits, target).item(), abs=1e-6)

    dice_only = BCEDiceLoss(bce_weight=0.0, dice_weight=1.0)(logits, target)
    assert dice_only.item() == pytest.approx(DiceLoss()(logits, target).item(), abs=1e-6)


def test_losses_are_differentiable() -> None:
    logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
    loss = BCEDiceLoss()(logits, target_mask())
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must match"):
        DiceLoss()(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 5, 5))
    with pytest.raises(ValueError, match="must match"):
        JaccardLoss()(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 5, 5))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bce_dice", BCEDiceLoss),
        ("dice", DiceLoss),
        ("jaccard", JaccardLoss),
        ("bce", BCELoss),
    ],
)
def test_build_loss_from_config(name: str, expected: type) -> None:
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml", overrides={"train": {"loss": name}})
    loss = build_loss(cfg)

    assert isinstance(loss, expected)
    value = loss(torch.zeros(1, 1, 8, 8), target_mask())
    assert torch.isfinite(value)
    assert value.ndim == 0


def test_build_loss_rejects_unknown_name() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    cfg["train"]["loss"] = "focal"
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss(cfg)


def test_batch_level_reduction_is_available() -> None:
    logits = torch.stack([perfect_logits()[0], -perfect_logits()[0]])
    target = target_mask().expand(2, 1, 8, 8).clone()

    per_image = DiceLoss(per_image=True)(logits, target)
    pooled = DiceLoss(per_image=False)(logits, target)

    assert per_image.item() == pytest.approx(0.5, abs=1e-4)
    # Pooled: intersection 16, sum(probs) 64, sum(target) 32 -> Dice 32/96.
    assert pooled.item() == pytest.approx(2.0 / 3.0, abs=1e-3)
    assert per_image.item() != pytest.approx(pooled.item(), abs=1e-3)
