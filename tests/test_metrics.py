"""Metric tests, checked against hand-computed values.

All examples are small enough that the expected numbers can be verified by hand:

    target = [[1, 1],
              [0, 0]]          (2 foreground pixels out of 4)

    pred all ones -> intersection 2, |pred| = 4, |target| = 2
      Dice = 2*2 / (4 + 2) = 4/6 = 0.666...   IoU = 2 / (4 + 2 - 2) = 2/4 = 0.5
"""

from __future__ import annotations

import pytest
import torch

from src.metrics import (
    SMALL_LESION_PIXELS,
    SegMetrics,
    binarize,
    dice_per_image,
    iou_per_image,
    mean_dice,
    mean_iou,
    summarize_by_size,
)


def half_mask() -> torch.Tensor:
    """Return a ``(1, 1, 2, 2)`` mask with the top row as foreground."""
    return torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])


def test_dice_hand_computed_all_ones_prediction() -> None:
    pred = torch.ones(1, 1, 2, 2)
    dice = dice_per_image(pred, half_mask())

    assert dice.shape == (1,)
    assert dice.item() == pytest.approx(4.0 / 6.0, abs=1e-6)


def test_iou_hand_computed_all_ones_prediction() -> None:
    pred = torch.ones(1, 1, 2, 2)
    iou = iou_per_image(pred, half_mask())

    assert iou.item() == pytest.approx(0.5, abs=1e-6)


def test_perfect_prediction_scores_one() -> None:
    mask = half_mask()
    assert dice_per_image(mask, mask).item() == pytest.approx(1.0, abs=1e-6)
    assert iou_per_image(mask, mask).item() == pytest.approx(1.0, abs=1e-6)


def test_empty_prediction_and_empty_target_score_one() -> None:
    empty = torch.zeros(1, 1, 4, 4)
    assert dice_per_image(empty, empty).item() == pytest.approx(1.0, abs=1e-5)
    assert iou_per_image(empty, empty).item() == pytest.approx(1.0, abs=1e-5)


def test_empty_prediction_with_lesion_scores_zero() -> None:
    empty = torch.zeros(1, 1, 2, 2)
    assert dice_per_image(empty, half_mask()).item() == pytest.approx(0.0, abs=1e-6)
    assert iou_per_image(empty, half_mask()).item() == pytest.approx(0.0, abs=1e-6)


def test_metrics_are_averaged_per_image_not_pooled() -> None:
    # Image 1: target half, prediction all ones -> Dice 4/6, IoU 0.5
    # Image 2: perfect                                       -> Dice 1,   IoU 1
    pred = torch.stack([torch.ones(1, 2, 2), half_mask()[0]])
    target = torch.stack([half_mask()[0], half_mask()[0]])

    dice = dice_per_image(pred, target)
    iou = iou_per_image(pred, target)

    assert dice[0].item() == pytest.approx(4.0 / 6.0, abs=1e-6)
    assert dice[1].item() == pytest.approx(1.0, abs=1e-6)
    # Mean of per-image scores (0.8333), not pooled (12/14 = 0.8571).
    assert float(dice.mean()) == pytest.approx((4.0 / 6.0 + 1.0) / 2.0, abs=1e-6)
    assert float(dice.mean()) != pytest.approx(12.0 / 14.0, abs=1e-4)
    assert iou[0].item() == pytest.approx(0.5, abs=1e-6)
    assert float(iou.mean()) == pytest.approx(0.75, abs=1e-6)


def test_accepts_3d_batch_without_channel_axis() -> None:
    pred = torch.ones(2, 2, 2)
    target = half_mask().squeeze(1).expand(2, 2, 2).clone()

    dice = dice_per_image(pred, target)

    assert dice.shape == (2,)
    assert dice[0].item() == pytest.approx(4.0 / 6.0, abs=1e-6)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        dice_per_image(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 3, 3))
    with pytest.raises(ValueError, match="same shape"):
        iou_per_image(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 3, 3))


def test_binarize_uses_strict_threshold_on_sigmoid() -> None:
    # sigmoid(0) == 0.5, which does not count as foreground with a strict >.
    logits = torch.tensor([0.0, 0.1, -0.1])
    assert binarize(logits).tolist() == [0.0, 1.0, 0.0]


def test_binarize_with_probabilities_and_custom_threshold() -> None:
    probs = torch.tensor([0.4, 0.6, 0.9])
    assert binarize(probs, threshold=0.5, from_logits=False).tolist() == [0.0, 1.0, 1.0]
    assert binarize(probs, threshold=0.7, from_logits=False).tolist() == [0.0, 0.0, 1.0]


def test_mean_dice_and_iou_from_logits() -> None:
    target = half_mask()
    # Large positive logits on the top row, large negative below -> perfect mask.
    logits = torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]])

    assert mean_dice(logits, target) == pytest.approx(1.0, abs=1e-5)
    assert mean_iou(logits, target) == pytest.approx(1.0, abs=1e-5)


def test_accumulator_computes_mean_and_keeps_rows() -> None:
    metrics = SegMetrics()
    metrics.update(torch.ones(1, 1, 2, 2), half_mask(), ids=["caseA_001"])
    metrics.update(torch.tensor([[[[10.0, 10.0], [-10.0, -10.0]]]]), half_mask(), ids=["caseA_002"])

    summary = metrics.compute()
    rows = metrics.per_image()

    assert summary["n"] == 2
    assert summary["dice"] == pytest.approx((4.0 / 6.0 + 1.0) / 2.0, abs=1e-5)
    assert summary["iou"] == pytest.approx((0.5 + 1.0) / 2.0, abs=1e-5)
    assert [row["id"] for row in rows] == ["caseA_001", "caseA_002"]
    assert rows[0]["lesion_pixels"] == 2.0
    assert rows[0]["small_lesion"] is True


def test_accumulator_flags_large_lesions() -> None:
    metrics = SegMetrics()
    target = torch.ones(1, 1, 16, 16) * 1.0  # 256 foreground pixels
    metrics.update(target, target, ids=["big"])

    assert metrics.per_image()[0]["small_lesion"] is False
    assert metrics.per_image()[0]["lesion_pixels"] > SMALL_LESION_PIXELS


def test_accumulator_empty_summary_is_zero() -> None:
    assert SegMetrics().compute() == {"dice": 0.0, "iou": 0.0, "n": 0.0}


def test_accumulator_reset_and_id_count_check() -> None:
    metrics = SegMetrics()
    metrics.update(torch.ones(1, 1, 2, 2), half_mask())
    metrics.reset()
    assert metrics.compute()["n"] == 0

    with pytest.raises(ValueError, match="ids for a batch"):
        metrics.update(torch.ones(2, 1, 2, 2), half_mask().expand(2, 1, 2, 2), ids=["only-one"])


def test_summarize_by_size_splits_rows() -> None:
    metrics = SegMetrics()
    metrics.update(torch.ones(1, 1, 2, 2), half_mask(), ids=["small"])  # 2 foreground pixels
    big = torch.ones(1, 1, 16, 16)
    metrics.update(big, big, ids=["big"])

    summary = summarize_by_size(metrics.per_image())

    assert summary["n_small"] == 1
    assert summary["n_large"] == 1
    assert summary["dice_large"] == pytest.approx(1.0, abs=1e-6)
    assert summary["dice_small"] == pytest.approx(4.0 / 6.0, abs=1e-6)


def test_summarize_by_size_empty_group() -> None:
    summary = summarize_by_size([])
    assert summary == {"dice_small": 0.0, "dice_large": 0.0, "n_small": 0, "n_large": 0}
