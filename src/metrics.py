"""Segmentation metrics: per-image Dice and IoU.

Conventions used everywhere in this repo:

* Metrics are computed **per image** and then averaged, never pooled over the
  whole batch. An image with a big polyp does not outvote a small one.
* Predictions are thresholded at 0.5 after a sigmoid (``from_logits=True``) or
  directly (``from_logits=False``). The comparison is strictly greater than the
  threshold; a prediction of exactly 0.5 counts as background.
* Empty-mask images (no polyp) score 1.0 when the prediction is also empty, which
  is the usual convention for a smoothed ratio with ``eps > 0``.
* Per-image scores are kept, not just the mean, so small-lesion performance can be
  analysed later (AGENTS.md section 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
from torch import Tensor

#: Below this many foreground pixels a mask is "small"; used for reporting only.
SMALL_LESION_PIXELS = 32


def binarize(pred: Tensor, threshold: float = 0.5, from_logits: bool = True) -> Tensor:
    """Threshold predictions into a 0/1 mask.

    Args:
        pred: Logits or probabilities, shape ``(B, 1, H, W)`` or ``(B, H, W)``.
        threshold: Decision threshold on the probability.
        from_logits: Apply sigmoid first when True.

    Returns:
        Float tensor with the same shape as ``pred``, values in {0, 1}.
    """
    probs = torch.sigmoid(pred) if from_logits else pred
    return (probs > threshold).to(probs.dtype)


def _flatten_per_image(tensor: Tensor) -> Tensor:
    """Flatten everything except the batch axis.

    Args:
        tensor: ``(B, ...)``.

    Returns:
        ``(B, N)``.
    """
    if tensor.ndim == 0:
        raise ValueError("Expected a batched tensor with at least 1 dimension")
    return tensor.reshape(tensor.shape[0], -1)


def dice_per_image(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Dice coefficient for each image in a batch.

    Args:
        pred: Binary predictions, ``(B, 1, H, W)`` or ``(B, H, W)``.
        target: Binary ground truth with the same shape.
        eps: Smoothing term.

    Returns:
        Tensor ``(B,)`` with values in [0, 1].

    Raises:
        ValueError: if shapes differ.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {tuple(pred.shape)} and {tuple(target.shape)}")
    pred_f = _flatten_per_image(pred.double())
    target_f = _flatten_per_image(target.double())
    intersection = (pred_f * target_f).sum(dim=1)
    denominator = pred_f.sum(dim=1) + target_f.sum(dim=1)
    return ((2.0 * intersection + eps) / (denominator + eps)).to(torch.float32)


def iou_per_image(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Jaccard index (IoU) for each image in a batch.

    Args:
        pred: Binary predictions, ``(B, 1, H, W)`` or ``(B, H, W)``.
        target: Binary ground truth with the same shape.
        eps: Smoothing term.

    Returns:
        Tensor ``(B,)`` with values in [0, 1].

    Raises:
        ValueError: if shapes differ.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {tuple(pred.shape)} and {tuple(target.shape)}")
    pred_f = _flatten_per_image(pred.double())
    target_f = _flatten_per_image(target.double())
    intersection = (pred_f * target_f).sum(dim=1)
    union = pred_f.sum(dim=1) + target_f.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).to(torch.float32)


@dataclass
class SegMetrics:
    """Accumulate Dice and IoU over a full split, keeping per-image rows.

    Example:
        >>> metrics = SegMetrics()
        >>> metrics.update(logits, masks, ids=["case_1", "case_2"])  # doctest: +SKIP
        >>> metrics.compute()  # doctest: +SKIP
        {'dice': 0.91, 'iou': 0.85, 'n': 2}

    Attributes:
        threshold: Decision threshold on the sigmoid output.
        from_logits: Whether predictions passed to :meth:`update` are logits.
        rows: Per-image records with id, dice, iou and lesion size.
    """

    threshold: float = 0.5
    from_logits: bool = True
    rows: list[dict[str, float | str | bool]] = field(default_factory=list)

    def update(self, pred: Tensor, target: Tensor, ids: Sequence[str] | None = None) -> None:
        """Add one batch of predictions.

        Args:
            pred: Logits/probabilities, ``(B, 1, H, W)`` or ``(B, H, W)``.
            target: Binary ground truth, same shape as ``pred``.
            ids: Optional per-image ids, length ``B``. Defaults to the batch index.

        Raises:
            ValueError: on shape mismatch or wrong number of ids.
        """
        binary = binarize(pred, threshold=self.threshold, from_logits=self.from_logits)
        dice = dice_per_image(binary, target)
        iou = iou_per_image(binary, target)
        lesion = _flatten_per_image(target).sum(dim=1)
        batch = dice.shape[0]
        if ids is not None and len(ids) != batch:
            raise ValueError(f"Got {len(ids)} ids for a batch of {batch}")
        for i in range(batch):
            self.rows.append(
                {
                    "id": str(ids[i]) if ids is not None else str(i),
                    "dice": float(dice[i]),
                    "iou": float(iou[i]),
                    "lesion_pixels": float(lesion[i]),
                    "small_lesion": bool(lesion[i] < SMALL_LESION_PIXELS),
                }
            )

    def compute(self) -> dict[str, float]:
        """Mean Dice and IoU over everything seen so far.

        Returns:
            ``{"dice": float, "iou": float, "n": int}``; zeros when empty.
        """
        if not self.rows:
            return {"dice": 0.0, "iou": 0.0, "n": 0.0}
        n = len(self.rows)
        return {
            "dice": sum(float(row["dice"]) for row in self.rows) / n,
            "iou": sum(float(row["iou"]) for row in self.rows) / n,
            "n": float(n),
        }

    def reset(self) -> None:
        """Drop all accumulated rows."""
        self.rows.clear()

    def per_image(self) -> list[dict[str, float | str | bool]]:
        """Return the per-image records (id, dice, iou, lesion_pixels, small_lesion).

        Returns:
            The internal list of rows.
        """
        return self.rows


def mean_dice(pred: Tensor, target: Tensor, threshold: float = 0.5, from_logits: bool = True) -> float:
    """Convenience wrapper: mean Dice over a batch.

    Args:
        pred: Logits/probabilities, ``(B, 1, H, W)``.
        target: Binary ground truth, same shape.
        threshold: Decision threshold.
        from_logits: Apply sigmoid first when True.

    Returns:
        Mean Dice as a Python float.
    """
    binary = binarize(pred, threshold=threshold, from_logits=from_logits)
    return float(dice_per_image(binary, target).mean())


def mean_iou(pred: Tensor, target: Tensor, threshold: float = 0.5, from_logits: bool = True) -> float:
    """Convenience wrapper: mean IoU over a batch.

    Args:
        pred: Logits/probabilities, ``(B, 1, H, W)``.
        target: Binary ground truth, same shape.
        threshold: Decision threshold.
        from_logits: Apply sigmoid first when True.

    Returns:
        Mean IoU as a Python float.
    """
    binary = binarize(pred, threshold=threshold, from_logits=from_logits)
    return float(iou_per_image(binary, target).mean())


def summarize_by_size(rows: Iterable[dict[str, float | str | bool]]) -> dict[str, float]:
    """Split per-image rows into small and large lesions and average each.

    Small polyps are where Meta-Polyp is reported to be weaker (AGENTS.md
    section 10), so this is kept as a first-class summary.

    Args:
        rows: Records produced by :class:`SegMetrics`.

    Returns:
        ``{"dice_small": float, "dice_large": float, "n_small": int, "n_large": int}``
        with 0.0 / 0 for an empty group.
    """
    small = [float(row["dice"]) for row in rows if row["small_lesion"]]
    large = [float(row["dice"]) for row in rows if not row["small_lesion"]]
    return {
        "dice_small": sum(small) / len(small) if small else 0.0,
        "dice_large": sum(large) / len(large) if large else 0.0,
        "n_small": len(small),
        "n_large": len(large),
    }
