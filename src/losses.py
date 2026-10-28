"""Loss functions for binary segmentation.

The model always returns **logits** (AGENTS.md section 3.4), so every loss here
takes logits and applies sigmoid internally where needed. Losses are computed on
``(B, 1, H, W)``.

Dice-family terms are computed per image and then averaged, matching the metric
convention in :mod:`src.metrics`, so the loss and the reported score agree about
what "good" means for a small polyp in a big image.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _sigmoid_dice_per_image(logits: Tensor, target: Tensor, eps: float) -> Tensor:
    """Per-image soft Dice from logits.

    Args:
        logits: ``(B, 1, H, W)`` raw model output.
        target: ``(B, 1, H, W)`` with values in {0, 1}.
        eps: Smoothing term.

    Returns:
        Tensor ``(B,)`` with per-image Dice in [0, 1].
    """
    probs = torch.sigmoid(logits).flatten(1).double()  # (B, N)
    flat_target = target.flatten(1).double()  # (B, N)
    intersection = (probs * flat_target).sum(dim=1)
    denominator = probs.sum(dim=1) + flat_target.sum(dim=1)
    return ((2.0 * intersection + eps) / (denominator + eps)).to(logits.dtype)


class DiceLoss(nn.Module):
    """Soft Dice loss on sigmoid probabilities.

    Args:
        eps: Smoothing term; keeps the loss finite for empty masks and gives
            both-empty a Dice of 1.
        per_image: Average per image (True) or pool the whole batch (False).
    """

    def __init__(self, eps: float = 1e-7, per_image: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.per_image = per_image

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute the loss.

        Args:
            logits: ``(B, 1, H, W)`` raw model output.
            target: ``(B, 1, H, W)`` binary ground truth.

        Returns:
            Scalar loss tensor.
        """
        if logits.shape != target.shape:
            raise ValueError(f"logits and target must match, got {tuple(logits.shape)} and {tuple(target.shape)}")
        if self.per_image:
            return 1.0 - _sigmoid_dice_per_image(logits, target, self.eps).mean()
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum()
        denominator = probs.sum() + target.sum()
        return 1.0 - (2.0 * intersection + self.eps) / (denominator + self.eps)


class JaccardLoss(nn.Module):
    """Soft Jaccard (IoU) loss, the objective Meta-Polyp trained with.

    Args:
        smooth: Meta-Polyp's smoothing factor, added to numerator and denominator.
        per_image: Average per image (True) or pool the whole batch (False).
    """

    def __init__(self, smooth: float = 1.0, per_image: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.per_image = per_image

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute the loss.

        Args:
            logits: ``(B, 1, H, W)`` raw model output.
            target: ``(B, 1, H, W)`` binary ground truth.

        Returns:
            Scalar loss tensor.
        """
        if logits.shape != target.shape:
            raise ValueError(f"logits and target must match, got {tuple(logits.shape)} and {tuple(target.shape)}")
        probs = torch.sigmoid(logits)
        if self.per_image:
            probs_f = probs.flatten(1).double()
            target_f = target.flatten(1).double()
            intersection = (probs_f * target_f).sum(dim=1)
            union = probs_f.sum(dim=1) + target_f.sum(dim=1) - intersection
            jaccard = (intersection + self.smooth) / (union + self.smooth)
            return (1.0 - jaccard).mean().to(logits.dtype)
        intersection = (probs * target).sum()
        union = probs.sum() + target.sum() - intersection
        return 1.0 - (intersection + self.smooth) / (union + self.smooth)


class BCEDiceLoss(nn.Module):
    """BCE-with-logits plus soft Dice - the default objective.

    Args:
        bce_weight: Weight on the BCE term.
        dice_weight: Weight on the Dice term.
        dice_eps: Smoothing term for the Dice term.
        per_image: Average Dice per image (True) or pool the batch (False).
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_eps: float = 1e-7,
        per_image: bool = True,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(eps=dice_eps, per_image=per_image)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute the weighted sum.

        Args:
            logits: ``(B, 1, H, W)`` raw model output.
            target: ``(B, 1, H, W)`` binary ground truth.

        Returns:
            Scalar loss tensor.
        """
        target = target.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = self.dice(logits, target)
        return self.bce_weight * bce + self.dice_weight * dice


class BCELoss(nn.Module):
    """Plain BCE-with-logits, kept as a baseline for ablations."""

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Compute BCE with logits.

        Args:
            logits: ``(B, 1, H, W)`` raw model output.
            target: ``(B, 1, H, W)`` binary ground truth.

        Returns:
            Scalar loss tensor.
        """
        return F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype))


def build_loss(cfg: Mapping[str, Any]) -> nn.Module:
    """Build the loss named by ``train.loss`` in the config.

    Args:
        cfg: Resolved config.

    Returns:
        A loss module taking ``(logits, target)``.

    Raises:
        ValueError: for an unknown loss name.
    """
    train = cfg["train"]
    name = str(train["loss"])
    if name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=float(train["bce_weight"]),
            dice_weight=float(train["dice_weight"]),
        )
    if name == "dice":
        return DiceLoss()
    if name == "jaccard":
        return JaccardLoss()
    if name == "bce":
        return BCELoss()
    raise ValueError(f"Unknown loss {name!r}; expected one of bce_dice, dice, jaccard, bce")
