"""Branch A encoder: ConvNeXt via timm, exposing 4 stage maps.

Interface required by AGENTS.md section 3.1: ``forward(x) -> list[Tensor]`` of four
maps at strides 4, 8, 16, 32, plus ``.out_channels``.
"""

from __future__ import annotations

from typing import Sequence

import timm
import torch
from torch import Tensor, nn

#: Strides every encoder in this repo must produce, in order.
EXPECTED_STRIDES: tuple[int, ...] = (4, 8, 16, 32)
#: Inputs must be divisible by this, because the deepest stage downsamples by 32.
SIZE_DIVISOR: int = 32


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt feature extractor (timm ``features_only``).

    Args:
        model_name: timm model with tag, e.g. ``"convnext_base.fb_in1k"``.
            ``convnext_small.fb_in1k`` and ``convnext_tiny.fb_in1k`` are the
            documented fallbacks.
        pretrained: Load the matching ImageNet weights. Never enabled in tests or
            the smoke run, so nothing is downloaded.
        out_indices: Stage indices to return; must select strides 4, 8, 16, 32.

    Attributes:
        out_channels: Channel count per stage, e.g. ``[128, 256, 512, 1024]`` for
            ``convnext_base``.
        strides: Stage strides, ``[4, 8, 16, 32]``.
        is_pretrained: Whether ImageNet weights were requested.

    Forward:
        ``x: (B, 3, H, W)`` -> ``list[Tensor]`` of
        ``(B, 128, H/4, W/4)``, ``(B, 256, H/8, W/8)``,
        ``(B, 512, H/16, W/16)``, ``(B, 1024, H/32, W/32)`` (base widths).
    """

    def __init__(
        self,
        model_name: str = "convnext_base.fb_in1k",
        pretrained: bool = False,
        out_indices: Sequence[int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.is_pretrained = bool(pretrained)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
        )
        info = self.backbone.feature_info
        self.out_channels: list[int] = list(info.channels())
        self.strides: list[int] = list(info.reduction())

        if len(self.out_channels) != 4 or tuple(self.strides) != EXPECTED_STRIDES:
            raise ValueError(
                f"{model_name} produces channels={self.out_channels} strides={self.strides}; "
                f"expected 4 maps at strides {EXPECTED_STRIDES}"
            )

    def forward(self, x: Tensor) -> list[Tensor]:
        """Extract the four stage feature maps.

        Args:
            x: ``(B, 3, H, W)`` float tensor, H and W divisible by 32.

        Returns:
            List of 4 tensors at strides 4, 8, 16, 32.

        Raises:
            ValueError: if H or W is not divisible by 32.
        """
        height, width = x.shape[-2], x.shape[-1]
        if height % SIZE_DIVISOR or width % SIZE_DIVISOR:
            raise ValueError(
                f"Input {height}x{width} must be divisible by {SIZE_DIVISOR} "
                "(the deepest stage downsamples by 32)"
            )
        return list(self.backbone(x))
