"""Branch B encoder: the Meta-Polyp CAFormer backbone via timm.

The Meta-Polyp paper (and its released code) uses a CAFormer-S18 encoder with
depth ``[3, 3, 9, 3]`` and stage widths ``[64, 128, 320, 512]``. Stages 1-2 use
ConvFormer (separable-convolution) token mixers and stages 3-4 use
self-attention. The matching timm model is ``caformer_s18.sail_in1k``.

Interface required by AGENTS.md section 3.1: ``forward(x) -> list[Tensor]`` of
four maps at strides 4, 8, 16, 32, plus ``.out_channels``.
"""

from __future__ import annotations

from typing import Sequence

import timm
from torch import Tensor, nn

from src.models.convnext_enc import EXPECTED_STRIDES, SIZE_DIVISOR

#: Stage widths of CAFormer-S18 / the Meta-Polyp encoder.
META_POLYP_CHANNELS: tuple[int, ...] = (64, 128, 320, 512)


class MetaPolypEncoder(nn.Module):
    """CAFormer-S18 feature extractor (timm ``features_only``).

    Args:
        model_name: timm model with tag, e.g. ``"caformer_s18.sail_in1k"``.
        pretrained: Load the matching ImageNet weights. Never enabled in tests or
            the smoke run, so nothing is downloaded there.
        out_indices: Stage indices to return; must select strides 4, 8, 16, 32.

    Attributes:
        out_channels: ``[64, 128, 320, 512]``, checked at construction.
        strides: ``[4, 8, 16, 32]``.
        is_pretrained: Whether ImageNet weights were requested.

    Forward:
        ``x: (B, 3, H, W)`` -> ``list[Tensor]`` of
        ``(B, 64, H/4, W/4)``, ``(B, 128, H/8, W/8)``,
        ``(B, 320, H/16, W/16)``, ``(B, 512, H/32, W/32)``.
    """

    def __init__(
        self,
        model_name: str = "caformer_s18.sail_in1k",
        pretrained: bool = False,
        out_indices: Sequence[int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.is_pretrained = bool(pretrained)
        # timm 1.0.30 supports features_only for caformer_s18, so no forward hooks
        # or forward_intermediates fallback is needed (verified before this class
        # was written; the feature_info below is the authoritative contract).
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
        )
        info = self.backbone.feature_info
        self.out_channels: list[int] = list(info.channels())
        self.strides: list[int] = list(info.reduction())

        if tuple(self.out_channels) != META_POLYP_CHANNELS or tuple(self.strides) != EXPECTED_STRIDES:
            raise ValueError(
                f"{model_name} produces channels={self.out_channels} strides={self.strides}; "
                f"the Meta-Polyp encoder must produce channels={list(META_POLYP_CHANNELS)} "
                f"at strides={list(EXPECTED_STRIDES)}"
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
