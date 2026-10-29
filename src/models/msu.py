"""Multi-scale Upsampling (MSU) links for the decoder.

This implements ``msu_mode: paper`` from AGENTS.md section 3.4:

* Decoder 4 -> 1x1 conv (``C4 -> C2``) -> x4 upsample -> add to Decoder 2.
* Decoder 3 -> 1x1 conv (``C3 -> C1``) -> x4 upsample -> add to Decoder 1.

The 1x1 convolution is applied before the upsample, as requested for this
milestone. A ReLU is applied after each addition. The released Meta-Polyp
implementation uses a different second link; that variant is reserved under
``msu_mode: released`` and intentionally raises ``NotImplementedError``.
"""

from __future__ import annotations

from typing import Sequence

import torch.nn.functional as F
from torch import Tensor, nn


class MSULink(nn.Module):
    """One MSU link: project channels, upsample x4, add to a decoder map.

    Args:
        in_channels: Channel count of the source decoder output, ``C_src``.
        out_channels: Channel count of the target decoder output, ``C_dst``.

    Attributes:
        conv: 1x1 convolution from ``C_src`` to ``C_dst``, without bias.
        act: ReLU applied after the residual addition.

    Forward:
        ``(source, target)`` with shapes ``(B, C_src, H, W)`` and
        ``(B, C_dst, 4H, 4W)`` -> ``(B, C_dst, 4H, 4W)``.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"MSULink channels must be positive, got {in_channels} -> {out_channels}"
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        """Apply the projected, upsampled source to the target map.

        Args:
            source: ``(B, C_src, H, W)`` source decoder output.
            target: ``(B, C_dst, 4H, 4W)`` target decoder output.

        Returns:
            ``(B, C_dst, 4H, 4W)`` after addition and activation.

        Raises:
            ValueError: if the tensors are not 4D, have the wrong channel counts,
                or the source is not exactly one quarter of the target spatially.
        """
        if source.ndim != 4 or target.ndim != 4:
            raise ValueError(
                "MSU expects two 4D tensors, got "
                f"{tuple(source.shape)} and {tuple(target.shape)}"
            )
        if source.shape[1] != self.in_channels:
            raise ValueError(
                f"MSU source has {source.shape[1]} channels, expected {self.in_channels}"
            )
        if target.shape[1] != self.out_channels:
            raise ValueError(
                f"MSU target has {target.shape[1]} channels, expected {self.out_channels}"
            )
        expected_target_hw = (source.shape[-2] * 4, source.shape[-1] * 4)
        if tuple(target.shape[-2:]) != expected_target_hw:
            raise ValueError(
                f"MSU target spatial size {tuple(target.shape[-2:])} does not match "
                f"4x source size {expected_target_hw}"
            )

        projected = self.conv(source)  # (B, C_dst, H, W)
        upsampled = F.interpolate(
            projected,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )  # (B, C_dst, 4H, 4W)
        return self.act(target + upsampled)


class MSU(nn.Module):
    """The two paper-mode MSU links.

    Args:
        channels: Fused decoder widths ``C1..C4`` in stage order (stride 4 to
            stride 32). Defaults to the Meta-Polyp widths.

    Attributes:
        d4_to_d2: 1x1 projection from Decoder 4 to Decoder 2, then x4.
        d3_to_d1: 1x1 projection from Decoder 3 to Decoder 1, then x4.

    The decoder calls ``d4_to_d2`` before it computes Decoder 1, so the D4
    information propagates through D2 into D1. It calls ``d3_to_d1`` after
    Decoder 1. There is deliberately no combined ``forward`` that would apply
    both links after D1 and make the D4 link dead.
    """

    def __init__(self, channels: Sequence[int] = (64, 128, 320, 512)) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"channels must have 4 entries (stages 1-4), got {len(channels)}")
        c1, c2, c3, c4 = (int(c) for c in channels)
        if min(c1, c2, c3, c4) <= 0:
            raise ValueError(f"channels must be positive, got {list(channels)}")
        self.channels = [c1, c2, c3, c4]
        self.d4_to_d2 = MSULink(c4, c2)
        self.d3_to_d1 = MSULink(c3, c1)
