"""Channel alignment and the plain U-Net decoder.

Milestone 2 decoder, per AGENTS.md section 3.4 without fusion and without MSU:

* Decoder 4: conv block on the fused (here: aligned) stage-4 map.
* Decoders 3, 2, 1: ``ConvTranspose2d`` x2 upsample of the previous decoder
  output, concat with the stage skip, then a conv block (3x3 conv + norm +
  activation, twice).
* Head: upsample x4 to full resolution, ``1x1`` conv to one channel.

The model returns **logits**; sigmoid is applied only in the loss and at inference.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


def make_norm(channels: int) -> nn.Module:
    """Create the norm layer used everywhere in the decoder.

    Args:
        channels: Channel count.

    Returns:
        ``BatchNorm2d``, matching the Meta-Polyp decoder's use of batch norm.
    """
    return nn.BatchNorm2d(channels)


class ChannelAlignment(nn.Module):
    """1x1 conv + norm, mapping one stage to the shared fused width ``C_i``.

    AGENTS.md section 3.2. Alignment runs before fusion so both branches speak the
    same channel language.

    Args:
        in_channels: Encoder channel count for this stage.
        out_channels: Target width ``C_i``.
        use_activation: Add a ReLU after the norm. Off by default because
            AGENTS.md section 3.2 specifies 1x1 conv + norm only; the decoder or
            the fusion block applies its own activation.

    Attributes:
        conv: 1x1 projection without bias.
        norm: BatchNorm over ``out_channels``.
    """

    def __init__(self, in_channels: int, out_channels: int, use_activation: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = make_norm(out_channels)
        self.act = nn.ReLU(inplace=True) if use_activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Align channels.

        Args:
            x: ``(B, C_in, H, W)``.

        Returns:
            ``(B, C_out, H, W)``.
        """
        return self.act(self.norm(self.conv(x)))


class ConvBlock(nn.Module):
    """Two 3x3 conv + norm + activation layers.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block.

        Args:
            x: ``(B, C_in, H, W)``.

        Returns:
            ``(B, C_out, H, W)``, same spatial size.
        """
        return self.block(x)


class PlainDecoder(nn.Module):
    """U-Net decoder with transposed-conv upsampling and concat skips.

    Args:
        channels: Fused width per stage ``C1..C4``, ordered stage 1 (stride 4) to
            stage 4 (stride 32). Defaults to the Meta-Polyp widths.
        head_channels: Width of the pre-head projection at full resolution.

    Forward:
        ``features``: 4 tensors, already aligned to ``channels``, ordered stage 1
        to stage 4 at strides 4, 8, 16, 32 ->
        logits ``(B, 1, H, W)`` at the input resolution.
    """

    def __init__(self, channels: Sequence[int] = (64, 128, 320, 512), head_channels: int = 32) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"channels must have 4 entries (stages 1-4), got {len(channels)}")
        c1, c2, c3, c4 = (int(c) for c in channels)
        self.channels = [c1, c2, c3, c4]

        self.dec4 = ConvBlock(c4, c4)

        self.up4_to_3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3)

        self.up3_to_2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)

        self.up2_to_1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        # Full-resolution head: x4 transposed conv, then 1x1 to one logit channel.
        self.head_up = nn.ConvTranspose2d(c1, head_channels, kernel_size=4, stride=4)
        self.head_norm = make_norm(head_channels)
        self.head_act = nn.ReLU(inplace=True)
        self.head_conv = nn.Conv2d(head_channels, 1, kernel_size=1)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        """Decode aligned multi-scale features to logits.

        Args:
            features: ``[F1, F2, F3, F4]`` with shapes
                ``(B, C1, H/4, W/4)`` ... ``(B, C4, H/32, W/32)``.

        Returns:
            ``(B, 1, H, W)`` logits.

        Raises:
            ValueError: if the feature list is not 4 maps or channel counts differ.
        """
        if len(features) != 4:
            raise ValueError(f"Expected 4 feature maps, got {len(features)}")
        f1, f2, f3, f4 = features
        for index, (feature, expected) in enumerate(zip(features, self.channels), start=1):
            if feature.shape[1] != expected:
                raise ValueError(f"Stage {index} has {feature.shape[1]} channels, expected {expected}")

        x = self.dec4(f4)  # (B, C4, H/32, W/32)
        x = self.dec3(self._cat(self.up4_to_3(x), f3))  # (B, C3, H/16, W/16)
        x = self.dec2(self._cat(self.up3_to_2(x), f2))  # (B, C2, H/8, W/8)
        x = self.dec1(self._cat(self.up2_to_1(x), f1))  # (B, C1, H/4, W/4)
        x = self.head_act(self.head_norm(self.head_up(x)))  # (B, head, H, W)
        return self.head_conv(x)  # (B, 1, H, W)

    @staticmethod
    def _cat(upsampled: Tensor, skip: Tensor) -> Tensor:
        """Concatenate an upsampled decoder map with its skip.

        Args:
            upsampled: ``(B, C, H, W)`` decoder output after 2x upsampling.
            skip: ``(B, C, H, W)`` aligned stage map.

        Returns:
            ``(B, 2C, H, W)``.

        Raises:
            ValueError: if spatial sizes differ, which means an unexpected input size.
        """
        if upsampled.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                f"Upsampled {tuple(upsampled.shape[-2:])} does not match skip {tuple(skip.shape[-2:])}"
            )
        return torch.cat([upsampled, skip], dim=1)
