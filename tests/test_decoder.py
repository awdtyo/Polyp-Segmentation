"""Shape and behaviour tests for channel alignment and the plain decoder."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.decoder import ChannelAlignment, ConvBlock, PlainDecoder

CHANNELS = (64, 128, 320, 512)


def stage_features(batch: int = 2, size: int = 64) -> list[torch.Tensor]:
    """Build aligned features at strides 4, 8, 16, 32.

    Args:
        batch: Batch size.
        size: Input spatial size.

    Returns:
        Four tensors ``(B, C_i, size/stride, size/stride)``.
    """
    return [
        torch.randn(batch, channels, size // stride, size // stride)
        for channels, stride in zip(CHANNELS, (4, 8, 16, 32))
    ]


def test_channel_alignment_changes_channels_only() -> None:
    align = ChannelAlignment(in_channels=128, out_channels=64, use_activation=False)
    out = align(torch.randn(2, 128, 16, 16))

    assert out.shape == (2, 64, 16, 16)
    assert any(isinstance(module, nn.BatchNorm2d) for module in align.modules())


def test_channel_alignment_widens() -> None:
    align = ChannelAlignment(in_channels=1024, out_channels=512)
    assert align(torch.randn(1, 1024, 8, 8)).shape == (1, 512, 8, 8)


def test_conv_block_preserves_spatial_size() -> None:
    block = ConvBlock(64, 64)
    out = block(torch.randn(2, 64, 12, 12))

    assert out.shape == (2, 64, 12, 12)


def test_decoder_output_is_full_resolution_logits() -> None:
    decoder = PlainDecoder(CHANNELS)
    out = decoder(stage_features(batch=2, size=64))

    assert out.shape == (2, 1, 64, 64)


def test_decoder_handles_128_and_batch_one() -> None:
    decoder = PlainDecoder(CHANNELS)
    out = decoder(stage_features(batch=1, size=128))

    assert out.shape == (1, 1, 128, 128)


def test_decoder_has_no_sigmoid() -> None:
    decoder = PlainDecoder(CHANNELS)
    assert not any(isinstance(module, nn.Sigmoid) for module in decoder.modules())


def test_decoder_rejects_wrong_stage_count() -> None:
    decoder = PlainDecoder(CHANNELS)
    with pytest.raises(ValueError, match="Expected 4 feature maps"):
        decoder(stage_features()[:3])


def test_decoder_rejects_wrong_channel_count() -> None:
    decoder = PlainDecoder(CHANNELS)
    features = stage_features()
    features[0] = torch.randn(2, 32, 16, 16)
    with pytest.raises(ValueError, match="Stage 1 has 32 channels"):
        decoder(features)


def test_decoder_gradients_flow_to_every_stage_path() -> None:
    decoder = PlainDecoder(CHANNELS)
    features = [feature.requires_grad_(True) for feature in stage_features()]
    loss = decoder(features).mean()
    loss.backward()

    assert decoder.head_conv.weight.grad is not None
    for index, feature in enumerate(features, start=1):
        assert feature.grad is not None, f"stage {index} received no gradient"
        assert float(feature.grad.abs().sum()) > 0.0
