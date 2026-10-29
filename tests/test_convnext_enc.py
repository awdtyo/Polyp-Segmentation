"""Shape and contract tests for the ConvNeXt encoder wrapper."""

from __future__ import annotations

import pytest
import torch

from src.models.convnext_enc import EXPECTED_STRIDES, SIZE_DIVISOR, ConvNeXtEncoder


@pytest.fixture(scope="module")
def tiny_encoder() -> ConvNeXtEncoder:
    """One shared ``convnext_tiny`` encoder (no weights download)."""
    return ConvNeXtEncoder("convnext_tiny.fb_in1k", pretrained=False)


@pytest.fixture(scope="module")
def base_encoder() -> ConvNeXtEncoder:
    """One shared ``convnext_base`` encoder, the default (no weights download)."""
    return ConvNeXtEncoder("convnext_base.fb_in1k", pretrained=False)


def test_attributes_match_timm(tiny_encoder: ConvNeXtEncoder) -> None:
    assert tiny_encoder.out_channels == [96, 192, 384, 768]
    assert tiny_encoder.strides == list(EXPECTED_STRIDES)
    assert tiny_encoder.is_pretrained is False


def test_default_base_channels(base_encoder: ConvNeXtEncoder) -> None:
    assert base_encoder.out_channels == [128, 256, 512, 1024]
    assert base_encoder.strides == [4, 8, 16, 32]


def test_forward_returns_four_maps_with_expected_shapes(tiny_encoder: ConvNeXtEncoder) -> None:
    out = tiny_encoder(torch.zeros(2, 3, 64, 64))

    assert isinstance(out, list) and len(out) == 4
    assert [tuple(feature.shape) for feature in out] == [
        (2, 96, 16, 16),
        (2, 192, 8, 8),
        (2, 384, 4, 4),
        (2, 768, 2, 2),
    ]


def test_base_forward_shapes(base_encoder: ConvNeXtEncoder) -> None:
    out = base_encoder(torch.zeros(1, 3, 64, 64))

    assert [tuple(feature.shape) for feature in out] == [
        (1, 128, 16, 16),
        (1, 256, 8, 8),
        (1, 512, 4, 4),
        (1, 1024, 2, 2),
    ]


def test_strides_are_4_8_16_32_on_a_256_input(tiny_encoder: ConvNeXtEncoder) -> None:
    out = tiny_encoder(torch.zeros(1, 3, 256, 256))
    heights = [feature.shape[-1] for feature in out]

    assert heights == [256 // stride for stride in EXPECTED_STRIDES]


def test_input_must_be_divisible_by_32(tiny_encoder: ConvNeXtEncoder) -> None:
    with pytest.raises(ValueError, match=f"divisible by {SIZE_DIVISOR}"):
        tiny_encoder(torch.zeros(1, 3, 65, 64))


def test_out_channels_match_actual_tensor_channels(tiny_encoder: ConvNeXtEncoder) -> None:
    out = tiny_encoder(torch.zeros(1, 3, 32, 32))

    assert [feature.shape[1] for feature in out] == tiny_encoder.out_channels


def test_parameter_count_is_positive(tiny_encoder: ConvNeXtEncoder) -> None:
    total = sum(p.numel() for p in tiny_encoder.parameters())
    assert total > 20e6  # convnext_tiny is ~28M
