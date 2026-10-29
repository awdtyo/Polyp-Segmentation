"""Tests for the Milestone-2 model: ConvNeXt encoder, plain decoder, assembled net.

Everything here runs on CPU with randomly initialised weights: ``pretrained`` is
always False, so no weights are downloaded. Shape tests use ``convnext_tiny``
because it has the same four-stage / stride-4-8-16-32 layout as the default
``convnext_base`` at a fraction of the cost; the default's channel layout is
asserted separately without a forward pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
import torch

from src.losses import build_loss
from src.models.convnext_enc import ConvNeXtEncoder
from src.models.decoder import ChannelAlignment, PlainDecoder
from src.models.net import PolypSegNet, build_model, count_parameters
from src.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The four strides every encoder must produce (AGENTS.md section 3.1).
EXPECTED_STRIDES = (4, 8, 16, 32)
#: Branch A channel layout documented for the default encoder.
CONVNEXT_BASE_CHANNELS = [128, 256, 512, 1024]


def baseline_config(model_overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a Milestone-2 (single-encoder baseline) config.

    Args:
        model_overrides: Optional ``model.*`` values to change, e.g. ``{"msu": True}``.

    Returns:
        Resolved config with a tiny, randomly initialised ConvNeXt and no fusion/MSU.
    """
    model: dict[str, Any] = {
        "encoder_a": "convnext_tiny.fb_in1k",
        "pretrained_a": False,
        "pretrained_b": False,
        "fusion": ["single_a", "single_a", "single_a", "single_a"],
        "msu": False,
    }
    model.update(dict(model_overrides or {}))
    return load_config(REPO_ROOT / "configs" / "base.yaml", overrides={"model": model})


def test_convnext_encoder_returns_four_stage_maps() -> None:
    encoder = ConvNeXtEncoder("convnext_tiny.fb_in1k", pretrained=False)
    maps = encoder(torch.randn(2, 3, 256, 256))

    assert len(maps) == 4
    assert tuple(encoder.strides) == EXPECTED_STRIDES
    for feature, channels, stride in zip(maps, encoder.out_channels, EXPECTED_STRIDES):
        assert feature.shape == (2, channels, 256 // stride, 256 // stride)


def test_convnext_base_has_the_documented_channel_layout() -> None:
    encoder = ConvNeXtEncoder("convnext_base.fb_in1k", pretrained=False)

    assert encoder.out_channels == CONVNEXT_BASE_CHANNELS
    assert tuple(encoder.strides) == EXPECTED_STRIDES
    total, _ = count_parameters(encoder)
    assert 80_000_000 < total < 95_000_000, f"convnext_base should be ~88M params, got {total}"


def test_encoder_rejects_input_not_divisible_by_32() -> None:
    encoder = ConvNeXtEncoder("convnext_tiny.fb_in1k", pretrained=False)

    with pytest.raises(ValueError, match="divisible by 32"):
        encoder(torch.randn(1, 3, 100, 100))


def test_channel_alignment_maps_to_the_fused_width() -> None:
    alignment = ChannelAlignment(256, 128)
    out = alignment(torch.randn(2, 256, 64, 64))

    assert out.shape == (2, 128, 64, 64)


def test_plain_decoder_returns_full_resolution_logits() -> None:
    channels = (64, 128, 320, 512)
    decoder = PlainDecoder(channels)
    features = [
        torch.randn(2, c, 64 // s, 64 // s) for c, s in zip(channels, EXPECTED_STRIDES)
    ]

    logits = decoder(features)

    assert logits.shape == (2, 1, 64, 64)


def test_plain_decoder_rejects_wrong_input() -> None:
    decoder = PlainDecoder((64, 128, 320, 512))

    with pytest.raises(ValueError, match="Expected 4 feature maps"):
        decoder([torch.randn(1, 64, 16, 16)])

    with pytest.raises(ValueError, match="channels"):
        decoder([torch.randn(1, 7, 16, 16)] + [torch.randn(1, c, 64 // s, 64 // s) for c, s in zip((128, 320, 512), (8, 16, 32))])


@pytest.mark.parametrize("size", [128, 256])
def test_net_forward_shape(size: int) -> None:
    model = build_model(baseline_config()).eval()

    with torch.no_grad():
        logits = model(torch.randn(1, 3, size, size))

    assert logits.shape == (1, 1, size, size)


def test_net_returns_logits_not_probabilities() -> None:
    model = build_model(baseline_config()).eval()
    # A zero-weight head with a +10 bias must emit exactly 10.0: if a sigmoid were
    # applied inside the model the output would be clamped to (0, 1).
    with torch.no_grad():
        model.decoder.head_conv.weight.zero_()
        model.decoder.head_conv.bias.fill_(10.0)
        logits = model(torch.randn(1, 3, 128, 128))

    assert torch.allclose(logits, torch.full_like(logits, 10.0))


def test_parameter_groups_partition_the_model() -> None:
    model = build_model(baseline_config())
    groups = model.parameter_groups()

    assert set(groups) == {"encoder_a", "encoder_b", "head"}
    assert groups["encoder_b"] == []  # Branch B arrives in Milestone 3
    seen = [id(param) for params in groups.values() for param in params]
    assert len(seen) == len(set(seen)), "a parameter appears in more than one group"
    assert set(seen) == {id(param) for param in model.parameters()}


def test_head_group_holds_alignment_and_decoder_only() -> None:
    model = build_model(baseline_config())
    groups = model.parameter_groups()

    head = {id(param) for param in groups["head"]}
    align = {id(param) for param in model.align_a.parameters()}
    decoder = {id(param) for param in model.decoder.parameters()}
    assert head == align | decoder
    assert not head & {id(param) for param in model.encoder_a.parameters()}


def test_count_parameters_is_positive_and_trainable_matches() -> None:
    model = build_model(baseline_config())
    total, trainable = count_parameters(model)

    assert total > 0
    assert total == trainable


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"msu": True}, "Milestone 4"),
        ({"fusion": ["concat"] * 4}, "Milestone 4"),
        ({"fusion": "single_b"}, "Milestone 3"),
        ({"skip_convformer": True}, "skip_convformer"),
    ],
)
def test_build_model_refuses_unimplemented_parts(overrides: Mapping[str, Any], message: str) -> None:
    with pytest.raises(NotImplementedError, match=message):
        build_model(baseline_config(overrides))


def test_net_rejects_more_than_one_encoder() -> None:
    encoder = ConvNeXtEncoder("convnext_tiny.fb_in1k", pretrained=False)

    with pytest.raises(NotImplementedError, match="Milestone 3"):
        PolypSegNet(encoder, channels=(8, 16, 32, 64), encoder_b=encoder)


def test_overfit_one_batch() -> None:
    """The model must drive one batch's loss clearly down in ~40 CPU steps.

    This is the check that catches a broken gradient path (AGENTS.md section 8).
    A narrow, shallow config is used so the test finishes in about half a minute:
    the architecture under test is the same, only the widths differ.
    """
    torch.manual_seed(0)
    model = PolypSegNet(
        ConvNeXtEncoder("convnext_tiny.fb_in1k", pretrained=False),
        channels=(8, 16, 32, 64),
        fusion="single_a",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    loss_fn = build_loss({"train": {"loss": "bce_dice", "bce_weight": 1.0, "dice_weight": 1.0}})

    generator = torch.Generator().manual_seed(1)
    images = torch.randn(2, 3, 64, 64, generator=generator)
    masks = torch.zeros(2, 1, 64, 64)
    masks[:, :, 16:48, 20:44] = 1.0

    losses: list[float] = []
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), masks)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < 0.35 * losses[0], f"loss did not fall enough: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert losses[-1] < 0.6, f"final loss {losses[-1]:.3f} is too high for one batch"
