"""CPU tests for Milestone 5 Multi-scale Upsampling (MSU).

The paper-mode links are tested directly and through the assembled model. Every
weight is random and every pretrained flag is false, so no download happens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from src.models.convnext_enc import ConvNeXtEncoder
from src.models.decoder import PlainDecoder
from src.models.msu import MSU
from src.models.net import PolypSegNet, build_model, count_parameters
from src.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANNELS = (64, 128, 320, 512)

#: Decoder state_dict keys at the Milestone 4 commit. MSU must add none of these
#: when ``decoder.msu`` is false.
MILESTONE4_DECODER_KEYS = frozenset(
    {
        "dec4.block.0.weight",
        "dec4.block.1.weight",
        "dec4.block.1.bias",
        "dec4.block.1.running_mean",
        "dec4.block.1.running_var",
        "dec4.block.1.num_batches_tracked",
        "dec4.block.3.weight",
        "dec4.block.4.weight",
        "dec4.block.4.bias",
        "dec4.block.4.running_mean",
        "dec4.block.4.running_var",
        "dec4.block.4.num_batches_tracked",
        "up4_to_3.weight",
        "up4_to_3.bias",
        "dec3.block.0.weight",
        "dec3.block.1.weight",
        "dec3.block.1.bias",
        "dec3.block.1.running_mean",
        "dec3.block.1.running_var",
        "dec3.block.1.num_batches_tracked",
        "dec3.block.3.weight",
        "dec3.block.4.weight",
        "dec3.block.4.bias",
        "dec3.block.4.running_mean",
        "dec3.block.4.running_var",
        "dec3.block.4.num_batches_tracked",
        "up3_to_2.weight",
        "up3_to_2.bias",
        "dec2.block.0.weight",
        "dec2.block.1.weight",
        "dec2.block.1.bias",
        "dec2.block.1.running_mean",
        "dec2.block.1.running_var",
        "dec2.block.1.num_batches_tracked",
        "dec2.block.3.weight",
        "dec2.block.4.weight",
        "dec2.block.4.bias",
        "dec2.block.4.running_mean",
        "dec2.block.4.running_var",
        "dec2.block.4.num_batches_tracked",
        "up2_to_1.weight",
        "up2_to_1.bias",
        "dec1.block.0.weight",
        "dec1.block.1.weight",
        "dec1.block.1.bias",
        "dec1.block.1.running_mean",
        "dec1.block.1.running_var",
        "dec1.block.1.num_batches_tracked",
        "dec1.block.3.weight",
        "dec1.block.4.weight",
        "dec1.block.4.bias",
        "dec1.block.4.running_mean",
        "dec1.block.4.running_var",
        "dec1.block.4.num_batches_tracked",
        "head_up.weight",
        "head_up.bias",
        "head_norm.weight",
        "head_norm.bias",
        "head_norm.running_mean",
        "head_norm.running_var",
        "head_norm.num_batches_tracked",
        "head_conv.weight",
        "head_conv.bias",
    }
)


def msu_config(msu: bool, msu_mode: str = "paper") -> dict[str, Any]:
    """Build a tiny single-branch config with MSU on or off.

    Args:
        msu: Whether to enable the MSU links.
        msu_mode: ``paper`` or ``released``.

    Returns:
        Resolved config with random ConvNeXt-Tiny weights.
    """
    return load_config(
        REPO_ROOT / "configs" / "base.yaml",
        overrides={
            "model": {
                "encoder": "convnext",
                "encoder_a": "convnext_tiny.fb_in1k",
                "pretrained_a": False,
                "pretrained_b": False,
                "fusion": ["single_a", "single_a", "single_a", "single_a"],
            },
            "decoder": {"msu": msu, "msu_mode": msu_mode},
        },
    )


def test_msu_link_shapes() -> None:
    msu = MSU(CHANNELS)
    d1 = torch.randn(2, 64, 16, 16)
    d2 = torch.randn(2, 128, 8, 8)
    d3 = torch.randn(2, 320, 4, 4)
    d4 = torch.randn(2, 512, 2, 2)

    d2_out = msu.d4_to_d2(d4, d2)
    d1_out = msu.d3_to_d1(d3, d1)

    assert d1_out.shape == d1.shape
    assert d2_out.shape == d2.shape
    assert torch.isfinite(d1_out).all()
    assert torch.isfinite(d2_out).all()


def test_d4_information_reaches_d2_through_the_link() -> None:
    """With the main D2 path zeroed, only the D4 link can change D2."""
    msu = MSU(CHANNELS)
    with torch.no_grad():
        msu.d4_to_d2.conv.weight.fill_(1.0)
        msu.d3_to_d1.conv.weight.fill_(1.0)

    d1 = torch.zeros(1, 64, 16, 16)
    d2 = torch.zeros(1, 128, 8, 8)
    d3 = torch.zeros(1, 320, 4, 4)
    d4 = torch.ones(1, 512, 2, 2)

    d1_without_d3 = msu.d3_to_d1(d3, d1)
    d2_with_d4 = msu.d4_to_d2(d4, d2)
    d2_with_different_d4 = msu.d4_to_d2(2.0 * d4, d2)

    assert torch.count_nonzero(d1_without_d3) == 0
    assert torch.count_nonzero(d2_with_d4) > 0
    assert not torch.allclose(d2_with_d4, d2_with_different_d4)


def test_d4_link_is_not_dead_in_the_decoder() -> None:
    """D4->D2 must run before D1, so changing D4 must change the logits."""
    torch.manual_seed(0)
    decoder = PlainDecoder((8, 16, 32, 64), msu=True)
    decoder.eval()
    with torch.no_grad():
        for parameter in decoder.dec2.parameters():
            parameter.zero_()

    f1 = torch.randn(2, 8, 16, 16)
    f2 = torch.randn(2, 16, 8, 8)
    f3 = torch.randn(2, 32, 4, 4)
    f4 = torch.randn(2, 64, 2, 2)

    with torch.no_grad():
        logits_a = decoder([f1, f2, f3, f4])
        logits_b = decoder([f1, f2, f3, 2.0 * f4])

    assert logits_a.shape == (2, 1, 64, 64)
    assert not torch.allclose(logits_a, logits_b)


def test_d3_information_reaches_d1_through_the_link() -> None:
    """With the main D1 path zeroed, only the D3 link can change D1."""
    msu = MSU(CHANNELS)
    with torch.no_grad():
        msu.d4_to_d2.conv.weight.fill_(1.0)
        msu.d3_to_d1.conv.weight.fill_(1.0)

    d1 = torch.zeros(1, 64, 16, 16)
    d2 = torch.zeros(1, 128, 8, 8)
    d3 = torch.ones(1, 320, 4, 4)
    d4 = torch.zeros(1, 512, 2, 2)

    d1_with_d3 = msu.d3_to_d1(d3, d1)
    d2_without_d4 = msu.d4_to_d2(d4, d2)
    d1_with_different_d3 = msu.d3_to_d1(2.0 * d3, d1)

    assert torch.count_nonzero(d2_without_d4) == 0
    assert torch.count_nonzero(d1_with_d3) > 0
    assert not torch.allclose(d1_with_d3, d1_with_different_d3)


def test_msu_parameter_count_is_the_two_1x1_projections() -> None:
    msu = MSU(CHANNELS)
    expected = 512 * 128 + 320 * 64

    assert sum(p.numel() for p in msu.parameters()) == expected == 86_016


def test_msu_increases_model_parameters_by_the_link_count() -> None:
    off = build_model(msu_config(False))
    on = build_model(msu_config(True))
    off_total, _ = count_parameters(off)
    on_total, _ = count_parameters(on)
    link_total = sum(p.numel() for p in on.decoder.msu.parameters())  # type: ignore[union-attr]

    assert on_total - off_total == link_total == 86_016


def test_msu_off_preserves_the_milestone4_decoder_state_dict_keys() -> None:
    model = PolypSegNet(
        encoder_a=ConvNeXtEncoder("convnext_tiny.fb_in1k", pretrained=False),
        channels=CHANNELS,
        fusion="single_a",
    )

    assert model.decoder.msu is None
    assert set(model.decoder.state_dict()) == MILESTONE4_DECODER_KEYS
    assert not any("msu" in key for key in model.state_dict())


def test_msu_gradients_reach_both_links_and_the_encoder() -> None:
    torch.manual_seed(0)
    model = PolypSegNet(
        encoder_a=ConvNeXtEncoder("convnext_tiny.fb_in1k", pretrained=False),
        channels=(8, 16, 32, 64),
        fusion="single_a",
        msu=True,
    )
    model(torch.randn(2, 3, 64, 64)).sum().backward()

    msu = model.decoder.msu
    assert msu is not None
    for link in (msu.d4_to_d2, msu.d3_to_d1):
        grad = link.conv.weight.grad
        assert grad is not None, "an MSU link received no gradient"
        assert float(grad.abs().sum()) > 0.0

    encoder_grads = [p.grad for p in model.encoder_a.parameters() if p.grad is not None]
    assert encoder_grads, "the encoder received no gradients"
    assert any(float(grad.abs().sum()) > 0.0 for grad in encoder_grads)

    head_ids = {id(p) for p in model.parameter_groups()["head"]}
    assert {id(p) for p in msu.parameters()} <= head_ids


def test_released_msu_is_accepted_by_validation_but_not_built() -> None:
    cfg = msu_config(True, "released")
    assert cfg["decoder"]["msu_mode"] == "released"

    with pytest.raises(NotImplementedError, match="released"):
        build_model(cfg)
    with pytest.raises(NotImplementedError, match="released"):
        PlainDecoder(CHANNELS, msu=True, msu_mode="released")
