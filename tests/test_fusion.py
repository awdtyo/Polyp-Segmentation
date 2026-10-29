"""CPU tests for Milestone 4 fusion.

All weights are random and every pretrained flag is false, so these tests never
download anything. Shape tests at 256 px use ``convnext_tiny`` for Branch A to
keep the CPU cost down; the default ConvNeXt-Base layout is already covered by
the encoder tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
import torch

from src.models.convnext_enc import ConvNeXtEncoder
from src.models.fusion import (
    ConcatFusion,
    CrossAttentionFusion,
    FusionStack,
    build_2d_sinusoidal_encoding,
    fusion_branch,
    resolve_fusion_modes,
)
from src.models.metapolyp_enc import MetaPolypEncoder
from src.models.net import PolypSegNet, build_model, count_parameters
from src.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = REPO_ROOT / "configs" / "exp001_convnext_base_baseline.yaml"
CAFORMER = REPO_ROOT / "configs" / "exp003_caformer_s18_baseline.yaml"
CHANNELS = (64, 128, 320, 512)
STRIDES = (4, 8, 16, 32)


def aligned_maps(batch: int = 2, size: int = 256, channels: tuple[int, ...] = CHANNELS) -> list[torch.Tensor]:
    """Random maps already aligned to the shared widths.

    Args:
        batch: Batch size.
        size: Input image size the stage shapes correspond to.
        channels: Stage widths ``C1..C4``.

    Returns:
        Four tensors at strides 4, 8, 16, 32.
    """
    return [torch.randn(batch, c, size // s, size // s) for c, s in zip(channels, STRIDES)]


def make_dual_net(fusion: str | list[str]) -> PolypSegNet:
    """Build a small dual-encoder net with random weights.

    Args:
        fusion: Preset or per-stage list.

    Returns:
        A ConvNeXt-tiny + CAFormer-S18 net, in train mode.
    """
    return PolypSegNet(
        encoder_a=ConvNeXtEncoder("convnext_tiny.fb_in1k", pretrained=False),
        channels=CHANNELS,
        encoder_b=MetaPolypEncoder("caformer_s18.sail_in1k", pretrained=False),
        fusion=fusion,
    )


def test_resolve_fusion_presets_and_lists() -> None:
    assert resolve_fusion_modes("hybrid") == ["concat", "concat", "xattn", "xattn"]
    assert resolve_fusion_modes("all_concat") == ["concat"] * 4
    assert resolve_fusion_modes("single_a") == ["single_a"] * 4
    assert resolve_fusion_modes("single_b") == ["single_b"] * 4
    assert resolve_fusion_modes("xattn") == ["xattn"] * 4
    explicit = ["concat", "concat", "xattn", "xattn"]
    assert resolve_fusion_modes(explicit) == explicit

    assert fusion_branch(resolve_fusion_modes("hybrid")) == "dual"
    assert fusion_branch(resolve_fusion_modes("all_concat")) == "dual"
    assert fusion_branch(resolve_fusion_modes("single_a")) == "single_a"
    assert fusion_branch(resolve_fusion_modes("single_b")) == "single_b"


@pytest.mark.parametrize(
    "fusion",
    [
        ["single_a", "concat", "xattn", "xattn"],
        ["single_a", "single_b", "single_a", "single_b"],
        ["concat", "concat", "concat"],
        ["concat", "concat", "hybrid", "xattn"],
    ],
)
def test_resolve_fusion_rejects_invalid_lists(fusion: list[str]) -> None:
    with pytest.raises(ValueError):
        resolve_fusion_modes(fusion)


def test_concat_fusion_returns_the_stage_width() -> None:
    fusion = ConcatFusion(12)
    first = torch.randn(2, 12, 16, 16)
    second = torch.randn(2, 12, 16, 16)

    out = fusion(first, second)

    assert out.shape == (2, 12, 16, 16)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(("channels", "size"), [(320, 16), (512, 8)])
@pytest.mark.parametrize("pos_enc_2d", [False, True])
def test_cross_attention_fusion_shapes(channels: int, size: int, pos_enc_2d: bool) -> None:
    fusion = CrossAttentionFusion(channels, num_heads=8, pos_enc_2d=pos_enc_2d)
    query = torch.randn(2, channels, size, size)
    key_value = torch.randn(2, channels, size, size)

    out = fusion(query, key_value)

    assert out.shape == (2, channels, size, size)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(("channels", "size"), [(320, 16), (512, 8)])
def test_cross_attention_output_depends_on_both_branches(channels: int, size: int) -> None:
    """Changing either branch's input must change the fused stage feature."""
    torch.manual_seed(0)
    fusion = CrossAttentionFusion(channels, num_heads=8, pos_enc_2d=False).eval()
    query = torch.randn(2, channels, size, size)
    key_value = torch.randn(2, channels, size, size)
    query_perturbed = query + 0.5 * torch.randn_like(query)
    key_value_perturbed = key_value + 0.5 * torch.randn_like(key_value)

    with torch.no_grad():
        base = fusion(query, key_value)
        changed_query = fusion(query_perturbed, key_value)
        changed_key_value = fusion(query, key_value_perturbed)

    assert not torch.allclose(base, changed_query), "the output ignores the ConvNeXt query branch"
    assert not torch.allclose(base, changed_key_value), "the output ignores the Meta-Polyp key/value branch"


def test_fusion_stack_shapes_at_256() -> None:
    maps_a = aligned_maps()
    maps_b = aligned_maps()

    for preset in ("hybrid", "all_concat"):
        stack = FusionStack(resolve_fusion_modes(preset), CHANNELS).eval()
        with torch.no_grad():
            fused = stack(maps_a, maps_b)
        assert len(fused) == 4
        for out, channels, stride in zip(fused, CHANNELS, STRIDES):
            assert out.shape == (2, channels, 256 // stride, 256 // stride)


def test_fusion_stack_rejects_single_branch_modes() -> None:
    with pytest.raises(ValueError, match="dual-branch"):
        FusionStack(["single_a"] * 4, CHANNELS)


@pytest.mark.parametrize(
    "encoder_factory",
    [
        lambda: ConvNeXtEncoder("convnext_tiny.fb_in1k", pretrained=False),
        lambda: MetaPolypEncoder("caformer_s18.sail_in1k", pretrained=False),
    ],
    ids=["convnext_tiny", "caformer_s18"],
)
def test_token_counts_at_256_are_256_and_64(encoder_factory: Callable[[], torch.nn.Module]) -> None:
    encoder = encoder_factory().eval()
    with torch.no_grad():
        maps = encoder(torch.randn(1, 3, 256, 256))

    stage_3_tokens = maps[2].shape[-2] * maps[2].shape[-1]
    stage_4_tokens = maps[3].shape[-2] * maps[3].shape[-1]
    assert (stage_3_tokens, stage_4_tokens) == (256, 64)


@pytest.mark.parametrize("preset", ["hybrid", "all_concat", "single_a", "single_b"])
def test_full_net_returns_logits_at_256_for_every_preset(preset: str) -> None:
    if preset == "single_a":
        net = PolypSegNet(
            encoder_a=ConvNeXtEncoder("convnext_tiny.fb_in1k", pretrained=False),
            channels=CHANNELS,
            fusion=preset,
        )
    elif preset == "single_b":
        net = PolypSegNet(
            encoder_a=None,
            channels=CHANNELS,
            encoder_b=MetaPolypEncoder("caformer_s18.sail_in1k", pretrained=False),
            fusion=preset,
        )
    else:
        net = make_dual_net(preset)
    net.eval()

    with torch.no_grad():
        logits = net(torch.randn(2, 3, 256, 256))

    assert logits.shape == (2, 1, 256, 256)
    assert torch.isfinite(logits).all()


def test_single_modes_match_the_existing_baseline_shapes_and_parameter_counts() -> None:
    tiny_convnext = ConvNeXtEncoder("convnext_tiny.fb_in1k", pretrained=False)
    single_a = PolypSegNet(encoder_a=tiny_convnext, channels=CHANNELS, fusion="single_a")
    baseline_a = build_model(
        load_config(
            BASELINE,
            overrides={
                "model": {
                    "encoder": "convnext",
                    "encoder_a": "convnext_tiny.fb_in1k",
                    "pretrained_a": False,
                    "msu": False,
                }
            },
        )
    )
    assert single_a.align_b is None and single_a.encoder_b is None
    assert single_a.fusion_stack is None
    assert count_parameters(single_a) == count_parameters(baseline_a)

    single_b = PolypSegNet(
        encoder_a=None,
        channels=CHANNELS,
        encoder_b=MetaPolypEncoder("caformer_s18.sail_in1k", pretrained=False),
        fusion="single_b",
    )
    baseline_b = build_model(load_config(CAFORMER, overrides={"model": {"pretrained_b": False}}))
    assert single_b.align_a is None and single_b.encoder_a is None
    assert single_b.fusion_stack is None
    assert count_parameters(single_b) == count_parameters(baseline_b)


def test_dual_parameter_groups_cover_both_encoders_and_new_modules() -> None:
    net = make_dual_net("hybrid")
    groups = net.parameter_groups()

    assert set(groups) == {"encoder_a", "encoder_b", "head"}
    assert groups["encoder_a"] and groups["encoder_b"] and groups["head"]
    seen = [id(param) for params in groups.values() for param in params]
    assert len(seen) == len(set(seen))
    assert set(seen) == {id(param) for param in net.parameters()}

    head = {id(param) for param in groups["head"]}
    fusion_params = {id(param) for param in net.fusion_stack.parameters()}
    decoder_params = {id(param) for param in net.decoder.parameters()}
    align_params = {
        id(param)
        for module in (net.align_a, net.align_b)
        for param in module.parameters()  # type: ignore[union-attr]
    }
    assert head == fusion_params | decoder_params | align_params


def test_dual_net_requires_both_encoders() -> None:
    with pytest.raises(ValueError, match="both encoder_a and encoder_b"):
        PolypSegNet(
            encoder_a=ConvNeXtEncoder("convnext_tiny.fb_in1k", pretrained=False),
            channels=CHANNELS,
            encoder_b=None,
            fusion="hybrid",
        )


def test_cross_attention_heads_must_divide_the_stage_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        CrossAttentionFusion(320, num_heads=7)


def test_positional_encoding_shape_and_no_parameters() -> None:
    encoding = build_2d_sinusoidal_encoding(16, 16, 320)
    off = CrossAttentionFusion(320, pos_enc_2d=False)
    on = CrossAttentionFusion(320, pos_enc_2d=True)

    assert encoding.shape == (1, 256, 320)
    assert torch.isfinite(encoding).all()
    assert sum(p.numel() for p in on.parameters()) == sum(p.numel() for p in off.parameters())


def test_positional_encoding_flag_changes_the_attention_output() -> None:
    torch.manual_seed(0)
    off = CrossAttentionFusion(320, pos_enc_2d=False).eval()
    on = CrossAttentionFusion(320, pos_enc_2d=True).eval()
    on.load_state_dict(off.state_dict())
    query = torch.randn(2, 320, 16, 16)
    key_value = torch.randn(2, 320, 16, 16)

    with torch.no_grad():
        without = off(query, key_value)
        with_encoding = on(query, key_value)

    assert not torch.allclose(without, with_encoding), "pos_enc_2d=True must change the attention input"
