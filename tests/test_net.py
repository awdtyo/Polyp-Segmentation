"""Tests for model assembly: shapes, parameter groups, and refusal to build
anything a later milestone owns."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.net import PolypSegNet, build_model, count_parameters
from src.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = REPO_ROOT / "configs" / "exp001_convnext_base_baseline.yaml"


@pytest.fixture(scope="module")
def baseline_cfg() -> dict:
    """Baseline config with tiny encoder and no weight download (tests must be offline)."""
    return load_config(
        BASELINE,
        overrides={"model": {"encoder_a": "convnext_tiny.fb_in1k", "pretrained_a": False}},
    )


@pytest.fixture(scope="module")
def model(baseline_cfg: dict) -> PolypSegNet:
    """One shared baseline model, eval mode, no dropout concerns."""
    net = build_model(baseline_cfg)
    net.eval()
    return net


def test_forward_returns_one_channel_logits_at_full_resolution(model: PolypSegNet) -> None:
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 64, 64))

    assert out.shape == (2, 1, 64, 64)
    assert torch.isfinite(out).all()


def test_batch_size_one_works(model: PolypSegNet) -> None:
    model.train()
    out = model(torch.zeros(1, 3, 64, 64))
    model.eval()

    assert out.shape == (1, 1, 64, 64)


def test_256_input_works_at_full_resolution(baseline_cfg: dict) -> None:
    """The configured default resolution really works end to end."""
    net = build_model(baseline_cfg)
    net.eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 3, 256, 256))

    assert out.shape == (1, 1, 256, 256)


def test_encoder_output_is_four_maps(model: PolypSegNet) -> None:
    with torch.no_grad():
        maps = model.encoder_a(torch.zeros(1, 3, 64, 64))

    assert len(maps) == 4
    assert [m.shape[1] for m in maps] == model.encoder_a.out_channels


def test_parameter_groups_are_disjoint_and_complete(model: PolypSegNet) -> None:
    groups = model.parameter_groups()

    assert set(groups) == {"encoder_a", "encoder_b", "head"}
    assert groups["encoder_b"] == []
    assert groups["head"], "alignment + decoder must be in the head group"

    ids = [id(p) for params in groups.values() for p in params]
    assert len(ids) == len(set(ids)), "a parameter appears in two groups"
    assert set(ids) == {id(p) for p in model.parameters()}, "a parameter is in no group"


def test_parameter_groups_split_the_encoder_from_the_head(model: PolypSegNet) -> None:
    groups = model.parameter_groups()
    encoder_params = sum(p.numel() for p in groups["encoder_a"])
    head_params = sum(p.numel() for p in groups["head"])

    assert encoder_params > 20e6  # convnext_tiny backbone
    assert 0 < head_params < encoder_params


def test_count_parameters_matches_the_module(model: PolypSegNet) -> None:
    total, trainable = count_parameters(model)

    assert total == sum(p.numel() for p in model.parameters())
    assert trainable == total


def test_count_parameters_tracks_frozen_layers(model: PolypSegNet) -> None:
    total, trainable = count_parameters(model)
    first = next(model.parameters())
    first.requires_grad_(False)
    try:
        new_total, new_trainable = count_parameters(model)
        assert new_total == total
        assert new_trainable == trainable - first.numel()
    finally:
        first.requires_grad_(True)


def test_msu_config_raises_until_milestone_5(baseline_cfg: dict) -> None:
    cfg = load_config(BASELINE, overrides={"model": {"encoder_a": "convnext_tiny.fb_in1k", "pretrained_a": False, "msu": True}})
    with pytest.raises(NotImplementedError, match="Milestone 5|MSU"):
        build_model(cfg)


def test_hybrid_fusion_raises_until_milestone_4() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml", overrides={"model": {"pretrained_a": False}})
    with pytest.raises(NotImplementedError, match="Milestone 4"):
        build_model(cfg)


def test_single_b_string_raises_until_milestone_3(baseline_cfg: dict) -> None:
    cfg = load_config(BASELINE, overrides={"model": {"pretrained_a": False, "fusion": "single_b"}})
    with pytest.raises(NotImplementedError, match="Milestone 3|Milestone 4"):
        build_model(cfg)


def test_skip_convformer_true_raises() -> None:
    cfg = load_config(
        BASELINE,
        overrides={"model": {"pretrained_a": False, "skip_convformer": True}},
    )
    with pytest.raises(NotImplementedError, match="skip_convformer"):
        build_model(cfg)


def test_unknown_msu_mode_is_rejected_at_config_load() -> None:
    with pytest.raises(ValueError, match="msu_mode"):
        load_config(BASELINE, overrides={"model": {"msu_mode": "other"}})


def test_net_rejects_an_encoder_returning_the_wrong_count() -> None:
    class TwoStageEncoder(torch.nn.Module):
        out_channels = [4, 8, 16, 32]

        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            return [x, x]

    with pytest.raises(ValueError, match="expected 4"):
        net = PolypSegNet(encoder_a=TwoStageEncoder(), channels=(64, 128, 320, 512))
        net(torch.zeros(1, 3, 64, 64))
