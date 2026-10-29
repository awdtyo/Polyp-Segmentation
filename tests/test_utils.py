"""Tests for config loading and seeding."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src import utils

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_yaml(path: Path, data: dict) -> Path:
    """Write a dict as YAML.

    Args:
        path: Destination.
        data: Content.

    Returns:
        The path.
    """
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_base_config_loads_with_expected_values() -> None:
    cfg = utils.load_config(REPO_ROOT / "configs" / "base.yaml")

    assert cfg["run"]["name"] == "exp000_base"
    assert cfg["run"]["seed"] == 42
    assert cfg["data"]["image_size"] == 256
    assert cfg["model"]["encoder"] == "dual"
    assert cfg["model"]["fusion"] == ["concat", "concat", "xattn", "xattn"]
    assert cfg["model"]["channels"] == [64, 128, 320, 512]
    assert cfg["decoder"]["msu"] is False
    assert cfg["decoder"]["msu_mode"] == "paper"
    assert cfg["train"]["loss"] == "bce_dice"
    assert cfg["logging"]["wandb"] is False


def test_partial_file_falls_back_to_defaults(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "exp.yaml", {"run": {"name": "exp001_minimal"}})
    cfg = utils.load_config(path)

    assert cfg["run"]["name"] == "exp001_minimal"
    assert cfg["data"]["image_size"] == 256  # default
    assert cfg["train"]["batch_size"] == 8  # default


def test_overrides_are_applied_last(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "exp.yaml", {"data": {"image_size": 128}})
    cfg = utils.load_config(path, overrides={"data": {"image_size": 64}, "train": {"batch_size": 2}})

    assert cfg["data"]["image_size"] == 64
    assert cfg["train"]["batch_size"] == 2


def test_defaults_are_not_mutated(tmp_path: Path) -> None:
    before = copy.deepcopy(utils.DEFAULT_CONFIG)
    path = write_yaml(tmp_path / "exp.yaml", {"data": {"image_size": 128}, "train": {"batch_size": 99}})
    cfg = utils.load_config(path)
    cfg["data"]["image_size"] = 32

    assert utils.DEFAULT_CONFIG == before


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(utils.ConfigError, match="not found"):
        utils.load_config(tmp_path / "nope.yaml")


def test_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(utils.ConfigError, match="empty"):
        utils.load_config(path)


def test_unknown_top_level_key_raises(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "exp.yaml", {"trainn": {"epochs": 3}})
    with pytest.raises(utils.ConfigError, match="unknown top-level keys"):
        utils.load_config(path)


def test_unknown_section_key_raises(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "exp.yaml", {"data": {"image_sizee": 128}})
    with pytest.raises(utils.ConfigError, match="unknown keys in 'data'"):
        utils.load_config(path)


def test_image_size_must_be_divisible_by_32(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "exp.yaml", {"data": {"image_size": 100}})
    with pytest.raises(utils.ConfigError, match="divisible by 32"):
        utils.load_config(path)


def test_fusion_list_must_have_four_entries(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "exp.yaml", {"model": {"fusion": ["concat", "xattn"]}})
    with pytest.raises(utils.ConfigError, match="4 entries"):
        utils.load_config(path)


def test_fusion_mode_must_be_known(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "exp.yaml", {"model": {"fusion": "attention"}})
    with pytest.raises(utils.ConfigError, match="model.fusion"):
        utils.load_config(path)


def test_fusion_single_string_mode_is_allowed(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "exp.yaml", {"model": {"fusion": "all_concat"}})
    cfg = utils.load_config(path)
    assert cfg["model"]["fusion"] == "all_concat"


def test_fusion_hybrid_preset_is_allowed(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "exp.yaml", {"model": {"fusion": "hybrid"}})
    cfg = utils.load_config(path)
    assert cfg["model"]["fusion"] == "hybrid"


def test_fusion_preset_name_is_not_a_stage_entry(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path / "exp.yaml", {"model": {"fusion": ["hybrid", "concat", "xattn", "xattn"]}}
    )
    with pytest.raises(utils.ConfigError, match="entry 'hybrid'"):
        utils.load_config(path)


def test_fusion_heads_must_be_positive(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "exp.yaml", {"model": {"fusion_heads": 0}})
    with pytest.raises(utils.ConfigError, match="fusion_heads"):
        utils.load_config(path)


def test_encoder_must_be_known(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "exp.yaml", {"model": {"encoder": "swin"}})
    with pytest.raises(utils.ConfigError, match="model.encoder"):
        utils.load_config(path)


def test_branch_b_configs_are_pinned_and_consistent() -> None:
    pretrained = utils.load_config(REPO_ROOT / "configs" / "exp003_caformer_s18_baseline.yaml")
    scratch = utils.load_config(REPO_ROOT / "configs" / "exp004_caformer_s18_scratch.yaml")

    for cfg in (pretrained, scratch):
        assert cfg["model"]["encoder"] == "caformer"
        assert cfg["model"]["encoder_b"] == "caformer_s18.sail_in1k"
        assert cfg["model"]["fusion"] == ["single_b"] * 4
        assert cfg["decoder"]["msu"] is False
        assert cfg["decoder"]["msu_mode"] == "paper"
        assert cfg["model"]["channels"] == [64, 128, 320, 512]

    assert pretrained["model"]["pretrained_b"] is True
    assert pretrained["train"]["lr_encoder_b"] == pytest.approx(1e-4)
    assert scratch["model"]["pretrained_b"] is False
    assert scratch["train"]["lr_encoder_b"] == pytest.approx(3e-4)


def test_dual_branch_configs_are_pinned_and_consistent() -> None:
    hybrid = utils.load_config(REPO_ROOT / "configs" / "exp005_hybrid_fusion.yaml")
    all_concat = utils.load_config(REPO_ROOT / "configs" / "exp006_all_concat.yaml")

    for cfg in (hybrid, all_concat):
        assert cfg["model"]["encoder"] == "dual"
        assert cfg["model"]["encoder_a"] == "convnext_base.fb_in1k"
        assert cfg["model"]["encoder_b"] == "caformer_s18.sail_in1k"
        assert cfg["model"]["pretrained_a"] is True
        assert cfg["model"]["pretrained_b"] is True
        assert cfg["decoder"]["msu"] is False
        assert cfg["decoder"]["msu_mode"] == "paper"
        assert cfg["model"]["channels"] == [64, 128, 320, 512]
        assert cfg["train"]["lr_convnext"] == pytest.approx(3e-5)
        assert cfg["train"]["lr_encoder_b"] == pytest.approx(1e-4)
        assert cfg["train"]["lr_head"] == pytest.approx(3e-4)

    assert hybrid["model"]["fusion"] == "hybrid"
    assert all_concat["model"]["fusion"] == "all_concat"


def test_exp007_enables_paper_msu() -> None:
    cfg = utils.load_config(REPO_ROOT / "configs" / "exp007_hybrid_msu.yaml")

    assert cfg["model"]["encoder"] == "dual"
    assert cfg["model"]["fusion"] == "hybrid"
    assert cfg["model"]["pretrained_a"] is True
    assert cfg["model"]["pretrained_b"] is True
    assert cfg["decoder"] == {"msu": True, "msu_mode": "paper"}


def test_unknown_loss_raises(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "exp.yaml", {"train": {"loss": "focal"}})
    with pytest.raises(utils.ConfigError, match="train.loss"):
        utils.load_config(path)


def test_get_dotted_key() -> None:
    cfg = utils.load_config(REPO_ROOT / "configs" / "base.yaml")

    assert utils.get(cfg, "data.image_size") == 256
    assert utils.get(cfg, "data.missing", 7) == 7
    with pytest.raises(KeyError, match="data.missing"):
        utils.get(cfg, "data.missing")


def test_save_config_roundtrip(tmp_path: Path) -> None:
    cfg = utils.load_config(REPO_ROOT / "configs" / "base.yaml")
    out = utils.save_config(cfg, tmp_path / "nested" / "config_resolved.yaml")

    assert out.is_file()
    reloaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert reloaded["data"]["image_size"] == 256
    assert reloaded["model"]["fusion"] == ["concat", "concat", "xattn", "xattn"]


def test_set_seed_makes_numpy_and_torch_repeatable() -> None:
    utils.set_seed(1234)
    first = (np.random.rand(3), torch.rand(3))
    utils.set_seed(1234)
    second = (np.random.rand(3), torch.rand(3))

    assert np.array_equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_set_seed_different_seed_differs() -> None:
    utils.set_seed(1)
    first = np.random.rand(3)
    utils.set_seed(2)
    second = np.random.rand(3)

    assert not np.array_equal(first, second)
