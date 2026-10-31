"""Tests that ``run.seed`` controls model init, data order and augmentation.

A run is only reproducible from its config if all three random streams are
seeded: the model's initial weights, the DataLoader shuffle order, and the
albumentations pipeline. These tests pin that wiring, and that ``train.main``
passes ``cfg["run"]["seed"]`` to :func:`src.utils.set_seed`.

Everything runs on CPU with random weights and the synthetic smoke data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from src import train as train_mod
from src.models.net import build_model
from src.train import build_dataloaders, synthesize_smoke_data
from src.utils import load_config, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def tiny_config(model_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolved config with a small, randomly initialised ConvNeXt.

    Args:
        model_overrides: Optional ``model.*`` values to replace.

    Returns:
        Resolved config: single-branch ConvNeXt-Tiny, nothing pretrained.
    """
    model: dict[str, Any] = {
        "encoder": "convnext",
        "encoder_a": "convnext_tiny.fb_in1k",
        "pretrained_a": False,
        "pretrained_b": False,
        "fusion": ["single_a", "single_a", "single_a", "single_a"],
    }
    model.update(dict(model_overrides or {}))
    return load_config(BASE_CONFIG, overrides={"model": model})


def synthetic_cfg(root: Path, seed: int) -> dict[str, Any]:
    """Config pointing at a synthetic dataset, for DataLoader tests.

    Args:
        root: Folder created by :func:`synthesize_smoke_data`.
        seed: Value for ``run.seed``.

    Returns:
        Resolved config with batch size 1 and no workers, so the order is exact.
    """
    cfg = tiny_config()
    cfg["run"].update({"name": "seeds", "out_dir": str(root / "runs"), "seed": seed})
    cfg["data"].update(
        {
            "root": str(root),
            "split_dir": str(root / "splits"),
            "image_ext": ".png",
            "mask_ext": ".png",
            "image_size": 64,
            "num_workers": 0,
            "pin_memory": False,
            "synthetic": True,
        }
    )
    cfg["train"].update({"epochs": 1, "batch_size": 1, "amp": False, "grad_accum_steps": 1})
    return cfg


def epoch_order(root: Path, seed: int) -> list[str]:
    """Build loaders from ``seed`` and return the ids in first-epoch order.

    Args:
        root: Synthetic dataset folder.
        seed: ``run.seed`` used to build the loaders.

    Returns:
        The training ids in the order the first epoch would visit them.
    """
    set_seed(seed)
    train_loader, _ = build_dataloaders(synthetic_cfg(root, seed))
    return [sample_id for batch in train_loader for sample_id in batch["id"]]


@pytest.fixture(scope="module")
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic dataset with 6 train images, enough for many permutations."""
    return synthesize_smoke_data(tmp_path_factory.mktemp("seed_data"), image_size=64, n_train=6, n_val=2)


def test_set_seed_makes_model_initialisation_reproducible() -> None:
    """The same seed gives identical initial weights; a different seed does not."""
    cfg = tiny_config()

    set_seed(42)
    first = build_model(cfg)
    set_seed(42)
    second = build_model(cfg)
    set_seed(43)
    third = build_model(cfg)

    first_state = first.state_dict()
    second_state = second.state_dict()
    third_state = third.state_dict()
    assert list(first_state) == list(second_state)
    assert all(torch.equal(first_state[key], second_state[key]) for key in first_state)
    assert any(
        not torch.equal(first_state[key], third_state[key]) for key in first_state
    ), "a different run.seed must change the initial weights"


def test_set_seed_makes_the_training_data_order_reproducible(data_root: Path) -> None:
    """Shuffling is driven by the seeded torch RNG, so the order repeats per seed."""
    pytest.importorskip("albumentations")
    first = epoch_order(data_root, 7)
    second = epoch_order(data_root, 7)
    others = [epoch_order(data_root, seed) for seed in (8, 9)]

    assert sorted(first) == sorted(second), "every train id must still be visited once"
    assert first == second, "the same run.seed must give the same shuffle order"
    assert any(order != first for order in others), "a different run.seed must change the order"


def test_set_seed_makes_the_augmentation_reproducible(data_root: Path) -> None:
    """The augmentation pipeline is built from run.seed, so samples repeat."""
    pytest.importorskip("albumentations")
    set_seed(7)
    train_loader, _ = build_dataloaders(synthetic_cfg(data_root, 7))
    first = train_loader.dataset[0]["image"]

    set_seed(7)
    train_loader, _ = build_dataloaders(synthetic_cfg(data_root, 7))
    same_seed = train_loader.dataset[0]["image"]

    set_seed(8)
    train_loader, _ = build_dataloaders(synthetic_cfg(data_root, 8))
    other_seed = train_loader.dataset[0]["image"]

    assert torch.equal(first, same_seed), "the same run.seed must reproduce the augmentation"
    assert not torch.equal(first, other_seed), "a different run.seed must change the augmentation"


def test_train_main_passes_the_config_seed_to_set_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``train.main`` seeds every RNG with ``run.seed`` before building the model."""
    cfg = tiny_config()
    cfg["run"].update({"name": "seed_wiring", "out_dir": str(tmp_path / "runs"), "seed": 123})
    monkeypatch.setattr(train_mod, "load_config", lambda path: cfg)

    calls: list[int] = []

    def record_seed(seed: int, *args: Any, **kwargs: Any) -> None:
        calls.append(int(seed))

    def stop_after_seeding(cfg_arg: Any) -> Any:
        raise RuntimeError("stop after the seed call")

    monkeypatch.setattr(train_mod, "set_seed", record_seed)
    monkeypatch.setattr(train_mod, "build_model", stop_after_seeding)

    with pytest.raises(RuntimeError, match="stop after the seed call"):
        train_mod.main(["--config", "ignored.yaml"])

    assert calls == [123], "set_seed must be called exactly once, with run.seed"
    assert "seed: 123" in capsys.readouterr().out
