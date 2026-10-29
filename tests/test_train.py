"""Tests for the training pipeline: LR groups, scheduler, resume, logging, smoke run.

Everything runs on CPU with random weights and synthetic images (the ``conftest``
fixture or :func:`src.train.synthesize_smoke_data`). No real data, no downloads,
no full training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch
import yaml
from torch import Tensor, nn

from src import train as train_mod
from src.datasets import PolypDataset, build_train_transform
from src.losses import build_loss
from src.models.net import build_model
from src.train import (
    METRICS_COLUMNS,
    append_metrics_row,
    build_dataloaders,
    build_optimizer,
    build_scheduler,
    get_learning_rates,
    load_checkpoint,
    make_grad_scaler,
    reseed_train_augmentation,
    save_checkpoint,
    synthesize_smoke_data,
    train_one_epoch,
    write_run_info,
)
from src.utils import load_config, save_config, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
BASELINE_CONFIG = REPO_ROOT / "configs" / "exp001_convnext_base_baseline.yaml"
CAFORMER_CONFIG = REPO_ROOT / "configs" / "exp003_caformer_s18_baseline.yaml"


def tiny_config(model_overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolved Milestone-2 config with a tiny, randomly initialised ConvNeXt.

    Args:
        model_overrides: Optional ``model.*`` values to replace.

    Returns:
        Resolved config (no fusion, no MSU, nothing pretrained).
    """
    model: dict[str, Any] = {
        "encoder_a": "convnext_tiny.fb_in1k",
        "pretrained_a": False,
        "pretrained_b": False,
        "fusion": ["single_a", "single_a", "single_a", "single_a"],
        "msu": False,
    }
    model.update(dict(model_overrides or {}))
    return load_config(BASE_CONFIG, overrides={"model": model})


def mini_run_config(root: Path, out_dir: Path, epochs: int) -> dict[str, Any]:
    """Config for a real (non-smoke) run on the synthetic smoke dataset.

    Args:
        root: Folder created by :func:`synthesize_smoke_data`.
        out_dir: Run output folder (``<out_dir>/<run_name>``).
        epochs: Total epoch count.

    Returns:
        Resolved config that :func:`src.train.main` can consume directly.
    """
    cfg = tiny_config()
    cfg["run"].update({"name": "mini", "out_dir": str(out_dir), "seed": 7})
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
    cfg["train"].update({"epochs": epochs, "batch_size": 2, "amp": False, "grad_accum_steps": 1})
    cfg["logging"].update({"wandb": False, "log_images_every": 1, "num_log_images": 2})
    return cfg


def make_batch(batch_size: int = 2, size: int = 64) -> dict[str, Any]:
    """Build one fixed synthetic batch.

    Args:
        batch_size: Number of images.
        size: Spatial size.

    Returns:
        ``{"image": (B,3,S,S), "mask": (B,1,S,S), "id": [str]}`` with a rectangle mask.
    """
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(batch_size, 3, size, size, generator=generator)
    masks = torch.zeros(batch_size, 1, size, size)
    masks[:, :, size // 4 : size // 2, size // 4 : size // 2] = 1.0
    return {"image": images, "mask": masks, "id": [f"id{i}" for i in range(batch_size)]}


def test_optimizer_has_differential_learning_rates() -> None:
    cfg = tiny_config()
    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg)

    rates = {group["name"]: group["lr"] for group in optimizer.param_groups}

    assert rates["encoder_a"] == pytest.approx(cfg["train"]["lr_convnext"])
    assert rates["head"] == pytest.approx(cfg["train"]["lr_head"])
    # Branch B has no parameters yet (Milestone 3), so its empty group is dropped
    # rather than handing AdamW an empty parameter list.
    assert "encoder_b" not in rates
    assert get_learning_rates(optimizer)["encoder_b"] == 0.0
    assert get_learning_rates(optimizer)["head"] == pytest.approx(rates["head"])
    for group in optimizer.param_groups:
        assert group["weight_decay"] == pytest.approx(cfg["train"]["weight_decay"])


def test_optimizer_covers_every_parameter_once() -> None:
    cfg = tiny_config()
    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg)

    seen = [id(param) for group in optimizer.param_groups for param in group["params"]]
    assert len(seen) == len(set(seen))
    assert set(seen) == {id(param) for param in model.parameters()}


def test_branch_b_optimizer_uses_the_encoder_b_learning_rate() -> None:
    cfg = load_config(CAFORMER_CONFIG, overrides={"model": {"pretrained_b": False}})
    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg)

    rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert rates["encoder_b"] == pytest.approx(cfg["train"]["lr_encoder_b"])
    assert "encoder_a" not in rates
    assert rates["head"] == pytest.approx(cfg["train"]["lr_head"])
    assert get_learning_rates(optimizer)["encoder_a"] == 0.0


def test_scheduler_warms_up_then_anneals_to_zero() -> None:
    cfg = tiny_config()
    cfg["train"].update({"epochs": 2, "warmup_epochs": 0.5})
    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = 10
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    # LambdaLR applies the schedule when it is constructed, so the group LR already
    # reflects step 0 here; compare against the configured base LR instead.
    base_lr = float(cfg["train"]["lr_convnext"])

    factors: list[float] = []
    for _ in range(int(cfg["train"]["epochs"]) * steps_per_epoch):
        optimizer.step()  # no gradients here; only silences the scheduler-order warning
        scheduler.step()
        factors.append(optimizer.param_groups[0]["lr"] / base_lr)

    # 0.5 warmup epochs over 10 steps per epoch = 5 warmup steps.
    assert factors[0] < factors[1] < factors[2] < factors[3], "warmup must raise the LR"
    assert max(factors) == pytest.approx(1.0, abs=1e-6), "the LR must reach its base value once"
    assert factors[-1] == pytest.approx(0.0, abs=1e-6), "cosine must end at (almost) zero"
    assert factors[6] > factors[10] > factors[-1], "the LR must fall monotonically after warmup"


def test_metrics_csv_header_is_written_once_and_rows_append(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    row = {column: 0 for column in METRICS_COLUMNS}
    row["epoch"] = 0
    append_metrics_row(path, row)
    row["epoch"] = 1
    append_metrics_row(path, row)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == list(METRICS_COLUMNS)
    assert len(lines) == 3


def test_metrics_csv_rejects_missing_columns(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="missing columns"):
        append_metrics_row(tmp_path / "metrics.csv", {"epoch": 0})


def test_metrics_csv_rejects_an_incompatible_header(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("epoch,val_dice\n0,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible header"):
        append_metrics_row(path, {column: 0 for column in METRICS_COLUMNS})


def test_run_info_records_provenance(tmp_path: Path) -> None:
    cfg = tiny_config()
    cfg["data"]["synthetic"] = True
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"synthetic-cache-bytes")
    cfg["data"]["cache_path"] = str(cache)
    model = build_model(cfg)

    info = write_run_info(
        tmp_path / "run_info.json", cfg, model, synthetic=True, config_file="configs/exp001.yaml"
    )

    on_disk = json.loads((tmp_path / "run_info.json").read_text(encoding="utf-8"))
    assert on_disk == info
    assert info["synth_data"] is True
    assert info["image_size"] == 256
    assert info["seed"] == 42
    assert info["model"]["encoder_a"] == "convnext_tiny.fb_in1k"
    assert info["model"]["pretrained_a"] is False
    assert info["params_total"] > 0
    assert info["params_trainable"] == info["params_total"]
    assert "git_commit" in info and "git_dirty" in info
    assert info["data_root"] == "data/raw"
    assert info["data_format"]["cache_path"] == str(cache)
    assert info["data_format"]["cache_sha256"] == hashlib.sha256(b"synthetic-cache-bytes").hexdigest()


def _train_step(model: nn.Module, batch: Mapping[str, Any], loss_fn: nn.Module, optimizer: Any, scaler: Any) -> float:
    """Run exactly one optimisation step and return the pre-step loss.

    Args:
        model: Model.
        batch: Fixed batch dict.
        loss_fn: Loss module.
        optimizer: Optimiser.
        scaler: GradScaler.

    Returns:
        Loss value at this step, as a float.
    """
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch["image"])
    loss = loss_fn(logits, batch["mask"])
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())


def test_resume_from_last_pt_gives_the_same_next_loss(tmp_path: Path) -> None:
    """Save after one step, reload, continue: the next loss must match exactly."""
    cfg = tiny_config()
    batch = make_batch()
    loss_fn = build_loss(cfg)
    checker = make_grad_scaler(False)

    # Reference: two consecutive steps in one clean process state.
    set_seed(0)
    reference_model = build_model(cfg)
    reference_optimizer = build_optimizer(reference_model, cfg)
    reference_scheduler = build_scheduler(reference_optimizer, cfg, steps_per_epoch=1)
    _train_step(reference_model, batch, loss_fn, reference_optimizer, checker)
    reference_scheduler.step()
    expected_next_loss = _train_step(reference_model, batch, loss_fn, reference_optimizer, checker)

    # Interrupted run: one step, checkpoint, then a brand new process state.
    set_seed(0)
    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=1)
    first_loss = _train_step(model, batch, loss_fn, optimizer, checker)
    scheduler.step()
    path = save_checkpoint(
        tmp_path / "last.pt", model, optimizer, scheduler, checker, epoch=0, best_val_dice=first_loss, cfg=cfg
    )

    set_seed(999)  # deliberately different RNG state than the saved run
    resumed_model = build_model(cfg)
    resumed_optimizer = build_optimizer(resumed_model, cfg)
    resumed_scheduler = build_scheduler(resumed_optimizer, cfg, steps_per_epoch=1)
    checkpoint = load_checkpoint(path, resumed_model, resumed_optimizer, resumed_scheduler, checker)

    assert checkpoint["epoch"] == 0
    assert resumed_scheduler.last_epoch == 1
    resumed_loss = _train_step(resumed_model, batch, loss_fn, resumed_optimizer, checker)

    assert resumed_loss == pytest.approx(expected_next_loss, rel=0, abs=1e-6)


def test_accumulation_flushes_the_trailing_partial_window() -> None:
    """Every batch must reach the optimizer, even when the epoch ends mid-window."""

    class ToySegmenter(nn.Module):
        """Minimal stand-in: 1x1 conv from 3 channels to one logit channel."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 1, kernel_size=1)

        def forward(self, x: Tensor) -> Tensor:
            return self.conv(x)

    cfg = tiny_config()
    cfg["train"].update({"grad_accum_steps": 3, "amp": False, "grad_clip_norm": 0.0})
    model = ToySegmenter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=2)
    batches = [make_batch(size=8) for _ in range(4)]

    stats = train_one_epoch(
        model, batches, build_loss(cfg), optimizer, scheduler, make_grad_scaler(False), cfg, torch.device("cpu")
    )

    assert stats["steps"] == 2, "4 micro-batches with accumulation 3 must give one full and one flushed step"
    assert scheduler.last_epoch == 2


def test_reseed_train_augmentation_changes_samples_between_epochs(synthetic_dataset: dict) -> None:
    pytest.importorskip("albumentations")
    dataset = PolypDataset(
        split_file=synthetic_dataset["split_file"],
        root=synthetic_dataset["root"],
        transform=build_train_transform(64, seed=0),
        image_size=64,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    reseed_train_augmentation(loader, 3)
    first = dataset[0]["image"]
    reseed_train_augmentation(loader, 3)
    same_epoch = dataset[0]["image"]
    reseed_train_augmentation(loader, 4)
    next_epoch = dataset[0]["image"]

    assert torch.equal(first, same_epoch), "the same epoch seed must reproduce the same sample"
    assert not torch.equal(first, next_epoch), "a new epoch seed must change the augmentation"


def test_build_dataloaders_uses_the_configured_cache(tmp_path: Path) -> None:
    """A non-empty data.cache_path must reach PolypDataset, for both splits."""
    from scripts.build_cache import build_cache

    root = synthesize_smoke_data(tmp_path / "data", image_size=64, n_train=2, n_val=2, seed=0)
    cache = tmp_path / "cache.npz"
    build_cache(root, cache, size=64, image_ext=".png", mask_ext=".png", progress=False)

    cfg = mini_run_config(root, tmp_path / "runs", epochs=1)
    cfg["data"]["cache_path"] = str(cache)

    train_loader, val_loader = build_dataloaders(cfg)

    assert train_loader.dataset.uses_cache
    assert val_loader.dataset.uses_cache
    assert train_loader.dataset[0]["image"].shape == (3, 64, 64)
    assert set(torch.unique(train_loader.dataset[0]["mask"]).tolist()) <= {0.0, 1.0}


def test_smoke_run_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)
    out_root = tmp_path / "colab_runs"

    exit_code = train_mod.main(
        ["--config", str(BASELINE_CONFIG), "--smoke", "--out-dir", str(out_root)]
    )

    assert exit_code == 0
    run_dir = out_root / "exp001_convnext_base_baseline_smoke"
    for name in ("config_resolved.yaml", "run_info.json", "metrics.csv", "last.pt", "best.pt"):
        assert (run_dir / name).is_file(), f"missing {name}"
    rows = (run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].split(",") == list(METRICS_COLUMNS)
    assert "peak_gpu_mem_mb" in rows[0].split(",")
    assert len(rows) == 2, "one epoch must write exactly one row"

    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    assert info["synth_data"] is True
    assert info["model"]["fusion"] == ["single_a"] * 4
    assert info["model"]["msu"] is False
    assert info["model"]["encoder"] == "convnext"
    assert info["peak_gpu_mem_mb"] == 0.0
    assert info["epochs_this_session"] == 1
    assert info["mean_epoch_time_s"] is not None and info["mean_epoch_time_s"] > 0
    assert info["data_format"]["cache_sha256"] is None
    assert (run_dir / "preds" / "epoch_0000.png").is_file()
    assert "SMOKE MODE" in capsys.readouterr().out


def test_caformer_smoke_run_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Branch B smoke run exercises the Meta-Polyp encoder without downloads."""
    monkeypatch.chdir(tmp_path)
    out_root = tmp_path / "runs"

    exit_code = train_mod.main(
        ["--config", str(CAFORMER_CONFIG), "--smoke", "--out-dir", str(out_root)]
    )

    assert exit_code == 0
    run_dir = out_root / "exp003_caformer_s18_baseline_smoke"
    for name in ("config_resolved.yaml", "run_info.json", "metrics.csv", "last.pt", "best.pt"):
        assert (run_dir / name).is_file(), f"missing {name}"

    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    assert info["model"]["encoder"] == "caformer"
    assert info["model"]["encoder_a"] is None
    assert info["model"]["encoder_b"] == "caformer_s18.sail_in1k"
    assert info["model"]["pretrained_b"] is False
    assert info["model"]["fusion"] == ["single_b"] * 4
    assert info["model"]["msu"] is False
    assert "caformer" in capsys.readouterr().out


def test_real_run_resumes_from_last_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    root = synthesize_smoke_data(tmp_path / "data", image_size=64, n_train=4, n_val=2, seed=1)
    out_dir = tmp_path / "runs"
    config_path = save_config(mini_run_config(root, out_dir, epochs=1), tmp_path / "mini1.yaml")

    assert train_mod.main(["--config", str(config_path)]) == 0
    run_dir = out_dir / "mini"
    rows_after_first = (run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows_after_first) == 2  # header + epoch 0
    assert "resumed from" not in capsys.readouterr().out

    # Second launch with one more epoch: must pick up last.pt, not restart.
    config_path = save_config(mini_run_config(root, out_dir, epochs=2), tmp_path / "mini2.yaml")
    assert train_mod.main(["--config", str(config_path)]) == 0

    rows_after_resume = (run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows_after_resume) == 3, "resume must append only the new epoch"
    assert rows_after_resume[0] == rows_after_first[0], "the existing header must not be rewritten"
    assert rows_after_resume[2].startswith("1,"), "the resumed run must continue at epoch 1"
    assert "resumed from" in capsys.readouterr().out

    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    assert info["synth_data"] is True
    assert yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))["train"]["epochs"] == 2
