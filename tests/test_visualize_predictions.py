"""Tests for scripts/visualize_predictions.py (synthetic data smoke test)."""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.net import build_model
from src.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "visualize_predictions.py"


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a synthetic run directory with a tiny model and fake data."""
    tmp = tmp_path_factory.mktemp("synthetic_run")

    # Create a minimal config based on exp001 but tiny
    cfg = load_config("configs/exp001_convnext_base_baseline.yaml")
    cfg["model"]["encoder_a"] = "convnext_tiny.fb_in1k"
    cfg["model"]["pretrained_a"] = False
    cfg["data"]["image_size"] = 64
    cfg["data"]["cache_path"] = ""
    cfg["data"]["synthetic"] = True
    cfg["run"]["name"] = "synthetic_test"
    cfg["train"]["batch_size"] = 2

    # Save config
    config_path = tmp / "config_resolved.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # Build and save a tiny model checkpoint
    model = build_model(cfg)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": {},
        "scheduler": {},
        "scaler": {},
        "epoch": 0,
        "best_val_dice": 0.5,
        "rng_state": {},
    }
    buf = io.BytesIO()
    torch.save(checkpoint, buf)
    (tmp / "best.pt").write_bytes(buf.getvalue())

    # Create run_info.json
    run_info = {
        "git_commit": "test",
        "config_file": "configs/exp001_convnext_base_baseline.yaml",
        "params_total": sum(p.numel() for p in model.parameters()),
        "mean_epoch_time_s": 1.0,
        "peak_gpu_mem_mb": 0.0,
    }
    (tmp / "run_info.json").write_text(json.dumps(run_info), encoding="utf-8")

    # Create synthetic dataset structure for kvasir_seg
    images_dir = tmp / "images"
    masks_dir = tmp / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    # Create 8 synthetic images and masks
    rng = np.random.default_rng(42)
    ids = []
    for i in range(8):
        img_id = f"syn_{i:03d}"
        ids.append(img_id)
        # Random RGB image
        img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(img, mode="RGB").save(images_dir / f"{img_id}.jpg")
        # Random mask with a blob
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        Image.fromarray(mask, mode="L").save(masks_dir / f"{img_id}.jpg")

    # Create split files
    splits_dir = tmp / "splits"
    splits_dir.mkdir()
    (splits_dir / "train.txt").write_text("\n".join(ids[:4]) + "\n", encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(ids[4:6]) + "\n", encoding="utf-8")
    (splits_dir / "test.txt").write_text("\n".join(ids[6:]) + "\n", encoding="utf-8")

    # Update config paths to point to synthetic data
    cfg["data"]["root"] = str(tmp)
    cfg["data"]["split_dir"] = str(splits_dir)
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # Create fake per_image_test_kvasir_seg.csv (as if evaluation ran)
    per_image_csv = tmp / "per_image_test_kvasir_seg.csv"
    with per_image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "dice", "iou", "mae", "lesion_pixels", "image_area", "small_lesion"])
        for img_id in ids[6:]:  # test split
            writer.writerow([img_id, "0.8", "0.7", "0.1", "1024", "4096", "0"])

    return tmp


def test_visualize_predictions_smoke(synthetic_run: Path) -> None:
    """Smoke test: run visualize_predictions.py on synthetic data, CPU only."""
    out_dir = synthetic_run / "figures_out"

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--run_dir", str(synthetic_run),
        "--datasets", "kvasir_seg",
        "--out_dir", str(out_dir),
        "--n_random", "2",
        "--n_failures", "1",
        "--seed", "42",
        "--split", "test",
        "--no_cuda",
    ]

    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)

    # Print output for debugging
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)

    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Check output files exist
    assert (out_dir / "kvasir_seg" / "kvasir_seg_predictions.png").is_file()
    assert (out_dir / "kvasir_seg" / "kvasir_seg_predictions.pdf").is_file()
    assert (out_dir / "combined_predictions.png").is_file()
    assert (out_dir / "combined_predictions.pdf").is_file()
    assert (out_dir / "kvasir_seg" / "kvasir_seg_selection.csv").is_file()

    # Check selection CSV has correct columns
    with (out_dir / "kvasir_seg" / "kvasir_seg_selection.csv").open() as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[:3] == ["dataset", "image_id", "selection_reason"]
        rows = list(reader)
        # Only 2 test images available, so 2 random selected (no failures possible)
        assert len(rows) == 2


def test_visualize_predictions_consistency_check(synthetic_run: Path) -> None:
    """Test that consistency check passes (recomputed Dice matches CSV)."""
    out_dir = synthetic_run / "figures_out2"

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--run_dir", str(synthetic_run),
        "--datasets", "kvasir_seg",
        "--out_dir", str(out_dir),
        "--n_random", "2",
        "--n_failures", "1",
        "--seed", "42",
        "--split", "test",
        "--no_cuda",
    ]

    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)

    assert result.returncode == 0
    assert "Consistency check: PASS" in result.stdout


def test_visualize_predictions_multi_run(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Test side-by-side comparison with two runs."""
    tmp = tmp_path_factory.mktemp("multi_run")

    # Create two synthetic runs with different configs
    for run_name, encoder_name in [("run_a", "convnext_tiny.fb_in1k"), ("run_b", "convnext_tiny.fb_in1k")]:
        run_dir = tmp / run_name
        run_dir.mkdir()

        cfg = load_config("configs/exp001_convnext_base_baseline.yaml")
        cfg["model"]["encoder_a"] = encoder_name
        cfg["model"]["pretrained_a"] = False
        cfg["data"]["image_size"] = 64
        cfg["data"]["cache_path"] = ""
        cfg["data"]["synthetic"] = True
        cfg["run"]["name"] = run_name
        cfg["train"]["batch_size"] = 2

        import yaml
        (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

        model = build_model(cfg)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": {},
            "scheduler": {},
            "scaler": {},
            "epoch": 0,
            "best_val_dice": 0.5,
            "rng_state": {},
        }
        buf = io.BytesIO()
        torch.save(checkpoint, buf)
        (run_dir / "best.pt").write_bytes(buf.getvalue())

        run_info = {"git_commit": "test", "params_total": sum(p.numel() for p in model.parameters())}
        (run_dir / "run_info.json").write_text(json.dumps(run_info), encoding="utf-8")

    # Shared synthetic data
    images_dir = tmp / "images"
    masks_dir = tmp / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    rng = np.random.default_rng(42)
    ids = []
    for i in range(8):
        img_id = f"syn_{i:03d}"
        ids.append(img_id)
        img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(img, mode="RGB").save(images_dir / f"{img_id}.jpg")
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        Image.fromarray(mask, mode="L").save(masks_dir / f"{img_id}.jpg")

    splits_dir = tmp / "splits"
    splits_dir.mkdir()
    (splits_dir / "train.txt").write_text("\n".join(ids[:4]) + "\n", encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(ids[4:6]) + "\n", encoding="utf-8")
    (splits_dir / "test.txt").write_text("\n".join(ids[6:]) + "\n", encoding="utf-8")

    # Update configs
    for run_name in ["run_a", "run_b"]:
        run_dir = tmp / run_name
        cfg = load_config(run_dir / "config_resolved.yaml")
        cfg["data"]["root"] = str(tmp)
        cfg["data"]["split_dir"] = str(splits_dir)
        (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

        per_image_csv = run_dir / "per_image_test_kvasir_seg.csv"
        with per_image_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "dice", "iou", "mae", "lesion_pixels", "image_area", "small_lesion"])
            for img_id in ids[6:]:
                writer.writerow([img_id, "0.8", "0.7", "0.1", "1024", "4096", "0"])

    out_dir = tmp / "figures_out"

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--run_dir", str(tmp / "run_a"),
        "--run_dir", str(tmp / "run_b"),
        "--datasets", "kvasir_seg",
        "--out_dir", str(out_dir),
        "--n_random", "1",
        "--n_failures", "1",
        "--seed", "42",
        "--split", "test",
        "--no_cuda",
    ]

    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0

    # Check both run predictions appear in output
    assert (out_dir / "kvasir_seg" / "kvasir_seg_predictions.png").is_file()
    assert (out_dir / "combined_predictions.png").is_file()

    # Selection CSV should have columns for both runs
    with (out_dir / "kvasir_seg" / "kvasir_seg_selection.csv").open() as f:
        reader = csv.reader(f)
        header = next(reader)
        assert "dice_run_a" in header
        assert "dice_run_b" in header