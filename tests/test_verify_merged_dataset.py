"""Tests for verify_merged_dataset.py using a synthetic merged-layout fixture."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "verify_merged_dataset.py"


@pytest.fixture()
def merged_fixture(tmp_path: Path) -> dict[str, Path]:
    """Create a synthetic merged dataset layout matching the new Drive structure.

    Returns dict with keys: data_root, report_path, splits_dir, train_dir, etc.
    """
    data_root = tmp_path / "polyp-seg-data"
    data_root.mkdir()

    # Create train/val/test with images/masks
    splits = ["train", "val", "test"]
    counts = {"train": 6, "val": 2, "test": 2}  # small for tests
    all_ids = []
    global_idx = 0

    for split in splits:
        img_dir = data_root / split / "images"
        msk_dir = data_root / split / "masks"
        img_dir.mkdir(parents=True)
        msk_dir.mkdir(parents=True)

        split_ids = []
        for i in range(counts[split]):
            # Use global index to ensure unique IDs across splits
            # Alternate between Kvasir and ClinicDB prefixes
            if global_idx % 2 == 0:
                sample_id = f"KvasirSEG__cju{global_idx:03d}"
            else:
                sample_id = f"CVCClinicDB__{global_idx:03d}"
            split_ids.append(sample_id)
            all_ids.append((split, sample_id))
            global_idx += 1

            # Create synthetic image (32x32 RGB)
            rng = np.random.default_rng(hash(sample_id) % 2**32)
            image = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(image, mode="RGB").save(img_dir / f"{sample_id}.png")

            # Create synthetic mask with some foreground
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:24, 8:24] = 255
            Image.fromarray(mask, mode="L").save(msk_dir / f"{sample_id}.png")

    # Create split CSVs
    splits_dir = data_root / "splits"
    splits_dir.mkdir()
    for split in splits:
        csv_path = splits_dir / f"{split}.csv"
        split_ids = [sid for sp, sid in all_ids if sp == split]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split",
                    "dataset",
                    "source_stem",
                    "image",
                    "mask",
                    "source_image_name",
                    "source_mask_name",
                ],
            )
            writer.writeheader()
            for sample_id in split_ids:
                source = "Kvasir-SEG" if sample_id.startswith("KvasirSEG") else "CVC-ClinicDB"
                writer.writerow(
                    {
                        "split": split,
                        "dataset": source,
                        "source_stem": sample_id,
                        "image": f"{split}/images/{sample_id}.png",
                        "mask": f"{split}/masks/{sample_id}.png",
                        "source_image_name": f"{sample_id}.png",
                        "source_mask_name": f"{sample_id}.png",
                    }
                )

    # Create preparation_report.json
    kvasir_count = sum(1 for _, sid in all_ids if sid.startswith("KvasirSEG"))
    clinic_count = sum(1 for _, sid in all_ids if sid.startswith("CVCClinicDB"))
    report = {
        "total": len(all_ids),
        "splits": {split: counts[split] for split in splits},
        "source_distribution": {
            split: {
                "Kvasir-SEG": sum(1 for sp, sid in all_ids if sp == split and sid.startswith("KvasirSEG")),
                "CVC-ClinicDB": sum(1 for sp, sid in all_ids if sp == split and sid.startswith("CVCClinicDB")),
            }
            for split in splits
        },
        "protocol": "Meta-Polyp paper Section IV-A: merged Kvasir-SEG + CVC-ClinicDB, 60/20/20",
        "split_seed": 42,
    }
    report_path = data_root / "preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "data_root": data_root,
        "report_path": report_path,
        "splits_dir": splits_dir,
        "total_ids": len(all_ids),
        "kvasir_count": kvasir_count,
        "clinic_count": clinic_count,
    }


def test_verify_merged_dataset_passes(merged_fixture: dict, tmp_path: Path) -> None:
    """The verification script should PASS on a valid synthetic layout."""
    out_splits = tmp_path / "out_splits"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--data-root",
        str(merged_fixture["data_root"]),
        "--write-splits",
        str(out_splits),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    assert result.returncode == 0, f"Verification failed: {result.stderr}"
    assert "ALL CHECKS PASSED" in result.stdout

    # Check that split files were written
    for split in ["train", "val", "test"]:
        split_file = out_splits / f"{split}.txt"
        assert split_file.is_file()
        ids = split_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(ids) > 0


def test_verify_merged_dataset_fails_on_missing_mask(merged_fixture: dict, tmp_path: Path) -> None:
    """The verification script should FAIL when a mask is missing."""
    # Remove one mask
    mask_to_remove = merged_fixture["data_root"] / "train" / "masks" / "KvasirSEG__cju000.png"
    mask_to_remove.unlink()

    cmd = [sys.executable, str(SCRIPT), "--data-root", str(merged_fixture["data_root"])]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "FAIL" in result.stdout or "missing mask" in result.stdout.lower()


def test_verify_merged_dataset_fails_on_duplicate_id(merged_fixture: dict, tmp_path: Path) -> None:
    """The verification script should FAIL when an ID appears in two splits."""
    # Copy a train ID to val CSV
    val_csv = merged_fixture["data_root"] / "splits" / "val.csv"
    rows = []
    with val_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Duplicate first train ID into val
    train_csv = merged_fixture["data_root"] / "splits" / "train.csv"
    with train_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        train_rows = list(reader)
    dup_row = train_rows[0].copy()
    dup_row["split"] = "val"
    rows.append(dup_row)
    with val_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Also need to copy the image/mask files to val folders
    dup_id = dup_row["source_stem"]
    for subdir in ["images", "masks"]:
        src = merged_fixture["data_root"] / "train" / subdir / f"{dup_id}.png"
        dst = merged_fixture["data_root"] / "val" / subdir / f"{dup_id}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(src, dst)

    cmd = [sys.executable, str(SCRIPT), "--data-root", str(merged_fixture["data_root"])]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "FAIL" in result.stdout
    assert "appears in both" in result.stdout


def test_verify_merged_dataset_counts_match_report(merged_fixture: dict) -> None:
    """Verify that the fixture counts match the preparation report."""
    # This is more of a sanity check on the fixture itself
    report = json.loads(merged_fixture["report_path"].read_text(encoding="utf-8"))
    assert report["total"] == merged_fixture["total_ids"]
    assert report["splits"]["train"] == 6
    assert report["splits"]["val"] == 2
    assert report["splits"]["test"] == 2
    assert report["source_distribution"]["train"]["Kvasir-SEG"] + report["source_distribution"]["train"]["CVC-ClinicDB"] == 6


def test_notebook_json_valid() -> None:
    """The merged Colab notebook should be valid JSON and have the expected cells."""
    nb_path = REPO_ROOT / "notebooks" / "colab_run_merged.ipynb"
    assert nb_path.is_file()
    import json
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] == 4
    assert "cells" in nb
    # Check for key cells: config, setup, data, train, results
    cell_types = [c["cell_type"] for c in nb["cells"]]
    assert cell_types.count("code") >= 6  # config, setup, packages, data, cache, sanity, train, results, eval, aggregate
    # Check config cell has the right variables
    config_cell = None
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and "CONFIG =" in "".join(cell["source"]):
            config_cell = cell
            break
    assert config_cell is not None, "Config cell not found"
    source = "".join(config_cell["source"])
    assert "exp006_merged_all_concat.yaml" in source
    assert "DATA_DRIVE" in source
    assert "RUNS_DIR" in source


def test_notebook_code_cells_compile() -> None:
    """Every code cell in the notebook should compile as valid Python."""
    nb_path = REPO_ROOT / "notebooks" / "colab_run_merged.ipynb"
    import json
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            try:
                compile(source, f"<notebook_cell_{i}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Cell {i} has syntax error: {e}\nSource:\n{source}")


def test_config_loads_and_matches_exp006() -> None:
    """The merged config should load through the real loader and differ only in expected keys."""
    from src.utils import load_config
    import yaml

    REPO_ROOT = Path(__file__).resolve().parents[1]
    
    # Load raw YAML to check file-level differences (before registry resolution)
    with open(REPO_ROOT / "configs" / "exp006_merged_all_concat.yaml") as f:
        raw_merged = yaml.safe_load(f)
    with open(REPO_ROOT / "configs" / "exp006_all_concat.yaml") as f:
        raw_base = yaml.safe_load(f)

    # Check the keys that MUST differ at file level
    assert raw_merged["run"]["name"] == "exp006_merged_all_concat"
    assert raw_base["run"]["name"] == "exp006_all_concat"
    assert raw_merged["train"]["epochs"] == 30
    assert raw_base["train"]["epochs"] == 100

    # Data location keys differ (placeholders for Colab override)
    assert raw_merged["data"]["root"] == "data/merged_raw"
    assert raw_base["data"]["root"] == "data/raw"
    assert raw_merged["data"]["split_dir"] == "data/merged_splits"
    assert raw_base["data"]["split_dir"] == "data/splits"
    assert raw_merged["data"]["image_ext"] == ".png"
    assert raw_base["data"]["image_ext"] == ".jpg"
    assert raw_merged["data"]["mask_ext"] == ".png"
    assert raw_base["data"]["mask_ext"] == ".jpg"
    assert raw_merged["data"]["cache_path"] == "data/cache/merged_256.npz"
    assert raw_base["data"]["cache_path"] == "data/cache/kvasir_seg_256.npz"
    assert raw_merged["data"]["split_ratios"] == [0.6, 0.2, 0.2]
    assert raw_base["data"]["split_ratios"] == [0.8, 0.1, 0.1]

    # Also test that both resolve without error through the real loader
    cfg_merged = load_config(REPO_ROOT / "configs" / "exp006_merged_all_concat.yaml")
    cfg_base = load_config(REPO_ROOT / "configs" / "exp006_all_concat.yaml")
    
    # After resolution, run.name and train.epochs should still differ
    assert cfg_merged["run"]["name"] == "exp006_merged_all_concat"
    assert cfg_base["run"]["name"] == "exp006_all_concat"
    assert cfg_merged["train"]["epochs"] == 30
    assert cfg_base["train"]["epochs"] == 100

    # Model, train (except epochs), decoder, logging should be identical after resolution
    for section in ["model", "decoder", "logging"]:
        assert cfg_merged[section] == cfg_base[section], f"Section {section} differs after resolution"
    
    # Train section should match except epochs
    train_merged = {k: v for k, v in cfg_merged["train"].items() if k != "epochs"}
    train_base = {k: v for k, v in cfg_base["train"].items() if k != "epochs"}
    assert train_merged == train_base, "Train section differs beyond epochs"


def test_train_cli_flags_match_notebook() -> None:
    """Check that the CLI flags used in the notebook match train.py argparse."""
    # The notebook uses: --config, --smoke, --out-dir, --no-resume, --max-steps
    import argparse
    from src.train import parse_args

    # Test that all flags used in notebook are accepted
    args = parse_args([
        "--config", "configs/exp006_merged_all_concat.yaml",
        "--smoke",
        "--out-dir", "/tmp/runs",
        "--no-resume",
        "--max-steps", "3",
    ])
    assert args.config.name == "exp006_merged_all_concat.yaml"
    assert args.smoke is True
    assert str(args.out_dir) == "/tmp/runs"
    assert args.no_resume is True
    assert args.max_steps == 3


def test_evaluate_cli_flags_match_notebook() -> None:
    """Check that evaluate CLI flags used in notebook match argparse."""
    from src.evaluate import parse_args

    args = parse_args([
        "--run_dir", "/tmp/runs/exp",
        "--split", "test",
        "--checkpoint", "best",
        "--results_csv", "/tmp/results.csv",
    ])
    assert str(args.run_dir) == "/tmp/runs/exp"
    assert args.split == "test"
    assert args.checkpoint == "best"
    assert str(args.results_csv) == "/tmp/results.csv"
    assert args.force is False


def test_aggregate_cli_flags_match_notebook() -> None:
    """Check that aggregate_results CLI flags match argparse."""
    from scripts.aggregate_results import parse_args

    args = parse_args([
        "--runs-dir", "/tmp/runs",
        "--out-dir", "/tmp/runs",
    ])
    assert str(args.runs_dir) == "/tmp/runs"
    assert str(args.out_dir) == "/tmp/runs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])