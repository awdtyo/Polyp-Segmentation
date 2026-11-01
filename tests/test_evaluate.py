"""Tests for src/evaluate.py: files, results row, duplicates and test safety.

Every test uses a real (tiny) smoke run on synthetic data at 64 px: the smoke
config is the only training this repo allows on the CPU box. Nothing downloads
pretrained weights.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src import evaluate as evaluate_mod
from src import train as train_mod
from src.data_registry import get_spec, resolve_data_root
from src.losses import build_loss
from src.metrics import SegMetrics
from src.models.net import build_model
from src.train import evaluate as train_evaluate
from src.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = REPO_ROOT / "configs" / "exp001_convnext_base_baseline.yaml"
RUN_NAME = "exp001_convnext_base_baseline_smoke"
TRAIN_DATASET = "kvasir_seg"


def _train_smoke_run(out_root: Path) -> Path:
    """Train one smoke run and return its run folder."""
    code = train_mod.main(["--config", str(BASELINE_CONFIG), "--smoke", "--out-dir", str(out_root)])
    assert code == 0
    return out_root / RUN_NAME


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A smoke run whose synthetic test id also appears in val (leaky on purpose)."""
    return _train_smoke_run(tmp_path_factory.mktemp("eval_runs"))


@pytest.fixture(scope="module")
def isolated_test_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A smoke run with the test id removed from val.txt, so test is isolated."""
    run_dir = _train_smoke_run(tmp_path_factory.mktemp("eval_runs_isolated"))
    splits = run_dir / "_smoke_data" / "splits"
    test_ids = {line.strip() for line in (splits / "test.txt").read_text().splitlines() if line.strip()}
    val_ids = [line for line in (splits / "val.txt").read_text().splitlines() if line.strip()]
    kept = [sample_id for sample_id in val_ids if sample_id not in test_ids]
    assert kept, "the synthetic val split must keep at least one non-test id"
    (splits / "val.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")
    return run_dir


def _read_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_processed_dataset(root: Path, name: str, size: int, n: int = 3) -> Path:
    """Create `<root>/processed/<name>/{data.npz,meta.json}` like the prep script."""
    processed = root / "processed" / name
    processed.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ids = np.asarray([f"{name}_{index:03d}" for index in range(n)], dtype=np.str_)
    images = rng.integers(0, 256, size=(n, size, size, 3), dtype=np.uint8)
    masks = np.zeros((n, size, size), dtype=np.uint8)
    masks[:, size // 4 : size // 2, size // 4 : size // 2] = 1
    np.savez_compressed(processed / "data.npz", ids=ids, images=images, masks=masks)
    (processed / "meta.json").write_text(
        json.dumps({"name": name, "image_size": size}), encoding="utf-8"
    )
    return processed


def test_summarize_rows_matches_hand_computed_values() -> None:
    """Mean, sample std and the small-lesion split must match hand arithmetic."""
    rows = [
        {"id": "a", "dice": 0.5, "iou": 0.25, "mae": 0.1, "lesion_pixels": 10.0, "small_lesion": True},
        {"id": "b", "dice": 1.0, "iou": 0.75, "mae": 0.2, "lesion_pixels": 100.0, "small_lesion": False},
        {"id": "c", "dice": 0.0, "iou": 0.0, "mae": 0.6, "lesion_pixels": 200.0, "small_lesion": False},
    ]
    summary = evaluate_mod.summarize_rows(rows)
    assert summary["n"] == 3
    assert summary["dice_mean"] == pytest.approx(0.5)
    assert summary["iou_mean"] == pytest.approx(1.0 / 3.0)
    assert summary["dice_std"] == pytest.approx(0.5)  # sample std of 0.5, 1.0, 0.0
    assert summary["mae_mean"] == pytest.approx(0.3)
    # Only one image qualifies as small (< 10), so small-lesion Dice is NaN.
    assert math.isnan(summary["small_lesion_dice"])
    assert summary["small_lesion_n"] == 1


def test_summarize_rows_small_lesion_returns_mean_when_enough_qualify() -> None:
    """With >= 10 small lesions, small_lesion_dice is the mean over them."""
    rows = []
    for index in range(12):
        rows.append(
            {
                "id": f"s{index}",
                "dice": 0.5 + 0.01 * index,
                "iou": 0.3,
                "mae": 0.2,
                "lesion_pixels": 10.0,
                "small_lesion": True,
            }
        )
    for index in range(3):
        rows.append(
            {
                "id": f"l{index}",
                "dice": 0.9,
                "iou": 0.8,
                "mae": 0.1,
                "lesion_pixels": 500.0,
                "small_lesion": False,
            }
        )
    summary = evaluate_mod.summarize_rows(rows)
    assert summary["small_lesion_n"] == 12
    expected = sum(0.5 + 0.01 * i for i in range(12)) / 12.0
    assert summary["small_lesion_dice"] == pytest.approx(expected, abs=1e-6)


def test_output_names_add_the_tta_suffix() -> None:
    """The TTA run is saved under a different name so it never overwrites a plain one."""
    assert evaluate_mod.output_names("test", "kvasir_seg", tta_flip=False) == (
        "eval_test_kvasir_seg.json",
        "per_image_test_kvasir_seg.csv",
    )
    assert evaluate_mod.output_names("val", "cvc_colondb", tta_flip=True) == (
        "eval_val_cvc_colondb_ttaflip.json",
        "per_image_val_cvc_colondb_ttaflip.csv",
    )


def test_evaluate_writes_files_and_one_row(smoke_run: Path, tmp_path: Path) -> None:
    """A val evaluation writes eval_val_kvasir_seg.json, the per-image CSV and one row."""
    results = tmp_path / "results.csv"
    code = evaluate_mod.main(
        ["--run_dir", str(smoke_run), "--split", "val", "--results_csv", str(results)]
    )
    assert code == evaluate_mod.EXIT_OK

    json_path = smoke_run / f"eval_val_{TRAIN_DATASET}.json"
    per_image_path = smoke_run / f"per_image_val_{TRAIN_DATASET}.csv"
    assert json_path.is_file()
    assert per_image_path.is_file()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["split"] == "val"
    assert payload["dataset"] == TRAIN_DATASET
    assert payload["tta"] == ""
    assert payload["n"] == 2
    assert payload["checkpoint"] == "best"
    assert payload["seed"] == 42
    assert payload["best_epoch"] == 1
    assert payload["epochs_trained"] == 1
    assert payload["params_total"] > 0
    assert payload["params_m"] == pytest.approx(payload["params_total"] / 1e6, rel=1e-6)
    assert payload["threshold"] == 0.5
    assert 0.0 <= payload["mae_mean"] <= 1.0

    with per_image_path.open(newline="", encoding="utf-8") as handle:
        per_image = list(csv.DictReader(handle))
    assert len(per_image) == 2
    assert set(per_image[0]) == {
        "id",
        "dice",
        "iou",
        "mae",
        "lesion_pixels",
        "image_area",
        "small_lesion",
    }
    assert payload["dice_mean"] == pytest.approx(
        sum(float(row["dice"]) for row in per_image) / len(per_image), abs=1e-6
    )
    assert payload["mae_mean"] == pytest.approx(
        sum(float(row["mae"]) for row in per_image) / len(per_image), abs=1e-6
    )

    rows = _read_results(results)
    assert len(rows) == 1
    assert list(rows[0]) == list(evaluate_mod.RESULTS_COLUMNS)
    row = rows[0]
    assert row["run_name"] == RUN_NAME
    assert row["split"] == "val"
    assert row["dataset"] == TRAIN_DATASET
    assert row["tta"] == ""
    assert row["config_name"] == RUN_NAME
    assert row["seed"] == "42"
    assert row["best_epoch"] == "1"
    assert row["epochs_trained"] == "1"
    assert row["small_lesion_n"] == "0"
    assert float(row["test_dice"]) == pytest.approx(payload["dice_mean"], abs=1e-6)
    assert float(row["test_iou"]) == pytest.approx(payload["iou_mean"], abs=1e-6)
    assert float(row["test_mae"]) == pytest.approx(payload["mae_mean"], abs=1e-6)


def test_evaluate_metrics_match_the_training_function(smoke_run: Path, tmp_path: Path) -> None:
    """The evaluation numbers equal src.train.evaluate on the same weights/loader."""
    results = tmp_path / "results.csv"
    code = evaluate_mod.main(
        ["--run_dir", str(smoke_run), "--split", "val", "--results_csv", str(results)]
    )
    assert code == evaluate_mod.EXIT_OK
    payload = json.loads((smoke_run / f"eval_val_{TRAIN_DATASET}.json").read_text(encoding="utf-8"))

    cfg = load_config(smoke_run / "config_resolved.yaml")
    cfg["model"] = {**cfg["model"], "pretrained_a": False, "pretrained_b": False}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = torch.load(smoke_run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    loader = evaluate_mod.build_split_loader(cfg, "val")
    result = train_evaluate(model, loader, build_loss(cfg).to(device), device)

    assert result["dice"] == pytest.approx(payload["dice_mean"], abs=1e-6)
    assert result["iou"] == pytest.approx(payload["iou_mean"], abs=1e-6)
    assert result["n"] == payload["n"]

    # And the same numbers via SegMetrics directly, so the metric path is pinned.
    metrics = SegMetrics(from_logits=True)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            metrics.update(
                model(batch["image"].to(device)),
                batch["mask"].to(device),
                ids=list(batch["id"]),
            )
    summary = metrics.compute()
    assert summary["dice"] == pytest.approx(payload["dice_mean"], abs=1e-6)


def test_flip_tta_writes_suffixed_outputs_beside_the_plain_run(
    isolated_test_run: Path, tmp_path: Path
) -> None:
    """--tta-flip writes *_ttaflip files and a separate row, leaving the plain run intact."""
    results = tmp_path / "results.csv"
    base = ["--run_dir", str(isolated_test_run), "--split", "test", "--results_csv", str(results)]
    assert evaluate_mod.main(base) == evaluate_mod.EXIT_OK
    assert evaluate_mod.main([*base, "--tta-flip", "--force"]) == evaluate_mod.EXIT_OK

    plain = isolated_test_run / f"eval_test_{TRAIN_DATASET}.json"
    tta = isolated_test_run / f"eval_test_{TRAIN_DATASET}_ttaflip.json"
    assert plain.is_file() and tta.is_file()
    payload = json.loads(tta.read_text(encoding="utf-8"))
    assert payload["tta"] == "flip"
    assert 0.0 <= payload["dice_mean"] <= 1.0

    rows = _read_results(results)
    assert len(rows) == 2, "the plain and the TTA row must coexist"
    assert {row["tta"] for row in rows} == {"", "flip"}


def test_external_dataset_is_evaluated_zero_shot(
    isolated_test_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-training dataset is read from processed/ and recorded with its own name."""
    data_root = tmp_path / "data"
    cfg = load_config(isolated_test_run / "config_resolved.yaml")
    size = int(cfg["data"]["image_size"])
    _write_processed_dataset(data_root, "cvc_clinicdb", size, n=3)
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    results = tmp_path / "results.csv"
    code = evaluate_mod.main(
        [
            "--run_dir",
            str(isolated_test_run),
            "--split",
            "test",
            "--dataset",
            "cvc_clinicdb",
            "--results_csv",
            str(results),
        ]
    )
    assert code == evaluate_mod.EXIT_OK

    json_path = isolated_test_run / "eval_test_cvc_clinicdb.json"
    per_image_path = isolated_test_run / "per_image_test_cvc_clinicdb.csv"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "cvc_clinicdb"
    assert payload["n"] == 3
    with per_image_path.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3

    rows = _read_results(results)
    assert len(rows) == 1
    assert rows[0]["dataset"] == "cvc_clinicdb"
    assert rows[0]["split"] == "test"


def test_missing_processed_dataset_fails_loudly(
    isolated_test_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A requested dataset that was never prepared is an error, not a silent skip."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "empty-data"))
    results = tmp_path / "results.csv"
    code = evaluate_mod.main(
        [
            "--run_dir",
            str(isolated_test_run),
            "--split",
            "test",
            "--dataset",
            "etis",
            "--results_csv",
            str(results),
        ]
    )
    assert code == evaluate_mod.EXIT_FAILURE
    assert "prepare_datasets.py" in capsys.readouterr().err
    assert not (isolated_test_run / "eval_test_etis.json").exists()
    assert not results.exists(), "a failed evaluation must not write a row"


def test_unknown_dataset_is_a_usage_error(isolated_test_run: Path, tmp_path: Path) -> None:
    """An unknown dataset name is rejected before anything is written."""
    results = tmp_path / "results.csv"
    code = evaluate_mod.main(
        ["--run_dir", str(isolated_test_run), "--dataset", "nope", "--results_csv", str(results)]
    )
    assert code == evaluate_mod.EXIT_USAGE
    assert not results.exists()


def test_duplicate_row_is_refused_then_force_replaces(smoke_run: Path, tmp_path: Path) -> None:
    """A second row for the same run and split is refused; --force replaces it."""
    results = tmp_path / "results.csv"
    args = ["--run_dir", str(smoke_run), "--split", "val", "--results_csv", str(results)]
    assert evaluate_mod.main(args) == evaluate_mod.EXIT_OK
    assert evaluate_mod.main(args) == evaluate_mod.EXIT_FAILURE
    assert len(_read_results(results)) == 1
    assert evaluate_mod.main([*args, "--force"]) == evaluate_mod.EXIT_OK
    rows = _read_results(results)
    assert len(rows) == 1, "--force must replace the row, not append a second one"


def test_test_split_overlap_is_detected(smoke_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A test id that also appears in val must abort the evaluation."""
    results = tmp_path / "results.csv"
    code = evaluate_mod.main(
        ["--run_dir", str(smoke_run), "--split", "test", "--results_csv", str(results)]
    )
    assert code == evaluate_mod.EXIT_FAILURE
    captured = capsys.readouterr()
    assert "TEST SPLIT" in captured.out
    assert "also appear in val.txt" in captured.err
    assert not (smoke_run / f"eval_test_{TRAIN_DATASET}.json").exists()
    assert not results.exists(), "a refused evaluation must not write a row"


def test_isolated_test_split_evaluation_writes_row(
    isolated_test_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no overlap, the test split runs once, prints the warning and logs a row."""
    results = tmp_path / "results.csv"
    code = evaluate_mod.main(
        ["--run_dir", str(isolated_test_run), "--split", "test", "--results_csv", str(results)]
    )
    assert code == evaluate_mod.EXIT_OK
    captured = capsys.readouterr()
    assert "TEST SPLIT" in captured.out
    assert "no overlap with train or val" in captured.out
    payload = json.loads((isolated_test_run / f"eval_test_{TRAIN_DATASET}.json").read_text(encoding="utf-8"))
    assert payload["split"] == "test"
    rows = _read_results(results)
    assert len(rows) == 1
    assert rows[0]["split"] == "test"


def test_results_csv_refuses_a_different_header(tmp_path: Path, smoke_run: Path) -> None:
    """Rows are never reordered: an incompatible header is an error, not a rewrite."""
    results = tmp_path / "results.csv"
    results.write_text("run_name,dice\nold,0.5\n", encoding="utf-8")
    code = evaluate_mod.main(
        ["--run_dir", str(smoke_run), "--split", "val", "--results_csv", str(results)]
    )
    assert code == evaluate_mod.EXIT_FAILURE
    assert results.read_text(encoding="utf-8") == "run_name,dice\nold,0.5\n"


def test_append_to_legacy_results_csv_migrates_once(tmp_path: Path, smoke_run: Path) -> None:
    """Appending to a legacy 16-column file backs it up, migrates it, and keeps old values."""
    legacy_header = ",".join(evaluate_mod.LEGACY_RESULTS_COLUMNS)
    legacy_rows = [
        "exp006_all_concat_30ep,test,kvasir_seg,,/content/config_session.yaml,b1b3ded85e1411adb6eb3a079ef4029230eb197f,42,20,127.778,0.909645,0.926230,0.875182,0.025532,0.000000,32.80,4550.1",
    ]
    results = tmp_path / "results.csv"
    results.write_text(legacy_header + "\n" + "\n".join(legacy_rows) + "\n", encoding="utf-8")

    new_row = {
        "run_name": "exp006_all_concat_30ep_seed43",
        "split": "test",
        "dataset": "kvasir_seg",
        "tta": "",
        "config_name": "exp006_all_concat_30ep_seed43",
        "config_file": "configs/exp006_all_concat.yaml",
        "git_commit": "abc123",
        "seed": 43,
        "best_epoch": 20,
        "epochs_trained": 30,
        "params_m": "127.778",
        "best_val_dice": "0.910000",
        "test_dice": "0.920000",
        "test_iou": "0.870000",
        "test_mae": "0.030000",
        "small_lesion_dice": "n/a",
        "small_lesion_n": "0",
        "mean_epoch_time_s": "33.00",
        "peak_gpu_mem_mb": "4550.1",
    }
    code = evaluate_mod.append_results_row(results, new_row)
    assert code == results

    # Backup created next to the file.
    backup = results.parent / evaluate_mod.LEGACY_BACKUP_NAME
    assert backup.is_file()
    backup_text = backup.read_text(encoding="utf-8")
    assert legacy_header in backup_text
    assert "exp006_all_concat_30ep,test,kvasir_seg,,/content/config_session.yaml," in backup_text

    # Old values are identical after migration.
    migrated = list(csv.DictReader(results.open(newline="", encoding="utf-8")))
    old = migrated[0]
    assert old["run_name"] == "exp006_all_concat_30ep"
    assert old["best_epoch"] == "20"  # old ``epochs`` became ``best_epoch``
    assert old["config_name"] == ""  # new column blank for legacy rows
    assert old["epochs_trained"] == ""  # new column blank for legacy rows
    assert old["test_dice"] == "0.926230"
    assert old["small_lesion_dice"] == "0.000000"

    # The new row carries all new columns.
    new = migrated[1]
    assert new["config_name"] == "exp006_all_concat_30ep_seed43"
    assert new["best_epoch"] == "20"
    assert new["epochs_trained"] == "30"
    assert new["small_lesion_dice"] == "n/a"
    assert new["small_lesion_n"] == "0"
    assert list(migrated[0]) == list(evaluate_mod.RESULTS_COLUMNS)


def test_config_override_is_read_and_recorded(isolated_test_run: Path, tmp_path: Path) -> None:
    """--config replaces the run's resolved config; a missing override is a usage error."""
    source = isolated_test_run / "config_resolved.yaml"
    override = tmp_path / "session_config.yaml"
    override.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    results = tmp_path / "results.csv"
    code = evaluate_mod.main(
        [
            "--run_dir",
            str(isolated_test_run),
            "--split",
            "val",
            "--config",
            str(override),
            "--results_csv",
            str(results),
        ]
    )
    assert code == evaluate_mod.EXIT_OK
    payload = json.loads(
        (isolated_test_run / f"eval_val_{TRAIN_DATASET}.json").read_text(encoding="utf-8")
    )
    assert payload["config_override"] == str(override)

    code = evaluate_mod.main(
        [
            "--run_dir",
            str(isolated_test_run),
            "--split",
            "val",
            "--config",
            str(tmp_path / "missing.yaml"),
            "--results_csv",
            str(results),
        ]
    )
    assert code == evaluate_mod.EXIT_USAGE


def test_processed_loader_shapes_and_binary_masks(
    isolated_test_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The zero-shot loader yields (B,3,H,W) images and (B,1,H,W) masks in {0,1}."""
    data_root = tmp_path / "data"
    cfg = load_config(isolated_test_run / "config_resolved.yaml")
    size = int(cfg["data"]["image_size"])
    _write_processed_dataset(data_root, "cvc_clinicdb", size, n=3)
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    resolved_root = resolve_data_root(str(cfg["data"].get("data_root", "")))
    loader = evaluate_mod.build_processed_loader(cfg, get_spec("cvc_clinicdb"), resolved_root)
    batches = list(loader)
    for batch in batches:
        assert batch["image"].shape[1:] == (3, size, size)
        assert batch["mask"].shape[1:] == (1, size, size)
        assert set(torch.unique(batch["mask"]).tolist()) <= {0.0, 1.0}
    seen = [sample_id for batch in batches for sample_id in batch["id"]]
    assert seen == ["cvc_clinicdb_000", "cvc_clinicdb_001", "cvc_clinicdb_002"]


def test_multi_dataset_evaluation_feeds_the_aggregate_matrix(
    isolated_test_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: three datasets -> three rows -> a config x dataset matrix.

    This ties Steps 3 and 4 together on a real (tiny) smoke checkpoint.
    """
    from scripts import aggregate_results as agg

    data_root = tmp_path / "data"
    cfg = load_config(isolated_test_run / "config_resolved.yaml")
    size = int(cfg["data"]["image_size"])
    _write_processed_dataset(data_root, "cvc_clinicdb", size, n=2)
    _write_processed_dataset(data_root, "cvc_colondb", size, n=4)
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    results = tmp_path / "results.csv"
    code = evaluate_mod.main(
        [
            "--run_dir",
            str(isolated_test_run),
            "--split",
            "test",
            "--dataset",
            TRAIN_DATASET,
            "--dataset",
            "cvc_clinicdb",
            "--dataset",
            "cvc_colondb",
            "--results_csv",
            str(results),
        ]
    )
    assert code == evaluate_mod.EXIT_OK

    rows = _read_results(results)
    assert [row["dataset"] for row in rows] == [TRAIN_DATASET, "cvc_clinicdb", "cvc_colondb"]
    assert [row["split"] for row in rows] == ["test", "test", "test"]
    assert (isolated_test_run / "eval_test_cvc_colondb.json").is_file()

    out_dir = tmp_path / "aggregate"
    assert agg.main(["--runs-dir", str(isolated_test_run.parent), "--out-dir", str(out_dir)]) == 0
    markdown = (out_dir / "aggregate_results.md").read_text(encoding="utf-8")
    assert "| Config | kvasir_seg | cvc_clinicdb | cvc_colondb | n seeds |" in markdown
    csv_rows = _read_aggregate(out_dir / "aggregate_results.csv")
    assert {row["dataset"] for row in csv_rows} == {"kvasir_seg", "cvc_clinicdb", "cvc_colondb"}
    assert len({row["config"] for row in csv_rows}) == 1, "all three rows belong to the same run"


def _read_aggregate(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
