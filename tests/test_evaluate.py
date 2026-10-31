"""Tests for src/evaluate.py: files, results row, duplicates and test safety.

Every test uses a real (tiny) smoke run on synthetic data at 64 px: the smoke
config is the only training this repo allows on the CPU box. Nothing downloads
pretrained weights.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from src import evaluate as evaluate_mod
from src import train as train_mod
from src.losses import build_loss
from src.metrics import SegMetrics
from src.models.net import build_model
from src.train import evaluate as train_evaluate
from src.utils import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = REPO_ROOT / "configs" / "exp001_convnext_base_baseline.yaml"
RUN_NAME = "exp001_convnext_base_baseline_smoke"


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


def test_summarize_rows_matches_hand_computed_values() -> None:
    """Mean, sample std and the small-lesion split must match hand arithmetic."""
    rows = [
        {"id": "a", "dice": 0.5, "iou": 0.25, "lesion_pixels": 10.0, "small_lesion": True},
        {"id": "b", "dice": 1.0, "iou": 0.75, "lesion_pixels": 100.0, "small_lesion": False},
        {"id": "c", "dice": 0.0, "iou": 0.0, "lesion_pixels": 200.0, "small_lesion": False},
    ]
    summary = evaluate_mod.summarize_rows(rows)
    assert summary["n"] == 3
    assert summary["dice_mean"] == pytest.approx(0.5)
    assert summary["iou_mean"] == pytest.approx(1.0 / 3.0)
    assert summary["dice_std"] == pytest.approx(0.5)  # sample std of 0.5, 1.0, 0.0
    assert summary["small_lesion_dice"] == pytest.approx(0.5)
    assert summary["small_lesion_n"] == 1


def test_evaluate_writes_files_and_one_row(smoke_run: Path, tmp_path: Path) -> None:
    """A val evaluation writes eval_val.json, per_image_val.csv and one CSV row."""
    results = tmp_path / "results.csv"
    code = evaluate_mod.main(
        ["--run_dir", str(smoke_run), "--split", "val", "--results_csv", str(results)]
    )
    assert code == evaluate_mod.EXIT_OK

    json_path = smoke_run / "eval_val.json"
    per_image_path = smoke_run / "per_image_val.csv"
    assert json_path.is_file()
    assert per_image_path.is_file()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["split"] == "val"
    assert payload["n"] == 2
    assert payload["checkpoint"] == "best"
    assert payload["seed"] == 42
    assert payload["epochs"] == 1
    assert payload["params_total"] > 0
    assert payload["params_m"] == pytest.approx(payload["params_total"] / 1e6, rel=1e-6)
    assert payload["threshold"] == 0.5

    with per_image_path.open(newline="", encoding="utf-8") as handle:
        per_image = list(csv.DictReader(handle))
    assert len(per_image) == 2
    assert set(per_image[0]) == {"id", "dice", "iou", "lesion_pixels", "small_lesion"}
    assert payload["dice_mean"] == pytest.approx(
        sum(float(row["dice"]) for row in per_image) / len(per_image), abs=1e-6
    )

    rows = _read_results(results)
    assert len(rows) == 1
    assert list(rows[0]) == list(evaluate_mod.RESULTS_COLUMNS)
    row = rows[0]
    assert row["run_name"] == RUN_NAME
    assert row["split"] == "val"
    assert row["seed"] == "42"
    assert row["epochs"] == "1"
    assert float(row["test_dice"]) == pytest.approx(payload["dice_mean"], abs=1e-6)
    assert float(row["test_iou"]) == pytest.approx(payload["iou_mean"], abs=1e-6)


def test_evaluate_metrics_match_the_training_function(smoke_run: Path, tmp_path: Path) -> None:
    """The evaluation numbers equal src.train.evaluate on the same weights/loader."""
    results = tmp_path / "results.csv"
    code = evaluate_mod.main(
        ["--run_dir", str(smoke_run), "--split", "val", "--results_csv", str(results)]
    )
    assert code == evaluate_mod.EXIT_OK
    payload = json.loads((smoke_run / "eval_val.json").read_text(encoding="utf-8"))

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
    assert not (smoke_run / "eval_test.json").exists()
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
    payload = json.loads((isolated_test_run / "eval_test.json").read_text(encoding="utf-8"))
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
