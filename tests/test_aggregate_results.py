"""Tests for scripts/aggregate_results.py on toy per-image files.

No training and no real data: the run folders are built directly in a temp
directory, exactly in the shape ``src.evaluate`` writes.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import pytest

from scripts import aggregate_results as agg


def make_run(
    root: Path,
    name: str,
    *,
    seed: int,
    per_image: dict[str, float],
    config_file: str | None = "configs/expA.yaml",
    epoch_time: float | None = 10.0,
    params_m: float = 122.677553,
) -> Path:
    """Write a toy ``eval_test.json`` + ``per_image_test.csv`` run folder.

    Args:
        root: Runs directory.
        name: Run folder name.
        seed: Training seed recorded in the JSON.
        per_image: ``{id: dice}``.
        config_file: Config path recorded in the JSON (None omits the key).
        epoch_time: Mean epoch time.
        params_m: Parameters in millions.

    Returns:
        The run folder.
    """
    run_dir = root / name
    run_dir.mkdir(parents=True)
    dice_mean = sum(per_image.values()) / len(per_image)
    payload = {
        "run_name": name,
        "split": "test",
        "seed": seed,
        "dice_mean": dice_mean,
        "iou_mean": dice_mean * 0.8,
        "small_lesion_dice": min(per_image.values()),
        "params_m": params_m,
        "mean_epoch_time_s": epoch_time,
    }
    if config_file is not None:
        payload["config_file"] = config_file
    (run_dir / "eval_test.json").write_text(json.dumps(payload), encoding="utf-8")
    with (run_dir / "per_image_test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "dice", "iou", "lesion_pixels", "small_lesion"])
        writer.writeheader()
        for image_id, dice in per_image.items():
            writer.writerow(
                {"id": image_id, "dice": dice, "iou": dice * 0.8, "lesion_pixels": 100, "small_lesion": 0}
            )
    return run_dir


@pytest.fixture()
def toy_runs(tmp_path: Path) -> Path:
    """expA with two seeds, expB with one seed, plus a training-only folder."""
    root = tmp_path / "runs"
    make_run(root, "expA_seed42", seed=42, per_image={"a": 0.8, "b": 0.6, "c": 1.0})
    make_run(root, "expA_seed43", seed=43, per_image={"a": 0.9, "b": 0.7, "c": 1.0})
    make_run(
        root,
        "expB_seed42",
        seed=42,
        per_image={"a": 0.7, "b": 0.5, "c": 0.9},
        config_file="configs/expB.yaml",
    )
    (root / "expA_seed44_training_only").mkdir()
    (root / "expA_seed44_training_only" / "metrics.csv").write_text("epoch\n0\n", encoding="utf-8")
    return root


def read_aggregate_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["config"]: row for row in csv.DictReader(handle)}


def test_aggregate_groups_seeds_and_marks_single_seed_std(toy_runs: Path, tmp_path: Path) -> None:
    """Two seeds give a real std; one seed gives n/a, never 0.0."""
    out_dir = tmp_path / "out"
    code = agg.main(["--runs-dir", str(toy_runs), "--out-dir", str(out_dir)])
    assert code == 0
    assert (out_dir / "aggregate_results.md").is_file()
    assert (out_dir / "aggregate_results.csv").is_file()

    rows = read_aggregate_csv(out_dir / "aggregate_results.csv")
    assert set(rows) == {"expA", "expB"}
    assert rows["expA"]["n_seeds"] == "2"
    assert rows["expB"]["n_seeds"] == "1"

    seed_means = [(0.8 + 0.6 + 1.0) / 3, (0.9 + 0.7 + 1.0) / 3]
    expected_mean = sum(seed_means) / 2
    assert float(rows["expA"]["dice_mean"]) == pytest.approx(expected_mean, abs=1e-6)
    expected_std = statistics.stdev(seed_means)
    assert float(rows["expA"]["dice_std"]) == pytest.approx(expected_std, abs=1e-6)
    assert rows["expB"]["dice_std"] == "n/a"
    assert rows["expB"]["iou_std"] == "n/a"

    markdown = (out_dir / "aggregate_results.md").read_text(encoding="utf-8")
    assert "| expA | 2 |" in markdown
    assert "n/a" in markdown


def test_aggregate_ignores_folders_without_eval_files(toy_runs: Path) -> None:
    """A training-only folder is skipped instead of crashing the aggregation."""
    runs = agg.load_runs(toy_runs)
    assert {run.run_name for run in runs} == {"expA_seed42", "expA_seed43", "expB_seed42"}


def test_config_key_falls_back_to_the_run_name_without_seed_suffix(tmp_path: Path) -> None:
    """Without a config_file field, ``expX_seed42`` and ``expX_seed43`` group together."""
    root = tmp_path / "runs"
    make_run(root, "expX_seed42", seed=42, per_image={"a": 1.0}, config_file=None)
    make_run(root, "expX_seed43", seed=43, per_image={"a": 0.0}, config_file=None)
    groups = agg.group_by_config(agg.load_runs(root))
    assert list(groups) == ["expX"]
    assert [run.seed for run in groups["expX"]] == [42, 43]


def test_paired_comparison_is_reproducible_and_centred_on_the_hand_value(toy_runs: Path) -> None:
    """The paired mean equals the hand-computed value and the fixed seed repeats the CI."""
    groups = agg.group_by_config(agg.load_runs(toy_runs))
    first = agg.paired_comparison(groups, "expA", "expB", n_boot=500, seed=7)
    second = agg.paired_comparison(groups, "expA", "expB", n_boot=500, seed=7)

    # B has one seed (42), so the only shared seed is 42: B - A is -0.1 on every image.
    assert first["mean_diff"] == pytest.approx(-0.1, abs=1e-12)
    assert first["n_seeds"] == 1
    assert first["n_images"] == 3
    assert first["ci_low"] <= first["mean_diff"] <= first["ci_high"]
    assert (first["ci_low"], first["ci_high"]) == (second["ci_low"], second["ci_high"])
    assert first["per_seed_diffs"][42] == pytest.approx(-0.1, abs=1e-12)


def test_paired_comparison_requires_shared_seeds(toy_runs: Path, tmp_path: Path) -> None:
    """Configs with no shared seed cannot be paired and must fail cleanly."""
    root = tmp_path / "runs2"
    make_run(root, "expA_seed42", seed=42, per_image={"a": 1.0})
    make_run(root, "expB_seed43", seed=43, per_image={"a": 0.5}, config_file="configs/expB.yaml")
    groups = agg.group_by_config(agg.load_runs(root))
    with pytest.raises(ValueError, match="share no seed"):
        agg.paired_comparison(groups, "expA", "expB")

    code = agg.main(["--runs-dir", str(root), "--compare", "expA", "expB"])
    assert code == 1


def test_bootstrap_ci_is_deterministic() -> None:
    """Same seed, same interval; a wider sample gives a wider interval."""
    values = [0.1, -0.2, 0.05, 0.3, -0.1]
    mean, low, high = agg.bootstrap_ci(values, n_boot=300, seed=1)
    again = agg.bootstrap_ci(values, n_boot=300, seed=1)
    assert (mean, low, high) == again
    assert low <= mean <= high
    assert agg.bootstrap_ci(values, n_boot=300, seed=2)[1] != low or agg.bootstrap_ci(
        values, n_boot=300, seed=2
    )[2] != high
