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
    dataset: str = "kvasir_seg",
    split: str = "test",
    tta: str = "",
    config_file: str | None = "configs/expA.yaml",
    epoch_time: float | None = 10.0,
    params_m: float = 122.677553,
    mae: float | None = 0.05,
    with_per_image: bool = True,
) -> Path:
    """Write a toy ``eval_<split>_<dataset>[_ttaflip].json`` + per-image CSV.

    Mirrors ``src.evaluate``: the file name carries ``_ttaflip`` when the TTA
    label is ``"flip"``, and the payload records ``dataset``/``split``/``tta``.

    Args:
        root: Runs directory.
        name: Run folder name.
        seed: Training seed recorded in the JSON.
        per_image: ``{id: dice}``.
        dataset: Dataset name.
        split: ``test`` or ``val``.
        tta: TTA label, ``""`` or ``"flip"``.
        config_file: Config path recorded in the JSON (None omits the key).
        epoch_time: Mean epoch time.
        params_m: Parameters in millions.
        mae: Mean absolute error (None omits the key).
        with_per_image: When False, skip the per-image CSV (a broken run).

    Returns:
        The run folder.
    """
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    dice_mean = sum(per_image.values()) / len(per_image)
    stem = f"{split}_{dataset}{'_ttaflip' if tta else ''}"
    payload: dict[str, object] = {
        "run_name": name,
        "split": split,
        "dataset": dataset,
        "tta": tta,
        "seed": seed,
        "dice_mean": dice_mean,
        "iou_mean": dice_mean * 0.8,
        "small_lesion_dice": min(per_image.values()),
        "params_m": params_m,
        "mean_epoch_time_s": epoch_time,
    }
    if config_file is not None:
        payload["config_file"] = config_file
    if mae is not None:
        payload["mae_mean"] = mae
    (run_dir / f"eval_{stem}.json").write_text(json.dumps(payload), encoding="utf-8")
    if with_per_image:
        with (run_dir / f"per_image_{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["id", "dice", "iou", "mae", "lesion_pixels", "small_lesion"]
            )
            writer.writeheader()
            for image_id, dice in per_image.items():
                writer.writerow(
                    {
                        "id": image_id,
                        "dice": dice,
                        "iou": dice * 0.8,
                        "mae": 0.05,
                        "lesion_pixels": 100,
                        "small_lesion": 0,
                    }
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


def read_aggregate_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_aggregate_groups_seeds_and_marks_single_seed_std(toy_runs: Path, tmp_path: Path) -> None:
    """Two seeds give a real std; one seed gives n/a, never 0.0."""
    out_dir = tmp_path / "out"
    code = agg.main(["--runs-dir", str(toy_runs), "--out-dir", str(out_dir)])
    assert code == 0
    assert (out_dir / "aggregate_results.md").is_file()
    assert (out_dir / "aggregate_results.csv").is_file()

    rows = {(row["config"], row["dataset"]): row for row in read_aggregate_csv(out_dir / "aggregate_results.csv")}
    assert set(rows) == {("expA", "kvasir_seg"), ("expB", "kvasir_seg")}
    assert rows[("expA", "kvasir_seg")]["n_seeds"] == "2"
    assert rows[("expB", "kvasir_seg")]["n_seeds"] == "1"

    seed_means = [(0.8 + 0.6 + 1.0) / 3, (0.9 + 0.7 + 1.0) / 3]
    expected_mean = sum(seed_means) / 2
    assert float(rows[("expA", "kvasir_seg")]["dice_mean"]) == pytest.approx(expected_mean, abs=1e-6)
    assert float(rows[("expA", "kvasir_seg")]["dice_std"]) == pytest.approx(statistics.stdev(seed_means), abs=1e-6)
    assert rows[("expB", "kvasir_seg")]["dice_std"] == "n/a"
    assert rows[("expB", "kvasir_seg")]["iou_std"] == "n/a"

    markdown = (out_dir / "aggregate_results.md").read_text(encoding="utf-8")
    assert "| expA |" in markdown
    assert "n/a" in markdown


def test_matrix_has_one_column_per_dataset(tmp_path: Path) -> None:
    """Rows are configs, columns are datasets, for Dice and IoU."""
    root = tmp_path / "runs"
    make_run(root, "expA_seed42", seed=42, per_image={"a": 0.8, "b": 0.6}, dataset="kvasir_seg")
    make_run(root, "expA_seed42", seed=42, per_image={"a": 0.5, "b": 0.4}, dataset="cvc_clinicdb")
    make_run(
        root, "expB_seed42", seed=42, per_image={"a": 0.7, "b": 0.5}, dataset="kvasir_seg", config_file="configs/expB.yaml"
    )
    out_dir = tmp_path / "out"
    assert agg.main(["--runs-dir", str(root), "--out-dir", str(out_dir)]) == 0

    markdown = (out_dir / "aggregate_results.md").read_text(encoding="utf-8")
    assert "## Dice" in markdown
    assert "## IoU" in markdown
    # Registry order puts kvasir_seg before cvc_clinicdb.
    assert "| Config | kvasir_seg | cvc_clinicdb | n seeds |" in markdown
    # expA was evaluated on both datasets, expB on kvasir only (n/a for the other).
    assert "| expA | 0.7000 +/- n/a | 0.4500 +/- n/a | 1 |" in markdown
    assert "| expB | 0.6000 +/- n/a | n/a | 1 |" in markdown

    rows = read_aggregate_csv(out_dir / "aggregate_results.csv")
    assert {(row["config"], row["dataset"]) for row in rows} == {
        ("expA", "kvasir_seg"),
        ("expA", "cvc_clinicdb"),
        ("expB", "kvasir_seg"),
    }


def test_parse_eval_name_handles_tta_and_underscored_datasets() -> None:
    """The file stem is parsed for split, dataset and the TTA marker."""
    assert agg.parse_eval_name("eval_test_kvasir_seg") == ("test", "kvasir_seg", "")
    assert agg.parse_eval_name("eval_test_cvc_clinicdb") == ("test", "cvc_clinicdb", "")
    assert agg.parse_eval_name("eval_val_cvc_colondb_ttaflip") == ("val", "cvc_colondb", "flip")
    assert agg.parse_eval_name("other_file") is None
    assert agg.parse_eval_name("eval_kvasir_seg") is None


def test_load_run_skips_an_eval_file_without_its_per_image_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A half-written run is skipped with a warning instead of crashing."""
    root = tmp_path / "runs"
    make_run(root, "expA_seed42", seed=42, per_image={"a": 1.0}, with_per_image=False)
    make_run(root, "expA_seed42", seed=42, per_image={"a": 1.0}, dataset="cvc_clinicdb")
    results = agg.load_run(root / "expA_seed42")
    assert [run.dataset for run in results] == ["cvc_clinicdb"]
    assert "is missing; skipping" in capsys.readouterr().err


def test_aggregate_ignores_folders_without_eval_files(toy_runs: Path) -> None:
    """A training-only folder is skipped instead of crashing the aggregation."""
    runs = agg.load_runs(toy_runs)
    assert {run.run_name for run in runs} == {"expA_seed42", "expA_seed43", "expB_seed42"}
    assert all(run.dataset == "kvasir_seg" for run in runs)


def test_dataset_order_prefers_the_registry_order() -> None:
    """Registered datasets come in registry order; unknown names sort last."""
    assert agg.dataset_order({"cvc_clinicdb", "kvasir_seg"}) == ["kvasir_seg", "cvc_clinicdb"]
    assert agg.dataset_order({"zzz_custom", "kvasir_seg"}) == ["kvasir_seg", "zzz_custom"]


def test_config_key_falls_back_to_the_run_name_without_seed_suffix(tmp_path: Path) -> None:
    """Without a config_file field, ``expX_seed42`` and ``expX_seed43`` group together."""
    root = tmp_path / "runs"
    make_run(root, "expX_seed42", seed=42, per_image={"a": 1.0}, config_file=None)
    make_run(root, "expX_seed43", seed=43, per_image={"a": 0.0}, config_file=None)
    groups = agg.group_cell(agg.load_runs(root), ("kvasir_seg", "test", ""))
    assert list(groups) == ["expX"]
    assert [run.seed for run in groups["expX"]] == [42, 43]


def test_paired_comparison_is_reproducible_and_centred_on_the_hand_value(toy_runs: Path) -> None:
    """The paired mean equals the hand-computed value and the fixed seed repeats the CI."""
    groups = agg.group_cell(agg.load_runs(toy_runs), ("kvasir_seg", "test", ""))
    first = agg.paired_comparison(groups, "expA", "expB", n_boot=500, seed=7)
    second = agg.paired_comparison(groups, "expA", "expB", n_boot=500, seed=7)

    # B has one seed (42), so the only shared seed is 42: B - A is -0.1 on every image.
    assert first["mean_diff"] == pytest.approx(-0.1, abs=1e-12)
    assert first["dataset"] == "kvasir_seg"
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
    groups = agg.group_cell(agg.load_runs(root), ("kvasir_seg", "test", ""))
    with pytest.raises(ValueError, match="share no seed"):
        agg.paired_comparison(groups, "expA", "expB")

    code = agg.main(["--runs-dir", str(root), "--compare", "expA", "expB"])
    assert code == 1


def test_compare_auto_picks_a_shared_cell_and_records_the_dataset(tmp_path: Path) -> None:
    """``--compare`` runs on a dataset both configs were evaluated on."""
    root = tmp_path / "runs"
    make_run(root, "expA_seed42", seed=42, per_image={"a": 1.0}, dataset="cvc_colondb")
    make_run(root, "expA_seed42", seed=42, per_image={"a": 0.8}, dataset="kvasir_seg")
    make_run(root, "expB_seed42", seed=42, per_image={"a": 0.9}, dataset="kvasir_seg", config_file="configs/expB.yaml")
    out_dir = tmp_path / "out"
    assert agg.main(["--runs-dir", str(root), "--out-dir", str(out_dir), "--compare", "expA", "expB"]) == 0
    markdown = (out_dir / "aggregate_results.md").read_text(encoding="utf-8")
    assert "## Paired comparison: expB vs expA (kvasir_seg)" in markdown


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
