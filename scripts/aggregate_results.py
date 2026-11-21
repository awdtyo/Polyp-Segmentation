"""Aggregate evaluated runs into config x dataset tables, and compare two configs.

Usage:
    python scripts/aggregate_results.py --runs-dir /path/to/runs
    python scripts/aggregate_results.py --runs-dir /path/to/runs \
        --compare exp005_hybrid_fusion exp007_hybrid_msu --dataset kvasir_seg

The script reads every ``eval_<split>_<dataset>[_ttaflip].json`` (and its
matching ``per_image_...csv``) from each run folder under ``--runs-dir`` -- the
files written by ``src.evaluate`` -- and groups them by config, ignoring the
seed suffix in the run name. One config's seeds make one cell of the matrix;
rows are configs, columns are datasets.

It writes two files next to the runs by default (``--out-dir`` overrides that):

* ``aggregate_results.md``: the human table. A Dice matrix and an IoU matrix
  (rows = configs, columns = datasets, cells = ``mean +/- std``), a details
  table (MAE, small-lesion Dice, parameters, epoch time), and the optional
  comparison.
* ``aggregate_results.csv``: the same numbers in long form, one row per
  ``(config, dataset, split, tta)``, so the matrix can be pivoted or joined.

A config trained with one seed has no sample standard deviation; its std is
written as ``n/a`` instead of 0.0, because 0.0 would look like a measured
spread. A cell that was never evaluated is ``n/a`` too.

``--compare A B`` adds a paired per-image comparison (B minus A). For every
image of one dataset the per-image Dice difference is averaged over the seeds
both configs share, then a bootstrap over images gives a 95% confidence
interval with a fixed seed, so the interval is reproducible. What this shows:
on *this* fixed test set, whether B tends to score higher than A image by image,
averaged over the training seeds. What it does not show: anything about other
datasets or splits, a correction for multiple comparisons, or that a CI
excluding zero is a real-world improvement. It also does not account for image
difficulty beyond pairing, and it cannot fix a leaky split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.evaluate import experiment_key

#: Suffix appended by the Colab launcher / probe protocol, e.g. ``_seed43``.
SEED_SUFFIX = re.compile(r"_seed\d+$")
#: One ``eval_*.json`` per (split, dataset, TTA) written by ``src.evaluate``.
EVAL_GLOB = "eval_*.json"
EVAL_PREFIX = "eval_"
PER_IMAGE_PREFIX = "per_image_"
#: Infix that marks a flip-TTA evaluation.
TTA_SUFFIX = "_ttaflip"
#: Splits ``src.evaluate`` can write.
SPLITS: tuple[str, ...] = ("test", "val")

#: Columns of the long-form aggregate CSV.
AGGREGATE_COLUMNS: tuple[str, ...] = (
    "config",
    "config_name",
    "dataset",
    "split",
    "tta",
    "n_seeds",
    "best_epoch",
    "epochs_trained",
    "dice_mean",
    "dice_std",
    "iou_mean",
    "iou_std",
    "mae_mean",
    "small_lesion_dice",
    "small_lesion_n",
    "params_m",
    "mean_epoch_time_s",
    "peak_gpu_mem_mb",
)

#: Marker for "not evaluated" or "no sample std is defined" (one seed only).
NOT_AVAILABLE = "n/a"
#: Cell preferred for the matrix and for ``--compare`` when several exist.
DEFAULT_SPLIT = "test"
DEFAULT_TTA = ""


@dataclass
class RunResult:
    """One evaluated ``(run, dataset, split, TTA)``, read from its eval files.

    Attributes:
        run_name: Folder name.
        config: Config identity (run name with any ``_seed<N>`` / ``_s<N>``
            suffix stripped). Different experiments are never merged into one
            group.
        seed: Training seed.
        dataset: Dataset name, e.g. ``kvasir_seg``.
        split: ``test`` or ``val``.
        tta: TTA label, ``""`` or ``"flip"``.
        dice_mean: Mean per-image Dice.
        iou_mean: Mean per-image IoU.
        mae_mean: Mean per-image MAE, or ``None`` when the file predates MAE.
        small_lesion_dice: Mean Dice on images below the small-lesion threshold.
        small_lesion_n: Number of images that qualified as small lesions.
        params_m: Parameters in millions.
        mean_epoch_time_s: Mean epoch time from ``run_info.json`` (may be None).
        best_epoch: Epoch at which ``best.pt`` was saved (may be None).
        epochs_trained: Total epochs run, from ``metrics.csv`` (may be None).
        peak_gpu_mem_mb: Peak GPU memory in MB (may be None).
        per_image_dice: ``{image_id: dice}`` for the paired comparison.
    """

    run_name: str
    config: str
    seed: int
    dataset: str
    split: str
    tta: str
    dice_mean: float
    iou_mean: float
    mae_mean: float | None
    small_lesion_dice: float
    small_lesion_n: int = 0
    params_m: float = 0.0
    mean_epoch_time_s: float | None = None
    best_epoch: int | None = None
    epochs_trained: int | None = None
    peak_gpu_mem_mb: float | None = None
    per_image_dice: dict[str, float] = field(default_factory=dict)

    @property
    def cell(self) -> tuple[str, str, str]:
        """``(dataset, split, tta)``, the cell this result belongs to."""
        return (self.dataset, self.split, self.tta)


def config_key(payload: Mapping[str, Any], run_name: str) -> str:
    """Return the config identity, ignoring the seed suffix.

    Grouping is always by experiment (the run name with any ``_seed<N>`` /
    ``_s<N>`` suffix stripped). ``config_file`` (the session config path) is
    deliberately *not* used, since every Colab row historically shares
    ``/content/config_session.yaml`` and different experiments would otherwise be
    averaged as if they were seeds.

    Args:
        payload: Parsed ``eval_*.json``.
        run_name: Run folder name / ``run_name`` column.

    Returns:
        The experiment name with a trailing seed suffix removed.
    """
    return experiment_key(run_name)


def parse_eval_name(stem: str) -> tuple[str, str, str] | None:
    """Split an ``eval_<split>_<dataset>[_ttaflip]`` stem into its parts.

    Args:
        stem: File stem, e.g. ``eval_test_cvc_clinicdb``.

    Returns:
        ``(split, dataset, tta)`` with ``tta`` in ``{"", "flip"}``, or ``None``
        when the stem is not an evaluate output name.
    """
    if not stem.startswith(EVAL_PREFIX):
        return None
    rest = stem[len(EVAL_PREFIX) :]
    tta = ""
    if rest.endswith(TTA_SUFFIX):
        tta = "flip"
        rest = rest[: -len(TTA_SUFFIX)]
    for split in SPLITS:
        prefix = f"{split}_"
        if rest.startswith(prefix) and len(rest) > len(prefix):
            return split, rest[len(prefix) :], tta
    return None


def load_run(run_dir: Path) -> list[RunResult]:
    """Load every evaluated dataset of one run folder.

    Args:
        run_dir: Candidate run folder.

    Returns:
        One :class:`RunResult` per ``eval_*.json`` with a matching
        ``per_image_*.csv``. A folder with neither (e.g. a training-only run)
        yields an empty list. An ``eval_*.json`` without its per-image CSV is
        skipped with a warning.
    """
    results: list[RunResult] = []
    for eval_path in sorted(run_dir.glob(EVAL_GLOB)):
        stem = eval_path.stem
        parsed = parse_eval_name(stem)
        per_image_path = run_dir / f"{PER_IMAGE_PREFIX}{stem[len(EVAL_PREFIX) :]}.csv"
        if not per_image_path.is_file():
            print(f"warning: {per_image_path} is missing; skipping {eval_path.name}", file=sys.stderr)
            continue
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        split = str(payload.get("split") or (parsed[0] if parsed else DEFAULT_SPLIT))
        dataset = str(payload.get("dataset") or (parsed[1] if parsed else ""))
        tta = str(payload.get("tta") or (parsed[2] if parsed else ""))
        per_image: dict[str, float] = {}
        with per_image_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                per_image[str(row["id"])] = float(row["dice"])
        epoch_time = payload.get("mean_epoch_time_s")
        mae = payload.get("mae_mean")
        small_lesion_dice = payload.get("small_lesion_dice")
        if small_lesion_dice is None:
            small_lesion_dice = 0.0
        results.append(
            RunResult(
                run_name=run_dir.name,
                config=config_key(payload, run_dir.name),
                seed=int(payload.get("seed", -1)),
                dataset=dataset,
                split=split,
                tta=tta,
                dice_mean=float(payload["dice_mean"]),
                iou_mean=float(payload["iou_mean"]),
                mae_mean=None if mae is None else float(mae),
                small_lesion_dice=float(small_lesion_dice),
                small_lesion_n=int(payload.get("small_lesion_n", 0) or 0),
                params_m=float(payload.get("params_m", 0.0)),
                mean_epoch_time_s=None if epoch_time is None else float(epoch_time),
                best_epoch=payload.get("best_epoch"),
                epochs_trained=payload.get("epochs_trained"),
                peak_gpu_mem_mb=payload.get("peak_gpu_mem_mb"),
                per_image_dice=per_image,
            )
        )
    return results


def load_runs(runs_dir: str | Path) -> list[RunResult]:
    """Load every evaluated result under ``runs_dir``.

    Args:
        runs_dir: Directory containing run folders.

    Returns:
        Results sorted by ``(config, dataset, split, tta, seed)``.

    Raises:
        FileNotFoundError: if ``runs_dir`` does not exist.
    """
    root = Path(runs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {root}")
    runs: list[RunResult] = []
    for path in sorted(root.iterdir()):
        if path.is_dir():
            runs.extend(load_run(path))
    runs.sort(key=lambda run: (run.config, run.dataset, run.split, run.tta, run.seed))
    return runs


def dataset_cells(runs: Sequence[RunResult]) -> list[tuple[str, str, str]]:
    """The distinct ``(dataset, split, tta)`` cells present in ``runs``, sorted."""
    return sorted({run.cell for run in runs})


def group_cell(runs: Sequence[RunResult], cell: tuple[str, str, str]) -> dict[str, list[RunResult]]:
    """Group the results of one cell by config.

    Args:
        runs: Evaluated results.
        cell: ``(dataset, split, tta)``.

    Returns:
        ``{config: [runs...]}`` with runs ordered by seed, configs sorted.
    """
    groups: dict[str, list[RunResult]] = {}
    for run in runs:
        if run.cell == cell:
            groups.setdefault(run.config, []).append(run)
    return {config: sorted(items, key=lambda run: run.seed) for config, items in sorted(groups.items())}


def _mean_std(values: Sequence[float]) -> tuple[float, float | None]:
    """Mean and sample std; std is ``None`` for fewer than two values."""
    if not values:
        return 0.0, None
    mean = statistics.fmean(values)
    return mean, (statistics.stdev(values) if len(values) > 1 else None)


def summarize_runs(runs: Sequence[RunResult]) -> list[dict[str, Any]]:
    """Summarise every cell of every config across its seeds.

    Args:
        runs: Evaluated results.

    Returns:
        One dict per ``(config, dataset, split, tta)`` with mean/std Dice and
        IoU, MAE, small-lesion Dice, parameters and mean epoch time.
        ``dice_std``/``iou_std`` are ``None`` when the config has a single seed.
    """
    rows: list[dict[str, Any]] = []
    for cell in dataset_cells(runs):
        groups = group_cell(runs, cell)
        for config, items in groups.items():
            dice_mean, dice_std = _mean_std([run.dice_mean for run in items])
            iou_mean, iou_std = _mean_std([run.iou_mean for run in items])
            mae_values = [run.mae_mean for run in items if run.mae_mean is not None]
            epoch_times = [run.mean_epoch_time_s for run in items if run.mean_epoch_time_s is not None]
            peak_mems = [run.peak_gpu_mem_mb for run in items if run.peak_gpu_mem_mb is not None]
            best_epochs = [run.best_epoch for run in items if run.best_epoch is not None]
            epoch_traineds = [run.epochs_trained for run in items if run.epochs_trained is not None]
            rows.append(
                {
                    "config": config,
                    "config_name": config,
                    "dataset": cell[0],
                    "split": cell[1],
                    "tta": cell[2],
                    "n_seeds": len(items),
                    "dice_mean": dice_mean,
                    "dice_std": dice_std,
                    "iou_mean": iou_mean,
                    "iou_std": iou_std,
                    "mae_mean": statistics.fmean(mae_values) if mae_values else None,
                    "small_lesion_dice": statistics.fmean([run.small_lesion_dice for run in items]),
                    "small_lesion_n": items[0].small_lesion_n,
                    "params_m": items[0].params_m,
                    "best_epoch": statistics.fmean(best_epochs) if best_epochs else None,
                    "epochs_trained": statistics.fmean(epoch_traineds) if epoch_traineds else None,
                    "mean_epoch_time_s": statistics.fmean(epoch_times) if epoch_times else None,
                    "peak_gpu_mem_mb": statistics.fmean(peak_mems) if peak_mems else None,
                    "seeds": [run.seed for run in items],
                }
            )
    return rows


def preferred_rows(
    rows: Sequence[Mapping[str, Any]], *, split: str = DEFAULT_SPLIT, tta: str = DEFAULT_TTA
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Pick one row per ``(config, dataset)``: the preferred ``(split, tta)``.

    Args:
        rows: Output of :func:`summarize_runs`.
        split: Preferred split.
        tta: Preferred TTA label.

    Returns:
        ``{(config, dataset): row}``; a dataset without the preferred cell falls
        back to whatever cell it does have, so nothing silently disappears.
    """
    chosen: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["config"]), str(row["dataset"]))
        score = (row["split"] == split, row["tta"] == tta)
        if key not in chosen or score > (chosen[key]["split"] == split, chosen[key]["tta"] == tta):
            chosen[key] = row
    return chosen


def dataset_order(names: Iterable[str]) -> list[str]:
    """Datasets in registry order, with any unknown names appended sorted.

    Args:
        names: Dataset names present in the runs.

    Returns:
        The names, registry-first, so a table lists datasets in the project's
        canonical order instead of alphabetically.
    """
    wanted = set(names)
    try:
        from src.data_registry import load_registry

        known = list(load_registry())
    except (ImportError, OSError, ValueError):
        known = []
    ordered = [name for name in known if name in wanted]
    ordered += sorted(name for name in wanted if name not in ordered)
    return ordered


def _format_std(value: float | None) -> str:
    """Format a std, using ``n/a`` when it is undefined (one seed)."""
    return NOT_AVAILABLE if value is None else f"{value:.4f}"


def _format_mean_std(mean: float, std: float | None) -> str:
    """Format ``mean +/- std`` with ``n/a`` for an undefined std."""
    return f"{mean:.4f} +/- {_format_std(std)}"


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap 95% confidence interval of the mean of ``values``.

    Args:
        values: Per-image differences.
        n_boot: Bootstrap resamples.
        seed: Fixed RNG seed, so the interval is reproducible.

    Returns:
        ``(mean, lower, upper)``; the interval is the 2.5/97.5 percentile of the
        resampled means.

    Raises:
        ValueError: if ``values`` is empty or ``n_boot`` is not positive.
    """
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(n_boot), array.size))
    means = array[indices].mean(axis=1)
    return float(array.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_comparison(
    groups: Mapping[str, Sequence[RunResult]],
    config_a: str,
    config_b: str,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired per-image Dice comparison of B against A, averaged over seeds.

    Args:
        groups: One cell's runs, ``{config: [runs...]}`` (see
            :func:`group_cell`); every run must belong to the same dataset.
        config_a: Reference config.
        config_b: Compared config.
        n_boot: Bootstrap resamples for the confidence interval.
        seed: Fixed bootstrap seed.

    Returns:
        Dict with ``config_a``, ``config_b``, ``n_seeds``, ``n_images``,
        ``mean_diff`` (B - A), ``ci_low``, ``ci_high``, ``per_seed_diffs`` and
        ``note``.

    Raises:
        KeyError: if either config is missing.
        ValueError: if the two configs share no seed or no test image.
    """
    runs_a = {run.seed: run for run in groups[config_a]}
    runs_b = {run.seed: run for run in groups[config_b]}
    shared_seeds = sorted(set(runs_a) & set(runs_b))
    if not shared_seeds:
        raise ValueError(f"{config_a} and {config_b} share no seed; a paired comparison needs the same seeds")
    datasets = {runs_a[seed_value].dataset for seed_value in shared_seeds} | {
        runs_b[seed_value].dataset for seed_value in shared_seeds
    }
    if len(datasets) > 1:
        raise ValueError(f"a paired comparison needs one dataset, got {sorted(datasets)}")

    common_ids = set(runs_a[shared_seeds[0]].per_image_dice)
    for seed_value in shared_seeds:
        common_ids &= set(runs_a[seed_value].per_image_dice) & set(runs_b[seed_value].per_image_dice)
    if not common_ids:
        raise ValueError(f"{config_a} and {config_b} share no test image ids")
    ids = sorted(common_ids)

    per_seed_diffs: dict[int, float] = {}
    for seed_value in shared_seeds:
        diffs = [
            runs_b[seed_value].per_image_dice[image_id] - runs_a[seed_value].per_image_dice[image_id]
            for image_id in ids
        ]
        per_seed_diffs[seed_value] = statistics.fmean(diffs)

    # Average each image's difference over the seeds, then bootstrap over images.
    per_image = [
        statistics.fmean(
            [
                runs_b[seed_value].per_image_dice[image_id] - runs_a[seed_value].per_image_dice[image_id]
                for seed_value in shared_seeds
            ]
        )
        for image_id in ids
    ]
    mean_diff, ci_low, ci_high = bootstrap_ci(per_image, n_boot=n_boot, seed=seed)
    return {
        "config_a": config_a,
        "config_b": config_b,
        "dataset": sorted(datasets)[0],
        "n_seeds": len(shared_seeds),
        "n_images": len(ids),
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "per_seed_diffs": per_seed_diffs,
        "note": (
            "Paired per-image test Dice (B - A), averaged over shared seeds; "
            "the 95% CI comes from bootstrapping images with a fixed seed. It "
            "describes this fixed test set and dataset only, does not correct "
            "for multiple comparisons, and a CI excluding 0 is not proof of a "
            "real-world gain."
        ),
    }


def write_aggregate_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write the long-form aggregate table as CSV.

    Args:
        path: Destination; parents are created.
        rows: Output of :func:`summarize_runs`.

    Returns:
        The written path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AGGREGATE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "config": row["config"],
                    "config_name": row["config_name"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "tta": row["tta"],
                    "n_seeds": row["n_seeds"],
                    "best_epoch": _fmt_optional_int(row.get("best_epoch")),
                    "epochs_trained": _fmt_optional_int(row.get("epochs_trained")),
                    "dice_mean": f"{row['dice_mean']:.6f}",
                    "dice_std": NOT_AVAILABLE if row["dice_std"] is None else f"{row['dice_std']:.6f}",
                    "iou_mean": f"{row['iou_mean']:.6f}",
                    "iou_std": NOT_AVAILABLE if row["iou_std"] is None else f"{row['iou_std']:.6f}",
                    "mae_mean": "" if row["mae_mean"] is None else f"{row['mae_mean']:.6f}",
                    "small_lesion_dice": _fmt_optional_float(row.get("small_lesion_dice")),
                    "small_lesion_n": row["small_lesion_n"],
                    "params_m": f"{row['params_m']:.3f}",
                    "mean_epoch_time_s": (
                        "" if row["mean_epoch_time_s"] is None else f"{row['mean_epoch_time_s']:.2f}"
                    ),
                    "peak_gpu_mem_mb": (
                        "" if row["peak_gpu_mem_mb"] is None else f"{row['peak_gpu_mem_mb']:.1f}"
                    ),
                }
            )
    return out


def _fmt_optional_int(value: Any) -> str:
    """Format an optional int, blank when None."""
    if value is None:
        return ""
    return str(int(value))


def _fmt_optional_float(value: Any) -> str:
    """Format an optional float, using ``n/a`` when missing or NaN."""
    if value is None:
        return NOT_AVAILABLE
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return NOT_AVAILABLE
    if math.isnan(fvalue):
        return NOT_AVAILABLE
    return f"{fvalue:.6f}"


def write_aggregate_markdown(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    comparison: Mapping[str, Any] | None = None,
    split: str = DEFAULT_SPLIT,
    tta: str = DEFAULT_TTA,
) -> Path:
    """Write the config x dataset matrices (and optional comparison) as Markdown.

    Args:
        path: Destination; parents are created.
        rows: Output of :func:`summarize_runs`.
        comparison: Optional output of :func:`paired_comparison`.
        split: Split shown in the matrices.
        tta: TTA label shown in the matrices.

    Returns:
        The written path.
    """
    chosen = preferred_rows(rows, split=split, tta=tta)
    configs = sorted({config for config, _ in chosen})
    datasets = dataset_order({dataset for _, dataset in chosen})

    lines = [
        "# Aggregated test results",
        "",
        "Config x dataset matrix. Rows are experiments (configs), columns are "
        "datasets, cells are per-image ``mean +/- sample std`` across training "
        "seeds. `n/a` means either a single seed (no spread is defined) or a "
        "dataset that was not evaluated. Split "
        f"`{split}`, TTA `{tta or 'off'}`; other cells are in "
        "`aggregate_results.csv`.",
        "",
    ]
    for label, mean_key, std_key in (
        ("Dice", "dice_mean", "dice_std"),
        ("IoU", "iou_mean", "iou_std"),
    ):
        header = "| Config | " + " | ".join(datasets) + " | n seeds |"
        separator = "|---" + "|---:" * len(datasets) + "|---:|"
        lines += [f"## {label} (per-image mean over seeds)", "", header, separator]
        for config in configs:
            cells = []
            n_seeds = 0
            for dataset in datasets:
                row = chosen.get((config, dataset))
                if row is None:
                    cells.append(NOT_AVAILABLE)
                else:
                    cells.append(_format_mean_std(row[mean_key], row[std_key]))
                    n_seeds = max(n_seeds, int(row["n_seeds"]))
            lines.append(f"| {config} | " + " | ".join(cells) + f" | {n_seeds} |")
        lines.append("")

    lines += [
        "## Details",
        "",
        "| Config | Dataset | Split | TTA | Seeds | Best epoch | Epochs trained | MAE | Small-lesion Dice | Small-lesion n | Params (M) | Mean epoch time (s) | Peak GPU mem (MB) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (str(item["config"]), str(item["dataset"]), str(item["split"]))):
        epoch_time = "" if row["mean_epoch_time_s"] is None else f"{row['mean_epoch_time_s']:.2f}"
        peak_mem = "" if row["peak_gpu_mem_mb"] is None else f"{row['peak_gpu_mem_mb']:.1f}"
        mae = "" if row["mae_mean"] is None else f"{row['mae_mean']:.4f}"
        small = _fmt_optional_float(row.get("small_lesion_dice"))
        lines.append(
            "| {config} | {dataset} | {split} | {tta} | {seeds} | {best_epoch} | {epochs_trained} | {mae} | {small} | {small_n} | {params:.2f} | {epoch} | {peak} |".format(
                config=row["config"],
                dataset=row["dataset"],
                split=row["split"],
                tta=row["tta"] or "-",
                seeds=row["n_seeds"],
                best_epoch=row.get("best_epoch", "") or "",
                epochs_trained=row.get("epochs_trained", "") or "",
                mae=mae,
                small=small,
                small_n=row["small_lesion_n"],
                params=row["params_m"],
                epoch=epoch_time,
                peak=peak_mem,
            )
        )
    if comparison is not None:
        lines += [
            "",
            f"## Paired comparison: {comparison['config_b']} vs {comparison['config_a']} "
            f"({comparison['dataset']})",
            "",
            (
                f"Mean per-image Dice difference (B - A): "
                f"{comparison['mean_diff']:+.4f} "
                f"(95% CI {comparison['ci_low']:+.4f} to {comparison['ci_high']:+.4f}), "
                f"{comparison['n_images']} test images, {comparison['n_seeds']} shared seeds."
            ),
            "",
            comparison["note"],
        ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def resolve_cell(
    runs: Sequence[RunResult],
    config_a: str,
    config_b: str,
    *,
    dataset: str | None,
    split: str,
    tta: str,
) -> tuple[str, str, str]:
    """Choose the cell a paired comparison runs on.

    Args:
        runs: Evaluated results.
        config_a: Reference config.
        config_b: Compared config.
        dataset: Explicit dataset, or ``None`` to auto-pick one that holds both
            configs in the requested split/TTA.
        split: Requested split.
        tta: Requested TTA label.

    Returns:
        The ``(dataset, split, tta)`` cell.

    Raises:
        ValueError: if no cell matches.
    """
    cells = dataset_cells(runs)
    if dataset is not None:
        target = (dataset, split, tta)
        if target not in cells:
            raise ValueError(f"no evaluated cell {target}; available cells: {cells}")
        return target
    preferred = [cell for cell in cells if cell[1] == split and cell[2] == tta]
    for cell in preferred:
        configs = set(group_cell(runs, cell))
        if {config_a, config_b} <= configs:
            return cell
    if preferred:
        return preferred[0]
    raise ValueError(f"no evaluated cell with split={split!r} tta={tta!r}; available cells: {cells}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Namespace with ``runs_dir``, ``out_dir``, ``compare``, ``dataset``,
        ``split``, ``tta``, ``bootstrap`` and ``bootstrap_seed``.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate evaluated runs into a config x dataset matrix and optionally compare two configs."
    )
    parser.add_argument("--runs-dir", required=True, type=Path, help="Directory containing run folders")
    parser.add_argument("--out-dir", type=Path, default=None, help="Where to write the tables (default: runs-dir)")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("CONFIG_A", "CONFIG_B"),
        default=None,
        help="Optional paired per-image comparison of B against A",
    )
    parser.add_argument("--dataset", default=None, help="Dataset for the matrix/--compare (default: auto)")
    parser.add_argument("--split", choices=SPLITS, default=DEFAULT_SPLIT, help="Split for the matrix (default: test)")
    parser.add_argument("--tta", default=DEFAULT_TTA, help="TTA label for the matrix (default: none)")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples (default: 2000)")
    parser.add_argument("--bootstrap-seed", type=int, default=42, help="Fixed bootstrap seed (default: 42)")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """Aggregate runs, write the tables and print a short summary.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 when no evaluated runs are found or
        the comparison cannot be made, 2 on bad arguments.
    """
    args = parse_args(argv)
    try:
        runs = load_runs(args.runs_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not runs:
        print(
            f"ERROR: no run with eval_*.json + per_image_*.csv under {args.runs_dir}",
            file=sys.stderr,
        )
        return 1

    configs_found = sorted({run.run_name for run in runs})
    print(
        f"Picked up {len(runs)} evaluated result(s) from {len(configs_found)} run folder(s): "
        + ", ".join(configs_found),
        flush=True,
    )

    rows = summarize_runs(runs)
    comparison = None
    if args.compare is not None:
        config_a, config_b = args.compare
        try:
            cell = resolve_cell(
                runs, config_a, config_b, dataset=args.dataset, split=args.split, tta=args.tta
            )
            comparison = paired_comparison(
                group_cell(runs, cell), config_a, config_b, n_boot=args.bootstrap, seed=args.bootstrap_seed
            )
        except (KeyError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    out_dir = args.out_dir if args.out_dir is not None else Path(args.runs_dir)
    md_path = write_aggregate_markdown(
        out_dir / "aggregate_results.md", rows, comparison=comparison, split=args.split, tta=args.tta
    )
    csv_path = write_aggregate_csv(out_dir / "aggregate_results.csv", rows)

    for row in rows:
        print(
            f"{row['config']} [{row['split']}/{row['dataset']}] {row['n_seeds']} seed(s) "
            f"dice {_format_mean_std(row['dice_mean'], row['dice_std'])} "
            f"iou {_format_mean_std(row['iou_mean'], row['iou_std'])}"
        )
    if comparison is not None:
        print(
            f"paired {comparison['config_b']} - {comparison['config_a']} on {comparison['dataset']}: "
            f"{comparison['mean_diff']:+.4f} "
            f"(95% CI {comparison['ci_low']:+.4f} to {comparison['ci_high']:+.4f})"
        )
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
