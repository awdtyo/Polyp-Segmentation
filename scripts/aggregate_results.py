"""Aggregate evaluated runs into one table, and optionally compare two configs.

Usage:
    python scripts/aggregate_results.py --runs-dir /path/to/runs
    python scripts/aggregate_results.py --runs-dir /path/to/runs \
        --compare exp005_hybrid_fusion exp007_hybrid_msu

The script reads ``eval_test.json`` and ``per_image_test.csv`` from every run
folder under ``--runs-dir`` (the files written by ``src.evaluate``) and groups
runs by config, ignoring the seed suffix in the run name. It writes
``aggregate_results.md`` and ``aggregate_results.csv`` next to the runs by
default (``--out-dir`` overrides that).

A config trained with one seed has no sample standard deviation; its std column
is written as ``n/a`` instead of 0.0, because 0.0 would look like a measured
spread.

``--compare A B`` adds a paired per-image comparison (B minus A). For every test
image the per-image Dice difference is averaged over the seeds both configs
share, then a bootstrap over images gives a 95% confidence interval with a fixed
seed, so the interval is reproducible. What this shows: on *this* fixed test
set, whether B tends to score higher than A image by image, averaged over the
training seeds. What it does not show: anything about other datasets or splits,
a correction for multiple comparisons, or that a CI excluding zero is a
real-world improvement. It also does not account for image difficulty beyond
pairing, and it cannot fix a leaky split.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: Suffix appended by the Colab launcher / probe protocol, e.g. ``_seed43``.
SEED_SUFFIX = re.compile(r"_seed\d+$")

#: Columns of the aggregate CSV.
AGGREGATE_COLUMNS: tuple[str, ...] = (
    "config",
    "n_seeds",
    "dice_mean",
    "dice_std",
    "iou_mean",
    "iou_std",
    "small_lesion_dice",
    "params_m",
    "mean_epoch_time_s",
)

#: Marker for "no sample std is defined" (one seed only).
NOT_AVAILABLE = "n/a"


@dataclass
class RunResult:
    """One evaluated run, as read from its eval/per-image files.

    Attributes:
        run_name: Folder name.
        config: Config identity (config file stem, seed suffix stripped).
        seed: Training seed.
        dice_mean: Mean per-image test Dice.
        iou_mean: Mean per-image test IoU.
        small_lesion_dice: Mean Dice on images below the small-lesion threshold.
        params_m: Parameters in millions.
        mean_epoch_time_s: Mean epoch time from ``run_info.json`` (may be None).
        per_image_dice: ``{image_id: dice}`` for the paired comparison.
    """

    run_name: str
    config: str
    seed: int
    dice_mean: float
    iou_mean: float
    small_lesion_dice: float
    params_m: float
    mean_epoch_time_s: float | None
    per_image_dice: dict[str, float] = field(default_factory=dict)


def config_key(payload: Mapping[str, Any], run_name: str) -> str:
    """Return the config identity, ignoring the seed suffix.

    Args:
        payload: Parsed ``eval_test.json``.
        run_name: Run folder name, used as a fallback.

    Returns:
        Config file stem when the payload records one, else the run name with a
        ``_seed<N>`` suffix removed.
    """
    config_file = payload.get("config_file")
    if isinstance(config_file, str) and config_file.strip():
        return Path(config_file).stem
    return SEED_SUFFIX.sub("", run_name)


def load_run(run_dir: Path) -> RunResult | None:
    """Load one run folder.

    Args:
        run_dir: Candidate run folder.

    Returns:
        The parsed :class:`RunResult`, or ``None`` when the folder has no
        ``eval_test.json`` / ``per_image_test.csv`` (e.g. a training-only run).
    """
    eval_path = run_dir / "eval_test.json"
    per_image_path = run_dir / "per_image_test.csv"
    if not eval_path.is_file() or not per_image_path.is_file():
        return None
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    per_image: dict[str, float] = {}
    with per_image_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            per_image[str(row["id"])] = float(row["dice"])
    epoch_time = payload.get("mean_epoch_time_s")
    return RunResult(
        run_name=run_dir.name,
        config=config_key(payload, run_dir.name),
        seed=int(payload.get("seed", -1)),
        dice_mean=float(payload["dice_mean"]),
        iou_mean=float(payload["iou_mean"]),
        small_lesion_dice=float(payload.get("small_lesion_dice", 0.0)),
        params_m=float(payload.get("params_m", 0.0)),
        mean_epoch_time_s=None if epoch_time is None else float(epoch_time),
        per_image_dice=per_image,
    )


def load_runs(runs_dir: str | Path) -> list[RunResult]:
    """Load every evaluated run under ``runs_dir``.

    Args:
        runs_dir: Directory containing run folders.

    Returns:
        Runs sorted by ``(config, seed)``.

    Raises:
        FileNotFoundError: if ``runs_dir`` does not exist.
    """
    root = Path(runs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {root}")
    runs = [load_run(path) for path in sorted(root.iterdir()) if path.is_dir()]
    return sorted((run for run in runs if run is not None), key=lambda run: (run.config, run.seed))


def group_by_config(runs: Sequence[RunResult]) -> dict[str, list[RunResult]]:
    """Group runs by config identity.

    Args:
        runs: Evaluated runs.

    Returns:
        ``{config: [runs...]}`` with runs ordered by seed.
    """
    groups: dict[str, list[RunResult]] = {}
    for run in runs:
        groups.setdefault(run.config, []).append(run)
    return {config: sorted(items, key=lambda run: run.seed) for config, items in sorted(groups.items())}


def _mean_std(values: Sequence[float]) -> tuple[float, float | None]:
    """Mean and sample std; std is ``None`` for fewer than two values."""
    if not values:
        return 0.0, None
    mean = statistics.fmean(values)
    return mean, (statistics.stdev(values) if len(values) > 1 else None)


def summarize_groups(groups: Mapping[str, Sequence[RunResult]]) -> list[dict[str, Any]]:
    """Summarise each config across its seeds.

    Args:
        groups: Output of :func:`group_by_config`.

    Returns:
        One dict per config with mean/std Dice and IoU, small-lesion Dice,
        parameters and mean epoch time. ``dice_std``/``iou_std`` are ``None``
        when the config has a single seed.
    """
    rows: list[dict[str, Any]] = []
    for config, runs in groups.items():
        dice_mean, dice_std = _mean_std([run.dice_mean for run in runs])
        iou_mean, iou_std = _mean_std([run.iou_mean for run in runs])
        epoch_times = [run.mean_epoch_time_s for run in runs if run.mean_epoch_time_s is not None]
        rows.append(
            {
                "config": config,
                "n_seeds": len(runs),
                "dice_mean": dice_mean,
                "dice_std": dice_std,
                "iou_mean": iou_mean,
                "iou_std": iou_std,
                "small_lesion_dice": statistics.fmean([run.small_lesion_dice for run in runs]),
                "params_m": runs[0].params_m,
                "mean_epoch_time_s": statistics.fmean(epoch_times) if epoch_times else None,
                "seeds": [run.seed for run in runs],
            }
        )
    return rows


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
        groups: Output of :func:`group_by_config`.
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
        "n_seeds": len(shared_seeds),
        "n_images": len(ids),
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "per_seed_diffs": per_seed_diffs,
        "note": (
            "Paired per-image test Dice (B - A), averaged over shared seeds; "
            "the 95% CI comes from bootstrapping images with a fixed seed. It "
            "describes this fixed test set only, does not correct for multiple "
            "comparisons, and a CI excluding 0 is not proof of a real-world gain."
        ),
    }


def write_aggregate_csv(path: str | Path, summary: Sequence[Mapping[str, Any]]) -> Path:
    """Write the aggregate table as CSV.

    Args:
        path: Destination; parents are created.
        summary: Output of :func:`summarize_groups`.

    Returns:
        The written path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AGGREGATE_COLUMNS))
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    "config": row["config"],
                    "n_seeds": row["n_seeds"],
                    "dice_mean": f"{row['dice_mean']:.6f}",
                    "dice_std": NOT_AVAILABLE if row["dice_std"] is None else f"{row['dice_std']:.6f}",
                    "iou_mean": f"{row['iou_mean']:.6f}",
                    "iou_std": NOT_AVAILABLE if row["iou_std"] is None else f"{row['iou_std']:.6f}",
                    "small_lesion_dice": f"{row['small_lesion_dice']:.6f}",
                    "params_m": f"{row['params_m']:.3f}",
                    "mean_epoch_time_s": (
                        "" if row["mean_epoch_time_s"] is None else f"{row['mean_epoch_time_s']:.2f}"
                    ),
                }
            )
    return out


def write_aggregate_markdown(
    path: str | Path,
    summary: Sequence[Mapping[str, Any]],
    comparison: Mapping[str, Any] | None = None,
) -> Path:
    """Write the aggregate table (and optional comparison) as Markdown.

    Args:
        path: Destination; parents are created.
        summary: Output of :func:`summarize_groups`.
        comparison: Optional output of :func:`paired_comparison`.

    Returns:
        The written path.
    """
    lines = [
        "# Aggregated test results",
        "",
        "Means and sample standard deviations across training seeds. `n/a` means "
        "the config has only one seed, so no spread is defined.",
        "",
        "| Config | Seeds | Test Dice | Test IoU | Small-lesion Dice | Params (M) | Mean epoch time (s) |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in summary:
        epoch_time = "n/a" if row["mean_epoch_time_s"] is None else f"{row['mean_epoch_time_s']:.2f}"
        lines.append(
            "| {config} | {n_seeds} | {dice} | {iou} | {small:.4f} | {params:.2f} | {epoch} |".format(
                config=row["config"],
                n_seeds=row["n_seeds"],
                dice=_format_mean_std(row["dice_mean"], row["dice_std"]),
                iou=_format_mean_std(row["iou_mean"], row["iou_std"]),
                small=row["small_lesion_dice"],
                params=row["params_m"],
                epoch=epoch_time,
            )
        )
    if comparison is not None:
        lines += [
            "",
            f"## Paired comparison: {comparison['config_b']} vs {comparison['config_a']}",
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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Namespace with ``runs_dir``, ``out_dir``, ``compare``, ``bootstrap``
        and ``bootstrap_seed``.
    """
    parser = argparse.ArgumentParser(description="Aggregate evaluated runs and optionally compare two configs.")
    parser.add_argument("--runs-dir", required=True, type=Path, help="Directory containing run folders")
    parser.add_argument("--out-dir", type=Path, default=None, help="Where to write the tables (default: runs-dir)")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("CONFIG_A", "CONFIG_B"),
        default=None,
        help="Optional paired per-image comparison of B against A",
    )
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
        print(f"ERROR: no run with eval_test.json + per_image_test.csv under {args.runs_dir}", file=sys.stderr)
        return 1

    groups = group_by_config(runs)
    summary = summarize_groups(groups)
    comparison = None
    if args.compare is not None:
        config_a, config_b = args.compare
        try:
            comparison = paired_comparison(
                groups, config_a, config_b, n_boot=args.bootstrap, seed=args.bootstrap_seed
            )
        except (KeyError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    out_dir = args.out_dir if args.out_dir is not None else Path(args.runs_dir)
    md_path = write_aggregate_markdown(out_dir / "aggregate_results.md", summary, comparison)
    csv_path = write_aggregate_csv(out_dir / "aggregate_results.csv", summary)

    for row in summary:
        print(
            f"{row['config']}: {row['n_seeds']} seed(s) "
            f"dice {_format_mean_std(row['dice_mean'], row['dice_std'])} "
            f"iou {_format_mean_std(row['iou_mean'], row['iou_std'])}"
        )
    if comparison is not None:
        print(
            f"paired {comparison['config_b']} - {comparison['config_a']}: "
            f"{comparison['mean_diff']:+.4f} "
            f"(95% CI {comparison['ci_low']:+.4f} to {comparison['ci_high']:+.4f})"
        )
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
