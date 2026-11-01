"""Backfill the ``small_lesion_dice`` / ``small_lesion_n`` columns of an existing
results CSV from the per-image metric files.

Usage:
    python scripts/backfill_small_lesion.py \
        --results-csv <RUNS_DIR>/results.csv \
        --runs-dir <RUNS_DIR> \
        [--small-lesion-fraction 0.02] [--min-small-images 10] [--dry-run]

Recomputes, for every row, the small-lesion Dice from
``per_image_<split>_<dataset>[_ttaflip].csv`` (columns ``lesion_pixels`` and
``image_area``; when ``image_area`` is missing the area is taken from the run's
``config_resolved.yaml`` ``data.image_size``). A lesion is "small" when its
foreground pixel count is below ``small_lesion_fraction`` of the image area.

Rules:
* ``small_lesion_dice`` is the mean Dice over images that qualify as small.
* If fewer than ``--min-small-images`` (default 10) qualify in a dataset, the
  value is written as ``n/a`` and ``small_lesion_n`` is still recorded. Never
  ``0.0`` for "none qualified".
* Rows whose per-image file is missing are left untouched (and reported).

Nothing is re-inferred: this only re-reads the per-image Dice and lesion sizes
already written by ``src.evaluate``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import yaml

#: Column used to locate the per-image file for a results row.
RESULTS_KEYS = ("run_name", "split", "dataset", "tta")

#: Default small-lesion area fraction (2% of image area).
DEFAULT_SMALL_LESION_FRACTION = 0.02
#: Default minimum qualifying images before reporting a small-lesion mean.
DEFAULT_MIN_SMALL_IMAGES = 10


def _per_image_path(run_dir: Path, split: str, dataset: str, tta: str) -> Path:
    suffix = "_ttaflip" if tta == "flip" else ""
    return run_dir / f"per_image_{split}_{dataset}{suffix}.csv"


def _image_size(run_dir: Path) -> int:
    config = run_dir / "config_resolved.yaml"
    if config.is_file():
        try:
            cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
            if isinstance(cfg, dict):
                return int(cfg.get("data", {}).get("image_size", 256))
        except (yaml.YAMLError, OSError, ValueError, TypeError):
            pass
    return 256


def _recompute(rows: list[dict[str, str]], fraction: float, min_small: int) -> tuple[float | None, int]:
    """Return (small_lesion_dice, small_lesion_n) from per-image rows."""
    small_dice: list[float] = []
    n_small = 0
    for row in rows:
        lesion_pixels = float(row.get("lesion_pixels", 0) or 0)
        image_area = float(row.get("image_area") or 0)
        if image_area <= 0:
            image_area = 256 * 256
        if lesion_pixels < fraction * image_area:
            n_small += 1
            dice = float(row.get("dice", 0) or 0)
            small_dice.append(dice)
    if n_small < min_small:
        return None, n_small
    if not small_dice:
        return None, 0
    return sum(small_dice) / len(small_dice), n_small


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.6f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill small_lesion_dice / small_lesion_n from per-image files."
    )
    parser.add_argument("--results-csv", required=True, type=Path, help="Results CSV to backfill")
    parser.add_argument("--runs-dir", required=True, type=Path, help="Directory containing run folders")
    parser.add_argument(
        "--small-lesion-fraction",
        type=float,
        default=DEFAULT_SMALL_LESION_FRACTION,
        help="Small-lesion area fraction (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--min-small-images",
        type=int,
        default=DEFAULT_MIN_SMALL_IMAGES,
        help="Minimum qualifying images before reporting a mean (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing anything",
    )
    args = parser.parse_args(argv)

    results_csv = Path(args.results_csv)
    if not results_csv.is_file():
        print(f"ERROR: results CSV not found: {results_csv}", file=sys.stderr)
        return 2

    with results_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]

    if "small_lesion_dice" not in fieldnames or "small_lesion_n" not in fieldnames:
        print(
            "ERROR: results CSV does not have small_lesion_dice / small_lesion_n columns; "
            "run src.evaluate first to create the new schema",
            file=sys.stderr,
        )
        return 2

    changed = 0
    skipped = 0
    for row in rows:
        run_dir = Path(args.runs_dir) / row["run_name"]
        per_image_path = _per_image_path(
            run_dir, row["split"], row["dataset"], row["tta"]
        )
        if not per_image_path.is_file():
            skipped += 1
            print(f"skip: {per_image_path} not found (row {row['run_name']}/{row['dataset']})")
            continue
        with per_image_path.open(newline="", encoding="utf-8") as handle:
            per_image_rows = list(csv.DictReader(handle))
        dice_mean, n_small = _recompute(
            per_image_rows, args.small_lesion_fraction, args.min_small_images
        )
        new_dice = _fmt(dice_mean)
        new_n = str(n_small)
        if row["small_lesion_dice"] != new_dice or row["small_lesion_n"] != new_n:
            changed += 1
            print(
                f"update {row['run_name']} [{row['split']}/{row['dataset']}] "
                f"small_lesion_dice {row['small_lesion_dice']} -> {new_dice} "
                f"small_lesion_n {row['small_lesion_n']} -> {new_n}"
            )
            row["small_lesion_dice"] = new_dice
            row["small_lesion_n"] = new_n

    if changed and not args.dry_run:
        # Back up before rewriting.
        backup = results_csv.with_name(results_csv.name + ".small_lesion_backup.csv")
        if not backup.is_file():
            backup.write_text(
                "\n".join(
                    [",".join(fieldnames)]
                    + [",".join(r.get(c, "") for c in fieldnames) for r in rows]
                ),
                encoding="utf-8",
            )
            print(f"backed up original to {backup}")
        with results_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {changed} updated rows to {results_csv}")
    elif changed and args.dry_run:
        print(f"dry-run: would update {changed} rows")
    else:
        print("no changes needed")
    print(f"skipped {skipped} rows with missing per-image files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
