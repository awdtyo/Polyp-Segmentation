"""Evaluate a finished run on a split and log one row per run/split.

Usage:
    python -m src.evaluate --run_dir runs/exp005_hybrid_fusion
    python -m src.evaluate --run_dir runs/exp005_hybrid_fusion --split test
    python -m src.evaluate --run_dir runs/exp005_hybrid_fusion --checkpoint last \
        --results_csv /content/drive/MyDrive/polyp-runs/results.csv

The run folder must contain ``config_resolved.yaml``, ``best.pt`` (or
``last.pt``) and, for provenance, ``run_info.json``. The model is rebuilt from
the resolved config and the checkpoint weights are loaded; pretrained timm
weights are never downloaded here because the checkpoint already holds them.

Outputs, all inside the run folder:

* ``eval_<split>.json``: mean/std Dice, mean/std IoU, n, small-lesion Dice and
  the provenance needed to reproduce the row.
* ``per_image_<split>.csv``: ``id, dice, iou, lesion_pixels, small_lesion`` for
  every evaluated image, so small-lesion performance can be analysed later.

One row is appended to the results CSV (default ``results/results.csv`` in the
repo). Existing rows are never reordered or dropped. A duplicate
``(run_name, split)`` is refused unless ``--force`` is given, in which case that
row is replaced in place.

Test-set safety: ``--split test`` prints an explicit line and refuses to run if
any test id also appears in ``train.txt`` or ``val.txt``. The test split is used
only here, never for model selection (AGENTS.md section 2).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from src.datasets import PolypDataset, read_split
from src.losses import build_loss
from src.metrics import summarize_by_size
from src.models.net import build_model, count_parameters
from src.train import evaluate as evaluate_split
from src.utils import ConfigError, load_config, set_seed

#: Columns of the results CSV, in order. ``split`` is needed so a val row and a
#: test row of the same run can coexist and be told apart.
RESULTS_COLUMNS: tuple[str, ...] = (
    "run_name",
    "split",
    "config_file",
    "git_commit",
    "seed",
    "epochs",
    "params_m",
    "best_val_dice",
    "test_dice",
    "test_iou",
    "small_lesion_dice",
    "mean_epoch_time_s",
    "peak_gpu_mem_mb",
)

#: Default results CSV, relative to the repository root.
DEFAULT_RESULTS_CSV = Path("results") / "results.csv"

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


class EvaluateError(RuntimeError):
    """Evaluation cannot proceed (missing files, bad header, overlap, ...)."""


class DuplicateResultError(EvaluateError):
    """The results CSV already has a row for this run name and split."""


class TestSplitOverlapError(EvaluateError):
    """A test id also appears in the train or val split."""


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Namespace with ``run_dir``, ``split``, ``checkpoint``, ``results_csv``
        and ``force``.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained run on one split.")
    parser.add_argument("--run_dir", required=True, type=Path, help="Run folder written by src.train")
    parser.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help="Split to evaluate (default: test; test is used only at the end)",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="Checkpoint to load (default: best)",
    )
    parser.add_argument(
        "--results_csv",
        type=Path,
        default=None,
        help=f"Results CSV to append to (default: {DEFAULT_RESULTS_CSV})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing (run_name, split) row instead of refusing it",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_run_config(run_dir: Path) -> dict[str, Any]:
    """Load ``config_resolved.yaml`` from a run folder.

    Args:
        run_dir: Run folder.

    Returns:
        Resolved config.

    Raises:
        EvaluateError: if the file is missing.
        ConfigError: if the file is not a valid config.
    """
    path = run_dir / "config_resolved.yaml"
    if not path.is_file():
        raise EvaluateError(f"Missing {path}; evaluate needs the run's resolved config")
    return load_config(path)


def read_run_info(run_dir: Path) -> dict[str, Any]:
    """Read ``run_info.json`` if it exists.

    Args:
        run_dir: Run folder.

    Returns:
        The parsed dict, or an empty dict when the file is missing or invalid.
        Evaluation can still proceed; provenance fields then come from the
        checkpoint and the config.
    """
    path = run_dir / "run_info.json"
    if not path.is_file():
        print(f"warning: {path} is missing; provenance will be incomplete", file=sys.stderr)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"warning: could not parse {path} ({exc})", file=sys.stderr)
        return {}


def check_test_isolation(cfg: Mapping[str, Any], split: str) -> None:
    """Print the test-split warning and verify the split files do not overlap.

    Args:
        cfg: Resolved config; ``data.split_dir`` is read.
        split: ``test`` or ``val``. Only ``test`` triggers the check.

    Raises:
        TestSplitOverlapError: if a test id also appears in train or val.
    """
    if split != "test":
        return
    split_dir = Path(cfg["data"]["split_dir"])
    print(
        "TEST SPLIT: evaluating the held-out test split. This split is used only "
        "here, never for model selection; do not tune on these numbers.",
        flush=True,
    )
    test_ids = set(read_split(split_dir / "test.txt"))
    for other in ("train", "val"):
        overlap = sorted(test_ids & set(read_split(split_dir / f"{other}.txt")))
        if overlap:
            preview = ", ".join(overlap[:5])
            more = "" if len(overlap) <= 5 else f" (+{len(overlap) - 5} more)"
            raise TestSplitOverlapError(
                f"{len(overlap)} test ids also appear in {other}.txt: {preview}{more}. "
                "Refusing to report a test number on a leaking split."
            )
    print(f"test isolation check: {len(test_ids)} test ids, no overlap with train or val")


def build_split_loader(cfg: Mapping[str, Any], split: str) -> DataLoader:
    """Build a deterministic loader for one split.

    Mirrors the validation loader in :func:`src.train.build_dataloaders`: no
    augmentation, resize + normalise only, no shuffling.

    Args:
        cfg: Resolved config.
        split: ``train``, ``val`` or ``test``.

    Returns:
        DataLoader over ``<split_dir>/<split>.txt``.

    Raises:
        EvaluateError: if the split file is missing.
    """
    data = cfg["data"]
    split_file = Path(data["split_dir"]) / f"{split}.txt"
    if not split_file.is_file():
        raise EvaluateError(f"Split file not found: {split_file}")
    dataset = PolypDataset(
        split_file=split_file,
        root=data["root"],
        transform=None,
        image_size=int(data["image_size"]),
        image_dir=data["image_dir"],
        mask_dir=data["mask_dir"],
        image_ext=data["image_ext"],
        mask_ext=data["mask_ext"],
        mask_suffix=data["mask_suffix"],
        mask_threshold=data["mask_threshold"],
        cache_path=data.get("cache_path") or None,
        normalize=data["normalize"],
        mean=data["mean"],
        std=data["std"],
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(data["num_workers"]),
        pin_memory=bool(data["pin_memory"]),
        drop_last=False,
    )


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Mean and sample std of per-image Dice/IoU plus the small-lesion summary.

    Args:
        rows: Per-image records from :class:`src.metrics.SegMetrics`.

    Returns:
        ``{"dice_mean", "dice_std", "iou_mean", "iou_std", "n",
        "small_lesion_dice", "small_lesion_n"}``. The std is 0.0 for a single
        image because a sample std is undefined there.
    """
    dice = [float(row["dice"]) for row in rows]
    iou = [float(row["iou"]) for row in rows]
    by_size = summarize_by_size(rows)
    return {
        "dice_mean": statistics.fmean(dice) if dice else 0.0,
        "dice_std": statistics.stdev(dice) if len(dice) > 1 else 0.0,
        "iou_mean": statistics.fmean(iou) if iou else 0.0,
        "iou_std": statistics.stdev(iou) if len(iou) > 1 else 0.0,
        "n": float(len(rows)),
        "small_lesion_dice": float(by_size["dice_small"]),
        "small_lesion_n": float(by_size["n_small"]),
    }


def write_eval_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write the evaluation summary as JSON.

    Args:
        path: Destination file; parents are created.
        payload: JSON-serialisable mapping.

    Returns:
        The written path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return out


def write_per_image_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write one row per image: id, dice, iou, lesion_pixels, small_lesion.

    Args:
        path: Destination file; parents are created.
        rows: Per-image records from :class:`src.metrics.SegMetrics`.

    Returns:
        The written path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ("id", "dice", "iou", "lesion_pixels", "small_lesion")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "dice": round(float(row["dice"]), 6),
                    "iou": round(float(row["iou"]), 6),
                    "lesion_pixels": int(float(row["lesion_pixels"])),
                    "small_lesion": int(bool(row["small_lesion"])),
                }
            )
    return out


def append_results_row(
    path: str | Path,
    row: Mapping[str, Any],
    *,
    force: bool = False,
) -> Path:
    """Append (or, with ``force``, replace) one row in the results CSV.

    Args:
        path: Results CSV.
        row: Values for every column in :data:`RESULTS_COLUMNS`.
        force: Replace an existing ``(run_name, split)`` row instead of raising.

    Returns:
        The written path.

    Raises:
        DuplicateResultError: if the row exists and ``force`` is false.
        EvaluateError: if an existing file has a different header.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if out.is_file():
        with out.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(RESULTS_COLUMNS):
                raise EvaluateError(
                    f"{out} has an incompatible header {reader.fieldnames}; "
                    f"expected {list(RESULTS_COLUMNS)}. Fix or move the file; rows are never rewritten."
                )
            rows = [dict(existing) for existing in reader]

    new_row = {column: str(row[column]) for column in RESULTS_COLUMNS}
    duplicate_index = next(
        (
            index
            for index, existing in enumerate(rows)
            if existing["run_name"] == new_row["run_name"] and existing["split"] == new_row["split"]
        ),
        None,
    )
    if duplicate_index is not None:
        if not force:
            raise DuplicateResultError(
                f"{out} already has a row for run {new_row['run_name']!r} split {new_row['split']!r}. "
                "Rerun with --force to replace it, or use a different run folder."
            )
        rows[duplicate_index] = new_row
    else:
        rows.append(new_row)

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULTS_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return out


def _checkpoint_epochs(checkpoint: Mapping[str, Any], cfg: Mapping[str, Any]) -> int:
    """Completed epoch count: checkpoint epoch + 1, else the configured epochs.

    Args:
        checkpoint: Loaded checkpoint dict.
        cfg: Resolved config.

    Returns:
        Number of completed epochs, at least 0.
    """
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, int) and epoch >= 0:
        return epoch + 1
    return int(cfg["train"]["epochs"])


def main(argv: Iterable[str] | None = None) -> int:
    """Evaluate one run and append one results row.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 on an evaluation failure, 2 on bad
        input/config.
    """
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: run folder not found: {run_dir}", file=sys.stderr)
        return EXIT_USAGE

    try:
        cfg = load_run_config(run_dir)
    except (ConfigError, EvaluateError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        check_test_isolation(cfg, args.split)
    except TestSplitOverlapError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    checkpoint_path = run_dir / f"{args.checkpoint}.pt"
    if not checkpoint_path.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return EXIT_USAGE

    run_info = read_run_info(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg["run"]["seed"]))

    # Pretrained weights are irrelevant here: the checkpoint supplies every
    # weight. Turning them off keeps evaluation offline and download-free.
    model_cfg = copy.deepcopy(dict(cfg))
    model_cfg["model"] = {
        **dict(cfg["model"]),
        "pretrained_a": False,
        "pretrained_b": False,
    }
    model = build_model(model_cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    try:
        loader = build_split_loader(cfg, args.split)
    except EvaluateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE

    loss_fn = build_loss(cfg).to(device)
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    started = time.time()
    # train.evaluate is the same per-image Dice/IoU code used during training.
    result = evaluate_split(model, loader, loss_fn, device, use_amp=use_amp)
    elapsed = time.time() - started

    rows = result["per_image"]
    summary = summarize_rows(rows)
    total_params = int(run_info.get("params_total") or count_parameters(model)[0])
    payload = {
        "run_name": cfg["run"]["name"],
        "split": args.split,
        "checkpoint": args.checkpoint,
        "config_file": str(run_info.get("config_file") or run_dir / "config_resolved.yaml"),
        "git_commit": run_info.get("git_commit"),
        "seed": int(cfg["run"]["seed"]),
        "epochs": _checkpoint_epochs(checkpoint, cfg),
        "image_size": int(cfg["data"]["image_size"]),
        "threshold": 0.5,
        "device": str(device),
        "n": int(summary["n"]),
        "dice_mean": summary["dice_mean"],
        "dice_std": summary["dice_std"],
        "iou_mean": summary["iou_mean"],
        "iou_std": summary["iou_std"],
        "small_lesion_dice": summary["small_lesion_dice"],
        "small_lesion_n": int(summary["small_lesion_n"]),
        "params_total": total_params,
        "params_m": round(total_params / 1e6, 6),
        "best_val_dice": float(checkpoint.get("best_val_dice", float("nan"))),
        "mean_epoch_time_s": run_info.get("mean_epoch_time_s"),
        "peak_gpu_mem_mb": run_info.get("peak_gpu_mem_mb"),
        "eval_seconds": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    json_path = write_eval_json(run_dir / f"eval_{args.split}.json", payload)
    per_image_path = write_per_image_csv(run_dir / f"per_image_{args.split}.csv", rows)

    results_row = {
        "run_name": payload["run_name"],
        "split": args.split,
        "config_file": payload["config_file"],
        "git_commit": payload["git_commit"] or "",
        "seed": payload["seed"],
        "epochs": payload["epochs"],
        "params_m": f"{payload['params_m']:.3f}",
        "best_val_dice": f"{payload['best_val_dice']:.6f}",
        "test_dice": f"{summary['dice_mean']:.6f}",
        "test_iou": f"{summary['iou_mean']:.6f}",
        "small_lesion_dice": f"{summary['small_lesion_dice']:.6f}",
        "mean_epoch_time_s": "" if payload["mean_epoch_time_s"] is None else f"{float(payload['mean_epoch_time_s']):.2f}",
        "peak_gpu_mem_mb": "" if payload["peak_gpu_mem_mb"] is None else f"{float(payload['peak_gpu_mem_mb']):.1f}",
    }
    results_csv = Path(args.results_csv) if args.results_csv is not None else DEFAULT_RESULTS_CSV
    try:
        append_results_row(results_csv, results_row, force=bool(args.force))
    except (DuplicateResultError, EvaluateError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    print(
        f"{payload['run_name']} [{args.split}] n={payload['n']} "
        f"dice {summary['dice_mean']:.4f} +/- {summary['dice_std']:.4f} "
        f"iou {summary['iou_mean']:.4f} +/- {summary['iou_std']:.4f} "
        f"small-lesion dice {summary['small_lesion_dice']:.4f} ({int(summary['small_lesion_n'])} images)"
    )
    print(f"wrote {json_path}")
    print(f"wrote {per_image_path}")
    print(f"appended {results_csv}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
