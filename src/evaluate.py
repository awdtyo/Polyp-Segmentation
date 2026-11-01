"""Evaluate a finished run on one or more datasets and log one row per dataset.

Usage:
    python -m src.evaluate --run_dir runs/exp005_hybrid_fusion
    python -m src.evaluate --run_dir runs/exp005_hybrid_fusion --split val --dataset kvasir_seg
    python -m src.evaluate --run_dir runs/exp005_hybrid_fusion --split test \\
        --dataset kvasir_seg --dataset cvc_clinicdb --dataset cvc_colondb
    python -m src.evaluate --run_dir runs/exp005_hybrid_fusion --tta-flip

The run folder must contain ``config_resolved.yaml``, ``best.pt`` (or
``last.pt``) and, for provenance, ``run_info.json``. The model is rebuilt from
the resolved config and the checkpoint weights are loaded; pretrained timm
weights are never downloaded here because the checkpoint already holds them.

Two kinds of dataset are handled:

* the **training dataset** (``data.dataset``, ``kvasir_seg``) is read through the
  committed split files, exactly as during training, so its ``val`` Dice
  reproduces the training log;
* every **other** dataset is read zero-shot from
  ``<DATA_ROOT>/processed/<name>/data.npz`` (see ``scripts/prepare_datasets.py``),
  using all of its images as held-out data.

The checkpoint is always the one chosen by Kvasir **val** Dice (``best.pt``);
external datasets are never used to select or tune a checkpoint.

## How the numbers are defined

* Images and masks live at ``data.image_size`` (256 px) after a bilinear image
  resize and a nearest-neighbour mask resize, with the masks binarised at
  ``> data.mask_threshold`` *before* any resizing (the same rule the Kvasir cache
  uses, AGENTS.md section 11).
* Dice and IoU are computed **per image** on the 0.5-thresholded sigmoid output
  and then averaged; the threshold is strictly greater than 0.5. An image whose
  ground truth is empty scores 1.0 when the prediction is empty too (the usual
  smoothed-ratio convention, ``src/metrics.py``).
* MAE is the mean absolute error between the predicted probability map and the
  binary ground truth, per image then averaged (the convention used in polyp
  segmentation papers). It is threshold-free.
* Flip TTA is off by default. When ``--tta-flip`` is given the model is also run
  on the horizontally flipped image, the flip is undone and the two probability
  maps are averaged before thresholding. Its outputs are written under a
  different name (``*_ttaflip``) so a TTA run never overwrites a plain one.

Outputs, all inside the run folder, one set per evaluated dataset:

* ``eval_<split>_<dataset>.json``: mean/std Dice, IoU and MAE, n, small-lesion
  Dice and the provenance needed to reproduce the row.
* ``per_image_<split>_<dataset>.csv``:
  ``id, dice, iou, mae, lesion_pixels, small_lesion`` for every image.

One row per ``(run_name, split, dataset, tta)`` is appended to the results CSV
(default ``results/results.csv`` in the repo). Existing rows are never reordered
or dropped. A duplicate is refused unless ``--force``, which replaces that row in
place.

Test-set safety: the training dataset's ``test`` split prints an explicit line
and refuses to run if any test id also appears in ``train.txt`` or ``val.txt``.
The test split is used only here, never for model selection (AGENTS.md section
2).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_registry import DatasetSpec, RegistryError, get_spec, resolve_data_root
from src.datasets import PolypDataset, read_split
from src.metrics import SegMetrics, SMALL_LESION_AREA_FRACTION, summarize_by_size
from src.models.net import build_model, count_parameters
from src.utils import ConfigError, load_config, set_seed

#: Columns of the results CSV, in order. ``split``, ``dataset`` and ``tta`` are
#: needed so every variation of a run can coexist and be told apart.
#:
#: ``best_epoch`` is the epoch at which ``best.pt`` was saved (the model-selection
#: signal) and ``epochs_trained`` is the total number of epochs run, read from the
#: run's ``metrics.csv``. The old ``epochs`` column held the best epoch and is
#: retired: new files use ``best_epoch`` / ``epochs_trained`` instead. Legacy
#: files that still carry ``epochs`` are migrated once on first append (see
#: :func:`_migrate_legacy_results_csv`); ``epochs` is then read as ``best_epoch``
#: by the reader and the aggregator.
#: ``config_name`` records the experiment identity (``cfg["run"]["name"]``) so
#: aggregation can group by experiment instead of by the session config path.
#: ``small_lesion_n`` records how many images qualified as small lesions.
RESULTS_COLUMNS: tuple[str, ...] = (
    "run_name",
    "split",
    "dataset",
    "tta",
    "config_name",
    "config_file",
    "git_commit",
    "seed",
    "best_epoch",
    "epochs_trained",
    "params_m",
    "best_val_dice",
    "test_dice",
    "test_iou",
    "test_mae",
    "small_lesion_dice",
    "small_lesion_n",
    "mean_epoch_time_s",
    "peak_gpu_mem_mb",
)

#: Header of results CSV files written before the schema migration. The only
#: difference from :data:`RESULTS_COLUMNS` is that ``epochs`` (the best epoch)
#: appears where ``best_epoch`` / ``epochs_trained`` now live, and
#: ``config_name`` / ``small_lesion_n`` are absent.
LEGACY_RESULTS_COLUMNS: tuple[str, ...] = (
    "run_name",
    "split",
    "dataset",
    "tta",
    "config_file",
    "git_commit",
    "seed",
    "epochs",
    "params_m",
    "best_val_dice",
    "test_dice",
    "test_iou",
    "test_mae",
    "small_lesion_dice",
    "mean_epoch_time_s",
    "peak_gpu_mem_mb",
)

#: Default results CSV, relative to the repository root.
DEFAULT_RESULTS_CSV = Path("results") / "results.csv"

#: Backup file written next to a legacy results CSV before its one-time
#: migration to the current header.
LEGACY_BACKUP_NAME = "results.legacy.csv"

#: Seed-suffix patterns stripped from a run name to obtain the experiment
#: identity used for grouping (``exp005_seed43`` → ``exp005``).
SEED_SUFFIX_PATTERNS = (r"_seed\d+$", r"_s\d+$")


def experiment_key(run_name: str) -> str:
    """Return the experiment identity for a run name.

    Trailing seed suffixes (``_seed<N>`` and ``_s<N>``) are stripped so that the
    seeds of one experiment group together and are never merged with a different
    experiment that happens to share a session config path.

    Args:
        run_name: Value of the ``run_name`` column (equals ``cfg["run"]["name"]``).

    Returns:
        The experiment name with any trailing seed suffix removed.
    """
    import re

    key = run_name
    for pattern in SEED_SUFFIX_PATTERNS:
        key = re.sub(pattern, "", key)
    return key


def _is_legacy_results_header(header: Sequence[str]) -> bool:
    """True when ``header`` is exactly the pre-migration results schema."""
    return list(header) == list(LEGACY_RESULTS_COLUMNS)


def _is_current_results_header(header: Sequence[str]) -> bool:
    """True when ``header`` is exactly the current results schema."""
    return list(header) == list(RESULTS_COLUMNS)


def _migrate_legacy_results_csv(path: Path) -> None:
    """Back up and rewrite a legacy results CSV to the current schema.

    The legacy file is copied to ``results.legacy.csv`` next to it (only the
    first migration creates the backup). The original is rewritten with the
    current header: every existing column is copied unchanged, the old
    ``epochs`` column becomes ``best_epoch``, and the new columns
    (``config_name``, ``epochs_trained``, ``small_lesion_n``) are left blank.

    Args:
        path: Path to the legacy results CSV.
    """
    backup = path.parent / LEGACY_BACKUP_NAME
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        legacy_columns = list(reader.fieldnames)
        rows = [dict(existing) for existing in reader]

    if not backup.is_file():
        backup.write_text(
            "\n".join(
                [",".join(legacy_columns)]
                + [",".join(row.get(column, "") for column in legacy_columns) for row in rows]
            ),
            encoding="utf-8",
        )
        print(
            f"Migrating legacy results CSV to the current schema; "
            f"legacy file backed up to {backup}",
            flush=True,
        )

    new_rows: list[dict[str, str]] = []
    for row in rows:
        new_row: dict[str, str] = {}
        for column in RESULTS_COLUMNS:
            if column == "best_epoch":
                new_row[column] = row.get("epochs", "")
            elif column in LEGACY_RESULTS_COLUMNS:
                new_row[column] = row.get(column, "")
            else:
                new_row[column] = ""
        new_rows.append(new_row)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULTS_COLUMNS))
        writer.writeheader()
        writer.writerows(new_rows)

#: Dataset evaluated through the committed split files when ``--dataset`` is not
#: given. Every other dataset is read zero-shot from ``processed/``.
DEFAULT_DATASET = "kvasir_seg"

#: File names inside ``<DATA_ROOT>/processed/<name>/``.
PROCESSED_DATA_FILENAME = "data.npz"
PROCESSED_META_FILENAME = "meta.json"

#: Infix added to output file names for a flip-TTA run.
TTA_SUFFIX = "_ttaflip"
#: Decision threshold on the sigmoid output, fixed everywhere in the repo.
THRESHOLD = 0.5

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


class EvaluateError(RuntimeError):
    """Evaluation cannot proceed (missing files, bad header, overlap, ...)."""


class DuplicateResultError(EvaluateError):
    """The results CSV already has a row for this run name, split, dataset and TTA."""


class TestSplitOverlapError(EvaluateError):
    """A test id also appears in the train or val split."""


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Namespace with ``run_dir``, ``split``, ``checkpoint``, ``dataset``,
        ``tta_flip``, ``config``, ``results_csv`` and ``force``.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained run on one or more datasets.")
    parser.add_argument("--run_dir", required=True, type=Path, help="Run folder written by src.train")
    parser.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help="Split for the training dataset (default: test; test is used only at the end). "
        "External datasets always use all their images and are recorded as 'test'.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="Checkpoint to load (default: best, i.e. the Kvasir val Dice optimum)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="NAME",
        help=f"Dataset to evaluate (repeatable). Default: {DEFAULT_DATASET}. Other names are read "
        "from <DATA_ROOT>/processed/<name>/ and must be prepared first.",
    )
    parser.add_argument(
        "--tta-flip",
        action="store_true",
        help="Average each prediction with a horizontal-flip pass (off by default). "
        "Outputs are written with a '_ttaflip' suffix.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override the run's config_resolved.yaml (data paths only in practice). Use when the "
        "training data lived at paths that no longer exist, e.g. a fresh Colab session.",
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
        help="Replace an existing (run_name, split, dataset, tta) row instead of refusing it",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_run_config(run_dir: Path, override: Path | None = None) -> dict[str, Any]:
    """Load ``config_resolved.yaml`` from a run folder, or an override file.

    Args:
        run_dir: Run folder.
        override: Optional config to read instead of the run's own
            ``config_resolved.yaml``. Used when the training data lived at paths
            that no longer exist (e.g. a fresh Colab session); the architecture
            should be the run's own, so only data paths are expected to differ.

    Returns:
        Resolved config.

    Raises:
        EvaluateError: if the file is missing.
        ConfigError: if the file is not a valid config.
    """
    path = Path(override) if override is not None else run_dir / "config_resolved.yaml"
    if not path.is_file():
        if override is not None:
            raise EvaluateError(f"Config override not found: {path}")
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


def _build_loader(dataset: PolypDataset, cfg: Mapping[str, Any]) -> DataLoader:
    """Wrap a dataset in the deterministic loader used for val/test.

    Args:
        dataset: The dataset to iterate.
        cfg: Resolved config.

    Returns:
        A non-shuffling, non-dropping DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=False,
    )


def build_split_loader(cfg: Mapping[str, Any], split: str) -> DataLoader:
    """Build a deterministic loader for one split of the training dataset.

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
    return _build_loader(dataset, cfg)


def read_processed_meta(processed_dir: str | Path) -> dict[str, Any]:
    """Read ``meta.json`` from a processed dataset folder, if present.

    Args:
        processed_dir: ``<DATA_ROOT>/processed/<name>``.

    Returns:
        The parsed dict, or an empty dict when the file is missing or unparseable.
    """
    path = Path(processed_dir) / PROCESSED_META_FILENAME
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def build_processed_loader(cfg: Mapping[str, Any], spec: DatasetSpec, data_root: str | Path) -> DataLoader:
    """Build a loader for a dataset prepared by ``scripts/prepare_datasets.py``.

    All images of the dataset are used (external datasets have no train/val/test
    split), read from ``processed/<name>/data.npz``.

    Args:
        cfg: Resolved config (image size, normalisation, batch size).
        spec: Registry entry.
        data_root: Data-tree root.

    Returns:
        DataLoader over every image in the processed cache.

    Raises:
        EvaluateError: if the processed cache is missing or malformed.
    """
    processed = spec.processed_dir(data_root)
    npz_path = processed / PROCESSED_DATA_FILENAME
    if not npz_path.is_file():
        raise EvaluateError(
            f"Processed data for {spec.name!r} not found: {npz_path}. "
            f"Build it with: python scripts/prepare_datasets.py --dataset {spec.name}"
        )
    with np.load(npz_path) as data:
        if "ids" not in data.files:
            raise EvaluateError(f"{npz_path} has no 'ids' array; rebuild it with scripts/prepare_datasets.py")
        ids = [str(value) for value in data["ids"]]

    meta = read_processed_meta(processed)
    run_size = int(cfg["data"]["image_size"])
    native_size = int(meta.get("image_size", run_size))
    if native_size != run_size:
        print(
            f"warning: {spec.name} was prepared at {native_size}px but the run uses {run_size}px; "
            f"evaluating at the prepared size {native_size}px.",
            file=sys.stderr,
        )

    dataset = PolypDataset(
        ids=ids,
        root=processed,
        image_size=native_size,
        cache_path=npz_path,
        normalize=cfg["data"]["normalize"],
        mean=cfg["data"]["mean"],
        std=cfg["data"]["std"],
    )
    return _build_loader(dataset, cfg)


def resolve_datasets(names: Sequence[str] | None) -> list[DatasetSpec]:
    """Look up the requested datasets in the registry, preserving order.

    Args:
        names: Requested names, or ``None`` for the default dataset.

    Returns:
        The matching specs, de-duplicated, in request order.

    Raises:
        EvaluateError: if a name is not in the registry.
    """
    ordered = list(names) if names else [DEFAULT_DATASET]
    specs: list[DatasetSpec] = []
    seen: set[str] = set()
    for name in ordered:
        try:
            spec = get_spec(name)
        except RegistryError as exc:
            raise EvaluateError(str(exc)) from exc
        if spec.name not in seen:
            seen.add(spec.name)
            specs.append(spec)
    return specs


def build_dataset_loader(
    cfg: Mapping[str, Any], spec: DatasetSpec, split: str, *, data_root: str | Path
) -> DataLoader:
    """Build the right loader for one dataset.

    The training dataset (``data.dataset``) is read through the committed split
    files; every other dataset is read zero-shot from its processed cache.

    Args:
        cfg: Resolved config.
        spec: Registry entry.
        split: Split for the training dataset.
        data_root: Data-tree root.

    Returns:
        A DataLoader over the dataset.
    """
    train_dataset = str(cfg["data"].get("dataset", DEFAULT_DATASET))
    if spec.name == train_dataset:
        return build_split_loader(cfg, split)
    return build_processed_loader(cfg, spec, data_root)


def evaluate_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    tta_flip: bool = False,
    use_amp: bool = False,
    small_lesion_area_fraction: float = SMALL_LESION_AREA_FRACTION,
) -> list[dict[str, float | str | bool]]:
    """Run a model over a loader and return one metric row per image.

    Args:
        model: The model to evaluate (set to eval mode here).
        loader: Deterministic loader of ``{"image", "mask", "id"}`` batches.
        device: Torch device.
        tta_flip: Also run the horizontally flipped image and average the two
            probability maps before thresholding.
        use_amp: Run the forward pass under ``torch.autocast`` (CUDA only in
            practice).
        small_lesion_area_fraction: Fraction of image area below which a lesion
            counts as "small" (default: 2%).

    Returns:
        Per-image records ``{"id", "dice", "iou", "mae", "lesion_pixels",
        "small_lesion"}``.
    """
    metrics = SegMetrics(
        threshold=THRESHOLD,
        from_logits=False,
        small_lesion_area_fraction=small_lesion_area_fraction,
    )
    mae_values: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                probs = torch.sigmoid(model(images))
                if tta_flip:
                    flipped = torch.sigmoid(model(torch.flip(images, dims=[3])))
                    probs = 0.5 * (probs + torch.flip(flipped, dims=[3]))
            probs = probs.float()
            mae_values.extend((probs - targets).abs().mean(dim=(1, 2, 3)).tolist())
            metrics.update(probs, targets, ids=list(batch["id"]))

    rows = list(metrics.per_image())
    if len(rows) != len(mae_values):
        raise EvaluateError(f"Collected {len(rows)} metric rows but {len(mae_values)} MAE values")
    for row, mae in zip(rows, mae_values):
        row["mae"] = float(mae)
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Mean and sample std of per-image Dice/IoU/MAE plus the small-lesion summary.

    Args:
        rows: Per-image records from :func:`evaluate_dataset`.

    Returns:
        ``{"dice_mean", "dice_std", "iou_mean", "iou_std", "mae_mean", "mae_std",
        "n", "small_lesion_dice", "small_lesion_n"}``. The std is 0.0 for a single
        image because a sample std is undefined there.
    """
    dice = [float(row["dice"]) for row in rows]
    iou = [float(row["iou"]) for row in rows]
    mae = [float(row["mae"]) for row in rows]
    by_size = summarize_by_size(rows)
    n_small = int(by_size["n_small"])
    dice_small = float(by_size["dice_small"])
    # Fewer than 10 qualifying images: no reliable small-lesion estimate; never
    # report 0.0 for "none qualified".
    if n_small < 10:
        dice_small = float("nan")
    return {
        "dice_mean": statistics.fmean(dice) if dice else 0.0,
        "dice_std": statistics.stdev(dice) if len(dice) > 1 else 0.0,
        "iou_mean": statistics.fmean(iou) if iou else 0.0,
        "iou_std": statistics.stdev(iou) if len(iou) > 1 else 0.0,
        "mae_mean": statistics.fmean(mae) if mae else 0.0,
        "mae_std": statistics.stdev(mae) if len(mae) > 1 else 0.0,
        "n": float(len(rows)),
        "small_lesion_dice": dice_small,
        "small_lesion_n": float(n_small),
    }


def output_names(split: str, dataset: str, *, tta_flip: bool) -> tuple[str, str]:
    """File names for one (split, dataset, TTA) evaluation.

    Args:
        split: ``test`` or ``val``.
        dataset: Dataset name.
        tta_flip: Whether flip TTA was used.

    Returns:
        ``(json_name, per_image_csv_name)``.
    """
    suffix = TTA_SUFFIX if tta_flip else ""
    return (f"eval_{split}_{dataset}{suffix}.json", f"per_image_{split}_{dataset}{suffix}.csv")


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
    """Write one row per image: id, dice, iou, mae, lesion_pixels, small_lesion.

    Args:
        path: Destination file; parents are created.
        rows: Per-image records from :func:`evaluate_dataset`.

    Returns:
        The written path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ("id", "dice", "iou", "mae", "lesion_pixels", "image_area", "small_lesion")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "dice": round(float(row["dice"]), 6),
                    "iou": round(float(row["iou"]), 6),
                    "mae": round(float(row["mae"]), 6),
                    "lesion_pixels": int(float(row["lesion_pixels"])),
                    "image_area": int(float(row.get("image_area", 0))),
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
        force: Replace an existing ``(run_name, split, dataset, tta)`` row
            instead of raising.

    Returns:
        The written path.

    Raises:
        DuplicateResultError: if the row exists and ``force`` is false.
        EvaluateError: if an existing file has a different header.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    existing_header: list[str] = []
    if out.is_file():
        with out.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_header = list(reader.fieldnames) if reader.fieldnames else []
            rows = [dict(existing) for existing in reader]

    if _is_legacy_results_header(existing_header):
        # One-time migration: back up the legacy file, rewrite it with the
        # current schema, then append normally below. Existing values are
        # preserved; new columns start blank (best_epoch comes from the old
        # ``epochs`` column). Rows were read above before the rewrite, so they
        # still carry legacy keys and are remapped after migration.
        _migrate_legacy_results_csv(out)
        existing_header = list(RESULTS_COLUMNS)
        remapped: list[dict[str, str]] = []
        for legacy in rows:
            new_row: dict[str, str] = {}
            for column in RESULTS_COLUMNS:
                if column == "best_epoch":
                    new_row[column] = legacy.get("epochs", "")
                elif column in LEGACY_RESULTS_COLUMNS:
                    new_row[column] = legacy.get(column, "")
                else:
                    new_row[column] = ""
            remapped.append(new_row)
        rows = remapped

    if out.is_file() and existing_header != list(RESULTS_COLUMNS):
        raise EvaluateError(
            f"{out} has an incompatible header {existing_header}; "
            f"expected {list(RESULTS_COLUMNS)}. Fix or move the file; rows are never rewritten."
        )

    key = ("run_name", "split", "dataset", "tta")
    new_row = {column: str(row[column]) for column in RESULTS_COLUMNS}
    duplicate_index = next(
        (
            index
            for index, existing in enumerate(rows)
            if all(existing[column] == new_row[column] for column in key)
        ),
        None,
    )
    if duplicate_index is not None:
        if not force:
            described = ", ".join(f"{column}={new_row[column]!r}" for column in key)
            raise DuplicateResultError(
                f"{out} already has a row for {described}. "
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


def _epochs_trained(run_dir: Path) -> int | None:
    """Total epochs completed, read from the run's ``metrics.csv``.

    Args:
        run_dir: Run folder written by :mod:`src.train`.

    Returns:
        The number of data rows in ``metrics.csv`` (one per epoch), or ``None``
        when the file is missing.
    """
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # header
            return sum(1 for _ in reader)
    except (OSError, StopIteration):
        return None


def main(argv: Iterable[str] | None = None) -> int:
    """Evaluate one run on one or more datasets and append one row per dataset.

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
        cfg = load_run_config(run_dir, override=args.config)
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

    data_root = resolve_data_root(str(cfg["data"].get("data_root", "")))
    try:
        specs = resolve_datasets(args.dataset)
    except EvaluateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
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

    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    total_params = int(run_info.get("params_total") or count_parameters(model)[0])
    train_dataset = str(cfg["data"].get("dataset", DEFAULT_DATASET))
    tta_label = "flip" if args.tta_flip else ""
    results_csv = Path(args.results_csv) if args.results_csv is not None else DEFAULT_RESULTS_CSV

    for spec in specs:
        # The training dataset keeps the requested split; external datasets are
        # wholly held-out data and are always recorded as 'test'.
        row_split = args.split if spec.name == train_dataset else "test"

        try:
            loader = build_dataset_loader(cfg, spec, args.split, data_root=data_root)
        except EvaluateError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_FAILURE

        started = time.time()
        try:
            rows = evaluate_dataset(
                model,
                loader,
                device,
                tta_flip=bool(args.tta_flip),
                use_amp=use_amp,
                small_lesion_area_fraction=float(cfg["data"]["small_lesion_area_fraction"]),
            )
        except EvaluateError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_FAILURE
        elapsed = time.time() - started

        summary = summarize_rows(rows)
        payload = {
            "run_name": cfg["run"]["name"],
            "split": row_split,
            "dataset": spec.name,
            "dataset_source": spec.source or spec.name,
            "tta": tta_label,
            "checkpoint": args.checkpoint,
            "config_file": str(run_info.get("config_file") or run_dir / "config_resolved.yaml"),
            "config_override": str(args.config) if args.config is not None else None,
            "git_commit": run_info.get("git_commit"),
            "seed": int(cfg["run"]["seed"]),
            "config_name": cfg["run"]["name"],
            "best_epoch": _checkpoint_epochs(checkpoint, cfg),
            "epochs_trained": _epochs_trained(run_dir),
            "image_size": int(cfg["data"]["image_size"]),
            "threshold": THRESHOLD,
            "device": str(device),
            "n": int(summary["n"]),
            "dice_mean": summary["dice_mean"],
            "dice_std": summary["dice_std"],
            "iou_mean": summary["iou_mean"],
            "iou_std": summary["iou_std"],
            "mae_mean": summary["mae_mean"],
            "mae_std": summary["mae_std"],
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

        json_name, per_image_name = output_names(row_split, spec.name, tta_flip=bool(args.tta_flip))
        json_path = write_eval_json(run_dir / json_name, payload)
        per_image_path = write_per_image_csv(run_dir / per_image_name, rows)

        results_row = {
            "run_name": payload["run_name"],
            "split": row_split,
            "dataset": spec.name,
            "tta": tta_label,
            "config_name": payload["config_name"],
            "config_file": payload["config_file"],
            "git_commit": payload["git_commit"] or "",
            "seed": payload["seed"],
            "best_epoch": payload["best_epoch"],
            "epochs_trained": payload["epochs_trained"],
            "params_m": f"{payload['params_m']:.3f}",
            "best_val_dice": f"{payload['best_val_dice']:.6f}",
            "test_dice": f"{summary['dice_mean']:.6f}",
            "test_iou": f"{summary['iou_mean']:.6f}",
            "test_mae": f"{summary['mae_mean']:.6f}",
            "small_lesion_dice": (
                "n/a"
                if (
                    summary["small_lesion_dice"] is None
                    or (
                        isinstance(summary["small_lesion_dice"], float)
                        and math.isnan(summary["small_lesion_dice"])
                    )
                )
                else f"{summary['small_lesion_dice']:.6f}"
            ),
            "small_lesion_n": str(int(summary["small_lesion_n"])),
            "mean_epoch_time_s": ""
            if payload["mean_epoch_time_s"] is None
            else f"{float(payload['mean_epoch_time_s']):.2f}",
            "peak_gpu_mem_mb": ""
            if payload["peak_gpu_mem_mb"] is None
            else f"{float(payload['peak_gpu_mem_mb']):.1f}",
        }
        try:
            append_results_row(results_csv, results_row, force=bool(args.force))
        except (DuplicateResultError, EvaluateError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_FAILURE

        print(
            f"{payload['run_name']} [{row_split}/{spec.name}] n={payload['n']} "
            f"dice {summary['dice_mean']:.4f} +/- {summary['dice_std']:.4f} "
            f"iou {summary['iou_mean']:.4f} +/- {summary['iou_std']:.4f} "
            f"mae {summary['mae_mean']:.4f}"
            + (" [flip TTA]" if args.tta_flip else "")
        )
        print(f"wrote {json_path}")
        print(f"wrote {per_image_path}")
        print(f"appended {results_csv}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
