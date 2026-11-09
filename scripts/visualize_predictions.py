#!/usr/bin/env python3
"""Visualize prediction figures for one or more runs.

Reuses the repo's model builder, config loader, checkpoint loading, preprocessing
and threshold so predictions exactly match the evaluated ones.

Usage:
    python scripts/visualize_predictions.py \
        --run_dir runs/exp006_all_concat_30ep \
        --datasets kvasir_seg cvc_clinicdb cvc_colondb \
        --out_dir figures \
        --n_random 4 --n_failures 2 --seed 42 \
        --split test

For side-by-side comparison of multiple runs:
    python scripts/visualize_predictions.py \
        --run_dir runs/exp005_hybrid_fusion runs/exp006_all_concat_30ep \
        --out_dir figures/compare
"""

from __future__ import annotations

# Ensure repo root is on sys.path when run as a script
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_registry import resolve_data_root
from src.datasets import PolypDataset, read_split, build_correction_transform_from_config
from src.metrics import SegMetrics
from src.evaluate import THRESHOLD
from src.models.net import build_model, count_parameters
from src.utils import ConfigError, load_config, set_seed

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

DEFAULT_DATASETS = ("kvasir_seg", "cvc_clinicdb", "cvc_colondb")
HELDOUT_DATASETS = ("etis", "cvc_300")
ALL_DATASETS = DEFAULT_DATASETS + HELDOUT_DATASETS

# Colours for overlays (RGB, 0-1)
GREEN = np.array([0.0, 1.0, 0.0])   # ground-truth contour
RED = np.array([1.0, 0.0, 0.0])     # prediction contour
GREEN_BW = "black"                   # line style handled separately
RED_BW = "black"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save prediction figures for one or more runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run_dir",
        action="append",
        required=True,
        type=Path,
        help="Run folder (repeatable for side-by-side comparison). Must contain "
             "best.pt and config_resolved.yaml.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=ALL_DATASETS,
        help="Datasets to visualize. Held-out datasets (etis, cvc_300) require "
             "--allow_heldout.",
    )
    parser.add_argument(
        "--allow_heldout",
        action="store_true",
        help="Allow ETIS and CVC-300 (held-out datasets).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for figures and CSV.",
    )
    parser.add_argument(
        "--n_random",
        type=int,
        default=4,
        help="Number of random examples per dataset (fixed seed).",
    )
    parser.add_argument(
        "--n_failures",
        type=int,
        default=2,
        help="Number of failure examples per dataset (lowest per-image Dice).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting random examples.",
    )
    parser.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help="Split for the training dataset (kvasir_seg). External datasets "
             "always use all images.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="Checkpoint to load.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Override DATA_ROOT (default: from config or env).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override config_resolved.yaml (data paths only).",
    )
    parser.add_argument(
        "--bw",
        action="store_true",
        help="Black-and-white mode for paper figures: grayscale masks, "
             "black contours with different line styles.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_run(run_dir: Path, checkpoint_name: str, config_override: Path | None) -> tuple[dict, torch.nn.Module, dict, Path]:
    """Load config, model, checkpoint, and run_info for a run directory."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = config_override if config_override else run_dir / "config_resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)

    # Disable pretrained downloads; checkpoint supplies all weights
    model_cfg = dict(cfg)
    model_cfg["model"] = {**cfg["model"], "pretrained_a": False, "pretrained_b": False}

    device = torch.device("cpu")  # model built on CPU first
    model = build_model(model_cfg).to(device)

    checkpoint_path = run_dir / f"{checkpoint_name}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    run_info_path = run_dir / "run_info.json"
    run_info = {}
    if run_info_path.is_file():
        try:
            run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    return cfg, model, run_info, checkpoint_path


def build_loader_for_dataset(
    cfg: Mapping[str, Any],
    dataset_name: str,
    split: str,
    data_root: Path | None,
) -> tuple[DataLoader, list[str], int]:
    """Build a DataLoader for a dataset, returning loader, ids, and image_size."""
    from src.data_registry import get_spec

    train_dataset = str(cfg["data"].get("dataset", "kvasir_seg"))
    image_size = int(cfg["data"]["image_size"])
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])
    pin_memory = bool(cfg["data"]["pin_memory"])

    if dataset_name == train_dataset:
        # Training dataset: use committed split files
        split_file = Path(cfg["data"]["split_dir"]) / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        ids = read_split(split_file)
        root = Path(cfg["data"]["root"])
        cache_path = cfg["data"].get("cache_path") or None
        dataset = PolypDataset(
            split_file=split_file,
            root=root,
            transform=None,
            correction_transform=build_correction_transform_from_config(cfg),
            image_size=image_size,
            image_dir=cfg["data"]["image_dir"],
            mask_dir=cfg["data"]["mask_dir"],
            image_ext=cfg["data"]["image_ext"],
            mask_ext=cfg["data"]["mask_ext"],
            mask_suffix=cfg["data"]["mask_suffix"],
            mask_threshold=cfg["data"]["mask_threshold"],
            cache_path=cache_path,
            normalize=cfg["data"]["normalize"],
            mean=cfg["data"]["mean"],
            std=cfg["data"]["std"],
        )
    else:
        # External dataset: use processed cache
        spec = get_spec(dataset_name)
        resolved_root = resolve_data_root(str(data_root) if data_root else cfg["data"].get("data_root", ""))
        processed_dir = spec.processed_dir(resolved_root)
        npz_path = processed_dir / "data.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(
                f"Processed data for {dataset_name} not found: {npz_path}. "
                f"Run scripts/prepare_datasets.py first."
            )
        with np.load(npz_path) as data:
            ids = [str(v) for v in data["ids"]]

        meta_path = processed_dir / "meta.json"
        native_size = image_size
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                native_size = int(meta.get("image_size", image_size))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        dataset = PolypDataset(
            ids=ids,
            root=processed_dir,
            image_size=native_size,
            cache_path=npz_path,
            correction_transform=build_correction_transform_from_config(cfg),
            normalize=cfg["data"]["normalize"],
            mean=cfg["data"]["mean"],
            std=cfg["data"]["std"],
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader, ids, image_size


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    cfg: Mapping[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run model on loader, return dict id -> (image, gt_mask, pred_mask)."""
    model.eval()
    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            ids = batch["id"]

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                probs = torch.sigmoid(logits)

            probs = probs.float().cpu().numpy()  # (B, 1, H, W)
            targets = targets.float().cpu().numpy()  # (B, 1, H, W)
            images_np = images.float().cpu().numpy()  # (B, 3, H, W)

            for i, sample_id in enumerate(ids):
                # Denormalize image for visualization
                img = images_np[i].transpose(1, 2, 0)  # (H, W, 3)
                mean = np.array(cfg["data"]["mean"]).reshape(1, 1, 3)
                std = np.array(cfg["data"]["std"]).reshape(1, 1, 3)
                if cfg["data"]["normalize"]:
                    img = img * std + mean
                img = np.clip(img, 0, 1)

                gt = targets[i, 0]  # (H, W)
                pred_prob = probs[i, 0]  # (H, W)
                pred_mask = (pred_prob > THRESHOLD).astype(np.float32)

                results[sample_id] = (img, gt, pred_mask)

    return results


def compute_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """Compute Dice coefficient for a single image."""
    pred_f = pred.flatten()
    gt_f = gt.flatten()
    intersection = np.sum(pred_f * gt_f)
    denominator = np.sum(pred_f) + np.sum(gt_f)
    return float((2.0 * intersection + eps) / (denominator + eps))


def select_examples(
    per_image_csv: Path,
    ids: list[str],
    n_random: int,
    n_failures: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Select random and failure example IDs from per-image CSV."""
    if not per_image_csv.is_file():
        raise FileNotFoundError(f"Per-image CSV not found: {per_image_csv}")

    dice_by_id: dict[str, float] = {}
    with per_image_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dice_by_id[row["id"]] = float(row["dice"])

    # Filter to IDs that exist in both the CSV and the dataset
    available_ids = [id_ for id_ in ids if id_ in dice_by_id]
    if not available_ids:
        raise ValueError(f"No overlapping IDs between dataset and {per_image_csv}")

    # Random selection
    rng = random.Random(seed)
    random_ids = rng.sample(available_ids, min(n_random, len(available_ids)))

    # Failure selection (lowest Dice)
    sorted_by_dice = sorted(available_ids, key=lambda id_: dice_by_id[id_])
    failure_ids = sorted_by_dice[:min(n_failures, len(sorted_by_dice))]

    # Remove duplicates (an image could be both random and failure)
    failure_ids = [id_ for id_ in failure_ids if id_ not in random_ids]

    return random_ids, failure_ids


def find_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Find contours in a binary mask."""
    # Try skimage first (best quality)
    try:
        from skimage.measure import find_contours as sk_find_contours
        return sk_find_contours(mask, level=0.5)
    except ImportError:
        pass

    # Fallback: use matplotlib's public contour API
    try:
        cs = plt.contour(mask, levels=[0.5], colors="none")
        contours = []
        for collection in cs.collections:
            for path in collection.get_paths():
                contours.append(path.vertices)
        plt.close()
        return contours
    except Exception:
        # Last resort: simple boundary tracing
        return _simple_find_contours(mask)


def _simple_find_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Simple contour finding using boundary detection."""
    # Pad mask to handle boundaries
    padded = np.pad(mask.astype(bool), 1, mode='constant', constant_values=False)
    # Find edges where padded changes from False to True
    h, w = padded.shape
    contours = []
    visited = np.zeros_like(padded, dtype=bool)

    # Simple approach: find all boundary points and connect them
    # This is a very basic implementation - in practice skimage or matplotlib should be available
    boundary = np.zeros_like(padded, dtype=bool)
    boundary[1:-1, 1:-1] = padded[1:-1, 1:-1] & (
        ~padded[:-2, 1:-1] | ~padded[2:, 1:-1] | ~padded[1:-1, :-2] | ~padded[1:-1, 2:]
    )

    y_coords, x_coords = np.where(boundary)
    if len(x_coords) == 0:
        return []

    # Return as a single contour (x, y) - subtract 1 for padding
    contour = np.column_stack([y_coords - 1, x_coords - 1]).astype(float)
    return [contour]


def draw_overlay(
    ax: plt.Axes,
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    bw: bool,
    title: str = "",
) -> None:
    """Draw image with GT and prediction contours."""
    if bw:
        # Grayscale image
        ax.imshow(image.mean(axis=2), cmap="gray", vmin=0, vmax=1)
        # GT contour: solid black
        for contour in find_contours(gt):
            ax.plot(contour[:, 1], contour[:, 0], "k-", linewidth=1.5, label="Ground Truth" if contour is find_contours(gt)[0] else "")
        # Pred contour: dashed black
        for contour in find_contours(pred):
            ax.plot(contour[:, 1], contour[:, 0], "k--", linewidth=1.5, label="Prediction" if contour is find_contours(pred)[0] else "")
    else:
        # Color image
        ax.imshow(image)
        # GT contour: green
        for contour in find_contours(gt):
            ax.plot(contour[:, 1], contour[:, 0], color=GREEN, linewidth=2, label="Ground Truth" if contour is find_contours(gt)[0] else "")
        # Pred contour: red
        for contour in find_contours(pred):
            ax.plot(contour[:, 1], contour[:, 0], color=RED, linewidth=2, label="Prediction" if contour is find_contours(pred)[0] else "")

    ax.set_title(title, fontsize=10)
    ax.axis("off")


def save_figures(
    out_dir: Path,
    dataset_name: str,
    run_names: list[str],
    all_results: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    selected_ids: dict[str, list[tuple[str, str]]],  # dataset -> list of (id, reason)
    per_image_dice: dict[str, dict[str, float]],  # run_name -> {id: dice}
    bw: bool,
    dpi: int,
) -> tuple[Path, Path, Path]:
    """Save PNG, PDF, and combined figure for one dataset."""
    dataset_out = out_dir / dataset_name
    dataset_out.mkdir(parents=True, exist_ok=True)

    example_ids = [id_ for id_, _ in selected_ids[dataset_name]]
    n_examples = len(example_ids)
    n_runs = len(run_names)

    # Columns: image, GT, [pred per run], overlay per run
    n_cols = 2 + n_runs + n_runs  # image, GT, pred1..predN, overlay1..overlayN
    fig, axes = plt.subplots(n_examples, n_cols, figsize=(4 * n_cols, 4 * n_examples), squeeze=False)

    for row_idx, (sample_id, reason) in enumerate(selected_ids[dataset_name]):
        # Get data from first run (image and GT are same across runs)
        first_run = run_names[0]
        image, gt, _ = all_results[first_run][sample_id]

        # Column 0: Input image
        ax = axes[row_idx, 0]
        if bw:
            ax.imshow(image.mean(axis=2), cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(image)
        ax.set_title(f"Input\n{sample_id} ({reason})", fontsize=10)
        ax.axis("off")

        # Column 1: Ground truth mask
        ax = axes[row_idx, 1]
        if bw:
            ax.imshow(gt, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(gt, cmap="gray", vmin=0, vmax=1)
        ax.set_title("Ground Truth", fontsize=10)
        ax.axis("off")

        # Columns 2..2+n_runs-1: Predicted masks per run
        for run_idx, run_name in enumerate(run_names):
            _, _, pred = all_results[run_name][sample_id]
            ax = axes[row_idx, 2 + run_idx]
            if bw:
                ax.imshow(pred, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(pred, cmap="gray", vmin=0, vmax=1)
            dice = per_image_dice[run_name].get(sample_id, float("nan"))
            ax.set_title(f"Pred ({run_name})\nDice={dice:.4f}", fontsize=9)
            ax.axis("off")

        # Columns 2+n_runs..: Overlays per run
        for run_idx, run_name in enumerate(run_names):
            _, _, pred = all_results[run_name][sample_id]
            ax = axes[row_idx, 2 + n_runs + run_idx]
            dice = per_image_dice[run_name].get(sample_id, float("nan"))
            draw_overlay(ax, image, gt, pred, bw, title=f"Overlay ({run_name})\nDice={dice:.4f}")

    # Add a single legend for the first overlay row
    if n_examples > 0 and n_runs > 0:
        ax = axes[0, 2 + n_runs]
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = dataset_out / f"{dataset_name}_predictions.png"
    pdf_path = dataset_out / f"{dataset_name}_predictions.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path, dataset_out


def save_combined_figure(
    out_dir: Path,
    run_names: list[str],
    all_results: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]],  # dataset -> run -> sample_id -> (img, gt, pred)
    selected_ids: dict[str, list[tuple[str, str]]],
    per_image_dice: dict[str, dict[str, float]],  # run_name -> {sample_id: dice}
    bw: bool,
    dpi: int,
) -> tuple[Path, Path]:
    """Save a combined figure with all datasets stacked."""
    # Flatten all examples
    all_examples = []
    for dataset_name, examples in selected_ids.items():
        for sample_id, reason in examples:
            all_examples.append((dataset_name, sample_id, reason))

    n_examples = len(all_examples)
    n_runs = len(run_names)
    n_cols = 2 + n_runs + n_runs

    fig, axes = plt.subplots(n_examples, n_cols, figsize=(4 * n_cols, 3.5 * n_examples), squeeze=False)

    for row_idx, (dataset_name, sample_id, reason) in enumerate(all_examples):
        first_run = run_names[0]
        image, gt, _ = all_results[dataset_name][first_run][sample_id]

        # Input
        ax = axes[row_idx, 0]
        if bw:
            ax.imshow(image.mean(axis=2), cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(image)
        ax.set_title(f"{dataset_name}: {sample_id} ({reason})", fontsize=9)
        ax.axis("off")

        # GT
        ax = axes[row_idx, 1]
        if bw:
            ax.imshow(gt, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(gt, cmap="gray", vmin=0, vmax=1)
        ax.set_title("GT", fontsize=9)
        ax.axis("off")

        # Predictions
        for run_idx, run_name in enumerate(run_names):
            _, _, pred = all_results[dataset_name][run_name][sample_id]
            ax = axes[row_idx, 2 + run_idx]
            if bw:
                ax.imshow(pred, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(pred, cmap="gray", vmin=0, vmax=1)
            dice = per_image_dice[run_name].get(sample_id, float("nan"))
            ax.set_title(f"Pred ({run_name})\nDice={dice:.4f}", fontsize=8)
            ax.axis("off")

        # Overlays
        for run_idx, run_name in enumerate(run_names):
            _, _, pred = all_results[dataset_name][run_name][sample_id]
            ax = axes[row_idx, 2 + n_runs + run_idx]
            dice = per_image_dice[run_name].get(sample_id, float("nan"))
            draw_overlay(ax, image, gt, pred, bw, title=f"Overlay ({run_name})\nDice={dice:.4f}")

    if n_examples > 0 and n_runs > 0:
        ax = axes[0, 2 + n_runs]
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = out_dir / "combined_predictions.png"
    pdf_path = out_dir / "combined_predictions.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def save_selection_csv(
    out_dir: Path,
    dataset_name: str,
    run_names: list[str],
    selected_ids: list[tuple[str, str]],
    per_image_dice: dict[str, dict[str, float]],
    per_image_csv_path: Path,
) -> Path:
    """Save CSV listing chosen image IDs, per-image Dice, and selection reason."""
    csv_path = out_dir / dataset_name / f"{dataset_name}_selection.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Read original per-image CSV for additional columns
    original_rows: dict[str, dict] = {}
    with per_image_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_rows[row["id"]] = row

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["dataset", "image_id", "selection_reason"] + [f"dice_{run}" for run in run_names]
        writer.writerow(header)
        for sample_id, reason in selected_ids:
            row = [dataset_name, sample_id, reason]
            for run_name in run_names:
                dice = per_image_dice[run_name].get(sample_id, float("nan"))
                row.append(f"{dice:.6f}" if not math.isnan(dice) else "nan")
            writer.writerow(row)

    return csv_path


def consistency_check(
    all_results: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]],
    per_image_dice: dict[str, dict[str, float]],
    selected_ids: dict[str, list[tuple[str, str]]],
    tol: float = 1e-3,
) -> bool:
    """Recompute Dice from predictions and compare with per-image CSV values."""
    all_pass = True
    for dataset_name, examples in selected_ids.items():
        for sample_id, _ in examples:
            for run_name in per_image_dice:
                if dataset_name not in all_results or run_name not in all_results[dataset_name]:
                    continue
                _, gt, pred = all_results[dataset_name][run_name][sample_id]
                recomputed = compute_dice(pred, gt)
                recorded = per_image_dice[run_name].get(sample_id, float("nan"))
                if not math.isnan(recorded) and abs(recomputed - recorded) > tol:
                    print(
                        f"WARNING: Dice mismatch for {run_name}/{dataset_name}/{sample_id}: "
                        f"recomputed={recomputed:.6f}, recorded={recorded:.6f}, diff={abs(recomputed-recorded):.6f}",
                        file=sys.stderr,
                    )
                    all_pass = False
    if all_pass:
        print("Consistency check: PASS (all selected images match within 1e-3)")
    return all_pass


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Validate held-out datasets
    for ds in args.datasets:
        if ds in HELDOUT_DATASETS and not args.allow_heldout:
            print(f"ERROR: Dataset {ds} is held out. Use --allow_heldout to include it.", file=sys.stderr)
            return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load all runs
    run_configs = []
    run_models = []
    run_infos = []
    run_names = []
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    use_amp = device.type == "cuda"

    print(f"Using device: {device}")
    print(f"Loading {len(args.run_dir)} run(s)...")

    for run_dir in args.run_dir:
        cfg, model, run_info, ckpt_path = load_run(run_dir, args.checkpoint, args.config)
        model.to(device)
        run_configs.append(cfg)
        run_models.append(model)
        run_infos.append(run_info)
        run_names.append(run_dir.name)
        print(f"  {run_dir.name}: {ckpt_path.name}, params={count_parameters(model)[0]/1e6:.3f}M")

    # Use the first run's config for data loading (they should be compatible)
    cfg = run_configs[0]

    # Determine data root
    data_root = args.data_root if args.data_root else None

    # For each dataset, run inference and generate figures
    all_results: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    per_image_dice: dict[str, dict[str, float]] = {}
    selected_ids: dict[str, list[tuple[str, str]]] = {}

    for dataset_name in args.datasets:
        print(f"\n=== Processing {dataset_name} ===")

        # Build loader
        loader, ids, image_size = build_loader_for_dataset(cfg, dataset_name, args.split, data_root)
        print(f"  {len(ids)} images, size {image_size}x{image_size}")

        # Run inference for each model
        dataset_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for run_idx, model in enumerate(run_models):
            run_name = run_names[run_idx]
            print(f"  Running inference for {run_name}...")
            start = time.time()
            results = run_inference(model, loader, device, use_amp, cfg)
            elapsed = time.time() - start
            print(f"    {elapsed:.1f}s, {len(results)} images")
            dataset_results[run_name] = results

        all_results[dataset_name] = dataset_results

        # Read per-image CSV from first run (for selection)
        first_run_dir = args.run_dir[0]
        split_suffix = args.split if dataset_name == cfg["data"].get("dataset", "kvasir_seg") else "test"
        per_image_csv = first_run_dir / f"per_image_{split_suffix}_{dataset_name}.csv"
        if not per_image_csv.is_file():
            print(f"  WARNING: Per-image CSV not found: {per_image_csv}, skipping selection")
            continue

        # Select examples
        random_ids, failure_ids = select_examples(per_image_csv, ids, args.n_random, args.n_failures, args.seed)
        selected = [(id_, "random") for id_ in random_ids] + [(id_, "failure") for id_ in failure_ids]
        selected_ids[dataset_name] = selected
        print(f"  Selected {len(random_ids)} random, {len(failure_ids)} failure examples")

        # Compute per-image Dice for all runs (for consistency check and labels)
        for run_idx, run_name in enumerate(run_names):
            dice_dict = {}
            for sample_id, (_, gt, pred) in dataset_results[run_name].items():
                dice_dict[sample_id] = compute_dice(pred, gt)
            per_image_dice[run_name] = dice_dict

    # Save figures
    print("\n=== Saving figures ===")
    for dataset_name in args.datasets:
        if dataset_name not in selected_ids:
            continue
        png, pdf, _ = save_figures(
            args.out_dir,
            dataset_name,
            run_names,
            {rn: all_results[dataset_name][rn] for rn in run_names},
            selected_ids,
            per_image_dice,
            args.bw,
            args.dpi,
        )
        print(f"  {dataset_name}: {png}, {pdf}")

        # Save selection CSV
        split_suffix = args.split if dataset_name == cfg["data"].get("dataset", "kvasir_seg") else "test"
        per_image_csv = args.run_dir[0] / f"per_image_{split_suffix}_{dataset_name}.csv"
        csv_path = save_selection_csv(
            args.out_dir,
            dataset_name,
            run_names,
            selected_ids[dataset_name],
            per_image_dice,
            per_image_csv,
        )
        print(f"  Selection CSV: {csv_path}")

    # Combined figure
    if selected_ids:
        png, pdf = save_combined_figure(
            args.out_dir,
            run_names,
            all_results,  # dataset -> run -> sample_id -> (img, gt, pred)
            selected_ids,
            per_image_dice,
            args.bw,
            args.dpi,
        )
        print(f"  Combined: {png}, {pdf}")

    # Consistency check
    print("\n=== Consistency check ===")
    consistency_check(all_results, per_image_dice, selected_ids)

    print(f"\nDone. Output directory: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())