"""Training loop with checkpoint/resume, differential LRs, AMP and metrics.csv.

Usage:
    python -m src.train --config configs/exp001_convnext_base_baseline.yaml
    python -m src.train --config configs/exp001_convnext_base_baseline.yaml --smoke

Resume is automatic: if ``<out_dir>/<run_name>/last.pt`` exists the run continues
from it. ``--no-resume`` starts fresh. ``--smoke`` shrinks everything (64 px,
2 images per batch, 3 optimiser steps, synthetic data in the run folder, no
pretrained download) to prove the pipeline runs end to end on CPU. The active
branch is still the one named by the config (ConvNeXt-tiny for Branch A,
caformer_s18 for Branch B). Smoke numbers are not results and are never
comparable to a real run.

Never run full training on the CPU dev box: only ``--smoke`` is allowed there.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import timm
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.datasets import PolypDataset, build_train_transform
from src.losses import build_loss
from src.metrics import SegMetrics
from src.models.net import build_model, count_parameters
from src.utils import ConfigError, load_config, save_config, set_seed

#: Columns of ``metrics.csv``, in order (AGENTS.md section 5).
METRICS_COLUMNS: tuple[str, ...] = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_dice",
    "val_iou",
    "lr_encoder_a",
    "lr_encoder_b",
    "lr_head",
    "peak_gpu_mem_mb",
    "epoch_time_s",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace with ``config``, ``smoke``, ``no_resume`` and ``max_steps``.
    """
    parser = argparse.ArgumentParser(description="Train the polyp segmentation model.")
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML, e.g. configs/exp001_convnext_base_baseline.yaml")
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU run on synthetic data; proves the pipeline only.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore an existing last.pt and start from scratch.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optimiser steps per epoch (smoke/debug).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override run.out_dir, e.g. a Drive folder on Colab.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def apply_smoke_overrides(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Shrink a config into a CPU smoke-test config.

    Args:
        cfg: Resolved config.

    Returns:
        A modified deep copy: 1 epoch, 64 px, batch 2, 3 steps, no pretrained
        download, no W&B, no workers, and the config's active branch only
        (ConvNeXt-tiny for Branch A, caformer_s18 for Branch B).
    """
    smoke = copy.deepcopy(dict(cfg))
    smoke["run"]["name"] = f"{smoke['run']['name']}_smoke"
    smoke["train"].update({"epochs": 1, "batch_size": 2, "amp": False, "grad_accum_steps": 1, "warmup_epochs": 0})
    smoke["data"].update(
        {"image_size": 64, "num_workers": 0, "pin_memory": False, "cache_path": ""}
    )
    encoder_kind = str(smoke["model"].get("encoder", "convnext"))
    branch_mode = "single_b" if encoder_kind == "caformer" else "single_a"
    # encoder_a is unused in the Branch B smoke test, but keeping a tiny name makes
    # the smoke config harmless if someone inspects it. Branch B keeps caformer_s18
    # so the real Meta-Polyp path is the one under test.
    smoke["model"].update({"encoder_a": "convnext_tiny.fb_in1k", "pretrained_a": False})
    smoke["model"].update(
        {
            "pretrained_b": False,
            "fusion": [branch_mode] * 4,
            "msu": False,
            "skip_convformer": False,
        }
    )
    smoke["logging"].update({"wandb": False, "log_images_every": 1, "num_log_images": 2})
    return smoke


def _ellipse_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a random filled ellipse.

    Args:
        height: Image height.
        width: Image width.
        rng: NumPy generator.

    Returns:
        ``(height, width)`` uint8 array, 255 inside the ellipse, 0 outside.
    """
    ys, xs = np.mgrid[0:height, 0:width]
    cy = rng.uniform(0.3 * height, 0.7 * height)
    cx = rng.uniform(0.3 * width, 0.7 * width)
    ry = rng.uniform(0.12 * height, 0.28 * height)
    rx = rng.uniform(0.12 * width, 0.28 * width)
    inside = ((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2 <= 1.0
    return (inside * 255).astype(np.uint8)


def synthesize_smoke_data(
    folder: str | Path,
    image_size: int = 64,
    n_train: int = 6,
    n_val: int = 2,
    seed: int = 0,
) -> Path:
    """Create a tiny synthetic dataset for ``--smoke`` (no download, not real data).

    Args:
        folder: Destination folder; receives ``images/``, ``masks/`` and ``splits/``.
        image_size: Square image size.
        n_train: Number of training images.
        n_val: Number of validation images.
        seed: RNG seed.

    Returns:
        The folder, ready for :meth:`PolypDataset.from_config`.

    Notes:
        Images are noisy gradients with a brighter ellipse; masks are that ellipse.
        It is learnable, which is what the smoke run needs, and worthless as data,
        which is why ``run_info.json`` marks the run as synthetic.
    """
    root = Path(folder)
    images_dir = root / "images"
    masks_dir = root / "masks"
    splits_dir = root / "splits"
    for path in (images_dir, masks_dir, splits_dir):
        path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    for index in range(n_train + n_val):
        sample_id = f"smoke_{index:03d}"
        mask = _ellipse_mask(image_size, image_size, rng)
        background = rng.integers(40, 90, size=(image_size, image_size, 3), dtype=np.uint8)
        image = np.clip(background.astype(np.int16) + (mask[..., None] // 2), 0, 255).astype(np.uint8)
        Image.fromarray(image, mode="RGB").save(images_dir / f"{sample_id}.png")
        Image.fromarray(mask, mode="L").save(masks_dir / f"{sample_id}.png")
        (train_ids if index < n_train else val_ids).append(sample_id)

    (splits_dir / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    (splits_dir / "test.txt").write_text("\n".join(val_ids[:1]) + "\n", encoding="utf-8")
    return root


def build_dataloaders(
    cfg: Mapping[str, Any],
    *,
    synthetic_root: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build the train and validation loaders.

    Args:
        cfg: Resolved (possibly smoke-) config.
        synthetic_root: When given, the dataset root for both splits is this
            folder and split files are read from ``synthetic_root/splits``.

    Returns:
        ``(train_loader, val_loader)``. Train gets augmentation and shuffling;
        validation is deterministic (resize + normalise only).
    """
    data = cfg["data"]
    root = synthetic_root if synthetic_root is not None else data["root"]
    split_dir = (synthetic_root / "splits") if synthetic_root is not None else data["split_dir"]
    image_size = int(data["image_size"])

    def make(split: str, train: bool) -> tuple[PolypDataset, DataLoader]:
        dataset = PolypDataset(
            split_file=Path(split_dir) / f"{split}.txt",
            root=root,
            transform=build_train_transform(image_size, seed=int(cfg["run"]["seed"])) if train else None,
            image_size=image_size,
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
        loader = DataLoader(
            dataset,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=train,
            num_workers=int(data["num_workers"]),
            pin_memory=bool(data["pin_memory"]),
            drop_last=False,
            persistent_workers=False,
        )
        return dataset, loader

    _, train_loader = make("train", train=True)
    _, val_loader = make("val", train=False)
    return train_loader, val_loader


def build_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    """AdamW with one parameter group per model part (differential LRs).

    Args:
        model: A model exposing ``parameter_groups()``.
        cfg: Resolved config; reads ``train.lr_convnext``, ``train.lr_encoder_b``,
            ``train.lr_head`` and ``train.weight_decay``.

    Returns:
        Configured ``AdamW``. Empty groups (no Branch B yet) are dropped.
    """
    learning_rates = {
        "encoder_a": float(cfg["train"]["lr_convnext"]),
        "encoder_b": float(cfg["train"]["lr_encoder_b"]),
        "head": float(cfg["train"]["lr_head"]),
    }
    groups = model.parameter_groups()  # type: ignore[attr-defined]
    param_groups = [
        {"params": params, "lr": learning_rates[name], "name": name}
        for name, params in groups.items()
        if params
    ]
    if not param_groups:
        raise RuntimeError("No trainable parameters found")
    return torch.optim.AdamW(param_groups, weight_decay=float(cfg["train"]["weight_decay"]))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Mapping[str, Any],
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing over optimiser steps, with optional linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        cfg: Resolved config; reads ``train.epochs`` and ``train.warmup_epochs``.
        steps_per_epoch: Optimiser steps per epoch.

    Returns:
        A per-step ``LambdaLR`` (its ``state_dict`` is checkpointed, so resume is exact).
    """
    total_steps = max(1, int(cfg["train"]["epochs"]) * max(1, steps_per_epoch))
    warmup_steps = max(0, int(round(float(cfg["train"]["warmup_epochs"]) * max(1, steps_per_epoch))))

    def lr_lambda(step: int) -> float:
        """Scale factor for ``step``.

        Args:
            step: Optimiser step index (0-based).

        Returns:
            Multiplier in [0, 1] on each group's base learning rate.
        """
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_grad_scaler(enabled: bool) -> Any:
    """Create a GradScaler that also exists on CPU-only torch builds.

    Args:
        enabled: Whether mixed precision is active (only on CUDA).

    Returns:
        A ``GradScaler`` instance (disabled when ``enabled`` is False).
    """
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(enabled: bool) -> Any:
    """Return the autocast context for CUDA fp16.

    Args:
        enabled: Whether to use autocast.

    Returns:
        A context manager (``nullcontext`` when disabled).
    """
    if not enabled:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def compute_loss(
    model: nn.Module,
    batch: Mapping[str, Any],
    loss_fn: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = False,
    channels_last: bool = False,
) -> Tensor:
    """Forward one batch and return the loss.

    Args:
        model: Segmentation model returning logits.
        batch: Dataset dict with ``image (B,3,H,W)`` and ``mask (B,1,H,W)``.
        loss_fn: Loss taking ``(logits, masks)``.
        device: Target device.
        use_amp: Wrap the forward in CUDA fp16 autocast.
        channels_last: Convert inputs to channels-last memory format.

    Returns:
        Scalar loss tensor (still attached to the graph).
    """
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    with _autocast(use_amp):
        logits = model(images)
        loss = loss_fn(logits, masks)
    return loss


def get_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Read the current LR of every named parameter group.

    Args:
        optimizer: Optimizer with named groups.

    Returns:
        Mapping ``group name -> LR``; missing names report 0.0.
    """
    rates = {"encoder_a": 0.0, "encoder_b": 0.0, "head": 0.0}
    for group in optimizer.param_groups:
        name = str(group.get("name", ""))
        rates[name] = float(group["lr"])
    return rates


def _optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Any,
    clip_norm: float,
) -> None:
    """Apply one optimiser step, after optional unscaled gradient clipping.

    Args:
        model: Model whose gradients are about to be applied.
        optimizer: Optimiser.
        scheduler: Per-step scheduler, advanced here.
        scaler: GradScaler (disabled on CPU); ``unscale_`` must run before clipping.
        clip_norm: Max global gradient norm; ``<= 0`` disables clipping.
    """
    if clip_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()


def reseed_train_augmentation(loader: DataLoader, seed: int) -> None:
    """Re-seed the training augmentation pipeline for one epoch.

    Args:
        loader: Training loader whose dataset carries the albumentations pipeline.
        seed: Seed for this epoch, usually ``run.seed + epoch``.

    Notes:
        The pipeline is built once, so without this every epoch would repeat the
        same random crops and flips and the reported numbers would not be
        reproducible from ``run.seed`` alone. Pipelines without
        ``set_random_seed`` (plain callables) are left untouched.
    """
    setter = getattr(getattr(loader.dataset, "transform", None), "set_random_seed", None)
    if callable(setter):
        setter(int(seed))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Any,
    cfg: Mapping[str, Any],
    device: torch.device,
    *,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Run one epoch of training.

    Args:
        model: Model in train mode (set here).
        loader: Training loader.
        loss_fn: Loss module.
        optimizer: Optimiser.
        scheduler: Per-step scheduler, stepped after every optimiser step.
        scaler: GradScaler (disabled on CPU).
        cfg: Resolved config; reads ``train.grad_accum_steps``,
            ``train.grad_clip_norm``, ``train.amp`` and ``train.channels_last``.
        device: Device.
        max_steps: Optional cap on optimiser steps (smoke runs).

    Returns:
        ``{"loss": mean micro-batch loss, "steps": optimiser steps}``.
    """
    model.train()
    accumulate = max(1, int(cfg["train"]["grad_accum_steps"]))
    clip_norm = float(cfg["train"]["grad_clip_norm"])
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    channels_last = bool(cfg["train"]["channels_last"]) and device.type == "cuda"

    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    loss_sum = 0.0
    micro_batches = 0
    steps = 0
    optimizer.zero_grad(set_to_none=True)

    stopped = False
    for micro_index, batch in enumerate(loader, start=1):
        loss = compute_loss(
            model, batch, loss_fn, device, use_amp=use_amp, channels_last=channels_last
        )
        scaler.scale(loss / accumulate).backward()
        loss_sum += float(loss.detach())
        micro_batches += 1

        if micro_index % accumulate == 0:
            _optimizer_step(model, optimizer, scheduler, scaler, clip_norm)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                stopped = True
                break

    # Flush a trailing partial accumulation window, so the last few batches of an
    # epoch are not silently trained on but never applied.
    if not stopped and micro_batches % accumulate != 0:
        _optimizer_step(model, optimizer, scheduler, scaler, clip_norm)
        steps += 1

    return {"loss": loss_sum / max(1, micro_batches), "steps": float(steps)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = False,
) -> dict[str, Any]:
    """Evaluate on a split: mean loss plus per-image Dice/IoU.

    Args:
        model: Model (switched to eval mode here).
        loader: Validation or test loader.
        loss_fn: Loss module.
        device: Device.
        use_amp: CUDA fp16 autocast.

    Returns:
        ``{"loss": float, "dice": float, "iou": float, "n": int, "per_image": [...]}``.
    """
    model.eval()
    metrics = SegMetrics(from_logits=True)
    total_loss = 0.0
    batches = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with _autocast(use_amp):
            logits = model(images)
            loss = loss_fn(logits, masks)
        total_loss += float(loss)
        batches += 1
        metrics.update(logits, masks, ids=list(batch["id"]))
    summary = metrics.compute()
    return {
        "loss": total_loss / max(1, batches),
        "dice": summary["dice"],
        "iou": summary["iou"],
        "n": int(summary["n"]),
        "per_image": metrics.per_image(),
    }


def _rng_state() -> dict[str, Any]:
    """Snapshot Python, NumPy and torch RNG state.

    Returns:
        Dict of RNG states, safe to ``torch.save``.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state produced by :func:`_rng_state`.

    Args:
        state: The saved mapping.
    """
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Any,
    epoch: int,
    best_val_dice: float,
    cfg: Mapping[str, Any],
) -> Path:
    """Write a full-training-state checkpoint (survives a Colab disconnect).

    Args:
        path: Destination file, e.g. ``last.pt``.
        model: Model.
        optimizer: Optimizer.
        scheduler: Scheduler.
        scaler: GradScaler.
        epoch: Epoch just finished (0-based index).
        best_val_dice: Best validation Dice so far.
        cfg: Resolved config, stored for provenance.

    Returns:
        The written path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": int(epoch),
            "best_val_dice": float(best_val_dice),
            "rng": _rng_state(),
            "config": copy.deepcopy(dict(cfg)),
        },
        out,
    )
    return out


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    scaler: Any = None,
    device: torch.device | str = "cpu",
    *,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, restoring as much state as the caller provides.

    Args:
        path: Checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        scaler: Optional GradScaler to restore.
        device: Map location.
        restore_rng: Restore RNG state, so a resumed run continues bit-identically.

    Returns:
        The raw checkpoint dict (``epoch``, ``best_val_dice``, ...).

    Raises:
        FileNotFoundError: if the checkpoint does not exist.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng and "rng" in checkpoint:
        _restore_rng_state(checkpoint["rng"])
    return checkpoint


def append_metrics_row(path: str | Path, row: Mapping[str, Any]) -> Path:
    """Append one epoch row to ``metrics.csv``, writing the header if needed.

    Args:
        path: CSV path.
        row: Values for :data:`METRICS_COLUMNS`.

    Returns:
        The written path.

    Raises:
        KeyError: if a column is missing from ``row``.
        ValueError: if an existing file has a different header (old run folder).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    missing = [column for column in METRICS_COLUMNS if column not in row]
    if missing:
        raise KeyError(f"metrics row missing columns: {missing}")
    write_header = not out.is_file() or out.stat().st_size == 0
    if not write_header:
        with out.open(newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle), [])
        if existing_header != list(METRICS_COLUMNS):
            raise ValueError(
                f"{out} has an incompatible header {existing_header}; "
                f"expected {list(METRICS_COLUMNS)}. Use a new run folder."
            )
    with out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRICS_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow({column: row[column] for column in METRICS_COLUMNS})
    return out


def _git_info(repo_dir: str | Path | None = None) -> dict[str, Any]:
    """Read the current commit hash and dirty flag.

    Args:
        repo_dir: Repository directory; defaults to the repository this file is in.

    Returns:
        ``{"git_commit": str|None, "git_dirty": bool|None}``; ``None`` if git fails.
    """
    directory = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parents[1]

    def run(args: Sequence[str]) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=str(directory),
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None

    commit = run(["rev-parse", "HEAD"])
    status = run(["status", "--porcelain"])
    dirty = None if status is None else bool(status)
    return {"git_commit": commit, "git_dirty": dirty}


def _sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.
        chunk_size: Read size in bytes (1 MiB default).

    Returns:
        Lower-case hexadecimal digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_info(
    path: str | Path,
    cfg: Mapping[str, Any],
    model: nn.Module,
    *,
    synthetic: bool,
    config_file: str | Path,
    repo_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write ``run_info.json`` (provenance for the paper).

    Args:
        path: Destination file.
        cfg: Effective config.
        model: The model, for parameter counts and encoder names.
        synthetic: Whether the data is the synthetic smoke set.
        config_file: The experiment config that was launched.
        repo_dir: Repository directory for git info.

    Returns:
        The dict that was written.
    """
    total, trainable = count_parameters(model)
    cache_path = cfg["data"].get("cache_path") or None
    cache_sha256 = _sha256_file(cache_path) if cache_path and Path(cache_path).is_file() else None
    info: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **_git_info(repo_dir),
        "config_file": str(config_file),
        "run_name": cfg["run"]["name"],
        "seed": int(cfg["run"]["seed"]),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "split_files": {
            split: str(Path(cfg["data"]["split_dir"]) / f"{split}.txt") for split in ("train", "val", "test")
        },
        "synth_data": bool(synthetic),
        "data_root": cfg["data"]["root"],
        "image_size": int(cfg["data"]["image_size"]),
        "data_format": {
            "image_dir": cfg["data"]["image_dir"],
            "mask_dir": cfg["data"]["mask_dir"],
            "image_ext": cfg["data"]["image_ext"],
            "mask_ext": cfg["data"]["mask_ext"],
            "mask_suffix": cfg["data"]["mask_suffix"],
            "mask_threshold": int(cfg["data"]["mask_threshold"]),
            "cache_path": cache_path,
            "cache_sha256": cache_sha256,
        },
        "normalization": {
            "mean": list(cfg["data"]["mean"]),
            "std": list(cfg["data"]["std"]),
            "source": "ImageNet (documented default, shared by both branches)",
        },
        "model": {
            "encoder": cfg["model"]["encoder"],
            "encoder_a": cfg["model"]["encoder_a"] if cfg["model"]["encoder"] == "convnext" else None,
            "encoder_b": cfg["model"]["encoder_b"] if cfg["model"]["encoder"] == "caformer" else None,
            "pretrained_a": bool(cfg["model"]["pretrained_a"]),
            "pretrained_b": bool(cfg["model"]["pretrained_b"]),
            "fusion": (
                [str(cfg["model"]["fusion"])] * 4
                if isinstance(cfg["model"]["fusion"], str)
                else list(cfg["model"]["fusion"])
            ),
            "msu": bool(cfg["model"]["msu"]),
            "msu_mode": cfg["model"]["msu_mode"],
            "skip_convformer": bool(cfg["model"]["skip_convformer"]),
            "channels": list(cfg["model"]["channels"]),
        },
        "params_total": int(total),
        "params_trainable": int(trainable),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def update_run_info_timing(
    path: str | Path,
    info: Mapping[str, Any],
    *,
    peak_gpu_mem_mb: float,
    epoch_times_s: Sequence[float],
) -> dict[str, Any]:
    """Rewrite ``run_info.json`` with measured peak memory and epoch time.

    Args:
        path: Existing ``run_info.json``.
        info: Dict returned by :func:`write_run_info`.
        peak_gpu_mem_mb: Maximum ``torch.cuda.max_memory_allocated`` this session.
        epoch_times_s: Wall-clock seconds for each epoch completed this session.

    Returns:
        The updated dict that was written.

    Notes:
        Values cover the current process only. After a reliable resume the
        earlier session's peak is not recoverable from ``last.pt``; together with
        the per-epoch rows in ``metrics.csv`` this is the honest record.
    """
    updated = dict(info)
    updated["peak_gpu_mem_mb"] = round(float(peak_gpu_mem_mb), 1)
    updated["mean_epoch_time_s"] = (
        round(float(sum(epoch_times_s)) / len(epoch_times_s), 2) if epoch_times_s else None
    )
    updated["epochs_this_session"] = len(epoch_times_s)
    out = Path(path)
    out.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    return updated


def log_prediction_grid(
    run_dir: str | Path,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Mapping[str, Any],
    epoch: int,
    max_images: int = 4,
) -> Path | None:
    """Save a small grid of image / ground truth / prediction for one epoch.

    Args:
        run_dir: Run folder; the PNG goes to ``run_dir/preds``.
        model: Model (switched to eval mode here).
        loader: Validation loader.
        device: Device.
        cfg: Resolved config (for de-normalisation statistics).
        epoch: Current epoch index, used in the file name.
        max_images: Number of samples in the grid.

    Returns:
        The written PNG path, or ``None`` if the loader is empty.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean = np.asarray(cfg["data"]["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(cfg["data"]["std"], dtype=np.float32).reshape(3, 1, 1)

    model.eval()
    batch = next(iter(loader), None)
    if batch is None:
        return None
    images = batch["image"][:max_images]
    masks = batch["mask"][:max_images]
    ids = list(batch["id"])[:max_images]
    with torch.no_grad():
        logits = model(images.to(device))
    probs = torch.sigmoid(logits).cpu().numpy()

    count = images.shape[0]
    fig, axes = plt.subplots(count, 3, figsize=(6, 2 * count), squeeze=False)
    for row in range(count):
        image = np.clip(images[row].numpy() * std + mean, 0, 1).transpose(1, 2, 0)
        axes[row][0].imshow(image)
        axes[row][0].set_title(f"{ids[row]}", fontsize=7)
        axes[row][1].imshow(masks[row, 0].numpy(), vmin=0, vmax=1, cmap="gray")
        axes[row][1].set_title("GT", fontsize=7)
        axes[row][2].imshow(probs[row, 0], vmin=0, vmax=1, cmap="magma")
        axes[row][2].set_title("pred", fontsize=7)
        for column in range(3):
            axes[row][column].axis("off")
    fig.tight_layout()
    out = Path(run_dir) / "preds" / f"epoch_{epoch:04d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _prepare_smoke_data(cfg: Mapping[str, Any], run_dir: Path) -> tuple[dict[str, Any], Path]:
    """Generate synthetic data for a smoke run and point the config at it.

    Args:
        cfg: Smoke config.
        run_dir: Run folder.

    Returns:
        ``(updated config, synthetic root)``.
    """
    root = synthesize_smoke_data(
        run_dir / "_smoke_data",
        image_size=int(cfg["data"]["image_size"]),
        n_train=6,
        n_val=2,
        seed=int(cfg["run"]["seed"]),
    )
    updated = copy.deepcopy(dict(cfg))
    updated["data"]["root"] = str(root)
    updated["data"]["split_dir"] = str(root / "splits")
    updated["data"]["image_ext"] = ".png"
    updated["data"]["mask_ext"] = ".png"
    updated["data"]["mask_suffix"] = ""
    updated["data"]["synthetic"] = True
    return updated, root


def main(argv: Iterable[str] | None = None) -> int:
    """Run training.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    if args.out_dir is not None:
        cfg["run"]["out_dir"] = str(args.out_dir)

    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
        print("SMOKE MODE: 64 px, synthetic data, 3 steps, no pretrained weights. Numbers are not results.")

    set_seed(int(cfg["run"]["seed"]))
    run_dir = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"run: {run_dir}  device: {device}  seed: {cfg['run']['seed']}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    else:
        print("gpu: none (CPU)")

    synthetic_root: Path | None = None
    if args.smoke:
        cfg, synthetic_root = _prepare_smoke_data(cfg, run_dir)
    synthetic_data = synthetic_root is not None or bool(cfg["data"]["synthetic"])

    model = build_model(cfg).to(device)
    total, trainable = count_parameters(model)
    active_encoder = str(cfg["model"]["encoder"])
    encoder_name = (
        cfg["model"]["encoder_b"] if active_encoder == "caformer" else cfg["model"]["encoder_a"]
    )
    print(
        f"model: {active_encoder} ({encoder_name})  "
        f"params: {total/1e6:.2f}M ({trainable/1e6:.2f}M trainable)"
    )

    train_loader, val_loader = build_dataloaders(cfg, synthetic_root=synthetic_root)
    print(f"data: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val")  # type: ignore[arg-type]

    loss_fn = build_loss(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    max_steps = args.max_steps if args.max_steps is not None else (3 if args.smoke else None)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, int(cfg["train"]["grad_accum_steps"]))))
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = make_grad_scaler(use_amp)

    save_config(cfg, run_dir / "config_resolved.yaml")
    run_info = write_run_info(
        run_dir / "run_info.json",
        cfg,
        model,
        synthetic=synthetic_data,
        config_file=args.config,
    )
    print(f"provenance: {run_info.get('git_commit')} dirty={run_info.get('git_dirty')}")

    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.csv"
    start_epoch = 0
    best_val_dice = -1.0
    peak_gpu_mem_mb = 0.0
    epoch_times_s: list[float] = []

    # Smoke runs always start clean; real runs resume automatically.
    resume_allowed = not args.no_resume and not args.smoke
    if resume_allowed and last_path.is_file():
        checkpoint = load_checkpoint(last_path, model, optimizer, scheduler, scaler, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_dice = float(checkpoint.get("best_val_dice", -1.0))
        print(f"resumed from {last_path} at epoch {start_epoch} (best val dice {best_val_dice:.4f})")
    elif args.no_resume and last_path.is_file():
        print(f"--no-resume: ignoring existing {last_path}")

    if bool(cfg["logging"]["wandb"]):
        try:
            import wandb

            wandb.init(project=cfg["logging"]["wandb_project"], name=cfg["run"]["name"], dir=str(run_dir), config=dict(cfg))
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"wandb disabled ({exc}); metrics.csv still written", file=sys.stderr)

    try:
        for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
            started = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            # Log the LR the epoch starts with: after the last scheduler step of a
            # one-epoch smoke run the LR is legitimately 0, which would hide it.
            rates = get_learning_rates(optimizer)
            reseed_train_augmentation(train_loader, int(cfg["run"]["seed"]) + epoch)
            train_stats = train_one_epoch(
                model, train_loader, loss_fn, optimizer, scheduler, scaler, cfg, device, max_steps=max_steps
            )
            val_stats = evaluate(model, val_loader, loss_fn, device, use_amp=use_amp)
            peak_epoch_mb = (
                float(torch.cuda.max_memory_allocated()) / (1024**2) if device.type == "cuda" else 0.0
            )
            peak_gpu_mem_mb = max(peak_gpu_mem_mb, peak_epoch_mb)
            elapsed = time.time() - started
            epoch_times_s.append(elapsed)

            row = {
                "epoch": epoch,
                "train_loss": round(train_stats["loss"], 6),
                "val_loss": round(val_stats["loss"], 6),
                "val_dice": round(val_stats["dice"], 6),
                "val_iou": round(val_stats["iou"], 6),
                "lr_encoder_a": rates["encoder_a"],
                "lr_encoder_b": rates["encoder_b"],
                "lr_head": rates["head"],
                "peak_gpu_mem_mb": round(peak_epoch_mb, 1),
                "epoch_time_s": round(elapsed, 2),
            }
            append_metrics_row(metrics_path, row)
            print(
                f"epoch {epoch}: train {row['train_loss']:.4f} val {row['val_loss']:.4f} "
                f"dice {row['val_dice']:.4f} iou {row['val_iou']:.4f} "
                f"peak {row['peak_gpu_mem_mb']:.0f}MB ({elapsed:.1f}s)"
            )

            if bool(cfg["logging"]["wandb"]):
                try:
                    import wandb

                    wandb.log({"epoch": epoch, **{k: v for k, v in row.items() if k != "epoch"}})
                except Exception:  # pragma: no cover
                    pass

            every = max(1, int(cfg["logging"]["log_images_every"]))
            if epoch % every == 0:
                grid = log_prediction_grid(
                    run_dir, model, val_loader, device, cfg, epoch, int(cfg["logging"]["num_log_images"])
                )
                if grid is not None and bool(cfg["logging"]["wandb"]):
                    try:
                        import wandb

                        wandb.log({f"preds/epoch_{epoch}": wandb.Image(str(grid))})
                    except Exception:  # pragma: no cover
                        pass

            if val_stats["dice"] > best_val_dice:
                best_val_dice = float(val_stats["dice"])
                save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_val_dice, cfg)
                print(f"  new best val dice {best_val_dice:.4f} -> {best_path}")
            # last.pt is written after the best check, so it always carries the
            # current best value; otherwise a disconnect could make resume
            # overwrite best.pt with a worse epoch.
            save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best_val_dice, cfg)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted; last.pt holds the last completed epoch", file=sys.stderr)
        return 130
    finally:
        try:
            update_run_info_timing(
                run_dir / "run_info.json",
                run_info,
                peak_gpu_mem_mb=peak_gpu_mem_mb,
                epoch_times_s=epoch_times_s,
            )
        except OSError as exc:  # pragma: no cover - e.g. Drive unmounted
            print(f"warning: could not update run_info.json ({exc})", file=sys.stderr)

    if bool(cfg["logging"]["wandb"]):
        try:
            import wandb

            wandb.finish()
        except Exception:  # pragma: no cover
            pass

    print(f"done. best val dice {best_val_dice:.4f}; see {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
