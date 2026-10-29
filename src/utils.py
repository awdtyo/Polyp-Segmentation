"""Config loading, seeding and small shared helpers.

Every hyperparameter lives in a YAML file under ``configs/``. A config file is
merged onto :data:`DEFAULT_CONFIG` so an experiment file only has to list what it
changes, and unknown keys are rejected so a typo cannot silently fall back to a
default.

Paths inside a config are relative to the repository root (the working directory
you launch training from). No absolute paths are hardcoded anywhere.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

#: ImageNet statistics, shared by both encoder branches (see AGENTS.md section 3).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Fusion settings accepted by ``model.fusion``: per-stage modes, the
#: ``hybrid`` preset, and the single-branch presets (AGENTS.md section 3.3).
FUSION_MODES: frozenset[str] = frozenset(
    {"concat", "xattn", "hybrid", "all_concat", "single_a", "single_b"}
)
#: Modes allowed inside the four-entry ``model.fusion`` list. Preset names are
#: only valid as a single string, not as a stage entry.
STAGE_FUSION_MODES: frozenset[str] = frozenset(
    {"concat", "xattn", "single_a", "single_b"}
)
#: Encoder choices accepted by ``model.encoder``.
ENCODER_KINDS: frozenset[str] = frozenset({"convnext", "caformer", "dual"})
#: Decoder MSU variants accepted by ``decoder.msu_mode``. Only ``paper`` is
#: implemented; ``released`` is accepted here and rejected by the model builder.
MSU_MODES: frozenset[str] = frozenset({"paper", "released"})

#: Loss names accepted by :func:`src.losses.build_loss`.
LOSS_NAMES: frozenset[str] = frozenset({"bce_dice", "dice", "jaccard", "bce"})


class ConfigError(ValueError):
    """Raised when a config file is missing keys, has unknown keys, or bad types."""


#: Every key an experiment config may set, with the defaults used when omitted.
DEFAULT_CONFIG: dict[str, Any] = {
    "run": {
        "name": "exp000_base",
        "seed": 42,
        "out_dir": "runs",
    },
    "data": {
        # TODO(dataset): the concrete dataset is not chosen yet - see README
        # "Open questions". These keys describe the folder layout only.
        "root": "data/raw",
        "image_dir": "images",
        "mask_dir": "masks",
        "split_dir": "data/splits",
        "image_ext": ".jpg",
        "mask_ext": ".jpg",
        "mask_suffix": "",
        # Optional npz cache built by scripts/build_cache.py. Empty means "read
        # the image and mask files directly".
        "cache_path": "",
        "image_size": 256,
        "split_ratios": [0.8, 0.1, 0.1],
        "split_seed": 42,
        "mask_threshold": 127,
        "normalize": True,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "num_workers": 2,
        "pin_memory": True,
        # True only for the smoke/test data. Recorded in run_info.json so a run on
        # synthetic images can never be mistaken for a real result.
        "synthetic": False,
    },
    "train": {
        "epochs": 100,
        "batch_size": 8,
        "loss": "bce_dice",
        "bce_weight": 1.0,
        "dice_weight": 1.0,
        "lr_convnext": 3.0e-5,
        "lr_encoder_b": 1.0e-4,
        "lr_head": 3.0e-4,
        "weight_decay": 1.0e-4,
        "warmup_epochs": 0,
        "grad_accum_steps": 1,
        "grad_clip_norm": 0.0,
        "amp": True,
        "channels_last": False,
    },
    "model": {
        "encoder": "dual",
        "encoder_a": "convnext_base.fb_in1k",
        "encoder_b": "caformer_s18.sail_in1k",
        "pretrained_a": True,
        "pretrained_b": True,
        # Fused widths, default to the Meta-Polyp stage widths.
        "channels": [64, 128, 320, 512],
        # Per-stage fusion type for stages 1..4 (strides 4, 8, 16, 32), or one
        # of the presets: hybrid, all_concat, single_a, single_b.
        "fusion": ["concat", "concat", "xattn", "xattn"],
        "fusion_heads": 8,
        "pos_enc_2d": False,
        # Reserved: the paper's Convformer skip block. Not implemented.
        "skip_convformer": False,
    },
    "decoder": {
        "msu": False,
        # paper: D4 x4 -> D2 and D3 x4 -> D1. released: reserved, raises.
        "msu_mode": "paper",
    },
    "logging": {
        "wandb": False,
        "wandb_project": "polyp-dual-encoder",
        "log_images_every": 5,
        "num_log_images": 4,
    },
}

_MISSING = object()


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML experiment config merged onto :data:`DEFAULT_CONFIG`.

    Args:
        path: YAML file, e.g. ``configs/base.yaml``.
        overrides: Optional nested mapping applied last (handy in tests and for
            CLI overrides).

    Returns:
        A new resolved config dict; the module-level defaults are never mutated.

    Raises:
        ConfigError: if the file is missing/empty/invalid, or validation fails.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raise ConfigError(f"Config file is empty: {config_path}")
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}: {config_path}")

    resolved = _deep_merge(DEFAULT_CONFIG, raw)
    if overrides is not None:
        resolved = _deep_merge(resolved, overrides)

    validate_config(resolved)
    return resolved


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto a deep copy of ``base``.

    Args:
        base: Fallback mapping (never mutated).
        override: Values that win.

    Returns:
        Merged dict.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Check structural invariants early, so a bad run fails in a second, not an hour.

    Args:
        cfg: Resolved config to validate.

    Raises:
        ConfigError: with every problem found, joined into one message.
    """
    problems: list[str] = []

    for section in DEFAULT_CONFIG:
        if section not in cfg:
            problems.append(f"missing section '{section}'")
        elif not isinstance(cfg[section], Mapping):
            problems.append(f"section '{section}' must be a mapping")
        else:
            unknown = sorted(set(cfg[section]) - set(DEFAULT_CONFIG[section]))
            if unknown:
                problems.append(f"unknown keys in '{section}': {unknown}")

    unknown_top = sorted(set(cfg) - set(DEFAULT_CONFIG))
    if unknown_top:
        problems.append(f"unknown top-level keys: {unknown_top}")

    if problems:
        raise ConfigError("; ".join(problems))

    run = cfg["run"]
    if not isinstance(run["name"], str) or not run["name"].strip():
        problems.append("run.name must be a non-empty string")
    if not isinstance(run["seed"], int):
        problems.append("run.seed must be an int")

    data = cfg["data"]
    size = data["image_size"]
    if not isinstance(size, int) or size <= 0:
        problems.append("data.image_size must be a positive int")
    elif size % 32 != 0:
        problems.append(f"data.image_size must be divisible by 32, got {size}")
    for key in ("root", "image_dir", "mask_dir", "split_dir", "image_ext", "mask_ext"):
        if not isinstance(data[key], str) or not data[key]:
            problems.append(f"data.{key} must be a non-empty string")
    if not isinstance(data["mask_suffix"], str):
        problems.append("data.mask_suffix must be a string")
    if not isinstance(data["cache_path"], str):
        problems.append("data.cache_path must be a string ('' disables the cache)")
    if not isinstance(data["mask_threshold"], int):
        problems.append("data.mask_threshold must be an int")
    ratios = data["split_ratios"]
    if not isinstance(ratios, (list, tuple)) or len(ratios) != 3:
        problems.append("data.split_ratios must be a list of 3 numbers")
    elif any(float(r) < 0 for r in ratios) or abs(sum(float(r) for r in ratios) - 1.0) > 1e-6:
        problems.append(f"data.split_ratios must be non-negative and sum to 1.0, got {list(ratios)}")
    if not isinstance(data["split_seed"], int):
        problems.append("data.split_seed must be an int")
    if len(data["mean"]) != 3 or len(data["std"]) != 3:
        problems.append("data.mean and data.std must each have 3 values")
    for key in ("normalize", "pin_memory", "synthetic"):
        if not isinstance(data[key], bool):
            problems.append(f"data.{key} must be a bool")

    train = cfg["train"]
    if train["loss"] not in LOSS_NAMES:
        problems.append(f"train.loss must be one of {sorted(LOSS_NAMES)}, got {train['loss']!r}")

    model = cfg["model"]
    if model["encoder"] not in ENCODER_KINDS:
        problems.append(
            f"model.encoder must be one of {sorted(ENCODER_KINDS)}, got {model['encoder']!r}"
        )
    fusion = model["fusion"]
    if isinstance(fusion, str):
        if fusion not in FUSION_MODES:
            problems.append(f"model.fusion must be one of {sorted(FUSION_MODES)}, got {fusion!r}")
    elif isinstance(fusion, (list, tuple)):
        if len(fusion) != 4:
            problems.append(f"model.fusion list must have 4 entries (stages 1-4), got {len(fusion)}")
        for mode in fusion:
            if mode not in STAGE_FUSION_MODES:
                problems.append(
                    f"model.fusion entry {mode!r} not in {sorted(STAGE_FUSION_MODES)}"
                )
    else:
        problems.append("model.fusion must be a string or a list of 4 strings")
    channels = model["channels"]
    if not isinstance(channels, (list, tuple)) or len(channels) != 4:
        problems.append("model.channels must be a list of 4 ints")
    elif any(not isinstance(c, int) or c <= 0 for c in channels):
        problems.append(f"model.channels must be 4 positive ints, got {list(channels)}")
    if not isinstance(model["fusion_heads"], int) or model["fusion_heads"] <= 0:
        problems.append(
            f"model.fusion_heads must be a positive int, got {model['fusion_heads']!r}"
        )
    for key in ("encoder_a", "encoder_b"):
        if not isinstance(model[key], str) or not model[key].strip():
            problems.append(f"model.{key} must be a non-empty timm model name")
    if not isinstance(model["skip_convformer"], bool):
        problems.append("model.skip_convformer must be a bool")
    for key in ("pretrained_a", "pretrained_b", "pos_enc_2d"):
        if not isinstance(model[key], bool):
            problems.append(f"model.{key} must be a bool")

    decoder = cfg["decoder"]
    if not isinstance(decoder["msu"], bool):
        problems.append("decoder.msu must be a bool")
    if decoder["msu_mode"] not in MSU_MODES:
        problems.append(
            f"decoder.msu_mode must be one of {sorted(MSU_MODES)}, got {decoder['msu_mode']!r}"
        )

    if problems:
        raise ConfigError("; ".join(problems))


def get(cfg: Mapping[str, Any], dotted_key: str, default: Any = _MISSING) -> Any:
    """Read a nested config value with a dotted key.

    Args:
        cfg: Config mapping.
        dotted_key: e.g. ``"data.image_size"``.
        default: Returned when the key is absent; if omitted, absence raises.

    Returns:
        The value at ``dotted_key``.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            if default is _MISSING:
                raise KeyError(f"Missing config key: {dotted_key}")
            return default
        node = node[part]
    return node


def save_config(cfg: Mapping[str, Any], path: str | Path) -> Path:
    """Write a resolved config to YAML (used as ``config_resolved.yaml`` per run).

    Args:
        cfg: Resolved config.
        path: Destination file; parent directories are created.

    Returns:
        The written path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(copy.deepcopy(dict(cfg)), handle, sort_keys=False, indent=2)
    return out


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch from the config and make runs repeatable.

    Args:
        seed: Seed value from ``run.seed``.
        deterministic: If True, also force deterministic cuDNN kernels. This is
            slower; on the CPU-only dev box it has no effect on speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
