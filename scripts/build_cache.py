"""Resize all Kvasir-SEG images and masks once and store a single ``.npz`` cache.

Why: every epoch re-decodes 1000 JPEGs and resizes them, which is the slowest
part of a CPU-bound Colab data pipeline. The cache is built offline, is not
committed (``data/*`` is git-ignored), and changes neither the split nor the
labels: images are resized with bilinear interpolation, masks are thresholded
*before* resizing and resized with nearest-neighbour, so they stay exactly 0/1.

Usage:
    python scripts/build_cache.py --config configs/exp001_convnext_base_baseline.yaml \
        --out data/cache/kvasir_seg_256.npz

    python scripts/build_cache.py --root data/raw --size 256 \
        --out data/cache/kvasir_seg_256.npz

The resulting file is read by :class:`src.datasets.PolypDataset` when
``data.cache_path`` points at it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

#: Default location, relative to the repository root.
DEFAULT_OUT: str = "data/cache/kvasir_seg_256.npz"


def discover_ids(
    root: str | Path,
    *,
    image_dir: str = "images",
    mask_dir: str = "masks",
    image_ext: str = ".jpg",
    mask_ext: str = ".jpg",
    mask_suffix: str = "",
) -> list[str]:
    """List sample ids that have both an image and a mask.

    Args:
        root: Folder containing the image and mask folders.
        image_dir: Image sub-folder.
        mask_dir: Mask sub-folder.
        image_ext: Image extension, e.g. ``".jpg"``.
        mask_ext: Mask extension.
        mask_suffix: Suffix between id and mask extension.

    Returns:
        Sorted list of ids present in both folders.

    Raises:
        FileNotFoundError: if either folder is missing.
        ValueError: if no complete pairs are found, or one folder has ids the
            other lacks (a silent mismatch would corrupt the cache order).
    """
    root_path = Path(root)
    image_folder = root_path / image_dir
    mask_folder = root_path / mask_dir
    for folder in (image_folder, mask_folder):
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

    def stems(folder: Path, extension: str, suffix: str) -> set[str]:
        """Collect stems for ``<stem><suffix><extension>`` files.

        Args:
            folder: Folder to scan.
            extension: File extension.
            suffix: Required suffix after the stem (usually empty).

        Returns:
            Set of stems.
        """
        found: set[str] = set()
        for path in folder.glob(f"*{extension}"):
            if not path.is_file() or path.name.startswith("."):
                continue
            stem = path.stem
            if suffix and not stem.endswith(suffix):
                continue
            found.add(stem[: -len(suffix)] if suffix else stem)
        return found

    image_ids = stems(image_folder, image_ext, "")
    mask_ids = stems(mask_folder, mask_ext, mask_suffix)

    missing_masks = sorted(image_ids - mask_ids)
    missing_images = sorted(mask_ids - image_ids)
    if missing_masks or missing_images:
        raise ValueError(
            f"Image/mask mismatch under {root_path}: {len(image_ids)} images, "
            f"{len(mask_ids)} masks, {len(missing_masks)} without mask, "
            f"{len(missing_images)} without image (first missing mask: "
            f"{missing_masks[:3]}, first missing image: {missing_images[:3]})"
        )
    if not image_ids:
        raise ValueError(f"No image/mask pairs found under {root_path}")
    return sorted(image_ids)


def _load_image(path: Path, size: int) -> np.ndarray:
    """Load an image as RGB uint8 and resize it.

    Args:
        path: Image path.
        size: Target side length.

    Returns:
        ``(size, size, 3)`` uint8 array.
    """
    with Image.open(path) as handle:
        image = handle.convert("RGB").resize((size, size), resample=Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def _load_mask(path: Path, size: int, threshold: int) -> np.ndarray:
    """Load a mask, binarise it, then resize it.

    Args:
        path: Mask path.
        size: Target side length.
        threshold: Pixel value above which a pixel is foreground.

    Returns:
        ``(size, size)`` uint8 array in {0, 1}.

    Notes:
        Thresholding happens before resizing (AGENTS.md section 6); nearest
        neighbour keeps the result binary.
    """
    with Image.open(path) as handle:
        mask = np.asarray(handle)
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D after channel reduction, got {mask.shape}: {path}")
    binary = (mask > threshold).astype(np.uint8)
    resized = Image.fromarray(binary * 255, mode="L").resize((size, size), resample=Image.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8)


def build_cache(
    root: str | Path,
    out_path: str | Path,
    *,
    size: int = 256,
    image_dir: str = "images",
    mask_dir: str = "masks",
    image_ext: str = ".jpg",
    mask_ext: str = ".jpg",
    mask_suffix: str = "",
    mask_threshold: int = 127,
    ids: Sequence[str] | None = None,
    progress: bool = True,
) -> dict[str, object]:
    """Build one ``.npz`` cache of resized images and binary masks.

    Args:
        root: Folder containing the image and mask folders.
        out_path: Destination ``.npz`` file; parent directories are created.
        size: Square output size, e.g. 256.
        image_dir: Image sub-folder.
        mask_dir: Mask sub-folder.
        image_ext: Image extension.
        mask_ext: Mask extension.
        mask_suffix: Suffix between id and mask extension.
        mask_threshold: Foreground threshold, applied before resizing.
        ids: Optional explicit ids (must be a subset of the pairs on disk).
            Default: every image/mask pair, sorted.
        progress: Print progress every 100 samples.

    Returns:
        Metadata: ``path``, ``n``, ``size``, ``bytes``, ``seconds``,
        ``image_dtype`` and ``mask_dtype``.

    Raises:
        ValueError: on an empty or invalid id list, or an unknown id.
    """
    root_path = Path(root)
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    available = discover_ids(
        root_path,
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_ext=image_ext,
        mask_ext=mask_ext,
        mask_suffix=mask_suffix,
    )
    if ids is None:
        chosen = available
    else:
        chosen = list(ids)
        unknown = sorted(set(chosen) - set(available))
        if unknown:
            raise ValueError(f"{len(unknown)} ids have no image/mask pair, e.g. {unknown[:3]}")
        if not chosen:
            raise ValueError("No ids requested for the cache")
        if len(set(chosen)) != len(chosen):
            raise ValueError("Duplicate ids passed to build_cache")

    images = np.empty((len(chosen), size, size, 3), dtype=np.uint8)
    masks = np.empty((len(chosen), size, size), dtype=np.uint8)
    started = time.time()

    for row, sample_id in enumerate(chosen):
        image_path = root_path / image_dir / f"{sample_id}{image_ext}"
        mask_path = root_path / mask_dir / f"{sample_id}{mask_suffix}{mask_ext}"
        images[row] = _load_image(image_path, size)
        masks[row] = _load_mask(mask_path, size, mask_threshold)
        if progress and (row + 1) % 100 == 0:
            elapsed = time.time() - started
            print(f"  {row + 1}/{len(chosen)} ({elapsed:.1f}s)", flush=True)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        ids=np.asarray(chosen, dtype=np.str_),
        images=images,
        masks=masks,
    )
    elapsed = time.time() - started
    metadata: dict[str, object] = {
        "path": str(out),
        "n": len(chosen),
        "size": int(size),
        "bytes": out.stat().st_size,
        "seconds": round(elapsed, 2),
        "image_dtype": str(images.dtype),
        "mask_dtype": str(masks.dtype),
    }
    if progress:
        print(
            f"cache: {out}  n={len(chosen)}  size={size}x{size}  "
            f"file={out.stat().st_size / 1e6:.1f} MB  built in {elapsed:.1f}s",
            flush=True,
        )
    return metadata


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Build the resized npz cache.")
    parser.add_argument("--config", type=Path, default=None, help="Experiment YAML; reads data.* from it.")
    parser.add_argument("--root", type=Path, default=None, help="Override data.root.")
    parser.add_argument("--out", type=Path, default=None, help=f"Output npz (default: {DEFAULT_OUT}).")
    parser.add_argument("--size", type=int, default=None, help="Override data.image_size (default 256).")
    parser.add_argument("--limit", type=int, default=None, help="Only cache the first N ids (debug).")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """Build the cache from CLI arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 2 on a configuration/data error.
    """
    args = parse_args(argv)

    data: Mapping[str, object]
    if args.config is not None:
        # When launched as ``python scripts/build_cache.py`` the repo root is not
        # on sys.path (sys.path[0] is scripts/), so add it before importing src.
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from src.utils import ConfigError, load_config

        try:
            cfg = load_config(args.config)
        except ConfigError as exc:
            print(f"Config error: {exc}", file=sys.stderr)
            return 2
        data = cfg["data"]
    else:
        data = {
            "root": "data/raw",
            "image_dir": "images",
            "mask_dir": "masks",
            "image_ext": ".jpg",
            "mask_ext": ".jpg",
            "mask_suffix": "",
            "mask_threshold": 127,
            "image_size": 256,
        }
        if args.root is None:
            print("Pass --config or --root.", file=sys.stderr)
            return 2

    root = args.root if args.root is not None else Path(str(data["root"]))
    size = args.size if args.size is not None else int(data["image_size"])  # type: ignore[arg-type]
    out = args.out if args.out is not None else Path(DEFAULT_OUT)

    try:
        ids = discover_ids(
            root,
            image_dir=str(data["image_dir"]),
            mask_dir=str(data["mask_dir"]),
            image_ext=str(data["image_ext"]),
            mask_ext=str(data["mask_ext"]),
            mask_suffix=str(data["mask_suffix"]),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        return 2
    if args.limit is not None:
        ids = ids[: args.limit]

    build_cache(
        root,
        out,
        size=size,
        image_dir=str(data["image_dir"]),
        mask_dir=str(data["mask_dir"]),
        image_ext=str(data["image_ext"]),
        mask_ext=str(data["mask_ext"]),
        mask_suffix=str(data["mask_suffix"]),
        mask_threshold=int(data["mask_threshold"]),  # type: ignore[arg-type]
        ids=ids,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
