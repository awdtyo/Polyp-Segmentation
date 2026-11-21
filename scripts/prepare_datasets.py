"""Prepare every registered dataset into ``data/processed/<name>/``.

For each dataset the raw folder ``<DATA_ROOT>/raw/<folder>`` is read once and
written as ``<DATA_ROOT>/processed/<name>/data.npz`` plus a ``meta.json``. The
npz has exactly the format the Kvasir cache uses (``scripts/build_cache.py``),
so ``PolypDataset`` reads it unchanged:

* ``ids``     ``(N,)`` unicode
* ``images``  ``(N, S, S, 3)`` uint8, resized with bilinear interpolation
* ``masks``   ``(N, S, S)`` uint8 in {0, 1}, binarised at ``mask_threshold``
  *before* resizing with nearest-neighbour

Images and masks pair up by file stem (the optional ``mask_suffix`` is stripped
from the mask stem first), so a dataset whose images and masks use different
extensions is handled. Extensions are discovered from disk, never assumed.

Usage::

    # every dataset present on disk (absent ones are skipped with a warning)
    python scripts/prepare_datasets.py

    # one dataset that must be present, failing loudly if it is not
    python scripts/prepare_datasets.py --dataset cvc_clinicdb

    # honour a run's preprocessing (so a checkpoint is evaluated as it trained)
    python scripts/prepare_datasets.py --config configs/exp003_caformer_s18_baseline.yaml

Safe to rerun: a dataset whose ``data.npz`` already exists at the requested size
is skipped. ``--force`` rebuilds it.

Loud failures (never silently fixed): unmatched image/mask pairs, empty masks,
mask values other than {0, 1} after thresholding. A count that differs from the
registry's ``expected_count`` is reported as a prominent warning and recorded in
``meta.json``; it is not raised, because a mismatch is a data problem for the
user to inspect, not something to "fix".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

# When run as ``python scripts/prepare_datasets.py`` the repo root is not on
# sys.path (sys.path[0] is scripts/), so add it before importing src.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_registry import (  # noqa: E402  (import after the sys.path fix)
    DEFAULT_DATA_ROOT,
    DatasetSpec,
    RegistryError,
    get_spec,
    load_registry,
    resolve_data_root,
)

#: npz and metadata file names inside a processed dataset folder.
DATA_FILENAME = "data.npz"
META_FILENAME = "meta.json"
#: Defaults matching the Kvasir cache and AGENTS.md section 6.
DEFAULT_SIZE = 256
DEFAULT_MASK_THRESHOLD = 127
IMAGE_RESAMPLE = "bilinear"
MASK_RESAMPLE = "nearest"

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


class PrepareError(RuntimeError):
    """Preparation cannot proceed (missing folders, unmatched pairs, bad masks)."""


@dataclass(frozen=True)
class Pair:
    """One image/mask pair, matched by stem.

    Attributes:
        sample_id: Shared file stem.
        image: Image path.
        mask: Mask path.
    """

    sample_id: str
    image: Path
    mask: Path


def discover_pairs(spec: DatasetSpec, data_root: str | Path) -> list[Pair]:
    """List every image/mask pair under a dataset's raw folder, sorted by id.

    Extensions are discovered from disk (a dataset with ``.tif`` images and
    ``.png`` masks works). A file whose stem has no counterpart, or a stem that
    appears twice in one folder, is a loud failure.

    Args:
        spec: Registry entry.
        data_root: Data-tree root.

    Returns:
        Pairs sorted by ``sample_id``.

    Raises:
        PrepareError: if the raw folders are missing, a pair is unmatched, a
            stem is ambiguous (two files, one stem) or nothing is found.
    """
    image_folder = spec.image_dir(data_root)
    mask_folder = spec.mask_dir(data_root)
    for folder in (image_folder, mask_folder):
        if not folder.is_dir():
            raise PrepareError(f"raw folder not found: {folder}")

    def by_stem(folder: Path, suffix: str) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            stem = path.stem
            if suffix:
                if not stem.endswith(suffix):
                    continue
                stem = stem[: -len(suffix)]
            if stem in found:
                raise PrepareError(
                    f"ambiguous stem {stem!r} in {folder}: {found[stem].name} and {path.name}"
                )
            found[stem] = path
        return found

    images = by_stem(image_folder, "")
    masks = by_stem(mask_folder, spec.mask_suffix)
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise PrepareError(
            f"{spec.name}: {len(images)} images, {len(masks)} masks; "
            f"{len(missing_masks)} without a mask {missing_masks[:3]}, "
            f"{len(missing_images)} without an image {missing_images[:3]}"
        )
    if not images:
        raise PrepareError(f"{spec.name}: no image/mask pairs under {image_folder}")
    return [Pair(sample_id, images[sample_id], masks[sample_id]) for sample_id in sorted(images)]


def _load_image(path: Path, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Load an image as RGB uint8, resized to ``size`` x ``size``.

    Args:
        path: Image path.
        size: Target side length.

    Returns:
        ``((size, size, 3) uint8 array, (original_width, original_height))``.
    """
    with Image.open(path) as handle:
        original = handle.size
        image = handle.convert("RGB").resize((size, size), resample=Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8), (int(original[0]), int(original[1]))


def _load_mask(path: Path, size: int, threshold: int) -> np.ndarray:
    """Load a mask, binarise it, then resize it to ``size`` x ``size``.

    Args:
        path: Mask path.
        size: Target side length.
        threshold: Pixel value above which a pixel is foreground.

    Returns:
        ``(size, size)`` uint8 array in {0, 1}.

    Raises:
        PrepareError: if the mask is neither 2D nor a stack of channels.
    """
    with Image.open(path) as handle:
        mask = np.asarray(handle)
    if mask.ndim == 3:
        mask = mask.max(axis=2)  # white-on-black RGB and palette masks both work
    if mask.ndim != 2:
        raise PrepareError(f"mask must be 2D after channel reduction, got {mask.shape}: {path}")
    binary = (mask > threshold).astype(np.uint8)
    resized = Image.fromarray(binary * 255, mode="L").resize((size, size), resample=Image.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8)


def _resolution_summary(resolutions: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Summarise a list of ``(width, height)`` pairs.

    Args:
        resolutions: Original sizes, one per image.

    Returns:
        ``{"n", "n_distinct", "min_width", "max_width", "min_height",
        "max_height", "most_common"}``, or a zeroed dict when empty.
    """
    if not resolutions:
        return {
            "n": 0,
            "n_distinct": 0,
            "min_width": 0,
            "max_width": 0,
            "min_height": 0,
            "max_height": 0,
            "most_common": None,
        }
    widths = [w for w, _ in resolutions]
    heights = [h for _, h in resolutions]
    counts: dict[tuple[int, int], int] = {}
    for size in resolutions:
        counts[size] = counts.get(size, 0) + 1
    most_common = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    return {
        "n": len(resolutions),
        "n_distinct": len(counts),
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "most_common": f"{most_common[0]}x{most_common[1]}",
    }


def read_meta(path: str | Path) -> dict[str, Any] | None:
    """Read a ``meta.json`` if present.

    Args:
        path: Path to the metadata file.

    Returns:
        The parsed dict, or ``None`` when the file is missing or unparseable.
    """
    meta_path = Path(path)
    if not meta_path.is_file():
        return None
    try:
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def write_meta(path: str | Path, meta: Mapping[str, Any]) -> Path:
    """Write ``meta.json``.

    Args:
        path: Destination; parents are created.
        meta: Metadata mapping.

    Returns:
        The written path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta), indent=2), encoding="utf-8")
    return out


def build_npz(
    spec: DatasetSpec,
    pairs: Sequence[Pair],
    *,
    size: int,
    mask_threshold: int,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Load every pair into ``(ids, images, masks)`` arrays.

    Args:
        spec: Registry entry (used in error messages).
        pairs: Pairs to load.
        size: Square output size.
        mask_threshold: Foreground threshold, applied before resizing.
        progress: Print progress every 100 samples.

    Returns:
        ``(ids, images, masks, resolutions)`` where ``resolutions`` is the
        original ``(width, height)`` of each image.

    Raises:
        PrepareError: on an empty mask or a mask value outside {0, 1}.
    """
    ids = np.asarray([pair.sample_id for pair in pairs], dtype=np.str_)
    images = np.empty((len(pairs), size, size, 3), dtype=np.uint8)
    masks = np.empty((len(pairs), size, size), dtype=np.uint8)
    resolutions: list[tuple[int, int]] = []
    started = time.time()

    for row, pair in enumerate(pairs):
        images[row], original = _load_image(pair.image, size)
        masks[row] = _load_mask(pair.mask, size, mask_threshold)
        resolutions.append(original)
        if progress and (row + 1) % 100 == 0:
            print(f"  {row + 1}/{len(pairs)} ({time.time() - started:.1f}s)", flush=True)

    empty = [str(ids[i]) for i in range(len(ids)) if int(masks[i].sum()) == 0]
    if empty:
        raise PrepareError(
            f"{spec.name}: {len(empty)} empty masks (no foreground after thresholding "
            f">{mask_threshold}), e.g. {empty[:5]}"
        )
    values = {int(v) for v in np.unique(masks)}
    if not values <= {0, 1}:
        raise PrepareError(
            f"{spec.name}: masks must be binary after thresholding, found values {sorted(values)}"
        )
    return ids, images, masks, resolutions


def prepare_dataset(
    spec: DatasetSpec,
    data_root: str | Path,
    *,
    size: int = DEFAULT_SIZE,
    mask_threshold: int = DEFAULT_MASK_THRESHOLD,
    out_root: str | Path | None = None,
    force: bool = False,
    limit: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Prepare one dataset, or skip it when it is already prepared.

    Args:
        spec: Registry entry.
        data_root: Data-tree root.
        size: Square output size.
        mask_threshold: Foreground threshold, applied before resizing.
        out_root: Processed-tree root (default ``<data_root>/processed``).
        force: Rebuild even when ``data.npz`` already exists.
        limit: Debug aid; only prepare the first N pairs.
        progress: Print per-100 progress lines.

    Returns:
        The dataset's ``meta.json`` payload, plus ``"skipped": True`` when an
        existing cache was kept.

    Raises:
        PrepareError: if the raw data is missing or invalid, or an existing
            cache was built at a different size (rerun with ``force``).
    """
    processed = (
        Path(out_root) / spec.name if out_root is not None else spec.processed_dir(data_root)
    )
    npz_path = processed / DATA_FILENAME
    meta_path = processed / META_FILENAME

    if npz_path.is_file() and not force:
        existing = read_meta(meta_path)
        existing_size = int(existing.get("image_size", -1)) if existing else -1
        if existing_size != size:
            raise PrepareError(
                f"{spec.name}: {npz_path} was prepared at {existing_size}px but {size}px was "
                f"requested. Rerun with --force to rebuild it at {size}px."
            )
        if progress:
            print(f"{spec.name}: already prepared at {size}px, skipping ({npz_path})", flush=True)
        return {**(existing or {}), "skipped": True}

    pairs = discover_pairs(spec, data_root)
    found_count = len(pairs)
    if limit is not None:
        pairs = pairs[:limit]
    ids, images, masks, resolutions = build_npz(
        spec, pairs, size=size, mask_threshold=mask_threshold, progress=progress
    )

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, ids=ids, images=images, masks=masks)

    mask_exts = sorted({pair.mask.suffix for pair in pairs})
    image_exts = sorted({pair.image.suffix for pair in pairs})
    count_matches = found_count == spec.expected_count
    meta: dict[str, Any] = {
        "name": spec.name,
        "source": spec.source or spec.name,
        "raw_folder": str(spec.raw_dir(data_root)),
        "processed_dir": str(processed),
        "data_file": str(npz_path),
        "image_count": len(pairs),
        "expected_count": spec.expected_count,
        "count_matches_expected": count_matches,
        "original_resolutions": _resolution_summary(resolutions),
        "image_size": int(size),
        "image_resample": IMAGE_RESAMPLE,
        "mask_resample": MASK_RESAMPLE,
        "mask_threshold": int(mask_threshold),
        "image_extensions": image_exts,
        "mask_extensions": mask_exts,
        "mask_suffix": spec.mask_suffix,
        "npz_bytes": npz_path.stat().st_size,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_meta(meta_path, meta)

    if not count_matches:
        print(
            f"WARNING: {spec.name}: found {found_count} pairs but the registry expects "
            f"{spec.expected_count}. Recorded in {meta_path}; inspect the raw data.",
            file=sys.stderr,
            flush=True,
        )
    if progress:
        print(
            f"{spec.name}: {len(pairs)} pairs -> {npz_path} "
            f"({meta['npz_bytes'] / 1e6:.1f} MB, {size}x{size})"
            f"{' [limit applied]' if limit is not None else ''}",
            flush=True,
        )
    return meta


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Prepare raw datasets into data/processed/<name>/.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="NAME",
        help="Prepare this dataset; must be present (repeatable). Default: all present datasets.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Run config; reads data.size/threshold.")
    parser.add_argument("--size", type=int, default=None, help=f"Output size (default: {DEFAULT_SIZE}).")
    parser.add_argument(
        "--mask-threshold", type=int, default=None, help=f"Mask threshold (default: {DEFAULT_MASK_THRESHOLD})."
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Override DATA_ROOT.")
    parser.add_argument("--out-root", type=Path, default=None, help="Override the processed tree root.")
    parser.add_argument("--limit", type=int, default=None, help="Only prepare the first N pairs (debug).")
    parser.add_argument("--force", action="store_true", help="Rebuild even when data.npz exists.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-dataset progress lines.")
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_settings(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve the output size and mask threshold from the CLI and an optional config.

    Args:
        args: Parsed CLI arguments.

    Returns:
        ``(size, mask_threshold)``.

    Raises:
        SystemExit(2): if ``--config`` cannot be loaded.
    """
    size: int = DEFAULT_SIZE
    threshold: int = DEFAULT_MASK_THRESHOLD
    if args.config is not None:
        from src.utils import ConfigError, load_config

        try:
            cfg = load_config(args.config)
        except ConfigError as exc:
            print(f"Config error: {exc}", file=sys.stderr)
            raise SystemExit(EXIT_USAGE) from exc
        size = int(cfg["data"]["image_size"])
        threshold = int(cfg["data"]["mask_threshold"])
    if args.size is not None:
        size = int(args.size)
    if args.mask_threshold is not None:
        threshold = int(args.mask_threshold)
    return size, threshold


def main(argv: Iterable[str] | None = None) -> int:
    """Prepare the requested datasets.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 on a data failure, 2 on bad input.
    """
    args = parse_args(argv)
    size, threshold = _resolve_settings(args)
    progress = not args.quiet

    try:
        registry = load_registry()
    except RegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE

    data_root = args.data_root if args.data_root is not None else resolve_data_root()

    if args.dataset:
        selected: list[DatasetSpec] = []
        for name in args.dataset:
            if name not in registry:
                print(f"ERROR: unknown dataset {name!r}; known: {sorted(registry)}", file=sys.stderr)
                return EXIT_USAGE
            spec = registry[name]
            if not spec.is_present(data_root):
                print(
                    f"ERROR: dataset {name!r} was requested but is not on disk at "
                    f"{spec.raw_dir(data_root)}",
                    file=sys.stderr,
                )
                return EXIT_FAILURE
            selected.append(spec)
    else:
        selected = [spec for spec in registry.values() if spec.is_present(data_root)]
        absent = [spec.name for spec in registry.values() if not spec.is_present(data_root)]
        if absent:
            print(
                f"skipping {len(absent)} dataset(s) not on disk: {absent} "
                f"(under {Path(data_root) / 'raw'})",
                file=sys.stderr,
                flush=True,
            )
        if not selected:
            print(f"ERROR: no dataset on disk under {Path(data_root) / 'raw'}", file=sys.stderr)
            return EXIT_FAILURE

    prepared = 0
    for spec in selected:
        try:
            prepare_dataset(
                spec,
                data_root,
                size=size,
                mask_threshold=threshold,
                out_root=args.out_root,
                force=bool(args.force),
                limit=args.limit,
                progress=progress,
            )
        except PrepareError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_FAILURE
        prepared += 1

    if progress:
        print(f"prepared {prepared} dataset(s) at {size}px under {Path(data_root) / 'processed'}", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
