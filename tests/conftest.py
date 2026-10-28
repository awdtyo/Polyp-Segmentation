"""Shared fixtures: a tiny synthetic polyp dataset built in a temp folder.

No real data is committed or downloaded. Every test that needs images uses this
fixture, which is what AGENTS.md section 6 asks for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

#: Native size of the synthetic images, deliberately not square and not 256.
SYNTHETIC_HW: tuple[int, int] = (40, 56)  # (height, width)


def _write_rgb(path: Path, seed: int) -> None:
    """Write a deterministic RGB image.

    Args:
        path: Destination path.
        seed: Value used to vary the pixels.
    """
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(*SYNTHETIC_HW, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _write_mask(path: Path, box: tuple[int, int, int, int], mode: str = "L") -> None:
    """Write a white rectangle on black, as a binary mask.

    Args:
        path: Destination path.
        box: ``(top, left, height, width)`` of the foreground rectangle.
        mode: ``"L"`` for grayscale, ``"RGB"`` for a red-on-black mask.
    """
    mask = np.zeros(SYNTHETIC_HW, dtype=np.uint8)
    top, left, height, width = box
    mask[top : top + height, left : left + width] = 255
    array = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1) if mode == "RGB" else mask
    Image.fromarray(array, mode=mode).save(path)


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> dict[str, object]:
    """Create ``<root>/images``, ``<root>/masks`` and a split file with 3 ids.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        Dict with ``root``, ``split_file``, ``split_dir``, ``ids`` and ``boxes``
        (foreground rectangle per id, in native pixel coordinates).
    """
    root = tmp_path / "data"
    images = root / "images"
    masks = root / "masks"
    splits = tmp_path / "splits"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    splits.mkdir(parents=True, exist_ok=True)

    boxes = {
        "caseA_001": (10, 10, 12, 12),
        "caseA_002": (0, 0, 8, 20),
        "caseB_001": (20, 30, 10, 10),
    }
    for index, (sample_id, box) in enumerate(boxes.items()):
        _write_rgb(images / f"{sample_id}.jpg", seed=index)
        _write_mask(masks / f"{sample_id}.png", box=box)

    split_file = splits / "train.txt"
    split_file.write_text("# synthetic\n\ncaseA_001\ncaseA_002\n", encoding="utf-8")
    return {
        "root": root,
        "split_file": split_file,
        "split_dir": splits,
        "ids": list(boxes),
        "boxes": boxes,
    }
