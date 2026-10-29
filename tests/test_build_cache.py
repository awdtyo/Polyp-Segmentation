"""Tests for the npz cache: builder, loader and dataset equivalence.

All data here is synthetic and written to ``tmp_path``; nothing is downloaded and
no real Kvasir-SEG image is read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.build_cache import build_cache, discover_ids, main
from src.datasets import PolypDataset, load_cache

SIZE = 32


@pytest.fixture()
def cache_source(tmp_path: Path) -> dict[str, object]:
    """Create a tiny image/mask folder with variable sizes and JPEG masks.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        Dict with ``root``, ``ids`` and a two-id ``split_file``.
    """
    root = tmp_path / "raw"
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)

    sizes = {"a_001": (40, 56), "b_002": (33, 47), "c_003": (64, 50)}
    rng = np.random.default_rng(0)
    for index, (sample_id, (height, width)) in enumerate(sizes.items()):
        image = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(images / f"{sample_id}.jpg")
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[5 : height // 2, 6 : width // 2] = 255
        mask[2:4, 2:4] = 200  # still foreground at threshold 127
        Image.fromarray(mask, mode="L").save(masks / f"{sample_id}.jpg", quality=90)

    split_file = tmp_path / "train.txt"
    split_file.write_text("c_003\na_001\n", encoding="utf-8")
    return {"root": root, "ids": sorted(sizes), "split_file": split_file}


def test_discover_ids_finds_pairs_and_rejects_mismatch(cache_source: dict) -> None:
    root: Path = cache_source["root"]

    assert discover_ids(root) == cache_source["ids"]

    (root / "masks" / "orphan.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="without image"):
        discover_ids(root)


def test_build_cache_writes_one_binary_npz(cache_source: dict, tmp_path: Path) -> None:
    out = tmp_path / "cache" / "tiny.npz"

    meta = build_cache(cache_source["root"], out, size=SIZE, progress=False)

    assert out.is_file()
    assert meta["n"] == 3
    assert meta["bytes"] == out.stat().st_size > 0
    with np.load(out) as data:
        assert data["images"].shape == (3, SIZE, SIZE, 3)
        assert data["images"].dtype == np.uint8
        assert data["masks"].shape == (3, SIZE, SIZE)
        assert data["masks"].dtype == np.uint8
        assert set(np.unique(data["masks"]).tolist()) <= {0, 1}
        assert [str(value) for value in data["ids"]] == cache_source["ids"]


def test_cached_dataset_matches_the_raw_pipeline(cache_source: dict, tmp_path: Path) -> None:
    cache_path = tmp_path / "tiny.npz"
    build_cache(cache_source["root"], cache_path, size=SIZE, progress=False)

    raw = PolypDataset(
        cache_source["split_file"],
        cache_source["root"],
        image_size=SIZE,
        normalize=False,
        mask_ext=".jpg",  # the fixture mirrors Kvasir-SEG's JPEG masks
    )
    cached = PolypDataset(
        cache_source["split_file"],
        cache_source["root"],
        image_size=SIZE,
        normalize=False,
        mask_ext=".jpg",
        cache_path=cache_path,
    )

    assert len(raw) == len(cached) == 2
    for index in range(len(raw)):
        raw_sample, cached_sample = raw[index], cached[index]
        assert raw_sample["id"] == cached_sample["id"]
        assert torch.equal(raw_sample["image"], cached_sample["image"])
        assert torch.equal(raw_sample["mask"], cached_sample["mask"])
        assert set(torch.unique(cached_sample["mask"]).tolist()) <= {0.0, 1.0}


def test_cache_id_lookup_does_not_depend_on_row_order(cache_source: dict, tmp_path: Path) -> None:
    cache_path = tmp_path / "tiny.npz"
    # Cache all three ids; the split uses two of them in reverse order.
    build_cache(cache_source["root"], cache_path, size=SIZE, progress=False)
    images, _masks, index = load_cache(cache_path)

    assert len(index) == 3
    assert images.shape[0] == 3
    assert index["c_003"] != index["a_001"]


def test_dataset_rejects_a_split_id_missing_from_cache(cache_source: dict, tmp_path: Path) -> None:
    cache_path = tmp_path / "tiny.npz"
    build_cache(cache_source["root"], cache_path, size=SIZE, ids=["a_001", "b_002"], progress=False)

    with pytest.raises(ValueError, match="missing from cache"):
        PolypDataset(
            cache_source["split_file"],
            cache_source["root"],
            image_size=SIZE,
            cache_path=cache_path,
        )


def test_missing_cache_file_has_a_helpful_error(cache_source: dict, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_cache.py"):
        PolypDataset(
            cache_source["split_file"],
            cache_source["root"],
            image_size=SIZE,
            cache_path=tmp_path / "nope.npz",
        )


def test_cli_builds_a_cache(cache_source: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cli.npz"

    code = main(
        [
            "--root",
            str(cache_source["root"]),
            "--out",
            str(out),
            "--size",
            str(SIZE),
            "--limit",
            "2",
        ]
    )

    assert code == 0
    assert out.is_file()
    assert "n=2" in capsys.readouterr().out
