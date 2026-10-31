"""Tests for ``scripts/prepare_datasets.py``.

Everything here is synthetic and written to ``tmp_path``: nothing downloads and
no real dataset image is read. The prep writes into the temp data root, so the
repository's ``data/`` tree is untouched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.prepare_datasets import (
    PrepareError,
    build_npz,
    discover_pairs,
    main,
    prepare_dataset,
    read_meta,
)
from src.data_registry import get_spec

SIZE = 32


def make_raw(
    data_root: Path,
    name: str = "kvasir_seg",
    *,
    count: int = 3,
    image_ext: str = ".png",
    mask_ext: str = ".png",
    mask_empty_for: set[str] | None = None,
    extra_image: str | None = None,
) -> Path:
    """Create a tiny synthetic raw dataset.

    Args:
        data_root: Data-tree root.
        name: Dataset name (must exist in the registry).
        count: Number of paired samples.
        image_ext: Extension for images.
        mask_ext: Extension for masks (may differ from ``image_ext``).
        mask_empty_for: Ids whose mask is entirely background.
        extra_image: Optional extra image id with no matching mask.

    Returns:
        The dataset's raw folder.
    """
    spec = get_spec(name)
    raw = spec.raw_dir(data_root)
    images = raw / spec.image_subdir
    masks = raw / spec.mask_subdir
    images.mkdir(parents=True)
    masks.mkdir(parents=True)

    empty = mask_empty_for or set()
    for index in range(count):
        sample_id = f"case_{index:03d}"
        array = np.full((20, 24, 3), 40 * (index + 1), dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(images / f"{sample_id}{image_ext}")
        mask = np.zeros((20, 24), dtype=np.uint8) if sample_id in empty else np.full((20, 24), 255, dtype=np.uint8)
        Image.fromarray(mask, mode="L").save(masks / f"{sample_id}{mask_ext}")
    if extra_image is not None:
        Image.fromarray(np.zeros((20, 24, 3), dtype=np.uint8), mode="RGB").save(
            images / f"{extra_image}{image_ext}"
        )
    return raw


def test_discover_pairs_matches_by_stem_across_extensions(tmp_path: Path) -> None:
    make_raw(tmp_path, count=2, image_ext=".tif", mask_ext=".png")

    pairs = discover_pairs(get_spec("kvasir_seg"), tmp_path)

    assert [pair.sample_id for pair in pairs] == ["case_000", "case_001"]
    assert {pair.image.suffix for pair in pairs} == {".tif"}
    assert {pair.mask.suffix for pair in pairs} == {".png"}


def test_discover_pairs_reports_an_unmatched_image(tmp_path: Path) -> None:
    make_raw(tmp_path, count=1, extra_image="orphan")

    with pytest.raises(PrepareError, match="without a mask"):
        discover_pairs(get_spec("kvasir_seg"), tmp_path)


def test_discover_pairs_reports_a_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(PrepareError, match="raw folder not found"):
        discover_pairs(get_spec("kvasir_seg"), tmp_path)


def test_prepare_dataset_writes_npz_and_meta(tmp_path: Path) -> None:
    make_raw(tmp_path, count=3)

    meta = prepare_dataset(get_spec("kvasir_seg"), tmp_path, size=SIZE, progress=False)

    npz = np.load(tmp_path / "processed" / "kvasir_seg" / "data.npz")
    assert npz.files == ["ids", "images", "masks"]
    assert npz["images"].shape == (3, SIZE, SIZE, 3)
    assert npz["masks"].shape == (3, SIZE, SIZE)
    assert npz["images"].dtype == np.uint8
    assert npz["masks"].dtype == np.uint8
    assert set(np.unique(npz["masks"])) <= {0, 1}
    assert [str(value) for value in npz["ids"]] == ["case_000", "case_001", "case_002"]

    assert meta["name"] == "kvasir_seg"
    assert meta["source"] == "Kvasir-SEG"
    assert meta["image_count"] == 3
    assert meta["expected_count"] == 1000
    assert meta["count_matches_expected"] is False  # 3 != 1000
    assert meta["image_size"] == SIZE
    assert meta["mask_threshold"] == 127
    assert meta["image_extensions"] == [".png"]
    assert meta["original_resolutions"]["most_common"] == "24x20"
    assert read_meta(tmp_path / "processed" / "kvasir_seg" / "meta.json") == meta


def test_prepare_dataset_is_idempotent_and_force_rebuilds(tmp_path: Path) -> None:
    make_raw(tmp_path, count=2)

    first = prepare_dataset(get_spec("kvasir_seg"), tmp_path, size=SIZE, progress=False)
    assert "skipped" not in first

    second = prepare_dataset(get_spec("kvasir_seg"), tmp_path, size=SIZE, progress=False)
    assert second.get("skipped") is True

    third = prepare_dataset(get_spec("kvasir_seg"), tmp_path, size=SIZE, force=True, progress=False)
    assert third.get("skipped") is not True or third["image_count"] == 2


def test_prepare_dataset_refuses_a_different_size_without_force(tmp_path: Path) -> None:
    make_raw(tmp_path, count=1)
    prepare_dataset(get_spec("kvasir_seg"), tmp_path, size=SIZE, progress=False)

    with pytest.raises(PrepareError, match="was prepared at"):
        prepare_dataset(get_spec("kvasir_seg"), tmp_path, size=SIZE * 2, progress=False)


def test_prepare_dataset_fails_loudly_on_an_empty_mask(tmp_path: Path) -> None:
    make_raw(tmp_path, count=2, mask_empty_for={"case_001"})

    with pytest.raises(PrepareError, match="empty masks"):
        prepare_dataset(get_spec("kvasir_seg"), tmp_path, size=SIZE, progress=False)


def test_prepare_dataset_warns_but_continues_on_a_count_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_raw(tmp_path, count=2)

    meta = prepare_dataset(get_spec("kvasir_seg"), tmp_path, size=SIZE, progress=False)

    assert meta["count_matches_expected"] is False
    assert "expects 1000" in capsys.readouterr().err


def test_build_npz_masks_are_binary(tmp_path: Path) -> None:
    make_raw(tmp_path, count=2)
    pairs = discover_pairs(get_spec("kvasir_seg"), tmp_path)

    ids, images, masks, resolutions = build_npz(
        get_spec("kvasir_seg"), pairs, size=SIZE, mask_threshold=127, progress=False
    )

    assert images.shape == (2, SIZE, SIZE, 3)
    assert set(np.unique(masks)) <= {0, 1}
    assert resolutions == [(24, 20), (24, 20)]
    assert list(ids) == ["case_000", "case_001"]


def test_main_skips_absent_datasets_and_prepares_present_ones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_raw(tmp_path, name="kvasir_seg", count=2)

    code = main(["--data-root", str(tmp_path), "--size", str(SIZE), "--quiet"])

    assert code == 0
    assert (tmp_path / "processed" / "kvasir_seg" / "data.npz").is_file()
    assert not (tmp_path / "processed" / "cvc_clinicdb").exists()
    assert "not on disk" in capsys.readouterr().err


def test_main_fails_when_an_explicit_dataset_is_absent(tmp_path: Path) -> None:
    make_raw(tmp_path, name="kvasir_seg", count=1)

    code = main(["--data-root", str(tmp_path), "--dataset", "etis", "--quiet"])

    assert code == 1


def test_main_unknown_dataset_is_usage_error(tmp_path: Path) -> None:
    code = main(["--data-root", str(tmp_path), "--dataset", "nope", "--quiet"])
    assert code == 2
