"""Tests for the dataset: loading, binarisation, resizing, normalisation, transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.datasets import PolypDataset, read_split
from src.utils import IMAGENET_MEAN, IMAGENET_STD, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SIZE = 32


def make_dataset(fixture: dict, *, image_size: int = SIZE, transform=None, normalize: bool = False, **kwargs) -> PolypDataset:
    """Build a dataset on the synthetic fixture.

    Args:
        fixture: The ``synthetic_dataset`` fixture dict.
        image_size: Output size.
        transform: Optional transform callable.
        normalize: Whether to apply ImageNet normalisation.
        **kwargs: Extra ``PolypDataset`` keyword arguments.

    Returns:
        A dataset over the fixture's 2 ids.
    """
    return PolypDataset(
        split_file=fixture["split_file"],
        root=fixture["root"],
        transform=transform,
        image_size=image_size,
        normalize=normalize,
        **kwargs,
    )


def test_len_matches_split(synthetic_dataset: dict) -> None:
    dataset = make_dataset(synthetic_dataset)
    assert len(dataset) == 2
    assert dataset.ids == ["caseA_001", "caseA_002"]


def test_item_shapes_dtypes_and_range(synthetic_dataset: dict) -> None:
    dataset = make_dataset(synthetic_dataset)
    sample = dataset[0]

    assert set(sample) == {"image", "mask", "id"}
    assert sample["image"].shape == (3, SIZE, SIZE)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].shape == (1, SIZE, SIZE)
    assert sample["mask"].dtype == torch.float32
    assert sample["id"] == "caseA_001"

    # normalize=False -> image is scaled to [0, 1].
    assert float(sample["image"].min()) >= 0.0
    assert float(sample["image"].max()) <= 1.0

    unique = torch.unique(sample["mask"]).tolist()
    assert set(unique) <= {0.0, 1.0}
    assert unique != [0.0]  # the synthetic mask has foreground


def test_mask_is_binary_after_downsampling(synthetic_dataset: dict) -> None:
    dataset = make_dataset(synthetic_dataset)
    mask = dataset[0]["mask"]
    # Nearest-neighbour resize keeps the mask strictly binary even at 32x32.
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_normalisation_matches_imagenet_statistics(synthetic_dataset: dict) -> None:
    raw = make_dataset(synthetic_dataset, normalize=False)[0]["image"]
    norm = make_dataset(synthetic_dataset, normalize=True)[0]["image"]

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    assert torch.allclose(norm, (raw - mean) / std, atol=1e-5)
    assert float(norm.min()) < 0.0  # normalisation really happened


def test_grayscale_image_is_repeated_to_three_channels(synthetic_dataset: dict, tmp_path: Path) -> None:
    root: Path = synthetic_dataset["root"]
    gray = np.full((20, 20), 128, dtype=np.uint8)
    Image.fromarray(gray, mode="L").save(root / "images" / "gray_001.jpg")
    Image.fromarray(np.zeros((20, 20), dtype=np.uint8), mode="L").save(root / "masks" / "gray_001.png")
    split = tmp_path / "splits" / "gray.txt"
    split.write_text("gray_001\n", encoding="utf-8")

    dataset = PolypDataset(split, root, image_size=SIZE, normalize=False)
    image = dataset[0]["image"]

    assert image.shape == (3, SIZE, SIZE)
    assert torch.allclose(image[0], image[1]) and torch.allclose(image[1], image[2])


def test_rgb_mask_uses_any_channel_nonzero(synthetic_dataset: dict, tmp_path: Path) -> None:
    root: Path = synthetic_dataset["root"]
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8), mode="RGB").save(root / "images" / "red_001.jpg")
    mask = np.zeros((20, 20, 3), dtype=np.uint8)
    mask[5:15, 5:15] = (255, 0, 0)  # pure red: only the red channel is non-zero
    Image.fromarray(mask, mode="RGB").save(root / "masks" / "red_001.png")
    split = tmp_path / "splits" / "red.txt"
    split.write_text("red_001\n", encoding="utf-8")

    dataset = PolypDataset(split, root, image_size=SIZE, normalize=False)
    out = dataset[0]["mask"]

    assert float(out.sum()) > 0.0
    assert set(torch.unique(out).tolist()) == {0.0, 1.0}


def test_mask_threshold_is_applied(synthetic_dataset: dict, tmp_path: Path) -> None:
    root: Path = synthetic_dataset["root"]
    Image.fromarray(np.zeros((20, 20), dtype=np.uint8), mode="L").save(root / "images" / "thr_001.jpg")
    mask = np.full((20, 20), 200, dtype=np.uint8)  # above the default threshold of 127
    Image.fromarray(mask, mode="L").save(root / "masks" / "thr_001.png")
    split = tmp_path / "splits" / "thr.txt"
    split.write_text("thr_001\n", encoding="utf-8")

    dataset = PolypDataset(split, root, image_size=SIZE, normalize=False, mask_threshold=127)
    assert float(dataset[0]["mask"].sum()) == float(SIZE * SIZE)

    dataset_high = PolypDataset(split, root, image_size=SIZE, normalize=False, mask_threshold=250)
    assert float(dataset_high[0]["mask"].sum()) == 0.0


def test_transform_result_is_used_for_both_image_and_mask(synthetic_dataset: dict) -> None:
    def invert_mask(image: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        """Leave the image alone, invert the mask."""
        return {"image": image, "mask": 1.0 - mask}

    plain = make_dataset(synthetic_dataset)[0]
    inverted = make_dataset(synthetic_dataset, transform=invert_mask)[0]

    assert torch.equal(plain["image"], inverted["image"])
    assert torch.allclose(inverted["mask"], 1.0 - plain["mask"])
    assert float(inverted["mask"].sum()) > 0.0


def test_missing_image_raises_file_not_found(synthetic_dataset: dict, tmp_path: Path) -> None:
    split = tmp_path / "splits" / "ghost.txt"
    split.write_text("ghost_001\n", encoding="utf-8")
    dataset = PolypDataset(split, synthetic_dataset["root"], image_size=SIZE)

    with pytest.raises(FileNotFoundError, match="Image not found"):
        _ = dataset[0]


def test_missing_mask_raises_file_not_found(synthetic_dataset: dict, tmp_path: Path) -> None:
    root: Path = synthetic_dataset["root"]
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8), mode="RGB").save(root / "images" / "nomask_001.jpg")
    split = tmp_path / "splits" / "nomask.txt"
    split.write_text("nomask_001\n", encoding="utf-8")
    dataset = PolypDataset(split, root, image_size=SIZE)

    with pytest.raises(FileNotFoundError, match="Mask not found"):
        _ = dataset[0]


def test_read_split_skips_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / "train.txt"
    path.write_text("# a comment\n\nfoo\nbar\n   \n#another\nbaz\n", encoding="utf-8")

    assert read_split(path) == ["foo", "bar", "baz"]


def test_read_split_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Split file not found"):
        read_split(tmp_path / "missing.txt")


def test_image_size_must_be_positive(synthetic_dataset: dict) -> None:
    with pytest.raises(ValueError, match="image_size must be positive"):
        make_dataset(synthetic_dataset, image_size=0)


def test_mask_suffix_is_respected(synthetic_dataset: dict, tmp_path: Path) -> None:
    root: Path = synthetic_dataset["root"]
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8), mode="RGB").save(root / "images" / "suf_001.jpg")
    suffixed = np.zeros((20, 20), dtype=np.uint8)
    suffixed[2:8, 2:8] = 255
    Image.fromarray(suffixed, mode="L").save(root / "masks" / "suf_001_m.png")
    split = tmp_path / "splits" / "suf.txt"
    split.write_text("suf_001\n", encoding="utf-8")

    dataset = PolypDataset(split, root, image_size=SIZE, normalize=False, mask_suffix="_m")
    assert float(dataset[0]["mask"].sum()) > 0.0


def test_from_config_builds_the_split(synthetic_dataset: dict) -> None:
    cfg = load_config(
        REPO_ROOT / "configs" / "base.yaml",
        overrides={
            "data": {
                "root": str(synthetic_dataset["root"]),
                "split_dir": str(synthetic_dataset["split_dir"]),
                "image_size": SIZE,
                # The synthetic fixture stores PNG masks; Kvasir-SEG (base.yaml)
                # uses JPEG masks.
                "mask_ext": ".png",
            }
        },
    )

    dataset = PolypDataset.from_config(cfg, split="train")

    assert len(dataset) == 2
    assert dataset[0]["image"].shape == (3, SIZE, SIZE)


def test_train_transform_helper_is_optional() -> None:
    """The augmentation pipeline needs albumentations; the dataset resizes after it."""
    pytest.importorskip("albumentations")
    from src.datasets import build_train_transform

    transform = build_train_transform(SIZE, seed=0)
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    mask = np.zeros((48, 48), dtype=np.float32)
    mask[10:20, 10:20] = 1.0
    out = transform(image=image, mask=mask)

    # A transform may change the size; image and mask must stay aligned, and the
    # dataset is what guarantees the final (3, image_size, image_size) shape.
    assert out["image"].ndim == 3 and out["image"].shape[2] == 3
    assert out["mask"].shape == out["image"].shape[:2]


def test_dataset_resizes_after_transform(synthetic_dataset: dict) -> None:
    pytest.importorskip("albumentations")
    from src.datasets import build_train_transform

    transform = build_train_transform(SIZE, seed=0)
    sample = make_dataset(synthetic_dataset, transform=transform, image_size=SIZE)[0]

    assert sample["image"].shape == (3, SIZE, SIZE)
    assert sample["mask"].shape == (1, SIZE, SIZE)
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}
