"""Dataset for binary polyp segmentation.

Everything the dataset needs about the on-disk layout comes from the config
(``data.*``). The layout assumed here is the simplest one that covers the usual
polyp benchmarks (Kvasir-SEG, CVC-ClinicDB, ETIS, ...):

    <root>/<image_dir>/<id><image_ext>       e.g. data/raw/images/cju0qkwl35piu0993l0dewej5.jpg
    <root>/<mask_dir>/<id><mask_suffix><mask_ext>   e.g. data/raw/masks/cju0qkwl35piu0993l0dewej5.png

``<id>`` is the stem written in the split file (``data/splits/train.txt``), one id
per line. Masks are binarised with ``data.mask_threshold`` *before* resizing, and
resized with nearest-neighbour so the output stays exactly 0 or 1.

The sample returned by ``__getitem__`` matches AGENTS.md section 6:
``{"image": (3, H, W) float32, "mask": (1, H, W) float32 in {0, 1}, "id": str}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils import IMAGENET_MEAN, IMAGENET_STD

#: A transform is any callable following the albumentations convention: it takes
#: ``image`` as uint8 ``(H, W, 3)`` and ``mask`` as float32 ``(H, W)`` in {0, 1},
#: and returns a dict with both keys.
Transform = Callable[..., dict[str, np.ndarray]]


def load_cache(cache_path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load a pre-resized ``.npz`` cache built by ``scripts/build_cache.py``.

    Args:
        cache_path: Path to the cache file.

    Returns:
        ``(images, masks, id_to_index)``: images ``(N, S, S, 3)`` uint8, masks
        ``(N, S, S)`` uint8 in {0, 1}, and the row for each sample id.

    Raises:
        FileNotFoundError: if the cache file does not exist.
        ValueError: if the cache is missing arrays or has an unexpected shape/dtype.

    Notes:
        The whole array pair is loaded into memory once per worker (about 262 MB
        for 1000 images at 256x256); workers forked on Linux share the pages.
    """
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Cache not found: {path}. Build it with scripts/build_cache.py "
            "or set data.cache_path to an empty string."
        )
    with np.load(path) as data:
        missing = {"ids", "images", "masks"} - set(data.files)
        if missing:
            raise ValueError(f"Cache {path} is missing arrays: {sorted(missing)}")
        ids = [str(value) for value in data["ids"]]
        images = data["images"]
        masks = data["masks"]

    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Cache images must be (N, S, S, 3), got {images.shape}: {path}")
    if masks.ndim != 3:
        raise ValueError(f"Cache masks must be (N, S, S), got {masks.shape}: {path}")
    if len(ids) != images.shape[0] or len(ids) != masks.shape[0]:
        raise ValueError(f"Cache ids ({len(ids)}) do not match image/mask counts: {path}")
    if images.dtype != np.uint8:
        raise ValueError(f"Cache images must be uint8, got {images.dtype}: {path}")
    if masks.dtype != np.uint8:
        raise ValueError(f"Cache masks must be uint8, got {masks.dtype}: {path}")
    if masks.size and int(masks.max()) > 1:
        raise ValueError(f"Cache masks must be 0/1, found max {int(masks.max())}: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Cache ids must be unique: {path}")

    return images, masks, {sample_id: index for index, sample_id in enumerate(ids)}


def read_split(split_file: str | Path) -> list[str]:
    """Read a split file into a list of sample ids.

    Blank lines and lines starting with ``#`` are ignored, so split files can be
    annotated. Order is preserved (reproducibility matters more than shuffling
    here - shuffling is the DataLoader's job).

    Args:
        split_file: Path to e.g. ``data/splits/train.txt``.

    Returns:
        List of non-empty ids.

    Raises:
        FileNotFoundError: if the split file does not exist.
    """
    path = Path(split_file)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    ids: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


class PolypDataset(Dataset):
    """Binary polyp segmentation dataset driven by a split file.

    Args:
        split_file: Text file with one sample id per line.
        root: Dataset root directory; image and mask folders live under it.
        transform: Optional augmentation callable (see :data:`Transform`).
        image_size: Output spatial size (square). Must be divisible by 32 in
            practice, because the encoders downsample by 32.
        image_dir: Sub-folder of ``root`` holding images.
        mask_dir: Sub-folder of ``root`` holding masks.
        image_ext: Image file extension, e.g. ``".jpg"``.
        mask_ext: Mask file extension, e.g. ``".png"``.
        mask_suffix: Appended to the id for mask lookup, e.g. ``"_mask"``.
        mask_threshold: Pixel value above which a mask pixel is foreground.
        cache_path: Optional ``.npz`` cache from ``scripts/build_cache.py``. When
            set, images and masks are read from it instead of from disk; every id
            in the split must be present.
        normalize: If True, normalise with ``mean``/``std``; else scale to [0, 1].
        mean: Per-channel mean, length 3.
        std: Per-channel std, length 3.

    Returns per item:
        ``{"image": (3, H, W) float32, "mask": (1, H, W) float32, "id": str}``
    """

    def __init__(
        self,
        split_file: str | Path,
        root: str | Path,
        transform: Transform | None = None,
        image_size: int = 256,
        *,
        image_dir: str = "images",
        mask_dir: str = "masks",
        image_ext: str = ".jpg",
        mask_ext: str = ".png",
        mask_suffix: str = "",
        mask_threshold: int = 127,
        cache_path: str | Path | None = None,
        normalize: bool = True,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.split_file = Path(split_file)
        self.root = Path(root)
        self.transform = transform
        self.image_size = int(image_size)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.mask_suffix = mask_suffix
        self.mask_threshold = int(mask_threshold)
        self.cache_path = Path(cache_path) if cache_path else None
        self.normalize = normalize
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must each have 3 values")

        self.ids: list[str] = read_split(self.split_file)

        self._cache_images: np.ndarray | None = None
        self._cache_masks: np.ndarray | None = None
        self._cache_index: dict[str, int] = {}
        if self.cache_path is not None:
            self._cache_images, self._cache_masks, self._cache_index = load_cache(self.cache_path)
            missing = [sample_id for sample_id in self.ids if sample_id not in self._cache_index]
            if missing:
                preview = ", ".join(missing[:5])
                raise ValueError(
                    f"{len(missing)} split ids are missing from cache {self.cache_path} "
                    f"(first: {preview}). Rebuild the cache from the same split."
                )

    @classmethod
    def from_config(cls, cfg: dict[str, Any], split: str, transform: Transform | None = None) -> "PolypDataset":
        """Build a dataset from a resolved config.

        Args:
            cfg: Resolved config (see :func:`src.utils.load_config`).
            split: One of ``"train"``, ``"val"``, ``"test"``; resolves to
                ``<data.split_dir>/<split>.txt``.
            transform: Optional augmentation callable.

        Returns:
            A :class:`PolypDataset` for that split.
        """
        data = cfg["data"]
        return cls(
            split_file=Path(data["split_dir"]) / f"{split}.txt",
            root=data["root"],
            transform=transform,
            image_size=data["image_size"],
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

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one sample.

        Args:
            index: Index into the split.

        Returns:
            ``{"image": (3, H, W) float32, "mask": (1, H, W) float32, "id": str}``
        """
        sample_id = self.ids[index]
        if self._cache_images is not None and self._cache_masks is not None:
            row = self._cache_index[sample_id]
            image = self._cache_images[row]  # (S, S, 3) uint8
            mask = self._cache_masks[row].astype(np.float32)  # (S, S) in {0, 1}
        else:
            image_path = self.root / self.image_dir / f"{sample_id}{self.image_ext}"
            mask_path = self.root / self.mask_dir / f"{sample_id}{self.mask_suffix}{self.mask_ext}"
            image = self._load_image(image_path)  # (h, w, 3) uint8
            mask = self._load_mask(mask_path)  # (h, w) float32 in {0, 1}

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = np.asarray(augmented["image"])
            mask = np.asarray(augmented["mask"])

        image = self._resize(image, self.image_size, resample=Image.BILINEAR)  # (H, W, 3) uint8
        mask = self._resize(mask, self.image_size, resample=Image.NEAREST)  # (H, W) float32
        mask = (mask >= 0.5).astype(np.float32)

        image_t = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3, H, W)
        if self.normalize:
            image_t = (image_t - torch.from_numpy(self.mean)) / torch.from_numpy(self.std)

        mask_t = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        return {"image": image_t.contiguous(), "mask": mask_t.contiguous(), "id": sample_id}

    def _load_image(self, path: Path) -> np.ndarray:
        """Load an image as RGB uint8.

        Args:
            path: Image path.

        Returns:
            Array ``(h, w, 3)`` uint8. Single-channel or palette images are
            repeated to 3 channels.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Image not found for id {path.stem!r}: {path}")
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            return np.asarray(image, dtype=np.uint8)

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load and binarise a mask.

        Args:
            path: Mask path.

        Returns:
            Array ``(h, w)`` float32 with values in {0, 1}.

        Notes:
            A grayscale mask is thresholded directly. A multi-channel mask is
            reduced with a per-pixel max first, so white-on-black RGB masks and
            colormapped masks both work.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Mask not found for id {path.stem!r}: {path}")
        with Image.open(path) as handle:
            mask = np.asarray(handle)
        if mask.ndim == 3:
            mask = mask.max(axis=2)
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D after channel reduction, got shape {mask.shape}: {path}")
        return (mask > self.mask_threshold).astype(np.float32)

    @staticmethod
    def _resize(array: np.ndarray, size: int, resample: int) -> np.ndarray:
        """Resize an image or mask to ``size x size``.

        Args:
            array: ``(h, w, 3)`` uint8 image or ``(h, w)`` float32 mask.
            size: Target side length.
            resample: PIL resampling filter (``BILINEAR`` for images,
                ``NEAREST`` for masks).

        Returns:
            Resized array with the same rank as the input.
        """
        height, width = array.shape[:2]
        if height == size and width == size:
            return array
        if array.ndim == 2:
            resized = Image.fromarray(array.astype(np.float32)).resize((size, size), resample=resample)
            return np.asarray(resized, dtype=np.float32)
        resized = Image.fromarray(array).resize((size, size), resample=resample)
        return np.asarray(resized, dtype=array.dtype)


def build_train_transform(image_size: int, *, seed: int | None = None) -> Transform:
    """Build the training augmentation pipeline (flips, rotation, scale, brightness, contrast).

    Geometric operations are applied to image and mask together by albumentations,
    so the mask never drifts from the image. Normalisation is *not* here: the
    dataset does it, so every split uses the same fixed mean/std.

    Args:
        image_size: Output spatial size.
        seed: Optional fixed seed for the augmentation RNG.

    Returns:
        An albumentations ``Compose`` callable.

    Raises:
        ImportError: if albumentations is not installed.
    """
    try:
        import albumentations as A
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("albumentations is required for training augmentations: pip install albumentations") from exc

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, border_mode=0, p=0.5),
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.85, 1.0), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ],
        seed=seed,
    )


def build_eval_transform() -> None:
    """Return the validation/test transform.

    Val and test only resize and normalise, and the dataset already does both
    deterministically, so there is nothing left to do here. Returning ``None``
    keeps the call sites identical to training.

    Returns:
        ``None``.
    """
    return None


def collate_ids(batch: Iterable[dict[str, Any]]) -> list[str]:
    """Small helper to collect ids from a batch of dataset samples.

    Args:
        batch: Iterable of dataset samples.

    Returns:
        List of ids, in batch order.
    """
    return [sample["id"] for sample in batch]
