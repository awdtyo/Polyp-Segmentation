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
    """Binary polyp segmentation dataset driven by a split file or an id list.

    Args:
        split_file: Text file with one sample id per line. May be ``None`` when
            ``ids`` is given instead.
        root: Dataset root directory; image and mask folders live under it.
        transform: Optional augmentation callable (see :data:`Transform`).
        image_size: Output spatial size (square). Must be divisible by 32 in
            practice, because the encoders downsample by 32.
        ids: Optional explicit sample ids, used instead of ``split_file``. This
            is how an external dataset prepared by ``scripts/prepare_datasets.py``
            (all images, no split file) is read. Exactly one of ``split_file``
            and ``ids`` must be provided.
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
        split_file: str | Path | None = None,
        root: str | Path = ".",
        transform: Transform | None = None,
        image_size: int = 256,
        *,
        ids: Sequence[str] | None = None,
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
        if (split_file is None) == (ids is None):
            raise ValueError("PolypDataset needs exactly one of split_file or ids")
        self.split_file = Path(split_file) if split_file is not None else None
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

        if ids is not None:
            self.ids = [str(sample_id) for sample_id in ids]
            if len(set(self.ids)) != len(self.ids):
                raise ValueError("ids must be unique")
        else:
            assert self.split_file is not None  # guarded above
            self.ids = read_split(self.split_file)

        self._cache_images: np.ndarray | None = None
        self._cache_masks: np.ndarray | None = None
        self._cache_index: dict[str, int] = {}
        if self.cache_path is not None:
            self._cache_images, self._cache_masks, self._cache_index = load_cache(self.cache_path)
            if self._cache_images is not None:
                cache_size = self._cache_images.shape[1]
                if cache_size != self.image_size:
                    raise ValueError(
                        f"Cache image size ({cache_size}) does not match requested "
                        f"image_size ({self.image_size}). Rebuild the cache at the "
                        f"correct size (scripts/build_cache.py --size {self.image_size}) "
                        f"or scripts/prepare_datasets.py --size {self.image_size}."
                    )
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

    @property
    def uses_cache(self) -> bool:
        """Whether samples are read from an npz cache instead of image files.

        Returns:
            True when ``cache_path`` was set and the cache loaded successfully.
        """
        return self._cache_images is not None

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


def build_train_transform(
    image_size: int,
    *,
    seed: int | None = None,
    hflip_p: float = 0.5,
    vflip_p: float = 0.5,
    rotation_degrees: float = 20.0,
    rotation_p: float = 0.5,
    random_resized_crop_scale: tuple[float, float] = (0.85, 1.0),
    random_resized_crop_p: float = 0.5,
    brightness_limit: float = 0.1,
    contrast_limit: float = 0.1,
    color_jitter_p: float = 0.0,
    brightness_jitter: float = 0.25,
    contrast_jitter: float = 0.25,
    saturation_jitter: float = 0.25,
    hue_jitter: float = 0.03,
    color_jitter_combined_p: float = 0.8,
    gauss_noise_p: float = 0.0,
    gauss_noise_var_limit: tuple[float, float] = (5.0, 20.0),
    gaussian_blur_p: float = 0.0,
    gaussian_blur_limit: int = 3,
    specular_inpaint: bool = False,
    specular_threshold: float = 0.95,
    clahe: bool = False,
    clahe_clip_limit: float = 2.0,
) -> Transform:
    """Build the training augmentation pipeline (configurable).

    Geometric operations are applied to image and mask together by albumentations,
    so the mask never drifts from the image. Colour and noise ops are image-only.
    Normalisation is *not* here: the dataset does it, so every split uses the same
    fixed mean/std.

    Defaults reproduce the original pipeline exactly (flips 0.5, rotate 20 deg p 0.5,
    RandomResizedCrop 0.85-1.0 p 0.5, RandomBrightnessContrast 0.1 p 0.5; no colour
    jitter, noise, specular inpainting or CLAHE).

    Args:
        image_size: Output spatial size.
        seed: Optional fixed seed for the augmentation RNG.
        hflip_p: Horizontal flip probability.
        vflip_p: Vertical flip probability.
        rotation_degrees: Rotation limit in degrees.
        rotation_p: Rotation probability.
        random_resized_crop_scale: (min, max) scale for RandomResizedCrop.
        random_resized_crop_p: RandomResizedCrop probability.
        brightness_limit: Brightness limit for the default RandomBrightnessContrast.
        contrast_limit: Contrast limit for the default RandomBrightnessContrast.
        color_jitter_p: Probability of ColorJitter (replaces RandomBrightnessContrast
            when > 0; when 0 the default RandomBrightnessContrast is used).
        brightness_jitter: ColorJitter brightness factor.
        contrast_jitter: ColorJitter contrast factor.
        saturation_jitter: ColorJitter saturation factor.
        hue_jitter: ColorJitter hue factor.
        color_jitter_combined_p: ColorJitter combined probability.
        gauss_noise_p: GaussNoise probability (0 = off).
        gauss_noise_var_limit: (min, max) variance limit for GaussNoise.
        gaussian_blur_p: GaussianBlur probability (0 = off).
        gaussian_blur_limit: Max kernel size for GaussianBlur.
        specular_inpaint: If True, apply specular-highlight detection + inpainting.
        specular_threshold: Brightness threshold for specular detection.
        clahe: If True, apply CLAHE on the L channel of LAB.
        clahe_clip_limit: CLAHE clip limit.

    Returns:
        An albumentations ``Compose`` callable.

    Raises:
        ImportError: if albumentations is not installed.
    """
    try:
        import albumentations as A
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("albumentations is required for training augmentations: pip install albumentations") from exc

    transforms = []
    if hflip_p > 0:
        transforms.append(A.HorizontalFlip(p=hflip_p))
    if vflip_p > 0:
        transforms.append(A.VerticalFlip(p=vflip_p))
    if rotation_degrees > 0 and rotation_p > 0:
        transforms.append(A.Rotate(limit=rotation_degrees, border_mode=0, p=rotation_p))
    if random_resized_crop_p > 0:
        transforms.append(A.RandomResizedCrop(size=(image_size, image_size), scale=random_resized_crop_scale, p=random_resized_crop_p))
    if color_jitter_p > 0:
        transforms.append(A.ColorJitter(brightness=brightness_jitter, contrast=contrast_jitter, saturation=saturation_jitter, hue=hue_jitter, p=color_jitter_p))
    elif brightness_limit > 0 or contrast_limit > 0:
        transforms.append(A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=0.5))
    if gauss_noise_p > 0:
        transforms.append(A.GaussNoise(var_limit=gauss_noise_var_limit, p=gauss_noise_p))
    if gaussian_blur_p > 0:
        transforms.append(A.GaussianBlur(blur_limit=(3, gaussian_blur_limit), p=gaussian_blur_p))
    if specular_inpaint:
        transforms.append(A.Lambda(name="specular_inpaint", image=_specular_inpaint, p=1.0, additional_targets={"mask": "mask"}))
    if clahe:
        transforms.append(A.CLAHE(clip_limit=clahe_clip_limit, p=1.0))

    return A.Compose(transforms, seed=seed)


def _specular_inpaint(image: np.ndarray, **kwargs) -> np.ndarray:
    """Detect specular highlights and inpaint them with surrounding tissue.

    Simple implementation: threshold bright pixels, dilate slightly, inpaint
    using cv2.inpaint with Telea method.
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, int(specular_threshold * 255), 255, cv2.THRESH_BINARY)
    if mask.sum() == 0:
        return image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def build_eval_transform() -> None:
    """Return the validation/test transform (no augmentation)."""
    return None


def collate_ids(batch: Iterable[dict[str, Any]]) -> list[str]:
    """Small helper to collect ids from a batch of dataset samples.

    Args:
        batch: Iterable of dataset samples.

    Returns:
        List of ids, in batch order.
    """
    return [sample["id"] for sample in batch]


def build_train_transform_from_config(cfg: dict, image_size: int, *, seed: int | None = None) -> Transform | None:
    """Build a training transform from a resolved config's ``data.augment`` block.

    If ``data.augment`` is absent or empty, returns the default pipeline (current
    behaviour). If ``data.augment.enabled`` is False, returns None (no augmentation).

    Args:
        cfg: Resolved config dict.
        image_size: Output spatial size.
        seed: Optional fixed seed for the augmentation RNG.

    Returns:
        An albumentations ``Compose`` callable, or None when augmentation is off.
    """
    data = cfg.get("data", {})
    aug = data.get("augment", None)
    if aug is None:
        # Default: current behaviour (backward compatible).
        return build_train_transform(image_size, seed=seed)
    if not aug.get("enabled", True):
        return None
    return build_train_transform(
        image_size,
        seed=seed,
        hflip_p=float(aug.get("hflip_p", 0.5)),
        vflip_p=float(aug.get("vflip_p", 0.5)),
        rotation_degrees=float(aug.get("rotation_degrees", 20.0)),
        rotation_p=float(aug.get("rotation_p", 0.5)),
        random_resized_crop_scale=tuple(aug.get("random_resized_crop_scale", [0.85, 1.0])),
        random_resized_crop_p=float(aug.get("random_resized_crop_p", 0.5)),
        brightness_limit=float(aug.get("brightness_limit", 0.1)),
        contrast_limit=float(aug.get("contrast_limit", 0.1)),
        color_jitter_p=float(aug.get("color_jitter_p", 0.0)),
        brightness_jitter=float(aug.get("brightness_jitter", 0.25)),
        contrast_jitter=float(aug.get("contrast_jitter", 0.25)),
        saturation_jitter=float(aug.get("saturation_jitter", 0.25)),
        hue_jitter=float(aug.get("hue_jitter", 0.03)),
        color_jitter_combined_p=float(aug.get("color_jitter_combined_p", 0.8)),
        gauss_noise_p=float(aug.get("gauss_noise_p", 0.0)),
        gauss_noise_var_limit=tuple(aug.get("gauss_noise_var_limit", [5.0, 20.0])),
        gaussian_blur_p=float(aug.get("gaussian_blur_p", 0.0)),
        gaussian_blur_limit=int(aug.get("gaussian_blur_limit", 3)),
        specular_inpaint=bool(aug.get("specular_inpaint", False)),
        specular_threshold=float(aug.get("specular_threshold", 0.95)),
        clahe=bool(aug.get("clahe", False)),
        clahe_clip_limit=float(aug.get("clahe_clip_limit", 2.0)),
    )
