"""Dataset registry and ``DATA_ROOT`` resolution (Milestone 8, Step 0).

Every polyp dataset the project knows about is described in
``configs/datasets.yaml``: its canonical name, the folder it lives in under the
raw data tree, its published image count, and how images pair with masks.

``DATA_ROOT`` is the root of the data tree and is resolved in this order:

1. the ``DATA_ROOT`` environment variable,
2. the ``data.data_root`` value of an experiment config,
3. ``<repo>/data`` (the default).

Under that root the layout is fixed::

    <DATA_ROOT>/raw/<folder>/          raw images and masks, as shipped
    <DATA_ROOT>/processed/<name>/      prepared cache (``data.npz`` + ``meta.json``)

Images and masks pair up by file stem (the optional ``mask_suffix`` is stripped
from the mask stem first), so a dataset whose images are ``.tif`` and whose masks
are ``.png`` is matched correctly.

No path is hard-coded anywhere else: callers ask this module for ``raw_dir`` and
``processed_dir``. A dataset that is not on disk is reported as absent, never as
an error; only an explicit request for a missing dataset is allowed to fail.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

#: Repository root, used to anchor the defaults below.
REPO_ROOT = Path(__file__).resolve().parents[1]
#: Default data-tree root when neither the environment nor a config sets one.
DEFAULT_DATA_ROOT: Path = REPO_ROOT / "data"
#: Committed registry file.
DEFAULT_REGISTRY_PATH: Path = REPO_ROOT / "configs" / "datasets.yaml"
#: Environment variable that overrides the data-tree root.
DATA_ROOT_ENV: str = "DATA_ROOT"
#: Names of the two sub-trees under the data root.
RAW_SUBDIR: str = "raw"
PROCESSED_SUBDIR: str = "processed"
#: Default sub-folders holding images and masks inside a raw dataset folder.
DEFAULT_IMAGE_SUBDIR: str = "images"
DEFAULT_MASK_SUBDIR: str = "masks"

#: Keys an entry in ``configs/datasets.yaml`` may set.
_ENTRY_KEYS: frozenset[str] = frozenset(
    {"name", "folder", "expected_count", "source", "image_subdir", "mask_subdir", "mask_suffix"}
)


class RegistryError(ValueError):
    """The registry file is missing or malformed, or a dataset name is unknown."""


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset as described in ``configs/datasets.yaml``.

    Attributes:
        name: Canonical short name, e.g. ``kvasir_seg``. Used for the processed
            folder and as the ``dataset`` key in results.
        folder: Raw folder name under ``<DATA_ROOT>/raw``.
        expected_count: Published number of image/mask pairs.
        image_subdir: Sub-folder of the raw folder holding images.
        mask_subdir: Sub-folder of the raw folder holding masks.
        mask_suffix: Optional suffix between mask stem and extension, e.g.
            ``"_mask"``. Empty for every dataset in this registry.
        source: Human-readable source name, e.g. ``"Kvasir-SEG"``. Recorded in
            each processed ``meta.json``.
    """

    name: str
    folder: str
    expected_count: int
    image_subdir: str = DEFAULT_IMAGE_SUBDIR
    mask_subdir: str = DEFAULT_MASK_SUBDIR
    mask_suffix: str = ""
    source: str = ""

    def raw_dir(self, data_root: str | Path) -> Path:
        """Raw folder for this dataset, ``<DATA_ROOT>/raw/<folder>``.

        Args:
            data_root: Data-tree root.

        Returns:
            Absolute (or caller-relative) path to the raw folder.
        """
        return Path(data_root) / RAW_SUBDIR / self.folder

    def processed_dir(self, data_root: str | Path) -> Path:
        """Prepared-cache folder, ``<DATA_ROOT>/processed/<name>``.

        Args:
            data_root: Data-tree root.

        Returns:
            Path to the processed folder for this dataset.
        """
        return Path(data_root) / PROCESSED_SUBDIR / self.name

    def image_dir(self, data_root: str | Path) -> Path:
        """Folder holding this dataset's raw images."""
        return self.raw_dir(data_root) / self.image_subdir

    def mask_dir(self, data_root: str | Path) -> Path:
        """Folder holding this dataset's raw masks."""
        return self.raw_dir(data_root) / self.mask_subdir

    def is_present(self, data_root: str | Path) -> bool:
        """Whether both the raw image and raw mask folders exist on disk.

        Args:
            data_root: Data-tree root.

        Returns:
            True when the dataset looks present (both folders are directories).
        """
        return self.image_dir(data_root).is_dir() and self.mask_dir(data_root).is_dir()


def load_registry(path: str | Path | None = None) -> dict[str, DatasetSpec]:
    """Load the dataset registry, preserving file order.

    Args:
        path: Registry YAML (default: ``configs/datasets.yaml`` next to the repo
            root).

    Returns:
        ``{name: DatasetSpec}`` in the order the file lists them.

    Raises:
        RegistryError: if the file is missing or malformed, has unknown or
            duplicate entries, or a non-positive ``expected_count``.
    """
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if not registry_path.is_file():
        raise RegistryError(f"Dataset registry not found: {registry_path}")
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or "datasets" not in raw:
        raise RegistryError(f"{registry_path} must be a mapping with a 'datasets' list")
    entries = raw["datasets"]
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        raise RegistryError(f"{registry_path}: 'datasets' must be a non-empty list")

    specs: dict[str, DatasetSpec] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise RegistryError(f"{registry_path}: dataset #{index} must be a mapping")
        unknown = sorted(set(entry) - _ENTRY_KEYS)
        if unknown:
            raise RegistryError(f"{registry_path}: dataset #{index} has unknown keys {unknown}")
        missing = [key for key in ("name", "folder", "expected_count") if key not in entry]
        if missing:
            raise RegistryError(f"{registry_path}: dataset #{index} is missing {missing}")
        try:
            expected_count = int(entry["expected_count"])
        except (TypeError, ValueError) as exc:
            raise RegistryError(
                f"{registry_path}: dataset #{index} expected_count must be an int, got {entry['expected_count']!r}"
            ) from exc
        spec = DatasetSpec(
            name=str(entry["name"]),
            folder=str(entry["folder"]),
            expected_count=expected_count,
            image_subdir=str(entry.get("image_subdir", DEFAULT_IMAGE_SUBDIR)),
            mask_subdir=str(entry.get("mask_subdir", DEFAULT_MASK_SUBDIR)),
            mask_suffix=str(entry.get("mask_suffix", "")),
            source=str(entry.get("source", "")),
        )
        if not spec.name or not spec.folder:
            raise RegistryError(f"{registry_path}: dataset #{index} name and folder must be non-empty")
        if spec.expected_count <= 0:
            raise RegistryError(f"{registry_path}: {spec.name} expected_count must be positive")
        if spec.name in specs:
            raise RegistryError(f"{registry_path}: duplicate dataset name {spec.name!r}")
        specs[spec.name] = spec
    return specs


def registry_names() -> list[str]:
    """Names of every registered dataset, in file order.

    Returns:
        List of dataset names.
    """
    return list(load_registry())


def get_spec(name: str) -> DatasetSpec:
    """Look up one dataset by name.

    Args:
        name: Dataset name, e.g. ``kvasir_seg``.

    Returns:
        The matching :class:`DatasetSpec`.

    Raises:
        RegistryError: if the name is not in the registry.
    """
    specs = load_registry()
    if name not in specs:
        raise RegistryError(f"Unknown dataset {name!r}; known datasets: {sorted(specs)}")
    return specs[name]


def resolve_data_root(config_value: str | None = None, *, env: Mapping[str, str] | None = None) -> Path:
    """Resolve the data-tree root: env var > config value > default.

    Args:
        config_value: Value of ``data.data_root`` from an experiment config
            (``""`` or ``None`` means unset).
        env: Environment mapping (defaults to :data:`os.environ`); injectable for
            tests.

    Returns:
        The resolved data-tree root as a :class:`~pathlib.Path`. Not checked for
        existence: a missing root is reported by the caller, not here.
    """
    environment = os.environ if env is None else env
    env_value = environment.get(DATA_ROOT_ENV)
    if env_value is not None and str(env_value).strip():
        return Path(str(env_value)).expanduser()
    if config_value is not None and str(config_value).strip():
        return Path(str(config_value)).expanduser()
    return DEFAULT_DATA_ROOT


def data_root_from_config(cfg: Mapping[str, Any]) -> Path:
    """Resolve the data-tree root from a resolved config.

    Args:
        cfg: Resolved config (see :func:`src.utils.load_config`).

    Returns:
        The resolved data-tree root.
    """
    data = cfg.get("data", {})
    config_value = data.get("data_root", "") if isinstance(data, Mapping) else ""
    return resolve_data_root(str(config_value))


def present_datasets(data_root: str | Path) -> list[DatasetSpec]:
    """Every registered dataset whose raw folders exist on disk.

    Args:
        data_root: Data-tree root.

    Returns:
        Specs present on disk, in registry order.
    """
    return [spec for spec in load_registry().values() if spec.is_present(data_root)]
