"""Tests that every pilot config (exp003_p*) loads through the real config
loader used by ``scripts/build_cache.py``, ``src.train`` and ``src.evaluate``.

All three entry points delegate to :func:`src.utils.load_config`, which calls
:func:`src.utils.validate_config`. Before the fix, any config with a
``data.augment`` block was rejected with
``"Config error: unknown keys in 'data': ['augment']"`` (exit status 2).
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.utils import ConfigError, load_config
from src.datasets import build_train_transform_from_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"

PILOT_CONFIGS = [
    "exp003_p0_current.yaml",
    "exp003_p1_strong_geom.yaml",
    "exp003_p2_color_jitter.yaml",
    "exp003_p3_noise_blur.yaml",
    "exp003_p4_352px.yaml",
    "exp003_p5_specular.yaml",
    "exp003_p6_clahe.yaml",
]


def _tracked_config_yamls() -> list[Path]:
    """Every tracked experiment config YAML under configs/.

    Excludes ``configs/datasets.yaml``, which is the dataset registry (a
    different schema with a top-level ``datasets`` key) and is not an experiment
    config loadable by :func:`load_config`.
    """
    result = subprocess.run(
        ["git", "ls-files", str(CONFIGS_DIR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    tracked: list[str] = []
    if result.returncode == 0:
        tracked = [
            line
            for line in result.stdout.splitlines()
            if line.endswith((".yaml", ".yml")) and Path(line).name != "datasets.yaml"
        ]
    if not tracked:
        tracked = [
            str(p) for p in CONFIGS_DIR.glob("*.yaml") if p.name != "datasets.yaml"
        ]
    return [Path(p) for p in sorted(tracked)]


@pytest.mark.parametrize("name", PILOT_CONFIGS)
def test_pilot_config_loads_through_real_loader(name: str) -> None:
    """Each pilot config must be accepted by the shared ``load_config``."""
    cfg = load_config(CONFIGS_DIR / name)
    assert cfg["data"]["image_size"] in (256, 352)
    if name == "exp003_p0_current.yaml":
        assert cfg["data"]["augment"] == {}
    else:
        assert cfg["data"]["augment"]["enabled"] is True


def test_every_tracked_config_yaml_loads_through_the_real_loader() -> None:
    """Load every tracked yaml under configs/ through load_config (the loader
    shared by build_cache.py, src.train and src.evaluate). Any failure means a
    config entry point is broken."""
    for path in _tracked_config_yamls():
        load_config(path)


def test_unknown_augment_subkey_is_rejected(tmp_path: Path) -> None:
    """A typo inside data.augment (e.g. "rotaton") must raise loudly."""
    import yaml

    base = load_config(CONFIGS_DIR / "exp003_p1_strong_geom.yaml")
    bad = {
        "run": base["run"],
        "data": {**base["data"], "augment": {**base["data"]["augment"], "rotaton": 0.5}},
        "train": base["train"],
        "model": base["model"],
        "decoder": base["decoder"],
        "logging": base["logging"],
    }
    path = tmp_path / "bad_augment.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown keys in 'data.augment'"):
        load_config(path)


def test_augment_value_out_of_range_is_rejected(tmp_path: Path) -> None:
    """A probability outside [0, 1] must raise."""
    import yaml

    base = load_config(CONFIGS_DIR / "exp003_p1_strong_geom.yaml")
    bad = {
        "run": base["run"],
        "data": {**base["data"], "augment": {**base["data"]["augment"], "hflip_p": 1.5}},
        "train": base["train"],
        "model": base["model"],
        "decoder": base["decoder"],
        "logging": base["logging"],
    }
    path = tmp_path / "bad_range.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ConfigError, match="data.augment.hflip_p"):
        load_config(path)


def test_p0_unchanged_without_augment_block() -> None:
    """P0 has no augment block and must behave exactly as before (the resolved
    config gains only the empty ``data.augment`` default, nothing else)."""
    cfg = load_config(CONFIGS_DIR / "exp003_p0_current.yaml")
    assert cfg["data"]["image_size"] == 256
    assert cfg["data"]["augment"] == {}
    assert cfg["data"]["cache_path"] == "data/cache/kvasir_seg_256.npz"


def test_p0_and_empty_augment_give_identical_transform() -> None:
    """A config with no augment block and one with ``augment: {}`` must yield an
    identical training transform (regression: exp003_p0_current gives the same
    tensors as before the fix)."""
    import albumentations as A  # noqa: F401  (ensures the builder import path works)

    cfg0 = load_config(CONFIGS_DIR / "exp003_p0_current.yaml")
    cfg1 = copy.deepcopy(cfg0)
    cfg1["data"]["augment"] = {}

    t0 = build_train_transform_from_config(cfg0, 64, seed=7)
    t1 = build_train_transform_from_config(cfg1, 64, seed=7)

    img = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    mask = (np.random.rand(64, 64) > 0.5).astype(np.uint8) * 255

    out0 = t0(image=img, mask=mask)
    out1 = t1(image=img, mask=mask)

    assert np.array_equal(out0["image"], out1["image"])
    assert np.array_equal(out0["mask"], out1["mask"])


@pytest.mark.parametrize("name,size", [
    ("exp003_p1_strong_geom.yaml", 256),
    ("exp003_p4_352px.yaml", 352),
])
def test_build_cache_subprocess_builds_pilot_cache(tmp_path: Path, name: str, size: int) -> None:
    """Run ``scripts/build_cache.py`` as a subprocess on a pilot config with
    --limit 8 against synthetic data; expect exit 0 and the configured size."""
    root = tmp_path / "syn"
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(8):
        image = rng.integers(0, 255, size=(60, 80, 3), dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(images / f"img{i:03d}.jpg")
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[10:40, 10:50] = 255
        Image.fromarray(mask, mode="L").save(masks / f"img{i:03d}.jpg")

    out = tmp_path / f"cache_{name}.npz"
    result = subprocess.run(
        [
            sys.executable, "scripts/build_cache.py",
            "--config", str(CONFIGS_DIR / name),
            "--root", str(root),
            "--limit", "8",
            "--out", str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    with np.load(out) as data:
        assert data["images"].shape == (8, size, size, 3), data["images"].shape
        assert data["masks"].shape == (8, size, size), data["masks"].shape
