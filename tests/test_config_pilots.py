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
    "exp006_p1_strong_geom.yaml",
    "exp006_p2_color_jitter.yaml",
    "exp006_p3_noise_blur.yaml",
]

EXP006 = "exp006_all_concat.yaml"
EXP006_PILOTS = ["exp006_p1_strong_geom.yaml", "exp006_p2_color_jitter.yaml", "exp006_p3_noise_blur.yaml"]
EXP003_PILOTS = ["exp003_p1_strong_geom.yaml", "exp003_p2_color_jitter.yaml", "exp003_p3_noise_blur.yaml"]

EXP009 = "exp009_concat_msu.yaml"
EXP009_PILOTS = ["exp009_p1_strong_geom.yaml", "exp009_p2_color_jitter.yaml"]


def _load(name: str) -> dict:
    return load_config(CONFIGS_DIR / name)


def _strip_comments(raw: str) -> str:
    """Return config lines with full-line comments and blank lines removed."""
    kept = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _config_values_equal(a_path: Path, b_path: Path) -> tuple[bool, list[str]]:
    """Compare two configs parameter-by-parameter (comments ignored)."""
    import yaml

    a = yaml.safe_load(_strip_comments(a_path.read_text(encoding="utf-8")))
    b = yaml.safe_load(_strip_comments(b_path.read_text(encoding="utf-8")))
    diffs: list[str] = []

    def walk(path: str, x: dict, y: dict) -> None:
        for key in sorted(set(x) | set(y)):
            kpath = f"{path}.{key}" if path else key
            if key not in x:
                diffs.append(f"only in b: {kpath} = {y[key]!r}")
            elif key not in y:
                diffs.append(f"only in a: {kpath} = {x[key]!r}")
            elif isinstance(x[key], dict) and isinstance(y[key], dict):
                walk(kpath, x[key], y[key])
            elif x[key] != y[key]:
                diffs.append(f"{kpath}: a={x[key]!r} b={y[key]!r}")

    walk("", a, b)
    return (len(diffs) == 0, diffs)


@pytest.mark.parametrize("name", EXP009_PILOTS + [EXP009])
def test_exp009_config_loads_through_real_loader(name: str) -> None:
    """Each new exp009 config must be accepted by load_config."""
    cfg = _load(name)
    assert cfg["run"]["name"] == name.replace(".yaml", "")
    assert cfg["data"]["image_size"] == 256


@pytest.mark.parametrize("name", EXP009_PILOTS + [EXP009])
def test_exp009_config_is_listed_with_pilots(name: str) -> None:
    """Sanity: the new exp009 files exist and are YAML experiment configs."""
    assert (CONFIGS_DIR / name).is_file()
    assert name.endswith(".yaml") and name != "datasets.yaml"


def test_exp009_concat_msu_equals_exp006_except_msu_and_name() -> None:
    """exp009_concat_msu must equal exp006_all_concat except run.name and
    decoder.msu (false -> true)."""
    ok, diffs = _config_values_equal(
        CONFIGS_DIR / "exp006_all_concat.yaml", CONFIGS_DIR / EXP009
    )
    allowed = ("run.name", "decoder.msu")
    unexpected = [d for d in diffs if not any(k in d for k in allowed)]
    assert ok or not unexpected, f"unexpected diffs: {unexpected}"
    assert any("run.name" in d for d in diffs), diffs
    assert any("decoder.msu" in d for d in diffs), diffs


@pytest.mark.parametrize("name", EXP009_PILOTS)
def test_exp009_pilot_equals_exp009_except_augment_and_name(name: str) -> None:
    """exp009_pN must equal exp009_concat_msu except run.name and data.augment."""
    ok, diffs = _config_values_equal(CONFIGS_DIR / EXP009, CONFIGS_DIR / name)
    allowed = ("run.name", "data.augment")
    unexpected = [d for d in diffs if not any(k in d for k in allowed)]
    assert ok or not unexpected, f"unexpected diffs: {unexpected}"
    assert any("run.name" in d for d in diffs), diffs
    assert any("data.augment" in d for d in diffs), diffs


@pytest.mark.parametrize("exp009_name,exp006_name", [
    ("exp009_p1_strong_geom.yaml", "exp006_p1_strong_geom.yaml"),
    ("exp009_p2_color_jitter.yaml", "exp006_p2_color_jitter.yaml"),
])
def test_exp009_pilot_equals_exp006_pilot_except_msu_and_name(
    exp009_name: str, exp006_name: str
) -> None:
    """exp009_pN must equal exp006_pN except run.name and decoder.msu
    (false -> true)."""
    ok, diffs = _config_values_equal(
        CONFIGS_DIR / exp006_name, CONFIGS_DIR / exp009_name
    )
    allowed = ("run.name", "decoder.msu")
    unexpected = [d for d in diffs if not any(k in d for k in allowed)]
    assert ok or not unexpected, f"unexpected diffs: {unexpected}"
    assert any("run.name" in d for d in diffs), diffs
    assert any("decoder.msu" in d for d in diffs), diffs


def test_exp009_msu_parameter_increase_matches_exp007() -> None:
    """The parameter increase from adding MSU to all-concat (exp006 -> exp009)
    must equal the increase for hybrid fusion (exp005 -> exp007), i.e. the two
    MSU links (86,016 params). Builds with pretrained=false so no weights are
    downloaded."""
    from src.models.net import build_model, count_parameters

    def total(name: str) -> int:
        cfg = _load(name)
        cfg["model"]["pretrained_a"] = False
        cfg["model"]["pretrained_b"] = False
        return count_parameters(build_model(cfg))[0]

    inc_hybrid = total("exp007_hybrid_msu.yaml") - total("exp005_hybrid_fusion.yaml")
    inc_concat = total(EXP009) - total("exp006_all_concat.yaml")
    assert inc_hybrid == inc_concat == 86_016, (inc_hybrid, inc_concat)


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


@pytest.mark.parametrize("name", EXP006_PILOTS)
def test_exp006_pilot_config_loads_through_real_loader(name: str) -> None:
    """Each new exp006 pilot config must be accepted by load_config."""
    cfg = load_config(CONFIGS_DIR / name)
    assert cfg["run"]["name"] == name.replace(".yaml", "")
    assert cfg["data"]["image_size"] == 256
    assert cfg["data"]["augment"]["enabled"] is True


@pytest.mark.parametrize("exp006_name,exp003_name", [
    ("exp006_p1_strong_geom.yaml", "exp003_p1_strong_geom.yaml"),
    ("exp006_p2_color_jitter.yaml", "exp003_p2_color_jitter.yaml"),
    ("exp006_p3_noise_blur.yaml", "exp003_p3_noise_blur.yaml"),
])
def test_exp006_pilot_augment_matches_exp003(exp006_name: str, exp003_name: str) -> None:
    """For each pN, exp006_pN's data.augment must equal exp003_pN's value for
    value (copied verbatim)."""
    exp006 = load_config(CONFIGS_DIR / exp006_name)
    exp003 = load_config(CONFIGS_DIR / exp003_name)
    assert exp006["data"]["augment"] == exp003["data"]["augment"]


@pytest.mark.parametrize("exp006_name,exp003_name", [
    ("exp006_p1_strong_geom.yaml", "exp003_p1_strong_geom.yaml"),
    ("exp006_p2_color_jitter.yaml", "exp003_p2_color_jitter.yaml"),
    ("exp006_p3_noise_blur.yaml", "exp003_p3_noise_blur.yaml"),
])
def test_exp006_pilot_model_decoder_match_exp006(exp006_name: str, exp003_name: str) -> None:
    """For each pN, exp006_pN's model and decoder sections must equal the
    exp006 config's; only run.name and data.augment may differ."""
    exp006 = load_config(CONFIGS_DIR / EXP006)
    pilot = load_config(CONFIGS_DIR / exp006_name)
    assert pilot["model"] == exp006["model"]
    assert pilot["decoder"] == exp006["decoder"]
    assert pilot["run"]["name"] != exp006["run"]["name"]
    assert pilot["train"] == exp006["train"]
    assert pilot["data"]["image_size"] == exp006["data"]["image_size"]
    assert pilot["data"]["augment"] != exp006["data"].get("augment", {})


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
