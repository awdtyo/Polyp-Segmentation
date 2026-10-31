"""Tests for the dataset registry and DATA_ROOT resolution (Milestone 8, Step 0).

CPU-only and fast: nothing here touches a real dataset, only the committed
registry file and temporary folders.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import data_registry
from src.data_registry import RegistryError

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The five datasets the project evaluates on, with their published sizes.
EXPECTED = {
    "kvasir_seg": 1000,
    "cvc_clinicdb": 612,
    "cvc_colondb": 380,
    "etis": 196,
    "cvc_300": 60,
}


def test_registry_lists_all_five_datasets_with_expected_counts() -> None:
    specs = data_registry.load_registry()

    assert list(specs) == list(EXPECTED)
    assert {name: spec.expected_count for name, spec in specs.items()} == EXPECTED


def test_registry_entries_use_the_committed_layout() -> None:
    for spec in data_registry.load_registry().values():
        assert spec.image_subdir == "images"
        assert spec.mask_subdir == "masks"
        assert spec.mask_suffix == ""


def test_registry_names_matches_the_registry() -> None:
    assert data_registry.registry_names() == list(EXPECTED)


def test_raw_and_processed_dirs_follow_the_data_root() -> None:
    spec = data_registry.get_spec("kvasir_seg")
    root = Path("/tmp/some-data-root")

    assert spec.raw_dir(root) == root / "raw" / "kvasir_seg"
    assert spec.processed_dir(root) == root / "processed" / "kvasir_seg"
    assert spec.image_dir(root) == root / "raw" / "kvasir_seg" / "images"
    assert spec.mask_dir(root) == root / "raw" / "kvasir_seg" / "masks"


def test_unknown_dataset_raises() -> None:
    with pytest.raises(RegistryError, match="Unknown dataset"):
        data_registry.get_spec("nope")


def test_missing_registry_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="not found"):
        data_registry.load_registry(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "content, match",
    [
        ("", "must be a mapping"),
        ("datasets: []", "non-empty list"),
        ("datasets:\n  - name: a\n", "is missing"),
        ("datasets:\n  - name: a\n    folder: a\n    expected_count: 0\n", "must be positive"),
        ("datasets:\n  - name: a\n    folder: a\n    expected_count: 1\n    bogus: 2\n", "unknown keys"),
        (
            "datasets:\n  - name: a\n    folder: a\n    expected_count: 1\n"
            "  - name: a\n    folder: b\n    expected_count: 1\n",
            "duplicate",
        ),
    ],
)
def test_malformed_registry_is_rejected(tmp_path: Path, content: str, match: str) -> None:
    path = tmp_path / "datasets.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(RegistryError, match=match):
        data_registry.load_registry(path)


def test_resolve_data_root_precedence(tmp_path: Path) -> None:
    config_value = str(tmp_path / "from-config")
    env_value = str(tmp_path / "from-env")

    # No env, no config -> the repo default.
    assert data_registry.resolve_data_root("", env={}) == data_registry.DEFAULT_DATA_ROOT
    assert data_registry.resolve_data_root(None, env={}) == data_registry.DEFAULT_DATA_ROOT
    # A config value wins over the default.
    assert data_registry.resolve_data_root(config_value, env={}) == Path(config_value)
    # The environment wins over the config value.
    assert data_registry.resolve_data_root(config_value, env={"DATA_ROOT": env_value}) == Path(env_value)


def test_resolve_data_root_ignores_a_blank_env_value(tmp_path: Path) -> None:
    assert data_registry.resolve_data_root("", env={"DATA_ROOT": "   "}) == data_registry.DEFAULT_DATA_ROOT
    assert data_registry.resolve_data_root(str(tmp_path), env={"DATA_ROOT": ""}) == tmp_path


def test_presence_and_present_datasets(tmp_path: Path) -> None:
    spec = data_registry.get_spec("kvasir_seg")

    assert spec.is_present(tmp_path) is False
    assert data_registry.present_datasets(tmp_path) == []

    (tmp_path / "raw" / "kvasir_seg" / "images").mkdir(parents=True)
    (tmp_path / "raw" / "kvasir_seg" / "masks").mkdir(parents=True)

    assert spec.is_present(tmp_path) is True
    assert [item.name for item in data_registry.present_datasets(tmp_path)] == ["kvasir_seg"]


def test_data_root_from_config(tmp_path: Path) -> None:
    assert data_registry.data_root_from_config({"data": {"data_root": str(tmp_path)}}) == tmp_path
    assert data_registry.data_root_from_config({"data": {}}) == data_registry.DEFAULT_DATA_ROOT
    assert data_registry.data_root_from_config({}) == data_registry.DEFAULT_DATA_ROOT
