"""Tests for the split generator: determinism, counts, and case-level grouping."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.make_splits import (
    group_ids,
    list_ids,
    main,
    read_ids_file,
    split_ids,
    write_splits,
)
from src.datasets import read_split

IDS = [f"cju{i:03d}" for i in range(100)]


def test_list_ids_returns_sorted_stems(tmp_path: Path) -> None:
    for name in ("b.png", "a.png", "c.jpg"):
        (tmp_path / name).write_bytes(b"x")

    assert list_ids(tmp_path, "*.png") == ["a", "b"]


def test_list_ids_empty_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No files matching"):
        list_ids(tmp_path, "*.png")


def test_list_ids_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list_ids(tmp_path / "nope")


def test_read_ids_file_skips_comments(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("# header\ncaseA\ncaseB\n", encoding="utf-8")

    assert read_ids_file(path) == ["caseA", "caseB"]


def test_read_ids_file_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("caseA\ncaseA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        read_ids_file(path)


def test_split_counts_and_disjointness() -> None:
    splits = split_ids(IDS, ratios=(0.8, 0.1, 0.1), seed=42)

    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10
    all_ids = splits["train"] + splits["val"] + splits["test"]
    assert len(all_ids) == len(set(all_ids)) == len(IDS)
    assert set(all_ids) == set(IDS)


def test_split_is_deterministic_for_a_seed() -> None:
    first = split_ids(IDS, seed=7)
    second = split_ids(IDS, seed=7)
    other = split_ids(IDS, seed=8)

    assert first == second
    assert first["train"] != other["train"]


def test_split_lists_are_sorted() -> None:
    splits = split_ids(IDS, seed=3)
    for name in ("train", "val", "test"):
        assert splits[name] == sorted(splits[name])


def test_group_split_keeps_cases_together() -> None:
    ids = [f"p{patient}_{slice_}" for patient in range(20) for slice_ in range(3)]
    splits = split_ids(ids, ratios=(0.7, 0.15, 0.15), seed=11, group_regex=r"^(?P<group>p\d+)_")

    def patients(split_name: str) -> set[str]:
        return {sample.split("_")[0] for sample in splits[split_name]}

    assert patients("train").isdisjoint(patients("val"))
    assert patients("train").isdisjoint(patients("test"))
    assert patients("val").isdisjoint(patients("test"))
    assert len(patients("train") | patients("val") | patients("test")) == 20
    # Every slice of a case sticks with its case.
    for name in ("train", "val", "test"):
        for patient in patients(name):
            assert len([s for s in splits[name] if s.startswith(f"{patient}_")]) == 3


def test_group_regex_must_match_every_id() -> None:
    with pytest.raises(ValueError, match="did not match"):
        split_ids(["caseA", "nounderscore"], group_regex=r"[^_]+_[^_]+")


def test_group_regex_needs_a_capture_group() -> None:
    with pytest.raises(ValueError, match="capture group"):
        group_ids(["abc"], r"abc")


def test_bad_ratios_raise() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_ids(IDS, ratios=(0.5, 0.1, 0.1))
    with pytest.raises(ValueError, match="non-negative"):
        split_ids(IDS, ratios=(-0.1, 0.6, 0.5))


def test_too_few_ids_raise() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        split_ids(["only-one", "only-two"], ratios=(0.5, 0.25, 0.25))


def test_write_splits_roundtrip_with_dataset_reader(tmp_path: Path) -> None:
    splits = split_ids(IDS, ratios=(0.8, 0.1, 0.1), seed=42)
    written = write_splits(splits, tmp_path, header="generated\nseed=42")

    assert set(written) == {"train", "val", "test"}
    assert read_split(written["train"]) == splits["train"]
    assert read_split(written["test"]) == splits["test"]
    assert written["train"].read_text(encoding="utf-8").startswith("# generated")


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "masks"
    source.mkdir()
    for i in range(10):
        (source / f"cju{i:02d}.png").write_bytes(b"x")
    out_dir = tmp_path / "splits"

    code = main(
        [
            "--source",
            str(source),
            "--pattern",
            "*.png",
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
            "--ratios",
            "0.6",
            "0.2",
            "0.2",
        ]
    )

    assert code == 0
    assert (out_dir / "train.txt").is_file()
    assert len(read_split(out_dir / "train.txt")) == 6
    assert len(read_split(out_dir / "val.txt")) == 2
    assert len(read_split(out_dir / "test.txt")) == 2
    assert "train: 6 ids" in capsys.readouterr().out


def test_cli_is_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "masks"
    source.mkdir()
    for i in range(12):
        (source / f"img{i:02d}.png").write_bytes(b"x")

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        main(["--source", str(source), "--pattern", "*.png", "--out-dir", str(out), "--seed", "5"])

    for name in ("train", "val", "test"):
        assert (out_a / f"{name}.txt").read_text() == (out_b / f"{name}.txt").read_text()
