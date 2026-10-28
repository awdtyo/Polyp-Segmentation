"""Build the train/val/test split files once, with a fixed seed, then commit them.

Two modes:

* ``--source DIR [--pattern GLOB]``: take file stems from a folder, e.g. the mask
  folder of Kvasir-SEG.
* ``--ids-file FILE``: take ids from a text file (one per line) when you already
  have a canonical list.

If the dataset has patient/case ids, pass ``--group-regex`` with a named group
``group`` (or use the first capture group). Whole groups are then assigned to a
single split, so slices from one case never leak across train/val/test
(AGENTS.md section 6).

Examples:
    python scripts/make_splits.py --source data/raw/masks --pattern "*.png" \\
        --out-dir data/splits --seed 42 --ratios 0.8 0.1 0.1

    python scripts/make_splits.py --source data/raw/images --pattern "*.jpg" \\
        --group-regex "^(?P<group>[^_]+)_" --out-dir data/splits
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


def list_ids(source: str | Path, pattern: str = "*") -> list[str]:
    """List sorted file stems in a folder, ignoring hidden files.

    Args:
        source: Folder to scan.
        pattern: Glob pattern, e.g. ``"*.png"``.

    Returns:
        Sorted list of stems.

    Raises:
        FileNotFoundError: if the folder does not exist.
        ValueError: if nothing matched.
    """
    folder = Path(source)
    if not folder.is_dir():
        raise FileNotFoundError(f"Source folder not found: {folder}")
    ids = sorted(path.stem for path in folder.glob(pattern) if path.is_file() and not path.name.startswith("."))
    if not ids:
        raise ValueError(f"No files matching {pattern!r} in {folder}")
    return ids


def read_ids_file(path: str | Path) -> list[str]:
    """Read ids from a text file, one per line, ignoring blanks and ``#`` comments.

    Args:
        path: Text file.

    Returns:
        List of ids.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"IDs file not found: {file_path}")
    ids = [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not ids:
        raise ValueError(f"No ids found in {file_path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate ids in {file_path}")
    return ids


def group_ids(ids: Sequence[str], group_regex: str | None) -> dict[str, list[str]]:
    """Group sample ids by case/patient id.

    Args:
        ids: Sample ids.
        group_regex: Regex with a named group ``group``, or at least one capture
            group. ``None`` puts every id in its own group.

    Returns:
        Mapping ``group -> [ids]``.

    Raises:
        ValueError: if the regex does not match an id or has no group to use.
    """
    if group_regex is None:
        return {sample_id: [sample_id] for sample_id in ids}

    pattern = re.compile(group_regex)
    groups: dict[str, list[str]] = {}
    for sample_id in ids:
        match = pattern.search(sample_id)
        if match is None:
            raise ValueError(f"group_regex {group_regex!r} did not match id {sample_id!r}")
        if "group" in pattern.groupindex:
            group = match.group("group")
        elif pattern.groups >= 1:
            group = match.group(1)
        else:
            raise ValueError("group_regex must define a named group 'group' or at least one capture group")
        groups.setdefault(group, []).append(sample_id)
    return groups


def _target_counts(total: int, ratios: Sequence[float]) -> list[int]:
    """Turn ratios into integer counts that sum exactly to ``total``.

    Args:
        total: Number of samples (or groups).
        ratios: Three ratios summing to 1.

    Returns:
        ``[n_train, n_val, n_test]``.
    """
    if len(ratios) != 3:
        raise ValueError(f"Expected 3 ratios, got {len(ratios)}")
    if any(r < 0 for r in ratios):
        raise ValueError("Ratios must be non-negative")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")
    if total < 3:
        raise ValueError(f"Need at least 3 samples to fill 3 splits, got {total}")
    counts = [int(total * ratios[0]), int(total * ratios[1])]
    counts.append(total - counts[0] - counts[1])
    # Guarantee at least one sample in val and test when possible.
    for index in (2, 1):
        if counts[index] == 0:
            donor = max(range(3), key=lambda i: counts[i])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[index] += 1
    return counts


def split_ids(
    ids: Sequence[str],
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    group_regex: str | None = None,
) -> dict[str, list[str]]:
    """Split ids into train/val/test, at case level when groups are given.

    Args:
        ids: All sample ids.
        ratios: Three ratios (train, val, test) summing to 1.
        seed: RNG seed; the same seed always yields the same split.
        group_regex: Optional regex extracting a case id (see :func:`group_ids`).

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}``, each list sorted.

    Raises:
        ValueError: on duplicate ids, bad ratios, or unmatched group regex.
    """
    unique = list(dict.fromkeys(ids))
    if len(unique) != len(ids):
        raise ValueError("Duplicate ids passed to split_ids")

    groups = group_ids(unique, group_regex)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)

    if group_regex is None:
        # Ungrouped: split on ids, so exact counts hold.
        targets = _target_counts(len(keys), ratios)
        assigned = {"train": keys[: targets[0]], "val": keys[targets[0] : targets[0] + targets[1]]}
        assigned["test"] = keys[targets[0] + targets[1] :]
    else:
        # Grouped: fill splits by group, approximating the ratios in group count.
        targets = _target_counts(len(keys), ratios)
        buckets: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
        for key in keys:
            for name, target in zip(SPLIT_NAMES, targets):
                if len(buckets[name]) < target:
                    buckets[name].append(key)
                    break
            else:
                buckets[SPLIT_NAMES[-1]].append(key)
        assigned = buckets

    return {
        name: sorted(sample_id for key in assigned[name] for sample_id in groups[key])
        for name in SPLIT_NAMES
    }


def write_splits(splits: Mapping[str, Sequence[str]], out_dir: str | Path, header: str | None = None) -> dict[str, Path]:
    """Write one text file per split.

    Args:
        splits: Mapping name -> ids.
        out_dir: Destination folder, created if needed.
        header: Optional comment block written as ``#`` lines at the top.

    Returns:
        Mapping name -> written path.
    """
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name in SPLIT_NAMES:
        path = folder / f"{name}.txt"
        lines = [f"# {line}" for line in (header or "").splitlines() if line.strip()]
        lines.extend(splits[name])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written[name] = path
    return written


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Create fixed train/val/test split files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", type=Path, help="Folder whose file stems become sample ids.")
    source.add_argument("--ids-file", type=Path, help="Text file with one id per line.")
    parser.add_argument("--pattern", default="*", help="Glob used with --source (default: *)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/splits"), help="Where to write the split files.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42).")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios (default: 0.8 0.1 0.1).",
    )
    parser.add_argument("--group-regex", default=None, help="Regex with a 'group' capture for case-level splitting.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """Run the split generator.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    ids = read_ids_file(args.ids_file) if args.ids_file else list_ids(args.source, args.pattern)
    splits = split_ids(ids, ratios=args.ratios, seed=args.seed, group_regex=args.group_regex)
    header = (
        f"Generated by scripts/make_splits.py\n"
        f"seed={args.seed} ratios={'/'.join(f'{r:g}' for r in args.ratios)} "
        f"group_regex={args.group_regex!r}\n"
        f"total_ids={len(ids)}"
    )
    written = write_splits(splits, args.out_dir, header=header)
    for name, path in written.items():
        print(f"{name}: {len(splits[name])} ids -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
