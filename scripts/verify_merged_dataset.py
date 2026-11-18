"""Verify the merged Kvasir-SEG + CVC-ClinicDB dataset layout.

This script checks the merged dataset prepared on the new Google Drive and
optionally writes split files (.txt) that PolypDataset can consume directly.

Usage:
    python scripts/verify_merged_dataset.py --data-root /content/drive/MyDrive/polyp-seg-data
    python scripts/verify_merged_dataset.py --data-root /content/drive/MyDrive/polyp-seg-data --write-splits /content/merged_splits
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def read_preparation_report(report_path: Path) -> dict[str, Any]:
    """Read and validate the preparation report."""
    if not report_path.is_file():
        raise FileNotFoundError(f"Preparation report not found: {report_path}")
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    required_keys = {"total", "splits", "source_distribution"}
    missing = required_keys - set(report.keys())
    if missing:
        raise ValueError(f"Preparation report missing keys: {missing}")
    return report


def read_split_csv(csv_path: Path) -> list[dict[str, str]]:
    """Read a split CSV file."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_source_from_id(sample_id: str) -> str:
    """Determine source dataset from the sample ID prefix."""
    if sample_id.startswith("KvasirSEG__"):
        return "Kvasir-SEG"
    if sample_id.startswith("CVCClinicDB__"):
        return "CVC-ClinicDB"
    return "unknown"


def verify_dataset(data_root: Path, write_splits: Path | None = None) -> bool:
    """Verify the merged dataset layout and return True if all checks pass."""
    all_ok = True

    def ok(msg: str) -> None:
        print(f"[PASS] {msg}")

    def fail(msg: str) -> None:
        nonlocal all_ok
        all_ok = False
        print(f"[FAIL] {msg}")

    def warn(msg: str) -> None:
        print(f"[WARN] {msg}")

    print(f"\n=== Verifying merged dataset at {data_root} ===\n")

    # 1. Check preparation report
    report_path = data_root / "preparation_report.json"
    try:
        report = read_preparation_report(report_path)
        ok(f"Preparation report found: {report_path}")
    except Exception as e:
        fail(f"Preparation report: {e}")
        return False

    # 2. Check train/val/test folders
    splits = ["train", "val", "test"]
    all_ids: dict[str, str] = {}  # id -> split
    all_ids_by_split: dict[str, list[str]] = {s: [] for s in splits}
    all_ids_by_source: dict[str, list[str]] = {"Kvasir-SEG": [], "CVC-ClinicDB": []}
    split_source_counts: dict[str, dict[str, int]] = {s: {"Kvasir-SEG": 0, "CVC-ClinicDB": 0} for s in splits}

    total_images = 0
    total_masks = 0
    empty_mask_count = 0
    mask_value_issues = 0

    for split in splits:
        images_dir = data_root / split / "images"
        masks_dir = data_root / split / "masks"
        csv_path = data_root / "splits" / f"{split}.csv"

        if not images_dir.is_dir():
            fail(f"Missing images folder: {images_dir}")
            continue
        if not masks_dir.is_dir():
            fail(f"Missing masks folder: {masks_dir}")
            continue
        if not csv_path.is_file():
            fail(f"Missing split CSV: {csv_path}")
            continue

        ok(f"Split '{split}': folders and CSV exist")

        # Read CSV
        try:
            csv_rows = read_split_csv(csv_path)
        except Exception as e:
            fail(f"Split '{split}' CSV read error: {e}")
            continue

        # Verify CSV rows match files
        csv_ids = [row["source_stem"] for row in csv_rows]
        image_files = {p.stem for p in images_dir.glob("*.png")}
        mask_files = {p.stem for p in masks_dir.glob("*.png")}

        # Check that every CSV id has image and mask
        missing_images = set(csv_ids) - image_files
        missing_masks = set(csv_ids) - mask_files
        if missing_images:
            fail(f"Split '{split}': {len(missing_images)} CSV ids missing image files (e.g., {list(missing_images)[:3]})")
        else:
            ok(f"Split '{split}': all {len(csv_ids)} CSV ids have image files")
        if missing_masks:
            fail(f"Split '{split}': {len(missing_masks)} CSV ids missing mask files (e.g., {list(missing_masks)[:3]})")
        else:
            ok(f"Split '{split}': all {len(csv_ids)} CSV ids have mask files")

        # Check for extra files not in CSV
        extra_images = image_files - set(csv_ids)
        extra_masks = mask_files - set(csv_ids)
        if extra_images:
            warn(f"Split '{split}': {len(extra_images)} image files not in CSV (e.g., {list(extra_images)[:3]})")
        if extra_masks:
            warn(f"Split '{split}': {len(extra_masks)} mask files not in CSV (e.g., {list(extra_masks)[:3]})")

        # Check ID uniqueness across splits
        for sample_id in csv_ids:
            if sample_id in all_ids:
                fail(f"ID '{sample_id}' appears in both '{all_ids[sample_id]}' and '{split}' splits")
            else:
                all_ids[sample_id] = split
                all_ids_by_split[split].append(sample_id)
                source = get_source_from_id(sample_id)
                all_ids_by_source[source].append(sample_id)
                split_source_counts[split][source] += 1

        total_images += len(image_files)
        total_masks += len(mask_files)

        # Verify mask binary values (sample a few)
        import numpy as np
        from PIL import Image
        for row in csv_rows[:5]:  # check first 5 masks per split
            mask_path = masks_dir / f"{row['source_stem']}.png"
            if mask_path.is_file():
                try:
                    mask = np.asarray(Image.open(mask_path))
                    if mask.ndim == 3:
                        mask = mask.max(axis=2)
                    unique_vals = set(np.unique(mask).tolist())
                    if not unique_vals.issubset({0, 255}) and not unique_vals.issubset({0, 1}):
                        mask_value_issues += 1
                        warn(f"Mask '{mask_path.name}' has non-binary values: {sorted(unique_vals)[:10]}")
                    if mask.sum() == 0:
                        empty_mask_count += 1
                except Exception as e:
                    warn(f"Could not read mask '{mask_path}': {e}")

    # 3. Total counts
    print(f"\n=== Counts ===")
    print(f"Total images on disk: {total_images}")
    print(f"Total masks on disk:  {total_masks}")
    print(f"Total IDs in CSVs:    {len(all_ids)}")
    print(f"Reported total:       {report.get('total', 'N/A')}")

    if len(all_ids) == report.get("total", -1):
        ok(f"Total count matches preparation report ({len(all_ids)})")
    else:
        fail(f"Total count mismatch: CSVs have {len(all_ids)}, report says {report.get('total')}")

    # Per-split counts
    for split in splits:
        expected = report["splits"].get(split, 0)
        actual = len(all_ids_by_split[split])
        if actual == expected:
            ok(f"Split '{split}': {actual} (matches report)")
        else:
            fail(f"Split '{split}': {actual} (report: {expected})")

    # Per-source counts
    print(f"\n=== Source Distribution ===")
    for source in ["Kvasir-SEG", "CVC-ClinicDB"]:
        total_source = len(all_ids_by_source[source])
        print(f"{source}: {total_source} total")
        for split in splits:
            count = split_source_counts[split][source]
            print(f"  {split}: {count}")
        # Check against report
        reported = sum(report["source_distribution"][split].get(source, 0) for split in splits)
        if total_source == reported:
            ok(f"{source} total matches report ({total_source})")
        else:
            fail(f"{source} total mismatch: found {total_source}, report {reported}")

    # 4. Cross-split ID overlap check (already done above during collection)

    # 5. Image-mask pairing (already checked above)

    # 6. Mask binary check
    if mask_value_issues == 0:
        ok("All sampled masks are binary (0/255 or 0/1)")
    else:
        warn(f"{mask_value_issues} masks had non-binary values (sampled first 5 per split)")

    # 7. Empty mask check
    if empty_mask_count == 0:
        ok("No empty masks found in sampled masks")
    else:
        warn(f"{empty_mask_count} empty masks found in sampled masks (first 5 per split)")

    # 8. Split protocol
    print(f"\n=== Split Protocol ===")
    protocol = report.get("protocol", "not specified")
    split_seed = report.get("split_seed", "not specified")
    print(f"Reported protocol: {protocol}")
    print(f"Reported split_seed: {split_seed}")
    if "random per image" in protocol.lower() or "random" in protocol.lower():
        warn("Split is random per image (not by patient/case sequence). Near-duplicate ClinicDB frames may leak across splits.")
    elif "sequence" in protocol.lower() or "patient" in protocol.lower() or "case" in protocol.lower():
        ok("Split appears to be by sequence/patient/case (no per-image leakage)")
    else:
        warn("Split protocol unclear from report")

    # 9. Write split files if requested
    if write_splits is not None:
        write_splits.mkdir(parents=True, exist_ok=True)
        for split in splits:
            split_file = write_splits / f"{split}.txt"
            with split_file.open("w", encoding="utf-8") as f:
                for sample_id in sorted(all_ids_by_split[split]):
                    f.write(f"{sample_id}\n")
            ok(f"Wrote {split_file} ({len(all_ids_by_split[split])} ids)")

    print(f"\n=== Summary ===")
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED - see above")

    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify merged Kvasir-SEG + CVC-ClinicDB dataset")
    parser.add_argument("--data-root", required=True, type=Path, help="Root of merged dataset (e.g., /content/drive/MyDrive/polyp-seg-data)")
    parser.add_argument("--write-splits", type=Path, default=None, help="Optional directory to write train.txt/val.txt/test.txt for PolypDataset")
    args = parser.parse_args(argv)

    if not args.data_root.is_dir():
        print(f"ERROR: Data root not found: {args.data_root}", file=sys.stderr)
        return 2

    ok = verify_dataset(args.data_root, args.write_splits)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())