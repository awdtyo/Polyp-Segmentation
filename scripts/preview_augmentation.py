#!/usr/bin/env python
"""Save a grid of original vs augmented image-mask overlays for visual inspection.

Usage:
    python scripts/preview_augmentation.py --config configs/exp003_p2_color_jitter.yaml \
        --split train --n 8 --out /tmp/aug_preview

Reads the config, builds the training transform, and for each of N random
training images saves side-by-side panels: original image+mask overlay and
augmented image+mask overlay. No model or checkpoint needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import PolypDataset, build_train_transform_from_config
from src.utils import load_config


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a mask (red) onto an RGB image."""
    out = image.copy().astype(np.float32)
    red = np.zeros_like(out)
    red[..., 0] = 255.0
    alpha = 0.4 * mask[..., None]
    out = out * (1 - alpha) + red * alpha
    return out.clip(0, 255).astype(np.uint8)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Preview training augmentation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("/tmp/aug_preview"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data = cfg["data"]
    image_size = int(data["image_size"])
    transform = build_train_transform_from_config(cfg, image_size, seed=args.seed)

    dataset = PolypDataset.from_config(cfg, args.split, transform=None)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.n, len(dataset)), replace=False)

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        img = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = sample["mask"][0].numpy().astype(np.uint8)
        # Apply transform manually
        if transform is not None:
            aug = transform(image=img, mask=mask)
            img_aug = np.asarray(aug["image"])
            mask_aug = np.asarray(aug["mask"]).astype(np.uint8)
        else:
            img_aug, mask_aug = img, mask

        orig_panel = _overlay(img, mask)
        aug_panel = _overlay(img_aug, mask_aug)
        combined = np.concatenate([orig_panel, aug_panel], axis=1)
        Image.fromarray(combined).save(out_dir / f"preview_{i:03d}.png")

    print(f"Saved {len(indices)} preview images to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
echo "created"