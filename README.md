# Dual-encoder polyp segmentation

ConvNeXt + Meta-Polyp (CAFormer) dual-encoder segmentation model: per-stage
fusion, U-Net decoder with Multi-scale Upsampling (MSU). The full specification
lives in [AGENTS.md](AGENTS.md); this README covers setup, layout and the data
contract.

Status: **Milestones 1-2 of 7 complete** (skeleton and dataset preparation;
ConvNeXt-Base baseline with a plain decoder, training loop and resume). See
[PROGRESS.md](PROGRESS.md) for the live checklist.

## Layout

```
AGENTS.md              project specification (read this first)
PROGRESS.md            milestone checklist, decisions in force, session notes
configs/base.yaml      default experiment config
data/splits/           committed split files (train/val/test .txt)
scripts/make_splits.py  generates the split files once, with a fixed seed
scripts/build_cache.py  resizes images/masks once into one .npz cache
src/utils.py           config loading/validation, seeding
src/datasets.py        PolypDataset, transforms
src/metrics.py         per-image Dice and IoU
src/losses.py          BCE+Dice, Dice, Jaccard, BCE
src/models/            encoders, fusion, decoder, MSU, net (milestones 2-5)
src/train.py           training entry point (milestone 2)
src/evaluate.py        evaluation entry point (milestone 6)
tests/                 CPU tests, a few seconds
results/results.csv    one row per evaluated run
notebooks/             thin Colab launcher (milestone 7)
```

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Colab, do not reinstall torch - it is preinstalled.

## Config

Every hyperparameter lives in YAML under `configs/`. Values omitted from a file
fall back to `DEFAULT_CONFIG` in `src/utils.py`; unknown keys are rejected so a
typo cannot silently change a run. Copy `configs/base.yaml` to
`expNNN_<description>.yaml` per experiment.

```
python - <<'PY'
from src.utils import load_config
cfg = load_config("configs/base.yaml")
print(cfg["data"]["image_size"], cfg["model"]["fusion"])
PY
```

`data.image_size` must be divisible by 32 (the encoders downsample by 32).

## Data contract

The project dataset is **Kvasir-SEG**: 1000 2D RGB colonoscopy images with 1000
matching masks, used for both training and evaluation. The verified layout is:

```
<data.root>/images/<id>.jpg
<data.root>/masks/<id>.jpg
```

`data.root` defaults to `data/raw`, which on the development machine is a
git-ignored symlink to the raw download:

```
ln -s ../../../dataset/Kvasir-SEG/Kvasir-SEG data/raw
```

Every image is RGB; every mask is a grayscale JPEG. Masks are **not strictly
binary** (JPEG compression leaves about 18 grey levels), so they are binarised
at `data.mask_threshold: 127` *before* resizing and resized with
nearest-neighbour. `data.mask_ext` is therefore `.jpg` and `data.mask_suffix` is
empty. `PolypDataset` returns
`{"image": (3, H, W) float32, "mask": (1, H, W) float32 in {0, 1}, "id": str}`;
images are normalised with ImageNet mean/std inside the dataset so both branches
share one preprocessing.

For faster epochs, build the optional cache once:

```
python scripts/build_cache.py --config configs/base.yaml \
    --out data/cache/kvasir_seg_256.npz
```

It resizes every pair to 256x256 and writes one `.npz` (`images` uint8,
`masks` uint8 0/1). For the current 1000-image Kvasir-SEG copy the file is
**143.5 MB** and takes about 11 s to build on the CPU box. Set
`data.cache_path` to that file to use it; an empty string reads the JPEGs
directly. The cache is an I/O optimisation only: it changes neither the split
nor the labels, and it is never committed.

> **Leakage caveat.** Kvasir-SEG provides no patient or case identifiers.
> Splits are therefore image level, so images from the same patient or
> endoscopy session cannot be guaranteed to stay in one split, and leakage
> between train/val/test cannot be ruled out. This must be stated in the paper
> write-up; it is a property of the public dataset, not something the code can
> fix.

## Splits

Generated once and committed, so a number is reproducible from
(config, commit, split file):

```
python scripts/make_splits.py --source data/raw/images --pattern "*.jpg" \
    --out-dir data/splits --seed 42 --ratios 0.8 0.1 0.1
```

The committed split is 800/100/100 (train/val/test) with seed 42, image level
and with no `--group-regex`; the three lists are disjoint and cover all 1000
ids. The upstream 880/120 `train.txt`/`val.txt` files shipped with the download
are not used. The test split is only ever read by `evaluate.py --split test`.

## Tests

```
python -m pytest
```

Covers: config merge/validation, dataset loading on a synthetic folder in a temp
dir (no real data), npz cache building and raw-vs-cache equivalence, Dice/IoU
against hand-computed values, loss behaviour and differentiability, split
generation (counts, determinism, group disjointness), and the committed
Kvasir-SEG split (800/100/100, disjoint). The albumentations augmentation test is
skipped when albumentations is not installed; everything else runs with torch +
Pillow + YAML only.

## Reproducibility rules

- One config per experiment, committed. Runs write `config_resolved.yaml` and
  `run_info.json` (milestone 2).
- Seeds are set from `run.seed` for Python, NumPy and torch.
- Train/val only for training and model selection; test is touched once.
- Data, checkpoints, run folders and `.env` never enter git.

## Backbone note

The Meta-Polyp encoder is `caformer.CAFormerS18(pretrained="imagenet")` from
`keras_cv_attention_models==1.3.9` (TensorFlow/Keras). Its stacks are
`num_blocks=[3, 3, 9, 3]`, `out_channels=[64, 128, 320, 512]`, with ConvFormer
(separable-conv) token mixers in stages 1-2 and self-attention in stages 3-4.
That is exactly timm's `caformer_s18`, which is what `model.encoder_b` points at.
`caformer_s36` is **not** the same model (depths `[3, 12, 18, 3]`), even though
it shares the channel widths.

## Roadmap

Milestones 1-2 are complete. Milestone 3 (Meta-Polyp encoder with the same
decoder) is next; later milestones add fusion, MSU, evaluation and the Colab
launcher (AGENTS.md section 9). See [PROGRESS.md](PROGRESS.md).
