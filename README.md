# Dual-encoder polyp segmentation

ConvNeXt + Meta-Polyp (CAFormer) dual-encoder segmentation model: per-stage
fusion, U-Net decoder with Multi-scale Upsampling (MSU). The full specification
lives in [AGENTS.md](AGENTS.md); this README covers setup, layout and the data
contract.

Status: **Milestone 1 of 7 complete** (repo skeleton, config loading, dataset,
Dice/IoU metrics, BCE+Dice and Jaccard losses, tests). No model or training code
yet.

## Layout

```
AGENTS.md              project specification (read this first)
configs/base.yaml      default experiment config
data/splits/           committed split files (train/val/test .txt)
scripts/make_splits.py  generates the split files once, with a fixed seed
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

Paths only, no dataset assumptions beyond this folder shape:

```
<data.root>/<data.image_dir>/<id><data.image_ext>
<data.root>/<data.mask_dir>/<id><data.mask_suffix><data.mask_ext>
```

`<id>` is the file stem written in the split file. `PolypDataset` returns
`{"image": (3, H, W) float32, "mask": (1, H, W) float32 in {0, 1}, "id": str}`.
Masks are binarised at `data.mask_threshold` before resizing and resized with
nearest-neighbour, so they stay exactly 0/1. Images are normalised with ImageNet
mean/std inside the dataset (`data.normalize`, `data.mean`, `data.std`), so both
branches share one preprocessing.

> The dataset itself is not chosen yet (TODO in AGENTS.md section 1). The layout
> above is the PraNet/Kvasir-SEG-style layout, which is what Meta-Polyp used. If
> your data differs (3D volumes, `CVC`-style naming, masks in `.tif`), say so and
> `datasets.py` gets adjusted rather than guessed.

## Splits

Generated once and committed, so a number is reproducible from
(config, commit, split file):

```
python scripts/make_splits.py --source data/raw/masks --pattern "*.png" \
    --out-dir data/splits --seed 42 --ratios 0.8 0.1 0.1
```

With patient/case ids, add `--group-regex "^(?P<group>[^_]+)_"` so all slices of
a case stay in one split. The test split is only ever read by
`evaluate.py --split test`.

## Tests

```
python -m pytest          # ~1.5 s on CPU
```

Covers: config merge/validation, dataset loading on a synthetic folder in a temp
dir (no real data), Dice/IoU against hand-computed values, loss behaviour and
differentiability, and split generation (counts, determinism, case-level
disjointness). The albumentations augmentation test is skipped when
albumentations is not installed; everything else runs with torch + Pillow + YAML
only.

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

Milestone 2 (ConvNeXt baseline + decoder + `train.py` with resume) starts only
after approval. See AGENTS.md section 9.
