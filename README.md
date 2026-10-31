# Dual-encoder polyp segmentation

ConvNeXt + Meta-Polyp (CAFormer) dual-encoder segmentation model: per-stage
fusion, U-Net decoder with Multi-scale Upsampling (MSU). The full specification
lives in [AGENTS.md](AGENTS.md); this README covers setup, layout and the data
contract.

Status: **Milestones 1-5 of 7 complete** (skeleton and dataset preparation;
ConvNeXt-Base baseline with a plain decoder, training loop and resume; Meta-Polyp
CAFormer-S18 branch with the same decoder; dual-encoder fusion with concat at
stages 1-2 and cross-attention at stages 3-4; paper-mode multi-scale upsampling
links). Evaluation and the final README remain. See
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
  convnext_enc.py      Branch A: ConvNeXt via timm
  metapolyp_enc.py     Branch B: CAFormer-S18 (Meta-Polyp encoder) via timm
  fusion.py            per-stage concat and cross-attention fusion
  msu.py               paper-mode multi-scale upsampling links
  decoder.py           channel alignment + U-Net decoder with optional MSU
  net.py               assembled dual/single-branch model, parameter groups
src/train.py           training entry point (milestone 2)
src/evaluate.py        evaluation entry point (milestone 6)
tests/                 CPU tests (about 1-2 minutes)
results/results.csv    one row per evaluated run
notebooks/             thin Colab launcher (`colab_run.ipynb`)
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
`model.encoder: convnext | caformer | dual` selects the branch layout. The
baselines use `single_a`/`single_b` fusion, for example
`model.fusion: [single_b, single_b, single_b, single_b]`, and
`decoder.msu: false`. The dual model sets `model.encoder: dual` and uses one of
the fusion presets `hybrid` (`[concat, concat, xattn, xattn]`), `all_concat`, or
an explicit four-stage list. `exp005` is the hybrid fusion config, `exp006` is
the all-concat ablation, and `exp007` is hybrid fusion with MSU. `decoder.msu`
defaults to `false`; `decoder.msu_mode: paper` is implemented as D4 x4 into D2
and D3 x4 into D1. `released` is accepted by config validation but the builder
raises `NotImplementedError` because that variant is deliberately reserved.

| Config | Encoder(s) | Fusion | MSU | Purpose |
|---|---|---|---|---|
| `base.yaml` | dual | hybrid | off | defaults for new experiments |
| `exp001_convnext_base_baseline.yaml` | ConvNeXt-Base | `single_a` | off | ConvNeXt baseline; first T4 probe |
| `exp002_convnext_small_baseline.yaml` | ConvNeXt-Small | `single_a` | off | fallback if ConvNeXt-Base runs out of memory on a T4 |
| `exp003_caformer_s18_baseline.yaml` | CAFormer-S18 | `single_b` | off | Meta-Polyp encoder baseline (pretrained) |
| `exp004_caformer_s18_scratch.yaml` | CAFormer-S18 | `single_b` | off | from-scratch extra baseline |
| `exp005_hybrid_fusion.yaml` | dual | `hybrid` | off | hybrid fusion (concat S1-S2, xattn S3-S4) |
| `exp006_all_concat.yaml` | dual | `all_concat` | off | all-concat ablation |
| `exp007_hybrid_msu.yaml` | dual | `hybrid` | `paper` | hybrid fusion + MSU |
| `exp008_convnext_base_lr1e4.yaml` | ConvNeXt-Base | `single_a` | off | fair-tuning check for the ConvNeXt baseline |

`exp008` is identical to `exp001` except `run.name` and `train.lr_convnext`
(`1.0e-4` instead of `3.0e-5`), so the ConvNeXt baseline can be compared with
`exp003` at the same encoder learning rate. `exp001` is left untouched.

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
binary** (all 1000 masks contain values other than 0/255; 19 grey levels
overall), so they are binarised at `data.mask_threshold: 127` *before* resizing
and resized with nearest-neighbour. `data.mask_ext` is therefore `.jpg` and
`data.mask_suffix` is empty. `PolypDataset` returns
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
Kvasir-SEG split (800/100/100, disjoint). Model tests cover both encoders
(4 maps, strides 4/8/16/32, expected widths), every implemented fusion mode
(concat, cross-attention and the presets), MSU link shapes, D4->D2 and D3->D1
information flow, MSU gradients, the unchanged Milestone-4 state dict when MSU
is off, the Branch A, Branch B and hybrid overfit-one-batch checks,
parameter-group coverage, and an end-to-end `--smoke` run for each branch layout,
including `exp005`, `exp006` and `exp007`. No test downloads pretrained weights:
every test passes `pretrained=False`. The albumentations augmentation test is
skipped when albumentations is not installed; everything else runs with torch +
Pillow + YAML only.

## Running on Colab

The launcher is `notebooks/colab_run.ipynb`. It is deliberately thin: it mounts
Drive, gets the repo, installs only missing Python packages, unzips the data,
builds the cache, runs `pytest`, and calls `python -m src.train`. Torch is never
reinstalled.

1. Upload the Kvasir-SEG zip to Drive and set `DRIVE_ZIP` in the first cell.
2. Open the notebook in Colab and choose **Runtime > Change runtime type > T4
   GPU**.
3. Add `GH_TOKEN` in **Colab Secrets** (key icon). The token is used for the
   clone/fetch only; it is not printed and is not written into `.git/config`.
4. Set `CONFIG`, `RUNS_DIR`, `REF` (commit, tag or branch to check out) and the
   optional `EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM` and `RUN_SUFFIX` overrides in
   the first cell.
5. Run the cells top to bottom.

`REF` is checked out instead of a plain `git pull`, so a rerun cannot silently
pick up newer code than the one recorded in `run_info.json`; a commit hash is
the most reproducible choice. `RUN_SUFFIX` is appended to `run.name`, so a
probe with a different epoch count, seed or batch size gets its own folder
instead of resuming into the previous one. `GRAD_ACCUM` overrides
`train.grad_accum_steps` when `BATCH_SIZE` has to drop on a T4.

The training cell streams the child process output line by line and prints the
exit code, raising only when it is non-zero. `src.train` also logs every
end-of-run step (checkpoint saves, prediction grid, `run_info.json` rewrite,
cleanup) before and after, so a failure after the last epoch is visible in the
streamed log instead of being lost in a buffered pipe.

`RUNS_DIR` is on Drive, so checkpoints survive a disconnect. Data is read from
`/content`, never directly from Drive. The cache is rebuilt in `/content` at the
start of each session and is never committed.

### Resume after a disconnect

Rerun the training cell with the same `RUNS_DIR`, `CONFIG` and run name. If
`<RUNS_DIR>/<run name>/last.pt` exists, `src.train` restores model, optimizer,
scheduler, scaler, epoch, best Dice and RNG state and prints a `resumed from ...`
message; only the remaining epochs run. Use `--no-resume` only when you intend
to restart from scratch. Do not resume a short probe with a different epoch
count: the cosine schedule would change. Use a separate `RUNS_DIR` for the probe.

### Recommended first three runs

1. **GPU pipeline check:** leave `SMOKE=True`. This uses synthetic data, 64 px
   and 3 optimiser steps (with `exp001`, a tiny ConvNeXt encoder). It proves the
   GPU path and notebook wiring; its Dice is meaningless.
2. **Five-epoch probe:** set `SMOKE=False`, `CONFIG` to
   `configs/exp001_convnext_base_baseline.yaml`, `EPOCHS=5`, `BATCH_SIZE=8` and a
   probe `RUNS_DIR`. If the T4 runs out of memory, try batch 4 and then 2,
   raising `train.grad_accum_steps` to keep the effective batch size. This is a
   memory/time probe, not a reported result.
3. **Choose the real run:** read `peak_gpu_mem_mb` and `mean_epoch_time_s` from
   `run_info.json` (and the per-epoch rows in `metrics.csv`). Pick the largest
   batch size that fits and the epoch count you can afford, then start the real
   run in a fresh `RUNS_DIR` with `EPOCHS=None`. If the T4 runs out of memory,
   reduce `BATCH_SIZE` and raise `train.grad_accum_steps` to keep the effective
   batch size; `exp002` (`convnext_small`) is the last resort, not a silent
   default change. No VRAM number is claimed here: measure it on your T4 and
   record it in the "Colab results" table in [PROGRESS.md](PROGRESS.md).

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

With timm 1.0.30, `caformer_s18.sail_in1k` supports
`features_only=True, out_indices=(0, 1, 2, 3)` directly; the wrapper reports
channels `[64, 128, 320, 512]` at reductions `[4, 8, 16, 32]`. No forward hooks
or `forward_intermediates` fallback are used. The pretrained configs of
`convnext_base.fb_in1k` and `caformer_s18.sail_in1k` both use ImageNet
mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`, so the shared
preprocessing in `data.mean` / `data.std` is correct.

Parameter counts with random weights (no download): the ConvNeXt-Base baseline
is **97.22 M** (87.56 M encoder + 9.66 M head); the CAFormer-S18 branch is
**32.55 M** (23.24 M encoder + 9.31 M head). The dual hybrid model (`exp005`) is
**122.68 M** before MSU, and the all-concat ablation (`exp006`) is **127.78 M**.
The two MSU links add **0.086 M** (86,016 parameters), so the hybrid + MSU model
(`exp007`) is **122.76 M** in total. A real `exp003`, `exp005` or `exp007` run
downloads the pinned `sail_in1k` weights through timm; `exp004` is the
from-scratch extra baseline. These are parameter counts, not memory
measurements: no VRAM number is claimed.

## Roadmap

Milestones 1-5 are complete. Milestone 6 (evaluation, results logging and the
ablation configs) is next, then the final README; Milestones 6-7 still need
explicit approval. The thin Colab launcher (`notebooks/colab_run.ipynb`) is
already in place. See [PROGRESS.md](PROGRESS.md).
