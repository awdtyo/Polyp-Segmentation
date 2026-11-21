# Datasets and the data tree

This document describes the data layout the code reads, the dataset registry,
the preparation step, and how the multi-dataset evaluation uses them. It is the
reference for Milestone 8 (multi-dataset evaluation); the model and training
rules are in [AGENTS.md](../AGENTS.md).

## 1. Data tree and `DATA_ROOT`

All data lives under one root, `DATA_ROOT`:

```
<DATA_ROOT>/raw/<folder>/          raw images and masks, exactly as shipped
<DATA_ROOT>/processed/<name>/      prepared cache (data.npz + meta.json)
<DATA_ROOT>/splits/                committed Kvasir split files (in the repo, not here)
```

`DATA_ROOT` is resolved, in order, by `src/data_registry.resolve_data_root`:

1. the `DATA_ROOT` environment variable,
2. `data.data_root` in the experiment config,
3. `<repo>/data` (the default).

On the development machine `<repo>/data/raw` is a git-ignored symlink to the raw
download. On Colab the whole tree lives on Drive (raw and processed) and
`processed/` is copied to `/content` for fast reads; see
[notebooks/colab_eval.ipynb](../notebooks/colab_eval.ipynb).

The entire `data/` tree is git-ignored (`.gitignore: data/*`), except
`data/splits/`, which is committed.

## 2. Registry

`configs/datasets.yaml` lists every dataset. Each entry has:

| Field | Meaning |
|---|---|
| `name` | key used everywhere (`data.dataset`, `--dataset`, `processed/<name>`) |
| `folder` | folder under `<DATA_ROOT>/raw/` |
| `expected_count` | published number of images; a mismatch is reported loudly, never fixed |
| `source` | human-readable dataset name (recorded in `meta.json` and the eval JSON) |
| `image_subdir` / `mask_subdir` | sub-folders of the raw folder |
| `mask_suffix` | optional suffix stripped from a mask stem before matching |

Images and masks are paired by **filename stem** (after stripping `mask_suffix`).
The extensions may differ and are discovered from disk, never assumed.

| name | source | expected | layout | status |
|---|---|---|---|---|
| `kvasir_seg` | Kvasir-SEG | 1000 | `images/*.jpg`, `masks/*.jpg` | present; training and evaluation set |
| `cvc_clinicdb` | CVC-ClinicDB | 612 | `images/*.png`, `masks/*.png` | present; zero-shot |
| `cvc_colondb` | CVC-ColonDB | 380 | `images/*.png`, `masks/*.png` | present; zero-shot |
| `etis` | ETIS-LaribPolypDB | 196 | `images/*.png`, `masks/*.png` | present; zero-shot (1225x966 native) |
| `cvc_300` | CVC-300 | 60 | `images/*.png`, `masks/*.png` | present; zero-shot (574x500 native) |

A dataset that is absent is **skipped with a warning**; it is only an error when
it is requested by name (`--dataset etis`). Kvasir-SEG provides no patient or
case identifiers, so its split is image level and cross-split leakage cannot be
ruled out (see the README leakage caveat).

## 3. Preparation (`scripts/prepare_datasets.py`)

Converts `raw/<folder>` into `processed/<name>/{data.npz,meta.json}` in exactly
the Kvasir cache format, so `PolypDataset` reads it with no extra code path.

```
python scripts/prepare_datasets.py                       # all present datasets
python scripts/prepare_datasets.py --dataset cvc_300     # one, must be present
python scripts/prepare_datasets.py --size 256 --mask-threshold 127
python scripts/prepare_datasets.py --force               # rebuild a different size
```

- `data.npz`: `ids` (unicode, N), `images` (N, S, S, 3) uint8, `masks` (N, S, S)
  uint8 in {0, 1}. Images use a bilinear resize, masks a nearest-neighbour resize
  **after** binarisation at `mask_threshold` (`> 127`), matching the Kvasir cache.
- `meta.json`: name, source, raw/processed paths, image count, expected count and
  match flag, original-resolution summary, output size, resample modes, mask
  threshold, extensions, npz size and timestamp.
- Idempotent: an existing `data.npz` at the same size is skipped; a different
  size is refused unless `--force`.
- Loud failures: missing raw folder, unmatched image/mask stem, ambiguous stem,
  empty mask, mask values outside {0, 1}.
- Normalisation (ImageNet mean/std) is **not** stored in the npz; it is applied
  at read time by `PolypDataset.__getitem__`, so one cache serves every run.

For the current copy: `kvasir_seg` 143.5 MB, `cvc_clinicdb` 60.0 MB,
`cvc_colondb` 45.2 MB. `data/processed/kvasir_seg/data.npz` is bit-for-bit
identical to the committed training cache `data/cache/kvasir_seg_256.npz`.

## 4. Zero-shot evaluation (`src/evaluate.py`)

```
python -m src.evaluate --run_dir <run> --split test \
    --dataset kvasir_seg --dataset cvc_clinicdb --dataset cvc_colondb
```

- The **training dataset** (`data.dataset`, `kvasir_seg`) is read through the
  committed split files, so its `val` Dice reproduces the training log. `test`
  prints an explicit warning and refuses to run on a split that leaks.
- Every **other** dataset is read zero-shot from `processed/<name>/data.npz`
  using **all** of its images, and is recorded as `split=test`.
- The checkpoint is always `best.pt`, i.e. the one selected by **Kvasir val
  Dice**. External datasets never select or tune a checkpoint.
- Dice and IoU are computed **per image** on the 0.5-thresholded sigmoid output
  and then averaged; MAE is the mean absolute error between the predicted
  probability map and the binary ground truth, also per image then averaged.
- Flip TTA is off by default; `--tta-flip` averages a flipped pass in probability
  space before thresholding and writes `*_ttaflip` outputs.

Outputs, inside the run folder, one set per `(split, dataset, tta)`:

| File | Contents |
|---|---|
| `eval_<split>_<dataset>[_ttaflip].json` | mean/std Dice, IoU, MAE, n, small-lesion Dice, provenance |
| `per_image_<split>_<dataset>[_ttaflip].csv` | `id, dice, iou, mae, lesion_pixels, small_lesion` per image |

One row per `(run_name, split, dataset, tta)` is appended to the results CSV
(default `results/results.csv`); a duplicate is refused unless `--force`.

`--config PATH` overrides the run's `config_resolved.yaml`, which is how the
Colab launcher points a run at prepared data when the training paths no longer
exist.

## 5. Aggregation (`scripts/aggregate_results.py`)

```
python scripts/aggregate_results.py --runs-dir <RUNS_DIR> --split test
```

Reads every `eval_*.json` (+ its `per_image_*.csv`) under `--runs-dir`, groups by
config (a `_seed<N>` suffix is stripped) **within** each `(dataset, split, tta)`
cell, and writes:

- `aggregate_results.md`: a **Dice** table and an **IoU** table, rows = configs,
  columns = datasets, cells = `mean +/- std` and `n/a` for a single seed or an
  unevaluated cell, plus a details table and an optional paired comparison.
- `aggregate_results.csv`: the long form, one row per `(config, dataset, split,
  tta)`, for pivoting or joining.

## 6. Small-lesion area fraction (`data.small_lesion_area_fraction`)

A lesion is "small" when its foreground pixel count is below
`data.small_lesion_area_fraction` of the image area. The threshold is **chosen
from ground-truth mask statistics only**, never from model scores, and is
configurable per experiment in the config file (recorded in each processed
`meta.json`).

**Default: 0.05 (5% of image area = 3277 px at 256x256).** This is the lowest of
the tested thresholds (1%, 2%, 3%, 5%, 10%) at which every dataset — including
the held-out Kvasir test split — has at least 10 qualifying images (the minimum
for reporting `small_lesion_dice`; below 10 it is written `n/a`, never `0.0`).
It was set on 2026-11-01 from the committed ground-truth masks at 256x256.

Per-dataset share of images qualifying as small at the 5% default
(ground-truth masks only):

| Dataset | n | qualify at 5% | share |
|---|---|---:|---:|
| Kvasir test | 100 | 18 | 18% |
| Kvasir all | 1000 | 200 | 20% |
| CVC-ClinicDB | 612 | 228 | 37% |
| CVC-ColonDB | 380 | 227 | 60% |
| ETIS | 196 | 141 | 72% |
| CVC-300 | 60 | 53 | 88% |

Resizing masks from native resolution to 256x256 (nearest-neighbour after
binarisation) changes the classification by at most one image per threshold, so
the 256px statistics are the ones that drive reporting.

Masks are resized with nearest-neighbour interpolation after binarisation at
`mask_threshold` (`> 127`); tiny polyps do not vanish materially — the area
fraction ranking is essentially identical at native and 256px resolution.

## 7. Limitations

- Every number is **single-seed and provisional**. Final tables need three seeds
  (42/43/44) on a frozen commit.
- `etis` and `cvc_300` are not on the development machine; their layout is
  unverified and their rows are missing until the data is added.
- Kvasir-SEG splits are image level; cross-split leakage cannot be ruled out.
- The external datasets are evaluated zero-shot and are not used for checkpoint
  selection or tuning in any way.

## 8. Configurable augmentation (`data.augment`)

All augmentation is train-only; val/test use deterministic resize+normalise.
If `data.augment` is absent, the default pipeline (flips 0.5, rotate 20 deg p 0.5,
RandomResizedCrop 0.85-1.0 p 0.5, RandomBrightnessContrast 0.1 p 0.5) is used,
matching the original behaviour exactly (regression test in
`tests/test_datasets.py`).

Fields (all optional, defaults shown):

| Field | Default | Effect |
|---|---|---|
| `enabled` | true | false disables all augmentation |
| `hflip_p` | 0.5 | Horizontal flip probability |
| `vflip_p` | 0.5 | Vertical flip probability |
| `rotation_degrees` | 20.0 | Rotation limit (degrees) |
| `rotation_p` | 0.5 | Rotation probability |
| `random_resized_crop_scale` | [0.85, 1.0] | Scale range for RandomResizedCrop |
| `random_resized_crop_p` | 0.5 | RandomResizedCrop probability |
| `brightness_limit` | 0.1 | RandomBrightnessContrast brightness |
| `contrast_limit` | 0.1 | RandomBrightnessContrast contrast |
| `color_jitter_p` | 0.0 | ColorJitter probability (replaces RBC when > 0) |
| `brightness_jitter` | 0.25 | ColorJitter brightness factor |
| `contrast_jitter` | 0.25 | ColorJitter contrast factor |
| `saturation_jitter` | 0.25 | ColorJitter saturation factor |
| `hue_jitter` | 0.03 | ColorJitter hue factor |
| `gauss_noise_p` | 0.0 | GaussNoise probability (off by default) |
| `gauss_noise_var_limit` | [5.0, 20.0] | GaussNoise variance range |
| `gaussian_blur_p` | 0.0 | GaussianBlur probability (off by default) |
| `gaussian_blur_limit` | 3 | Max blur kernel size |
| `specular_inpaint` | false | Specular highlight detection + inpainting |
| `specular_threshold` | 0.95 | Brightness threshold for specular detection |
| `clahe` | false | CLAHE on L channel of LAB |
| `clahe_clip_limit` | 2.0 | CLAHE clip limit |

Geometric ops (flips, rotate, crop) apply identically to image and mask via
albumentations; masks use nearest-neighbour and padding value 0 (no false
positives). Colour, noise, CLAHE and specular ops are image-only.

Seed all random augmentation from the run seed for reproducibility.

## 9. Resolution (256 vs 352)

- `data.image_size` is configurable (default 256, 352 option). Must be divisible by 32.
- `scripts/build_cache.py --size 352` rebuilds from raw images (not by upscaling
  the 256 cache). Same for `scripts/prepare_datasets.py --size 352` on all
  external datasets.
- Keep 256 and 352 processed data in separate folders/filenames so both coexist.
- Evaluation reads `image_size` from each run's own config and uses processed
  data at that size. A 352 model that tries to load a 256 cache raises
  `ValueError: Cache image size (256) does not match requested image_size (352)`.
- GPU memory at 352 for dual-encoder configs: expect ~1.9x the 256 footprint
  (area scales by 352²/256² ≈ 1.89). On a T4 (16 GB), the existing
  `batch_size: 8` with `grad_accum_steps: 1` may need to be reduced to 4 with
  `grad_accum_steps: 2` to keep the same effective batch size. No VRAM number
  is claimed until measured on Colab.
