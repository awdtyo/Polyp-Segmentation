# PROGRESS.md

Living status file. AGENTS.md is the specification; this file records what is
actually done, what is in flight, and what is blocked. Update it at the end of
every session.

Last updated: 2026-10-29.

## Milestone checklist

- [x] 1. Skeleton: repo layout, `requirements.txt`, `.gitignore`, config
      loading, dataset class with synthetic-data test, metrics, losses, tests
      passing.
- [x] 2. Baseline: ConvNeXt encoder + plain decoder (no fusion, no MSU),
      `train.py` with checkpoint and resume, `--smoke` run passes.
- [ ] 3. Meta-Polyp encoder alone with the same decoder.
- [ ] 4. Fusion: concat at stages 1 and 2, cross-attention at stages 3 and 4,
      plus the `all_concat` variant.
- [ ] 5. MSU links on and off.
- [ ] 6. `evaluate.py`, results logging, and ablation configs for: A only,
      B only, all concat, hybrid fusion, hybrid fusion + MSU, hybrid fusion
      without MSU.
- [ ] 7. `notebooks/colab_run.ipynb` launcher and a README section on running
      on Colab.

## Decisions in force

Full reasoning and supersession history live in AGENTS.md section 11. Short
version:

- Dataset: Kvasir-SEG, 1000 2D RGB images, used for training and evaluation.
- Layout: `images/<id>.jpg`, `masks/<id>.jpg`; masks are grayscale JPEG, so
  they are binarised at >127 before resizing.
- Splits: image level, 800/100/100, seed 42, no group regex. Kvasir-SEG has no
  patient ids, so cross-split leakage cannot be ruled out.
- `image_size: 256`; `convnext_base.fb_in1k` default, `convnext_small` fallback
  config.
- Branch B: `caformer_s18.sail_in1k`. No TensorFlow -> PyTorch weight remap.
- MSU follows the paper (decoder 3 x4 -> decoder 1); `model.msu_mode:
  paper | released`, only `paper` implemented.
- The Convformer skip block is omitted; `model.skip_convformer: false` is
  reserved and `true` raises `NotImplementedError`.
- albumentations for training augmentation only; `metrics.csv` always written;
  W&B optional and off by default.
- Final results: 3 seeds (42/43/44), report mean and std.

## Last session (2026-10-29)

- Read AGENTS.md and PROGRESS.md, then oriented the repo: one commit existed
  (Milestone 1). The working tree already carried uncommitted draft Milestone 2
  files from the previous session; they were reviewed, finished and committed
  rather than discarded.
- Inspected `/home/aditya/Desktop/polyp/dataset` without assumptions. Found
  `Kvasir-SEG/Kvasir-SEG/{images,masks,annotated_images,bbox}` plus 1000 CSVs
  and a PDF; 1000 images and 1000 masks with identical stems; every image RGB,
  every mask grayscale JPEG; sizes range from 332x352 to 1920x1072 (most common
  622x530), and every mask contains values other than 0/255. Confirmed
  `images/<id>.jpg` / `masks/<id>.jpg`, with the caveat that JPEG masks are not
  strictly binary.
- Wrote the dataset, split, cache and results decisions into AGENTS.md
  section 11 and created this file.
- Generated and committed `data/splits/{train,val,test}.txt`: 800/100/100,
  seed 42, disjoint, all 1000 ids covered. The upstream 880/120 split is unused.
- Added `scripts/build_cache.py` and cache loading in `PolypDataset`
  (`data.cache_path`). Built the real 256x256 cache:
  `data/cache/kvasir_seg_256.npz`, **143.5 MB**, about 11 s on this CPU box.
  Raw and cached samples were checked to be byte-identical, and the cache is
  git-ignored.
- Finished Milestone 2: ConvNeXt wrapper, plain U-Net decoder with
  `ConvTranspose2d` upsampling and concat skips, assembled net, differential-LR
  AdamW, cosine schedule with optional warmup, AMP/GradScaler on CUDA, gradient
  accumulation/clipping, automatic resume from `last.pt`, `best.pt`,
  `metrics.csv`, `run_info.json`, prediction grids, and `--smoke`.
- Parameter counts (CPU, randomly initialised, no weights downloaded):
  `convnext_base.fb_in1k` baseline **97.22 M** (87.56 M encoder + 9.66 M
  alignment/decoder/head); `convnext_small.fb_in1k` fallback **58.93 M**
  (49.45 M + 9.47 M). No VRAM number is claimed; that needs a T4 run.
- Tests: `pytest -q` -> **152 passed** on CPU. The smoke run passed and wrote
  `config_resolved.yaml`, `run_info.json`, `metrics.csv`, `last.pt`, `best.pt`
  and a prediction grid. Its synthetic val Dice (0.1986) is a pipeline check,
  not a result.
- Fixed two issues found while finishing: `last.pt` is now written after the
  best-check so a disconnect cannot make resume overwrite `best.pt` with a worse
  epoch, and `metrics.csv` logs the LR at epoch start rather than the
  post-anneal value (which was 0 in a one-epoch smoke run).

### Open questions

- Confirm on Colab whether ConvNeXt-Base at 256x256 fits the free T4; if not,
  switch to `configs/exp002_convnext_small_baseline.yaml` (a config change, not
  a silent default change).
- Decide `pretrained_b` for the first branch-B run (Milestone 3), and whether
  to build the npz cache on Colab or read JPEGs directly.
- Real-data training is not started. Milestones 3-7 remain.

## Session 2026-10-29 (second session)

Step 0 decisions, also recorded in AGENTS.md section 11:

- **T4 memory:** ConvNeXt-Base stays the default. On OOM, reduce batch size and
  use gradient accumulation first; `exp002` (convnext_small) is a last resort,
  never a silent default change.
- **Branch B:** `caformer_s18.sail_in1k`, pretrained. The from-scratch config
  (`exp004`) is an extra baseline only.
- **Cache:** rebuilt on Colab at the start of each session from the raw Drive
  zip with `scripts/build_cache.py`; never committed. Its SHA-256 goes into
  `run_info.json`.
- **Splits:** development and ablations use the committed 800/100/100 split
  (seed 42). The official upstream 880/120 lists are not used yet; they may be
  added later as a separate config for a final published-number comparison.
- **Mask threshold:** stays `> 127`, applied before resizing.
- **Final results:** 3 seeds (42/43/44), mean and std.
- **Milestones 4-7:** pending and must not start without approval.

These resolve the open questions from the previous session: pretrained Branch B
is the default, and the Colab cache is rebuilt per session.

## Colab results

To be filled in by the user after each run on the free T4 (16 GB). Values come
from `metrics.csv` / `run_info.json`; the smoke row is a pipeline check, not a
result. No VRAM number is claimed until it is measured.

| Run | Config | Batch size | Peak VRAM (GB) | Mean epoch time (s) | Val Dice |
|---|---|---:|---:|---:|---:|
| GPU smoke (3 steps, synthetic) | exp001 | | | | |
| 5-epoch memory/time probe | exp001 | | | | |
| Real run | exp001 | | | | |

