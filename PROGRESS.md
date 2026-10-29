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
- [x] 3. Meta-Polyp encoder alone with the same decoder.
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
- One branch per baseline: `model.encoder: convnext | caformer` selects it;
  dual-branch fusion is Milestone 4.
- MSU follows the paper (decoder 3 x4 -> decoder 1); `model.msu_mode:
  paper | released`, only `paper` implemented.
- The Convformer skip block is omitted; `model.skip_convformer: false` is
  reserved and `true` raises `NotImplementedError`.
- albumentations for training augmentation only; `metrics.csv` always written;
  W&B optional and off by default.
- Final results: 3 seeds (42/43/44), report mean and std.
- First T4 probe: `exp001`, batch 8, then 4 and 2 with gradient accumulation
  if memory runs out. No VRAM number is claimed until measured.
- Pretrained timm downloads are Colab-only; tests and smoke runs stay
  `pretrained: false`.
- The upstream 880/120 split comparison is deferred, not cancelled.
- Milestone 4 (fusion) is approved. Milestones 5-7 are still pending.

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

### Open questions at the end of that session (now resolved)

- T4 fit: resolved by the 2026-10-29 memory policy (reduce batch size and use
  gradient accumulation first; `exp002` only as a last resort).
- `pretrained_b` and the Colab cache: resolved in the second and third session
  notes below; Branch B is pretrained by default and the cache is rebuilt per
  Colab session.
- Real-data training is still not started. Milestones 4-7 remain.

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

## Session 2026-10-29 (third session)

### Decisions, push and Colab launcher

- Recorded the T4 memory policy, pretrained Branch B, per-session Colab cache,
  development split policy, `>127` mask threshold, three-seed final results and
  the "Milestones 4-7 need approval" rule in AGENTS.md section 11 and here;
  committed as `0c5cb7c` and pushed to `origin/main`.
- Added `notebooks/colab_run.ipynb` and commit `7aa7aeb`: Drive mount,
  `GH_TOKEN` from Colab Secrets (never printed, not stored in `.git/config`),
  clone/pull, missing-package install (never torch), dataset unzip with
  automatic `images/` + `masks/` discovery, cache build, `pytest -q`, and a
  temp session config that points `data.root`, `data.cache_path` and
  `run.out_dir` outside the repo. The launcher uses `python -m src.train`.
- `src/train.py` now writes `peak_gpu_mem_mb` per epoch and
  `peak_gpu_mem_mb`, `mean_epoch_time_s` and `epochs_this_session` to
  `run_info.json`; it records the cache SHA-256, prints the GPU name, and
  refuses to append to a `metrics.csv` with an incompatible header. `--out-dir`
  overrides a Drive run folder. Tests at that commit: **153 passed**.
- README has a "Running on Colab" section and the recommended three runs;
  this file has the empty "Colab results" table above. No VRAM number has been
  measured yet.

### Milestone 3: Meta-Polyp encoder alone

- Verified before choosing the wrapper that timm **1.0.30** supports
  `features_only=True, out_indices=(0, 1, 2, 3)` for
  `caformer_s18.sail_in1k`. It reports channels `[64, 128, 320, 512]` at
  reductions `[4, 8, 16, 32]`, so no `forward_intermediates` or stage-hook
  fallback was needed.
- Added `src/models/metapolyp_enc.py`: `MetaPolypEncoder` with the same
  `forward(x) -> list[Tensor]` and `.out_channels` interface as the ConvNeXt
  wrapper, plus the 32-divisibility check and an exact width/stride check.
- Added `model.encoder: convnext | caformer` (default `convnext`) and made
  `PolypSegNet` single-branch: Branch A uses `align_a`, Branch B uses
  `align_b`, both feed the same `PlainDecoder`. Dual-branch and mixed fusion
  still raise `NotImplementedError` (Milestone 4).
- Configs: `exp003_caformer_s18_baseline.yaml` (pretrained Branch B,
  `lr_encoder_b: 1e-4`) and `exp004_caformer_s18_scratch.yaml`
  (`pretrained_b: false`, `lr_encoder_b: 3e-4`, extra baseline only).
- Normalisation check: `convnext_base.fb_in1k` and
  `caformer_s18.sail_in1k` both declare mean `(0.485, 0.456, 0.406)` and std
  `(0.229, 0.224, 0.225)`. They match, so the shared ImageNet
  `data.mean`/`data.std` is unchanged.
- Parameter counts (CPU, random weights, no download): ConvNeXt-Base baseline
  **97.22 M** (87.56 M encoder + 9.66 M head); CAFormer-S18 branch **32.55 M**
  (23.24 M encoder + 9.31 M head).
- Tests: **166 passed** in 50.5 s on CPU. Added Meta-Polyp shape/stride and
  input-divisibility tests, Branch B parameter-group and overfit-one-batch
  tests, a full `(2, 1, 256, 256)` output test, Branch B config consistency
  tests, and a Branch B `--smoke` end-to-end test. No test downloads weights.
- Standalone Branch B `--smoke` run passed: `model: caformer
  (caformer_s18.sail_in1k)  params: 32.55M`, synthetic val Dice `0.0000`
  (meaningless pipeline check, not a result).
- No pretrained weights were downloaded and no real training was run.

### Open questions (current)

- Does ConvNeXt-Base at 256x256 fit the free T4? The first probe uses
  `exp001` with batch size 8, then 4, then 2 with gradient accumulation; it
  must measure peak VRAM and epoch time. Nothing is claimed until measured.
- Before the first real Branch B run, confirm that downloading the pinned
  `caformer_s18.sail_in1k` ImageNet weights is allowed. Downloads are now
  approved on Colab only; nothing has been downloaded locally.
- The upstream 880/120 split comparison remains optional and separate from the
  committed 800/100/100 development split. It is deferred, not cancelled.
- Milestone 4 (fusion) is approved. Milestones 5-7 remain pending explicit
  approval.

