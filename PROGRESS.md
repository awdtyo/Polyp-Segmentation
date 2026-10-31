# PROGRESS.md

Living status file. AGENTS.md is the specification; this file records what is
actually done, what is in flight, and what is blocked. Update it at the end of
every session.

Last updated: 2026-10-31.

## Milestone checklist

- [x] 1. Skeleton: repo layout, `requirements.txt`, `.gitignore`, config
      loading, dataset class with synthetic-data test, metrics, losses, tests
      passing.
- [x] 2. Baseline: ConvNeXt encoder + plain decoder (no fusion, no MSU),
      `train.py` with checkpoint and resume, `--smoke` run passes.
- [x] 3. Meta-Polyp encoder alone with the same decoder.
- [x] 4. Fusion: concat at stages 1 and 2, cross-attention at stages 3 and 4,
      plus the `all_concat` variant.
- [x] 5. MSU links on and off.
- [x] 6. `evaluate.py`, results logging, and ablation configs for: A only,
      B only, all concat, hybrid fusion, hybrid fusion + MSU, hybrid fusion
      without MSU. **Done 2026-10-31**: `src/evaluate.py` (test-split guard,
      one results row per run/split, per-image CSV), `scripts/aggregate_results.py`
      (mean/std over seeds, `n/a` for one seed, paired bootstrap comparison),
      seed tests, the Colab `SEED`/evaluate/aggregate cells, and the README
      "Full ablation" section. The ablation runs themselves have not been
      launched; that is the user's Colab work.
- [ ] 7. `notebooks/colab_run.ipynb` launcher and a README section on running
      on Colab. **Partial**: launcher (streaming output, `REF`, `RUN_SUFFIX`,
      `GRAD_ACCUM`) and the README section are in place; the final README pass
      is still to do.
- [ ] 8. Multi-dataset evaluation (user-specified, outside AGENTS.md section 9):
      zero-shot evaluation of the Kvasir-trained checkpoints on Kvasir-SEG test
      plus CVC-ClinicDB, CVC-ColonDB, ETIS and CVC-300. **In progress**: Steps 0
      (dataset registry + `DATA_ROOT` resolution), 1 (eval/prep report), 2
      (`scripts/prepare_datasets.py`, run on the real data), 3 (multi-dataset
      `src/evaluate.py`), 4 (config x dataset aggregation), 5 (test suite for
      the multi-dataset path), 6 (`notebooks/colab_eval.ipynb`) and the Step 7
      docs (`docs/datasets.md`, the README evaluation-protocol pass) are done.
      Step 7's real-number validation (exp003 Kvasir val first, then the full
      exp003/exp005/exp006/exp007 x dataset table) needs the Colab checkpoints
      and remains. Everything here is single-seed and provisional, not a final
      result.

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
  dual-branch fusion sets `model.encoder: dual` and runs both encoders.
- Fusion presets: `hybrid` (concat S1-S2, xattn S3-S4), `all_concat`,
  `single_a`, `single_b`; an explicit four-stage list is also accepted.
- MSU follows the paper: D4 x4 -> D2 and D3 x4 -> D1. `decoder.msu` defaults
  false; `decoder.msu_mode: paper | released`, only `paper` implemented.
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
- Milestone 6 is approved and in progress (evaluate, aggregation, seeds, tests,
  README ablation section). Milestone 7 is partial (launcher and README section
  exist).
- exp008 is a 5-epoch probe: if it clearly beats exp001, it replaces exp001 as
  the ConvNeXt baseline in the final ablations; otherwise exp001 stays.
- Final protocol: one epoch count N for every run (chosen from the 30-epoch
  validation curves), seeds 42/43/44, same committed split, batch size, image
  size and frozen commit (tag `ablation-v1`). Test split evaluated once at the
  end only.

## Session 2026-10-29 (first session)

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

Measured by the user on the free T4 (16 GB) and copied from `metrics.csv` /
`run_info.json`. The smoke row is a pipeline check, not a result. Blank cells
were not measured; they are never filled by guesswork.

| Run | Config | Batch size | Peak VRAM (GB) | Mean epoch time (s) | Val Dice |
|---|---|---:|---:|---:|---|
| GPU smoke (3 steps, synthetic) | exp001 | n/a | n/a | n/a | n/a |
| 5-epoch memory/time probe | exp001 | 8 (to confirm in `config_resolved.yaml`) | 2.794 | 23.85 | 0.8655 (best, epoch 4) |
| 5-epoch memory/time probe | exp003 | n/a | 1.951 | 15.66 | 0.8869 (best, epoch 3; last epoch 0.8845) |
| Real run | exp001 | | | | |

Notes:

- The GPU smoke row is `n/a` on purpose. The pytest smoke run on a T4 used
  `convnext_tiny` at 64 px and peaked at 703 MB, which is a pipeline check of
  the smoke path, not a measurement of either model in the table.
- Peak VRAM is `torch.cuda.max_memory_allocated()`; the table is in GB, the
  measured values were 2794 MB (exp001) and 1951 MB (exp003).
- exp001 and exp003 used **different encoder learning rates** (`exp001`
  `train.lr_convnext: 3.0e-5`, `exp003` `train.lr_encoder_b: 1.0e-4`). The two
  rows are therefore not a clean architecture comparison, and the Dice gap must
  not be reported as a ConvNeXt vs CAFormer result. `exp008` exists to repeat
  the ConvNeXt baseline at `1e-4`.
- The "Real run" row is still empty: no full-length run has been measured.

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

## Session 2026-10-29 (fourth session)

### Step 0 decisions

- Recorded the first-T4-probe policy (`exp001`, batch 8, fallback 4 then 2 with
  gradient accumulation), Colab-only pretrained downloads, the deferred
  upstream 880/120 comparison, and the Milestone 4 approval in AGENTS.md
  section 11 and this file. Committed as `38393b6` and pushed to `origin/main`.

### Milestone 4: dual-encoder fusion

- Added `src/models/fusion.py`:
  - `ConcatFusion`: concat on channels, 3x3 conv + BatchNorm + ReLU back to
    `C_i`.
  - `CrossAttentionFusion`: flatten to `(B, H*W, C_i)`, ConvNeXt query,
    Meta-Polyp key/value, `nn.MultiheadAttention(8, batch_first=True)`,
    residual add to the query, LayerNorm, reshape. Optional fixed 2-D
    sinusoidal encoding is implemented and tested; it is off in `exp005` and
    `exp006`.
  - `resolve_fusion_modes` accepts the `hybrid`, `all_concat`, `single_a` and
    `single_b` presets, repeated `concat`/`xattn`, or an explicit four-stage
    list. `FusionStack` rejects single-branch modes because the model assembly
    skips the unused encoder instead.
- `src/models/net.py` now assembles the dual model when
  `model.encoder: dual`: both encoders run on the same input, each stage is
  aligned twice, fused, then decoded by the existing `PlainDecoder`. The
  `single_a`/`single_b` presets still skip the other encoder and keep the
  Milestone 2-3 baselines unchanged (parameter counts match the old baselines).
  `fusion_heads` and `pos_enc_2d` are read from the config.
- Parameter groups stay `encoder_a`, `encoder_b`, `head`; `head` now contains
  alignment, fusion, decoder and the segmentation head. All three LRs are
  logged in `metrics.csv`.
- Added `configs/exp005_hybrid_fusion.yaml` (`fusion: hybrid`) and
  `configs/exp006_all_concat.yaml` (`fusion: all_concat`). Both use the pinned
  pretrained encoders, `image_size: 256`, batch 8 and `msu: false`.
- `src/train.py` smoke mode preserves the configured branch layout and fusion
  preset instead of forcing a single branch, so `exp005`/`exp006` smoke tests
  exercise the real dual path. `run_info.json` records `encoder: dual`, both
  encoder names, and the resolved per-stage fusion list.
- `src/utils.py` accepts `model.encoder: dual`, validates `fusion_heads`, and
  rejects preset names inside a per-stage list.

### Parameter counts (CPU, random weights, no download)

| Component | exp005 hybrid | exp006 all_concat |
|---|---:|---:|
| Encoder A (ConvNeXt-Base) | 87.564 M | 87.564 M |
| Encoder B (CAFormer-S18) | 23.237 M | 23.237 M |
| Alignment A | 0.731 M | 0.731 M |
| Alignment B | 0.387 M | 0.387 M |
| Fusion (S1-S4) | 1.832 M | 6.932 M |
| Decoder + head | 8.926 M | 8.926 M |
| **Total** | **122.678 M** | **127.778 M** |

Fusion by stage, hybrid: 0.074 / 0.295 / 0.412 / 1.052 M (concat, concat,
xattn, xattn). All-concat: 0.074 / 0.295 / 1.844 / 4.720 M. These are parameter
counts, not VRAM measurements.

### Tests and smoke runs

- `pytest -q`: **199 passed** in 77.44 s on CPU. No test downloads weights.
  New coverage: fusion preset parsing and rejection of invalid lists; concat
  and cross-attention shapes at 256 px; cross-attention output changes when
  either branch input changes; stage-3/stage-4 token counts are 256/64; full
  net output for all four presets at `(2, 3, 256, 256)`; single modes match the
  old baseline parameter counts; dual parameter groups; dual optimizer LRs;
  positional-encoding behavior; hybrid overfit-one-batch; and `--smoke` for
  both `exp005` and `exp006`.
- Standalone CPU smoke runs passed for `exp005` and `exp006`:
  `model: dual (convnext_tiny.fb_in1k + caformer_s18.sail_in1k)` with 62.75 M
  and 67.85 M smoke parameters respectively. Synthetic val Dice was `0.0000`
  in both; that is a pipeline check, not a result. No epoch time or VRAM number
  from these CPU runs is reported as a measurement.
- No pretrained weights were downloaded locally, no full training was run, and
  no VRAM or real-data metric is claimed.

### Open questions (current)

- Does ConvNeXt-Base at 256x256 fit the free T4? The first probe uses
  `exp001` with batch size 8, then 4, then 2 with gradient accumulation; it
  must measure peak VRAM and epoch time. Nothing is claimed until measured.
- The dual hybrid model has 122.68 M parameters before MSU. It needs its own
  T4 memory measurement after the `exp001` probe; no VRAM number is claimed
  from the CPU parameter count.
- Downloads of the pinned `caformer_s18.sail_in1k` weights are approved on
  Colab only. Nothing has been downloaded locally.
- The upstream 880/120 split comparison remains optional and separate from the
  committed 800/100/100 development split. It is deferred, not cancelled.
- Milestone 4 is complete. Milestones 5-7 remain pending explicit approval.

## Session 2026-10-29 (fifth session)

### Colab GPU test compatibility

- `tests/test_train.py::test_smoke_run_end_to_end` no longer asserts
  `peak_gpu_mem_mb == 0.0` on every machine. A small device-aware helper now
  requires exactly `0.0` on CPU and `> 0.0` on CUDA, and it checks both
  `run_info.json` and the first `metrics.csv` row. The same test also checks
  that `run_info.json` records `device: cpu | cuda` consistently with
  `torch.cuda.is_available()`.
- Audited the other end-to-end tests for device assumptions. The `mini` and
  `--smoke` configs keep AMP disabled, the CPU unit tests use CPU tensors and a
  disabled scaler, and the only timing assertion remains `mean_epoch_time_s >
  0`, which holds for both CPU and GPU. No other hard-coded CPU-only peak or
  timing assertions were found. The CUDA branch itself was not run locally
  (CPU-only box); Colab remains the GPU check.
- `notebooks/colab_run.ipynb` now runs the pytest cell with `check=False`,
  captures stdout/stderr, prints the last 20 lines (including pytest's summary)
  and a clear warning on failure. Training is not stopped by a test failure;
  the section markdown says so.
- `pytest` (the repo's `pytest.ini` supplies `-q`): **199 passed in 86.97 s**
  on CPU; an explicit `pytest -q` run also passed and was double-quiet because
  `-q` is already configured. The notebook JSON parses and the edited pytest
  cell compiles.

## Session 2026-10-29 (sixth session)

### Step 1: Colab GPU test compatibility

- Already complete at session start (commit `379aea8`), so it was not redone.
  `tests/test_train.py` uses a device-aware peak-memory expectation (`0.0` on
  CPU, `> 0` on CUDA) and checks both `run_info.json` and `metrics.csv`; the
  notebook pytest cell warns on failure without blocking training.

### Milestone 5: Multi-scale Upsampling (MSU)

- Added `src/models/msu.py` with `MSULink` and `MSU`. Each link is a 1x1
  `Conv2d` (no bias) that projects the source width, a x4 bilinear upsample, a
  residual addition into the target decoder map, and ReLU after the addition.
  Paper mode has exactly two links: Decoder 4 -> Decoder 2 and Decoder 3 ->
  Decoder 1.
- `PlainDecoder` now accepts `msu` and `msu_mode`. When `msu: false` no MSU
  module is registered (`self.msu is None`), so the decoder state_dict is
  unchanged from Milestone 4 (63 keys, test-pinned). When `msu: true`, the
  D4->D2 link is applied before Decoder 1 is computed; otherwise the D4 link
  would be dead because nothing downstream would read the updated D2. D3->D1 is
  applied after Decoder 1.
- `msu_mode: released` is accepted by config validation but `build_model` and
  `PlainDecoder` raise `NotImplementedError` with a clear message. It was not
  implemented in this milestone.
- Config keys moved to the requested `decoder` section: `decoder.msu` (default
  `false`) and `decoder.msu_mode: paper | released`. This replaces
  `model.msu`/`model.msu_mode`; all repo configs were migrated, and
  `configs/base.yaml` now defaults to `msu: false`. Added
  `configs/exp007_hybrid_msu.yaml` (hybrid fusion, MSU on, paper mode, both
  encoders pretrained).
- `src/train.py` smoke mode now preserves the configured MSU setting instead of
  forcing it off, so the `exp007` smoke run exercises the real MSU path.
  `run_info.json` and `config_resolved.yaml` record `msu` and `msu_mode`.
- `src/utils.py` validates the new `decoder` section; unknown keys still fail.
  `msu_mode: released` passes validation and is rejected by the model builder.
- Optional GroupNorm was not added. BatchNorm remains in `ChannelAlignment`,
  `ConcatFusion` (stages 1-2), every decoder `ConvBlock`, and the decoder head
  norm; cross-attention uses LayerNorm. GroupNorm can be added as a separate
  config change later if wanted.

### Parameter counts (CPU, random weights, no download)

| Component | Parameters |
|---|---:|
| MSU links (512x128 + 320x64 1x1 convs) | 86,016 (0.086 M) |
| `exp005` hybrid, MSU off | 122.677553 M |
| `exp007` hybrid, MSU on | 122.763569 M |

The increase is exactly the two MSU links: 86,016 parameters. These are
parameter counts, not VRAM measurements.

### Tests and smoke runs

- `pytest`: **211 passed in 87.76 s** on CPU (12 new tests versus the previous
  199). New coverage: MSU link shapes; D4->D2 and D3->D1 information flow with
  the main path zeroed; a decoder-level check that the D4 link is not dead;
  exact Milestone-4 decoder state_dict keys with `msu: false`; MSU parameter
  count and model parameter increase; gradients reaching both links and the
  encoder; full-net output shapes for all four fusion presets with MSU on and
  off; `released` rejection; and `exp007` end-to-end smoke.
- `python -m src.train --config configs/exp007_hybrid_msu.yaml --smoke
  --out-dir /tmp/opencode/exp007_msu_smoke` passed on CPU (3 steps, 64 px,
  synthetic data). It reported `model: dual (convnext_tiny.fb_in1k +
  caformer_s18.sail_in1k)  params: 62.84M`, `msu: true`, `msu_mode: paper`.
  Synthetic val Dice `0.1980` is a pipeline check, not a result.
- No pretrained weights were downloaded locally, no full training was run, and
  no VRAM or real-data metric is claimed.

### Open questions (current)

- MSU adds only 86,016 parameters, but `exp007` still needs its own T4 memory
  and epoch-time measurement; no VRAM number is claimed.
- The first real T4 probe remains `exp001`, batch 8, then 4 and 2 with gradient
  accumulation if needed.
- Milestone 5 is complete. Milestones 6 (evaluation, results logging, ablation
  configs) and 7 (final README) remain pending explicit approval.

## Session 2026-10-31 (seventh session)

### Orientation

- Repo was clean, `main` level with `origin/main` at `32b68c3`, no unpushed
  work. `pytest`: 211 passed. Milestone 5 was verified complete at the start
  (MSU links, `decoder.msu`/`decoder.msu_mode`, `exp007`, state_dict
  equivalence with MSU off, gradient flow, MSU-on overfit test and `--smoke`),
  so Part 2A was skipped as agreed.

### End-of-run instrumentation (`src/train.py`)

- Every post-training step now goes through `run_post_step()` and logs
  `[end-of-run] <step>: start` / `done` (or `FAILED`) with the full traceback.
  Steps: prediction grid, `save best.pt`, `save last.pt` (labelled "final
  checkpoint save" on the last epoch), `run_info` rewrite, `wandb.finish`
  cleanup and the final summary.
- Required steps are the checkpoint writes: a failure raises
  `PostTrainingStepError`, main prints an ERROR with the likely cause (free
  space / writability) and returns `EXIT_REQUIRED_STEP_FAILED = 3`.
- Optional steps (prediction grid, `run_info` rewrite, cleanup, summary) print a
  loud `WARNING` with the traceback and leave the exit code unchanged. The
  `finally` handler now catches every exception, not only `OSError`.
- New tests: a failing prediction grid keeps exit code 0 and logs the warning
  plus traceback; a failing checkpoint returns the non-zero code and still
  leaves `metrics.csv`.

### The exp001 exit-code-1 cause: not found, not guessed

- The Colab exp001 run wrote 5 epochs of `metrics.csv` and `run_info.json` but
  exited 1. A local `--smoke` run (matplotlib 3.11.2 present) exits 0, so the
  failure could not be reproduced on the CPU box.
- What the code allowed after a successful last epoch: an exception from the
  last epoch's grid save, `best.pt`/`last.pt` write (Drive full or unmounted),
  or the `finally` `run_info` rewrite (which only caught `OSError`), plus a
  `BrokenPipeError` on the final print. The most likely candidate is a
  checkpoint write to Drive failing after the last `metrics.csv` row, which
  matches "5 epochs recorded, exit 1". The traceback was lost because the
  notebook ran the child with a piped, block-buffered stdout.
- Rather than guess, both sides are now fixed: the child is streamed
  unbuffered and every end-of-run step is logged with its traceback. The next
  Colab failure will name the step and show the exception.

### Colab launcher

- The training cell uses `subprocess.Popen(..., stdout=PIPE,
  stderr=STDOUT, text=True, bufsize=1)` with `-u` and `PYTHONUNBUFFERED=1`,
  prints every line live, prints the exit code and raises only when it is
  non-zero.
- `REF` (commit, tag or branch) is fetched and checked out instead of a plain
  `git pull`; a branch is reset to `origin/REF`, a tag or commit is checked out
  detached. `RUN_SUFFIX` is appended to `run.name`; `GRAD_ACCUM` overrides
  `train.grad_accum_steps`.

### Measured T4 results and exp008

- The "Colab results" table now holds the user's measured values: exp001 5
  epochs, 2.794 GB peak, 23.85 s mean epoch, best val Dice 0.8655 (epoch 4);
  exp003 5 epochs, 1.951 GB, 15.66 s, best 0.8869 (epoch 3, last 0.8845). The
  GPU smoke row is `n/a` (T4 pytest smoke used `convnext_tiny` at 64 px, 703 MB)
  and the real-run row is still empty.
- The two runs used different encoder learning rates (`exp001` 3.0e-5,
  `exp003` 1.0e-4), so they are not a clean architecture comparison. Added
  `configs/exp008_convnext_base_lr1e4.yaml`, identical to `exp001` except
  `run.name` and `train.lr_convnext: 1.0e-4`; `exp001` is untouched.
- README gained a config table covering every experiment file.

### Parameter counts (CPU, random weights, no download, recomputed)

| Config | Parameters |
|---|---:|
| `exp005` hybrid fusion, MSU off | 122.677553 M |
| `exp006` all concat | 127.777841 M |
| `exp007` hybrid fusion, MSU on | 122.763569 M |
| `exp008` ConvNeXt-Base at 1e-4 | 97.221377 M |

### Tests

- `pytest`: **213 passed** (2 new end-of-run tests). `--smoke` runs for
  `exp001` and `exp008` exit 0 and log every end-of-run step. No pretrained
  weights were downloaded, no full training was run, and no VRAM number was
  measured on this box.

### Open questions (current)

- Milestone 6 was approved and implemented in the eighth session; the ablation
  runs themselves are still the user's Colab work.
- The exact Colab exit-1 cause is still unknown; the instrumentation and
  streaming should identify it on the next run.
- exp003's batch size was not recorded; `config_resolved.yaml` from that run
  should confirm it (the table marks it `n/a` until then).
- Whether to run `exp008` (ConvNeXt at 1e-4) before or alongside the
  ablations is an open scheduling question.


## Session 2026-10-31 (eighth session)

### Orientation

Clean tree at `714741e`, `main == origin/main`, `pytest` 213 passed. Milestone 6
was approved at the start of the session, together with the exp008 probe rule
and the final comparison protocol (both recorded in AGENTS.md section 11).

### Milestone 6a: `src/evaluate.py` (`1c89622`)

- `python -m src.evaluate --run_dir <run> [--split test|val]
  [--checkpoint best|last] [--results_csv <path>] [--force]`.
- Rebuilds the model from `config_resolved.yaml` with `pretrained_a/b: false`
  (the checkpoint supplies every weight, so evaluation never downloads), loads
  the checkpoint, and reuses `src.train.evaluate`, so the numbers come from the
  same per-image Dice/IoU code as training.
- Writes `<run>/eval_<split>.json` (mean/std Dice and IoU, n, small-lesion Dice,
  provenance) and `<run>/per_image_<split>.csv`
  (`id, dice, iou, lesion_pixels, small_lesion`).
- Appends one row to the results CSV (default `results/results.csv`): run name,
  split, config, git commit, seed, epochs, params (M), best val Dice, test Dice,
  test IoU, small-lesion Dice, mean epoch time, peak GPU memory. Rows are never
  reordered; a duplicate `(run_name, split)` is refused unless `--force`
  replaces that row in place.
- Test-set safety: prints a `TEST SPLIT:` line and aborts if any test id also
  appears in `train.txt` or `val.txt`.
- The results header gained a `split` column so a val row and a test row of one
  run can coexist. `results/results.csv` is still header-only and committed.
- `tests/test_evaluate.py`: 7 tests on module-scoped smoke runs (files and row,
  metrics equal to `src.train.evaluate` and to `SegMetrics`, duplicate refused
  then `--force` replaces, overlap detected, isolated test run writes a row,
  incompatible header refused without rewriting).

### Milestone 6b: `scripts/aggregate_results.py` (`2055764`)

- Reads `eval_test.json` + `per_image_test.csv` from every run folder under
  `--runs-dir`, groups by config (the `_seed<N>` suffix and `config_file` stem
  are ignored), and writes `aggregate_results.md`/`.csv` into `--out-dir`
  (default: the runs directory).
- Columns: config, number of seeds, mean ± std test Dice and IoU, small-lesion
  Dice, parameters, mean epoch time. One seed gives `n/a`, never `0.0`.
- `--compare A B` adds a paired per-image Dice difference (B - A) averaged over
  shared seeds, with a fixed-seed bootstrap 95% CI, and states what it does and
  does not show. Configs sharing no seed raise a clean error.
- `tests/test_aggregate_results.py`: 6 tests on toy per-image files (grouping,
  hand-computed mean/std, `n/a`, skip training-only folders, seed-suffix
  fallback, reproducible paired comparison, no shared seed).

### Milestone 6c: seeds, notebook and README

- `tests/test_seeds.py` (4 tests): the same `run.seed` reproduces the initial
  weights, the DataLoader shuffle order and the training augmentation, a
  different seed changes all three, and `train.main` calls `set_seed` once with
  `cfg["run"]["seed"]` before building the model.
- `notebooks/colab_run.ipynb`: new `SEED` setting (sets `run.seed` and appends
  `_seed<N>` to the run name), `RESULTS_CSV` on Drive, an evaluate-after-training
  cell (`--split test`, or `val` in `SMOKE` mode because the synthetic test ids
  are a subset of val), and a final aggregate cell.
- `README.md`: "Full ablation" section (six models, `git tag ablation-v1` to
  freeze the commit, one epoch count N from 30-epoch curves, seeds 42/43/44,
  evaluate the test split once per run, aggregate, and where every file goes).
  Colab steps, tests and roadmap updated.

### Tests

- `pytest`: **230 passed** (213 before this session, +7 evaluate, +6 aggregate,
  +4 seeds).
- Manual checks on a local `--smoke` run: val evaluation wrote both files and one
  row; a duplicate row was refused; `--force` replaced it (header + 1 row); a
  test evaluation on the synthetic split aborted with
  "1 test ids also appear in val.txt".
- No pretrained weights were downloaded, no full training was run, and no VRAM
  number was measured in this session.

### Open questions (current)

- The ablation runs have not been launched: N, the T4 batch size and the final
  table are still the user's Colab work.
- Should `results/results.csv` stay header-only in the repo (current choice), or
  should final rows be committed at the end? Committing them would make the
  table reproducible from git alone but mixes data into the repo.
- The paired bootstrap compares test images only; a case-level or
  multi-dataset comparison is not implemented.

## Session 2026-10-31 (ninth session)

### Orientation (no assumptions from earlier sessions)

Tree at `91bf9ac`, `main` **1 commit ahead** of `origin/main` (the untimed
M6d timing-breakdown commit), plus one **uncommitted** file:
`notebooks/colab_run.ipynb` (a `tqdm` epoch bar for the streamed child log).
`git fetch` showed nothing new on the remote. `pytest` with the project venv
(`.venv/bin/python -m pytest`, timm 1.0.30 / torch 2.15.0+cpu): **231 passed**
at orientation. Note: the repo's `pytest.ini` adds `-q`, so a double `-q` hides
the summary line; the exit status is still 0 and the collected count is 231.

Milestone status at orientation matched the checklist: M1-M6 done, M7 partial
(launcher and README section exist, final README pass outstanding). Milestone 5
was verified complete, so no M5 code was written this session.

### Part 2A: Milestone 5 was already complete (no change)

Verified in the tree rather than from notes: `src/models/msu.py` (`MSULink`,
`MSU`), `decoder.msu` / `decoder.msu_mode` in `PlainDecoder` and config
validation, `configs/exp007_hybrid_msu.yaml`, and tests for all five required
properties: link shapes, D4->D2 and D3->D1 information flow (including a
dead-link check), D4->D2 applied *before* Decoder 1, exact Milestone-4 decoder
state_dict keys with `msu: false`, an overfit-one-batch test that builds the
model with `msu=True`, and an `exp007` `--smoke` end-to-end test. `released`
raises `NotImplementedError`.

Parameter counts (CPU, random weights, no download, recomputed this session):
`exp005` 122,677,553 (122.677553 M); `exp006` 127,777,841 (127.777841 M);
`exp007` 122,763,569 (122.763569 M), i.e. +86,016 MSU parameters over
`exp005`. `--smoke` for `exp007` exits 0 with `params: 62.84M`.

### Part 2B: the exit-code mechanism, and a fix

The exp001 Colab run (exit 1 after five complete epochs) still cannot be
reproduced: a local `--smoke` run on a healthy stdout exits 0. Reading the
end-of-run path (`main()` after the loop) narrowed the possibilities to three,
and one of them is now **reproduced and fixed** rather than guessed at:

- **Reproduced:** if stdout's reading end is gone when the end-of-run log line
  is printed, `print` raises `BrokenPipeError` and the process exits non-zero
  (120 here) even though `metrics.csv` and both checkpoints are complete.
  `python -m src.train ... | head -1` shows it. On Colab stdout/stderr go
  through the kernel's output channel, which can close (tab closed, kernel not
  draining, output truncated), and that is consistent with "5 epochs written,
exit 1, cause invisible". It is **not proven** to be the exp001 cause: that run
  used `subprocess.run` with an inherited stdout, so no pipe existed on the
  notebook side.
- **Fixed:** new `_emit()` writes a log line and returns `False` instead of
  raising when the stream is closed; every end-of-run print (step start/done,
  warnings, the failure message, the summary) uses it. New
  `_flush_std_streams()` runs in `__main__` before `SystemExit`, so Python's
  shutdown flush of a dead pipe cannot replace `main()`'s code with 120.
- Unchanged: mid-run prints still fail loudly (a run that cannot log should
  not silently keep training), and a failed checkpoint write still returns
  `EXIT_REQUIRED_STEP_FAILED = 3`.
- Still unknown: whether exp001 died in the loop (e.g. an uncaught exception
  from a dataloader or `append_metrics_row` after the row was written) or at
  interpreter teardown. The `[end-of-run] <step>: start/done/FAILED`
  instrumentation plus live streaming will name it if it recurs.

Tests added (`tests/test_train.py`): `_emit` swallows a closed stream and
returns `False`; `_flush_std_streams` survives a pipe whose flush fails; and an
end-to-end `--smoke` run whose stdout fails on every end-of-run line still
returns exit code 0 with `metrics.csv`, `run_info.json`, `last.pt` and `best.pt`
present.

### Part 2B: Colab launcher

The launcher requirements (live streaming via `subprocess.Popen(...,
stdout=PIPE, stderr=STDOUT)`, printed exit code, raise only when non-zero,
`RUN_SUFFIX`, `GRAD_ACCUM`, `REF` checkout instead of a plain `git pull`) were
already in place from earlier sessions. The uncommitted `tqdm` epoch bar found
in the working tree at session start was reviewed: `bar` advances on the
`epoch N: train ... dice ... iou ...` lines, postfix carries the four metrics,
a `resumed from ... at epoch N` line pre-fills skipped epochs, and
`RAW_STREAM = True` bypasses the bar. One defect was fixed: `tqdm.write`
defaults to stderr, so every log line was going to stderr; it now writes to
`sys.stdout`. All 20 notebook cells compile and both regexes were checked
against a real epoch line and resume line. The bar itself is untested here
(it needs Colab/tqdm), which is a limitation, not a claim.

### Tests

- `.venv/bin/python -m pytest -q`: **234 passed** (231 at orientation + 3 new).
  Every check is CPU-only; no pretrained weights were downloaded and no full
training was run. No VRAM number is claimed from this box.

### Open questions (current)

- The exact exp001 exit-1 cause remains unknown; the next Colab failure will
  name the step and print the traceback.
- The timeline is unchanged: the ablation runs (exp005/exp006/exp007 plus
  exp001/exp003/exp004 baselines, seeds 42/43/44) are still the user's Colab
  work.
- `results/results.csv` stays header-only in the repo.

## Session 2026-10-31 (tenth session)

### Orientation

Clean tree at `0db9499`, `main == origin/main`, **234 passed** with
`.venv/bin/python -m pytest -q`. Both requested items already existed on disk
from earlier commits this date: the per-epoch timing breakdown in `91bf9ac`
(M6d) and the streamed `tqdm` epoch bar in `d321bcd`. This session therefore
verified them from the code instead of re-implementing them, fixed one defect
in the notebook, and produced the measurements and proposals below.

### Task 1: timing breakdown (verified; no new code needed)

`src/train.py` already times train, validation, prediction grid, `best.pt` and
`last.pt` with `time.perf_counter()`, writes `train_time_s`, `val_time_s`,
`grid_time_s`, `best_save_time_s`, `last_save_time_s` at the end of
`METRICS_COLUMNS`, and prints them in the epoch line after the unchanged
`epoch N: train X val Y dice Z iou W peak ...MB (...)` fields.
`epoch_time_s` is the full epoch wall clock, so the 37 s logged versus ~98 s
observed gap is the grid plus both checkpoint writes, which used to be logged
outside the epoch total. Older `metrics.csv` files stay resumable:
`append_metrics_row` accepts a header that is a prefix of the current one, keeps
those columns and writes no new ones (covered by
`test_metrics_csv_accepts_a_legacy_header_and_keeps_its_columns`).

### Task 1: expected checkpoint size for `exp005`

Measured on CPU with random weights (no download, no training), by serialising
the exact payload `save_checkpoint` writes and computing the AdamW state from
the parameter count:

| Part | Size |
|---|---:|
| `exp005` parameters | 122,677,553 |
| weights + metadata (`torch.save` file) | **491.0 MB** (0.46 GiB) |
| AdamW state (2 x fp32 tensors per parameter) | **981.5 MB** (0.91 GiB) |
| **full `last.pt` / `best.pt`** | **~1.47 GB** (1.37 GiB) |
| written per epoch (both checkpoints) | **~2.95 GB** |

`best.pt` is not weights-only: both checkpoints carry model, optimizer,
scheduler, scaler, epoch, best Dice, RNG state and config. 2.95 GB per epoch to
Google Drive is the likely bulk of the missing ~61 s (a full Drive write of a
1.5 GB file typically costs several to tens of seconds, and it happens twice).

### Task 1: proposals to cut the time (NOT implemented)

1. **`last.pt` local, mirrored to Drive every N epochs.** Write `last.pt` to
   `/content/last.pt` each epoch and copy it to the run folder every
   `save_every_n_epochs` (proposed default 5), plus once at the end. Resume
   reads Drive when the local file is missing, so a disconnect loses at most N-1
   epochs instead of the session. Needs one new config key, a "local then copy"
   path in `save_checkpoint` and a resume test.
2. **Weights-only `best.pt`.** ~491 MB instead of 1.47 GB, and the optimizer
   state is useless for `evaluate.py`. Model selection only needs weights + the
   epoch and best-Dice scalars; `last.pt` keeps the full state for resume.
3. **Make the prediction grid optional.** `log_images_every` already gates it;
   setting it to 0 (or a `logging.log_images: false` flag) removes a
   forward pass and a PNG write on Drive every N epochs. If the PNG is the slow
   part, writing it to `/content` and copying at the end would help more.
4. Optional, not requested: copy `metrics.csv` once per epoch only (it is tiny),
   and keep the grid images in `/content` for the session.
   None of these change training semantics, and none is implemented yet.

### Task 1: the `lr_scheduler.step()` warning

Not an ordering bug. `_optimizer_step` runs `scaler.step(optimizer)` →
`scaler.update()` → `optimizer.zero_grad()` → `scheduler.step()`, which is the
required order. PyTorch's warning fires only at `scheduler._step_count == 1`
(torch 2.15: `optim/lr_scheduler.py`, "Detected call ... before
`optimizer.step()`") when `optimizer._opt_called` is still false; that flag is
set by the wrapper around `optimizer.step`, which AMP skips when GradScaler
finds inf/nan. With `init_scale=65536` in fp16 the very first step is commonly
skipped, so the warning is the expected one-time AMP symptom.

A real, smaller issue does exist: `scheduler.step()` is unconditional, so a
**skipped** AMP step still advances the cosine schedule. Reproduced with a
scaler that skips step 1: the LR moved 1e-3 → 9.6194e-4 on the skipped step and
the warning appeared once. The cost is one step out of
`epochs * steps_per_epoch` (~1 in 37,500 at 100 epochs, batch 8), i.e.
negligible for the results, but a fix is a two-line guard around
`scheduler.step()` (`if scaler_stepped`) and was **not** applied here.

### Task 2: notebook

The bar, `RAW_STREAM`, `REF`, `SEED`, `RUN_SUFFIX`, `GRAD_ACCUM`, the exit-code
print and the evaluate/aggregate cells were already merged. One defect was
found and fixed: the `tqdm.write(...)` line had lost its loop indentation, so
the training cell was `IndentationError`-invalid and would not have run on
Colab. The bar fields were also renamed to `train_loss` / `val_loss` /
`val_dice` / `val_iou` so the bar shows the requested names. Verified by
compiling every code cell and by executing the extracted bar block against
synthetic child output (resume line at epoch 1 + epochs 1-2): 3/3 epochs
advanced, postfix `{'train_loss': '0.1000', 'val_loss': '0.1000',
'val_dice': '0.8900', 'val_iou': '0.8000'}`, all five log lines printed above
the bar. Not run on Colab itself.

### Tests

- `.venv/bin/python -m pytest -q`: **234 passed**, no new tests this session
  (the timed columns, legacy header and smoke paths are already covered).
- No pretrained weights downloaded, no training run, no VRAM number claimed.

### Open questions (current)

- Approve or reject the three speed proposals above; none is implemented.
- Should the skipped-step scheduler advance be fixed? It changes the LR curve by
  one step out of tens of thousands.

## Session 2026-10-31 (eleventh session) - Milestone 8, Step 0

Milestone 8 (multi-dataset evaluation) is user-specified work outside AGENTS.md
section 9. Rules for it: do not change training code, model code or the Kvasir
800/100/100 split; never select checkpoints or tune on the external datasets
(selection stays Kvasir val Dice); all numbers are single-seed and provisional.

### Step 0: dataset registry and `DATA_ROOT` resolution (done)

What the raw data actually is (inspected, not assumed). `<repo>/data/raw` is a
symlink to `../../../dataset/Kvasir-SEG/Kvasir-SEG`, and under it sit one folder
per dataset:

| dataset | folder | layout | pairs | expected |
|---|---|---|---|---|
| Kvasir-SEG | `data/raw/kvasir_seg` | `images/*.jpg`, `masks/*.jpg` | 1000/1000 | 1000 |
| CVC-ClinicDB | `data/raw/cvc_clinicdb` | `images/*.png`, `masks/*.png` | 612/612 | 612 |
| CVC-ColonDB | `data/raw/cvc_colondb` | `images/*.png`, `masks/*.png` | 380/380 | 380 |
| ETIS-LaribPolypDB | `data/raw/etis` | absent | - | 196 |
| CVC-300 | `data/raw/cvc_300` | absent | - | 60 |

No count mismatch for the three present datasets. Masks share the image stem
with no `_mask` suffix; external images are `.png` while Kvasir is `.jpg`/`.jpg`,
so pairing is by stem with the extension discovered from disk.

Changes:

- `configs/datasets.yaml` (new): the registry. Five entries with name, folder,
  expected count, `image_subdir`/`mask_subdir`/`mask_suffix`, plus the
  `DATA_ROOT` resolution and stem-matching rules in a header comment.
- `src/data_registry.py` (new): `DatasetSpec` (`raw_dir`, `processed_dir`,
  `image_dir`, `mask_dir`, `is_present`), `load_registry`, `registry_names`,
  `get_spec`, `resolve_data_root` (env `DATA_ROOT` > `data.data_root` >
  `<repo>/data`), `data_root_from_config`, `present_datasets`, `RegistryError`.
  A missing dataset is reported as absent, never as an error.
- `src/utils.py`: `data.dataset` and `data.data_root` defaults; `load_config`
  now calls `_apply_data_registry`, which sets `data.root` to the registry raw
  folder and fills `image_dir`/`mask_dir`/`mask_suffix` from the registry entry.
  A layout key is rewritten only while it still holds its default, so an
  explicit root (synthetic smoke data, a Colab config) is never touched. This
  keeps `src/train.py` unchanged. `validate_config` gained the two new keys.
  Side effect: `run_info.json`'s `data_root` now records the resolved absolute
  raw folder instead of the literal `data/raw`; the provenance test was updated
  to match.
- `tests/test_data_registry.py` (new, 16 tests) and 4 new tests in
  `tests/test_utils.py`; `tests/test_train.py` provenance assertion updated.

Verification: `.venv/bin/python -m pytest -o addopts="" -q` -> **256 passed**
(was 236; +16 registry, +4 config-wiring, and the provenance test now asserts
the resolved root). `data/` is already gitignored (`.gitignore: data/*`, with
`data/splits/` re-included); `git check-ignore data/raw` confirms it. The three
present datasets resolve to their real folders; `etis` and `cvc_300` report
absent.

Left to do for Milestone 8: Step 1 (report + plan), Step 2
(`scripts/prepare_datasets.py` -> `data/processed/<name>/{data.npz,meta.json}`),
Step 3 (`src/evaluate.py` multi-dataset), Step 4 (`aggregate_results.py` matrix),
Step 5 (tests), Step 6 (Colab cells), Step 7 (validation; exp003 Kvasir val
first), `docs/datasets.md` and the results table.

Open questions (current):

- Commit `d5508ca` ("fusion: add the xattn_swap preset and the exp013 config")
  is model code outside the Milestone 8 scope and is on local `main`, one commit
  ahead of `origin/main` (`179aad7`). Keep it, or leave it unpushed until after
  Milestone 8? It was not created by this session's Milestone 8 work.

## Session 2026-10-31 (twelfth session) - Milestone 8, Step 1 + Step 2

### Step 1: how the existing eval plumbing and the Kvasir prep work (report)

Read, not changed: `src/evaluate.py`, `scripts/aggregate_results.py`,
`scripts/build_cache.py`, `src/datasets.py`.

- `src/evaluate.py` CLI: `--run_dir --split {test,val} --checkpoint {best,last}`
  (default `test`/`best`), plus `--results_csv` and `--force`. It reuses
  `src.train.evaluate` for the val loader and metric loop, writes `eval_<split>.json`
  and `per_image_<split>.csv` into the run folder, and appends one row to
  `results/results.csv` keyed on `(run_name, split)`; a duplicate row is refused
  unless `--force`, and a different header is always refused. `test` is guarded
  as the only split that may produce a reported number.
- `scripts/aggregate_results.py` reads only `eval_test.json` + `per_image_test.csv`,
  groups runs by config (stripping a `_seed\d+` suffix), reports mean/std and
  `n/a` when only one seed exists, plus a paired bootstrap comparison.
- Kvasir cache (`scripts/build_cache.py` + `data/cache/kvasir_seg_256.npz`, 143 MB):
  keys `ids` (U25, 1000), `images` (1000, 256, 256, 3) uint8, `masks`
  (1000, 256, 256) uint8 {0, 1}. Masks are binarised at >127 *before* a
  nearest-neighbour resize; images are RGB bilinear. ImageNet mean/std
  normalisation is **not** in the npz, it is applied in `PolypDataset.__getitem__`
  at read time. Splits are the committed 800/100/100 files in `data/splits/`.
- New layout for M8: `<DATA_ROOT>/raw/<folder>` (read-only source) and
  `<DATA_ROOT>/processed/<name>/{data.npz,meta.json}` (what evaluation reads),
  with `<DATA_ROOT>` resolved by `src/data_registry.resolve_data_root`. The
  `processed/<name>/data.npz` format is byte-for-byte the Kvasir cache format,
  so `PolypDataset` reads it with no new code path.

### Step 2: `scripts/prepare_datasets.py` (done)

- `scripts/prepare_datasets.py` (new): `raw/<folder> -> processed/<name>/{data.npz,meta.json}`
  for every registered dataset, in exactly the Kvasir cache format. Extensions
  are discovered from disk (Kvasir `.jpg`/`.jpg`, external `.png`/`.png`), pairs
  match by stem. Loud failures: missing raw folder, unmatched image/mask stem,
  ambiguous stem, empty masks, mask values outside {0, 1}. A count that differs
  from the registry's `expected_count` is a prominent warning recorded in
  `meta.json` (`count_matches_expected`), never silently "fixed". Idempotent:
  an existing `data.npz` at the same size is skipped; a different size is
  refused unless `--force`. `--dataset NAME` must be present (absent -> exit 1),
  no `--dataset` skips absent datasets with a warning (nothing present -> exit 1),
  an unknown name is a usage error (exit 2). Also `--config` (read a run's
  `data.image_size`/`mask_threshold`), `--size`, `--mask-threshold`,
  `--data-root`, `--out-root`, `--limit`, `--quiet`.
- `meta.json` per dataset: name, source, raw/processed paths, image count,
  expected count + match flag, original-resolution summary, image size,
  resample modes, mask threshold, image/mask extensions, mask suffix, npz bytes,
  created timestamp.
- `src/data_registry.py` + `configs/datasets.yaml`: added the `source` field
  (human-readable name, e.g. `Kvasir-SEG`) that lands in `meta.json`.
- `tests/test_prepare_datasets.py` (new, 12 tests, CPU, all in `tmp_path`):
  stem matching across differing extensions, unmatched image reported, missing
  folder reported, npz+meta written, idempotent + `--force`, different-size
  refusal, empty mask fails loudly, count mismatch warns but continues, masks
  binary, `main` skips absent / fails on an explicit absent dataset / usage error
  on an unknown name.

Run on the real data (local, CPU). `python scripts/prepare_datasets.py`:

| dataset | pairs | expected | npz | notes |
|---|---|---|---|---|
| kvasir_seg | 1000 | 1000 | 143.5 MB | 333 distinct resolutions, most common 622x530 |
| cvc_clinicdb | 612 | 612 | 60.0 MB | all 384x288 |
| cvc_colondb | 380 | 380 | 45.2 MB | all 574x500 |
| etis, cvc_300 | absent | - | - | skipped with a warning |

All three npz: `ids` unicode, `images` (N,256,256,3) uint8, `masks` (N,256,256)
uint8 in {0, 1}, zero empty masks. **Strong check:** `data/processed/kvasir_seg/data.npz`
is bit-for-bit identical to the committed `data/cache/kvasir_seg_256.npz`
(ids, images and masks all `np.array_equal`), so the M8 prep reproduces the
Kvasir pipeline exactly. `data/processed/` lives under the gitignored `data/*`.

Verification: `.venv/bin/python -m pytest -o addopts="" -q` -> **268 passed**
(was 256; +12 prep tests).

Left to do for Milestone 8: Step 3 (`src/evaluate.py` multi-dataset: per-(run,
dataset) files/rows, `dataset` column in `results/results.csv`, optional flip
TTA off by default), Step 4 (`aggregate_results.py` config x dataset matrix),
Step 5 (missing-dataset skip vs explicit-request failure, evaluate smoke, val
reproduction within tolerance), Step 6 (`notebooks/colab_eval.ipynb`), Step 7
(validation; exp003 Kvasir val first, then the full table), `docs/datasets.md`
and the evaluation-protocol section.

Open questions (current): unchanged from the eleventh session (commit `d5508ca`
is unpushed model code outside M8).

## Session 2026-10-31 (thirteenth session) - Milestone 8, Step 3

### Step 3: multi-dataset `src/evaluate.py` (done)

`src/evaluate.py` now evaluates one run on one or more datasets and writes one
result per `(run_name, split, dataset, tta)`. The single-dataset interface
(`--run_dir`, `--split`, `--checkpoint`, `--results_csv`, `--force`) is kept; new
flags are `--dataset NAME` (repeatable, default `kvasir_seg`) and `--tta-flip`
(off by default).

- Loaders: the training dataset (`data.dataset`, `kvasir_seg`) is read through
  the committed split files exactly as before, so its `val` Dice reproduces the
  training log (`build_split_loader`). Every other dataset is read zero-shot
  from `<DATA_ROOT>/processed/<name>/data.npz` with all of its images
  (`build_processed_loader`); a missing npz fails loudly and prints the
  `scripts/prepare_datasets.py --dataset NAME` fix. `resolve_datasets` rejects an
  unknown name (exit 2). External datasets are always recorded as `split=test`
  whatever `--split` says.
- Metrics: Dice and IoU per image at the fixed 0.5-threshold on the sigmoid
  output, then averaged (unchanged convention). MAE is added (probability map
  vs. binary ground truth, per image then averaged). Flip TTA is optional: the
  flipped pass is undone and the two maps are averaged in probability space
  before thresholding. Small-lesion Dice is still reported.
- Outputs per dataset, inside the run folder: `eval_<split>_<dataset>.json` and
  `per_image_<split>_<dataset>.csv` (`id, dice, iou, mae, lesion_pixels,
  small_lesion`). A TTA run adds `_ttaflip` so it never overwrites a plain run.
- `results/results.csv` gained the header
  `run_name,split,dataset,tta,config_file,git_commit,seed,epochs,params_m,
  best_val_dice,test_dice,test_iou,test_mae,small_lesion_dice,mean_epoch_time_s,
  peak_gpu_mem_mb`; duplicate key is `(run_name, split, dataset, tta)`.
- `src/datasets.py`: `PolypDataset` now takes exactly one of `split_file=` or
  `ids=` so an external dataset with no split file can be read from its cache.
  Training paths are untouched (`from_config` still passes `split_file=`).
- `tests/test_evaluate.py`: rewritten for the new file names; 5 new tests (TTA
  suffix, TTA and plain rows coexisting, zero-shot external dataset via a
  monkeypatched `DATA_ROOT`, missing-processed-dataset loud failure, unknown
  dataset usage error).

Verification: focused `tests/test_evaluate.py tests/test_datasets.py` -> **29
passed**; full suite `.venv/bin/python -m pytest -o addopts="" -q` -> **273
passed** (was 268; +5 evaluate tests). The checkpoint used for every dataset is
`best.pt`, i.e. the one selected by Kvasir val Dice; external datasets never
select or tune it.

Open questions (current): unchanged from the eleventh session (commit `d5508ca`
is unpushed model code outside M8).

## Session 2026-10-31 (fourteenth session) - Milestone 8, Step 4

### Step 4: `scripts/aggregate_results.py` config x dataset matrix (done)

Rewritten to read every `eval_<split>_<dataset>[_ttaflip].json` + matching
`per_image_...csv` written by `src.evaluate` (the old hardcoded
`eval_test.json`/`per_image_test.csv` names are gone) and to group by config
(seed suffix stripped) *within* each `(dataset, split, tta)` cell.

- `aggregate_results.md` is now the human matrix: a **Dice** table and an
  **IoU** table with rows = configs, columns = datasets, cells = `mean +/- std`
  and `n/a` for a single seed or an unevaluated cell, plus a details table
  (MAE, small-lesion Dice, params, epoch time) and the optional comparison.
  Datasets are listed in registry order.
- `aggregate_results.csv` is the long form (one row per
  `(config, dataset, split, tta)`): `config, dataset, split, tta, n_seeds,
  dice_mean, dice_std, iou_mean, iou_std, mae_mean, small_lesion_dice, params_m,
  mean_epoch_time_s`, so the matrix can be pivoted or joined.
- `--compare A B` now runs on one dataset: `--dataset` picks it, otherwise the
  script auto-selects the preferred `(split, tta)` cell that holds both configs.
  New flags `--dataset`, `--split`, `--tta`.
- New helpers: `parse_eval_name`, `load_run` (per-folder, list),
  `dataset_cells`, `group_cell`, `summarize_runs`, `preferred_rows`,
  `dataset_order`, `resolve_cell`. `bootstrap_ci`/`paired_comparison` kept.
- `tests/test_aggregate_results.py`: rewritten for the new names; 11 tests
  (grouping + `n/a`, dataset-column matrix, stem parsing, warning on a missing
  per-image CSV, registry-ordered columns, auto cell for `--compare`, plus the
  kept bootstrap/paired tests).

Verification: `tests/test_aggregate_results.py` -> **11 passed**; full suite
`.venv/bin/python -m pytest -o addopts="" -q` -> **278 passed** (was 273; +5).

Left to do for Milestone 8: Step 5 (missing-dataset skip vs explicit-request
failure, evaluate smoke, val reproduction within tolerance), Step 6
(`notebooks/colab_eval.ipynb`), Step 7 (validation; exp003 Kvasir val first,
then the full table) plus `docs/datasets.md`, the evaluation-protocol section
and the README pass (the README still shows the old `eval_test.json` names).

Open questions (current): unchanged from the eleventh session (commit `d5508ca`
is unpushed model code outside M8).

## Session 2026-10-31 (fifteenth session) - Milestone 8, Step 5

### Step 5: tests for the multi-dataset path (done)

Extended `tests/test_evaluate.py` with two tests that close out Step 5 and tie
Steps 3 and 4 together on a real (tiny) smoke checkpoint:

- `test_processed_loader_shapes_and_binary_masks`: the zero-shot loader
  (`build_processed_loader`) over a prepared `processed/<name>/data.npz` yields
  `(B, 3, H, W)` images and `(B, 1, H, W)` masks whose values are a subset of
  `{0.0, 1.0}`. Ids are collected across every batch (the smoke config uses
  `train.batch_size: 2`, so the first batch holds only two of the three images),
  then checked against the prepared order.
- `test_multi_dataset_evaluation_feeds_the_aggregate_matrix`: `evaluate_mod.main`
  on `kvasir_seg` + `cvc_clinicdb` + `cvc_colondb` writes three results rows (all
  `split=test`) and the matching `eval_test_<dataset>.json` files, and
  `scripts.aggregate_results.main` over the same runs folder produces the matrix
  header `| Config | kvasir_seg | cvc_clinicdb | cvc_colondb | n seeds |` with
  CSV rows covering all three datasets for one config.

The rest of the Step 5 checklist was already covered by earlier work:

- Adapters: shapes, mask values in {0, 1}, stem matching across differing
  extensions, unmatched-pair and count checks: `tests/test_prepare_datasets.py`
  (12 tests, Step 2).
- Missing dataset skipped with a warning, failing only when requested by name:
  `tests/test_prepare_datasets.py::test_main_skips_absent_datasets_and_prepares_present_ones`
  and `::test_main_fails_when_an_explicit_dataset_is_absent` for prep;
  `tests/test_evaluate.py::test_missing_processed_dataset_fails_loudly` (explicit
  `--dataset` with no npz -> exit 1, message names
  `scripts/prepare_datasets.py --dataset NAME`) and
  `::test_unknown_dataset_is_a_usage_error` (exit 2) for evaluation.
- Evaluate smoke and val reproduction within tolerance:
  `tests/test_evaluate.py::test_evaluate_metrics_match_the_training_function`
  compares `src.evaluate` output against `src.train.evaluate` and `SegMetrics`.

Verification: `tests/test_evaluate.py tests/test_aggregate_results.py` -> **25
passed**; full suite `.venv/bin/python -m pytest -o addopts="" -q` -> **280
passed** (was 278; +2). CPU-only, no pretrained weights downloaded, no full
training run.

Left to do for Milestone 8: Step 6 (`notebooks/colab_eval.ipynb`), Step 7
(validation; exp003 Kvasir val first, then the full table) plus `docs/datasets.md`,
the evaluation-protocol section and the README pass (the README still shows the
old `eval_test.json` names).

Open questions (current): unchanged from the eleventh session (commit `d5508ca`
is unpushed model code outside M8).

## Session 2026-10-31 (sixteenth session) - Milestone 8, Step 6

### Step 6: `notebooks/colab_eval.ipynb` (done)

New launcher for the multi-dataset flow, separate from `colab_run.ipynb`
(training). 17 cells, all code cells compile. It is safe to rerun:

- Settings: `DRIVE_DATA_ROOT` (raw + processed on Drive), `LOCAL_DATA_ROOT`
  (per-session copy on `/content`), `RUNS_DIR` / `RESULTS_CSV` on Drive, `REF`,
  `PREP_SIZE` (256), `RUN_PREP`, `EVAL_SPLIT` (`test`), `FORCE` (True),
  `TTA_FLIP` (False), `ONLY_RUNS` (None = every non-smoke run with `best.pt`).
- Mounts Drive, clones/checks out the repo at `REF`, installs only missing
  packages, verifies the GPU.
- Prep: `scripts/prepare_datasets.py --size 256 --data-root <Drive>` builds every
  dataset present under `<DRIVE_DATA_ROOT>/raw/<folder>` (idempotent; absent
  datasets skipped with a warning).
- Copies `<DRIVE_DATA_ROOT>/processed` to `/content/polyp-data/processed` and
  sets `DATA_ROOT` so evaluation reads locally.
- Writes a per-run **session config** that changes only data paths: `split_dir`
  to the committed splits, `cache_path` to the local prepared Kvasir npz. This is
  needed because a run's `config_resolved.yaml` records the training-session
  `/content` paths, which are gone in a fresh session.
- Evaluates each run on `kvasir_seg` first (split `EVAL_SPLIT`) plus every
  prepared external dataset (recorded as `test`), with `--checkpoint best` and
  `--results_csv` on Drive; `--force` replaces its own row on a rerun.
- Aggregates with `scripts/aggregate_results.py --split <EVAL_SPLIT>` and prints
  `aggregate_results.md`.
- Last cell documents where checkpoints (`best.pt`, `last.pt`), eval json /
  per-image csv, `results.csv` and the aggregate files live on Drive.

### Supporting change: `src/evaluate.py --config` (new flag)

`src/evaluate.py` gained `--config PATH` to read a different resolved config
instead of the run's own `config_resolved.yaml`, for exactly the fresh-session
case above. It is data-path oriented; the architecture stays the run's own
because the session config is a copy of it. The payload records the override in
a new `config_override` field (the results CSV header is unchanged; `config_file`
still points at the training config). `load_run_config(run_dir, override=...)`
raises a clear `EvaluateError` when the override file is missing (exit 2).

Tests: `tests/test_evaluate.py::test_config_override_is_read_and_recorded`
(override is read and recorded; a missing override is a usage error). Focused
`tests/test_evaluate.py` -> **15 passed**.

Verification: full suite `.venv/bin/python -m pytest -o addopts="" -q` -> **281
passed** (was 280; +1). Notebook JSON parses and all code cells compile (checked
with `ast.parse`). Not run on Colab. CPU-only, no pretrained weights downloaded,
no training run.

Left to do for Milestone 8: Step 7 (local validation; exp003 Kvasir val first,
then the full table) plus `docs/datasets.md`, the evaluation-protocol section and
the README pass (the README still shows the old `eval_test.json` names).

Open questions (current): unchanged from the eleventh session (commit `d5508ca`
is unpushed model code outside M8).

## Session 2026-10-31 (seventeenth session) - Milestone 8, Step 7 (docs)

### `docs/datasets.md` (new)

Reference document for the data side of Milestone 8, six sections: (1) the data
tree and the `DATA_ROOT` resolution order (env -> `data.data_root` -> `<repo>/data`),
with the `data/*` gitignore rule and the `data/splits/` exception; (2) the
registry fields and the five datasets (kvasir_seg / cvc_clinicdb / cvc_colondb
present, etis / cvc_300 absent-unverified), stem matching; (3) the preparation
step, the `data.npz` / `meta.json` format, idempotence, the loud-failure list and
the cache sizes (143.5 / 60.0 / 45.2 MB); (4) zero-shot `src/evaluate.py`
including the new `--config`; (5) `scripts/aggregate_results.py`; (6) the
limitations (single-seed provisional, missing datasets, image-level leakage,
zero-shot external use).

### README pass to the M8 names (done)

- Status header: Milestones 1-6 complete, Milestone 8 complete, Milestone 7
  partial (was "Milestones 1-5 of 7 complete; Milestone 6 pending").
- Layout: added `configs/datasets.yaml`, `scripts/prepare_datasets.py`,
  `scripts/aggregate_results.py`, `src/data_registry.py`, `docs/datasets.md` and
  `notebooks/colab_eval.ipynb`; `results/results.csv` now says
  `(run, split, dataset, tta)`.
- Full ablation step 4 and the results-file table: `eval_test.json` /
  `per_image_test.csv` -> `eval_<split>_<dataset>[_ttaflip].json` /
  `per_image_<split>_<dataset>[_ttaflip].csv`; one row per
  `(run, split, dataset, tta)`; the aggregate row now reads config x dataset.
- New **Evaluation protocol** section: checkpoint = Kvasir val Dice (`best.pt`),
  the two dataset kinds (Kvasir via committed splits, external zero-shot from
  `processed/`), the per-image Dice/IoU/MAE rules, TTA off by default, the output
  files and the results-row key, the aggregation matrix, and the
  single-seed-provisional caveat.
- Roadmap updated to match.

Docs-only change: no code path changed. Verification: full suite
`.venv/bin/python -m pytest -o addopts="" -q` -> **281 passed** (unchanged;
proves the docs edits broke nothing).

Left to do for Milestone 8: Step 7's real-number validation only. It needs the
Colab checkpoints (there are no real checkpoints on the CPU box, only the smoke
run): first evaluate exp003 on Kvasir val and compare to the ~0.91 training-log
value, stopping to diagnose on a mismatch; then evaluate exp003, exp005, exp006
and exp007 on every present dataset and report the full config x dataset table,
labelled single-seed and provisional. `notebooks/colab_eval.ipynb` runs exactly
that flow.

Open questions (current): unchanged (commit `d5508ca` is unpushed model code
outside M8).

