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
