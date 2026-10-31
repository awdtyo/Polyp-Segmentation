# AGENTS.md

Project: dual-encoder polyp segmentation (ConvNeXt + Meta-Polyp encoder, per-stage fusion, U-Net decoder with multi-scale upsampling).

Read this whole file before writing code. If anything here conflicts with a request in chat, stop and ask.
At session start, read PROGRESS.md; at session end, update it.

## 1. Goal

Build, train and evaluate a 2D segmentation model that outputs a binary polyp mask from an input image.

- Research context: internship project on polyp segmentation using computer vision. Results go into a paper-style write-up, so every number must be reproducible from a config, a commit hash and a fixed split.
- Inspiration: NeXtSwin-X (dual-branch ConvNeXt + Swin with cross-attention fusion, classification) and Meta-Polyp (MetaFormer encoder + U-Net decoder with Multi-scale Upsampling, segmentation). Our model replaces Swin with the Meta-Polyp encoder and adapts the cross-attention fusion from pooled vectors to feature maps.
- Dataset: TODO (the user will fill in: name, number of images, 2D slices or 3D volumes, mask format, folder layout). Do not guess. Write the dataset code against the interface in section 6 and ask the user for the missing details.

## 2. Environment and hard rules

- Dev machine: Linux, CPU only, 16 GB RAM, no CUDA. Training runs on Google Colab free tier (T4 16 GB, about 12 GB RAM) and is launched by the user, not by you.
- NEVER run full training locally. You may only run `--smoke` runs (tiny subset, tiny image size, a few steps, CPU) and `pytest`.
- NEVER download datasets or pretrained weights without telling the user first. Pretrained weights are fetched through timm on the user's machine or Colab.
- NEVER commit data, checkpoints, wandb folders or `.env` files. Keep `.gitignore` up to date.
- NEVER invent results, metrics or citations. If you did not run it, say so.
- NEVER change the architecture in section 3 silently. Propose the change, explain why, and wait for approval.
- The test split is used only by `evaluate.py --split test`. Training and model selection use train and val only.
- Python 3.10+, PyTorch, timm, albumentations (or torchvision transforms), PyYAML, pytest, wandb (optional, must work offline or disabled). Do not use Ultralytics or OpenVINO. Do not reinstall torch in Colab instructions.
- Use type hints, small functions, and docstrings that state tensor shapes. No hardcoded absolute paths: all paths come from the config.
- One commit per milestone (section 9) with a clear message. Do not squash unrelated work into one commit.

## 3. Architecture spec

Input: image tensor `(B, 3, H, W)`. If the source is single-channel, repeat it to 3 channels in the dataset. H and W must be divisible by 32. Default `H = W = 256`, configurable.

Normalization: ImageNet mean and std for both branches (one shared preprocessing), unless a branch's pretrained config in timm says otherwise. Log which one was used.

### 3.1 Encoders (both return 4 feature maps at strides 4, 8, 16, 32)

- Branch A, ConvNeXt: `timm.create_model("convnext_base", pretrained=cfg.pretrained_a, features_only=True, out_indices=(0, 1, 2, 3))`. Channels `[128, 256, 512, 1024]`. This branch is pretrained (ImageNet). Config key allows `convnext_small` and `convnext_tiny` as fallbacks if memory is tight.
- Branch B, Meta-Polyp encoder: 4 stages with channels `[64, 128, 320, 512]`, stages 1 and 2 use ConvFormer blocks (separable-conv token mixer), stages 3 and 4 use Transformer blocks (self-attention token mixer). This matches the MetaFormer-baselines CAFormer layout (likely `caformer_s18` or `caformer_s36` in timm). FIRST TASK: read the Meta-Polyp repo (https://github.com/huyquoctrinh/MetaPolyp-CBMS2023) and report which backbone and variant it uses before choosing. Pretrained or from scratch is a config flag `pretrained_b`; do not assume.
- Both encoders are wrapped behind one interface: `forward(x) -> list[Tensor]` of 4 maps, and each exposes `.out_channels`.

### 3.2 Channel alignment

For each stage `i`, apply a 1x1 conv + norm to each branch's map so both have `C_i` channels. Default `C = [64, 128, 320, 512]` (Meta-Polyp widths). Configurable.

### 3.3 Per-stage fusion (`fusion.py`)

- Stages 1 and 2 (strides 4, 8): concatenation fusion. Steps: align channels, concatenate on channel axis (2*C_i), conv 3x3 + norm + activation back to C_i.
- Stages 3 and 4 (strides 16, 32): cross-attention fusion. Flatten maps to tokens `(B, H*W, C_i)`. Query from ConvNeXt tokens, key and value from Meta-Polyp tokens. `nn.MultiheadAttention` with 8 heads, `batch_first=True`. Output added back to the query tokens (residual), then LayerNorm, then reshape to `(B, C_i, H, W)`. Optional 2D sinusoidal positional encoding behind a config flag, default off.
- Fusion type per stage is a config list so ablations need no code changes, for example `fusion: [concat, concat, xattn, xattn]`. Also support `all_concat`, `single_a` (ConvNeXt only) and `single_b` (Meta-Polyp only) for baselines. In single-branch modes, skip the other encoder entirely.

### 3.4 Decoder with Multi-scale Upsampling (`decoder.py`, `msu.py`)

Let `F1..F4` be the fused maps (strides 4, 8, 16, 32).

- Decoder 4 (bottleneck): conv block on `F4`.
- Decoders 3, 2, 1: upsample the previous decoder output 2x with `ConvTranspose2d`, concatenate the fused skip `F_i`, then a conv block (conv 3x3 + norm + activation, twice).
- Multi-scale Upsampling (MSU), configurable on/off: the output of Decoder 4 is upsampled 4x (conv then upsample) and added into Decoder 2's output; the output of Decoder 3 is upsampled 4x the same way and added into Decoder 1's output. After each addition apply the activation. Use a 1x1 conv inside the MSU to match channels before the addition. This follows Meta-Polyp Eq. 6 and 7: `X_decoded = Conv(Up(X))`, `X' = act(X' + X_decoded)`.
- Head: upsample the Decoder 1 output to full resolution (x4), 1x1 conv to 1 channel. The model returns logits. Apply sigmoid only in the loss and in inference.

## 4. Training spec (`train.py`)

- Optimizer: AdamW with differential learning rates in parameter groups: ConvNeXt backbone low (start 3e-5), Meta-Polyp encoder (start 1e-4 if pretrained, 3e-4 if from scratch), all new modules (alignment, fusion, decoder, MSU, head) higher (start 3e-4). Log each group's LR. Cosine annealing scheduler, optional warmup.
- Loss: BCE with logits + Dice by default. Also support Jaccard loss (Meta-Polyp used Jaccard with a smoothing factor). Loss choice is a config key.
- Mixed precision: `torch.autocast` fp16 with `GradScaler` on CUDA. Disabled on CPU. `channels_last` optional.
- Gradient accumulation and gradient clipping as config keys.
- Metrics (`metrics.py`): Dice and IoU on thresholded predictions (threshold 0.5), computed per image then averaged. Unit-test them against hand-computed examples.
- Seeds: set Python, NumPy and torch seeds from the config. Log the seed.
- Checkpointing and resume (critical, Colab disconnects): every epoch save `last.pt` containing model, optimizer, scheduler, scaler, epoch, best metric and RNG state to `cfg.out_dir/<run_name>/`. Save `best.pt` when val Dice improves. On start, if `last.pt` exists, resume automatically. Keep only `last.pt` and `best.pt`.
- Dataloader: `num_workers` from config (Colab: 2), `pin_memory=True`. Data is read from a local path (copied from Drive by the user), never directly from Drive.

## 5. Experiment tracking

- One YAML per experiment in `configs/`, named like `exp003_concat12_xattn34_msu.yaml`. All hyperparameters live there.
- At run start, write `config_resolved.yaml` and `run_info.json` (git commit hash, git dirty flag, torch and timm versions, GPU name, split file name, seed) into the run folder.
- Log per epoch: train loss, val loss, val Dice, val IoU, LR per group, epoch time. Log a small grid of val images with prediction and ground-truth masks every N epochs.
- W&B is optional: `logging.wandb: true/false`. Always also write `metrics.csv` in the run folder so nothing depends on a web service.
- After a run finishes, `evaluate.py` appends one row to `results/results.csv`: run name, commit, config file, params (M), best val Dice, test Dice, test IoU, epochs, train time. Never overwrite existing rows.

## 6. Data interface (`datasets.py`)

- `PolypDataset(split_file, root, transform, image_size)` returns a dict: `image` float tensor `(3, H, W)`, `mask` float tensor `(1, H, W)` with values 0 or 1, `id` string.
- Splits are text files in `data/splits/{train,val,test}.txt`, generated once by `scripts/make_splits.py` with a fixed seed and then committed. Splitting must be at patient or case level if the dataset has patient IDs (ask the user). Never split randomly per slice if slices from the same case could leak across splits.
- Augmentations (train only): flips, small rotation, scale, brightness and contrast. Same geometric transform applied to image and mask. Val and test: resize and normalize only.
- Dataset loading code must be tested on a tiny synthetic folder created in `tests/`.

## 7. Repo layout

```
AGENTS.md
README.md
requirements.txt
configs/
data/splits/
scripts/make_splits.py
src/
  datasets.py  metrics.py  losses.py  utils.py
  models/
    convnext_enc.py  metapolyp_enc.py  fusion.py  msu.py  decoder.py  net.py
  train.py  evaluate.py
tests/
results/results.csv
notebooks/colab_run.ipynb      # thin launcher only: mount Drive, pull repo, copy data, call train.py
```

## 8. Testing rules

Every module needs tests that run on CPU in seconds:

- Shape tests: each encoder returns 4 maps with the expected channels and strides for a `(2, 3, 256, 256)` input; each fusion mode returns `(B, C_i, H_i, W_i)`; the full net returns `(B, 1, H, W)` for every fusion config and with MSU on and off.
- Overfit-one-batch test: the full net must drive the loss on a single synthetic batch clearly down in about 50 steps on CPU. If it does not, there is a bug.
- Metric tests, resume test (save, reload, continue gives the same next-step loss), and a `--smoke` run of `train.py` end to end.
- Run `pytest -q` before every commit. Report failures honestly.

## 9. Milestones (stop after each and report)

1. Skeleton: repo layout, `requirements.txt`, `.gitignore`, config loading, dataset class with synthetic-data test, metrics, losses, tests passing.
2. Baseline: ConvNeXt encoder + plain decoder (no fusion, no MSU), `train.py` with checkpoint and resume, `--smoke` run passes.
3. Meta-Polyp encoder alone with the same decoder.
4. Fusion: concat at stages 1 and 2, cross-attention at stages 3 and 4, plus the `all_concat` variant.
5. MSU links on and off.
6. `evaluate.py`, results logging, and ablation configs for: A only, B only, all concat, hybrid fusion, hybrid fusion + MSU, hybrid fusion without MSU.
7. `notebooks/colab_run.ipynb` launcher and a README section explaining how to run on Colab (Drive mount, GitHub token from Colab Secrets, copy dataset zip to `/content`, resume behavior).

For each milestone, the report must contain: what changed, files touched, test output summary, parameter count of the model if relevant, and open questions for the user.

## 10. Known risks to keep in mind

- Memory on T4: ConvNeXt-Base plus a second encoder plus decoder at 256x256 should be tested with a CPU parameter count and a note on expected VRAM; the user will confirm on Colab. Offer `convnext_small` as a fallback config rather than changing the default.
- Cross-attention on stage 1 or 2 tokens is expensive: at 256x256 input, stage 1 has 64x64 = 4096 tokens and stage 2 has 32x32 = 1024 tokens, versus 256 and 64 tokens at stages 3 and 4. That is why stages 1 and 2 use concatenation.
- Meta-Polyp reports weaker results on small polyps. Keep per-image Dice available so small-lesion performance can be analyzed later.
- Both encoders are pretrained only if the config says so. Pretrained and from-scratch runs are not comparable without saying so.

## 11. Decisions log

Decisions taken after sections 1-10 were written. An entry here overrides an
earlier section. Nothing below changes without a new dated entry. Each entry says
what was decided and what it replaced.

- **2026-10-28 - Image size and encoder size.** `image_size: 256` stays the
  default. ConvNeXt-Base stays the default encoder; `convnext_small` ships as a
  separate fallback config, never as a silent change to the default.
- **2026-10-28 - Pretrained tags are pinned.** Branch A is
  `convnext_base.fb_in1k`, Branch B is `caformer_s18.sail_in1k`. Pinning the tag
  means the exact weights are named in the config, not implied by a library
  default. `caformer_s18` is the timm match for the Meta-Polyp encoder (depths
  `[3, 3, 9, 3]`, channels `[64, 128, 320, 512]`, ConvFormer stages 1-2,
  Transformer stages 3-4); `caformer_s36` is not a match.
- **2026-10-28 - MSU follows the paper, not the released code.** Decoder 3
  output upsampled x4 into Decoder 1 output (paper Eq. 6/7 and section III-E),
  not the released repo's second link (F2-level decoder output upsampled x8, see
  the Step 1 inspection report). New config key `model.msu_mode: paper | released`
  exists so the released variant can be ablated later without code changes.
  Only `paper` is implemented; `released` must raise `NotImplementedError`.
- **2026-10-28 - The Convformer skip block is not implemented.** The released
  repo runs a Convformer block on every skip before merging; we keep the plain
  decoder's concat skip. `model.skip_convformer: false` is reserved as a config
  flag, and setting it to `true` must raise `NotImplementedError` rather than
  quietly doing nothing or quietly adding an unapproved module.
- **2026-10-28 - No TensorFlow -> PyTorch weight remap.** The Meta-Polyp `.h5`
  checkpoint is not converted. Both branches use timm ImageNet weights, and a
  run's pretrained flags are recorded in `run_info.json`.
- **2026-10-28 - Splits are case level, 80/10/10, fixed seed (superseded
  2026-10-29).** `data.split_ratios` and `data.split_seed` record it.
  `scripts/make_splits.py --group-regex` is the only supported way to build
  splits when the data has case or patient ids; file-level random splits are not
  acceptable. Kvasir-SEG has no case ids, so the 2026-10-29 entry below replaces
  this rule for this dataset.
- **2026-10-28 - Augmentation and logging.** albumentations is installed and used
  for training augmentation only (val/test: resize + normalise). `metrics.csv` is
  always written and W&B stays optional, default off, so every number survives
  without a web service.
- **2026-10-29 - Dataset is Kvasir-SEG.** 1000 2D RGB colonoscopy images with
  1000 matching masks, used for both training and evaluation. Verified layout on
  2026-10-29: `Kvasir-SEG/Kvasir-SEG/images/<id>.jpg` and
  `Kvasir-SEG/Kvasir-SEG/masks/<id>.jpg`, identical sorted stems (1000/1000),
  image sizes variable (width 332-1920, height 352-1072; the most common is
  622x530). Masks are grayscale JPEG, not PNG and not strictly binary (all 1000
  masks contain values other than 0/255; 19 grey levels overall), so
  `data.mask_ext` is `.jpg` and masks are binarised with
  `data.mask_threshold: 127` **before** resizing. This closes the section 1
  dataset TODO and replaces the PENDING dataset entry.
- **2026-10-29 - Splits are image level, 800/100/100, seed 42.** Generated once
  by `scripts/make_splits.py` from sorted image stems with no `--group-regex`
  and committed in `data/splits/`. Kvasir-SEG has no patient or case ids, so
  leakage between splits cannot be ruled out; this limitation is stated in the
  README and must be repeated in the paper write-up. The upstream 880/120
  `train.txt`/`val.txt` files are not used. This replaces the case-level
  80/10/10 entry above.
- **2026-10-29 - Data cache.** `scripts/build_cache.py` resizes every image and
  mask once to 256x256 and stores one `.npz` (images uint8, masks uint8 0/1)
  outside git under `data/cache/`. `PolypDataset` reads the cache when
  `data.cache_path` is set; the cache is an I/O optimisation only and never
  changes the split or the labels.
- **2026-10-29 - Final results use three seeds.** Every model in the final
  comparison is trained with seeds 42, 43 and 44 on the same committed split;
  tables report mean and standard deviation, never a single best run.
- **2026-10-29 - T4 memory policy.** ConvNeXt-Base remains the default at
  `image_size: 256`. If a real T4 run runs out of memory, first reduce
  `train.batch_size` and raise `train.grad_accum_steps` to keep the effective
  batch size. Switch to `configs/exp002_convnext_small_baseline.yaml` only as a
  last resort. The default encoder is never changed silently.
- **2026-10-29 - Branch B is pretrained.** Branch B is
  `caformer_s18.sail_in1k` with `pretrained_b: true` for the main runs. A
  from-scratch run (`pretrained_b: false`,
  `configs/exp004_caformer_s18_scratch.yaml`) is an extra baseline only, never
  the default.
- **2026-10-29 - Cache lifecycle on Colab.** The npz cache is rebuilt at the
  start of every Colab session from the raw dataset zip on Drive with
  `scripts/build_cache.py`. The cache file lives outside git and is never
  committed. Its SHA-256 is written to `run_info.json`, so a run can be tied to
  the exact cache it read.
- **2026-10-29 - Split policy for development and final comparison.** All
  development and ablations use the committed image-level 800/100/100 split
  (seed 42). The official upstream 880/120 lists shipped with the Kaggle
  download (`train.txt` and `val.txt`) are NOT used yet; they may be added later
  as a separate config for a final comparison with published numbers. This is a
  policy clarification, not a replacement of the 2026-10-29 split entry.
- **2026-10-29 - Mask threshold stays >127.** Masks are binarised with
  `pixel > 127` before any resizing. The cache and future augmentation work do
  not change this rule.
- **2026-10-29 - Final results use three seeds (confirmed).** This confirms the
  earlier 2026-10-29 entry: seeds 42, 43 and 44, mean and standard deviation in
  the final tables. No single-run best number is reported as the result.
- **2026-10-29 - Milestones 4-7 need explicit approval.** Fusion (concat at
  stages 1-2, cross-attention at stages 3-4, `all_concat`; Milestone 4), MSU
  links (`msu_mode: paper | released`; Milestone 5), `evaluate.py` plus ablation
  configs, `results.csv` appending and multi-seed runs (Milestone 6), and the
  final README (Milestone 7) must not start without the user's approval.
- **2026-10-29 - First T4 probe.** The first memory/time probe uses `exp001`
  (ConvNeXt-Base) with `train.batch_size: 8`. If the T4 runs out of memory,
  fall back to batch size 4 and then 2, raising `train.grad_accum_steps` to keep
  the effective batch size. Do not switch to `exp002` before those fallbacks
  are measured. No VRAM number is claimed until this probe is run.
- **2026-10-29 - Pretrained downloads are Colab-only.** Downloading the pinned
  timm ImageNet weights is allowed on Colab only. Local tests and smoke runs
  always use `pretrained: false`; nothing is downloaded on the CPU dev box.
- **2026-10-29 - Upstream 880/120 comparison is deferred.** The comparison
  against the official upstream `train.txt`/`val.txt` lists is deferred, not
  cancelled. Development and ablations continue on the committed 800/100/100
  split (seed 42).
- **2026-10-29 - Milestone 4 approved; 5-7 pending.** Dual-encoder fusion is
  approved and is the only new architecture work in this session. MSU
  (Milestone 5), `evaluate.py`/ablation configs (Milestone 6) and the final
  README (Milestone 7) remain pending explicit approval.
- **2026-10-29 - Milestone 5 approved; MSU implemented; config moved under
  `decoder`.** MSU lives in `src/models/msu.py`. The canonical keys are now
  `decoder.msu` (default `false`) and `decoder.msu_mode: paper | released`,
  replacing `model.msu` / `model.msu_mode` from the 2026-10-28 entry. Paper
  mode projects with a 1x1 conv, upsamples x4, adds D4 into D2 before D1 is
  decoded, adds D3 into D1, and applies ReLU after each addition. `released` is
  accepted by config validation but raises `NotImplementedError`.
  `decoder.msu: false` keeps the exact Milestone 4 state_dict keys. The optional
  `norm: batch | group` switch was not implemented; BatchNorm remains in channel
  alignment, concat fusion, the decoder conv blocks and the head norm.
  Milestones 6-7 remain pending explicit approval.
- **2026-10-31 - Milestone 6 approved.** `src/evaluate.py`,
  `scripts/aggregate_results.py`, the seed/Colab wiring, the evaluation and
  aggregation tests, and the README ablation section are approved. Milestone 7
  (the final README pass) stays partial and separate.
- **2026-10-31 - exp008 is a 5-epoch probe that may replace exp001.** exp008
  (`configs/exp008_convnext_base_lr1e4.yaml`, ConvNeXt-Base at
  `train.lr_convnext: 1.0e-4`) is run as a 5-epoch probe on the committed split.
  If it clearly beats exp001 on best val Dice, exp008 replaces exp001 as the
  ConvNeXt baseline in the final ablation table; exp001 is then kept only as the
  3e-5 reference. If it does not clearly beat exp001, exp001 stays the baseline.
  "Clearly beats" means a gain larger than the seed-to-seed spread seen in the
  probes; no numeric threshold is claimed before those numbers exist.
- **2026-10-31 - Final comparison protocol.** Every final run uses one epoch
  count N, chosen once from the 30-epoch validation curves (not per config),
  three seeds (42, 43, 44), the same committed split, the same batch size, the
  same image size, and a frozen commit (a git tag such as `ablation-v1`). The
  test split is evaluated once at the end with
  `python -m src.evaluate --run_dir <run> --split test` and is never used for
  model selection. `scripts/aggregate_results.py` groups the seeds per config and
  reports mean and std; a single run is never reported as the result.
- **2026-10-31 - Epoch timing breakdown.** `metrics.csv` gains
  `train_time_s`, `val_time_s`, `grid_time_s`, `best_save_time_s` and
  `last_save_time_s`, appended after `epoch_time_s` so old files stay readable,
  and the epoch log line prints them. `epoch_time_s` is now the full epoch wall
  clock; it used to stop before the prediction grid and both checkpoint writes,
  which is why a log line could say 37 s while the epoch took about 98 s on
  Colab. `append_metrics_row` accepts an older header that is a prefix of the
  current one and writes rows with that file's columns, so a resumed run keeps
  its existing `metrics.csv`.
