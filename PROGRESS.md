# PROGRESS.md

Living status file. AGENTS.md is the specification; this file records what is
actually done, what is in flight, and what is blocked. Update it at the end of
every session.

Last updated: 2026-10-29.

## Milestone checklist

- [x] 1. Skeleton: repo layout, `requirements.txt`, `.gitignore`, config
      loading, dataset class with synthetic-data test, metrics, losses, tests
      passing.
- [ ] 2. Baseline: ConvNeXt encoder + plain decoder (no fusion, no MSU),
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

- Read AGENTS.md and PROGRESS.md, then oriented the repo: one commit exists
  (Milestone 1), all 143 tests passed on the CPU venv.
- Inspected `/home/aditya/Desktop/polyp/dataset` without assumptions. Found
  `Kvasir-SEG/Kvasir-SEG/{images,masks,annotated_images,bbox}` plus 1000 CSVs
  and a PDF; 1000 images and 1000 masks with identical stems; every image RGB,
  every mask grayscale JPEG; variable sizes around 570-626 x 528-547. Confirmed
  the expected `images/<id>.jpg` / `masks/<id>.jpg` pairing, with the caveat
  that JPEG masks are not strictly binary.
- Wrote the dataset, split, cache and results decisions into AGENTS.md
  section 11 and created this file.
- Note: the working tree also contained uncommitted draft Milestone 2 files
  from the previous session (encoder, decoder, net, train, tests). They are
  being reviewed and finished in this session rather than discarded.

### Open questions

- None blocking Milestone 2. Pending for later: whether the Colab T4 fits
  ConvNeXt-Base at 256x256, and whether branch B is trained pretrained or from
  scratch in the first real run.
