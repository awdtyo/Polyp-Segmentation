# results.csv schema and grouping rule

## Grouping rule (aggregation)

`scripts/aggregate_results.py` groups results rows by **experiment identity**, never by the
session config path (`config_file`).

The grouping key is the `run_name` column (which equals `cfg["run"]["name"]` and the run
folder name) with any trailing seed suffix stripped:

* `_seed<N>` (e.g. `exp005_seed43` → `exp005`)
* `_s<N>` (e.g. `exp005_s43` → `exp005`)

Seeds are counted only within one group. When a group has a single seed, the std is
`n/a`, never `0.0`. The details table has one row per `(config, dataset, split, tta)`;
params, best epoch, epochs trained, mean epoch time and peak GPU memory come from that
group's own runs (mean over its seeds, or the single-run value when single-seed).

`config_file` is kept in the CSV for backward compatibility but is deliberately not used
for grouping, because every Colab row historically shares `/content/config_session.yaml`
and different experiments would otherwise be averaged as if they were seeds.

A `config_name` column records the experiment identity (`cfg["run"]["name"]`) in every
new row, but grouping never depends on it.

## Legacy migration

Results CSVs written before the schema change have the 16-column header ending in
`...,epochs,params_m,...,small_lesion_dice,mean_epoch_time_s,peak_gpu_mem_mb`.

On the first append to such a legacy file, `src/evaluate.py`:

1. Backs it up as `results.legacy.csv` next to it (only the first migration creates the
   backup).
2. Rewrites it once with the current header, copying existing values unchanged, mapping
   the old `epochs` column (which held the best epoch) to `best_epoch`, and leaving the
   new columns `config_name`, `epochs_trained` and `small_lesion_n` blank.
3. Logs that a migration happened, then resumes normal append-only behaviour.

Legacy files remain readable: the reader treats the old `epochs` column as `best_epoch`.

## Current header (`RESULTS_COLUMNS`)

| Column | Meaning |
|---|---|
| `run_name` | Run folder name / `cfg["run"]["name"]` |
| `split` | `test` or `val` |
| `dataset` | Dataset name (e.g. `kvasir_seg`, `cvc_clinicdb`, `cvc_colondb`) |
| `tta` | `""` or `flip` |
| `config_name` | Experiment identity (new; legacy rows: blank) |
| `config_file` | Config path used at run time |
| `git_commit` | Git commit of the run |
| `seed` | Training seed |
| `best_epoch` | Epoch at which `best.pt` was saved (model-selection signal) |
| `epochs_trained` | Total epochs run, from the run's `metrics.csv` |
| `params_m` | Parameters in millions |
| `best_val_dice` | Best validation Dice |
| `test_dice` | Test Dice (per-image mean) |
| `test_iou` | Test IoU (per-image mean) |
| `test_mae` | Test MAE on probabilities vs binary GT |
| `small_lesion_dice` | Mean Dice over small-lesion images, or `n/a` if fewer than 10 qualify |
| `small_lesion_n` | Number of images that qualified as small lesions |
| `mean_epoch_time_s` | Mean epoch time (s) |
| `peak_gpu_mem_mb` | Peak GPU memory (MB) |

## Small-lesion definition

A lesion is "small" when its foreground pixel count is below
`data.small_lesion_area_fraction` of the image area (**default `0.05`, i.e. 5% =
3277 px at 256x256**). The threshold is chosen from ground-truth mask statistics
only, never from model scores, and is configurable in the experiment config.

The default is 5% because it is the lowest tested threshold (1%, 2%, 3%, 5%,
10%) at which every dataset — including the held-out Kvasir test split — has at
least 10 qualifying images (the minimum for reporting `small_lesion_dice`; below
10 it is written `n/a`, never `0.0`). Set on 2026-11-01 from committed
ground-truth masks at 256x256.

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

`small_lesion_dice` is the mean Dice over images that qualify as small. If fewer
than 10 images qualify in a dataset, the value is written as `n/a` and
`small_lesion_n` is still recorded; `0.0` is never written for "none qualified".

The column can be backfilled for existing runs without re-running inference:
`python scripts/backfill_small_lesion.py --results-csv <runs>/results.csv --runs-dir <runs>`

## Legacy migration and backup naming

Results CSVs written before the schema change have the 16-column header ending in
`...,epochs,params_m,...,small_lesion_dice,mean_epoch_time_s,peak_gpu_mem_mb`.

On the first append to such a legacy file, `src/evaluate.py`:

1. Backs it up next to itself as `results.legacy.csv`. If that path already
   exists, it uses a timestamped name `results.legacy-YYYYMMDD-HHMMSS.csv`
   instead, so an existing backup is never overwritten.
2. Rewrites the original once with the current header, copying existing values
   unchanged, mapping the old `epochs` column (which held the best epoch) to
   `best_epoch`, and leaving the new columns `config_name`, `epochs_trained` and
   `small_lesion_n` blank.
3. Logs that a migration happened, then resumes normal append-only behaviour.

Legacy files remain readable: the reader treats the old `epochs` column as
`best_epoch`.
