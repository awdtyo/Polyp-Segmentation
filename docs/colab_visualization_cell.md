# Colab cell(s) to paste into colab_eval.ipynb for prediction visualization
# Run this AFTER the evaluation cells (which prepare data and run src.evaluate)
# Assumes the notebook already defines: REPO_DIR, RUNS_DIR, DRIVE_DATA_ROOT, LOCAL_DATA_ROOT

# -----------------------------------------------------------------------------
# Prediction visualization for exp006_all_concat_30ep
# -----------------------------------------------------------------------------
# This cell runs scripts/visualize_predictions.py on the exp006 run and saves
# figures to RUNS_DIR/../figures (i.e., /content/drive/MyDrive/polyp/figures).
# stderr is visible on failure.

# %% 
# --- CONFIGURATION ---
RUN_NAME = "exp006_all_concat_30ep"     # folder name under RUNS_DIR
FIGURES_DIR = Path(RUNS_DIR).parent / "figures"  # /content/drive/MyDrive/polyp/figures
DATASETS = ["kvasir_seg", "cvc_clinicdb", "cvc_colondb"]  # add "etis" if prepared and --allow_heldout
N_RANDOM = 4
N_FAILURES = 2
SEED = 42
SPLIT = "test"
CHECKPOINT = "best"
USE_CUDA = True  # set False to force CPU

# --- RUN ---
import subprocess
import sys
from pathlib import Path

run_dir = Path(RUNS_DIR) / RUN_NAME
if not run_dir.is_dir():
    raise FileNotFoundError(f"Run directory not found: {run_dir}")

out_dir = FIGURES_DIR / RUN_NAME
out_dir.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable,
    str(REPO_DIR / "scripts" / "visualize_predictions.py"),
    "--run_dir", str(run_dir),
    "--datasets", *DATASETS,
    "--out_dir", str(out_dir),
    "--n_random", str(N_RANDOM),
    "--n_failures", str(N_FAILURES),
    "--seed", str(SEED),
    "--split", SPLIT,
    "--checkpoint", CHECKPOINT,
]
if not USE_CUDA:
    cmd.append("--no_cuda")

print(f"Running visualization for {RUN_NAME}")
print(f"Output directory: {out_dir}")
print(f"Command: {' '.join(cmd)}")
print("---")

result = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=False)

# Print stdout/stderr (streamed in real-time would be better but this works)
if result.stdout:
    print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)

if result.returncode != 0:
    raise RuntimeError(f"Visualization failed with exit code {result.returncode}")

print(f"\n✓ Figures saved to: {out_dir}")
print(f"  - {RUN_NAME}/kvasir_seg/kvasir_seg_predictions.png/pdf")
print(f"  - {RUN_NAME}/cvc_clinicdb/cvc_clinicdb_predictions.png/pdf")
print(f"  - {RUN_NAME}/cvc_colondb/cvc_colondb_predictions.png/pdf")
print(f"  - combined_predictions.png/pdf")
print(f"  - *_selection.csv files")

# -----------------------------------------------------------------------------
# Optional: Side-by-side comparison with another run (e.g., exp005 hybrid fusion)
# -----------------------------------------------------------------------------
# Uncomment and modify to compare multiple runs:
#
# RUN_NAMES = ["exp005_hybrid_fusion", "exp006_all_concat_30ep"]
# out_dir = FIGURES_DIR / "comparison_exp005_vs_exp006"
# out_dir.mkdir(parents=True, exist_ok=True)
#
# cmd = [
#     sys.executable,
#     str(REPO_DIR / "scripts" / "visualize_predictions.py"),
#     *[arg for rn in RUN_NAMES for arg in ("--run_dir", str(Path(RUNS_DIR) / rn))],
#     "--datasets", *DATASETS,
#     "--out_dir", str(out_dir),
#     "--n_random", str(N_RANDOM),
#     "--n_failures", str(N_FAILURES),
#     "--seed", str(SEED),
#     "--split", SPLIT,
#     "--checkpoint", CHECKPOINT,
# ]
# if not USE_CUDA:
#     cmd.append("--no_cuda")
#
# result = subprocess.run(cmd, cwd=REPO_DIR, text=True, capture_output=False)
# if result.stdout: print(result.stdout)
# if result.stderr: print(result.stderr, file=sys.stderr)
# if result.returncode != 0: raise RuntimeError(f"Visualization failed: {result.returncode}")
# print(f"Comparison figures saved to: {out_dir}")

# -----------------------------------------------------------------------------
# Optional: Black-and-white mode for paper figures
# -----------------------------------------------------------------------------
# Add "--bw" to cmd for grayscale masks with black contours (different line styles).