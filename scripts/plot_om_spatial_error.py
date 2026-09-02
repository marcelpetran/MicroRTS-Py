"""
Generate a clean opponent-model spatial-error plot from a wandb CSV export.

The export (om/eval_spatial_error) contains, for every training epoch, the
expected spatial error (in tiles, lower is better) of the OM's predicted
subgoal heatmap, logged separately for each training run.

The opponent model conditions only on the observation history and does *not*
consume the belief map, so this metric is independent of the policy's
belief-map setting. The different runs here correspond to OM-hyperparameter
fine-tuning trials. We therefore pool all runs into a single mean line with a
shaded +/-1 std band, mirroring the style of plot_eval_returns.py.
"""

import os
import tempfile
from pathlib import Path

# Redirect matplotlib's config/cache dir to a writable temp dir so it does not
# try to write to the (read-only) home directory.
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SPATIAL_ERROR_CSV = "wandb_export_2026-08-21T18_46_21.442+02_00.csv"

EPISODES_PER_EPOCH = 500  # matches --episodes_per_epoch

# Output location next to the report
OUT_DIR = Path("report/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot styling
COLOR_OM = "#2ca02c"  # green (consistent with the OM line in plot_eval_returns.py)
BAND_ALPHA = 0.18

METRIC = "om/eval_spatial_error"

# wandb groups runs under "<group>_<run>"; the exported column looks like
# "map4_vs_greedy_<run> - om/eval_spatial_error".
COL_PREFIX = "map4_vs_greedy_"

# All available runs are pooled: the opponent model does not use the belief
# map, so the belief/no-belief distinction is irrelevant for this metric and
# the runs simply correspond to different OM fine-tuning trials.
ALL_RUNS = [
    "belief_map",
    "belief_map_run_2",
    "belief_map_run_3",
    "belief_map_run_4",
    "no_belief_map",
    "no_belief_map_run_2",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def collect_runs(df: pd.DataFrame, run_names: list[str]) -> np.ndarray:
    """Stack the metric columns for the given exact run names -> (epochs, n_runs)."""
    cols = []
    for run in run_names:
        col = f"{COL_PREFIX}{run} - {METRIC}"
        if col not in df.columns:
            raise ValueError(f"Missing column for run='{run}': {col!r}")
        cols.append(col)
    return df[cols].to_numpy(dtype=float)


def mean_std(arr: np.ndarray):
    """Return mean and +/-1 std across the run axis (axis=1)."""
    mu = np.nanmean(arr, axis=1)
    sd = np.nanstd(arr, axis=1, ddof=0)
    return mu, sd


def episodes_axis(df: pd.DataFrame) -> np.ndarray:
    return df["epoch"].to_numpy(dtype=int) * EPISODES_PER_EPOCH


# --------------------------------------------------------------------------- #
# Load & plot
# --------------------------------------------------------------------------- #
df = pd.read_csv(SPATIAL_ERROR_CSV)
x = episodes_axis(df)

arr = collect_runs(df, ALL_RUNS)
mu, sd = mean_std(arr)
n_runs = arr.shape[1]

fig, ax = plt.subplots(figsize=(7.0, 4.6))
ax.plot(x, mu, color=COLOR_OM, lw=2, label="OM agent")
ax.fill_between(x, mu - sd, mu + sd, color=COLOR_OM, alpha=BAND_ALPHA)
ax.set_title(
    r"OM spatial error on MAP$_4$ vs.\ greedy "
    f"(mean $\\pm$ std over {n_runs} runs)"
)
ax.set_xlabel("Training episodes")
ax.set_ylabel("Average evaluation spatial error (tiles)")
ax.grid(True, ls=":", alpha=0.5)
ax.legend(frameon=False)
fig.tight_layout()

for ext in ("pdf", "png"):
    out = OUT_DIR / f"om_spatial_error_map4.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
plt.close(fig)

print("done.")
