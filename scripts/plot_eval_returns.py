"""
Generate clean evaluation-return plots from two wandb CSV exports.

Layout: two panels
  left  -> no belief map:  classic vs OM
  right -> with belief map: classic vs OM

For each condition we aggregate the available runs (columns) into a
mean line with a shaded +/-1 std band.
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
CLASSIC_CSV = "wandb_export_2026-08-21T18_38_46.785+02_00.csv"
OM_CSV = "wandb_export_2026-08-21T18_38_11.786+02_00.csv"

EPISODES_PER_EPOCH = 500  # matches --episodes_per_epoch

# Output location next to the report
OUT_DIR = Path("report/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot styling
COLOR_CLASSIC = "#1f77b4"  # blue
COLOR_OM = "#2ca02c"  # green
BAND_ALPHA = 0.18

# wandb run-name prefixes present in the exports
CONDITIONS = {
    "no_belief_map": ["no_belief_map", "no_belief_map_run_2"],
    "belief_map": [
        "belief_map",
        "belief_map_run_2",
        "belief_map_run_3",
        "belief_map_run_4",
    ],
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def collect_runs(df: pd.DataFrame, agent: str, run_names: list[str]) -> np.ndarray:
    """
    Stack the `eval_return` columns for the given exact run names.

    A column for run `r` and `agent` is exactly
        "map4_vs_greedy_{r} - {agent}/eval_return"
    We match the full column name (not a loose substring) so that a run name
    such as "belief_map" does not also match the "no_belief_map" columns, which
    it would under prefix matching ("...no_belief_map - ..." ends with
    "...belief_map - ...").
    """
    cols = []
    for r in run_names:
        col = f"map4_vs_greedy_{r} - {agent}/eval_return"
        if col not in df.columns:
            raise ValueError(
                f"No eval_return column found for agent='{agent}' run='{r}' ({col!r})"
            )
        cols.append(col)
    arr = df[cols].to_numpy(dtype=float)  # (epochs, n_runs)
    return arr


def mean_std(arr: np.ndarray):
    """Return mean and +/-1 std across the run axis (axis=1)."""
    mu = np.nanmean(arr, axis=1)
    sd = np.nanstd(arr, axis=1, ddof=0)
    return mu, sd


def episodes_axis(df: pd.DataFrame) -> np.ndarray:
    return df["epoch"].to_numpy(dtype=int) * EPISODES_PER_EPOCH


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
classic = pd.read_csv(CLASSIC_CSV)
om = pd.read_csv(OM_CSV)

# sanity: both share the epoch grid
assert len(classic) == len(om), "epoch counts differ between the two exports"
assert np.array_equal(classic["epoch"].to_numpy(), om["epoch"].to_numpy()), (
    "epoch grids differ between the two exports"
)

x = episodes_axis(classic)

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)

panel_specs = [
    ("no_belief_map", "No belief map", "classic vs. OM, no belief map"),
    ("belief_map", "With belief map", "classic vs. OM, with belief map"),
]

for ax, (cond, title, _) in zip(axes, panel_specs):
    prefixes = CONDITIONS[cond]
    c_arr = collect_runs(classic, "classic", prefixes)
    o_arr = collect_runs(om, "om", prefixes)

    c_mu, c_sd = mean_std(c_arr)
    o_mu, o_sd = mean_std(o_arr)

    # Classic
    ax.plot(x, c_mu, color=COLOR_CLASSIC, lw=2, label="Classic agent")
    ax.fill_between(x, c_mu - c_sd, c_mu + c_sd, color=COLOR_CLASSIC, alpha=BAND_ALPHA)

    # OM
    ax.plot(x, o_mu, color=COLOR_OM, lw=2, label="OM agent")
    ax.fill_between(x, o_mu - o_sd, o_mu + o_sd, color=COLOR_OM, alpha=BAND_ALPHA)

    ax.set_title(title)
    ax.set_xlabel("Training episodes")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(frameon=False)

axes[0].set_ylabel("Average evaluation return")
fig.suptitle(
    r"Evaluation return on MAP$_4$ vs.\ greedy opponent "
    f"(mean $\\pm$ std over {len(CONDITIONS['belief_map'])}/{len(CONDITIONS['no_belief_map'])} runs)",
    y=1.02,
)
fig.tight_layout()

# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
for ext in ("pdf", "png"):
    out = OUT_DIR / f"eval_return_map4.{ext}"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")

# Also save the two panels as separate files for flexible inclusion
for cond, title, _ in panel_specs:
    prefixes = CONDITIONS[cond]
    c_arr = collect_runs(classic, "classic", prefixes)
    o_arr = collect_runs(om, "om", prefixes)
    c_mu, c_sd = mean_std(c_arr)
    o_mu, o_sd = mean_std(o_arr)

    fig_s, ax_s = plt.subplots(figsize=(6.2, 4.6))
    ax_s.plot(x, c_mu, color=COLOR_CLASSIC, lw=2, label="Classic agent")
    ax_s.fill_between(
        x, c_mu - c_sd, c_mu + c_sd, color=COLOR_CLASSIC, alpha=BAND_ALPHA
    )
    ax_s.plot(x, o_mu, color=COLOR_OM, lw=2, label="OM agent")
    ax_s.fill_between(x, o_mu - o_sd, o_mu + o_sd, color=COLOR_OM, alpha=BAND_ALPHA)
    ax_s.set_title(title)
    ax_s.set_xlabel("Training episodes")
    ax_s.set_ylabel("Average evaluation return")
    ax_s.grid(True, ls=":", alpha=0.5)
    ax_s.legend(frameon=False)
    fig_s.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"eval_return_map4_{cond}.{ext}"
        fig_s.savefig(out, dpi=200, bbox_inches="tight")
        print(f"saved {out}")
    plt.close(fig_s)

plt.close(fig)
print("done.")
