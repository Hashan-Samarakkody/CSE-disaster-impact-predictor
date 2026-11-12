"""Build Figure 18: Kaplan-Meier survival curve for Y3 recovery time.

Adds:
  - Censoring tick marks (show_censors=True)
  - Numbers at risk below the x-axis (add_at_risk_counts)

Output: docs/figures/fig_18_km_survival_y3.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.artifact_store import artifact_file
from src.visualization.result_figures import FIGURE_DIR, PALETTE, WIDTH_FULL, H_MED, stamp

FIGURE_NAME = "fig_18_km_survival_y3"
DPI = 300


def main() -> None:
    import pandas as pd
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts

    dataset = pd.read_parquet(artifact_file("dataset.parquet"))

    y3_col = "Y3_ASPI_Recovery_Time"
    obs_col = "Y3_event_observed"

    valid = dataset[y3_col].notna() & dataset[obs_col].notna()
    durations = dataset.loc[valid, y3_col].to_numpy(dtype=float)
    event_observed = dataset.loc[valid, obs_col].astype(bool).to_numpy()

    n_total = int(valid.sum())
    n_events = int(event_observed.sum())
    n_censored = n_total - n_events

    kmf = KaplanMeierFitter(label=f"ASPI recovery (n={n_total})")
    kmf.fit(durations, event_observed=event_observed)

    fig, ax = plt.subplots(figsize=(WIDTH_FULL, H_MED + 0.8))

    kmf.plot_survival_function(
        ax=ax,
        show_censors=True,
        censor_styles={"marker": "|", "ms": 6, "mew": 1.5},
        color=PALETTE["blue"],
        ci_show=True,
        ci_alpha=0.15,
    )

    add_at_risk_counts(kmf, ax=ax, fontsize=8, rows_to_show=["At risk"])

    ax.set_xlabel("Trading sessions after disaster", fontsize=10)
    ax.set_ylabel("Probability of not yet recovering", fontsize=10)
    ax.set_title("Kaplan-Meier estimate of recovery time", fontsize=11)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color=PALETTE["grey"], lw=0.8, ls="--", alpha=0.6)
    ax.legend(fontsize=9)

    stamp(fig, f"{n_events} recovered · {n_censored} censored · "
               f"numbers at risk shown below curve · tick marks denote censored observations")

    fig.tight_layout()

    out = FIGURE_DIR / f"{FIGURE_NAME}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
