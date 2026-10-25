"""Explicit cumulative-incidence baseline for Y3 recovery — item B.3.

Adds a CIF (Cumulative Incidence Function) baseline alongside the KM median.
The KM CIF = 1 - S_KM(t) is already embedded in km_train_baseline rows of
recovery_grid_predictions.parquet (P_T_le_t columns).  This script extracts it,
adds Greenwood-based CI, and emits recovery_cif_baseline.parquet.

Columns: model, horizon, cif_estimate, ci_low, ci_high, n_at_risk, n_events,
         n_censored_before, km_survival, greenwood_var
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.artifact_store import artifact_file, save_frame

PROB_TIMES = [10, 20, 30, 60, 90]


def _km_at_t(durations, events, t):
    """KM estimate and Greenwood CI at time t (not IPCW-weighted, standard)."""
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events)
    s = float(kmf.predict(t))
    # Greenwood variance: sum d_i / (n_i * (n_i - d_i)) up to t
    tbl = kmf.event_table
    tbl = tbl[tbl.index <= t]
    # n_at_risk and removed are in the event table
    at_risk = tbl["at_risk"].to_numpy(float)
    deaths = tbl["observed"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where((at_risk > deaths) & (at_risk > 0),
                         deaths / (at_risk * (at_risk - deaths)), 0.0)
    gw_var = s ** 2 * terms.sum()
    se = float(np.sqrt(gw_var))
    ci_lo = max(0.0, s - 1.96 * se)
    ci_hi = min(1.0, s + 1.96 * se)
    # CIF = 1 - S
    n_at_risk_at_t = int((durations >= t).sum())
    n_events_at_t = int(((durations < t) & events.astype(bool)).sum())
    return {
        "cif_estimate": 1.0 - s,
        "km_survival": s,
        "ci_low": 1.0 - ci_hi,   # CI on CIF (reversed from survival CI)
        "ci_high": 1.0 - ci_lo,
        "greenwood_var": gw_var,
        "n_at_risk": n_at_risk_at_t,
        "n_events": n_events_at_t,
        "n_censored_before": int(((durations < t) & (~events.astype(bool))).sum()),
    }


def main():
    preds = pd.read_parquet(artifact_file("recovery_grid_predictions.parquet"))
    dataset = pd.read_parquet(artifact_file("dataset.parquet")).reset_index(drop=True)

    # Full held-out durations and events (pooled across folds)
    # Use the km_train_baseline rows as representative (durations are per-event, same for all models)
    km_rows = preds[preds["model"] == "km_train_baseline"].drop_duplicates("row")
    durations = km_rows["duration"].to_numpy(float)
    events = km_rows["event_observed"].to_numpy(bool)

    rows = []
    for t in PROB_TIMES:
        stats = _km_at_t(durations, events, float(t))
        rows.append({
            "model": "cif_km_baseline",
            "horizon": int(t),
            **stats,
        })

    # Also report the train-fold CIF: for each fold's km_train_baseline, extract P_T_le_t
    # (which is 1 - S_KM fitted on that fold's training set), giving a per-fold CIF baseline
    fold_rows = []
    for fold_i, grp in preds[preds["model"] == "km_train_baseline"].groupby("fold"):
        for t in PROB_TIMES:
            col = f"P_T_le_{t}"
            if col in grp.columns:
                fold_rows.append({
                    "model": "cif_km_baseline",
                    "fold": int(fold_i),
                    "horizon": t,
                    "mean_predicted_cif": float(grp[col].mean()),
                    "n": len(grp),
                })

    out = pd.DataFrame(rows)
    fold_out = pd.DataFrame(fold_rows) if fold_rows else pd.DataFrame()

    save_frame(out, "recovery_cif_baseline",
               ("Explicit cumulative-incidence (CIF) baseline for Y3 recovery. "
                "CIF = 1 - KM(t) fitted on pooled held-out events with Greenwood CI. "
                "Equivalent to the km_train_baseline P_T_le_t column but with uncertainty. "
                "Companion to recovery_grid_metrics and survival_discrimination."))

    if not fold_out.empty:
        save_frame(fold_out, "recovery_cif_baseline_by_fold",
                   "Per-fold mean CIF for the KM baseline, for calibration cross-checks.")

    print(f"wrote recovery_cif_baseline.parquet ({len(out)} rows, {len(PROB_TIMES)} horizons)")
    if not fold_out.empty:
        print(f"wrote recovery_cif_baseline_by_fold.parquet ({len(fold_out)} rows)")

    # Print summary
    print("\nCIF baseline summary:")
    for _, r in out.iterrows():
        print(f"  t={r.horizon:2d}: CIF={r.cif_estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]  "
              f"n_at_risk={r.n_at_risk}  n_events={r.n_events}")


if __name__ == "__main__":
    main()
