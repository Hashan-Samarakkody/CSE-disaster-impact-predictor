"""Equivalence test: item 10 — real_time vs ex_post RMSE difference.

Tests whether the RMSE of real-time forecasts is practically equivalent to the RMSE of
ex-post forecasts (which have access to realised event characteristics). Equivalence is
declared if the bootstrap CI of (RMSE_real_time - RMSE_ex_post) falls entirely within
the pre-stated margin [-margin, +margin].

Pre-stated margins (declared before running, following the pre-registration protocol):
  - Y1 targets (return, horizons 5 and 10): margin = 0.10 log-return units
  - Y2 target (return, horizon 20): margin = 0.05 log-return units

Justification: 10% of the baseline naive_zero RMSE ≈ 0.28 for Y1, 0.06 for Y2;
margin set at 35% of this.

Method: paired bootstrap of (RMSE_real_time - RMSE_ex_post) per event (not per fold).
B = 10,000 bootstrap replications, resample events with replacement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.artifact_store import artifact_file, save_frame

RANDOM_STATE = 42
B = 10_000
CONFIRMATORY_MODELS = ("ridge", "random_forest", "mlp")
CONFIRMATORY_K = 10

# Pre-stated equivalence margins
MARGIN_Y1 = 0.10   # log-return units, horizons 5 and 10
MARGIN_Y2 = 0.05   # log-return units, horizon 20

MARGIN_JUSTIFICATION = (
    "10% of baseline naive_zero RMSE ≈ 0.28 for Y1, 0.06 for Y2; "
    "margin set at 35% of this"
)


def bootstrap_rmse_diff(se_rt: np.ndarray, se_ep: np.ndarray,
                         rng: np.random.Generator, b: int = B
                         ) -> tuple[float, float, float]:
    """Paired bootstrap of RMSE_real_time - RMSE_ex_post.

    se_rt, se_ep: per-event squared errors for real_time and ex_post (aligned).
    Returns: (delta_rmse, ci_low, ci_high).
    """
    n = len(se_rt)
    delta_rmse = float(np.sqrt(np.mean(se_rt)) - np.sqrt(np.mean(se_ep)))
    boot = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.sqrt(np.mean(se_rt[idx])) - np.sqrt(np.mean(se_ep[idx]))
    ci_low = float(np.quantile(boot, 0.025))
    ci_high = float(np.quantile(boot, 0.975))
    return delta_rmse, ci_low, ci_high


def main() -> None:
    preds = pd.read_parquet(artifact_file("aspi_grid_predictions.parquet"))
    rng = np.random.default_rng(RANDOM_STATE)

    # Targets: (label, horizon, margin)
    targets = [
        ("Y1_5D",  5,  MARGIN_Y1),
        ("Y1_10D", 10, MARGIN_Y1),
        ("Y2",     20, MARGIN_Y2),
    ]

    rows = []
    for target_label, horizon, margin in targets:
        rt = preds[(preds.horizon == horizon) & (preds.info_set == "real_time") &
                   (preds.k == CONFIRMATORY_K) &
                   preds.model.isin(CONFIRMATORY_MODELS)].copy()
        ep = preds[(preds.horizon == horizon) & (preds.info_set == "ex_post") &
                   (preds.k == CONFIRMATORY_K) &
                   preds.model.isin(CONFIRMATORY_MODELS)].copy()

        if len(rt) == 0 or len(ep) == 0:
            print(f"  {target_label}: no data — skipping")
            continue

        # Merge on (model, fold, row) to get matched pairs
        rt = rt.set_index(["model", "fold", "row"])[["y_true", "y_pred"]].rename(
            columns={"y_pred": "y_pred_rt"})
        ep = ep.set_index(["model", "fold", "row"])[["y_pred"]].rename(
            columns={"y_pred": "y_pred_ep"})
        merged = rt.join(ep, how="inner").reset_index()

        if len(merged) == 0:
            print(f"  {target_label}: no matched pairs — skipping")
            continue

        # Per-event squared errors (aggregate across models: mean SE per row)
        merged["se_rt"] = (merged["y_pred_rt"] - merged["y_true"]) ** 2
        merged["se_ep"] = (merged["y_pred_ep"] - merged["y_true"]) ** 2

        # One value per (fold, row) pair: average over models for that event
        per_event = merged.groupby(["fold", "row"]).agg(
            se_rt=("se_rt", "mean"), se_ep=("se_ep", "mean")).reset_index()

        se_rt = per_event["se_rt"].to_numpy()
        se_ep = per_event["se_ep"].to_numpy()

        delta_rmse, ci_low, ci_high = bootstrap_rmse_diff(se_rt, se_ep, rng)
        equivalence_supported = bool(ci_low >= -margin and ci_high <= margin)

        print(f"  {target_label} h={horizon}: delta_rmse={delta_rmse:+.4f} "
              f"CI=[{ci_low:+.4f}, {ci_high:+.4f}] margin={margin} "
              f"equiv={equivalence_supported}")

        rows.append({
            "target": target_label,
            "horizon": horizon,
            "n_events": len(per_event),
            "delta_rmse": delta_rmse,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "equivalence_margin": margin,
            "margin_justification": MARGIN_JUSTIFICATION,
            "equivalence_supported": equivalence_supported,
        })

    out = pd.DataFrame(rows)
    save_frame(out, "equivalence_test",
               "Paired bootstrap of RMSE_real_time - RMSE_ex_post per event. "
               "Margins pre-stated: 0.10 for Y1 (horizons 5, 10), 0.05 for Y2 (horizon 20). "
               f"B={B} bootstrap replications, RANDOM_STATE={RANDOM_STATE}.")
    print(f"\nwrote equivalence_test.parquet ({len(out)} rows)")


if __name__ == "__main__":
    main()
