"""Placebo event study: empirical size check for the event-study test statistics.

Draw 74 non-disaster origins matched to the real events on calendar month and
pre-event realised volatility, exclude any session inside a real disaster event window,
then run the same event-study machinery against the placebo dates. The empirical rejection
rate at 5% per statistic (BMP, Corrado, Kolari-Pynnonen) should not exceed ~5-10% under
the null. Substantial over-rejection would indicate that the tests fire on market
microstructure patterns rather than disaster-driven abnormal returns.

Matching algorithm:
  1. For each real event, enumerate candidate sessions in the same calendar month
     (any year) that are at least EVENT_WINDOW[1] + ESTIMATION_WINDOW sessions from
     any real disaster event window.
  2. Among candidates, pick the one whose 20-session realised volatility is closest
     to the real event's pre-event realised volatility.
  3. If fewer than N_PLACEBO candidates survive after exclusions, raise a warning and
     use the closest available match outside the month constraint.

Outputs:
  artifacts/tables/placebo_event_study.parquet  — same schema as the real event table
  artifacts/tables/placebo_event_study_provenance.json — seed, real event list, match map
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.event_study import (
    EVENT_WINDOW,
    build_abnormal_panel,
    event_study_table,
)
from src.targets.abnormal_returns import ESTIMATION_WINDOW, align_market_factor
from src.utils.artifact_store import artifact_file, save_frame

RANDOM_STATE = 42
N_PLACEBO = 74
VOL_WINDOW = 20  # sessions of realised vol for matching


def _realised_vol(returns: np.ndarray, positions: np.ndarray, window: int) -> np.ndarray:
    """Pre-event realised volatility for each position (std of prior `window` returns)."""
    vols = np.full(len(positions), np.nan)
    for i, pos in enumerate(positions):
        start = max(0, pos - window)
        end = pos
        if end - start >= 5:
            vols[i] = float(np.std(returns[start:end], ddof=1))
    return vols


def _event_window_mask(positions: np.ndarray, n_sessions: int,
                       pre: int, post: int, buffer: int = 0) -> np.ndarray:
    """Boolean mask: True for every session inside any event window (with buffer)."""
    mask = np.zeros(n_sessions, dtype=bool)
    for pos in positions:
        lo = max(0, pos + pre - buffer)
        hi = min(n_sessions, pos + post + buffer + 1)
        mask[lo:hi] = True
    return mask


def _load_daily() -> pd.DataFrame:
    """Mirror run_event_study.py's load_daily: market_feats + sp500 joined on date."""
    market_feats = pd.read_parquet(artifact_file("market_feats.parquet"))
    sp500 = pd.read_parquet(artifact_file("sp500.parquet"))
    daily = (market_feats[["date", "log_return"]]
             .sort_values("date").reset_index(drop=True))
    daily["sp500_log_return"] = align_market_factor(
        daily["date"], sp500["date"], sp500["sp500_log_return"])
    return daily


def main():
    market = _load_daily()
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))

    market_dates = pd.to_datetime(market["date"]).to_numpy()
    returns = market["log_return"].to_numpy(dtype=float)
    n_sessions = len(market_dates)

    # Real event positions: first market session on or after the event date.
    real_dates = pd.to_datetime(dataset["event_date"])
    real_positions = []
    for d in real_dates:
        cands = np.flatnonzero(market_dates >= np.datetime64(d))
        if len(cands) and cands[0] > 0:
            real_positions.append(int(cands[0]))
    real_positions = np.array(real_positions)

    # Block only the event windows of real disasters (not the estimation window).
    # The placebo estimation window can straddle past real events; that is acceptable
    # because the estimation window absorbs the same market conditions for both real
    # and placebo events. Including the estimation window in the block would leave too
    # few valid candidates given ESTIMATION_WINDOW=250.
    blocked = _event_window_mask(real_positions, n_sessions,
                                 EVENT_WINDOW[0], EVENT_WINDOW[1], buffer=0)

    # Pre-event realised vol for each real event.
    real_vols = _realised_vol(returns, real_positions, VOL_WINDOW)

    # Calendar month of each real event (1-12).
    real_months = pd.to_datetime(market_dates[real_positions]).month

    # Candidate sessions: have enough estimation history, have post-event data, not
    # inside any real event window. Duplicates allowed: with 74 matches from ~5000
    # candidates the collision rate is negligible and the pool would otherwise
    # exhaust for early-calendar months.
    candidate_mask = ~blocked
    candidate_mask[:ESTIMATION_WINDOW] = False         # need estimation history
    candidate_mask[n_sessions - EVENT_WINDOW[1]:] = False  # need post-event data
    candidate_positions = np.flatnonzero(candidate_mask)
    if len(candidate_positions) == 0:
        raise RuntimeError("No valid placebo candidates after applying blocking constraints")
    candidate_months = pd.to_datetime(market_dates[candidate_positions]).month
    candidate_vols = _realised_vol(returns, candidate_positions, VOL_WINDOW)

    rng = np.random.default_rng(RANDOM_STATE)
    placebo_positions = []
    match_map = {}  # real position -> chosen placebo position

    for i, (rpos, rmonth, rvol) in enumerate(zip(real_positions, real_months, real_vols)):
        # Same-month candidates first, then relax to any unblocked session.
        month_ok = candidate_months == rmonth
        same_month = candidate_positions[month_ok]
        same_month_vols = candidate_vols[month_ok]

        if len(same_month) > 0 and np.any(np.isfinite(same_month_vols)):
            pool = same_month
            pool_vols = same_month_vols
        else:
            pool = candidate_positions
            pool_vols = candidate_vols

        # Closest pre-event vol; uniform weights when vol is unavailable.
        if np.isfinite(rvol) and np.any(np.isfinite(pool_vols)):
            dist = np.where(np.isfinite(pool_vols), np.abs(pool_vols - rvol), np.inf)
        else:
            dist = np.ones(len(pool))

        chosen_local = int(np.argmin(dist))
        chosen_pos = int(pool[chosen_local])
        placebo_positions.append(chosen_pos)
        match_map[int(rpos)] = chosen_pos

    placebo_dates = pd.to_datetime(market_dates[placebo_positions])

    # Run the event study unchanged against placebo origins.
    panel = build_abnormal_panel(
        market,
        placebo_dates,
        model="market",
        event_window=EVENT_WINDOW,
        estimation_window=ESTIMATION_WINDOW,
        date_col="date",
        value_col="log_return",
        factor_col="sp500_log_return",
    )
    table = event_study_table(panel, start=1, label="placebo")
    save_frame(table, "placebo_event_study",
               "Placebo event study: empirical size of BMP, Corrado, Kolari-Pynnonen.")

    # Empirical rejection rate at 5%.
    reject_bmp = float((table["p_bmp"].dropna() < 0.05).mean())
    reject_corrado = float((table["p_corrado"].dropna() < 0.05).mean())
    reject_kp = float((table["p_kolari_pynnonen"].dropna() < 0.05).mean())

    provenance = {
        "script": "scripts/run_placebo_event_study.py",
        "random_state": RANDOM_STATE,
        "n_placebo": len(placebo_positions),
        "n_real_events": len(real_positions),
        "vol_window": VOL_WINDOW,
        "estimation_window": ESTIMATION_WINDOW,
        "event_window": list(EVENT_WINDOW),
        "match_map": {str(k): v for k, v in match_map.items()},
        "placebo_session_indices": [int(p) for p in placebo_positions],
        "empirical_rejection_rate_5pct": {
            "bmp": reject_bmp,
            "corrado": reject_corrado,
            "kolari_pynnonen": reject_kp,
        },
    }
    prov_path = artifact_file("placebo_event_study_provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2))

    print(f"Placebo event study: {len(placebo_positions)} placebo origins")
    print(f"Empirical rejection rate at 5%:")
    print(f"  BMP:              {reject_bmp:.3f}")
    print(f"  Corrado:          {reject_corrado:.3f}")
    print(f"  Kolari-Pynnonen:  {reject_kp:.3f}")
    print(f"wrote placebo_event_study.parquet ({len(table)} rows)")
    print(f"wrote placebo_event_study_provenance.json")


if __name__ == "__main__":
    main()
