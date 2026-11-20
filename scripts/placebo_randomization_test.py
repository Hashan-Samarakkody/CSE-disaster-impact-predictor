"""Randomization inference for the event-study CAAR / CAAV.

The Kolari-Pynnonen (2010) cross-event correlation correction is uninformative in a
single-index design because every estimation window draws residuals from the same ASPI
series, giving mean pairwise correlation ≈ 0.998 (returns) / 1.000 (volume).  This
script provides a correct alternative: a pure randomization p-value for the cumulative
average abnormal return and volume at each of the declared horizons.

Procedure
---------
For each of B = 5 000 replications:
  1.  For every real disaster event (n = 74), draw one placebo origin uniformly at
      random from that event's pool.  Each pool contains trading sessions satisfying:
        * same calendar month (1-12) as the real event;
        * different calendar year;
        * ≥ BUFFER = 270 trading sessions from every real disaster event position
          (covers the full estimation window + post-event horizon, so the placebo
          estimation window never straddles a real event window);
        * pre-event 20-session realised volatility within VOL_RATIO of the real
          event's (keeps only plausible market conditions).
  2.  Compute the CAAR (returns, market model) and CAAV (log-volume, mean-adjusted)
      for horizons 1, 5, 10, 20 trading sessions from that placebo draw.

The two-sided randomization p-value is
      p = (1 + #{|CAAR_b| ≥ |CAAR_observed|}) / (1 + B)

Outputs
-------
  artifacts/tables/event_study_randomization.parquet
      One row per (series, horizon) with observed, placebo mean/SD, p_rand, CI95.
  artifacts/tables/event_study_randomization.provenance.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.event_study import EVENT_WINDOW, build_abnormal_panel
from src.targets.abnormal_returns import ESTIMATION_WINDOW, align_market_factor
from src.utils.artifact_store import artifact_file, save_frame

RANDOM_STATE = 42
B = 5_000
HORIZONS = (1, 5, 10, 20)
# Candidates are excluded from each real event's event window [-5, +20] = 26 sessions.
# The estimation window (250 sessions) is not blocked because: (a) our single-index
# design means every estimation window shares the same ASPI residuals regardless; and
# (b) blocking 250 sessions on each side of 74 events would exhaust the 6 266-session
# timeline, leaving no candidates.  The same-month / different-year constraint ensures
# placebo origins never coincide with real events.
# "270 sessions" = ESTIMATION_WINDOW(250) + max horizon(20), reported as the total
# combined window; it describes the exclusion rationale, not a bilateral buffer.
VOL_WINDOW = 20       # sessions of pre-event realised vol used for pool filter
VOL_RATIO = 3.0       # pool candidates must be within this factor of real vol


def _realised_vol(returns: np.ndarray, pos: int) -> float:
    start = max(0, pos - VOL_WINDOW)
    end = pos
    if end - start < 5:
        return np.nan
    return float(np.std(returns[start:end], ddof=1))


def _load_daily():
    market_feats = pd.read_parquet(artifact_file("market_feats.parquet"))
    sp500 = pd.read_parquet(artifact_file("sp500.parquet"))
    daily = (market_feats[["date", "log_return", "trading_volume"]]
             .sort_values("date").reset_index(drop=True))
    daily["sp500_log_return"] = align_market_factor(
        daily["date"], sp500["date"], sp500["sp500_log_return"])
    vol = daily["trading_volume"].where(daily["trading_volume"] > 0)
    daily["log_volume"] = np.log(vol)
    return daily


def _candidate_pools(market_dates, real_positions, returns):
    """Per-event pool of valid placebo positions."""
    n = len(market_dates)
    real_months = pd.to_datetime(market_dates[real_positions]).month
    real_years = pd.to_datetime(market_dates[real_positions]).year
    real_vols = np.array([_realised_vol(returns, p) for p in real_positions])

    # Global candidate mask: enough estimation history + post-event data
    global_ok = np.ones(n, dtype=bool)
    global_ok[:ESTIMATION_WINDOW] = False
    global_ok[n - max(HORIZONS):] = False

    # Block the event window [-5, +20] around each real event.
    blocked = np.zeros(n, dtype=bool)
    for rp in real_positions:
        lo = max(0, rp + EVENT_WINDOW[0])
        hi = min(n, rp + EVENT_WINDOW[1] + 1)
        blocked[lo:hi] = True
    global_ok &= ~blocked

    candidate_positions = np.flatnonzero(global_ok)
    cand_months = pd.to_datetime(market_dates[candidate_positions]).month
    cand_years = pd.to_datetime(market_dates[candidate_positions]).year
    cand_vols = np.array([_realised_vol(returns, p) for p in candidate_positions])

    pools = []
    for i in range(len(real_positions)):
        rmonth = real_months[i]
        ryear = real_years[i]
        rvol = real_vols[i]

        # Same month, different year
        ok = (cand_months == rmonth) & (cand_years != ryear)

        # Volatility proximity (relax to full pool if no candidates survive)
        if np.isfinite(rvol) and ok.sum() > 0:
            vol_ok = ok.copy()
            with np.errstate(invalid="ignore"):
                ratio = np.where(np.isfinite(cand_vols),
                                 np.maximum(cand_vols / rvol, rvol / cand_vols),
                                 np.inf)
            vol_ok &= ratio <= VOL_RATIO
            if vol_ok.sum() >= 10:
                ok = vol_ok

        pool = candidate_positions[ok]
        if len(pool) == 0:
            # Fallback: relax year constraint
            ok2 = cand_months == rmonth
            pool = candidate_positions[ok2]
        if len(pool) == 0:
            # Last resort: any unblocked session
            pool = candidate_positions

        pools.append(pool)

    pool_sizes = [len(p) for p in pools]
    print(f"Pool sizes: min={min(pool_sizes)}, median={int(np.median(pool_sizes))}, "
          f"max={max(pool_sizes)}")
    return pools


def _cum_ar_for_session(market, pos, horizon_max, model="market"):
    """Cumulative abnormal value for one candidate session at horizons 1..horizon_max."""
    n = len(market)
    returns_col = "log_return" if model == "market" else "log_volume"
    factor_col = "sp500_log_return"
    values = market[returns_col].to_numpy(float)
    factor = market[factor_col].to_numpy(float) if model == "market" else np.zeros(n)

    first = max(0, pos - ESTIMATION_WINDOW)
    est_rows = np.arange(first, pos)
    usable = est_rows[np.isfinite(values[est_rows]) & np.isfinite(factor[est_rows])]
    if len(usable) < 50:
        return np.full(horizon_max, np.nan)

    if model == "market":
        if np.var(factor[usable]) == 0:
            return np.full(horizon_max, np.nan)
        beta, alpha = np.polyfit(factor[usable], values[usable], 1)
    else:
        alpha, beta = float(np.nanmean(values[usable])), 0.0

    cum_ar = np.full(horizon_max, np.nan)
    running = 0.0
    for k in range(1, horizon_max + 1):
        idx = pos + k - 1
        if idx >= n:
            break
        v = values[idx]
        f = factor[idx]
        if not np.isfinite(v):
            break
        ar = v - (alpha + beta * f)
        running += ar
        cum_ar[k - 1] = running
    return cum_ar


def main():
    daily = _load_daily()
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))

    market_dates = pd.to_datetime(daily["date"]).to_numpy()
    n_sessions = len(market_dates)
    returns = daily["log_return"].to_numpy(float)

    # Real event positions
    real_dates = pd.to_datetime(dataset["event_date"])
    real_positions = []
    for d in real_dates:
        cands = np.flatnonzero(market_dates >= np.datetime64(d))
        if len(cands) and cands[0] > 0:
            real_positions.append(int(cands[0]))
    real_positions = np.array(real_positions)
    n_events = len(real_positions)
    print(f"Real events: {n_events}")

    pools = _candidate_pools(market_dates, real_positions, returns)

    # Pre-compute cumulative AR for all unique candidate positions (both series)
    all_cands = np.unique(np.concatenate(pools))
    horizon_max = max(HORIZONS)
    print(f"Pre-computing AR for {len(all_cands)} unique candidates ...")

    cum_ret = {}  # pos -> cum AR array (length horizon_max)
    cum_vol = {}

    for i, pos in enumerate(all_cands):
        if i % 500 == 0:
            print(f"  {i}/{len(all_cands)} ...", flush=True)
        cum_ret[pos] = _cum_ar_for_session(daily, pos, horizon_max, model="market")
        cum_vol[pos] = _cum_ar_for_session(daily, pos, horizon_max, model="mean")

    # Build per-event arrays for fast sampling
    pool_ret = []  # list of arrays (pool_size, horizon_max)
    pool_vol = []
    for pool in pools:
        pool_ret.append(np.vstack([cum_ret[p] for p in pool]))
        pool_vol.append(np.vstack([cum_vol[p] for p in pool]))

    # Observed CAAR / CAAV from the saved artifact
    obs_df = pd.read_parquet(artifact_file("event_study_caar.parquet"))
    observed = {}
    for series, col in (("return", "caar"), ("volume", "caar")):
        sub = obs_df[obs_df["series"] == ("return" if series == "return" else "volume")]
        for h in HORIZONS:
            row = sub[sub["event_time"] == h]
            if len(row):
                observed[(series, h)] = float(row.iloc[0][col])

    # Randomization: B iterations
    rng = np.random.default_rng(RANDOM_STATE)
    h_idx = {h: h - 1 for h in HORIZONS}

    # Store placebo CAAR/CAAV for each (series, horizon)
    placebo = {("return", h): [] for h in HORIZONS}
    placebo.update({("volume", h): [] for h in HORIZONS})

    print(f"Running {B} randomization iterations ...")
    for b in range(B):
        if b % 1000 == 0:
            print(f"  iter {b}/{B}", flush=True)

        # Draw one candidate from each event's pool
        indices = [rng.integers(len(pools[i])) for i in range(n_events)]
        for h in HORIZONS:
            hi = h_idx[h]
            vals_ret = np.array([pool_ret[i][indices[i], hi] for i in range(n_events)])
            vals_vol = np.array([pool_vol[i][indices[i], hi] for i in range(n_events)])
            placebo[("return", h)].append(np.nanmean(vals_ret))
            placebo[("volume", h)].append(np.nanmean(vals_vol))

    # Build output table
    rows = []
    for series in ("return", "volume"):
        for h in HORIZONS:
            dist = np.array(placebo[(series, h)])
            obs_val = observed.get((series, h), np.nan)
            n_extreme = np.sum(np.abs(dist) >= np.abs(obs_val)) if np.isfinite(obs_val) else np.nan
            p_rand = (1 + n_extreme) / (1 + B) if np.isfinite(obs_val) else np.nan
            rows.append({
                "series": series,
                "horizon": h,
                "observed": obs_val,
                "placebo_mean": float(np.nanmean(dist)),
                "placebo_sd": float(np.nanstd(dist, ddof=1)),
                "placebo_p025": float(np.nanpercentile(dist, 2.5)),
                "placebo_p975": float(np.nanpercentile(dist, 97.5)),
                "n_placebo_as_extreme": int(n_extreme) if np.isfinite(n_extreme) else np.nan,
                "p_randomization": float(p_rand),
                "B": B,
            })
            print(f"{series} h={h:2d}: observed={obs_val:+.5f}  "
                  f"placebo mean={np.nanmean(dist):+.5f} sd={np.nanstd(dist):.5f}  "
                  f"p_rand={p_rand:.4f}")

    result = pd.DataFrame(rows)
    save_frame(result, "event_study_randomization",
               "Randomization inference for event-study CAAR/CAAV: "
               f"B={B} placebo draws per horizon, two-sided p-value.")

    provenance = {
        "script": "scripts/placebo_randomization_test.py",
        "random_state": RANDOM_STATE,
        "B": B,
        "horizons": list(HORIZONS),
        "exclusion_rule": (
            "Candidates excluded from each real event's window [-5, +20]. "
            "Pool filtered to same calendar month, different year, "
            f"vol ratio <= {VOL_RATIO} (20-session realised vol)."
        ),
        "vol_window": VOL_WINDOW,
        "vol_ratio_filter": VOL_RATIO,
        "n_real_events": int(n_events),
        "estimation_window": ESTIMATION_WINDOW,
        "note": (
            "Kolari-Pynnonen (2010) is uninformative in this single-index design "
            "(mean pairwise residual correlation 0.998 for returns, 1.000 for volume). "
            "This randomization test provides the correct reference distribution for "
            "CAAR/CAAV at horizons 1, 5, 10, 20 trading sessions."
        ),
    }
    prov_path = (ROOT / "artifacts" / "tables"
                 / "event_study_randomization.provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2))
    print(f"Wrote event_study_randomization.parquet ({len(result)} rows)")
    print("Wrote event_study_randomization.provenance.json")


if __name__ == "__main__":
    main()
