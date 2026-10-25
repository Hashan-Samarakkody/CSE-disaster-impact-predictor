"""Abnormal return variants: item 6 — four expected-return model specifications.

Re-estimate the event study under four variants of the expected-return model and report
all results without selecting the most favourable. This allows the reader to assess whether
the inference depends on the market-model specification.

Variants
--------
(a) contemporaneous_sp500  — market model with same-session S&P 500 return (the
    current implementation in run_event_study.py).
(b) lagged_sp500           — one-session-lagged S&P 500, the standard correction for
    non-overlapping trading hours (US closes after CSE; the move reaches Colombo the
    next session).
(c) constant_mean          — constant-mean return model (Brown & Warner 1985, mean-
    adjusted), no factor.
(d) zero_expected          — zero expected return; abnormal return equals the raw return.

Both series (return, volume) are run under every variant. The output has one row per
(variant, series, event_time).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.event_study import (
    EVENT_WINDOW,
    AbnormalPanel,
    build_abnormal_panel,
    event_study_table,
    average_cross_correlation,
    cumulative_abnormal,
    cross_sectional_t,
    bmp_t,
    kolari_pynnonen_t,
    corrado_rank_t,
)
from src.targets.abnormal_returns import (
    ESTIMATION_WINDOW,
    MIN_ESTIMATION_ROWS,
    align_market_factor,
)
from src.utils.artifact_store import artifact_file, save_frame

CUMULATION_START = 1


def load_daily() -> pd.DataFrame:
    """Daily ASPI return, log volume and SP500 factor, on the CSE calendar."""
    market_feats = pd.read_parquet(artifact_file("market_feats.parquet"))
    sp500 = pd.read_parquet(artifact_file("sp500.parquet"))
    daily = (market_feats[["date", "aspi_close", "trading_volume", "log_return"]]
             .sort_values("date").reset_index(drop=True))
    daily["sp500_log_return"] = align_market_factor(
        daily["date"], sp500["date"], sp500["sp500_log_return"])
    volume = daily["trading_volume"].where(daily["trading_volume"] > 0)
    daily["log_volume"] = np.log(volume)
    return daily


def build_zero_expected_panel(daily: pd.DataFrame, event_dates,
                               event_window=EVENT_WINDOW,
                               estimation_window: int = ESTIMATION_WINDOW,
                               min_estimation: int = MIN_ESTIMATION_ROWS,
                               date_col: str = "date",
                               value_col: str = "log_return") -> AbnormalPanel:
    """Build abnormal panel with zero expected return (AR = raw return).

    This is a degenerate case of the market model with alpha=0, beta=0.
    Residual std is estimated as the unconditional std of the estimation-window returns
    (ddof=1, consistent with the zero-parameter expected-return specification).
    """
    frame = daily.sort_values(date_col).reset_index(drop=True)
    dates = pd.to_datetime(frame[date_col])
    date_values = dates.to_numpy()
    values = frame[value_col].to_numpy(dtype=float)
    zeros = np.zeros(len(frame))  # zero factor for Corrado rank test compatibility

    k_values = np.arange(event_window[0], event_window[1] + 1)
    rows_ar, resid_std, n_est, dev, ss, residuals, kept = [], [], [], [], [], [], []
    window_start, window_end = [], []

    for event_id, event_date in enumerate(pd.to_datetime(pd.Series(list(event_dates)))):
        candidates = np.flatnonzero(date_values >= np.datetime64(event_date))
        if len(candidates) == 0 or candidates[0] == 0:
            continue
        position = int(candidates[0])

        last = position - 1
        first = max(0, last - estimation_window + 1)
        est_rows = np.arange(first, last + 1)
        usable = est_rows[np.isfinite(values[est_rows])]
        if len(usable) < min_estimation:
            continue

        # Zero expected return: alpha=0, beta=0
        resid = values[usable]  # residuals are raw returns themselves
        ddof = max(len(usable) - 1, 1)  # one degree of freedom for the mean (constrained to 0)
        sigma = float(np.sqrt(np.sum(resid ** 2) / ddof))

        event_rows = position + k_values - 1
        inside = (event_rows >= 0) & (event_rows < len(values))
        ar = np.full(len(k_values), np.nan)
        fd = np.full(len(k_values), np.nan)
        safe = event_rows[inside]
        ar[inside] = values[safe]   # abnormal = raw return (expected = 0)
        fd[inside] = zeros[safe]    # no factor deviation

        rows_ar.append(ar)
        dev.append(fd)
        resid_std.append(sigma)
        n_est.append(len(usable))
        ss.append(0.0)  # no factor SS
        residuals.append(pd.Series(resid, index=dates.iloc[usable].to_numpy()))
        window_start.append(date_values[safe[0]])
        window_end.append(date_values[safe[-1]])
        kept.append(event_id)

    if not rows_ar:
        raise ValueError("no event had a usable estimation window")
    return AbnormalPanel(np.vstack(rows_ar), k_values, np.array(resid_std),
                         np.array(n_est), np.vstack(dev), np.array(ss),
                         residuals, np.array(kept),
                         np.array(window_start), np.array(window_end))


def run_variant(daily: pd.DataFrame, event_dates, variant: str, series: str) -> pd.DataFrame:
    """Run one (variant, series) combination and return the event_study_table rows."""
    if series == "return":
        value_col = "log_return"
        factor_col = "sp500_log_return"
    elif series == "volume":
        value_col = "log_volume"
        factor_col = "sp500_log_return"
    else:
        raise ValueError(f"Unknown series: {series!r}")

    if variant == "contemporaneous_sp500":
        # Current implementation: market model with same-session S&P
        model = "market" if series == "return" else "mean"
        panel = build_abnormal_panel(daily, event_dates, model=model,
                                     value_col=value_col, factor_col=factor_col)

    elif variant == "lagged_sp500":
        # One-session lag of the S&P 500 (timezone correction)
        d = daily.copy()
        d["sp500_log_return"] = d["sp500_log_return"].shift(1)
        model = "market" if series == "return" else "mean"
        panel = build_abnormal_panel(d, event_dates, model=model,
                                     value_col=value_col, factor_col="sp500_log_return")

    elif variant == "constant_mean":
        # Mean-adjusted model: no factor for either series
        panel = build_abnormal_panel(daily, event_dates, model="mean",
                                     value_col=value_col, factor_col=factor_col)

    elif variant == "zero_expected":
        # Zero expected return: AR = raw return
        panel = build_zero_expected_panel(daily, event_dates, value_col=value_col)

    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    table = event_study_table(panel, start=CUMULATION_START, label=series)
    n_usable = panel.abnormal.shape[0]

    # Attach metadata columns
    table["variant"] = variant
    table["estimation_window"] = ESTIMATION_WINDOW
    table["min_obs"] = n_usable     # events with a usable estimation window
    table["gap_sessions"] = 0       # no gap between estimation window end and event

    return table


def main() -> None:
    daily = load_daily()
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))
    event_dates = pd.to_datetime(dataset["event_date"])
    print(f"{len(event_dates)} events")

    VARIANTS = (
        "contemporaneous_sp500",
        "lagged_sp500",
        "constant_mean",
        "zero_expected",
    )
    SERIES = ("return", "volume")

    OUTPUT_COLS = [
        "variant", "series", "event_time", "n_events",
        "caar", "ci_low", "ci_high",
        "p_bmp", "p_corrado", "p_kolari_pynnonen",
        "estimation_window", "min_obs", "gap_sessions",
    ]

    all_tables = []
    for variant in VARIANTS:
        for series in SERIES:
            print(f"  running variant={variant!r} series={series!r} ...")
            try:
                tbl = run_variant(daily, event_dates, variant, series)
                all_tables.append(tbl)
                settled = tbl[tbl.event_time == 5].iloc[0]
                print(f"    n_events={settled.n_events} CAAR(1,5)={settled.caar:+.4f} "
                      f"p_bmp={settled.p_bmp:.4f}")
            except Exception as exc:
                print(f"    ERROR: {exc}")

    combined = pd.concat(all_tables, ignore_index=True)

    # Keep only declared output columns; drop test statistics not in the spec
    for col in OUTPUT_COLS:
        if col not in combined.columns:
            combined[col] = np.nan

    out = combined[OUTPUT_COLS].copy()
    save_frame(out, "abnormal_return_variants",
               "Event study results under four expected-return model specifications "
               "(contemporaneous_sp500, lagged_sp500, constant_mean, zero_expected) "
               "for both return and volume series. All variants reported without selection.")
    print(f"\nwrote abnormal_return_variants.parquet ({len(out)} rows)")


if __name__ == "__main__":
    main()
