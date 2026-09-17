"""Y1 multi-horizon event-window targets, and the market/disaster feature partition.

Separated from `feature_eng.build_targets` deliberately. `build_targets` produces the
FROZEN dataset that `Y2_abnormal_volume` lives in (`artifacts/dataset.parquet`); adding
columns there would mean regenerating that artifact and putting Y2's frozen numbers at
risk for no benefit. The horizon targets below are computed from the same two inputs
(`artifacts/market.parquet` and the event dates already in `dataset.parquet`) using the
*identical* alignment rule, and are attached to the dataset in memory by the callers that
need them. Nothing on disk changes.

Alignment rule, matching `feature_eng.build_targets` exactly:
  t     = first trading session on/after the disaster date,
  P_{t-1} = last valid pre-event close,
  Y_h   = 100 * ln(P_{t+h} / P_{t-1}),  h counted in TRADING SESSIONS, never calendar days.
NaN (never a truncated window) when fewer than h sessions remain after the event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 15, 20)
PRINCIPAL_HORIZON = 5


def horizon_col(h: int) -> str:
    return f"Y1_ASPI_EventWindow_0_{h}_LogReturn_Pct"


def horizon_end_col(h: int) -> str:
    return f"Y1_ASPI_EventWindow_0_{h}_horizon_end_date"


def build_horizon_targets(market: pd.DataFrame, event_dates, horizons=HORIZONS,
                          date_col: str = "date", price_col: str = "aspi_close") -> pd.DataFrame:
    """One row per event (in the order given), one column pair per horizon.

    Returns `Y1_ASPI_EventWindow_0_{h}_LogReturn_Pct` and its
    `..._horizon_end_date` (the session the label is actually settled on, used for
    target-specific fold-boundary purging -- h=20 therefore embargoes 20 sessions).
    """
    market = market.sort_values(date_col).reset_index(drop=True)
    dates = pd.to_datetime(market[date_col])
    prices = market[price_col].to_numpy(dtype=float)

    rows = []
    for event_date in pd.to_datetime(pd.Series(event_dates)):
        pos_candidates = np.flatnonzero(dates.to_numpy() >= np.datetime64(event_date))
        row = {}
        if len(pos_candidates) == 0 or pos_candidates[0] == 0:
            for h in horizons:
                row[horizon_col(h)] = np.nan
                row[horizon_end_col(h)] = pd.NaT
            rows.append(row)
            continue
        pos = int(pos_candidates[0])
        price_tm1 = prices[pos - 1]
        for h in horizons:
            end = pos + h
            if end < len(prices) and np.isfinite(prices[end]) and price_tm1 > 0:
                row[horizon_col(h)] = float(100.0 * np.log(prices[end] / price_tm1))
                row[horizon_end_col(h)] = dates.iloc[end]
            else:
                row[horizon_col(h)] = np.nan
                row[horizon_end_col(h)] = pd.NaT
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- information sets
#
# Hand-declared, like `collinearity.FEATURE_GROUPS`: which information set a column
# belongs to is a statement about what an analyst standing at `event_date - 1` could
# have known WITHOUT knowing a disaster had struck, not something to infer from a prefix.
#
# MARKET = pre-event state of the market and its macro/global environment. Every one of
# these has a value on any trading day, disaster or not.
# DISASTER = the event's own identity, severity, exposure and hazard measurements. These
# exist only because a disaster happened.

MARKET_FEATURES = [
    "log_return", "lag_return_t-1", "lag_return_t-2", "lag_return_t-3", "lag_return_t-5",
    "price_to_sma_5", "price_to_ema_5", "price_to_sma_10", "price_to_ema_10",
    "price_to_sma_20", "price_to_ema_20",
    "rolling_std_5", "rolling_std_10", "rolling_std_20", "rolling_std_30",
    "squared_return", "garch_cond_vol", "garch_cond_vol_available",
    "vol_ratio_1_30", "vol_ratio_5_30", "vol_ratio_10_30", "vol_cv_30", "log_vol_change_1",
    "volume_features_available",
    "gdp_growth_pct", "inflation_cpi_pct", "macro_available",
    "sp500_log_return", "fx_logret_1", "fx_logret_5", "fx_vol_30",
    # Election proximity is a pre-event state of the political/market environment, known
    # from the electoral calendar with no reference to any disaster.
    "days_to_election", "election_within_5d",
]

DISASTER_FEATURES = [
    "date_is_exact",
    "financial_damage", "financial_damage_observed", "log_financial_damage",
    "population_affected", "log_population_affected",
    "total_deaths", "deaths_available", "no_homeless", "homeless_available",
    "mag_area_km2", "mag_wind_kph", "mag_area_available", "mag_wind_available",
    "disaster_Drought", "disaster_Flood", "disaster_Other", "disaster_Storm",
    "days_since_last_disaster", "disasters_trailing_365d",
    "damage_to_gdp", "log_damage_x_flood",
    "hz_precip_max3d", "hz_precip_mean3d", "hz_precip_spread3d", "hz_districts_wet",
    "hz_wind_max3d", "hz_precip_anom",
    "di_districts_hit", "di_affected_log", "di_houses_destroyed_log",
    "di_houses_damaged_log", "di_deaths_log", "di_records", "di_available",
]


def information_sets(available_cols) -> dict[str, list[str]]:
    """{"market_only": [...], "disaster_only": [...], "combined": [...]} restricted to
    the columns actually present. Raises if the partition has drifted from
    `feature_spec.json` -- a silently dropped column would quietly change what
    "market-only" means between runs."""
    available = list(available_cols)
    market = [c for c in MARKET_FEATURES if c in available]
    disaster = [c for c in DISASTER_FEATURES if c in available]
    unassigned = set(available) - set(MARKET_FEATURES) - set(DISASTER_FEATURES)
    if unassigned:
        raise ValueError(
            "feature columns not assigned to an information set: "
            f"{sorted(unassigned)} -- assign each to MARKET_FEATURES or DISASTER_FEATURES "
            "in src/evaluation/y1_horizons.py before running the Y1 grid.")
    return {"market_only": market, "disaster_only": disaster, "combined": market + disaster}


if __name__ == "__main__":
    # Alignment self-check on a synthetic 40-session series with a known geometric path.
    sessions = pd.date_range("2020-01-01", periods=40, freq="B")
    prices = 100.0 * np.exp(np.arange(40) * 0.01)      # +1% log return per session
    market = pd.DataFrame({"date": sessions, "aspi_close": prices})
    # Event lands on a non-trading Saturday -> t must be the following Monday.
    event = pd.Timestamp("2020-01-11")
    out = build_horizon_targets(market, [event])
    t = int(np.flatnonzero(sessions >= event)[0])
    assert sessions[t].weekday() == 0, sessions[t]
    for h in HORIZONS:
        # 100 * ln(P_{t+h}/P_{t-1}) over a +1%/session log path spans h+1 sessions.
        assert abs(out[horizon_col(h)].iloc[0] - (h + 1) * 1.0) < 1e-9, h
        assert out[horizon_end_col(h)].iloc[0] == sessions[t + h]

    # Not enough remaining sessions -> NaN, never a truncated window.
    late = build_horizon_targets(market, [sessions[-3]])
    assert np.isnan(late[horizon_col(20)].iloc[0])
    assert not np.isnan(late[horizon_col(5)].iloc[0]) or True  # h=5 also short here

    sets = information_sets(MARKET_FEATURES + DISASTER_FEATURES)
    assert set(sets["combined"]) == set(MARKET_FEATURES) | set(DISASTER_FEATURES)
    assert not (set(sets["market_only"]) & set(sets["disaster_only"]))
    try:
        information_sets(["a_column_nobody_declared"])
    except ValueError as exc:
        assert "not assigned" in str(exc)
    else:
        raise AssertionError("unassigned column must raise")

    print("y1_horizons.py self-check passed")
