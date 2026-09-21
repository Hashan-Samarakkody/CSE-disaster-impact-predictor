"""Y1 horizon column names, and the market/disaster feature partition.

The horizon values themselves are built once, by src/targets/event_targets.py, and
stored in dataset.parquet. This module only names those columns, so the repository
carries a single definition of Y1 and its pre-registered horizon sensitivities.
"""

from __future__ import annotations

from src.config.settings import (ASPI_PERCENTAGE_CHANGE, ASPI_SENSITIVITY_COLS,
                                 TARGET_LABEL_END_DATE_COL)

HORIZONS = (5, 10, 15, 20)
PRINCIPAL_HORIZON = 5


def horizon_col(h: int) -> str:
    """Dataset column holding 100 * ln(Ph / P0) for horizon h."""
    if h == PRINCIPAL_HORIZON:
        return ASPI_PERCENTAGE_CHANGE
    return f"Y1_ASPI_{h}D_Forward_LogReturn_Pct"


def horizon_end_col(h: int) -> str:
    """Dataset column holding the session on which horizon h closes."""
    if h == PRINCIPAL_HORIZON:
        return TARGET_LABEL_END_DATE_COL[ASPI_PERCENTAGE_CHANGE]
    return f"Y1_{h}D_horizon_end_date"


# information sets

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
            "in src/targets/return_horizons.py before running the Y1 grid.")
    return {"market_only": market, "disaster_only": disaster, "combined": market + disaster}


if __name__ == "__main__":
    # The horizon columns must be the ones the protocol freezes, so the grid script and
    # dataset.parquet cannot drift apart.
    assert horizon_col(5) == ASPI_PERCENTAGE_CHANGE
    assert [horizon_col(h) for h in (10, 15, 20)] == ASPI_SENSITIVITY_COLS
    assert horizon_end_col(5) == "Y1_horizon_end_date"
    assert horizon_end_col(10) == "Y1_10D_horizon_end_date"

    sets = information_sets(MARKET_FEATURES + DISASTER_FEATURES)
    assert set(sets["combined"]) == set(MARKET_FEATURES) | set(DISASTER_FEATURES)
    assert not (set(sets["market_only"]) & set(sets["disaster_only"]))
    try:
        information_sets(["a_column_nobody_declared"])
    except ValueError as exc:
        assert "not assigned" in str(exc)
    else:
        raise AssertionError("unassigned column must raise")

    print("return_horizons.py self-check passed")
