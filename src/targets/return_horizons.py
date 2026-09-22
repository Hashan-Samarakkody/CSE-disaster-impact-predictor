"""Y1 horizon column names, and the two feature partitions.

The horizon values themselves are built once, by src/targets/event_targets.py, and
stored in dataset.parquet. This module only names those columns, so the repository
carries a single definition of Y1 and its declared horizon sensitivities.

Two orthogonal partitions of the same feature list live here. The first splits by data
source (market against disaster). The second splits by availability at the prediction
origin (real time against ex post), which is the one that decides whether a result is a
forecast or an ex post attribution. See docs/audit.md Part 8.3.
"""

from __future__ import annotations

from src.config.settings import (ASPI_PERCENTAGE_CHANGE, ASPI_SENSITIVITY_COLS,
                                 NON_FEATURE_COLS, TARGET_LABEL_END_DATE_COL)

HORIZONS = (1, 5, 10, 15, 20)
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


# availability at the prediction origin

# Demonstrably knowable at the prediction origin: market state, the disaster's own
# identity and timing, and the electoral calendar. These are the only columns a
# real-time forecaster could have had.
REALTIME_FEATURES = [
    "log_return", "lag_return_t-1", "lag_return_t-2", "lag_return_t-3", "lag_return_t-5",
    "price_to_sma_5", "price_to_ema_5", "price_to_sma_10", "price_to_ema_10",
    "price_to_sma_20", "price_to_ema_20",
    "rolling_std_5", "rolling_std_10", "rolling_std_20", "rolling_std_30",
    "squared_return", "garch_cond_vol", "garch_cond_vol_available",
    "vol_ratio_1_30", "vol_ratio_5_30", "vol_ratio_10_30", "vol_cv_30", "log_vol_change_1",
    "volume_features_available",
    "sp500_log_return", "fx_logret_1", "fx_logret_5", "fx_vol_30",
    "days_to_election", "election_within_5d",
    "disaster_Drought", "disaster_Flood", "disaster_Other", "disaster_Storm",
    "days_since_last_disaster", "disasters_trailing_365d",
    "date_is_exact",
]

# Finalised after the prediction origin. EM-DAT settles damage and casualty figures over
# days to months, DesInventar compiles local loss records afterwards, NASA POWER
# reanalysis publishes with a lag of several days, and the World Bank series are annual.
# Any result that depends on these is an ex post attribution, not a forecast.
EXPOST_FEATURES = [
    "financial_damage", "financial_damage_observed", "log_financial_damage",
    "population_affected", "log_population_affected",
    "total_deaths", "deaths_available", "no_homeless", "homeless_available",
    "mag_area_km2", "mag_wind_kph", "mag_area_available", "mag_wind_available",
    "damage_to_gdp", "log_damage_x_flood",
    "hz_precip_max3d", "hz_precip_mean3d", "hz_precip_spread3d", "hz_districts_wet",
    "hz_wind_max3d", "hz_precip_anom",
    "di_districts_hit", "di_affected_log", "di_houses_destroyed_log",
    "di_houses_damaged_log", "di_deaths_log", "di_records", "di_available",
    "gdp_growth_pct", "inflation_cpi_pct", "macro_available",
]


def availability_class(column: str) -> str:
    """Either "real_time" or "ex_post"; raises if the column belongs to neither."""
    if column in REALTIME_FEATURES:
        return "real_time"
    if column in EXPOST_FEATURES:
        return "ex_post"
    raise ValueError(
        f"feature column not assigned an availability class: {column!r} -- add it to "
        "REALTIME_FEATURES or EXPOST_FEATURES in src/targets/return_horizons.py. A "
        "silently unassigned column would change what real_time means between runs.")


def information_sets(available_cols) -> dict[str, list[str]]:
    """Five information sets over the columns actually present: the three source sets
    ("market_only", "disaster_only", "combined") and the two availability sets
    ("real_time", "ex_post"). Raises if a column is missing from either partition."""
    available = list(available_cols)
    market = [c for c in MARKET_FEATURES if c in available]
    disaster = [c for c in DISASTER_FEATURES if c in available]
    unassigned = set(available) - set(MARKET_FEATURES) - set(DISASTER_FEATURES)
    if unassigned:
        raise ValueError(
            "feature columns not assigned to an information set: "
            f"{sorted(unassigned)} -- assign each to MARKET_FEATURES or DISASTER_FEATURES "
            "in src/targets/return_horizons.py before running the Y1 grid.")
    unclassified = set(available) - set(REALTIME_FEATURES) - set(EXPOST_FEATURES)
    if unclassified:
        raise ValueError(
            "feature columns not assigned an availability class: "
            f"{sorted(unclassified)} -- assign each to REALTIME_FEATURES or "
            "EXPOST_FEATURES in src/targets/return_horizons.py before running the Y1 grid.")
    return {"market_only": market, "disaster_only": disaster, "combined": market + disaster,
            "real_time": [c for c in REALTIME_FEATURES if c in available],
            "ex_post": [c for c in EXPOST_FEATURES if c in available]}


if __name__ == "__main__":
    # The horizon columns must be the ones the protocol freezes, so the grid script and
    # dataset.parquet cannot drift apart.
    assert horizon_col(5) == ASPI_PERCENTAGE_CHANGE
    assert PRINCIPAL_HORIZON == 5 and HORIZONS[0] == 1
    assert [horizon_col(h) for h in HORIZONS if h != PRINCIPAL_HORIZON] == ASPI_SENSITIVITY_COLS
    assert horizon_end_col(5) == "Y1_horizon_end_date"
    assert horizon_end_col(1) == "Y1_1D_horizon_end_date"
    assert horizon_end_col(10) == "Y1_10D_horizon_end_date"
    # Every horizon column and its label-end date must be a declared non-feature: this is
    # exactly how the 15 and 20 session returns leaked into the model once before.
    for _h in HORIZONS:
        assert horizon_col(_h) in NON_FEATURE_COLS, _h
        assert horizon_end_col(_h) in NON_FEATURE_COLS, _h

    sets = information_sets(MARKET_FEATURES + DISASTER_FEATURES)
    assert set(sets) == {"market_only", "disaster_only", "combined", "real_time", "ex_post"}
    assert set(sets["combined"]) == set(MARKET_FEATURES) | set(DISASTER_FEATURES)
    assert not (set(sets["market_only"]) & set(sets["disaster_only"]))

    # The availability partition must cover the same columns, exactly once each.
    assert not (set(REALTIME_FEATURES) & set(EXPOST_FEATURES))
    assert (set(REALTIME_FEATURES) | set(EXPOST_FEATURES)
            == set(MARKET_FEATURES) | set(DISASTER_FEATURES))
    assert set(sets["real_time"]) | set(sets["ex_post"]) == set(sets["combined"])
    assert availability_class("sp500_log_return") == "real_time"
    assert availability_class("gdp_growth_pct") == "ex_post"
    assert availability_class("hz_precip_max3d") == "ex_post"

    for bad in (["a_column_nobody_declared"],):
        try:
            information_sets(bad)
        except ValueError as exc:
            assert "not assigned" in str(exc)
        else:
            raise AssertionError("unassigned column must raise")
    try:
        availability_class("a_column_nobody_declared")
    except ValueError as exc:
        assert "availability class" in str(exc)
    else:
        raise AssertionError("unclassified column must raise")

    print("return_horizons.py self-check passed")
