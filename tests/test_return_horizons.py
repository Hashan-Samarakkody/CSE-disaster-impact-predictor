"""Y1 horizon column naming and the market/disaster information partition."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config.settings import (ASPI_PERCENTAGE_CHANGE, ASPI_SENSITIVITY_COLS,
                                 NON_FEATURE_COLS)
from src.targets.return_horizons import (DISASTER_FEATURES, EXPOST_FEATURES, HORIZONS,
                                         MARKET_FEATURES, PRINCIPAL_HORIZON,
                                         REALTIME_FEATURES, availability_class,
                                         horizon_col, horizon_end_col, information_sets)
from src.utils.artifact_store import artifact_file


SENSITIVITY_HORIZONS = (1, 10, 15, 20)


def test_horizon_names_are_the_frozen_protocol_columns():
    """One definition of Y1 only. If these names drift from settings, the grid script
    silently starts modelling a different target from the one in dataset.parquet."""
    assert horizon_col(5) == ASPI_PERCENTAGE_CHANGE
    assert [horizon_col(h) for h in SENSITIVITY_HORIZONS] == ASPI_SENSITIVITY_COLS
    assert horizon_end_col(5) == "Y1_horizon_end_date"
    assert [horizon_end_col(h) for h in SENSITIVITY_HORIZONS] == [
        f"Y1_{h}D_horizon_end_date" for h in SENSITIVITY_HORIZONS]


def test_one_session_horizon_is_declared_and_can_never_be_a_predictor():
    """T4. The one-session return is a robustness test, never primary, and it must be a
    registered non-feature: this is exactly how the 15 and 20 session returns leaked."""
    assert HORIZONS == (1, 5, 10, 15, 20)
    assert PRINCIPAL_HORIZON == 5
    for h in HORIZONS:
        assert horizon_col(h) in NON_FEATURE_COLS, h
        assert horizon_end_col(h) in NON_FEATURE_COLS, h


def test_availability_partition_covers_every_feature_exactly_once():
    """T1. real_time and ex_post must tile the same column set the source partition does,
    with no overlap, or "real-time" quietly means something different between runs."""
    assert not set(REALTIME_FEATURES) & set(EXPOST_FEATURES)
    assert (set(REALTIME_FEATURES) | set(EXPOST_FEATURES)
            == set(MARKET_FEATURES) | set(DISASTER_FEATURES))

    spec = json.loads((artifact_file("feature_spec.json")).read_text(encoding="utf-8"))
    sets = information_sets(spec["FEATURE_COLS"])
    assert set(sets) == {"market_only", "disaster_only", "combined", "real_time", "ex_post"}
    assert not set(sets["real_time"]) & set(sets["ex_post"])
    assert set(sets["real_time"]) | set(sets["ex_post"]) == set(spec["FEATURE_COLS"])


def test_ex_post_set_holds_the_variables_that_settle_after_the_event():
    """The partition is only useful if it is right about the specific columns the review
    named: World Bank macro, EM-DAT severity, DesInventar and NASA POWER."""
    for column in ("gdp_growth_pct", "inflation_cpi_pct", "macro_available",
                   "financial_damage", "total_deaths", "population_affected",
                   "di_affected_log", "hz_precip_max3d", "mag_wind_kph"):
        assert availability_class(column) == "ex_post", column
    for column in ("log_return", "lag_return_t-1", "rolling_std_30", "sp500_log_return",
                   "fx_logret_1", "days_to_election", "disaster_Flood",
                   "days_since_last_disaster", "date_is_exact"):
        assert availability_class(column) == "real_time", column


def test_unclassified_column_raises_rather_than_silently_defaulting():
    """T1 guard. A column missing from the availability partition must raise the same way
    an unassigned source column already does."""
    with pytest.raises(ValueError, match="availability class"):
        availability_class("a_column_nobody_declared")
    with pytest.raises(ValueError, match="not assigned"):
        information_sets(["a_column_nobody_declared"])


def test_feature_spec_records_the_availability_class_of_every_feature():
    """T1 acceptance: the persisted spec, not just the source, carries the class."""
    spec = json.loads((artifact_file("feature_spec.json")).read_text(encoding="utf-8"))
    recorded = spec["AVAILABILITY_CLASS"]
    assert set(recorded) == set(spec["FEATURE_COLS"])
    assert set(recorded.values()) == {"real_time", "ex_post"}
    for column, klass in recorded.items():
        assert klass == availability_class(column), column


def test_every_horizon_column_exists_in_the_built_dataset():
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))
    for h in HORIZONS:
        assert horizon_col(h) in dataset.columns, h
        assert horizon_end_col(h) in dataset.columns, h
        assert dataset[horizon_col(h)].notna().mean() > 0.9, h


def test_horizon_values_follow_the_protocol_alignment():
    """Ph = market[position + h - 1] and P0 = market[position - 1], checked against the
    raw market series rather than against the code that wrote the column."""
    market = pd.read_parquet(artifact_file("market.parquet")).sort_values("date")
    market = market.reset_index(drop=True)
    dates = pd.to_datetime(market["date"]).to_numpy()
    prices = market["aspi_close"].to_numpy(float)
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))

    for row in dataset.itertuples():
        position = int(np.flatnonzero(dates >= np.datetime64(row.event_date))[0])
        assert prices[position - 1] == pytest.approx(row.P0)
        for h in HORIZONS:
            value = getattr(row, horizon_col(h))
            if not np.isfinite(value):
                continue
            want = 100.0 * np.log(prices[position + h - 1] / prices[position - 1])
            assert value == pytest.approx(want, abs=1e-9), (row.event_date, h)


def test_horizon_end_dates_are_monotone_in_h():
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))
    ends = dataset[[horizon_end_col(h) for h in HORIZONS]].dropna()
    assert len(ends) > 0
    increasing = ends.apply(lambda r: list(r) == sorted(r) and len(set(r)) == len(r), axis=1)
    assert increasing.all()


def test_information_sets_partition_every_feature_column():
    """Every column in the frozen feature spec must be assigned to exactly one side, or
    'market-only' silently means something different between runs."""
    spec = json.loads((artifact_file("feature_spec.json")).read_text(encoding="utf-8"))
    sets = information_sets(spec["FEATURE_COLS"])
    assert not set(sets["market_only"]) & set(sets["disaster_only"])
    assert set(sets["combined"]) == set(spec["FEATURE_COLS"])
    assert len(sets["combined"]) == len(spec["FEATURE_COLS"])


def test_unassigned_column_raises_rather_than_silently_dropping():
    with pytest.raises(ValueError, match="not assigned"):
        information_sets(["a_column_nobody_declared"])


def test_no_disaster_column_leaks_into_the_market_set():
    """The market set defines the 'what the market already knew' baseline; a severity or
    hazard column in it would make that comparison meaningless."""
    leak_prefixes = ("disaster_", "hz_", "di_", "mag_", "log_damage", "financial_damage")
    assert not [c for c in MARKET_FEATURES if c.startswith(leak_prefixes)]
    assert "population_affected" not in MARKET_FEATURES
    assert "total_deaths" not in MARKET_FEATURES
    assert set(MARKET_FEATURES).isdisjoint(DISASTER_FEATURES)
