"""Y1 multi-horizon target construction and the market/disaster information partition."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.targets.return_horizons import (DISASTER_FEATURES, HORIZONS, MARKET_FEATURES,
                                         build_horizon_targets, horizon_col,
                                         horizon_end_col, information_sets)
from src.utils.artifact_store import artifact_file



@pytest.fixture
def synthetic_market():
    sessions = pd.date_range("2020-01-01", periods=60, freq="B")
    prices = 100.0 * np.exp(np.arange(60) * 0.01)     # +1% log return per session
    return pd.DataFrame({"date": sessions, "aspi_close": prices}), sessions


def test_horizons_are_trading_sessions_not_calendar_days(synthetic_market):
    """The series is business-day only, so a Saturday event must align to the next Monday
    and every horizon must be counted in sessions from there."""
    market, sessions = synthetic_market
    event = pd.Timestamp("2020-01-11")            # a Saturday
    out = build_horizon_targets(market, [event])
    t = int(np.flatnonzero(sessions >= event)[0])
    assert sessions[t].weekday() == 0
    for h in HORIZONS:
        assert out[horizon_end_col(h)].iloc[0] == sessions[t + h]


def test_formula_is_log_return_from_pre_event_close(synthetic_market):
    """Y_h = 100 ln(P_{t+h} / P_{t-1}), baselined on the LAST PRE-EVENT close, so the
    window spans h+1 sessions of a +1%/session path."""
    market, sessions = synthetic_market
    out = build_horizon_targets(market, [sessions[20]])
    for h in HORIZONS:
        assert out[horizon_col(h)].iloc[0] == pytest.approx((h + 1) * 1.0, abs=1e-9)


def test_short_window_is_nan_never_truncated(synthetic_market):
    market, sessions = synthetic_market
    out = build_horizon_targets(market, [sessions[-3]])
    assert np.isnan(out[horizon_col(20)].iloc[0])
    assert pd.isna(out[horizon_end_col(20)].iloc[0])


def test_event_before_first_session_is_nan(synthetic_market):
    market, sessions = synthetic_market
    out = build_horizon_targets(market, [sessions[0]])      # no P_{t-1} exists
    assert out[[horizon_col(h) for h in HORIZONS]].isna().all(axis=None)


def test_horizon_end_dates_are_monotone_in_h(synthetic_market):
    market, sessions = synthetic_market
    out = build_horizon_targets(market, [sessions[10]])
    ends = [out[horizon_end_col(h)].iloc[0] for h in HORIZONS]
    assert ends == sorted(ends) and len(set(ends)) == len(ends)


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


def test_real_market_series_produces_finite_targets_for_most_events():
    """Smoke test against the cached artifacts: the alignment rule must actually resolve
    on the real CSE calendar, not just a synthetic business-day index."""
    if not (artifact_file("market.parquet")).exists():
        pytest.skip("artifacts/market.parquet missing")
    market = pd.read_parquet(artifact_file("market.parquet"))
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))
    out = build_horizon_targets(market, pd.to_datetime(dataset["event_date"]))
    assert len(out) == len(dataset)
    for h in HORIZONS:
        assert out[horizon_col(h)].notna().mean() > 0.9, h
    # h=5 must reproduce the frozen pipeline's own Y1 column, which uses the identical
    # formula and alignment, a drift here means one of the two is wrong.
    frozen = dataset["Y1_ASPI_5D_Forward_LogReturn_Pct"].to_numpy(float)
    both = np.isfinite(frozen) & out[horizon_col(5)].notna().to_numpy()
    np.testing.assert_allclose(out[horizon_col(5)].to_numpy(float)[both], frozen[both],
                               atol=1e-9)
