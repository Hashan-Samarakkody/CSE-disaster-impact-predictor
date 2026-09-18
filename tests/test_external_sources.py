"""Tests for the external data blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.external_sources import (
    DI_WINDOW,
    EXTERNAL_FEATURE_BLOCKS,
    HEAVY_RAIN_MM_3D,
    POWER_POINTS,
    build_desinventar_features,
    build_election_features,
    build_fx_features,
    build_hazard_features,
    extend_market_series,
)

EVENT = pd.Timestamp("2019-06-10")


@pytest.fixture
def power():
    days = pd.date_range("2019-01-01", "2019-12-31")
    return pd.concat([
        pd.DataFrame({"date": days, "district": name,
                      "precip_mm": np.where(days == EVENT, 100.0, 1.0),
                      "wind_max_ms": np.where(days == EVENT, 20.0, 5.0)})
        for name in POWER_POINTS])


@pytest.fixture
def desinventar():
    return pd.DataFrame({
        "date": [pd.Timestamp("2019-06-12"), pd.Timestamp("2019-06-30"),
                 pd.Timestamp("2019-06-11")],
        "evento": ["FLOOD", "FLOOD", "EPIDEMIC"],
        "name1": ["Colombo", "Galle", "Kandy"],
        "afectados": [1000.0, 5.0, 900.0], "vivdest": [10.0, 1.0, 7.0],
        "vivafec": [20.0, 2.0, 9.0], "muertos": [3.0, 0.0, 11.0]})


# ------------------------------------------------------------------ Block A: hazard

def test_hazard_accumulates_exactly_three_days(power):
    row = build_hazard_features([EVENT], power).iloc[0]
    # Event day 100mm plus the two prior days at 1mm each.
    assert row.hz_precip_max3d == pytest.approx(102.0)
    assert row.hz_precip_mean3d == pytest.approx(102.0)


def test_hazard_window_excludes_the_day_after(power):
    """t+1 must not enter the 3-day accumulation ending at t."""
    quiet = build_hazard_features([EVENT - pd.Timedelta(days=3)], power).iloc[0]
    assert quiet.hz_precip_max3d == pytest.approx(3.0)


def test_hazard_wet_district_count_uses_the_published_threshold(power):
    row = build_hazard_features([EVENT], power).iloc[0]
    assert row.hz_districts_wet == float(len(POWER_POINTS))
    assert HEAVY_RAIN_MM_3D == 50.0  # Sri Lanka Dept. of Meteorology advisory level
    calm = build_hazard_features([pd.Timestamp("2019-05-20")], power).iloc[0]
    assert calm.hz_districts_wet == 0.0


def test_hazard_spread_is_zero_when_districts_agree(power):
    assert build_hazard_features([EVENT], power).iloc[0].hz_precip_spread3d == pytest.approx(0.0)


def test_hazard_spread_is_positive_when_one_district_is_hit(power):
    localised = power.copy()
    hit = (localised.date == EVENT) & (localised.district != "Colombo")
    localised.loc[hit, "precip_mm"] = 1.0
    row = build_hazard_features([EVENT], localised).iloc[0]
    assert row.hz_precip_spread3d > 10.0
    assert row.hz_districts_wet == 1.0


def test_hazard_anomaly_baseline_stops_before_the_event_window(power):
    """The 30-day baseline ends at t-3, so the storm cannot inflate its own baseline."""
    row = build_hazard_features([EVENT], power).iloc[0]
    # Baseline is a flat 1mm/day -> 3mm over three days; 102/3 - 1 = 33.
    assert row.hz_precip_anom == pytest.approx(33.0)


# ------------------------------------------------------------ Block B: DesInventar

def test_desinventar_respects_the_declared_window(desinventar):
    assert DI_WINDOW == (-7, 14)
    row = build_desinventar_features([EVENT], desinventar).iloc[0]
    # +2 days is inside the window, +20 days is not.
    assert row.di_districts_hit == 1.0
    assert row.di_affected_log == pytest.approx(np.log1p(1000.0))


def test_desinventar_excludes_non_natural_event_types(desinventar):
    """The EPIDEMIC row sits inside the window but must not contribute."""
    row = build_desinventar_features([EVENT], desinventar).iloc[0]
    assert row.di_deaths_log == pytest.approx(np.log1p(3.0))  # not 3 + 11
    assert row.di_records == 1.0


def test_desinventar_flags_unmatched_events_instead_of_reporting_zero_damage(desinventar):
    """The defect this flag exists to prevent: a zero that means 'unknown'."""
    row = build_desinventar_features([pd.Timestamp("2019-01-01")], desinventar).iloc[0]
    assert row.di_available == 0.0
    assert row.di_records == 0.0
    assert row.di_affected_log == 0.0  # zero, but di_available says why


# -------------------------------------------------------------------- Block C: FX

@pytest.fixture
def fx():
    return pd.DataFrame({"date": pd.date_range("2019-01-01", periods=200),
                         "lkr_usd": np.linspace(180.0, 200.0, 200)})


def test_fx_never_reads_the_event_day(fx):
    """The exchange rate is a market price: same-day use would leak."""
    row = build_fx_features([EVENT], fx).iloc[0]
    prior = fx[fx.date < EVENT].lkr_usd.to_numpy()
    assert row.fx_logret_1 == pytest.approx(np.log(prior[-1] / prior[-2]))
    assert row.fx_logret_5 == pytest.approx(np.log(prior[-1] / prior[-6]))


def test_fx_returns_nan_without_enough_history(fx):
    row = build_fx_features([pd.Timestamp("2019-01-05")], fx).iloc[0]
    assert np.isnan(row.fx_logret_1) and np.isnan(row.fx_vol_30)


def test_fx_volatility_is_positive_on_a_noisy_series():
    rng = np.random.default_rng(0)
    noisy = pd.DataFrame({"date": pd.date_range("2019-01-01", periods=200),
                          "lkr_usd": 180 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))})
    assert build_fx_features([EVENT], noisy).iloc[0].fx_vol_30 > 0


# -------------------------------------------------------------- Block D: elections

@pytest.fixture
def polls():
    return pd.DataFrame({"date": [pd.Timestamp("2019-06-15"), pd.Timestamp("2020-08-05")],
                         "name": ["presidential", "parliamentary"]})


def test_election_distance_is_signed(polls):
    ahead = build_election_features([EVENT], polls).iloc[0]
    assert ahead.days_to_election == 5.0 and ahead.election_within_5d == 1.0
    behind = build_election_features([pd.Timestamp("2019-06-20")], polls).iloc[0]
    assert behind.days_to_election == -5.0 and behind.election_within_5d == 1.0


def test_election_flag_is_off_beyond_five_days(polls):
    row = build_election_features([pd.Timestamp("2019-06-25")], polls).iloc[0]
    assert row.election_within_5d == 0.0


# ------------------------------------------------------- Block E: series extension

def test_extension_never_overwrites_the_archive():
    archive = pd.DataFrame({"date": pd.date_range("2023-06-01", "2023-06-28"),
                            "aspi_close": 9000.0, "trading_volume": 1e6})
    ext = pd.DataFrame({"date": pd.date_range("2023-06-20", "2023-07-31"),
                        "aspi_close": 9500.0})
    merged = extend_market_series(archive, ext)
    overlap = merged[merged.date <= pd.Timestamp("2023-06-28")]
    assert (overlap.aspi_close == 9000.0).all()
    assert merged.date.max() == pd.Timestamp("2023-07-31")


def test_extension_leaves_volume_missing_rather_than_zero():
    """A zero volume would be a fabricated measurement; NaN is dropped per target."""
    archive = pd.DataFrame({"date": pd.date_range("2023-06-01", "2023-06-28"),
                            "aspi_close": 9000.0, "trading_volume": 1e6})
    ext = pd.DataFrame({"date": pd.date_range("2023-06-29", "2023-07-31"),
                        "aspi_close": 9500.0})
    merged = extend_market_series(archive, ext)
    assert merged[merged.date > pd.Timestamp("2023-06-28")].trading_volume.isna().all()


def test_extension_is_a_noop_when_the_source_adds_nothing():
    archive = pd.DataFrame({"date": pd.date_range("2023-06-01", "2023-06-28"),
                            "aspi_close": 9000.0, "trading_volume": 1e6})
    stale = pd.DataFrame({"date": pd.date_range("2023-06-01", "2023-06-10"),
                          "aspi_close": 1.0})
    assert len(extend_market_series(archive, stale)) == len(archive)


def test_declared_blocks_match_the_builders(power, desinventar, fx, polls):
    """The block registry drives the ablation, so it must not drift from the builders."""
    built = set()
    for frame in (build_hazard_features([EVENT], power),
                  build_desinventar_features([EVENT], desinventar),
                  build_fx_features([EVENT], fx),
                  build_election_features([EVENT], polls)):
        built |= set(frame.columns) - {"event_date"}
    declared = {c for cols in EXTERNAL_FEATURE_BLOCKS.values() for c in cols}
    assert built == declared


# --------------------------------------- pooled_frame alignment under skipped folds

def test_pooled_frame_uses_recorded_folds_when_one_is_skipped():
    """A skipped fold in the MIDDLE must not shift the event mapping."""
    from types import SimpleNamespace

    from src.evaluation.metrics import pooled_frame

    splits = [SimpleNamespace(train_index=np.arange(3), test_index=np.array([3, 4])),
              SimpleNamespace(train_index=np.arange(5), test_index=np.array([5, 6])),
              SimpleNamespace(train_index=np.arange(7), test_index=np.array([7, 8]))]
    dataset = pd.DataFrame({"event_date": pd.date_range("2020-01-01", periods=9),
                            "disaster_type": ["Flood"] * 9})

    # Fold 1 was skipped; folds 0 and 2 were stored.
    results = {"m": {"t": {"y_true": [np.array([1.0, 2.0]), np.array([7.0, 8.0])],
                           "y_pred": [np.array([1.1, 2.1]), np.array([7.1, 8.1])],
                           "folds": [0, 2]}}}
    frame = pooled_frame(results, "m", "t", splits, dataset)

    assert list(frame.fold) == [0, 0, 2, 2]
    assert list(frame.row_index) == [3, 4, 7, 8]
    # The last two rows must map to fold 2's events, not fold 1's.
    assert frame.event_date.iloc[-1] == pd.Timestamp("2020-01-09")


def test_pooled_frame_falls_back_to_offset_without_recorded_folds():
    """Results pickled before fold tracking existed must still map correctly."""
    from types import SimpleNamespace

    from src.evaluation.metrics import pooled_frame

    splits = [SimpleNamespace(train_index=np.arange(3), test_index=np.array([3, 4])),
              SimpleNamespace(train_index=np.arange(5), test_index=np.array([5, 6]))]
    dataset = pd.DataFrame({"event_date": pd.date_range("2020-01-01", periods=7),
                            "disaster_type": ["Flood"] * 7})
    # One fold stored, none recorded -> the stacked-model case, fold 0 forfeited.
    results = {"m": {"t": {"y_true": [np.array([5.0, 6.0])],
                           "y_pred": [np.array([5.1, 6.1])]}}}
    frame = pooled_frame(results, "m", "t", splits, dataset)
    assert list(frame.row_index) == [5, 6]
