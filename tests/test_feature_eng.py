import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.feature_eng import FeatureEngineer, FeatureEngineeringConfig


def test_engineer_market_features_generates_lags_and_no_lookahead_rolling_std():
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    prices = np.linspace(100.0, 140.0, 40)
    volumes = np.linspace(1000.0, 2000.0, 40)
    df = pd.DataFrame({"date": dates, "aspi_close": prices, "trading_volume": volumes})

    fe = FeatureEngineer()
    out = fe.engineer_market_features(df)

    assert {"lag_return_t-1", "lag_return_t-2", "lag_return_t-3", "lag_return_t-5"}.issubset(out.columns)
    assert "rolling_std_5" in out.columns

    idx = 10
    expected = out["log_return"].shift(1).iloc[idx - 4 : idx + 1].std()
    assert np.isclose(out.loc[idx, "rolling_std_5"], expected, equal_nan=True)


def test_volume_features_use_only_information_available_before_the_row():
    # The whole point of the volume block is that it is the Y2 functional evaluated one
    # trading day early. If any term touched row t's own volume it would leak the target,
    # so the test perturbs the LAST row's volume and asserts nothing before it moves.
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    prices = np.linspace(100.0, 160.0, 60)
    volumes = np.linspace(1000.0, 5000.0, 60)
    df = pd.DataFrame({"date": dates, "aspi_close": prices, "trading_volume": volumes})

    fe = FeatureEngineer()
    out = fe.engineer_market_features(df)

    vol_cols = ["vol_ratio_1_30", "vol_ratio_5_30", "vol_ratio_10_30",
                "vol_cv_30", "log_vol_change_1"]
    assert set(vol_cols).issubset(out.columns)

    bumped = df.copy()
    bumped.loc[bumped.index[-1], "trading_volume"] *= 100.0
    out_bumped = fe.engineer_market_features(bumped)

    # Every row is unchanged -- including the last one, whose features look back to t-1.
    pd.testing.assert_frame_equal(out[vol_cols], out_bumped[vol_cols])

    # And the ratio is the real quantity, not a placeholder: at row t it compares the
    # t-1 volume against the mean of the 30 sessions ending at t-1.
    idx = 45
    shifted = df["trading_volume"].shift(1)
    expected = shifted.iloc[idx] / shifted.iloc[idx - 29 : idx + 1].mean() - 1.0
    assert np.isclose(out.loc[idx, "vol_ratio_1_30"], expected)


def test_volume_features_are_absent_when_no_volume_column_is_supplied():
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    df = pd.DataFrame({"date": dates, "aspi_close": np.linspace(100.0, 140.0, 40)})

    out = FeatureEngineer().engineer_market_features(df)

    assert not [c for c in out.columns if c.startswith("vol_")]


def test_engineer_disaster_features_filters_biological_and_low_impact_events():
    # Three Floods so the surviving type clears the rare-type pooling threshold and this
    # test keeps testing what it is named for: the Epidemic row is dropped as biological,
    # and the 900-affected row is dropped as below the >=1000 threshold.
    disaster_df = pd.DataFrame(
        {
            "event_date": ["2024-01-05", "2024-01-10", "2024-01-20", "2024-02-01", "2024-02-10"],
            "disaster_type": ["Flood", "Epidemic", "Cyclone", "Flood", "Flood"],
            "financial_damage": [1_000_000, 5_000_000, 2_000_000, 3_000_000, 4_000_000],
            "population_affected": [5_000, 10_000, 900, 2_000, 1_000],
        }
    )

    # Pinned to the pre-registered >=1000 threshold. The config default moved to 700 on
    # 2026-09-12, under which the 900-affected Cyclone row survives -- this test is named
    # for the biological filter, so it keeps its original threshold rather than quietly
    # changing what it asserts. The 700 default is covered by
    # test_inclusion_threshold_is_read_from_config.
    fe = FeatureEngineer(FeatureEngineeringConfig(min_affected=1000))
    out = fe.engineer_disaster_features(disaster_df)

    assert len(out) == 3
    assert set(out["disaster_type"]) == {"Flood"}
    assert "log_financial_damage" in out.columns
    assert "log_population_affected" in out.columns
    # The 1,000-affected event is retained: the threshold is >= 1000, matching the
    # filter the thesis and the notebook both state.
    assert 1_000 in set(out["population_affected"])


def test_engineer_disaster_features_pools_types_with_too_few_events():
    # Cyclone appears once. A one-hot column with a single positive case is a
    # memorisation key in-sample and all-zero out-of-sample, so it is pooled into
    # "Other" rather than given its own indicator.
    disaster_df = pd.DataFrame(
        {
            "event_date": ["2024-01-05", "2024-01-10", "2024-01-20", "2024-02-01"],
            "disaster_type": ["Flood", "Flood", "Flood", "Cyclone"],
            "financial_damage": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
            "population_affected": [5_000, 6_000, 7_000, 8_000],
        }
    )

    out = FeatureEngineer().engineer_disaster_features(disaster_df)

    assert set(out["disaster_type"]) == {"Flood", "Other"}
    assert "disaster_Other" in out.columns
    assert "disaster_Cyclone" not in out.columns


def test_build_targets_caps_recovery_days_at_90():
    dates = pd.date_range("2024-01-01", periods=140, freq="B")
    prices = np.concatenate([np.full(20, 100.0), np.linspace(80, 95, 120)])
    volumes = np.linspace(1000.0, 2000.0, len(dates))
    market_df = pd.DataFrame({"date": dates, "aspi_close": prices, "trading_volume": volumes})
    disaster_df = pd.DataFrame({"event_date": [dates[20]]})

    fe = FeatureEngineer()
    targets = fe.build_targets(market_df, disaster_df)

    assert len(targets) == 1
    assert targets.iloc[0]["Y3_recovery_days"] == 90.0


# ------------------------------------------------- cumulative event-window returns

def test_car_targets_accumulate_from_the_pre_event_close():
    """Y1_car_k = ln(P[pos+k] / P[pos-1]) -- same denominator as Y1, later numerator.

    Getting the denominator wrong (using P[pos] instead of P[pos-1]) would drop the
    event day itself out of the window, which is the whole point of a [0,+k] window.
    """
    days = pd.bdate_range("2020-01-01", periods=60)
    # A flat series with one known jump lets every target be computed by hand.
    price = np.full(len(days), 100.0)
    price[10:] = 110.0                      # event lands on index 10
    market = pd.DataFrame({"date": days, "aspi_close": price, "trading_volume": 1e6})
    events = pd.DataFrame({"event_date": [days[10]]})

    fe = FeatureEngineer(FeatureEngineeringConfig())
    t = fe.build_targets(market, events).iloc[0]

    expected = float(np.log(110.0 / 100.0))
    assert t.Y1_aspi_log_return == pytest.approx(expected)
    # Price is flat at 110 after the jump, so 5- and 10-day CARs equal the day-0 return.
    assert t.Y1_car_5 == pytest.approx(expected)
    assert t.Y1_car_10 == pytest.approx(expected)


def test_car_targets_capture_drift_the_day0_return_misses():
    days = pd.bdate_range("2020-01-01", periods=60)
    price = np.full(len(days), 100.0)
    price[10] = 99.0            # small dip on the event day
    price[11:] = 90.0           # the real damage shows up afterwards
    market = pd.DataFrame({"date": days, "aspi_close": price, "trading_volume": 1e6})
    events = pd.DataFrame({"event_date": [days[10]]})

    t = FeatureEngineer(FeatureEngineeringConfig()).build_targets(market, events).iloc[0]
    assert t.Y1_aspi_log_return == pytest.approx(np.log(99.0 / 100.0))
    assert t.Y1_car_5 == pytest.approx(np.log(90.0 / 100.0))
    # The window return is a much larger loss than the single day showed.
    assert t.Y1_car_5 < t.Y1_aspi_log_return


def test_car_is_nan_rather_than_a_truncated_window():
    """A partial accumulation must never be reported as a full one."""
    days = pd.bdate_range("2020-01-01", periods=14)
    market = pd.DataFrame({"date": days, "aspi_close": np.linspace(100, 120, len(days)),
                           "trading_volume": 1e6})
    events = pd.DataFrame({"event_date": [days[10]]})   # only 3 rows remain after it
    t = FeatureEngineer(FeatureEngineeringConfig()).build_targets(market, events).iloc[0]
    assert np.isnan(t.Y1_car_5) and np.isnan(t.Y1_car_10)


def test_inclusion_threshold_is_read_from_config():
    """The 1000 -> 700 change must live in one declared place, not a literal."""
    assert FeatureEngineeringConfig().min_affected == 700
    events = pd.DataFrame({
        "event_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
        "disaster_type": ["Flood"] * 3,
        "financial_damage": [0.0] * 3,
        "population_affected": [650.0, 800.0, 1500.0],
    })
    kept = FeatureEngineer(FeatureEngineeringConfig()).engineer_disaster_features(events)
    assert len(kept) == 2                       # 650 excluded, 800 and 1500 kept
    strict = FeatureEngineer(FeatureEngineeringConfig(min_affected=1000))
    assert len(strict.engineer_disaster_features(events)) == 1
