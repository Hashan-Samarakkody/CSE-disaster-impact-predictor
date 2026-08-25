import numpy as np
import pandas as pd

from src.data_pipeline.feature_eng import FeatureEngineer


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


def test_engineer_disaster_features_filters_biological_and_low_impact_events():
    disaster_df = pd.DataFrame(
        {
            "event_date": ["2024-01-05", "2024-01-10", "2024-01-20"],
            "disaster_type": ["Flood", "Epidemic", "Cyclone"],
            "financial_damage": [1_000_000, 5_000_000, 2_000_000],
            "population_affected": [5_000, 10_000, 900],
        }
    )

    fe = FeatureEngineer()
    out = fe.engineer_disaster_features(disaster_df)

    assert len(out) == 1
    assert out.iloc[0]["disaster_type"] == "Flood"
    assert "log_financial_damage" in out.columns
    assert "log_population_affected" in out.columns


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
