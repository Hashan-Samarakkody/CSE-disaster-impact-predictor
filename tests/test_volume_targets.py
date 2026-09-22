"""T5: Y2 response-window sensitivities, the percentage form, and the outlier variant.

The property that matters most is the constraint: volume is unavailable for the 2000
archive year and for events after 2023, and a missing label must stay missing. Zero-filling
would assert that turnover sat exactly at its baseline when in fact it is unknown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.settings import (NON_FEATURE_COLS, VOLUME_DERIVED_COLS,
                                 VOLUME_SENSITIVITY_COLS)
from src.targets.event_targets import (VOLUME_SENSITIVITY_WINDOWS, abnormal_volume_percent,
                                       build_event_targets, volume_window_col,
                                       volume_window_end_col, winsorise)
from src.utils.artifact_store import artifact_file

ALL_WINDOWS = (1, 5, 10, 20)


def _market(volumes, n=160):
    sessions = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"date": sessions, "aspi_close": np.full(n, 100.0),
                         "trading_volume": volumes}), sessions


def test_missing_volume_propagates_as_missing_and_is_never_zero_filled():
    """The constraint. No volume column at all, and every Y2 view must be missing."""
    sessions = pd.date_range("2020-01-01", periods=160, freq="B")
    market = pd.DataFrame({"date": sessions, "aspi_close": np.full(160, 100.0)})
    out = build_event_targets(market, pd.DataFrame({"event_date": [sessions[60]]}))

    for w in ALL_WINDOWS:
        assert out[volume_window_col(w)].isna().all(), w
    for column in VOLUME_DERIVED_COLS:
        assert out[column].isna().all(), column


def test_a_gap_inside_the_response_window_is_missing_not_partial():
    """A window with an unobserved session must not be reported from the sessions that
    happen to be present."""
    volumes = np.full(160, 1000.0)
    market, sessions = _market(volumes)
    market.loc[62, "trading_volume"] = np.nan          # inside the 5-session window
    out = build_event_targets(market, pd.DataFrame({"event_date": [sessions[60]]}))
    assert np.isnan(out[volume_window_col(5)].iloc[0])
    assert np.isnan(out[volume_window_col(10)].iloc[0])
    # The 1-session window closes before the gap, so it is still observed.
    assert np.isfinite(out[volume_window_col(1)].iloc[0])


def test_every_window_is_built_against_the_same_thirty_session_baseline():
    """Doubling volume for the 20 sessions after the origin must read ln(2) at every
    window, because the baseline is the same 30 pre-event sessions throughout."""
    volumes = np.full(160, 1000.0)
    volumes[60:80] = 2000.0
    market, sessions = _market(volumes)
    out = build_event_targets(market, pd.DataFrame({"event_date": [sessions[60]]}))
    for w in ALL_WINDOWS:
        assert out[volume_window_col(w)].iloc[0] == pytest.approx(np.log(2.0)), w


def test_each_window_closes_on_its_own_session():
    volumes = np.full(160, 1000.0)
    market, sessions = _market(volumes)
    out = build_event_targets(market, pd.DataFrame({"event_date": [sessions[60]]}))
    for w in ALL_WINDOWS:
        assert out[volume_window_end_col(w)].iloc[0] == sessions[60 + w - 1], w


def test_the_percentage_form_satisfies_the_documented_identity():
    """The protocol's own worked example: a log ratio of 0.4055 is +50 percent."""
    assert abnormal_volume_percent(0.4055) == pytest.approx(50.0, abs=1e-2)
    assert abnormal_volume_percent(np.log(1.5)) == pytest.approx(50.0)
    assert abnormal_volume_percent(0.0) == pytest.approx(0.0)
    assert abnormal_volume_percent(np.log(0.5)) == pytest.approx(-50.0)
    assert np.isnan(abnormal_volume_percent(np.nan))


def test_winsorising_clips_the_tails_and_leaves_missing_values_alone():
    values = np.array([np.nan, -9.0, -0.2, 0.0, 0.3, 9.0])
    clipped = winsorise(values, quantiles=(0.05, 0.95))
    assert np.isnan(clipped[0])
    assert clipped[1] > -9.0 and clipped[-1] < 9.0
    assert clipped[2] == -0.2 and clipped[4] == 0.3
    assert np.isfinite(clipped[1:]).all()


def test_the_new_columns_can_never_become_predictors():
    for column in VOLUME_SENSITIVITY_COLS + VOLUME_DERIVED_COLS:
        assert column in NON_FEATURE_COLS, column
    for w in VOLUME_SENSITIVITY_WINDOWS:
        assert volume_window_end_col(w) in NON_FEATURE_COLS, w


def test_the_built_dataset_carries_every_window_with_the_same_coverage():
    path = artifact_file("dataset.parquet")
    if not path.exists():
        pytest.skip("run notebooks/02_features_targets.ipynb first")
    dataset = pd.read_parquet(path)
    observed = {w: int(dataset[volume_window_col(w)].notna().sum()) for w in ALL_WINDOWS}
    # Volume is missing for the same events at every window, so the counts agree.
    assert len(set(observed.values())) == 1, observed
    assert observed[5] == 61, observed

    primary = dataset[volume_window_col(5)]
    percent = dataset["Y2_5D_Forward_AbnormalVolume_Pct"]
    both = primary.notna()
    np.testing.assert_allclose(percent[both].to_numpy(),
                               abnormal_volume_percent(primary[both].to_numpy()), atol=1e-9)
