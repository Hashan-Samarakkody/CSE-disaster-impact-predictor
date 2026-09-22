"""T6: the market model behind the abnormal-return target.

The one property that matters is the constraint: the estimation window must close before
the prediction origin, so no post-event session can influence the expected return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.settings import ABNORMAL_RETURN_HORIZONS, NON_FEATURE_COLS
from src.targets.abnormal_returns import (ESTIMATION_WINDOW, abnormal_return_col,
                                          abnormal_return_end_col, align_market_factor,
                                          build_abnormal_return_targets,
                                          estimate_market_model)
from src.utils.artifact_store import artifact_file


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(7)
    n = 600
    sessions = pd.date_range("2018-01-01", periods=n, freq="B")
    market = rng.normal(0.0, 0.01, n)
    asset = 1.25 * market + rng.normal(0.0, 0.002, n)
    prices = 100.0 * np.exp(np.cumsum(asset))
    return pd.DataFrame({"date": sessions, "aspi_close": prices,
                         "log_return": asset, "sp500_log_return": market})


def test_estimation_window_ends_strictly_before_the_prediction_origin(synthetic):
    """The constraint T6 names. `origin_row` is the first post-event session, so the last
    row the model may see is `origin_row - 1`, the close observed before the origin."""
    origin = 400
    model = estimate_market_model(synthetic["log_return"], synthetic["sp500_log_return"], origin)
    assert model is not None
    assert model.last_row == origin - 1
    assert model.last_row < origin
    assert model.first_row == origin - ESTIMATION_WINDOW
    assert model.n == ESTIMATION_WINDOW


def test_post_origin_sessions_cannot_move_the_fitted_coefficients(synthetic):
    """Corrupting every session from the origin onward must leave the model identical."""
    origin = 400
    before = estimate_market_model(synthetic["log_return"], synthetic["sp500_log_return"], origin)

    corrupted = synthetic.copy()
    corrupted.loc[origin:, "log_return"] = 99.0
    corrupted.loc[origin:, "sp500_log_return"] = -99.0
    after = estimate_market_model(corrupted["log_return"], corrupted["sp500_log_return"], origin)

    assert before.alpha == pytest.approx(after.alpha)
    assert before.beta == pytest.approx(after.beta)


def test_a_planted_shock_is_recovered_as_the_abnormal_return():
    """A known shock on the post-event sessions must come back as the abnormal return,
    and a quiet event must come back as zero."""
    n = 600
    sessions = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(11)
    market = rng.normal(0.0, 0.01, n)
    asset = 1.5 * market
    origin, shock = 400, -0.003
    asset[origin:origin + 5] += shock
    frame = pd.DataFrame({"date": sessions, "aspi_close": 100.0 * np.exp(np.cumsum(asset)),
                          "log_return": asset, "sp500_log_return": market})

    out = build_abnormal_return_targets(frame, [sessions[origin], sessions[300]], (5,))
    assert out[abnormal_return_col(5)].iloc[0] == pytest.approx(100 * 5 * shock, abs=1e-6)
    assert out[abnormal_return_col(5)].iloc[1] == pytest.approx(0.0, abs=1e-6)


def test_too_little_history_gives_a_missing_value_not_a_thin_fit(synthetic):
    assert estimate_market_model(synthetic["log_return"], synthetic["sp500_log_return"], 10) is None
    out = build_abnormal_return_targets(synthetic, [synthetic["date"].iloc[10]], (5,))
    assert np.isnan(out[abnormal_return_col(5)].iloc[0])


def test_calendar_alignment_never_drops_a_local_session():
    """The two markets keep different holidays, and the alignment must handle both
    directions without leaving a hole that voids the whole event window."""
    local = pd.date_range("2020-01-01", periods=10, freq="B")

    # The factor market was shut on one local session: that session saw no market move,
    # so it carries zero rather than a missing value.
    factor_dates = local.delete(4)
    aligned = align_market_factor(local, factor_dates, np.full(len(factor_dates), 0.01))
    assert np.isnan(aligned[0])                          # no preceding interval
    assert np.isfinite(aligned[1:]).all()
    assert aligned[4] == pytest.approx(0.0)
    assert aligned[5] == pytest.approx(0.01)

    # The local market was shut while the factor kept trading: the next local session
    # accumulates both factor days, rather than discarding one.
    open_local = local.delete(4)
    accumulated = align_market_factor(open_local, local, np.full(len(local), 0.01))
    assert accumulated[4] == pytest.approx(0.02)
    assert np.isfinite(accumulated[1:]).all()


def test_abnormal_columns_are_present_and_can_never_be_predictors():
    """Acceptance: one column per horizon with its own label-end date, all non-features."""
    for h in ABNORMAL_RETURN_HORIZONS:
        assert abnormal_return_col(h) in NON_FEATURE_COLS, h
        assert abnormal_return_end_col(h) in NON_FEATURE_COLS, h

    path = artifact_file("dataset.parquet")
    if not path.exists():
        pytest.skip("run notebooks/02_features_targets.ipynb first")
    dataset = pd.read_parquet(path)
    for h in ABNORMAL_RETURN_HORIZONS:
        assert abnormal_return_col(h) in dataset.columns, h
        assert abnormal_return_end_col(h) in dataset.columns, h
        assert dataset[abnormal_return_col(h)].notna().all(), h
