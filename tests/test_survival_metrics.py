"""Censoring-aware Y3 metrics.

The point of these tests is that each metric must behave differently from its naive,
censoring-blind counterpart, otherwise switching to them bought nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.survival_metrics import (PROB_TIMES, cluster_bootstrap_ci,
                                             harrell_c_index, integrated_brier_score,
                                             probability_calibration,
                                             uncensored_point_errors)


@pytest.fixture
def censored_sample():
    rng = np.random.default_rng(0)
    n = 300
    true_t = rng.exponential(20, n)
    cens_t = rng.exponential(60, n)
    return np.minimum(true_t, cens_t), true_t <= cens_t, true_t


def test_c_index_detects_order_and_ignores_scale(censored_sample):
    d, e, true_t = censored_sample
    assert harrell_c_index(d, e, true_t) > 0.95
    # A monotone rescale is the same ranking, so concordance must not move.
    assert harrell_c_index(d, e, true_t) == pytest.approx(harrell_c_index(d, e, true_t * 7 + 3))


def test_c_index_of_noise_is_chance(censored_sample):
    d, e, _ = censored_sample
    rng = np.random.default_rng(1)
    assert abs(harrell_c_index(d, e, rng.normal(size=len(d))) - 0.5) < 0.1


def test_c_index_is_nan_when_undefined(censored_sample):
    d, e, _ = censored_sample
    assert np.isnan(harrell_c_index(d, e, np.ones(len(d))))          # constant prediction
    assert np.isnan(harrell_c_index(d[:5], np.zeros(5, bool), d[:5]))  # no observed events


def test_brier_prefers_the_true_curve(censored_sample):
    d, e, _ = censored_sample
    times = np.array(PROB_TIMES)
    s_true = np.exp(-np.outer(np.ones(len(d)), times) / 20.0)
    s_flat = np.full((len(d), len(times)), 0.5)
    assert (integrated_brier_score(d, e, s_true)["integrated_brier_score"]
            < integrated_brier_score(d, e, s_flat)["integrated_brier_score"])


def test_brier_reports_every_requested_time(censored_sample):
    d, e, _ = censored_sample
    s = np.full((len(d), len(PROB_TIMES)), 0.5)
    out = integrated_brier_score(d, e, s)
    assert set(out["brier_by_time"]) == set(map(float, PROB_TIMES))
    assert out["integrated_brier_score"] == pytest.approx(
        np.mean(list(out["brier_by_time"].values())))


def test_calibration_gap_is_small_for_a_correct_curve(censored_sample):
    d, e, _ = censored_sample
    times = np.array(PROB_TIMES)
    cdf = 1.0 - np.exp(-np.outer(np.ones(len(d)), times) / 20.0)
    assert probability_calibration(d, e, cdf)["calibration_gap"].abs().max() < 0.1


def test_calibration_is_wrong_for_an_overconfident_curve(censored_sample):
    d, e, _ = censored_sample
    cdf = np.full((len(d), len(PROB_TIMES)), 0.99)
    assert probability_calibration(d, e, cdf)["calibration_gap"].max() > 0.2


def test_point_errors_use_only_observed_recoveries(censored_sample):
    d, e, true_t = censored_sample
    out = uncensored_point_errors(d, e, true_t)
    assert out["n_uncensored"] == int(e.sum())
    assert out["mae_uncensored"] == pytest.approx(0.0, abs=1e-9)
    # Corrupting a CENSORED row's prediction must not change the score at all, that is
    # the whole difference from an ordinary RMSE over every row.
    p = true_t.copy()
    p[~e] += 1000.0
    assert uncensored_point_errors(d, e, p)["rmse_uncensored"] == pytest.approx(
        out["rmse_uncensored"])


def test_point_errors_handle_an_all_censored_sample():
    out = uncensored_point_errors([1.0, 2.0], [False, False], [1.0, 2.0])
    assert out["n_uncensored"] == 0 and np.isnan(out["mae_uncensored"])


def test_cluster_bootstrap_returns_a_ci_and_its_draws(censored_sample):
    d, e, true_t = censored_sample
    fn = lambda i: harrell_c_index(d[i], e[i], true_t[i])  # noqa: E731
    lo, hi = cluster_bootstrap_ci(fn, np.arange(len(d)) // 3, n_boot=300)
    lo2, hi2, draws = cluster_bootstrap_ci(fn, np.arange(len(d)) // 3, n_boot=300,
                                           return_draws=True)
    assert (lo, hi) == (lo2, hi2)          # same seed, same resampling
    assert lo > 0.5 and hi <= 1.0
    assert len(draws) >= 100


def test_cluster_bootstrap_gives_nan_when_too_few_usable_draws():
    lo, hi = cluster_bootstrap_ci(lambda i: float("nan"), np.arange(30), n_boot=200)
    assert np.isnan(lo) and np.isnan(hi)
