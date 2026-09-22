"""Tests for the invariants the study's credibility rests on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import truncate_overlapping_windows
from src.features.sector_panel import event_block_bootstrap, grouped_walk_forward
from src.evaluation.collinearity import REDUNDANCY_THRESHOLD, redundant_drop_set
from src.training.walk_forward import generate_walk_forward_splits


# chronological walk-forward

def test_expanding_mode_never_trains_on_the_future():
    """T9. The whole point of a chronological split: every training index must precede
    every test index, in both modes."""
    for mode in ("sliding", "expanding"):
        splits = list(generate_walk_forward_splits(74, 30, 10, 10, mode=mode))
        assert len(splits) == 4, mode
        for split in splits:
            assert split.train_index.max() < split.test_index.min(), mode


def test_expanding_and_sliding_modes_never_overlap_train_with_test():
    for mode in ("sliding", "expanding"):
        for split in generate_walk_forward_splits(74, 30, 10, 10, mode=mode):
            assert not set(split.train_index) & set(split.test_index), mode


def test_expanding_mode_grows_the_training_block_monotonically():
    """Expanding keeps the earliest events instead of dropping them, so the block starts
    at index zero and only ever gets longer."""
    splits = list(generate_walk_forward_splits(74, 30, 10, 10, mode="expanding"))
    sizes = [len(s.train_index) for s in splits]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes), sizes
    assert all(s.train_index[0] == 0 for s in splits)
    assert sizes == [30, 40, 50, 60]


def test_sliding_mode_is_unchanged_and_is_the_default():
    """T9 requires the default to stay sliding so no existing result moves."""
    default = list(generate_walk_forward_splits(74, 30, 10, 10))
    explicit = list(generate_walk_forward_splits(74, 30, 10, 10, mode="sliding"))
    assert len(default) == len(explicit)
    for a, b in zip(default, explicit):
        assert (a.train_index == b.train_index).all()
        assert (a.test_index == b.test_index).all()
    assert all(len(s.train_index) == 30 for s in default)


def test_both_modes_score_exactly_the_same_test_events():
    """A robustness comparison between the two is only meaningful if the held-out events
    are identical, which they must be by construction."""
    sliding = list(generate_walk_forward_splits(74, 30, 10, 10, mode="sliding"))
    expanding = list(generate_walk_forward_splits(74, 30, 10, 10, mode="expanding"))
    for a, b in zip(sliding, expanding):
        assert (a.test_index == b.test_index).all()


def test_an_unknown_window_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        list(generate_walk_forward_splits(74, 30, 10, 10, mode="rolling"))


def test_walk_forward_never_trains_on_the_future():
    """Thesis 3.7.1 forbids k-fold. Every training index must precede every test index."""
    for s in generate_walk_forward_splits(74, train_window=30, test_window=10, step=10):
        assert s.train_index.max() < s.test_index.min()


def test_walk_forward_train_and_test_never_intersect():
    for s in generate_walk_forward_splits(74, train_window=30, test_window=10, step=10):
        assert not set(s.train_index) & set(s.test_index)


def test_walk_forward_geometry_at_the_studys_actual_size():
    """N=74 with the 30/10/10 config must give 4 folds and 40 pooled test points."""
    splits = list(generate_walk_forward_splits(74, 30, 10, 10))
    assert len(splits) == 4
    assert sum(len(s.test_index) for s in splits) == 40
    # Test windows must tile without overlapping, otherwise pooled points are double
    # counted and every interval is too narrow.
    pooled = np.concatenate([s.test_index for s in splits])
    assert len(pooled) == len(set(pooled))


def test_walk_forward_yields_nothing_when_the_sample_is_too_small():
    assert list(generate_walk_forward_splits(20, 30, 10, 10)) == []


# sector panel: event grouping

@pytest.fixture
def panel():
    """6 events x 20 sectors, the shape of the real panel."""
    return pd.DataFrame({
        "event_id": np.repeat(np.arange(6), 20),
        "sector": list(range(20)) * 6,
        "y": np.random.default_rng(0).normal(size=120),
    })


def test_sector_folds_never_split_an_event_across_the_boundary(panel):
    """An event's 20 sectors co-move on the shock day. Splitting them across train and
    test leaks the answer and inflates every panel score."""
    for tr, te in grouped_walk_forward(panel, train_events=3, test_events=1, step=1):
        tr_events = set(panel.event_id.iloc[tr])
        te_events = set(panel.event_id.iloc[te])
        assert not tr_events & te_events


def test_sector_folds_keep_every_sector_of_a_test_event_together(panel):
    for _, te in grouped_walk_forward(panel, train_events=3, test_events=1, step=1):
        assert len(te) == 20  # all 20 sectors of the single test event


def test_sector_folds_are_chronological(panel):
    for tr, te in grouped_walk_forward(panel, train_events=3, test_events=1, step=1):
        assert panel.event_id.iloc[tr].max() < panel.event_id.iloc[te].min()


# event-clustered bootstrap must not shrink

def test_event_block_bootstrap_is_wider_than_resampling_rows_independently():
    """The whole point of clustering. Treating 20 correlated sector rows as 20
    independent observations shrinks the interval by roughly sqrt(20) and manufactures
    significance out of the co-movement.
    """
    rng = np.random.default_rng(0)
    n_events, n_sectors = 20, 20
    event_ids = np.repeat(np.arange(n_events), n_sectors)
    # A shared per-event shock is what makes sectors correlated within an event.
    shock = rng.normal(0, 1.0, n_events).repeat(n_sectors)
    y = shock + rng.normal(0, 0.1, n_events * n_sectors)
    pred_a = shock * 0.5
    pred_b = np.zeros_like(y)

    clustered = event_block_bootstrap(event_ids, y, pred_a, pred_b, n_boot=1000)
    clustered_width = clustered["ci_high"] - clustered["ci_low"]

    # Naive row-level bootstrap: every row treated as its own independent "event".
    naive = event_block_bootstrap(np.arange(len(y)), y, pred_a, pred_b, n_boot=1000)
    naive_width = naive["ci_high"] - naive["ci_low"]
    assert clustered_width > naive_width, (clustered_width, naive_width)
    assert clustered["n_events"] == n_events and clustered["n_rows"] == n_events * n_sectors


def test_event_block_bootstrap_point_estimate_favours_the_better_model():
    rng = np.random.default_rng(1)
    event_ids = np.repeat(np.arange(10), 5)
    y = rng.normal(size=50)
    good, bad = y * 0.9, np.zeros_like(y)
    out = event_block_bootstrap(event_ids, y, good, bad, n_boot=500)
    assert out["delta_rmse"] > 0  # positive favours model a, which here is the good one


# the pre-declared collinearity drop rule

def test_drop_rule_keeps_the_least_derived_member():
    """Pre-declared rule: within a group correlated above |rho| >= 0.95, keep the most
    primitive column. Deterministic, and fixed before any score was seen."""
    n = 200
    base = np.linspace(1, 100, n)
    X = pd.DataFrame({
        "financial_damage": base,
        "log_financial_damage": np.log1p(base),   # one derivation step off the raw column
        "rolling_std_5": np.random.default_rng(0).normal(size=n),
    })
    _keep, drop, _detail = redundant_drop_set(X)
    assert "log_financial_damage" in drop
    assert "financial_damage" not in drop


def test_drop_rule_leaves_uncorrelated_features_alone():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"f{i}": rng.normal(size=200) for i in range(5)})
    _keep, drop, _detail = redundant_drop_set(X)
    assert drop == []


def test_drop_rule_threshold_is_the_declared_one():
    assert REDUNDANCY_THRESHOLD == 0.95


# Y3 window contamination

def test_truncate_overlapping_windows_flags_events_inside_a_prior_window():
    """29 of 64 Y3 windows contained a later qualifying disaster. The helper that
    handles it existed uncalled for three phases; this pins its behaviour."""
    events = pd.DataFrame({"event_date": pd.to_datetime(
        ["2020-01-01", "2020-02-01", "2020-09-01"])})  # 2nd lands 31 days after the 1st
    out = truncate_overlapping_windows(events, horizon_days=90)
    assert len(out) == len(events)
    # Whatever column it adds, the first event must be marked as contaminated and the
    # last (9 months clear of anything) must not.
    added = [c for c in out.columns if c != "event_date"]
    assert added, "truncate_overlapping_windows added no column"
    flag = added[0]
    assert out[flag].iloc[0] != out[flag].iloc[2]
