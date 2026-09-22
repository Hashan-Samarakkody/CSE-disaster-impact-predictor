"""Integrity of the Y1/Y3 improvement-grid artifacts."""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest
from src.utils.artifact_store import artifact_file

REQUIRED = ["aspi_grid_predictions.parquet", "aspi_grid_metrics.parquet",
            "aspi_grid_verdicts.parquet", "aspi_direction_metrics.parquet",
            "recovery_grid_predictions.parquet", "recovery_grid_metrics.parquet"]

pytestmark = pytest.mark.skipif(
    not all(artifact_file(f).exists() for f in REQUIRED),
    reason="improvement artifacts missing -- run scripts/run_aspi_return_grid.py and "
           "scripts/run_recovery_survival_grid.py")


@pytest.fixture(scope="module")
def y1_oof():
    return pd.read_parquet(artifact_file("aspi_grid_predictions.parquet"))


@pytest.fixture(scope="module")
def y3_oof():
    return pd.read_parquet(artifact_file("recovery_grid_predictions.parquet"))


def test_y1_every_configuration_scores_the_same_test_events(y1_oof):
    """Protocol 3.1: paired evaluation. Every candidate AND every baseline must predict
    exactly the same rows, per horizon, or the bootstrap deltas are not paired."""
    for horizon, g in y1_oof.groupby("horizon"):
        per_config = g.groupby(["info_set", "k", "model"])["row"].apply(
            lambda s: tuple(sorted(s)))
        assert per_config.nunique() == 1, f"horizon {horizon}: test events differ by config"


def test_y1_targets_are_identical_across_configurations(y1_oof):
    """The same event at the same horizon has one true value; a mismatch would mean two
    configurations were fitted against different targets."""
    for (horizon, row), g in y1_oof.groupby(["horizon", "row"]):
        assert g["y_true"].nunique() == 1, (horizon, row)


def test_y1_all_three_baselines_present_at_every_horizon(y1_oof):
    baselines = {"naive_zero", "naive_train_mean", "market_only_expected"}
    for horizon, g in y1_oof.groupby("horizon"):
        assert baselines <= set(g[g.info_set == "baseline"]["model"]), horizon


def test_y1_naive_zero_baseline_really_predicts_zero(y1_oof):
    assert (y1_oof[y1_oof.model == "naive_zero"]["y_pred"] == 0.0).all()


def test_y1_predictions_are_finite(y1_oof):
    assert np.isfinite(y1_oof["y_pred"]).all()
    assert np.isfinite(y1_oof["y_true"]).all()


def test_y1_verdict_column_agrees_with_its_interval():
    v = pd.read_parquet(artifact_file("aspi_grid_verdicts.parquet"))
    assert (v["boot_beats"] == (v["ci_low"] > 0)).all()
    assert (v.loc[v["boot_beats"], "verdict"] == "A - statistically supported").all()
    # Holm can only ever be stricter than the single-comparison criterion.
    assert not (v["holm_significant"] & ~v["boot_beats"]).any()


def test_y1_direction_labels_partition_the_horizon(y1_oof):
    """`negative_return = Y_h < 0` must match the pooled targets, per horizon."""
    d = pd.read_parquet(artifact_file("aspi_direction_metrics.parquet"))
    for horizon, g in d.groupby("horizon"):
        truth = y1_oof[(y1_oof.horizon == horizon) & (y1_oof.model == "naive_zero")]
        assert g["n"].nunique() == 1 and g["n"].iloc[0] == len(truth)
        assert g["n_negative"].iloc[0] == int((truth["y_true"] < 0).sum())


def test_y3_every_model_scores_the_same_test_events(y3_oof):
    per_model = y3_oof.groupby("model")["row"].apply(lambda s: tuple(sorted(s)))
    assert per_model.nunique() == 1


def test_y3_durations_and_censoring_match_the_frozen_dataset(y3_oof):
    data = pd.read_parquet(artifact_file("dataset.parquet")).reset_index(drop=True)
    for model, g in y3_oof.groupby("model"):
        rows = g["row"].to_numpy()
        np.testing.assert_allclose(g["duration"].to_numpy(float),
                                   data["Y3_ASPI_Recovery_Time"].to_numpy(float)[rows],
                                   err_msg=model)
        assert (g["event_observed"].to_numpy() ==
                ~data["Y3_censored"].to_numpy(bool)[rows]).all(), model


def test_y3_recovery_probabilities_are_monotone_and_bounded(y3_oof):
    """P(T <= t) is a CDF: non-decreasing in t and inside [0, 1]."""
    cols = ["P_T_le_10", "P_T_le_20", "P_T_le_30", "P_T_le_60", "P_T_le_90"]
    p = y3_oof[cols].to_numpy(float)
    assert np.isfinite(p).all()
    assert (p >= -1e-9).all() and (p <= 1 + 1e-9).all()
    assert (np.diff(p, axis=1) >= -1e-9).all()


def test_y3_predicted_medians_are_inside_the_design_window(y3_oof):
    assert y3_oof["pred_median"].between(0.0, 90.0).all()


def test_y3_concordance_verdicts_match_their_own_intervals():
    """The recovery grid's claim column must follow from its own numbers. Under the
    frozen target protocol no configuration clears chance, so this guards against a
    verdict string drifting away from the interval that produced it rather than pinning
    a particular finding."""
    m = pd.read_parquet(artifact_file("recovery_grid_metrics.parquet"))
    # 15 original configurations plus the Aalen-Johansen competing-risks arm (T2).
    assert len(m) == 16
    assert "aalen_johansen_competing_risks" in set(m["model"])
    assert m["c_index"].between(0.0, 1.0).all()
    assert (m["c_index_ci_low"] <= m["c_index"]).all()
    assert (m["c_index"] <= m["c_index_ci_high"]).all()
    beats = m["c_index_ci_low"] > 0.5
    assert (m["c_index_beats_chance"].astype(bool) == beats).all()
    assert (m.loc[~m["c_index_beats_chance"].astype(bool), "holm_significant"] == False).all()
