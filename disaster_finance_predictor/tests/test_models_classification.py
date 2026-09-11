import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from src.evaluation.verification import diebold_mariano, paired_bootstrap_delta
from src.models.classifiers import (LABELS, classification_metrics, label_adverse_move,
                                    label_recovers_in_90)
from src.models.hurdle import CAP, HurdleRecoveryModel


def _targets(n=64, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Y1_aspi_log_return": rng.normal(0, 0.014, n),
        "Y2_abnormal_volume": rng.normal(-0.13, 0.58, n),
        "Y3_recovery_days": np.where(rng.random(n) < 0.14, 90.0, rng.exponential(6, n).round()),
    })


def test_all_labels_are_binary_and_defined_for_every_event():
    y = _targets()
    train_idx = np.arange(30)
    for name, fn in LABELS.items():
        lab = fn(y, train_idx, None)
        assert len(lab) == len(y), name
        assert set(np.unique(lab.dropna())) <= {0.0, 1.0}, name


def test_adverse_move_cut_point_uses_training_rows_only():
    """The tercile must be computed on the training window. If it leaked from the test
    rows the label would shift when only test-row targets change."""
    y = _targets()
    train_idx = np.arange(30)
    base = label_adverse_move(y, train_idx)

    perturbed = y.copy()
    perturbed.loc[perturbed.index[40:], "Y1_aspi_log_return"] *= 10.0
    after = label_adverse_move(perturbed, train_idx)

    # Training-row labels are untouched by anything happening after the training window.
    assert np.array_equal(base.iloc[train_idx].to_numpy(), after.iloc[train_idx].to_numpy())


def test_adverse_move_gives_a_workable_prevalence_where_a_sigma_cut_would_not():
    y = _targets()
    train_idx = np.arange(30)
    prevalence = label_adverse_move(y, train_idx).iloc[train_idx].mean()
    assert 0.2 <= prevalence <= 0.45, prevalence


def test_majority_rule_scores_exactly_half_on_balanced_accuracy():
    """This is the property that makes balanced accuracy the right headline metric here:
    the trivial classifier cannot game it, unlike plain accuracy."""
    truth = np.array([1] * 80 + [0] * 20)
    m = classification_metrics(truth, np.ones(100))
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["mcc"] == pytest.approx(0.0) or np.isnan(m["mcc"])
    assert m["accuracy"] == pytest.approx(0.8)
    assert not m["beats_baseline"]


def test_a_chance_classifier_is_not_credited_with_beating_the_baseline():
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 2, 200)
    assert not classification_metrics(truth, rng.random(200))["beats_baseline"]
    assert classification_metrics(truth, truth.astype(float))["beats_baseline"]


def test_hurdle_predictions_stay_inside_the_target_support():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 3))
    recovers = X[:, 0] + rng.normal(0, 0.3, 150) > 0
    y = np.where(recovers, np.clip(np.abs(X[:, 1]) * 8, 0, 89), CAP)

    model = HurdleRecoveryModel(
        LogisticRegression(max_iter=1000),
        RandomForestRegressor(n_estimators=50, max_depth=3, random_state=0)).fit(X, y)
    pred = model.predict(X)

    assert pred.min() >= 0.0 and pred.max() <= CAP
    assert np.mean(np.abs(pred - y)) < np.mean(np.abs(np.mean(y) - y))


def test_hurdle_degrades_gracefully_when_a_fold_has_no_censoring():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    model = HurdleRecoveryModel(LogisticRegression(max_iter=1000),
                                RandomForestRegressor(n_estimators=10, random_state=0))
    model.fit(X, np.full(40, 5.0))
    assert model.degenerate_
    assert np.allclose(model.predict(X), 5.0)


def test_recovers_in_90_is_the_censoring_indicator():
    y = _targets()
    lab = label_recovers_in_90(y)
    assert np.array_equal(lab.to_numpy(), (y["Y3_recovery_days"] < 90).astype(float).to_numpy())


def test_verification_tests_agree_on_an_obvious_win_and_an_obvious_tie():
    rng = np.random.default_rng(3)
    truth = rng.normal(0, 1, 150)
    good, bad = truth + rng.normal(0, 0.2, 150), truth + rng.normal(0, 1.2, 150)

    boot, dm = paired_bootstrap_delta(truth, good, bad), diebold_mariano(truth, good, bad)
    assert boot["significant"] and dm["significant"]

    a, b = truth + rng.normal(0, 0.5, 150), truth + rng.normal(0, 0.5, 150)
    assert not paired_bootstrap_delta(truth, a, b)["significant"]
