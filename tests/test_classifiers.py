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
        "Y1_ASPI_5D_Forward_LogReturn_Pct": rng.normal(0, 0.014, n),
        "Y2_abnormal_volume": rng.normal(-0.13, 0.58, n),
        "Y3_recovery_days": np.where(rng.random(n) < 0.14, 90.0, rng.exponential(6, n).round()),
        # Y1_EventWindow_0_10_LogReturn_Pct: cumulative event-window return, added
        "Y1_EventWindow_0_10_LogReturn_Pct": rng.normal(0, 0.04, n),
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
    perturbed.loc[perturbed.index[40:], "Y1_ASPI_5D_Forward_LogReturn_Pct"] *= 10.0
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


def test_labels_propagate_missing_targets_instead_of_asserting_the_negative_class():
    """A missing target must produce NaN, never 0.0."""
    import numpy as np
    import pandas as pd

    from src.models.classifiers import LABELS

    y = pd.DataFrame({
        "Y1_ASPI_5D_Forward_LogReturn_Pct": [0.01, -0.01, 0.02, -0.02],
        "Y2_abnormal_volume": [0.5, -0.3, np.nan, np.nan],
        "Y3_recovery_days": [0.0, 90.0, 5.0, 90.0],
    })
    train_idx = np.arange(4)

    volume = LABELS["C2_volume_spike"](y, train_idx, None)
    assert volume.isna().sum() == 2, list(volume)
    assert list(volume.iloc[:2]) == [1.0, 0.0]

    # Y1 and Y3 are complete here, so those labels must be unaffected.
    for name in ("C1_negative_return", "C1b_adverse_move", "C3_recovers_in_90"):
        assert LABELS[name](y, train_idx, None).isna().sum() == 0, name


def test_missing_labels_are_dropped_by_a_notna_mask():
    """The downstream contract: masking on notna() must actually remove those rows."""
    import numpy as np
    import pandas as pd

    from src.models.classifiers import LABELS

    y = pd.DataFrame({
        "Y1_ASPI_5D_Forward_LogReturn_Pct": [0.01] * 6,
        "Y2_abnormal_volume": [0.5, -0.3, np.nan, 0.2, np.nan, -0.1],
        "Y3_recovery_days": [0.0] * 6,
    })
    labels = LABELS["C2_volume_spike"](y, np.arange(6), None)
    kept = np.arange(6)[labels.notna().to_numpy()]
    assert list(kept) == [0, 1, 3, 5]


def test_new_labels_are_registered_and_balanced():
    """C3b, pre-declared in docs/audit.md Part 3 section 7.3."""
    import numpy as np
    import pandas as pd

    from src.models.classifiers import LABELS

    assert "C3b_slow_recovery" in LABELS
    assert "C4_car5_negative" not in LABELS

    rng = np.random.default_rng(0)
    n = 40
    y = pd.DataFrame({
        "Y1_ASPI_5D_Forward_LogReturn_Pct": rng.normal(0, 0.014, n),
        "Y2_abnormal_volume": rng.normal(0, 0.5, n),
        "Y3_recovery_days": np.arange(n, dtype=float),      # 0..39, median 19.5
        "Y1_EventWindow_0_10_LogReturn_Pct": rng.normal(0, 0.04, n),
    })
    train_idx = np.arange(n)

    slow = LABELS["C3b_slow_recovery"](y, train_idx)
    # A median split is ~50% by construction, the whole point, versus C3's 0.90.
    assert 0.45 <= slow.mean() <= 0.55, slow.mean()
    # And it must actually be the median cut, not the 90-day cap.
    assert slow.iloc[0] == 0.0 and slow.iloc[-1] == 1.0


def test_slow_recovery_cut_comes_from_training_rows_only():
    """No test information may reach the label, exactly as for C1b's tercile."""
    import numpy as np
    import pandas as pd

    from src.models.classifiers import LABELS

    # Training window is all small values; the test tail is huge. A cut computed on the
    # full column would sit far above the training median and mislabel the training rows.
    y = pd.DataFrame({
        "Y1_ASPI_5D_Forward_LogReturn_Pct": np.zeros(20),
        "Y2_abnormal_volume": np.zeros(20),
        "Y3_recovery_days": np.concatenate([np.arange(10, dtype=float),
                                            np.full(10, 500.0)]),
        "Y1_EventWindow_0_10_LogReturn_Pct": np.zeros(20),
    })
    lab = LABELS["C3b_slow_recovery"](y, np.arange(10))   # train on the first 10 only
    assert lab.iloc[:10].mean() == pytest.approx(0.5)      # median of 0..9 is 4.5
    assert (lab.iloc[10:] == 1.0).all()                    # every 500 is "slow"


def test_new_labels_propagate_missing_targets():
    import numpy as np
    import pandas as pd

    from src.models.classifiers import LABELS

    y = pd.DataFrame({
        "Y1_ASPI_5D_Forward_LogReturn_Pct": [0.01] * 4,
        "Y2_abnormal_volume": [0.1] * 4,
        "Y3_recovery_days": [1.0, 2.0, np.nan, 4.0],
        "Y1_EventWindow_0_10_LogReturn_Pct": [0.0] * 4,
    })
    assert LABELS["C3b_slow_recovery"](y, np.arange(4)).isna().sum() == 1


def test_auc_scorer_returns_nan_instead_of_raising_on_degenerate_splits():
    """A split with one class on either side scores NaN, matching what sklearn's own
    roc_auc plus error_score produces, but without raising."""
    from sklearn.ensemble import RandomForestClassifier
    from src.models.classifiers import roc_auc_or_nan

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))

    # Validation side holds one class: the metric is undefined there.
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, [0, 1] * 10)
    assert np.isnan(roc_auc_or_nan(model, X, np.zeros(20, dtype=int)))

    # Training side held one class, so the model has a single probability column.
    single = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, np.zeros(20, dtype=int))
    assert single.predict_proba(X).shape[1] == 1
    assert np.isnan(roc_auc_or_nan(single, X, np.array([0, 1] * 10)))


def test_auc_scorer_matches_sklearn_on_a_healthy_split():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from src.models.classifiers import roc_auc_or_nan

    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y)
    expected = roc_auc_score(y, model.predict_proba(X)[:, 1])
    assert roc_auc_or_nan(model, X, y) == pytest.approx(expected)
