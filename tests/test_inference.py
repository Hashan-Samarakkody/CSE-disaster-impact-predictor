"""Smoke tests for the demo app's prediction serving.

Requires the artifacts the full pipeline produces (dataset, feature_spec,
final_rf_models, final_classifiers, final_hurdle_model), skipped if absent, since
these are regenerable outputs, not something a fresh checkout carries.
"""

import pytest

pytest.importorskip("pandas")

from pathlib import Path
from src.utils.artifact_store import artifact_file

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
pytestmark = pytest.mark.skipif(
    not (artifact_file("final_classifiers.pkl")).exists(),
    reason="run the pipeline + scripts/train_final_models.py first")


@pytest.fixture(scope="module")
def bundle():
    from src.models.inference import get_bundle
    return get_bundle()


def test_lists_all_real_events(bundle):
    # N=74 under the inclusion threshold frozen at >=1000 affected (2026-09-16
    # methodology-audit review); was 76 while the threshold had drifted to 700/500.
    events = bundle.list_events()
    assert len(events) == 74
    assert "event_date" in events.columns


def test_unmodified_event_round_trips_through_the_same_feature_values(bundle):
    """No override applied -> the feature row must equal the real cached row exactly,
    so a prediction with no override reproduces the real, already-scored input.
    """
    events = bundle.list_events()
    event_id = int(events["event_id"].iloc[0])
    row = bundle.build_feature_row(event_id, {})
    real = bundle.event_row(event_id)
    for col in bundle.feature_cols:
        if col in real.index and real[col] == real[col]:
            expected = real[col]
        else:
            expected = bundle.median_impute_values.get(col, 0.0)
        assert row[col] == pytest.approx(float(expected)), col


def test_severity_override_changes_only_the_dependent_columns(bundle):
    events = bundle.list_events()
    event_id = int(events["event_id"].iloc[0])
    base = bundle.build_feature_row(event_id, {})
    bumped = bundle.build_feature_row(event_id, {"financial_damage": 999_000_000.0})

    assert bumped["financial_damage"] == pytest.approx(999_000_000.0)
    assert bumped["log_financial_damage"] > base["log_financial_damage"]
    unrelated = ["log_return", "gdp_growth_pct", "hz_precip_max3d", "fx_logret_5"]
    for col in unrelated:
        assert bumped[col] == pytest.approx(base[col]), col


def test_regression_predictions_stay_inside_each_targets_definitional_support(bundle):
    events = bundle.list_events()
    event_id = int(events["event_id"].iloc[0])
    row = bundle.build_feature_row(event_id, {})
    pred = bundle.predict_regression(row)

    assert pred["Y2_abnormal_volume"] >= -1.0
    assert 0.0 <= pred["Y3_recovery_days"] <= 90.0


def test_classification_predictions_are_valid_probabilities_with_verdict_metadata(bundle):
    events = bundle.list_events()
    event_id = int(events["event_id"].iloc[0])
    row = bundle.build_feature_row(event_id, {})
    pred = bundle.predict_classification(row)

    # C4_car5_negative removed 2026-09-16 (methodology-audit finding #8): identical to
    # C1_negative_return once Y1 absorbed EventWindow_0_5's formula.
    assert set(pred) == {"C1_negative_return", "C1b_adverse_move", "C2_volume_spike",
                         "C3_recovers_in_90", "C3b_slow_recovery"}
    for name, info in pred.items():
        assert 0.0 <= info["probability"] <= 1.0, name
        assert isinstance(info["beats_baseline"], bool), name

    # Measured result after wiring collinearity-drop into feature selection here too
    assert pred["C2_volume_spike"]["beats_baseline"]
    assert not pred["C1_negative_return"]["beats_baseline"]
    # New confirmed result after the Y3 adverse-response-gate fix (finding #11,
    assert pred["C3b_slow_recovery"]["beats_baseline"]


def test_hurdle_prediction_stays_inside_the_censored_support(bundle):
    events = bundle.list_events()
    event_id = int(events["event_id"].iloc[0])
    row = bundle.build_feature_row(event_id, {})
    assert 0.0 <= bundle.predict_hurdle(row) <= 90.0
