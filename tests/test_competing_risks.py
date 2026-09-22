"""T2 and T3: informative censoring, and the demotion of the Y3 regression.

A subsequent qualifying disaster is a COMPETING event, not independent censoring. An event
that has not recovered is more likely to be overtaken by a new one, so treating it as
ordinary censoring credits those events with a recovery they may never have had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.settings import (MARKET_RECOVERY_DAYS, Y3_DIAGNOSTIC_LABEL,
                                 Y3_REGRESSION_IS_DIAGNOSTIC)
from src.models.survival_recovery import (EVENT_CENSORED, EVENT_COMPETING, EVENT_RECOVERY,
                                          aalen_johansen_recovery, competing_risk_codes)
from src.utils.artifact_store import artifact_file

GRID = np.arange(0.0, 91.0, 1.0)


def test_censor_reasons_map_onto_competing_risk_codes():
    observed = np.array([True, False, False, False])
    reason = np.array(["recovered", "next_disaster", "90_day_cap", "no_drawdown"],
                      dtype=object)
    codes = competing_risk_codes(observed, reason)
    assert list(codes) == [EVENT_RECOVERY, EVENT_COMPETING, EVENT_CENSORED, EVENT_CENSORED]


def test_competing_risks_does_not_credit_recoveries_to_overtaken_events():
    """The property T2 exists for. When the competing event strikes early, Kaplan-Meier
    assumes those events would have recovered at the background rate; Aalen-Johansen knows
    they cannot. The competing-risks incidence must therefore be strictly lower."""
    from lifelines import KaplanMeierFitter

    durations = np.concatenate([np.full(20, 3.0), np.arange(10, 50, 2.0)])
    observed = np.concatenate([np.zeros(20, bool), np.ones(20, bool)])
    reason = np.array(["next_disaster"] * 20 + ["recovered"] * 20, dtype=object)

    competing = 1.0 - aalen_johansen_recovery(durations, competing_risk_codes(observed, reason), GRID)
    km = KaplanMeierFitter().fit(durations + 0.5, event_observed=observed)
    independent = 1.0 - np.atleast_1d(np.asarray(km.predict(GRID + 0.5), dtype=float))

    assert competing[-1] < independent[-1] - 0.05
    # Over the horizons the study reports, the competing-risks incidence never exceeds the
    # independent-censoring one. The two step functions can cross by a hair at the very
    # first grid points, where both are still essentially zero.
    horizons = GRID >= 20
    assert (competing[horizons] <= independent[horizons] + 1e-9).all()


def test_the_two_estimators_agree_when_there_is_no_competing_event():
    """No competing event, no difference. The estimator must not move a result that has
    nothing informative in it."""
    from lifelines import KaplanMeierFitter

    durations = np.concatenate([np.arange(5, 45, 2.0), np.full(10, 90.0)])
    observed = np.concatenate([np.ones(20, bool), np.zeros(10, bool)])
    reason = np.array(["recovered"] * 20 + ["90_day_cap"] * 10, dtype=object)
    codes = competing_risk_codes(observed, reason)
    assert not (codes == EVENT_COMPETING).any()

    competing = aalen_johansen_recovery(durations, codes, GRID)
    km = KaplanMeierFitter().fit(durations + 0.5, event_observed=observed)
    independent = np.atleast_1d(np.asarray(km.predict(GRID + 0.5), dtype=float))
    # Step placement differs by at most one grid point, hence the tolerance.
    np.testing.assert_allclose(competing, independent, atol=0.05)


def test_the_curve_is_a_valid_survival_function():
    durations = np.array([3.0, 5.0, 8.0, 12.0, 20.0, 40.0])
    reason = np.array(["recovered", "next_disaster", "recovered", "90_day_cap",
                       "recovered", "next_disaster"], dtype=object)
    observed = np.array([True, False, True, False, True, False])
    curve = aalen_johansen_recovery(durations, competing_risk_codes(observed, reason), GRID)
    assert ((curve >= 0.0) & (curve <= 1.0)).all()
    assert (np.diff(curve) <= 1e-9).all()          # non-increasing


def test_no_recovery_at_all_leaves_the_curve_flat():
    durations = np.array([5.0, 9.0, 15.0])
    codes = np.array([EVENT_COMPETING, EVENT_CENSORED, EVENT_COMPETING])
    np.testing.assert_allclose(aalen_johansen_recovery(durations, codes, GRID), 1.0)


def test_the_recovery_grid_carries_a_competing_risks_arm():
    path = artifact_file("recovery_grid_metrics.parquet")
    if not path.exists():
        pytest.skip("run scripts/run_recovery_survival_grid.py first")
    metrics = pd.read_parquet(path).set_index("model")
    assert "aalen_johansen_competing_risks" in metrics.index
    assert "km_train_baseline" in metrics.index

    calibration = pd.read_parquet(artifact_file("recovery_probability_calibration.parquet"))
    both = calibration[calibration["model"].isin(
        ["aalen_johansen_competing_risks", "km_train_baseline"])]
    wide = both.pivot(index="t", columns="model", values="predicted_P_T_le_t")
    # On this sample the competing-risks arm must not exceed the independent-censoring
    # arm at the long horizons, where the overtaken events would otherwise be credited.
    assert (wide["aalen_johansen_competing_risks"].iloc[-1]
            <= wide["km_train_baseline"].iloc[-1] + 1e-9)


def test_the_exclusion_sensitivity_run_is_written_separately():
    """T2 requires the drop-them treatment beside the model-them treatment, in its own
    artifact so neither overwrites the other."""
    path = artifact_file("recovery_grid_metrics_excl_competing.parquet")
    if not path.exists():
        pytest.skip("run scripts/run_recovery_survival_grid.py first")
    excluded = pd.read_parquet(path)
    primary = pd.read_parquet(artifact_file("recovery_grid_metrics.parquet"))
    assert excluded["n"].max() < primary["n"].max()
    assert set(excluded["model"]) == set(primary["model"])


def test_the_y3_regression_is_flagged_as_a_diagnostic_everywhere():
    """T3: no Y3 RMSE or R squared may appear in an exported table that is not labelled."""
    assert Y3_REGRESSION_IS_DIAGNOSTIC is True

    path = (artifact_file("dataset.parquet").parents[2] / "docs" / "thesis_materials"
            / "final_table_recovery_regression_diagnostic.csv")
    if not path.exists():
        pytest.skip("run scripts/build_final_tables.py first")
    table = pd.read_csv(path)
    assert (table["result_class"] == Y3_DIAGNOSTIC_LABEL).all()
    assert (table["censoring_aware"] == False).all()          # noqa: E712
    assert (table["target"] == MARKET_RECOVERY_DAYS).all()
    # Every error column present must carry the diagnostic marker in its own name.
    for column in table.columns:
        if "RMSE" in column or "R2" in column:
            assert "diagnostic" in column, column


def test_the_primary_recovery_table_is_the_survival_one():
    path = (artifact_file("dataset.parquet").parents[2] / "docs" / "thesis_materials"
            / "final_table_recovery.csv")
    if not path.exists():
        pytest.skip("run scripts/build_final_tables.py first")
    table = pd.read_csv(path)
    if "result_class" not in table.columns:
        pytest.skip("stale table, re-run scripts/build_final_tables.py")
    assert (table["result_class"] == "primary (censoring-aware survival)").all()
    assert "C_index" in table.columns
    for column in table.columns:
        if "RMSE" in column or "MAE" in column:
            assert "diagnostic" in column or "uncensored" in column, column
