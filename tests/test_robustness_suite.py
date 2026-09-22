"""T15: the robustness suite is a closed, pre-declared list, and the table proves it.

The list lives in docs/audit.md Part 8.7. These tests check that what ran matches what was
declared, that nothing was quietly dropped, and that every check reports the four things it
is required to report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.artifact_store import artifact_file

pytestmark = pytest.mark.skipif(
    not (artifact_file("robustness_suite.parquet")).exists(),
    reason="run scripts/run_robustness_suite.py first")

# Every check the pre-declaration names, by its `check` label in the table.
DECLARED_CHECKS = {
    "principal",
    "exact dates",
    "horizon",
    "return definition",
    "information set",
    "crisis period",
    "window mode",
    "influential events",
    "disaster type",
    "event clustering",
    "oversampling",
    "missingness",
}


@pytest.fixture(scope="module")
def suite():
    return pd.read_parquet(artifact_file("robustness_suite.parquet"))


def test_every_declared_check_is_present(suite):
    """Nothing from the closed list may be missing, whatever its result."""
    assert set(suite["check"]) == DECLARED_CHECKS, set(suite["check"]) ^ DECLARED_CHECKS


def test_no_check_was_added_after_the_declaration(suite):
    """The list is closed in both directions: nothing extra either."""
    assert not set(suite["check"]) - DECLARED_CHECKS


def test_every_row_reports_its_sample_size(suite):
    assert suite["n_held_out"].notna().all()
    assert (suite["n_held_out"] >= 0).all()


def test_a_check_that_could_not_run_says_why_instead_of_being_omitted(suite):
    """The acceptance criterion: an unrunnable check appears WITH a reason."""
    unrunnable = suite[suite["n_held_out"] == 0]
    assert len(unrunnable) > 0, "expected some subgroups to be too small to run"
    assert unrunnable["note"].str.len().gt(0).all()
    assert unrunnable["delta_rmse"].isna().all()


def test_every_runnable_check_carries_an_uncertainty_interval(suite):
    runnable = suite[suite["n_held_out"] > 0]
    assert len(runnable) > 0
    assert runnable["ci_low"].notna().all()
    assert runnable["ci_high"].notna().all()
    assert (runnable["ci_low"] <= runnable["ci_high"]).all()


def test_every_runnable_check_compares_against_the_benchmark(suite):
    runnable = suite[(suite["n_held_out"] > 0) & suite["delta_rmse"].notna()]
    assert runnable["rmse_model"].notna().all()
    assert runnable["rmse_benchmark"].notna().all()
    np.testing.assert_allclose(
        runnable["delta_rmse"].to_numpy(),
        (runnable["rmse_benchmark"] - runnable["rmse_model"]).to_numpy(), atol=1e-9)


def test_small_subgroups_are_reported_with_their_size_and_never_as_a_finding(suite):
    """Drought (5) and Other (3) are too small to support a standalone comparison, and the
    pre-declaration says so. They must still appear, with their size stated."""
    subgroups = suite[suite["check"] == "disaster type"]
    assert len(subgroups) == 4
    for name in ("Flood", "Storm", "Drought", "Other"):
        row = subgroups[subgroups["variant"].str.startswith(name)]
        assert len(row) == 1, name
        assert "n=" in row["variant"].iloc[0], name
    for name in ("Drought", "Other"):
        row = subgroups[subgroups["variant"].str.startswith(name)].iloc[0]
        assert not row["beats_benchmark"], name
        assert "too small" in row["note"], name


def test_both_window_modes_were_run(suite):
    modes = suite[suite["check"] == "window mode"]
    assert set(modes["variant"]) == {"sliding", "expanding"}
    assert (modes["n_held_out"] > 0).all()


def test_raw_and_abnormal_returns_were_compared_on_the_same_events(suite):
    """The comparison is only meaningful on identical held-out events."""
    definitions = suite[suite["check"] == "return definition"]
    assert len(definitions) == 2
    assert definitions["n_held_out"].nunique() == 1


def test_both_availability_information_sets_were_run(suite):
    sets = suite[suite["check"] == "information set"]
    assert set(sets["variant"]) == {"real_time", "ex_post"}
    assert sets["n_held_out"].nunique() == 1
