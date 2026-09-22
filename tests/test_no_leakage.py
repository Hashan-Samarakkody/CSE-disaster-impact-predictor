"""T13: the leakage guards, promoted out of a notebook cell and into the suite.

`assert_no_target_leakage` existed and worked, but ran only inside
notebooks/02_features_targets.ipynb, so it never fired under pytest. A check that only
runs when someone opens a notebook is not a check. Nothing here executes a notebook.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config.settings import (NON_FEATURE_COLS, TARGET_LABEL_END_DATE_COL,
                                 assert_no_target_leakage)
from src.training.walk_forward import WalkForwardSplit, purge_horizon_overlap
from src.utils.artifact_store import artifact_file

pytestmark = pytest.mark.skipif(
    not (artifact_file("feature_spec.json")).exists(),
    reason="run notebooks/02_features_targets.ipynb first")


@pytest.fixture(scope="module")
def spec():
    return json.loads((artifact_file("feature_spec.json")).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dataset():
    return pd.read_parquet(artifact_file("dataset.parquet"))


def test_no_outcome_column_reached_the_feature_matrix(spec, dataset):
    """The persisted spec, not a fresh in-memory computation: what the models were
    actually fitted on must contain no target, audit, censoring or label-end column."""
    features = set(spec["FEATURE_COLS"])
    leaked = sorted(features & NON_FEATURE_COLS)
    assert leaked == [], leaked
    assert_no_target_leakage(spec["FEATURE_COLS"])

    # And the same columns really are present in the table the models read.
    assert features <= set(dataset.columns)


def test_the_guard_actually_fires_when_a_target_is_reintroduced(spec):
    """A guard nobody has seen fail is not known to work. Deliberately reintroduce the
    defect, confirm it raises, and leave the real state untouched."""
    for offender in ("Y1_ASPI_5D_Forward_LogReturn_Pct", "Y2_V_future5",
                     "Y3_censor_reason", "Y1_horizon_end_date"):
        with pytest.raises(AssertionError, match="leaked"):
            assert_no_target_leakage(list(spec["FEATURE_COLS"]) + [offender])

    # The real feature set is still clean afterwards.
    assert_no_target_leakage(spec["FEATURE_COLS"])


def test_every_feature_is_dated_at_or_before_the_prediction_origin(spec, dataset):
    """Every feature describes the state of the world at the prediction origin. The event
    level table carries one row per event, snapshotted at `asof_date`, so the check is
    that no feature column is a post-origin timestamp and that the snapshot date itself
    precedes the origin for every event."""
    origin = pd.to_datetime(dataset["prediction_origin_session"])
    event_date = pd.to_datetime(dataset["event_date"])

    # The market state is snapshotted the day before the event, which is at or before the
    # origin session for every event.
    assert (event_date <= origin).all()

    # No feature column may itself be a date, which would carry a settlement time.
    for column in spec["FEATURE_COLS"]:
        assert not pd.api.types.is_datetime64_any_dtype(dataset[column]), column

    # Every label-end date, the one thing that IS post-origin, is a declared non-feature.
    for column in TARGET_LABEL_END_DATE_COL.values():
        assert column not in spec["FEATURE_COLS"], column
        assert (pd.to_datetime(dataset[column]).dropna()
                >= origin[pd.to_datetime(dataset[column]).notna()]).all(), column


def test_the_purge_drops_a_training_event_whose_label_reaches_into_the_test_period():
    """The embargo, on a fold built for the purpose. The training event settles after the
    first test event begins, so it must be dropped; the one that settles before must not."""
    event_dates = pd.Series(pd.to_datetime(
        ["2020-01-01", "2020-02-01", "2020-03-01", "2020-06-01"]))
    # Row 1's label settles AFTER the first test event (row 2) begins; row 0's does not.
    label_end = pd.Series(pd.to_datetime(
        ["2020-01-15", "2020-04-01", "2020-03-20", "2020-06-20"]))
    split = WalkForwardSplit(np.array([0, 1]), np.array([2, 3]))

    purged = purge_horizon_overlap(split, event_dates, label_end)
    assert list(purged.train_index) == [0]
    assert list(purged.test_index) == [2, 3]


def test_the_purge_is_the_thing_doing_the_work():
    """Reintroduce the defect: without the purge, the contaminated row survives. This is
    the failure the test above is guarding against."""
    event_dates = pd.Series(pd.to_datetime(
        ["2020-01-01", "2020-02-01", "2020-03-01", "2020-06-01"]))
    label_end = pd.Series(pd.to_datetime(
        ["2020-01-15", "2020-04-01", "2020-03-20", "2020-06-20"]))
    split = WalkForwardSplit(np.array([0, 1]), np.array([2, 3]))

    unpurged = list(split.train_index)
    purged = list(purge_horizon_overlap(split, event_dates, label_end).train_index)
    assert 1 in unpurged and 1 not in purged


def test_every_target_family_column_in_the_dataset_is_a_declared_non_feature(dataset, spec):
    """Feature construction is a denylist, so a target-family column missing from
    NON_FEATURE_COLS is silently admitted. Catch the omission here rather than in a result."""
    suspicious = [c for c in dataset.columns
                  if c.startswith(("Y1_", "Y2_", "Y3_", "market_model_"))]
    for column in suspicious:
        assert column not in spec["FEATURE_COLS"], column
        assert column in NON_FEATURE_COLS, f"{column} is not a declared non-feature"
