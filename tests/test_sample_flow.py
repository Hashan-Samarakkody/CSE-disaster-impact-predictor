"""T12: the sample selection accounting must reconcile with the pipeline it describes."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.feature_engineering import build_sample_flow
from src.utils.artifact_store import artifact_file

pytestmark = pytest.mark.skipif(
    not (artifact_file("sample_flow.parquet")).exists(),
    reason="run notebooks/02_features_targets.ipynb first")


@pytest.fixture(scope="module")
def flow():
    return pd.read_parquet(artifact_file("sample_flow.parquet"))


def test_every_stage_subtracts_exactly_what_it_removed(flow):
    """The whole point of the table is that it closes. A stage whose remaining count does
    not equal the previous remaining minus its own removals is an accounting error."""
    for i in range(1, len(flow)):
        previous = flow["n_remaining"].iloc[i - 1]
        assert previous - flow["n_removed"].iloc[i] == flow["n_remaining"].iloc[i], \
            flow["stage"].iloc[i]


def test_the_counts_are_the_ones_the_study_reports(flow):
    """110 raw records, 94 after the two inclusion filters, 74 modelled."""
    assert flow["n_remaining"].iloc[0] == 110
    after_inclusion = flow.loc[flow["stage"] == "below the affected threshold", "n_remaining"]
    assert int(after_inclusion.iloc[0]) == 94
    assert flow["n_remaining"].iloc[-1] == 74


def test_the_final_count_matches_the_modelling_table(flow):
    """The table can never disagree with the pipeline: the last remaining count is the
    number of rows the models are actually fitted on."""
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))
    assert flow["n_remaining"].iloc[-1] == len(dataset)


def test_recomputing_the_flow_reproduces_the_cached_table(flow):
    """The artifact is not a hand-maintained note; recomputing it from the same inputs
    must give the same answer."""
    rebuilt = build_sample_flow(pd.read_parquet(artifact_file("disasters.parquet")),
                                pd.read_parquet(artifact_file("market.parquet")))
    pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True),
                                  flow.reset_index(drop=True), check_dtype=False)


def test_every_stage_names_the_criterion_it_applied(flow):
    """A count with no stated criterion is not an accounting."""
    assert flow["criterion"].str.len().gt(0).all()
    assert flow["stage"].is_unique
