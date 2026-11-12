"""Regression tests for analysis labels and per-target sample accounting."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.build_analysis_map import build_analysis_map
from scripts.build_sample_accounting import build_sample_accounting
from scripts.run_abnormal_return_variants import run_variant


def test_analysis_map_uses_source_metadata_and_separates_primary_comparisons():
    grid = pd.DataFrame([
        {
            "horizon": 5,
            "info_set": info_set,
            "k": 10,
            "model": model,
            "baseline": baseline,
            "confirmatory": True,
            "n": 40,
            "p_one_sided": 0.5,
            "p_holm": 1.0,
            "holm_significant": False,
        }
        for info_set in ("real_time", "ex_post")
        for model in ("ridge", "random_forest", "mlp")
        for baseline in ("naive_zero", "naive_train_mean",
                         "naive_train_median", "market_only_expected")
    ])
    old_results = pd.DataFrame([{
        "target": "Y1_ASPI_5D_Forward_LogReturn_Pct",
        "model": "ridge",
        "baseline": "naive_zero",
        "n": 40,
        "confirmatory": True,
        "p_model_worse": 0.4,
        "p_holm": 1.0,
        "holm_significant": False,
    }])
    direction = pd.DataFrame([
        {
            "horizon": horizon,
            "model": model,
            "n": 40,
            "p_one_sided": 0.5,
            "p_holm": 1.0,
            "holm_significant": False,
        }
        for horizon in range(1, 6)
        for model in ("logistic", "random_forest")
    ])
    classification = pd.DataFrame([
        {"label": "C1", "model": "majority_baseline", "n": 34},
        {"label": "C1", "model": "random_forest", "n": 34},
    ])

    result = build_analysis_map(old_results, grid, direction, classification)

    primary = result[result["source_artifact"] == "aspi_grid_verdicts.parquet"]
    assert len(primary) == 24
    assert set(primary["info_set"]) == {"real_time", "ex_post"}
    assert set(primary["k"]) == {10}
    assert set(primary["tier"]) == {"confirmatory"}
    assert set(primary["family_size"]) == {24}

    older = result[result["source_artifact"] == "verdict_table.parquet"]
    assert set(older["tier"]) == {"exploratory"}
    assert set(older["info_set"]) == {"combined"}
    assert set(older["k"]) == {20}

    direction_rows = result[
        result["source_artifact"] == "aspi_direction_metrics.parquet"]
    assert set(direction_rows["info_set"]) == {"combined"}
    assert set(direction_rows["k"]) == {10}
    assert set(direction_rows["family_size"]) == {10}

    classification_rows = result[
        result["source_artifact"] == "classification_summary.parquet"]
    assert len(classification_rows) == 1
    assert classification_rows.iloc[0]["info_set"] == "combined"
    assert classification_rows.iloc[0]["k"] == 20


def test_sample_accounting_counts_only_target_valid_held_out_rows():
    n = 74
    volume = np.arange(n, dtype=float)
    volume[[0, 1, 2, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73]] = np.nan
    volume_baseline = volume.copy()
    volume_baseline[73] = 73.0
    dataset = pd.DataFrame({
        "Y1_ASPI_5D_Forward_LogReturn_Pct": np.ones(n),
        "Y1_ASPI_10D_Forward_LogReturn_Pct": np.ones(n),
        "Y2_5D_Forward_AbnormalVolume_LogRatio": volume,
        "Y2_V_base": volume_baseline,
        "Y2_V_future5": volume,
        "Y3_ASPI_Recovery_Time": np.ones(n),
    })
    splits = [
        SimpleNamespace(train_index=np.arange(start, start + 30),
                        test_index=np.arange(test_start, test_start + 10))
        for start, test_start in ((0, 30), (10, 40), (20, 50), (30, 60))
    ]

    result = build_sample_accounting(dataset, splits)
    volume_row = result[result["target"] == "Y2_5D"].iloc[0]

    assert volume_row["n_labelled"] == 61
    assert volume_row["n_with_baseline"] == 62
    assert volume_row["n_with_forward_window"] == 61
    assert volume_row["n_held_out"] == 34
    assert volume_row["n_training_eligible"] == 27


def test_zero_expected_volume_uses_log_changes_not_log_volume_levels():
    sessions = pd.date_range("2020-01-01", periods=360, freq="B")
    origin = 200
    log_volume = np.full(len(sessions), 10.0)
    log_volume[origin:] += np.log(2.0)
    daily = pd.DataFrame({
        "date": sessions,
        "log_volume": log_volume,
        "log_return": np.zeros(len(sessions)),
        "sp500_log_return": np.zeros(len(sessions)),
    })

    result = run_variant(daily, [sessions[origin]], "zero_expected", "volume")
    first_event_session = result[result["event_time"] == 1].iloc[0]

    assert first_event_session["caar"] == pytest.approx(np.log(2.0))
