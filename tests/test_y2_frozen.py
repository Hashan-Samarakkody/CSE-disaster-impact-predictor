"""Y2 must not move.

`Y2_abnormal_volume` is the study's one statistically supported continuous target
(`docs/RESULTS_AUDIT.txt` sec. 2: GP and SVR both beat both naive baselines with a
bootstrap CI excluding zero). The Y1/Y3 improvement work (2026-09-17) is allowed to touch
shared infrastructure; it is NOT allowed to change a single Y2 number while doing so.

Every assertion below compares the LIVE artifacts against `artifacts/frozen_baseline.json`
(written by `scripts/freeze_baseline.py` at commit 1fbf6275, before any improvement work).
A failure here is a stop-and-diagnose signal, not a number to re-freeze: regenerate the
baseline only after the cause is understood and documented in
`docs/Y2_FROZEN_VALIDATION_REPORT.md`.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
BASELINE = ART / "frozen_baseline.json"
TARGET = "Y2_abnormal_volume"

# Float tolerance. Y2's own scale is ~0.5 (a volume ratio), so 1e-9 is far below any
# numerically meaningful change while still absorbing platform float noise.
ATOL = 1e-9


@pytest.fixture(scope="module")
def frozen():
    if not BASELINE.exists():
        pytest.skip(f"{BASELINE.name} missing -- run scripts/freeze_baseline.py first")
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live_results():
    return pickle.loads((ART / "results_regression.pkl").read_bytes())


def _arr(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def test_y2_target_values_unchanged(frozen):
    live = pd.read_parquet(ART / "dataset.parquet")[TARGET].to_numpy(dtype=float)
    want = _arr(frozen["targets"][TARGET])
    assert len(live) == len(want), f"event count moved: {len(want)} -> {len(live)}"
    np.testing.assert_allclose(live, want, atol=ATOL, equal_nan=True)


def test_y2_event_sample_unchanged(frozen):
    dataset = pd.read_parquet(ART / "dataset.parquet")
    assert [str(d.date()) for d in pd.to_datetime(dataset["event_date"])] == frozen["event_dates"]


def test_y2_fold_definitions_unchanged(frozen):
    splits = pickle.loads((ART / "splits.pkl").read_bytes())
    live = [{"train": [int(i) for i in np.asarray(s.train_index)],
             "test": [int(i) for i in np.asarray(s.test_index)]} for s in splits]
    assert live == frozen["folds"]


@pytest.mark.parametrize("model", ["ridge", "random_forest", "xgboost", "gp", "svr",
                                   "quantile", "mlp", "ensemble", "stacked",
                                   "naive_zero", "naive_train_mean"])
def test_y2_pooled_oof_predictions_unchanged(frozen, live_results, model):
    store = live_results[model][TARGET]
    want = frozen["models"][model][TARGET]
    np.testing.assert_allclose(np.concatenate(store["y_true"]), _arr(want["y_true"]),
                               atol=ATOL, equal_nan=True)
    np.testing.assert_allclose(np.concatenate(store["y_pred"]), _arr(want["y_pred"]),
                               atol=ATOL, equal_nan=True)


@pytest.mark.parametrize("metric", ["rmse", "mae", "r2"])
def test_y2_per_fold_metrics_unchanged(frozen, live_results, metric):
    for model, per_target in live_results.items():
        np.testing.assert_allclose(_arr(per_target[TARGET][metric]),
                                   _arr(frozen["models"][model][TARGET][metric]),
                                   atol=ATOL, equal_nan=True,
                                   err_msg=f"{model}/{TARGET}/{metric}")


def test_y2_pooled_r2_unchanged(frozen, live_results):
    for model, per_target in live_results.items():
        assert per_target[TARGET]["pooled_r2"] == pytest.approx(
            frozen["models"][model][TARGET]["pooled_r2"], abs=ATOL), model


def test_y2_bootstrap_verdict_unchanged(frozen):
    live = pd.read_parquet(ART / "verdict_table.parquet")
    live = live[live["target"] == TARGET].reset_index(drop=True)
    want = pd.DataFrame([r for r in frozen["verdict"] if r["target"] == TARGET])
    assert len(live) == len(want), "Y2 verdict row count moved"
    key = ["model", "baseline"]
    live, want = live.sort_values(key).reset_index(drop=True), want.sort_values(key).reset_index(drop=True)
    for col in ["delta_rmse", "ci_low", "ci_high", "p_model_worse", "rmse_model", "rmse_baseline"]:
        np.testing.assert_allclose(live[col].to_numpy(float), want[col].to_numpy(float),
                                   atol=ATOL, equal_nan=True, err_msg=col)
    for col in ["verdict", "boot_beats", "holm_significant"]:
        assert list(live[col]) == list(want[col]), col


def test_y2_classification_unchanged(frozen):
    """C2_volume_spike is Y2's classification arm -- AUC and balanced accuracy frozen too."""
    live = pd.read_parquet(ART / "classification_summary.parquet")
    live = live[live["label"] == "C2_volume_spike"].sort_values("model").reset_index(drop=True)
    want = pd.DataFrame([r for r in frozen["classification"] if r["label"] == "C2_volume_spike"]
                        ).sort_values("model").reset_index(drop=True)
    assert list(live["model"]) == list(want["model"])
    for col in ["auc", "balanced_accuracy", "auc_boot_lo", "auc_boot_hi", "mcc", "pr_auc"]:
        np.testing.assert_allclose(live[col].to_numpy(float), want[col].to_numpy(float),
                                   atol=ATOL, equal_nan=True, err_msg=col)
    assert list(live["beats_baseline"]) == list(want["beats_baseline"])
