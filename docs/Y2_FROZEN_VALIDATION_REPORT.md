# Y2 frozen-validation report

**Verdict: NO MATERIAL CHANGE. No change of any kind, at 1e-9 tolerance.**

`Y2_abnormal_volume` is the study's one statistically supported continuous target. The
2026-09-17 Y1/Y3 improvement work was required to leave it untouched. This report is the
before/after evidence, and `tests/test_y2_frozen.py` is the mechanical enforcement.

* **Before** = commit `1fbf6275`, captured in `artifacts/frozen_baseline.json`
  (`scripts/freeze_baseline.py`), taken before a single line of improvement code existed.
* **After** = the same artifacts re-read at the end of the improvement work.
* **Tolerance** = `1e-9` absolute, on a target whose own scale is ~0.5. This is float
  noise, not a materiality threshold.

## How Y2 was protected

The improvement work adds files; it changes no file the frozen pipeline reads.

* The four Y1 horizon targets are built **in memory** from `artifacts/market.parquet`
  (`src/evaluation/y1_horizons.py`), not by regenerating `artifacts/dataset.parquet`.
  `dataset.parquet` — which is where `Y2_abnormal_volume` lives — is never rewritten, so
  Y2's target values cannot move.
* `src/data_pipeline/feature_eng.py`, `src/training/walk_forward.py`,
  `src/evaluation/verification.py` and `notebooks/_shared.py` were **read, not edited**.
  Shared infrastructure is consumed by the new scripts, never modified by them.
* `src/evaluation/survival_metrics.py` and `src/evaluation/y1_horizons.py` are new
  modules with no importer inside the frozen pipeline.
* `notebooks/04_modeling_regression.ipynb`, `notebooks/05_modeling_classification.ipynb`
  and `scripts/run_survival_model.py` were not re-run, so `results_regression.pkl`,
  `verdict_table.parquet` and `classification_summary.parquet` are byte-identical to the
  freeze.

The only shared-file edit anywhere in this work is an **additive** keyword argument
(`return_draws=False`) on `survival_metrics.cluster_bootstrap_ci`, a module that did not
exist at freeze time and that nothing in the Y2 path imports.

## Before vs after

| Quantity | Before | After | Change |
|---|---|---|---|
| Target values (all 74 rows) | see `frozen_baseline.json` | identical | **0** |
| Observations scored (pooled OOF) | 34 | 34 | **0** |
| Fold definitions (4 folds, train/test index arrays) | identical | identical | **0** |
| Pooled OOF predictions, all 11 models | identical | identical | **0** |

### Regression metrics (pooled out-of-fold, n = 34)

| Model | RMSE before | RMSE after | MAE before | MAE after | R2 before | R2 after |
|---|---|---|---|---|---|---|
| gp | 0.45535 | 0.45535 | 0.36114 | 0.36114 | +0.27042 | +0.27042 |
| svr | 0.45790 | 0.45790 | 0.36431 | 0.36431 | +0.26222 | +0.26222 |
| random_forest | 0.48548 | 0.48548 | 0.39716 | 0.39716 | +0.17068 | +0.17068 |
| ensemble | 0.51955 | 0.51955 | 0.40637 | 0.40637 | +0.05017 | +0.05017 |
| xgboost | 0.54343 | 0.54343 | 0.44155 | 0.44155 | -0.03912 | -0.03912 |
| ridge | 0.54557 | 0.54557 | 0.45165 | 0.45165 | -0.04733 | -0.04733 |
| mlp | 0.62240 | 0.62240 | 0.47767 | 0.47767 | -0.36308 | -0.36308 |

### Bootstrap CI and statistical verdict (episode-clustered, 10,000 draws)

| Model | Baseline | delta RMSE | CI low | CI high | Verdict | Holm |
|---|---|---|---|---|---|---|
| gp | naive_zero | +0.08684 | +0.02521 | +0.15398 | **BEATS BASELINE** | False |
| gp | naive_train_mean | +0.09309 | +0.01229 | +0.16685 | **BEATS BASELINE** | False |
| svr | naive_zero | +0.08429 | +0.03435 | +0.14178 | **BEATS BASELINE** | **True** |
| svr | naive_train_mean | +0.09054 | +0.02991 | +0.14331 | **BEATS BASELINE** | False |
| random_forest | naive_zero | +0.05671 | -0.00948 | +0.12244 | better, not distinguishable | False |
| random_forest | naive_train_mean | +0.06296 | -0.01850 | +0.13280 | better, not distinguishable | False |
| ridge | naive_zero | -0.00338 | -0.09906 | +0.07457 | worse than baseline | False |
| xgboost | naive_zero | -0.00123 | -0.09488 | +0.09718 | worse than baseline | False |
| mlp | naive_zero | -0.08021 | -0.20958 | +0.04571 | worse than baseline | False |
| quantile | naive_zero | -0.35666 | -0.54618 | -0.15812 | worse than baseline | False |

All 18 Y2 rows are identical before and after. Nine of them are omitted from the table for
length only; `tests/test_y2_frozen.py::test_y2_bootstrap_verdict_unchanged` asserts every
one of them.

### Classification arm (`C2_volume_spike`)

| Model | AUC before | AUC after | Balanced acc. before | Balanced acc. after | Beats baseline |
|---|---|---|---|---|---|
| logistic | 0.7107 | 0.7107 | 0.5643 | 0.5643 | True |
| rf_clf | 0.6964 | 0.6964 | 0.6286 | 0.6286 | False |
| xgb_clf | 0.6893 | 0.6893 | 0.5071 | 0.5071 | False |
| majority_baseline | 0.5000 | 0.5000 | 0.5000 | 0.5000 | False |

## Mechanical enforcement

`tests/test_y2_frozen.py` — 20 tests, all passing — asserts, against
`artifacts/frozen_baseline.json`:

1. `test_y2_target_values_unchanged` — all 74 Y2 values.
2. `test_y2_event_sample_unchanged` — the 74 event dates, in order.
3. `test_y2_fold_definitions_unchanged` — every fold's train and test index arrays.
4. `test_y2_pooled_oof_predictions_unchanged` — `y_true` and `y_pred`, per model (11).
5. `test_y2_per_fold_metrics_unchanged` — RMSE, MAE, R2, per fold, per model.
6. `test_y2_pooled_r2_unchanged` — pooled R2 per model.
7. `test_y2_bootstrap_verdict_unchanged` — delta, CI bounds, one-sided p, verdict string,
   `boot_beats`, `holm_significant`.
8. `test_y2_classification_unchanged` — C2 AUC, balanced accuracy, AUC CI, MCC, PR-AUC.

Full suite at the end of the work: **119 passed** (99 pre-existing + 20 new).

If a future change makes any of these fail, that is a stop-and-diagnose signal. The
baseline is not to be regenerated to make the test pass; the cause is to be found,
documented in this file, and only then re-frozen.
