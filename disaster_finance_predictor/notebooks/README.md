# Pipeline stages

The analysis used to live in one 59-cell notebook. That made every figure tweak cost a
full re-execution, including re-parsing 24 Excel workbooks and re-hitting two live APIs.
It is now a numbered pipeline where each stage writes its outputs to `../artifacts/` and
later stages read them back.

Run them in order. Each stage prints the cache contents on entry, so it is always visible
which upstream stage produced the inputs and when.

| Stage | Notebook | Produces | Cost |
|---|---|---|---|
| 01 | `01_data_acquisition.ipynb` | `market`, `disasters`, `macro`, `sp500` | **Slow** — Excel archive + live World Bank / Yahoo |
| 02 | `02_features_targets.ipynb` | `dataset`, `market_feats`, `in_scope`, `feature_spec`, `splits` | Fast |
| 03 | `03_eda_diagnostics.ipynb` | `collinearity_rule`, descriptive figures | Fast, fits nothing |
| 04 | `04_modeling_regression.ipynb` | `results_regression`, `results_ablations`, `selected_features`, `train_fit_r2` | **Slow** — every model family × every fold |
| 05 | `05_modeling_classification.ipynb` | `results_classification`, `classification_summary`, `hurdle_table` | Medium |
| 06 | `06_evaluation.ipynb` | `verdict_table`, `conformal_coverage`, model figures | Fast, refits nothing |
| 07 | `07_explainability.ipynb` | SHAP figures | Medium |
| 08 | `08_sector_panel.ipynb` | `results_sector`, sector figures | Slow |
| 09 | `09_synthesis.ipynb` | — (written record only) | Instant |

To execute one stage:

```bash
cd disaster_finance_predictor/notebooks
python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3000 04_modeling_regression.ipynb
```

## What lives where

`_shared.py` holds the paths, the artifact load/save helpers, `RANDOM_STATE`, the target
bounds and the clipping function. Every notebook opens with `from _shared import *`, so
none of them re-declares any of it.

Reusable logic lives in `../src/`, not in the notebooks:

- `src/evaluation/figures.py` — every figure, plus `apply_thesis_style()` and PNG export
- `src/evaluation/metrics.py` — pooling, bootstrap CIs, conformal and Wilson intervals
- `src/evaluation/collinearity.py` — correlation, VIF, and the pre-declared redundancy rule
- `src/evaluation/verification.py` — paired bootstrap and Diebold–Mariano baseline tests
- `src/models/classifiers.py` — the pre-registered binary labels and their metrics
- `src/models/hurdle.py` — the two-stage Y3 model

`python -m pytest tests/ -q` covers all of it and needs no data files and no notebook run.

## Reproducibility

Stage 01 pulls World Bank and Yahoo data live from unpinned libraries, so the same code
can return different numbers months apart. Every cached artifact therefore carries a
`.provenance.json` sidecar recording the retrieval timestamp, the Python version and the
version of every relevant library. Re-running stage 01 overwrites both.

`../artifacts/` is regenerable and is not committed. Figures in `../../docs/figures/`
**are** committed, so the thesis can cite them by a stable filename.

## The archived notebook

`archive/CSE_Disaster_Impact_Pipeline_v3.ipynb` is the executed pre-split monolith, kept
as the record of the results reported before the split. Two defects were found in it and
fixed in stage 04, so its numbers should not be quoted:

- the ensemble blend weighted members by the held-out RMSE of the very fold it was
  predicting, which is selection on test performance;
- the feature table contained no volume-derived column at all, so Y2 was predicted with
  no information about its own driving series.
