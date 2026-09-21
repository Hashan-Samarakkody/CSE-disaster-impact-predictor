# Experiments

The nine notebooks are the pipeline. The scripts in `scripts/` are the experiments run on
top of it. This file explains what each one does, what it writes, and which claim in
`docs/results.md` it supports.

Every script is additive. None of them rewrites an artifact the notebooks read, which is
why the frozen volume target holds by construction rather than by care.

## 1. The frozen baseline

`scripts/freeze_baseline.py`

Writes `artifacts/results/frozen_baseline.json`, a snapshot of the pipeline: the commit
hash, every event id and date, every target value, the fold definitions, every model's
pooled out of fold predictions, the per fold and pooled metrics, the bootstrap verdict
rows and the classification summary.

It exists because work on the return and recovery targets is allowed to touch shared
infrastructure but not to move the volume target.
`tests/test_volume_target_frozen.py` reads this file back and fails if any volume number
changes. That snapshot cannot be rebuilt from a later state, so it is the one file under
`artifacts/` that is version controlled.

The current snapshot was taken on 2026-09-21 at commit `f076bd9`, replacing the one taken
at commit `1fbf6275`, which pinned the pre protocol volume definition. Run this script
once per deliberate, pre declared change to a target definition, and never otherwise:
re running it overwrites the reference and defeats its purpose.

## 2. The return prediction grid

`scripts/run_aspi_return_grid.py`, about seventy five minutes.

Executes the pre declared grid in `docs/audit.md` Part 2: four horizons of five, ten,
fifteen and twenty trading sessions, four information sets, three feature capacities and
five model families. That is 240 configurations, each scored on the same chronological
folds and the same test events, each purged against its own horizon, so the twenty session
horizon carries a twenty session embargo.

The four information sets are the point of the script. Market only uses strictly pre event
market, macro, exchange rate and global market state. Disaster only uses event identity,
severity, exposure and hazard. Combined is the union. The fourth decomposes the target into
a normal market component, estimated from the daily index series using only sessions whose
own label settled strictly before the event, and a disaster residual modelled on top.

It writes five tables to `artifacts/tables/`:

1. `aspi_grid_predictions.parquet`, one row per configuration and test event
2. `aspi_grid_metrics.parquet`, pooled metrics per configuration
3. `aspi_grid_verdicts.parquet`, all 720 paired bootstrap comparisons with Holm correction
4. `aspi_grid_feature_stability.parquet`, how often each feature was selected across folds
5. `aspi_direction_metrics.parquet`, the secondary direction classification analysis

It also caches `aspi_expected_return_market_only.parquet`, the per event expected return estimates,
because those depend only on frozen inputs and cost three minutes to recompute.

Supports: the finding that return magnitude is not predictable, that no configuration beats
its baseline, and the one positive result in the study, that return direction at ten
sessions is predictable above chance.

## 3. The recovery survival grid

`scripts/run_recovery_survival_grid.py`, about one minute.

Replaces point regression on recovery duration with censoring aware survival analysis, on
genuine events only. Synthetic oversampled rows are excluded from every survival fit,
because interpolation produces a duration but not a valid event indicator, so a synthetic
row would assert a recovery that was never observed.

Fits fifteen models across three feature capacities: Weibull and log normal accelerated
failure time, a two stage architecture that first estimates the probability of a drawdown
and then models duration for the drawdown cases, a penalised Cox model fitted only where
the event count supports it, and two baselines, a training fold Kaplan Meier curve and a
training fold median.

Writes `recovery_grid_predictions.parquet`, `recovery_grid_metrics.parquet`,
`recovery_probability_calibration.parquet` and `recovery_category_metrics.parquet`.

Supports: the finding that exact recovery duration cannot be predicted, and the ranking
result, which under the frozen target protocol no longer clears chance. The best
concordance interval still contains 0.5, so the recovery target is reported as not
predictable rather than as suggestive.

## 4. Final tables

`scripts/build_final_tables.py`

Assembles the two grids into the tables the thesis quotes, writing
`docs/thesis_materials/final_table_aspi.csv`,
`docs/thesis_materials/final_table_recovery.csv` and the markdown fragments in
`docs/results.md`. It refits nothing and retypes no number by hand.

## 5. Supporting scripts

`scripts/audit_results.py` regenerates the full metric audit, every model against every
target, that forms the last part of `docs/results.md`.

`scripts/train_final_models.py` refits the regression, classification and hurdle models on
all real rows and saves the bundle the demo app loads. These are in sample fits, used for
explanation and demonstration only, and are never scored as predictions.

`scripts/make_architecture_diagram.py` redraws `docs/images/pipeline.png` from the stage
table in the script itself, so the diagram cannot drift from the documented pipeline.

## 6. Recorded earlier experiments

These four scripts are the recorded runs behind specific sections of `docs/audit.md`. They
are kept because the audit cites their numbers, including the ones that failed.

`scripts/run_survival_model.py` is the first censoring aware treatment of the recovery
target, superseded as the primary analysis by the grid in section 3 but retained as the
comparison it is cited as.

`scripts/run_garch_ablation.py` measures whether the conditional volatility feature earns
its place.

`scripts/run_aspi_return_experiments.py` is the earlier controlled return experiment series, run
before the pre declared grid and reported as such.

`scripts/run_headline_confirmation.py` re confirms the return and volume headline numbers
against the cached predictions.

## 7. Reproducing everything

```bash
python -m pytest tests/ -q
python scripts/run_aspi_return_grid.py
python scripts/run_recovery_survival_grid.py
python scripts/build_final_tables.py
python scripts/audit_results.py
```

The seed is fixed at 42 in `src/config/settings.py` and is never varied to obtain a better
score. Any script that writes an artifact writes a provenance sidecar beside it recording
the library versions that produced it.
