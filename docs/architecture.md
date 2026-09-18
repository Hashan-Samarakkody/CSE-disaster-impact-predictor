# Architecture

How this project is put together and why each piece exists. Read this before the code.

It assumes you know basic Python and some introductory machine learning. Where a
specialised term is unavoidable it is explained on first use.

![Pipeline](images/pipeline.png)

Regenerate the diagram with `python scripts/make_architecture_diagram.py`.

## 1. Research objective

The study asks what happens to the Colombo Stock Exchange after a natural disaster in Sri
Lanka, and whether any part of that response can be predicted out of sample from
information available before the market reacts.

It is an ex post impact attribution study, not an early warning system. Several features
are damage assessments that EM-DAT finalises weeks or months after an event, so the model
cannot be run the day before a disaster. It answers the question: given a disaster of known
severity, what was the market response.

## 2. The three prediction targets

The repository has exactly three research targets. They are defined in
`src/targets/event_targets.py` and every one of them is measured per event.

1. **ASPI percentage change**, column `Y1_ASPI_5D_Forward_LogReturn_Pct`, computed by
   `calculate_aspi_percentage_change`. The forward log return in percent from the last pre
   event close over five trading sessions: 100 ln(P at t plus 5 divided by P at t minus 1),
   where t is the first trading session on or after the disaster date.
2. **Volume crash magnitude**, column `Y2_abnormal_volume`, computed by
   `calculate_volume_crash_magnitude`. Event day traded volume relative to its own trailing
   thirty session mean, minus one.
3. **Market recovery days**, column `Y3_recovery_days`, computed by
   `calculate_market_recovery_days`. The number of trading sessions until the index regains
   its pre event level, right censored at ninety sessions or at the next qualifying
   disaster, whichever comes first.

Two things that look like extra targets are not. The column
`Y1_EventWindow_0_10_LogReturn_Pct` is the same formula for target one over ten sessions,
carried as a pre registered sensitivity analysis. The five classification labels in
`src/models/classifiers.py` are binary views of the same three targets, for example
"was the return negative" and "did recovery take longer than the median". Neither adds a
fourth research question.

Column names are frozen. They appear inside cached artifacts and inside the regression test
that holds the volume target fixed, so renaming them would invalidate the research record.
`src/config/settings.py` carries readable aliases, `ASPI_PERCENTAGE_CHANGE`,
`VOLUME_CRASH_MAGNITUDE` and `MARKET_RECOVERY_DAYS`, which is what new code should import.

## 3. Major data sources

| Source | What it gives | Where it lands |
|---|---|---|
| Colombo Stock Exchange yearly workbooks, `data/raw/cse_securities_YYYY.xls` | per security traded volume, summed to a market wide daily series | `market.parquet` |
| CSE market indices workbook, `data/raw/cse_market_indices_daily.xls` | the daily ASPI close and every sector index | `market.parquet`, `sector_panel.parquet` |
| EM-DAT export, `data/raw/emdat_sri_lanka_disasters.xlsx` | disaster records, dates, affected population, deaths, damage | `disasters.parquet` |
| World Bank via `wbgapi` | annual GDP growth and inflation, lagged for publication delay | `macro.parquet` |
| S&P 500 via `yfinance` | global market control | `sp500.parquet` |
| NASA POWER | district level precipitation and wind, the measured hazard intensity | `artifacts/external/` |
| DesInventar | independent local loss records | `artifacts/external/` |
| FRED | the Sri Lankan rupee exchange rate | `artifacts/external/` |
| Wikidata | the national election calendar, used as a confounder control | `artifacts/external/` |

The raw inputs and their terms of use are documented in `docs/data_sources.md`.

## 4. Overall data flow

Raw workbooks and the live sources become four cached tables in stage 01. Those become one
event level table of features and the three targets, plus a fixed list of chronological
folds, in stage 02. Stage 03 describes that table without modifying it. Stages 04 and 05
fit models and cache every out of fold prediction. Stage 06 scores those cached predictions
statistically. Stage 07 explains the fitted models. Stage 08 repeats the analysis at sector
level. Stage 09 is the written synthesis.

No stage passes data to the next in memory. Each one reads what it needs from `artifacts/`
and writes what it produces back there. That is deliberate. Stage 01 parses two dozen Excel
workbooks and hits four live web sources, and stages 03, 06 and 09 need none of that, so
caching turns a figure tweak from a full re run into seconds of work. It also closes a
reproducibility gap: the live sources are unpinned and can return revised numbers months
later, so every cached file is written with a `.provenance.json` sidecar recording the
retrieval time and the version of every library involved.

## 5. Notebook execution order

Run the notebooks in numeric order. Later stages fail with a clear message naming the
missing artifact if an earlier stage has not run.

| Stage | Notebook | Reads | Writes | Cost |
|---|---|---|---|---|
| 01 | `01_data_acquisition.ipynb` | `data/raw/`, live sources | `market`, `disasters`, `macro`, `sp500` | slow |
| 02 | `02_features_targets.ipynb` | stage 01 output | `dataset`, `market_feats`, `in_scope`, `feature_spec`, `splits` | fast |
| 03 | `03_eda_diagnostics.ipynb` | `dataset` | exploratory and diagnostic figures | fast, fits nothing |
| 04 | `04_modeling_regression.ipynb` | `dataset`, `splits` | `results_regression`, `results_ablations`, `selected_features`, ablation tables | slowest |
| 05 | `05_modeling_classification.ipynb` | `dataset`, `splits` | `results_classification`, `classification_summary`, `hurdle_table` | medium |
| 06 | `06_evaluation.ipynb` | stages 04 and 05 | `verdict_table`, `conformal_coverage`, `exact_date_sensitivity`, figures | fast, refits nothing |
| 07 | `07_explainability.ipynb` | `final_rf_models` | SHAP figures | medium |
| 08 | `08_sector_panel.ipynb` | `dataset`, sector sheet | `results_sector`, sector tables, figure | slow |
| 09 | `09_synthesis.ipynb` | every table above | written record only | instant |

Stages 06 and 07 fit nothing. They read cached out of fold predictions, which makes it
possible to redraw a figure or add a test without re running an hour of model fitting, and
impossible to accidentally tune a model while replotting it.

Each notebook is documented individually in `docs/notebooks/`.

## 6. Standalone scripts

Four scripts run after the notebooks and read only their cached outputs. They are additive:
none of them writes an artifact the notebooks read, which is what makes the frozen volume
target hold by construction rather than by care.

| Script | Purpose | Cost |
|---|---|---|
| `scripts/freeze_baseline.py` | snapshot every target value, fold, prediction and metric into `frozen_baseline.json` | instant |
| `scripts/run_aspi_return_grid.py` | the 240 configuration return prediction grid | about 75 minutes |
| `scripts/run_recovery_survival_grid.py` | the censoring aware recovery survival grid | about 1 minute |
| `scripts/build_final_tables.py` | assemble the final result tables from the two grids | instant |

Three further scripts support the pipeline: `scripts/audit_results.py` regenerates the
metric audit in `docs/results.md`, `scripts/train_final_models.py` refits the models the
demo app uses, and `scripts/make_architecture_diagram.py` redraws the diagram above.
`scripts/run_survival_model.py`, `scripts/run_garch_ablation.py`,
`scripts/run_aspi_return_experiments.py` and `scripts/run_headline_confirmation.py` are the recorded
experiment runs behind specific sections of `docs/audit.md`.

These are described in `docs/experiments.md`.

## 7. Directory structure

```
README.md                  what the project is and how to run it
LICENSE                    all rights reserved, permission required
config/requirements.txt    pinned dependency list
data/raw/                  CSE workbooks and the EM-DAT export
data/external/             market capitalisation and the ADB reference workbook
notebooks/                 the nine pipeline stages, 01 to 09
notebooks/_shared.py       one import line for every notebook
src/config/                paths, the seed, and the three target definitions
src/data/                  loaders for every raw and live source
src/features/              market and disaster feature engineering, the sector panel
src/targets/               the three research targets and the return horizon variants
src/training/              walk forward splits, purged inner cross validation, oversampling
src/models/                regression, classification, hurdle, survival, the demo bundle
src/evaluation/            metrics, collinearity, survival metrics, the statistical verdict
src/visualization/         the figure suites
src/utils/                 the artifact cache
apps/streamlit_app.py      the demo web app
scripts/                   pipeline runners and experiment runners
artifacts/                 generated cache, see below
tests/                     the test suite
docs/                      this file and the rest of the documentation
```

## 8. Where generated output lands

```
artifacts/tables/    every cached DataFrame, as parquet or csv, with provenance sidecars
artifacts/models/    pickled fitted models, fold definitions and results dictionaries
artifacts/results/   plain JSON payloads, including the frozen baseline
artifacts/figures/   figures produced by the experiment scripts
artifacts/external/  cached downloads from the live external sources
docs/figures/        the figures the thesis cites, tracked in version control
docs/thesis_materials/  the final result tables, as csv
```

`artifacts/` is regenerated by the pipeline and is not version controlled, with one
exception: `artifacts/results/frozen_baseline.json` is a historical snapshot of the volume
target that cannot be rebuilt from a later state, so it is tracked deliberately.

Code never hard codes these paths. `src/utils/artifact_store.py` routes every read and
write by file type, so `artifact_file("dataset.parquet")` resolves to the tables directory
without the caller needing to know.

## 9. How the notebooks and src relate

Notebooks are the narrative. They load configuration, call reusable functions, show
intermediate checks, fit models, evaluate, and save research outputs. Reusable logic lives
in `src/` and is covered by tests that need neither a data file nor a notebook run.

Every notebook opens with `from _shared import *`. That module is a thin facade over
`src/config/settings.py`, `src/utils/artifact_store.py` and the training helpers, so a
notebook needs one import line rather than fifteen and every stage agrees on where files
live, what the seed is, and how the three targets are defined.

## 10. The validation design, in one place

Because it is the part most easily got wrong, the chronological discipline is worth stating
plainly:

1. Events are ordered by date and split by a rolling walk forward, thirty training events,
   ten test events, step ten. No shuffled cross validation anywhere.
2. Every target purges its own label horizon against the fold boundary. A training event
   whose label depends on prices dated on or after the first test event is dropped for that
   fold.
3. Hyperparameter search runs a purged inner split one level deeper, on the training rows
   only. Where purging makes that impossible the number of splits is reduced, and then
   pre specified defaults are used. The purge is never dropped.
4. Feature selection is refit inside every inner split, so an inner validation row's own
   label cannot influence which columns reach the model scored on it.
5. Missing values are imputed from training rows only, one fold at a time.
6. Synthetic oversampled rows appear only in training folds, never in validation or test.
7. The statistical verdict is a paired bootstrap that resamples disaster episodes rather
   than individual events, because events a fortnight apart are not independent draws.

The reasoning behind each of these, and the audit that produced them, is in `docs/audit.md`.
