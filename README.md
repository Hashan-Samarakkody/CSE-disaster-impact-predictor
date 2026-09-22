# Predicting the impact of natural disasters on the Colombo Stock Exchange

An undergraduate BSc (Hons) thesis project. For every qualifying natural disaster in Sri
Lanka between 2000 and 2025, it models three things the Colombo Stock Exchange did
afterwards, and tests whether any of them can be predicted out of sample.

Seventy four real events. No synthetic events in any test set, no relaxed inclusion
criteria, no gap filled outcomes.

## Research objective

The study asks whether a disaster's measurable characteristics carry information about the
market response beyond what the market's own condition already implies. It is an ex post
impact attribution study, not an early warning system: several features are damage
assessments that EM-DAT finalises weeks after an event, so the model cannot be run the day
before a disaster. It answers the question, given a disaster of known severity, what was
the market response.

Every feature is classified by availability at the prediction origin, real time or ex post,
and the two information sets are reported separately, so a reader can see exactly which
results depend on information a forecaster could not have had.

The target definitions and evaluation rules were frozen in this repository on 19 September 2026 at commit `928a255`, before any performance under them was observed. The feature set was not pre registered: GARCH volatility, NASA POWER hazard intensity, DesInventar severity and election proximity were all added after the proposal, and the study does not claim otherwise.

## The three prediction targets

Definitions are frozen in `docs/TARGET_DEFINITION_PROTOCOL.md` and implemented in
`src/targets/event_targets.py`. `P0` is the last ASPI close known before the disaster,
`Pk` and `Vk` are the close and the market wide volume of the kth complete trading
session after it.

1. **ASPI return magnitude**, `Y1_ASPI_5D_Forward_LogReturn_Pct`. The forward log return
   in percent over the first five complete sessions, `100 ln(P5 / P0)`.
2. **Forward abnormal trading volume**, `Y2_5D_Forward_AbnormalVolume_LogRatio`. The
   natural logarithm of the ratio of mean volume over those five sessions to the mean of
   the thirty sessions before the event, `ln(mean(V1..V5) / mean(V-30..V-1))`. It is a
   log ratio, not a ratio minus one. The interpretable companion is the percentage form
   `100 * (exp(Y2) - 1)`, stored as `Y2_5D_Forward_AbnormalVolume_Pct`: a log ratio of
   0.4055 reads as turnover 50 percent above normal. The sign is two sided, because a
   disaster can raise or suppress turnover, and 23 of the 61 observed values are positive.
3. **Market recovery duration**, `Y3_ASPI_Recovery_Time`. A time to event outcome. Whether
   a drawdown occurs at all is stage one; conditional on one, the duration is the number
   of sessions until the index regains `P0`, right censored at ninety sessions or at the
   next qualifying disaster.

These three are the whole research question. There is no fourth target. The one, ten,
fifteen and twenty session return columns are declared horizon sensitivity analyses of
target one, the market adjusted abnormal return is target one with the world market's
contribution removed, the alternative volume windows are target two measured over one, ten
and twenty sessions, and the six classification labels are binary views of the same three
targets. None of them is ever promoted to primary.
`docs/target_definitions.md` states each one precisely, with the observed distributions
and a worked example.

## What the study found

All numbers below come from the walk forward evaluation in this repository, re run in full
on 2026-09-21 under the frozen target protocol.

| Target | What is predictable | Best validated model | Evidence | Supported |
|---|---|---|---|---|
| Y1, return magnitude | nothing | none | 0 of 720 paired bootstrap comparisons across a pre declared 240 configuration grid had an interval excluding zero. Best pooled R squared anywhere in the grid is 0.092 | No |
| Y1, return direction at ten sessions | the sign of the return | logistic regression, combined features, ten selected features | ROC AUC 0.817, episode clustered interval [0.657, 0.940], Holm corrected p below 0.001, balanced accuracy 0.757 | Yes |
| Y2, forward abnormal volume | the size of the response | ensemble of the regression families | pooled R squared 0.334, delta RMSE interval excludes zero against both naive baselines, Holm corrected p 0.021 against naive zero | Yes |
| Y3, recovery duration | nothing, in either duration or ranking | none | every model loses to the training mean baseline on RMSE, and the best Harrell concordance is 0.535 with an interval of [0.371, 0.697] | No |

Read that table with its negatives intact. Exact return magnitude and recovery duration are
not predictable at this sample size, and the study reports that rather than working around
it. The full numbers are in `docs/results.md` and what they license is in
`docs/interpretation.md`.

## High level methodology

Events are ordered by date and validated by a rolling chronological walk forward of thirty
training events, ten test events, step ten. There is no shuffled cross validation anywhere.
Every target purges its own label horizon against the fold boundary, hyperparameter search
runs a purged inner split one level deeper, feature selection is refit inside every inner
split, missing values are imputed from training rows only, and synthetic oversampled rows
appear in training folds only.

A model beats its baseline when a paired bootstrap interval on the error difference excludes
zero. The bootstrap resamples disaster episodes rather than individual events, because two
disasters a fortnight apart are not independent draws. A family wise correction is reported
alongside as a stricter diagnostic and is never folded into the primary criterion.

`docs/architecture.md` explains the whole design. `docs/audit.md` is the complete research
record, including the experiments that failed.

## Repository structure

```
README.md                  this file
LICENSE                    all rights reserved, permission required before any use
config/requirements.txt    pinned dependency list
data/raw/                  CSE workbooks and the EM-DAT export
data/external/             reference workbooks used by the thesis text
notebooks/                 the nine pipeline stages, 01 to 09
src/config/                paths, the seed, and the three target definitions
src/data/                  loaders for every raw and live source
src/features/              market and disaster feature engineering, the sector panel
src/targets/               the three research targets, the horizon variants and the
                           market adjusted abnormal return
src/training/              walk forward splits, purged inner cross validation, oversampling
src/models/                regression, classification, hurdle, survival, the demo bundle
src/evaluation/            metrics, collinearity, survival metrics, the forecast verdict,
                           and the event study inference that tests the realised response
src/visualization/         the figure suites
src/utils/                 the artifact cache
apps/streamlit_app.py      the demo web app
scripts/                   pipeline runners and experiment runners
artifacts/                 generated cache, not version controlled
tests/                     the test suite, including 20 that hold the volume target frozen
docs/                      all documentation, listed below
```

## Installation

Python 3.12, although 3.10 and later should work.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS and Linux
pip install -r config/requirements.txt
```

Two dependency files sit in `config/` rather than the repository root, so that the root
holds only this file, the licence and the ignore rules:

| File | What it is | When to use it |
|---|---|---|
| `config/requirements.txt` | the declared direct dependencies, with lower bounds such as `numpy>=1.24` | ordinary installation, forward compatible |
| `config/requirements.lock.txt` | every direct and transitive dependency pinned to the exact version that produced the reported results, 141 packages | reproducing the published numbers |

```bash
pip install -r config/requirements.lock.txt    # the exact environment
pip install -r config/requirements.txt         # the loose, forward-compatible set
```

The results were produced on **Python 3.12.10**, Windows. The lock file is generated from
the installed environment rather than maintained by hand.

## Data requirements

The raw inputs in `data/raw/` are required before anything runs: the CSE daily index
workbook, twenty four yearly per security workbooks, and the EM-DAT export. Stage 01
additionally retrieves six live sources over the network. Every source, and the terms that
come with it, is documented in `docs/data_sources.md`.

## How to run the pipeline

Run the notebooks in numeric order. Each stage reads what it needs from `artifacts/` and
writes what it produces back there, so a later stage fails with a clear message naming the
missing artifact if an earlier one has not run.

Stage 01 is the exception. Its six external sources are live and unpinned, so re running it
can return revised figures and move every downstream number. Its outputs are already cached
under `artifacts/tables/` and `artifacts/external/`, and it ships without stored cell
outputs for that reason. Start at stage 02 unless a data refresh is intended.

```
notebooks/01_data_acquisition.ipynb        slow, network access, live unpinned sources
notebooks/02_features_targets.ipynb        fast
notebooks/03_eda_diagnostics.ipynb         fast, fits nothing
notebooks/04_modeling_regression.ipynb     slowest, over an hour
notebooks/05_modeling_classification.ipynb medium
notebooks/06_evaluation.ipynb              fast, refits nothing
notebooks/07_explainability.ipynb          medium
notebooks/08_sector_panel.ipynb            slow
notebooks/09_synthesis.ipynb               written record only
```

Then the experiment scripts:

```bash
python scripts/run_event_study.py               # did the market react at all
python scripts/run_aspi_return_grid.py          # the return grid, a few hours
python scripts/run_recovery_survival_grid.py    # survival plus competing risks
python scripts/run_robustness_suite.py          # the closed pre-declared checks, once
python scripts/train_final_models.py            # fits the bundle the demo app serves
python scripts/build_final_tables.py
python scripts/audit_results.py
```

And the tests:

```bash
python -m pytest tests/ -q
```

## Expected outputs

Cached tables in `artifacts/tables/`, fitted models and fold definitions in
`artifacts/models/`, JSON payloads in `artifacts/results/`, downloads in
`artifacts/external/`, tracked figures in `docs/figures/`, and the final result tables in
`docs/thesis_materials/`.

## Reproducibility

The seed is fixed at 42 in `src/config/settings.py` and is never varied to obtain a better
score. Every cached artifact is written with a provenance sidecar naming the retrieval time
and the library versions that produced it, because several sources are pulled live and
unpinned. No path anywhere is absolute or machine specific.

One file under `artifacts/` is version controlled deliberately:
`artifacts/results/frozen_baseline.json`, a snapshot of the volume target taken before the
improvement work began. It cannot be rebuilt from a later state, and
`tests/test_volume_target_frozen.py` compares the live pipeline against it.

## Documentation

| Document | What it covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | how the whole system fits together, start here |
| [docs/TARGET_DEFINITION_PROTOCOL.md](docs/TARGET_DEFINITION_PROTOCOL.md) | the frozen specification of the three targets, the binding definition |
| [docs/target_definitions.md](docs/target_definitions.md) | the same three targets written for a thesis reader, with observed distributions and a worked example |
| [docs/notebooks/](docs/notebooks/) | one file per notebook stage, nine in total |
| [docs/audit.md](docs/audit.md) | the complete research record: the frozen protocol, every pre declaration, the full dated change log, and every rejected variant |
| [docs/results.md](docs/results.md) | every number the executed repository produced |
| [docs/interpretation.md](docs/interpretation.md) | what may and may not be claimed, and the thesis revision guide |
| [docs/experiments.md](docs/experiments.md) | what each script in `scripts/` does |
| [docs/testing.md](docs/testing.md) | what the test suite checks and why |
| [docs/data_sources.md](docs/data_sources.md) | every input, its provenance and its terms |
| [docs/refactor_validation.md](docs/refactor_validation.md) | what was actually executed and verified |
| [docs/improvements_to_thesis/](docs/improvements_to_thesis/) | where the written thesis and the implementation disagree |

## Demo application

```bash
streamlit run apps/streamlit_app.py
```

The app replays a real historical event through models refitted on all real rows. Those are
in sample fits for demonstration, not out of sample predictions, and the app says so.

## Limitations

Seventy four events and forty pooled out of fold test points. There is no final lockbox
holdout, because at this sample size setting one aside would cost folds the study cannot
spare, so an adaptive selection risk remains that the bootstrap and the family wise
correction reduce but do not eliminate. Across four folds, 35.7 per cent of selected
features were selected in exactly one fold, so no single fold's selection is treated as a
finding. The recovery target has fourteen observed recoveries in the pooled test set
against twenty six censored ones, which is thin for a survival model with twenty
covariates and is the main reason that target returns no result.

## License

All rights reserved. This repository is published for examination and reference only. It is
not open source. Written permission is required before any use, including academic and
educational use. See [LICENSE](LICENSE) and contact hashansamarakkody@gmail.com.

Third party data carries its own terms, which this licence does not override.

**Open question raised by the Revision 2 review (D5), awaiting the owner's decision.** The
review asks that the code, seeds, versions and workflow be provided for verification,
which the current all rights reserved terms do not permit. The recommended resolution is
to license the CODE under an open licence, for example MIT or BSD 3-Clause, while keeping
separate and explicit terms for the third party data, which cannot be redistributed under
any of them. This has deliberately NOT been changed: a licence is a legal instrument and
only the repository owner can alter it.

The models here are a student research exercise. They are not investment advice.
