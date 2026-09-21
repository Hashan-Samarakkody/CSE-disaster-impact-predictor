# Refactor validation

What the repository refactors changed, what was actually executed afterwards, and what was
verified. Nothing here is claimed unless it was run.

Two passes are recorded. Part A is the 2026-09-21 validation pass, which repaired the
repository after the target definitions were re frozen on 2026-09-19 and re ran the whole
workflow under them. Part B is the original 2026-09-17 restructuring pass, kept unchanged
as the record of what moved where. Part B's numbers describe the pre protocol targets and
are historical.

---

# Part A. The 2026-09-21 validation pass

## A.1 Why it was needed

On 2026-09-19 the three targets were re specified and frozen in
`docs/TARGET_DEFINITION_PROTOCOL.md`. The notebooks and `src/targets/event_targets.py`
were updated with them, but the rest of the repository was not, and the test suite was
left failing: **25 of 153 tests failed** at the start of this pass.

## A.2 Defects found and fixed

1. **A second, contradictory definition of target one.**
   `src/targets/return_horizons.py` built its own horizon columns as
   `100 ln(P[pos + h] / P[pos - 1])`, one session later than the protocol's
   `Ph = market[position + h - 1]`. The return grid modelled those columns while every
   other stage modelled the dataset's own, so the two disagreed on all 74 events, by up to
   3.38 percentage points at the five session horizon. The builder is deleted; the module
   now only names the columns, and the grid reads them from `dataset.parquet`. One
   definition of Y1 now exists in the repository.
2. **The same off by one in the market only baseline.** `stage_a_expected_returns` in
   `scripts/run_aspi_return_grid.py` estimated a forward return over `price[p + h]`, so
   the study's primary baseline predicted a different quantity from the target it was
   compared against. Corrected, along with the purge bound that depends on it, and the
   stale cached estimates were deleted and rebuilt.
3. **The Y3 hurdle model was fed the no drawdown events.** Under the protocol those events
   carry duration 0 with no recovery observed, so the hurdle read them as censored and
   predicted the 90 session cap against a true zero. Its mean absolute error was 56.0
   sessions against 12.1 for predicting zero. It is now fitted and scored on drawdown
   events only, as the protocol's stage two requires, giving 27.5 sessions. It still loses
   to every baseline, and that result is kept.
4. **`label_adverse_move_sigma` was dead code** that referenced a module deleted in the
   2026-09-17 refactor and a target definition that no longer exists. Removed.
5. **`LABEL_DESCRIPTIONS` had no entry for `C0_drawdown_occurs`**, so the demo app would
   have raised a `KeyError` as soon as the final classifier bundle was refitted with the
   six current labels. Added.
6. **The demo app hard coded a result.** Its classification caption asserted that
   `C2_volume_spike` is the one label that clears both the majority rule and chance. Under
   the current run no label does. The caption is now derived from the table it sits under.
7. **Figure axis labels described the old Y2.** `src/visualization/result_figures.py`
   labelled the target as a ratio minus one. Corrected to the log ratio, and the figures
   regenerated.
8. **Stale tests, rewritten rather than deleted.** The Y3 session counting tests encoded
   the pre protocol convention; the classification label test passed `dataset=None` to a
   label that now needs it; two tests pinned findings rather than behaviour, asserting that
   the recovery concordance interval must exclude 0.5 and that specific classification
   labels must beat their baseline. Those two now assert internal consistency between each
   artifact's numbers and its own verdict column, which catches drift without freezing a
   conclusion in place.
9. **70 decorative comment separators, four oversized docstrings, two oversized comment
   blocks and several truncated comments**, the latter left mid sentence by an earlier
   shortening pass.
10. **Notebook 01 shipped pre refactor stored outputs** naming
    `src/data_pipeline/cse_raw_loaders.py`, `data/2000 data.xls` and absolute machine
    paths, none of which exist. Its outputs are cleared and the reason is documented in the
    notebook and in `docs/notebooks/01_data_acquisition.md`. It is deliberately not re run:
    its six sources are live and unpinned.
11. **One broken documentation link**, to the deleted `architecture/data_acquisition.md`.
12. **`docs/target_definitions.md` contradicted the frozen protocol outright**, describing
    the superseded endpoints, the ratio minus one volume target and the trough anchored
    recovery scan. Rewritten against the implementation, with the observed distributions
    recomputed and the worked example re derived from the raw market series.

## A.3 What was executed

| What | Status |
|---|---|
| `pytest tests/ -q` | **152 passed, 0 failed** |
| `pyflakes` over `src`, `scripts`, `tests`, `apps`, `notebooks` | clean |
| Module self checks | 12 of 12 pass |
| Notebook 02, features and targets | not re run; unchanged by this pass, and its output is independently re verified below |
| Notebook 03, exploratory analysis | executed end to end |
| Notebook 04, regression modelling | not re run; unchanged by this pass, its artifacts date from the 2026-09-19 protocol run |
| Notebook 05, classification and hurdle | executed end to end, twice, the second time after the hurdle fix |
| Notebook 06, evaluation | executed end to end |
| Notebook 07, explainability | executed end to end |
| Notebook 08, sector panel | executed end to end, twice, the second time after a column rename |
| Notebook 09, synthesis | contains no code cells |
| Notebook 01, data acquisition | deliberately not executed, see defect 10 |
| Uncaught errors in any notebook output | **0 across all nine** |
| `scripts/run_aspi_return_grid.py` | executed, 87 minutes, 252 configurations and 720 comparisons rewritten |
| `scripts/run_recovery_survival_grid.py` | executed, all four recovery artifacts rewritten |
| `scripts/run_survival_model.py` | executed |
| `scripts/run_garch_ablation.py` | executed |
| `scripts/run_headline_confirmation.py` | executed |
| `scripts/train_final_models.py` | executed, both bundles rewritten |
| `scripts/build_final_tables.py` | executed, 240 and 15 row tables |
| `scripts/audit_results.py` | executed |
| `scripts/generate_feature_dictionary.py` | executed |
| `scripts/make_architecture_diagram.py` | executed |
| `scripts/freeze_baseline.py` | executed once, deliberately, see A.5 |

## A.4 Independent verification of the target columns

`tests/test_return_horizons.py` now recomputes every horizon value for every event
directly from `market.parquet`, using the protocol's own indexing, and compares it against
`dataset.parquet` at a tolerance of 1e-9. It also checks `P0` for every event. All 74
events and all four horizons agree. That is an independent check of the target column, not
a check of the code that wrote it against itself.

## A.5 The frozen baseline was re taken, deliberately

`artifacts/results/frozen_baseline.json` pinned the pre protocol volume target, a ratio
minus one over the event day window. The protocol replaced it with a log ratio over the
five sessions after the prediction origin, so twenty assertions in
`tests/test_volume_target_frozen.py` were failing against a definition that no longer
exists. The baseline was re taken on 2026-09-21 at commit `f076bd9`, and the fact is
recorded here, in `docs/testing.md` and in `docs/experiments.md` rather than done
silently. Re freezing is justified only by a deliberate, pre declared change to a target
definition.

## A.6 Result changes caused by the re specification

These are consequences of the author's own change of definition, not of the repair work.

| Result | Before the protocol | After |
|---|---|---|
| Return magnitude | 0 of 720 comparisons significant | unchanged, 0 of 720 |
| Return direction at ten sessions | AUC 0.752, interval [0.567, 0.896], Holm p 0.032 | AUC 0.817, interval [0.657, 0.940], Holm p below 0.001 |
| Forward abnormal volume | Gaussian process and support vector regression beat both baselines | the ensemble beats both baselines, pooled R squared 0.334, Holm p 0.021 against naive zero |
| Recovery duration | concordance 0.657, interval [0.522, 0.769], reported as suggestive | concordance 0.535, interval [0.371, 0.697], no result |
| Recovery hurdle | reported as losing | still losing, and now measured on the right sample |

The recovery result is the one that reverses. It is documented in
`docs/improvements_to_thesis/improvements_to_thesis.md` entry 2, and no attempt was made
to recover it.

## A.7 Remaining limitations

1. **Notebook 04 was not re executed in this pass.** Nothing this pass changed affects it:
   its inputs, its target columns and the modules it imports are unchanged, and its
   artifacts are from the 2026-09-19 run under the current protocol. Notebooks 05 to 08
   were re run on top of those artifacts and agree with them. A full six hour re execution
   would confirm bit level stability and has not been performed.
2. **Notebook 01 was not executed**, for the reason in defect 10.
3. **The sector panel still zero fills its own feature matrix** rather than using the
   global median values the index level pipeline uses. It is a secondary analysis and the
   gap is disclosed rather than closed, because closing it would change a result for a
   stylistic reason.
4. **The thesis document was not supplied**, so
   `docs/improvements_to_thesis/improvements_to_thesis.md` says which of its entries are
   unverified against the thesis text and asks the author to check them.
5. **A deprecation warning from `nbformat`** about cells missing an id field is still
   emitted. It is harmless with the current version and is not suppressed, because
   silencing a forward compatibility warning hides a real future break.

---

# Part B. The 2026-09-17 restructuring pass

The numbers and target names in this part describe the pre protocol targets and are kept
as the historical record of what moved where.


## 1. Repository execution status

**Executed and verified on 2026-09-18.** Eight of the nine notebooks were run end to end
after the refactor, every script was run, and the test suite passes. The ninth contains no
code. Only notebook 01 was deliberately left un run, for the reason in section 11.

| What | Status |
|---|---|
| Test suite, `pytest tests/ -q` | **153 passed**, run repeatedly, including after every notebook re execution |
| Every module import | **35 modules, 0 failures**, discovered by package walk |
| Module self checks | **13 of 13 pass** |
| Static check, `pyflakes` over `src`, `scripts`, `tests`, `apps` | **clean, no findings** |
| Notebook 02, features and targets | **executed end to end**, artifacts byte identical to the pre refactor versions |
| Notebook 03, exploratory analysis | **executed end to end** |
| Notebook 05, classification | **executed end to end**, regenerated `classification_summary.parquet` |
| Notebook 06, evaluation | **executed end to end**, regenerated `verdict_table.parquet` |
| Notebook 07, explainability | **executed end to end** |
| Notebook 08, sector panel | **executed end to end**, regenerated every sector artifact |
| Notebook 04, regression modelling | **executed end to end**, regenerated every stage 04 artifact |
| Notebook 01, data acquisition | **deliberately not run.** See section 11 |
| Notebook 09, synthesis | contains no code cells |
| Uncaught errors in any notebook output | **0 across all nine** |
| `scripts/freeze_baseline.py` | run once at the freeze, deliberately not re run |
| `scripts/run_aspi_return_grid.py` | executed, about 75 minutes |
| `scripts/run_recovery_survival_grid.py` | executed, reproduced all four recovery artifacts |
| `scripts/build_final_tables.py` | executed, 240 and 15 row tables |
| `scripts/audit_results.py` | executed |
| `scripts/train_final_models.py` | executed, wrote both final model bundles |
| `scripts/run_survival_model.py` | executed |
| `scripts/run_garch_ablation.py` | executed |
| `scripts/run_headline_confirmation.py` | executed |
| `scripts/generate_feature_dictionary.py` | executed, all 68 feature columns now documented |
| `scripts/make_architecture_diagram.py` | executed, redrew the diagram |

**The strongest single piece of evidence.** Notebooks 05 and 06 were re executed by the
refactored code, which rewrote `classification_summary.parquet` and `verdict_table.parquet`.
All twenty frozen volume target assertions still pass at a tolerance of 1e-9. The refactored
pipeline therefore reproduces the frozen target's predictions, metrics, bootstrap intervals,
verdict strings, Holm flags and classification arm exactly, through the modelling and
evaluation stages, not merely at the point of target construction.

## 1a. Defects found by this verification pass, and fixed

1. **`scripts/generate_feature_dictionary.py` read a stale artifact path.** It built
   `artifacts/feature_spec.json` by hand rather than going through the cache helper, so it
   crashed after the artifact cache was reorganised into subdirectories. It now uses
   `artifact_file`, and the run additionally revealed that four of the sixty eight feature
   columns had never been documented. All four missingness flags are now described.
2. **`scripts/run_headline_confirmation.py` asked the cache to route a directory.** The
   bulk path rewrite turned a figures directory into `artifact_file("figures")`, which has
   no extension and raised a bare `KeyError`. The call site now imports the figure directory
   from settings, and `artifact_store._directory_for` raises a message that names the
   problem instead of a bare `KeyError`, so the same mistake anywhere else is legible.
3. **Nine subpackages had no `__init__.py`.** They resolved as namespace packages, so
   imports worked, but package discovery tools found nothing. Each now has one.
4. **Two modules had no runnable self check**, and a third still announced its old filename.
   `src/targets/event_targets.py` now has one covering all three targets, the resilient
   event case, competing event censoring and the short horizon case.
   `src/evaluation/metrics.py` now has one covering the error metrics, the skill score and
   bound clipping. `src/targets/return_horizons.py` announces its own name.
5. **`scripts/train_final_models.py` printed a path it no longer wrote to.** It reported
   `artifacts/final_classifiers.pkl` while actually writing to `artifacts/models/`. It now
   prints the real destination.

6. **Notebook 05 emitted about 135 scoring warnings on standard error, and the first
   version of this report misdiagnosed them.** That version said the cause was a single
   class on the validation side of an inner split, and that fixing it would be a methodology
   change. Both statements were wrong. Instrumenting the real data showed 6 degenerate
   splits out of 60: five with one class on the validation side, where the metric is
   genuinely undefined, and one with a single class on the *training* side, where the fitted
   model returns a single column of probabilities and the failure happens before the metric
   is reached. The search already anticipated both, through `error_score=np.nan`, so every
   degenerate split was already scoring NaN. The warnings were sklearn announcing a handled
   condition, not an unhandled one. `roc_auc_or_nan` in `src/models/classifiers.py` now
   returns that same NaN directly instead of raising. Scores are unchanged by construction,
   and this was verified three ways: on synthetic splits of both degenerate kinds and a
   healthy control, where the scores are identical and the warnings drop from six to zero;
   by re running notebook 05, where the warning count drops from 135 to zero and
   `classification_summary.parquet` comes back exactly identical in all 19 columns and 20
   rows; and by the frozen target assertions, which cover this notebook's own classification
   arm. The only difference anywhere was one floating point unit in the last place in the
   hurdle table's mean absolute error, from summation order under parallel fitting. Two
   tests in `tests/test_classifiers.py` now pin the scorer's behaviour.

## 2. Final structure

```
README.md
LICENSE
.gitignore
apps/streamlit_app.py
config/requirements.txt
data/raw/            27 files
data/external/       2 files
docs/                architecture, audit, results, interpretation, experiments,
                     testing, data_sources, refactor_validation,
                     notebooks/ (9 files), improvements_to_thesis/,
                     figures/, images/, thesis_materials/
notebooks/           01 to 09, plus _shared.py
scripts/             12 scripts
src/config/          settings.py
src/data/            cse_market_data, emdat_disasters, macro_indicators,
                     external_sources, preprocessing
src/features/        feature_engineering, sector_panel
src/targets/         event_targets, return_horizons
src/training/        walk_forward, inner_cv, time_aware_smogn
src/models/          classifiers, hurdle, shallow_mlp, mlp_subprocess_runner,
                     survival_recovery, inference
src/evaluation/      metrics, verification, collinearity, survival_metrics
src/visualization/   result_figures, eda_figures
src/utils/           artifact_store
artifacts/           tables/ models/ results/ figures/ external/
tests/               12 test modules
```

The repository root now holds only `README.md`, `LICENSE` and `.gitignore`, plus
directories.

## 3. Important files moved

| From | To | Why |
|---|---|---|
| `app.py` | `apps/streamlit_app.py` | the root must hold no scripts |
| `requirements.txt` | `config/requirements.txt` | not technically required at the root; the README points pip at the new path |
| `architecture.md` | `docs/architecture.md` | documentation belongs under `docs/` |
| `architecture/images/pipeline.png` | `docs/images/pipeline.png` | one documentation tree, not two |
| `src/data_pipeline/*.py` | `src/data/`, `src/features/` | loaders separated from feature engineering |
| `src/inference.py` | `src/models/inference.py` | it serves a fitted model bundle |
| `src/evaluation/figures.py`, `eda_figures.py` | `src/visualization/` | plotting is not evaluation |
| `src/evaluation/y1_horizons.py` | `src/targets/return_horizons.py` | it constructs targets |
| `src/sampling/time_aware_smogn.py` | `src/training/time_aware_smogn.py` | augmentation happens during training |
| all `data/*.xls`, `*.xlsx`, `*.csv` | `data/raw/` and `data/external/` | raw inputs separated from reference material |
| every cached artifact | `artifacts/tables/`, `models/`, `results/` | routed by type through one helper |

## 4. Files renamed

**Data.** `07Market Indices - Daily.xls` to `cse_market_indices_daily.xls`. The twenty four
yearly workbooks from `2000 data.xls`, `2015 Data .xlsx` and `2023 Data (2).xls` to
`cse_securities_YYYY.xls`, which removed a stray space and a duplicate marker from the
filenames and let the discovery pattern in `src/data/cse_market_data.py` become a single
clean expression. The two EM-DAT exports to `emdat_sri_lanka_disasters.xlsx` and
`emdat_sri_lanka_disasters_superseded_2026_02_09.xlsx`.

**Modules.** `cse_raw_loaders` to `cse_market_data`, `emdat_loader` to `emdat_disasters`,
`macro_loader` to `macro_indicators`, `preprocessor` to `preprocessing`, `feature_eng` to
`feature_engineering`, `y1_horizons` to `return_horizons`, `figures` to `result_figures`.

**Scripts.** `run_y1_improvement.py` to `run_aspi_return_grid.py`,
`run_y3_improvement.py` to `run_recovery_survival_grid.py`, `run_y1_experiments.py` to
`run_aspi_return_experiments.py`, `run_y1_y2_confirmation.py` to
`run_headline_confirmation.py`.

**Artifacts.** `y1_improve_oof` to `aspi_grid_predictions`, `y1_improve_metrics` to
`aspi_grid_metrics`, `y1_improve_verdicts` to `aspi_grid_verdicts`, `y1_improve_stability`
to `aspi_grid_feature_stability`, `y1_improve_direction` to `aspi_direction_metrics`,
`y1_stage_a_expected` to `aspi_expected_return_market_only`, `y3_improve_oof` to
`recovery_grid_predictions`, `y3_improve_metrics` to `recovery_grid_metrics`,
`y3_improve_calibration` to `recovery_probability_calibration`, `y3_improve_categories` to
`recovery_category_metrics`, `final_table_y1.csv` to `final_table_aspi.csv`,
`final_table_y3.csv` to `final_table_recovery.csv`.

**Tests.** `test_feature_eng` to `test_feature_engineering`, `test_y2_frozen` to
`test_volume_target_frozen`, `test_y1_horizons` to `test_return_horizons`, `test_figures` to
`test_result_figures`, `test_models_classification` to `test_classifiers`,
`test_improvement_artifacts` to `test_experiment_artifacts`.

## 5. Files deleted, and why

| Deleted | Reason |
|---|---|
| `.pytest_cache/`, every `__pycache__/` | regenerated tooling state, now ignored |
| `.scratch/model_diagrams.html` | scratch file, referenced by nothing in the repository |
| `docs/EXTERNAL_DATA_PRE_DECLARATION.md`, `FINAL_ANALYSIS_PROTOCOL.md`, `METHODOLOGY_AUDIT.md`, `THESIS_AMENDMENTS.md`, `FINAL_METHOD_COMPARISON.md`, `Y1_Y3_IMPROVEMENT_PREDECLARATION.md`, `Y2_FROZEN_VALIDATION_REPORT.md` | **content merged, not lost**, into `docs/audit.md` as Parts 1 to 7 |
| `docs/Y1_Y3_FINAL_RESULTS.md`, `docs/RESULTS_AUDIT.txt`, `docs/thesis_materials/final_tables.md` | merged into `docs/results.md` |
| `docs/THESIS_UPDATE_GUIDE.md` | merged into `docs/interpretation.md` Part 4 |
| `docs/architecture/data_acquisition.md`, `evaluation.md`, `feature_engineering.md`, `modeling.md` | superseded by the single `docs/architecture.md` |

No experimental script, no result file and no negative result was deleted. Every rejected
variant and failed experiment that the previous documents recorded is still present in
`docs/audit.md` Part 7.

## 6. Modules created

| Module | Extracted from | What it now owns |
|---|---|---|
| `src/config/settings.py` | `notebooks/_shared.py` | all paths, the seed, the three target definitions, their bounds and their purge columns |
| `src/utils/artifact_store.py` | `notebooks/_shared.py` | the artifact cache, now routing by file type into `tables/`, `models/` and `results/` |
| `src/training/inner_cv.py` | `notebooks/_shared.py` and `scripts/run_aspi_return_grid.py` | both purged inner cross validation variants, previously duplicated |
| `src/targets/event_targets.py` | `src/features/feature_engineering.py` | target construction, split out of feature engineering and decomposed into one named function per target |

`notebooks/_shared.py` is now a thin facade over those modules, so each notebook still opens
with a single import line.

## 7. Duplicate logic removed, and paths corrected

1. The purged inner cross validation existed in two places with slightly different fallback
   behaviour. Both now live in `src/training/inner_cv.py`, as `purged_inner_cv` and the
   stricter `purged_inner_splits`.
2. Roughly 105 hard coded artifact paths across scripts and tests were replaced by
   `artifact_file(name)`, which routes by extension. No script now knows which subdirectory
   an artifact lives in.
3. Every import of a moved module was rewritten, in Python files, notebooks and
   documentation, and verified by importing all twenty four modules and running the suite.
4. The yearly workbook discovery pattern and the index filename default were updated to the
   new data filenames.
5. The architecture diagram output path was updated, and the diagram regenerated.
6. Two figure modules imported each other through the old package; those imports were
   repointed and both now import shared helpers explicitly.

## 8. Code quality pass

| Check | Before | After |
|---|---|---|
| Docstrings longer than five lines | 93 | **0** |
| Comment blocks longer than three lines | 45 | **0** |
| Decorative separator comments | 8 | **0** |
| Emoji characters | 60 across 8 files | **0** |
| Unused imports and variables (pyflakes) | 15 | **0** |
| Code cells without a preceding markdown cell | 0 | 0 |
| Notebook headings without hierarchical numbering | 0 | 0 |
| Notebooks with a Final Outputs section | 0 of 9 | **9 of 9** |

The long docstrings and comments were shortened, not discarded. Where they carried research
rationale rather than code explanation, that rationale is in `docs/audit.md`.

Twelve notebook markdown cells that read as development history, with phrasing such as "a
bug found and fixed during this review", were rewritten into the what, why, later use and
justification structure. Two of them also described behaviour the code no longer has: an
earlier calendar day recovery calculation and an earlier return definition. Both now
describe what the code actually does.

## 9. Three target verification

All three research targets are present, are computed by the refactored code, and produce
identical values to the pre refactor implementation.

| Target | Column | Function | Verified |
|---|---|---|---|
| ASPI percentage change | `Y1_ASPI_5D_Forward_LogReturn_Pct` | `calculate_aspi_percentage_change` | bit identical to the pre refactor values |
| Volume crash magnitude | `Y2_abnormal_volume` | `calculate_volume_crash_magnitude` | bit identical, and additionally held by 20 assertions against the frozen baseline |
| Market recovery days | `Y3_recovery_days` | `calculate_market_recovery_days` | bit identical, including the censoring flag, the censor reason and the drawdown flag |

The verification was run directly after extracting target construction into its own module:
all three targets, the ten session variant, the three label settlement dates and the three
censoring columns were recomputed and compared against `dataset.parquet` at a tolerance of
1e-12. Every column matched.

Notebook 02 was then executed end to end and its regenerated `dataset.parquet`,
`feature_spec.json` and `splits.pkl` were compared against the pre refactor versions. All
numeric columns matched, the feature specification was identical, and every fold's index
arrays were identical.

No fourth target exists. The ten session return column is a pre registered sensitivity
analysis of target one, and the five classification labels are binary views of the same
three targets. Both are documented as such in `docs/architecture.md` section 2.

## 10. Generated result verification

| Artifact | Regenerated | Result |
|---|---|---|
| `dataset.parquet`, `feature_spec.json`, `splits.pkl` | yes, notebook 02 | identical to the pre refactor versions |
| exploratory and diagnostic figures | yes, notebook 03 | 29 figures in `docs/figures/`; one figure was renamed by the run to match its target column, and the stale file was removed |
| `recovery_grid_*.parquet`, `recovery_probability_calibration.parquet`, `recovery_category_metrics.parquet` | yes, script | reproduced, concordance 0.657 for the best model, unchanged |
| `final_table_aspi.csv`, `final_table_recovery.csv` | yes, script | 240 and 15 rows, zero return configurations significant, two recovery models with an interval excluding chance |
| `docs/images/pipeline.png` | yes, script | redrawn at the new path |
| `results_regression.pkl`, `results_ablations.pkl`, `selected_features.pkl`, `train_fit_r2.json`, `final_rf_models.pkl` and the ablation tables | yes, notebook 04 | regenerated in full; the frozen target assertions pass against them |

## 11. What was not executed, and why

**Notebook 01 was deliberately not run.** It retrieves six live, unpinned external sources.
Re running it can return revised figures, which would move every downstream number including
the frozen volume target, and that is a data change rather than a refactor validation. Its
code paths were updated for the new data filenames and directory, and those updates were
verified by resolving both file constants and by confirming that the workbook discovery
pattern matches all twenty four yearly files.

**Notebook 04 ran to completion**, taking about six hours of processor time. It
regenerated `results_regression.pkl`, `results_ablations.pkl`, `selected_features.pkl`,
`train_fit_r2.json`, `final_rf_models.pkl` and every ablation table. Notebooks 06 and 07
were then re executed on top of that fresh output, and the full suite was re run.

**All twenty frozen volume target assertions pass against the regenerated stage 04
output.** The refactored code therefore reproduces the frozen target end to end, from raw
cached inputs through feature and target construction, model fitting, classification and the
statistical verdict, at a tolerance of 1e-9. This closes the gap that the first version of
this report recorded as its largest.

**Notebook 09 contains no code cells**, so there is nothing to execute.

## 12. Remaining warnings and unresolved issues

1. **The licence changed from a permissive one to all rights reserved**, at the owner's
   instruction. Anyone who previously relied on the old terms is affected, and hosting
   platforms will no longer display a recognised open source licence.
2. **The sector panel notebook still blanket zero fills** its own feature matrix rather than
   using the global median values, a residual gap carried over from before the refactor. It
   is a secondary analysis and the gap is disclosed rather than fixed here.
3. **A deprecation warning is emitted by `nbformat`** about notebook cells missing an id
   field. It is harmless with the current version and was not suppressed, because silencing
   a forward compatibility warning hides a real future break.
4. **The thesis document was not available** for the comparison in
   `docs/improvements_to_thesis/`, so that document says which of its entries are unverified
   and asks the author to check them.
