# Refactor validation

What the 2026-09-17 repository refactor changed, what was actually executed afterwards, and
what was verified. Nothing below is claimed unless it was run.

## 1. Repository execution status

**Partially executed, and the gap is stated rather than papered over.**

| What | Status |
|---|---|
| Test suite, `pytest tests/ -q` | **151 passed**, run repeatedly during and after the refactor |
| Notebook 02, features and targets | **executed end to end**, in place, via `nbconvert --execute` |
| Notebook 03, exploratory analysis | **executed end to end**, in place, twice |
| Notebook 04, regression modelling | **started and not completed.** It ran for about two and a half hours of CPU time without finishing and was stopped. See section 8. |
| Notebooks 01, 05, 06, 07, 08, 09 | **not executed.** See section 8 for why, per notebook |
| `scripts/build_final_tables.py` | executed, regenerated both final tables |
| `scripts/run_recovery_survival_grid.py` | executed, regenerated all four recovery artifacts |
| `scripts/make_architecture_diagram.py` | executed, redrew the pipeline diagram at its new path |
| `scripts/run_aspi_return_grid.py` | executed earlier in the session, before the refactor, under its previous name |
| Static checks: `pyflakes` over `src`, `scripts`, `tests`, `apps` | **clean, no findings** |

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
| `results_regression.pkl` and the stage 04 ablation tables | no | see section 11 |

## 11. What was not executed, and why

**Notebook 01 was deliberately not re run.** It retrieves six live, unpinned external
sources. Re running it can return revised figures, which would move every downstream number
including the frozen volume target, and that is a data change rather than a refactor
validation. Its code paths were updated for the new data filenames and directory, and those
updates were verified by resolving both file constants and by confirming that the workbook
discovery pattern matches all twenty four yearly files.

**Notebook 04 was started and did not finish.** It ran for roughly two and a half hours of
CPU time inside this session and was stopped before completing. `nbconvert` writes a
notebook only after the last cell succeeds, so nothing was partially written: the cached
stage 04 artifacts are exactly the pre refactor ones, and the notebook file on disk is
unchanged apart from the markdown edits described in section 8. The consequence is that
`results_regression.pkl` and the stage 04 ablation tables were **not** regenerated by the
refactored code, so this report does not claim they were.

**Notebooks 05 to 08 were not executed** because each depends on stage 04's output, and
re running them against artifacts produced by the pre refactor code would test nothing
meaningful about the refactor. **Notebook 09 contains no code cells.**

What is known about stage 04 despite this: the modules it imports all import cleanly, every
path it uses resolves, its inputs were regenerated identically by stage 02, and the twenty
frozen target assertions that read its cached output still pass. What is not known is
whether a complete re execution reproduces its numbers exactly. Confirming that requires a
run of several hours and is the first thing to do with more time.

## 12. Remaining warnings and unresolved issues

1. **Stage 04 has not been re executed end to end.** Section 11. This is the largest gap in
   this validation.
2. **The licence changed from a permissive one to all rights reserved**, at the owner's
   instruction. Anyone who previously relied on the old terms is affected, and hosting
   platforms will no longer display a recognised open source licence.
3. **The sector panel notebook still blanket zero fills** its own feature matrix rather than
   using the global median values, a residual gap carried over from before the refactor. It
   is a secondary analysis and the gap is disclosed rather than fixed here.
4. **A deprecation warning is emitted by `nbformat`** about notebook cells missing an id
   field. It is harmless with the current version and was not suppressed, because silencing
   a forward compatibility warning hides a real future break.
5. **The thesis document was not available** for the comparison in
   `docs/improvements_to_thesis/`, so that document says which of its entries are unverified
   and asks the author to check them.
