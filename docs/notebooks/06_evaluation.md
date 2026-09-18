# Notebook 06, Evaluation

Documents `notebooks/06_evaluation.ipynb`.

## Purpose

Decide whether anything beats its baseline. This stage reads the cached out of fold
predictions from stages 04 and 05 and refits nothing, which is what makes it impossible to
tune a model while claiming to replot it.

## Inputs

`results_regression.pkl`, `results_classification.pkl`, `dataset.parquet`, `splits.pkl` and
`train_fit_r2.json`.

## Main processing stages

Section 6.2 produces the predictability verdict, per target and with the exact date
sensitivity analysis. Section 6.3 builds the results table and discusses selection
procedures and optimistic bias. Section 6.4 draws the figures that show whether the gap is
real. Section 6.5 names a recommended model per target and shows the overfitting gap.
Section 6.6 computes rolling conformal prediction intervals and their achieved coverage.
Section 6.7 reports directional accuracy as a secondary metric. Section 6.8 states what was
produced.

## Important functions called

| Function | Module | What it does |
|---|---|---|
| `verdict_table` | `src/evaluation/verification.py` | every model against every baseline on every target |
| `paired_bootstrap_delta` | `src/evaluation/verification.py` | the paired bootstrap interval on the error difference |
| `build_episode_ids` | `src/evaluation/verification.py` | groups events less than a fortnight apart into one episode |
| `diebold_mariano` | `src/evaluation/verification.py` | the small sample corrected forecast comparison, reported as a diagnostic |
| `plot_model_vs_baseline`, `plot_skill_forest`, `plot_pred_vs_actual`, `plot_overfitting_gap`, `plot_conformal_coverage`, `plot_roc_with_ci` | `src/visualization/result_figures.py` | the evaluation figures |

## Important outputs

`verdict_table.parquet`, `conformal_coverage.parquet`, `exact_date_sensitivity.parquet`,
and the evaluation figures in `docs/figures/`.

## Relationship with other notebooks

Reads stages 04 and 05. Feeds stage 09 and `docs/results.md`.

## Methodological decisions made here

1. **The paired bootstrap is the only gate.** A model beats its baseline when the paired
   interval on the error difference excludes zero. Nothing else promotes a result.
2. **The bootstrap resamples disaster episodes, not individual events.** Two disasters a
   fortnight apart plausibly share one market shock, so drawing them independently would
   overstate the effective sample size. Events within fourteen days are chained into one
   episode and drawn or withheld together.
3. **The forecast comparison test is reported, not required.** Its assumptions of a roughly
   stationary, weakly dependent error series fit an irregular event panel with overlapping
   horizons worse than the bootstrap's plain exchangeability assumption. Requiring both
   would let the weaker assumption veto the stronger one.
4. **A family wise correction is reported alongside, never folded in.** Many comparisons are
   run and some significant result is expected by chance, so a Holm corrected p value is
   reported as a stricter diagnostic. It does not silently reclassify a single comparison
   result, because which comparisons belong to one family is itself a judgment call.
5. **The stacked model is aligned, not silently zipped.** It forfeits the first fold to its
   meta learner, so it has fewer test points than the others. Those comparisons are aligned
   on the shared tail and flagged.
6. **A sensitivity analysis on date precision.** Five of the seventy four events carry month
   only precision in EM-DAT. The headline result is re scored on the sixty nine exact day
   events using the same cached predictions, as a check on whether it depends on those five.

## Justification

The paired bootstrap makes no assumption beyond exchangeability of events, which matches an
irregular event panel better than the stationarity assumptions a forecast comparison test
requires. Clustering the resample by episode is the standard correction when observations
arrive in correlated groups. Reporting the family wise correction separately follows the
convention that a multiplicity adjustment is a diagnostic about a body of results, not a
property of any single comparison.

## Execution requirements

Stages 04 and 05 must have run. This stage is fast because it refits nothing, so it can be
re run whenever the presentation of a result changes.
