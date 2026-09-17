# Notebook 04, Regression Modelling and Ablations

Documents `notebooks/04_modeling_regression.ipynb`.

## Purpose

Fit every regression model on the chronological walk forward and cache every out of fold
prediction, so that the evaluation stage can score them without refitting anything. This is
the most expensive stage in the pipeline and the one that produces the numbers the thesis
quotes.

## Inputs

`dataset.parquet`, `splits.pkl` and `feature_spec.json` from stage 02.

## Main processing stages

Section 4.2 illustrates the oversampling on one fold. Section 4.3 runs the walk forward
loop, with the oversampling ablation, the neural comparison, the ensemble blend and the
stacked meta learner as subsections. Section 4.4 adds the naive baselines the models must
beat. Sections 4.5 to 4.9 run the pre declared ablations: a denser fold configuration, a
reduced feature count, the final in sample refit for the explainability stage, the external
data block ablation, and the collinearity and principal component comparisons. Section 4.10
caches everything. Section 4.11 states what was produced.

## Important functions called

| Function | Module | What it does |
|---|---|---|
| `purge_horizon_overlap` | `src/training/walk_forward.py` | drops training rows whose label reaches into the test period |
| `median_impute_from_train` | `src/training/walk_forward.py` | imputes the sixteen genuinely missing columns from training rows only |
| `purged_inner_cv` | `src/training/inner_cv.py` | the purged inner splits used for hyperparameter search |
| `CollinearityDropper`, `CollinearityRFTopK` | `src/evaluation/collinearity.py` | feature selection as pipeline steps, so it refits inside every inner split |
| `time_aware_smogn` | `src/training/time_aware_smogn.py` | training fold oversampling, capped and time windowed |
| `evaluate_regression`, `clip_to_bounds` | `src/evaluation/metrics.py` | scoring and projection onto each target's support |

## Important outputs

`results_regression.pkl` is the important one: every model's out of fold predictions and
per fold metrics, for all four regression columns. Also `results_ablations.pkl`,
`selected_features.pkl`, `train_fit_r2.json`, the ablation tables
`smogn_ablation.parquet`, `ablation_blocks.parquet`, `ablation_block_contrib.parquet`,
`pca_ablation_results.pkl` and `corr_pairs_above_075.parquet`, and `final_rf_models.pkl`.

## Relationship with other notebooks

Reads stage 02. Feeds stage 06 for the statistical verdict and stage 07 for the SHAP
explanations.

## Methodological decisions made here

1. **Feature selection is nested inside the inner cross validation.** The selector is a
   pipeline step inside the search, so it is refit on every inner split and every
   hyperparameter candidate. Without that, an inner validation row's own label could
   influence which columns even reached the model being scored on it.
2. **Each target gets its own selection.** No target is forced to reuse another's feature
   ranking.
3. **The linear model is treated differently, on purpose.** A random forest importance
   ranking is not the right selection criterion for a linear model, so the ridge model fits
   on the full collinearity pruned set and lets its own regularisation shrink, rather than
   taking a top k cut chosen by a tree.
4. **Oversampling is measured, not assumed.** Section 4.3.1 runs the same walk forward with
   the oversampling switched off, so its effect is reported rather than taken on faith. A
   wider variant tried earlier made every model worse and is recorded as rejected in
   `docs/audit.md` rather than deleted.
5. **Each target defines its own minority mask.** The oversampler targets the tail that
   matters for that target: the most severe negative returns for target one, the largest
   volume spikes for target two, the slowest recoveries for target three.
6. **The final refit is in sample and labelled as such.** `final_rf_models.pkl` is fitted on
   all real rows for the explainability stage. It is never scored as a prediction.
7. **Baselines are part of the design.** A constant zero prediction and the training fold
   mean are run through the identical loop, so they are scored on exactly the same events.

## Justification

Nesting selection inside the search follows the standard treatment of selection bias in
small sample model evaluation: any step that reads the target is part of the model and must
sit inside the resampling. The purge follows the fold boundary embargo described in the
financial machine learning literature, which exists precisely for targets measured over a
forward window. Comparing against a constant zero baseline rather than only against the
training mean matters here because a disaster event study's null hypothesis is that there is
no measurable effect, which is a prediction of zero, not a prediction of the average.

## Execution requirements

Stage 02 must have run. This stage fits hundreds of models and is by far the slowest, well
over an hour on an ordinary laptop. The neural comparison is trained in a subprocess on
Windows to avoid a known runtime conflict between torch and the Intel OpenMP library.
