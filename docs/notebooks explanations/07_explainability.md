# Notebook 07, Explainability

Documents `notebooks/07_explainability.ipynb`.

## Purpose

Show which variables the fitted models actually rely on, globally across the sample and
locally for one real event. This is an interpretation stage, not an evidence stage.

## Inputs

`final_rf_models.pkl` and `selected_features.pkl` from stage 04, and `dataset.parquet` from
stage 02.

## Main processing stages

Section 7.2 computes global feature attributions per target. Section 7.2.1 renders the
summary plot. Section 7.3 explains a single real event, chosen as the largest observed index
drop in the sample. Section 7.4 states what was produced.

## Important functions called

The `shap` library's tree explainer, applied to the random forest models saved by stage 04,
with the figure styling helpers from `src/visualization/result_figures.py`.

## Important outputs

A global attribution figure per target and one local explanation figure, written to
`docs/figures/`. No cached table and no model is written.

## Relationship with other notebooks

Reads stage 04. Nothing downstream depends on it. It supports the thesis explainability
chapter.

## Methodological decisions made here

1. **The models explained here are in sample fits.** `final_rf_models.pkl` is refitted on
   all real rows, because a global attribution needs one model rather than four fold
   specific ones. That makes these figures a description of what a model trained on
   everything relies on, not evidence about out of sample behaviour. They are never
   presented as predictive evidence.
2. **The local example is chosen by an objective rule.** The event shown is the largest
   observed index drop, chosen by that rule rather than by which explanation looked most
   convincing.
3. **Attribution size is not evidence of predictability.** A model with no out of sample
   skill still produces an attribution ranking. The ranking here should be read alongside
   the feature stability table in `docs/results.md`, which shows that 35.6 per cent of
   features were selected in exactly one of four folds.

## Justification

Shapley additive explanations give a locally accurate, consistent decomposition of a single
prediction, which is why they are used here rather than raw impurity importances, which are
biased toward high cardinality features. The in sample caveat is stated explicitly because
an attribution figure looks equally convincing whether or not the model generalises, and in
this study two of the three targets do not.

## Execution requirements

Stage 04 must have run. The `shap` library must be installed, and the stage takes a few
minutes because tree attribution over the full feature matrix is not free.
