# Notebook 03, Exploratory Analysis and Diagnostics

Documents `notebooks/03_eda_diagnostics.ipynb`.

## Purpose

Describe the modelling table before anything is fitted to it: what is missing, what is
extreme, how the targets are distributed, how the features relate to each other and to the
targets, and what the walk forward geometry actually looks like. This stage fits no model
and modifies no cached table.

## Inputs

`dataset.parquet`, `splits.pkl` and `feature_spec.json` from stage 02, plus the sector index
sheet for the coverage figure.

## Main processing stages

Section 3.2 covers completeness, provenance and the event timeline. Section 3.3 covers
outliers. Section 3.4 covers distributions and normality. Section 3.5 covers the
relationships between predictors and targets. Section 3.6 covers class balance and panel
coverage. Sections 3.7 and 3.8 cover the target distributions and the dependence between
the return and recovery targets. Section 3.9 covers feature correlation, variance inflation
and the pre declared redundancy rule. Section 3.10 draws the fold geometry. Section 3.11
examines what the damage features actually measure. Section 3.12 states what was produced.

## Important functions called

| Function | Module | What it does |
|---|---|---|
| `plot_missingness_matrix`, `outlier_table`, `plot_outlier_panel` | `src/visualization/eda_figures.py` | completeness and extreme value diagnostics |
| `plot_feature_distributions`, `plot_target_scatter_matrix`, `plot_class_balance` | `src/visualization/eda_figures.py` | distribution and association panels |
| `plot_target_distributions`, `plot_target_dependence_y1_y3`, `plot_vif`, `plot_walk_forward_folds` | `src/visualization/result_figures.py` | the diagnostic figures the thesis cites |
| `redundant_drop_set`, `top_correlated_pairs`, `compute_vif` | `src/evaluation/collinearity.py` | the collinearity diagnostics and the pre declared drop rule |

## Important outputs

Figures written to `docs/figures/`, which is version controlled so the thesis can cite a
stable path. No cached table is written here.

## Relationship with other notebooks

Reads stage 02. Nothing downstream depends on it, which is deliberate: an exploratory stage
that later stages depended on would be a route for exploratory decisions to reach the
models.

## Methodological decisions made here

1. **The redundancy rule is reported here but applied in stage 04.** This notebook shows
   which columns the rule would drop. The drop itself happens inside every fold during
   fitting, on training rows only, because a drop decided on the full sample would be a
   selection made with test rows visible.
2. **The threshold was fixed before any model was scored.** An absolute correlation of 0.95
   was pre declared. Choosing it after seeing results would be selection on the test set.
3. **Outliers are reported, not removed.** Removing events because they increase error
   would be sample manipulation. The panel exists so a reader can see which events are
   extreme and judge the results accordingly.
4. **The recovery target is shown to depend on the return target.** They are not
   independent outcomes, and section 3.8 makes that explicit rather than leaving a reader
   to assume three unrelated questions.

## Justification

Reporting distributional diagnostics before modelling is what lets a reader judge whether a
metric is meaningful. Two cases here matter in particular. The recovery target has a median
of four sessions and a mean of about seventeen, so a mean based error metric is dominated by
a handful of slow recoveries, which is part of why the primary recovery metric is rank based.
The class balance panel shows that one label has a prevalence near ninety per cent, which is
why accuracy is never reported without its prevalence and why balanced accuracy is used
instead.

## Execution requirements

Stage 02 must have run. This stage is fast and fits nothing, so it can be re run freely
whenever a figure needs redrawing.
