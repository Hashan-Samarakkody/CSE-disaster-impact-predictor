# Improvements to the thesis

What the written thesis needs to change so that it describes the implementation that now
exists in this repository.

## Scope and an honest limitation of this document

The thesis document itself was not supplied for this comparison. Nothing below is invented
from a guess about what the thesis says. Every entry is anchored in something the
repository's own prior documentation asserted, which the thesis is likely to mirror because
those documents were written alongside it. Where a statement could not be verified against
the thesis text, the entry says so and asks the author to check.

Three further entries are not mismatches at all but results that postdate any thesis draft
written before 2026-09-17, and they are marked as new material rather than corrections.

---

## 1. The recovery target headline

**Thesis section.** Results, recovery duration.

**Current thesis statement.** The recovery duration model achieves a pooled out of sample R
squared of about +0.155, with an ensemble as the best model.

**Actual repository implementation.** That number exists and is still reported, but it is no
longer the primary result. It was computed by point regression that treats every censored
observation as if it were an exact recovery time. Seventeen of the seventy four events are
censored, eight at the ninety day cap and nine by a competing later disaster. The primary
analysis is now a right censored survival model fitted to genuine events only, reported by
Harrell concordance rather than by R squared.

**Mismatch.** The headline number was produced under a likelihood that does not match the
data shape, and it flatters the result.

**Required thesis improvement.** Demote the R squared to a disclosed diagnostic of that
distortion. Promote the survival result: concordance 0.657, episode clustered interval
[0.522, 0.769], against 0.477 for a Kaplan Meier baseline, failing the family wise
correction at p 0.165 and therefore reported as suggestive rather than established. Add the
recovery probability output, which is the practically useful part.

**Repository evidence.** `scripts/run_recovery_survival_grid.py`,
`artifacts/tables/recovery_grid_metrics.parquet`,
`artifacts/tables/recovery_probability_calibration.parquet`,
`docs/thesis_materials/final_table_recovery.csv`, `src/evaluation/survival_metrics.py`.

**Result impact.** Methodology, results, discussion and conclusions.

---

## 2. Return magnitude predictability

**Thesis section.** Results, index return.

**Current thesis statement.** Likely to report a best pooled R squared near +0.14 for the
five session return and to treat it as a weak positive.

**Actual repository implementation.** A pre declared grid of 240 configurations, four
horizons by four information sets by three feature capacities by five model families, was
run against three baselines. None of the 720 comparisons produced an interval excluding
zero. The best pooled R squared anywhere in the grid is +0.075, with a delta error interval
of [-0.560, +1.045].

**Mismatch.** A weak positive presented as a partial success is now a systematic negative
result, established across a far larger and pre registered search.

**Required thesis improvement.** Rewrite the section as a negative finding, with the grid
size, the three baselines and the interval quoted. This is a stronger and more defensible
claim than the original, not a weaker one.

**Repository evidence.** `scripts/run_aspi_return_grid.py`,
`artifacts/tables/aspi_grid_verdicts.parquet`,
`docs/thesis_materials/final_table_aspi.csv`, `docs/results.md`.

**Result impact.** Results, discussion, conclusions.

---

## 3. Whether disaster information adds anything

**Thesis section.** Results and discussion, the incremental value of disaster severity data.

**Current thesis statement.** Likely to claim or imply that the external severity and hazard
blocks improved prediction, on the strength of the block ablation.

**Actual repository implementation.** The block ablation still shows a handful of blocks with
intervals excluding zero, and those remain in the record. The direct test, the combined
information set against a market only information set on identical folds and identical test
events, does not support the claim. Combined wins at three of four horizons and loses at the
fourth, with every interval spanning zero.

**Mismatch.** The block ablation answers a narrower question than the one the thesis needs.
It shows that removing a block changes error, not that disaster information beats market
state.

**Required thesis improvement.** Report both. State that the incremental value of disaster
specific information for return magnitude is not demonstrated, and keep the block ablation as
the secondary evidence it is.

**Repository evidence.** `src/targets/return_horizons.py` for the information set partition,
`artifacts/tables/aspi_grid_metrics.parquet`, `artifacts/tables/ablation_blocks.parquet`.

**Result impact.** Results, discussion, conclusions.

---

## 4. The target naming

**Thesis section.** Methodology, target definitions.

**Current thesis statement.** The three targets are named in the thesis in its own words.

**Actual repository implementation.** The implementation carries frozen column names that
begin with Y1, Y2 and Y3, because those names appear inside cached artifacts and inside the
regression test that holds the volume target fixed. Readable aliases exist in
`src/config/settings.py`: `ASPI_PERCENTAGE_CHANGE`, `VOLUME_CRASH_MAGNITUDE` and
`MARKET_RECOVERY_DAYS`.

**Mismatch.** Two vocabularies for the same three quantities. Whether the thesis uses the
repository's names is unverified and the author should check.

**Required thesis improvement.** Add one table to the methodology chapter mapping the thesis
name, the readable alias and the frozen column name for each of the three targets, so a
reader moving between the document and the code is never in doubt.

**Repository evidence.** `src/config/settings.py`, `docs/architecture.md` section 2.

**Result impact.** Methodology only. No number changes.

---

## 5. Apparent extra targets

**Thesis section.** Methodology, scope.

**Current thesis statement.** Three prediction targets.

**Actual repository implementation.** The dataset also carries a ten session variant of the
return target and five binary classification labels. The variant is a pre registered
sensitivity analysis of target one. The labels are binary views of the same three targets.
Neither is a fourth research question.

**Mismatch.** A reader opening the dataset counts more than three target shaped columns and
may reasonably wonder whether the scope grew.

**Required thesis improvement.** Add one sentence to the scope section stating that the
horizon variant and the classification labels are derived views of the same three targets,
and pointing at `docs/architecture.md` section 2.

**Repository evidence.** `src/config/settings.py`, `src/models/classifiers.py`,
`src/targets/event_targets.py`.

**Result impact.** Methodology and presentation. No number changes.

---

## 6. New material: return direction at ten sessions

**Thesis section.** Results, new subsection required.

**Current thesis statement.** No equivalent. The earlier direction analysis covered the five
session horizon only, where it failed.

**Actual repository implementation.** Direction was tested at all four pre declared horizons
as an explicitly separate question from magnitude. At ten sessions, logistic regression on
the combined information set with ten features reaches ROC AUC 0.752, episode clustered
interval [0.567, 0.896], and it is the only result in the study that survives the family
wise correction, at p 0.032.

**Mismatch.** New evidence, not a correction.

**Required thesis improvement.** Add a results subsection, clearly separated from magnitude,
and immediately follow it with the asymmetry: sensitivity 0.476 against specificity 0.842,
so the model is a reliable indicator that the index will not fall and an unreliable detector
of falls. State explicitly that classification success is not regression success.

**Repository evidence.** `artifacts/tables/aspi_direction_metrics.parquet`,
`scripts/run_aspi_return_grid.py`.

**Result impact.** Results, discussion and conclusions. This is the study's strongest new
claim.

---

## 7. New material: feature selection stability

**Thesis section.** Results, feature importance.

**Current thesis statement.** Likely to name individual features as important, on the
strength of importance rankings or single fold selections.

**Actual repository implementation.** Selection stability is now measured. Across four folds,
35.6 per cent of all selected features were selected in exactly one fold.

**Mismatch.** Any claim that a particular feature matters needs that number beside it.

**Required thesis improvement.** Add the stability paragraph and restrict feature level
claims to the variables with a selection frequency of 0.75 or above. The most consistently
selected disaster variable is the DesInventar log affected population, and the most
consistently selected market variable is the one session lagged return.

**Repository evidence.** `artifacts/tables/aspi_grid_feature_stability.parquet`.

**Result impact.** Results and discussion. It constrains what the explainability chapter may
claim.

---

## 8. New material: the failed decomposition

**Thesis section.** Methodology and results.

**Current thesis statement.** No equivalent.

**Actual repository implementation.** A normal market plus disaster residual architecture was
implemented and tested: a market only expected return model estimated from daily sessions
that settled strictly before each event, with a disaster residual modelled on top. It did not
improve prediction at any horizon and was the worst of the four information sets at fifteen
and twenty sessions.

**Mismatch.** New negative result.

**Required thesis improvement.** Report it. A tested and failed hypothesis is a contribution,
and omitting it would leave the pre declaration in `docs/audit.md` describing an experiment
the thesis never mentions.

**Repository evidence.** `scripts/run_aspi_return_grid.py`,
`artifacts/tables/aspi_grid_metrics.parquet` rows with information set
`normal_plus_residual`.

**Result impact.** Methodology, results, discussion.

---

## 9. Repository structure and file paths

**Thesis section.** Any appendix or methodology passage citing a file path.

**Current thesis statement.** Paths from the pre refactor layout, for example
`src/data_pipeline/feature_eng.py` or `app.py`.

**Actual repository implementation.** The layout was reorganised on 2026-09-17. Target
construction now lives in `src/targets/`, feature engineering in `src/features/`, loaders in
`src/data/`, figures in `src/visualization/`, the demo in `apps/streamlit_app.py`, and the
dependency list in `config/requirements.txt`.

**Mismatch.** Any path the thesis cites from the old layout is now broken.

**Required thesis improvement.** Re check every file path the thesis cites against
`docs/architecture.md` section 7.

**Repository evidence.** `docs/refactor_validation.md` lists every move and rename.

**Result impact.** Presentation and reproducibility. No number changes.

---

# Verified final research outputs

Everything below was produced by executing this repository. No value here is estimated or
carried over from an earlier run, and where a result could not be produced the reason is
stated.

## ASPI percentage change

**Target statistics.** Seventy four events, seventy four observed values for the five
session horizon. Standard deviation 3.05 percentage points, mean +0.43.

**Validation.** Chronological walk forward, thirty training events, ten test events, step
ten, four folds, forty pooled out of fold test points.

**Evaluation metrics.** Best configuration in the pre declared grid: pooled R squared
+0.075, ten session horizon, disaster only information set, elastic net, ten features. Best
five session configuration: pooled R squared +0.062, combined information set, random
forest. Neither is distinguishable from its baseline.

**Model comparisons.** 240 configurations, 720 paired comparisons against a zero return
baseline, a training fold mean and a market only expected return model. Comparisons whose
interval excludes zero: zero.

**Direction, reported as a separate question.** Ten session horizon, logistic regression,
combined information set, ten features: ROC AUC 0.752, interval [0.567, 0.896], balanced
accuracy 0.659, Matthews correlation 0.339, precision recall AUC 0.771, sensitivity 0.476,
specificity 0.842, Holm corrected p 0.032.

**Result files.** `artifacts/tables/aspi_grid_metrics.parquet`,
`artifacts/tables/aspi_grid_verdicts.parquet`,
`artifacts/tables/aspi_direction_metrics.parquet`,
`artifacts/tables/aspi_grid_feature_stability.parquet`,
`docs/thesis_materials/final_table_aspi.csv`.

**Figures.** Predicted against actual, the skill forests and the model against baseline
panel, in `docs/figures/`.

## Volume crash magnitude

**Target statistics.** Sixty one observed values of seventy four events, because volume data
is absent for part of the early archive. Mean minus 0.129, standard deviation 0.580, median
minus 0.346, quartiles minus 0.531 and plus 0.162.

**Validation.** The same folds, thirty four pooled out of fold test points.

**Evaluation metrics.** Gaussian process: RMSE 0.455, MAE 0.361, pooled R squared +0.270.
Support vector regression: RMSE 0.458, MAE 0.364, pooled R squared +0.262. Random forest:
RMSE 0.485, pooled R squared +0.171.

**Model comparisons.** Gaussian process and support vector regression both beat the zero
baseline and the training fold mean with intervals excluding zero. Support vector regression
against the zero baseline also survives the family wise correction. Eleven models were
compared in total.

**Classification arm.** Volume spike label, logistic regression: ROC AUC 0.711, interval
[0.507, 0.889], balanced accuracy 0.564, beats baseline true.

**Result files.** `artifacts/models/results_regression.pkl`,
`artifacts/tables/verdict_table.parquet`,
`artifacts/tables/classification_summary.parquet`,
`artifacts/results/frozen_baseline.json`.

**Verification.** Every number above is held fixed by twenty assertions in
`tests/test_volume_target_frozen.py` at a tolerance of 1e-9.

## Market recovery days

**Target statistics.** Seventy four events. Fifty seven observed recoveries, seventeen
censored, of which nine were censored by a competing later disaster and eight at the ninety
day cap. Fifty one events showed a genuine post event drawdown. Median four trading
sessions, mean 16.6, with fifty seven per cent recovering within five sessions.

**Validation.** The same folds, forty pooled out of fold test points, thirty one observed
recoveries and nine censored.

**Evaluation metrics.** Two stage drawdown classifier plus Weibull survival model at twenty
features: concordance 0.657, interval [0.522, 0.769], integrated Brier score 0.194, median
absolute error among uncensored recoveries four sessions, worst calibration gap 4.5
percentage points, Holm corrected p 0.165. Kaplan Meier baseline: concordance 0.477.

**Model comparisons.** Fifteen models. Two have intervals excluding chance on a single
comparison, neither survives the family wise correction. The plain single stage survival
models score between 0.539 and 0.598.

**Recovery probability calibration.** Predicted against Kaplan Meier observed, for the best
model: 0.655 against 0.650 at ten sessions, 0.713 against 0.758 at twenty, 0.745 against
0.788 at thirty, 0.802 against 0.788 at sixty, 0.823 against 0.788 at ninety.

**Category result.** The pre specified recovery within twenty sessions category is not
predictable. Best AUC 0.612 with an interval of [0.347, 0.854], and no model beats chance.
Prevalence is 0.789, which leaves little to discriminate.

**Result files.** `artifacts/tables/recovery_grid_metrics.parquet`,
`artifacts/tables/recovery_grid_predictions.parquet`,
`artifacts/tables/recovery_probability_calibration.parquet`,
`artifacts/tables/recovery_category_metrics.parquet`,
`docs/thesis_materials/final_table_recovery.csv`.

## Results that could not be produced

1. **A market only arm of the direction analysis.** The direction analysis was pre declared
   on the combined information set only. Adding a market only arm after seeing that the
   combined arm works would be a post hoc extension, and the stop condition in the
   pre declaration forbids it. The consequence is that no incremental claim can be made for
   direction, and that is stated as a limitation rather than filled in.
2. **A final lockbox holdout.** Not achievable at seventy four events across a thirty by ten
   walk forward. Setting aside a further tail would cost folds the study cannot spare. The
   residual adaptive selection risk is disclosed instead.
3. **A random survival forest for recovery duration.** Declared in advance as unsuitable at
   thirty one observed recoveries, and deliberately not fitted.
