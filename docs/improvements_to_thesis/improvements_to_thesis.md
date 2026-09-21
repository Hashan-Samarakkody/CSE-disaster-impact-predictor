# Improvements to the thesis

What the written thesis needs to change so that it describes the implementation that now
exists in this repository.

## Scope and an honest limitation of this document

The thesis document itself was not supplied for this comparison. Nothing below is invented
from a guess about what the thesis says. Every entry is anchored in something the
repository's own prior documentation asserted, which the thesis is likely to mirror because
those documents were written alongside it. Where a statement could not be verified against
the thesis text, the entry says so and asks the author to check.

Some entries are not mismatches at all but results that postdate any earlier thesis draft,
and they are marked as new material rather than corrections.

Every number quoted below comes from a complete re run of the repository on 2026-09-21
under the target definitions frozen on 2026-09-19.

---

## 1. The targets were re specified, and that supersedes every earlier result

**Thesis section.** Methodology, target definitions, and every results section that quotes
a number.

**Current thesis statement.** Whatever the thesis says, it almost certainly describes the
pre 2026-09-19 definitions: a five session return ending at `P[t0+5]`, a volume target
expressed as a ratio minus one over the event day window, and a recovery clock whose scan
begins at the trough with no drawdown events recorded as recovered.

**Actual repository implementation.** On 2026-09-19 the three targets were re specified in
`docs/TARGET_DEFINITION_PROTOCOL.md` and frozen there before any performance under the new
definitions was observed. The changes are:

| Quantity | Before | Now |
|---|---|---|
| Y1 endpoint | `P[t0+5]`, the sixth post event session | `P5 = market[position + 4]`, the fifth complete session after the event becomes known |
| Y2 | `mean(V[0..+4]) / V_base - 1`, a ratio minus one | `ln(mean(V1..V5) / V_base)`, a log ratio, unbounded both ways |
| Y2 column | `Y2_abnormal_volume` | `Y2_5D_Forward_AbnormalVolume_LogRatio` |
| Y3 drawdown gate | `P[0..+5]`, six sessions | `P1..P5`, five sessions |
| Y3 scan origin | the trough of the gate window | `k = 1`, the first session after the origin |
| Y3 no drawdown | recorded as `recovered`, not censored | its own state: duration 0, no event observed, censor reason `no_drawdown` |
| Y3 column | `Y3_recovery_days` | `Y3_ASPI_Recovery_Time` |

**Mismatch.** Every result number published before that date was produced under the old
definitions and does not carry over. This is not a correction of an error: the author
changed the specification deliberately, and the earlier arithmetic was internally correct
under the earlier definitions.

**Required thesis improvement.** Replace the target definition section wholesale with
`docs/target_definitions.md`, which states each target precisely with the observed
distributions and a worked example, and cite
`docs/TARGET_DEFINITION_PROTOCOL.md` as the frozen specification. Then replace every
results number. Do not mix numbers from the two regimes anywhere in the document.

**Repository evidence.** `docs/TARGET_DEFINITION_PROTOCOL.md`,
`docs/target_definitions.md`, `src/targets/event_targets.py`, and the naming note at the
head of `docs/audit.md`.

**Result impact.** Methodology, results, discussion and conclusions. Everything.

---

## 2. The recovery target headline

**Thesis section.** Results, recovery duration.

**Current thesis statement.** The recovery duration model achieves a pooled out of sample
R squared of about +0.155, with an ensemble as the best model.

**Actual repository implementation.** Under the frozen protocol this target returns
nothing. The best of fifteen survival configurations reaches a Harrell concordance of
0.535 with an episode clustered interval of [0.371, 0.697], against 0.503 for a training
fold Kaplan Meier baseline. No interval in the family excludes 0.5. On the point
regression side every model loses to the training mean baseline, and the two stage hurdle
reaches a mean absolute error of 27.5 sessions against 18.3 for that baseline.

The reason is the observed event count. Forty pooled test points carry only fourteen
observed recoveries against twenty six censored ones, and twenty two of the seventy four
events never fell below their pre event level at all.

**Mismatch.** A reported positive becomes a clean negative. The earlier R squared was
computed by point regression that treats every censored observation as an exact recovery
time, which flatters the result.

**Required thesis improvement.** Rewrite the section as a negative finding. Report the
concordance and its interval, state that every point model loses to the training mean,
and give the observed recovery count as the reason. Keep the calibration table as evidence
of what was attempted, not as a positive result: its best worst case gap is 0.162, against
0.166 for the marginal survival curve, so the probabilities are no better calibrated than
the baseline.

**Repository evidence.** `scripts/run_recovery_survival_grid.py`,
`artifacts/tables/recovery_grid_metrics.parquet`,
`artifacts/tables/recovery_probability_calibration.parquet`,
`artifacts/tables/hurdle_table.parquet`,
`docs/thesis_materials/final_table_recovery.csv`, `src/evaluation/survival_metrics.py`.

**Result impact.** Methodology, results, discussion and conclusions.

---

## 3. Return magnitude predictability

**Thesis section.** Results, index return.

**Current thesis statement.** Likely to report a best pooled R squared near +0.14 for the
five session return and to treat it as a weak positive.

**Actual repository implementation.** A pre declared grid of 240 configurations, four
horizons by four information sets by three feature capacities by five model families, was
run against three baselines. None of the 720 comparisons produced an interval excluding
zero. The best pooled R squared anywhere in the grid is +0.092, at ten sessions on the
disaster only set, and its delta RMSE interval against the market only expected return
baseline is [-0.388, +1.046].

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

## 4. Whether disaster information adds anything

**Thesis section.** Results and discussion, the incremental value of disaster severity
data.

**Current thesis statement.** Likely to claim or imply that the external severity and
hazard blocks improved prediction, on the strength of the block ablation.

**Actual repository implementation.** The block ablation remains in the record. The direct
test, the combined information set against the market only expected return baseline on
identical folds and identical test events, does not support the claim: combined wins at
three of four horizons and loses at the fourth, with every interval spanning zero. On
lowest RMSE alone the disaster only set is best at five and ten sessions and the market
only set at twenty, which is not a pattern that supports an incremental claim either way.

**Mismatch.** The block ablation answers a narrower question than the one the thesis needs.
It shows that removing a block changes error, not that disaster information beats market
state.

**Required thesis improvement.** Report both. State that the incremental value of disaster
specific information for return magnitude is not demonstrated, and keep the block ablation
as the secondary evidence it is.

**Repository evidence.** `src/targets/return_horizons.py` for the information set
partition, `artifacts/tables/aspi_grid_metrics.parquet`,
`artifacts/tables/ablation_blocks.parquet`.

**Result impact.** Results, discussion, conclusions.

---

## 5. The target naming

**Thesis section.** Methodology, target definitions.

**Current thesis statement.** The three targets are named in the thesis in its own words.

**Actual repository implementation.** The implementation carries frozen column names that
begin with Y1, Y2 and Y3, because those names appear inside cached artifacts and inside the
regression test that holds the volume target fixed. Readable aliases exist in
`src/config/settings.py`: `ASPI_PERCENTAGE_CHANGE`, `VOLUME_CRASH_MAGNITUDE` and
`MARKET_RECOVERY_DAYS`.

Two of those aliases are now misleading in themselves. `VOLUME_CRASH_MAGNITUDE` names a
quantity that is positive for twenty three of its sixty one observations, and
`MARKET_RECOVERY_DAYS` counts trading sessions, not days.

**Mismatch.** Two vocabularies for the same three quantities, and two alias names that
misdescribe what they hold. Whether the thesis uses the repository's names is unverified
and the author should check.

**Required thesis improvement.** Use "forward abnormal trading volume" and "recovery
duration in trading sessions" throughout the prose, and add one table to the methodology
chapter mapping the thesis name, the alias and the frozen column name for each target, so
a reader moving between the document and the code is never in doubt.

**Repository evidence.** `src/config/settings.py`, `docs/architecture.md` section 2,
`docs/target_definitions.md` sections 3 and 4.

**Result impact.** Methodology and presentation. No number changes.

---

## 6. Apparent extra targets

**Thesis section.** Methodology, scope.

**Current thesis statement.** Three prediction targets.

**Actual repository implementation.** The dataset also carries ten, fifteen and twenty
session variants of the return target and six binary classification labels. The variants
are pre registered sensitivity analyses of target one. The labels are binary views of the
same three targets, including `C0_drawdown_occurs`, which is stage one of the protocol's
own two stage recovery architecture rather than a separate question.

**Mismatch.** A reader opening the dataset counts more than three target shaped columns and
may reasonably wonder whether the scope grew.

**Required thesis improvement.** Add one sentence to the scope section stating that the
horizon variants and the classification labels are derived views of the same three targets,
and pointing at `docs/architecture.md` section 2.

**Repository evidence.** `src/config/settings.py`, `src/models/classifiers.py`,
`src/targets/event_targets.py`.

**Result impact.** Methodology and presentation. No number changes.

---

## 7. New material: return direction at ten sessions

**Thesis section.** Results, new subsection required.

**Current thesis statement.** No equivalent. The earlier direction analysis covered the five
session horizon only, where it failed.

**Actual repository implementation.** Direction was tested at all four pre declared horizons
as an explicitly separate question from magnitude. At ten sessions, logistic regression on
the combined information set with ten features reaches ROC AUC 0.817, episode clustered
interval [0.657, 0.940], and it is the only result in the study that survives the family
wise correction, at Holm p below 0.001.

**Mismatch.** New evidence, not a correction.

**Required thesis improvement.** Add a results subsection, clearly separated from magnitude,
and immediately follow it with the asymmetry: sensitivity 0.619 against specificity 0.895,
so the model is a reliable indicator that the index will not fall and a weaker detector of
falls. State explicitly that classification success is not regression success, and that the
result does not hold at five or twenty sessions.

**Repository evidence.** `artifacts/tables/aspi_direction_metrics.parquet`,
`scripts/run_aspi_return_grid.py`.

**Result impact.** Results, discussion and conclusions. This is the study's strongest
claim.

---

## 8. New material: feature selection stability

**Thesis section.** Results, feature importance.

**Current thesis statement.** Likely to name individual features as important, on the
strength of importance rankings or single fold selections.

**Actual repository implementation.** Selection stability is measured. Across four folds,
35.7 per cent of all selected features were selected in exactly one fold and 20.0 per cent
in every fold.

**Mismatch.** Any claim that a particular feature matters needs that number beside it.

**Required thesis improvement.** Add the stability paragraph and restrict feature level
claims to variables with a selection frequency of 0.75 or above. At the ten session horizon
on the combined information set, the one session lagged return is selected in every fold
and the DesInventar log affected population in three of four.

**Repository evidence.** `artifacts/tables/aspi_grid_feature_stability.parquet`.

**Result impact.** Results and discussion. It constrains what the explainability chapter may
claim.

---

## 9. New material: the failed decomposition

**Thesis section.** Methodology and results.

**Current thesis statement.** No equivalent.

**Actual repository implementation.** A normal market plus disaster residual architecture was
implemented and tested: a market only expected return model estimated from daily sessions
that settled strictly before each event, with a disaster residual modelled on top. It did not
improve prediction at any horizon and was the worst of the four information sets at three of
the four horizons.

**Mismatch.** New negative result.

**Required thesis improvement.** Report it. A tested and failed hypothesis is a contribution,
and omitting it would leave the pre declaration in `docs/audit.md` describing an experiment
the thesis never mentions.

**Repository evidence.** `scripts/run_aspi_return_grid.py`,
`artifacts/tables/aspi_grid_metrics.parquet` rows with information set
`normal_plus_residual`.

**Result impact.** Methodology, results, discussion.

---

## 10. New material: the hurdle model and the no drawdown state

**Thesis section.** Methodology and results, recovery duration.

**Current thesis statement.** Likely to describe a hurdle model that classifies recovery
within the window and then regresses duration, fitted on all events.

**Actual repository implementation.** Under the frozen protocol an event with no drawdown
carries duration zero as a recorded state, not as an instant recovery, and the protocol
forbids treating it as an ordinary duration observation. The hurdle is therefore fitted and
scored on drawdown events only, twenty three pooled test rows. A defect found during the
2026-09-21 validation pass had been feeding the no drawdown events in as censored
durations, which made the model predict the ninety session cap against a true zero and
produced a mean absolute error of 56.0 sessions. Corrected, it is 27.5, still worse than
every baseline.

**Mismatch.** If the thesis quotes any hurdle error figure, it is wrong under either
version.

**Required thesis improvement.** State the exclusion rule explicitly, quote the corrected
figure, and keep the negative result.

**Repository evidence.** `src/models/hurdle.py`, `artifacts/tables/hurdle_table.parquet`,
notebook 05 section 5.4, `scripts/train_final_models.py`.

**Result impact.** Methodology and results.

---

## 11. Repository structure and file paths

**Thesis section.** Any appendix or methodology passage citing a file path.

**Current thesis statement.** Paths from the pre refactor layout, for example
`src/data_pipeline/feature_eng.py` or `app.py`.

**Actual repository implementation.** The layout was reorganised on 2026-09-17. Target
construction now lives in `src/targets/`, feature engineering in `src/features/`, loaders in
`src/data/`, figures in `src/visualization/`, the demo in `apps/streamlit_app.py`, and the
dependency list in `config/requirements.txt`.

**Mismatch.** Any path the thesis cites from the old layout is now broken.

**Required thesis improvement.** Re check every file path the thesis cites against
`docs/architecture.md`.

**Repository evidence.** `docs/refactor_validation.md` lists every move and rename.

**Result impact.** Presentation and reproducibility. No number changes.

---

# Verified final research outputs

Everything below was produced by executing this repository in full on 2026-09-21, under
the target definitions frozen on 2026-09-19. No value here is estimated or carried over
from an earlier run, and where a result could not be produced the reason is stated.

## Target one, ASPI return magnitude and direction

**Target statistics.** Seventy four events, seventy four observed values at the five
session horizon. Mean +0.339 percentage points, standard deviation 2.680, median -0.121,
range -5.397 to +8.797. 51.4 per cent of events are negative.

**Validation.** Chronological walk forward, thirty training events, ten test events, step
ten, four folds, forty pooled out of fold test points.

**Evaluation metrics.** Best configuration anywhere in the pre declared grid: pooled R
squared +0.092, RMSE 4.830, MAE 3.549, at the ten session horizon on the disaster only
information set, elastic net with five features. Best five session configuration: pooled R
squared +0.038, RMSE 2.690, disaster only, random forest with five features. Neither is
distinguishable from its baseline.

**Model comparisons.** 240 configurations, 720 paired comparisons against a zero return
baseline, a training fold mean and a market only expected return model. Comparisons whose
interval excludes zero: **zero**. Surviving the Holm correction: zero.

**Direction, reported as a separate question.** Ten session horizon, logistic regression,
combined information set, ten features: ROC AUC 0.817, episode clustered interval
[0.657, 0.940], balanced accuracy 0.757, Matthews correlation 0.530, precision recall AUC
0.825, sensitivity 0.619, specificity 0.895, Holm corrected p below 0.001. At five
sessions AUC 0.574 and the interval contains chance; at fifteen sessions AUC 0.694 with an
interval excluding chance but no correction survival; at twenty sessions AUC 0.638 and the
interval contains chance.

**Result files.** `artifacts/tables/aspi_grid_metrics.parquet`,
`artifacts/tables/aspi_grid_verdicts.parquet`,
`artifacts/tables/aspi_direction_metrics.parquet`,
`artifacts/tables/aspi_grid_feature_stability.parquet`,
`docs/thesis_materials/final_table_aspi.csv`.

## Target two, forward abnormal trading volume

**Target statistics.** Sixty one observed values of seventy four events, because volume is
absent for the 2000 archive year and for post 2023 events. Mean -0.134 log points,
standard deviation 0.594, median -0.083, quartiles -0.494 and +0.321, range -1.464 to
+1.120. Twenty three of the sixty one are positive.

**Validation.** The same folds, thirty four pooled out of fold test points.

**Evaluation metrics, pooled.** Ensemble: RMSE 0.488, MAE 0.382, pooled R squared +0.334.
Shallow network: RMSE 0.491, pooled R squared +0.326. Stacked: RMSE 0.549 on twenty four
points, pooled R squared +0.315. Random forest: RMSE 0.500, MAE 0.400, pooled R squared
+0.303. Gaussian process: +0.281. Extreme gradient boosting: +0.254.

**Model comparisons.** Nine of the thirteen comparisons in the whole study whose interval
excludes zero belong to this target. The ensemble beats both naive baselines, and its
advantage over the zero baseline survives the family wise correction at Holm p 0.021. The
random forest beats both baselines on a single test. Eleven models were compared.

**Classification arm.** Volume spike label, random forest: accuracy 0.647, balanced
accuracy 0.612, precision 0.545, recall 0.462, F1 0.500, ROC AUC 0.656, bootstrap interval
[0.454, 0.839]. It does not clear both the majority rule and chance, so the classification
arm of this target is **not** a positive result even though the regression arm is.

**Result files.** `artifacts/models/results_regression.pkl`,
`artifacts/tables/verdict_table.parquet`,
`artifacts/tables/classification_summary.parquet`,
`artifacts/results/frozen_baseline.json`.

**Verification.** Every number above is held fixed by twenty assertions in
`tests/test_volume_target_frozen.py` at a tolerance of 1e-9, against a baseline re frozen
on 2026-09-21.

## Target three, market recovery duration

**Target statistics.** Seventy four events. Thirty six observed recoveries and thirty eight
without one: twenty two never fell below their pre event level, ten were censored by a
competing later disaster and six reached the ninety session cap. Fifty two events showed a
qualifying drawdown. Median duration four sessions, mean 14.257, standard deviation 24.787.
Among the thirty six observed recoveries the median is five sessions, the upper quartile
8.5 and the maximum thirty five.

**Validation.** The same folds, forty pooled out of fold test points carrying fourteen
observed recoveries and twenty six censored ones.

**Evaluation metrics.** Best of fifteen survival configurations, the two stage log normal
model at twenty features: concordance 0.535, episode clustered interval [0.371, 0.697],
integrated Brier score 0.338, mean absolute error among uncensored recoveries 12.8
sessions, Holm corrected p 1.000. Kaplan Meier training fold baseline: concordance 0.503.
**No configuration has an interval excluding 0.5.**

**Point regression.** Every model loses to the training mean baseline on RMSE. The best
pooled R squared is -0.037, held by the training mean itself. The two stage hurdle, fitted
and scored on drawdown events only, reaches a mean absolute error of 27.5 sessions against
18.3 for the training mean and 21.0 for predicting zero.

**Category result.** The pre specified recovery within twenty sessions category is not
predictable. Best AUC 0.440 with an interval of [0.135, 0.736]; no model beats chance and
most score below it. Prevalence is 0.65.

**Calibration.** The smallest worst case calibration gap across the five horizons is 0.162,
held by the single stage log normal model at twenty features, against 0.166 for the
Kaplan Meier baseline. The probabilities are therefore no better calibrated than the
marginal survival curve.

**Result files.** `artifacts/tables/recovery_grid_metrics.parquet`,
`artifacts/tables/recovery_grid_predictions.parquet`,
`artifacts/tables/recovery_probability_calibration.parquet`,
`artifacts/tables/recovery_category_metrics.parquet`,
`artifacts/tables/hurdle_table.parquet`,
`docs/thesis_materials/final_table_recovery.csv`.

## Secondary analyses

**Sector panel.** Thirty events across the sector indices, 594 rows. Nothing is
significant and every point estimate is negative: the models are worse than the naive
baselines at sector level. `artifacts/tables/sector_panel_verdict.parquet`.

**Conformal prediction intervals.** At a nominal 0.8, observed coverage is 0.667 for the
five session return, 0.500 for volume and 0.867 for recovery duration. Coverage undershoots
for the two targets that matter, which is the expected behaviour of split conformal
prediction at this calibration size. `artifacts/tables/conformal_coverage.parquet`.

**Event date precision.** Five of the seventy four events carry a month or a year rather
than an exact day. Re scoring on exact day events only moves the volume result from +0.303
to +0.251 and changes no conclusion.
`artifacts/tables/exact_date_sensitivity.parquet`.

**Oversampling.** Time aware SMOGN helps volume, pooled R squared +0.216 to +0.303, and
recovery, -0.175 to -0.116, and has no effect on either return column.
`artifacts/tables/smogn_ablation.parquet`.

**Conditional volatility.** Adding the GARCH conditional volatility feature changes the
five session return result by an amount indistinguishable from zero, delta RMSE -0.009
with an interval of [-0.268, +0.379].

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
   this number of observed recoveries, and deliberately not fitted.
4. **Any recovery duration result at all.** Fourteen observed recoveries in the pooled test
   set against a survival model with up to twenty covariates is not a workable ratio. This
   is reported as a negative finding, not as a failure to run the analysis: all fifteen
   configurations were fitted and all fifteen are reported.
5. **A re execution of notebook 01.** Its six external sources are live and unpinned, so re
   running it is a data refresh rather than a reproduction and would move every downstream
   number. The cached tables and their provenance sidecars stand as the record.
