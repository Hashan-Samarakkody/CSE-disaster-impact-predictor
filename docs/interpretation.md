# Interpretation

What this study can claim, what it cannot claim, and how to write both up. Every claim
below traces to a number in `docs/results.md`, and every number there traces to a file in
`artifacts/`.

## Contents

1. The one paragraph answer
2. What can be said about each of the three targets
3. Claims that must not be made
4. Thesis revision guide, chapter by chapter

---

# Part 1. The one paragraph answer

When a qualifying disaster strikes Sri Lanka, this study can say two things about the
Colombo Stock Exchange. First, it can estimate how unusual trading volume will be over the
five sessions that follow. That estimate is imprecise but better than chance, and it is
the study's strongest continuous result. Second, at a ten session horizon it can indicate
with reasonable confidence which way the index will move, and that is the one finding that
survives a family wise correction. It cannot say how far the index will move, and it can
say nothing useful about recovery duration, in days or in rank order. Both of those were
tested systematically and found to be unpredictable at this sample size.

The economically interesting version of the same sentence: the market's attention to
disasters is predictable, its repricing is not. The event study sharpens that further: the realised return response is not merely unforecastable, it is not detectable at all, CAAR(1,5) = +0.0006 with every test p above 0.85.

---

# Part 2. What can be said about each of the three targets

## 2.1 Forward abnormal volume: the size can be estimated

Evidence: pooled out of sample R squared of 0.334 for the ensemble and 0.303 for the
random forest, pooled RMSE 0.488 against a target whose own pooled standard deviation is
0.607, on 34 test points. Under the T8 model parity rule the random forest is the
strongest model in the confirmatory set, and its paired episode clustered bootstrap
interval excludes zero against both naive baselines, [0.037, 0.175] against the zero
baseline. It does not survive the Holm family wise correction, at corrected p 0.078.

Two qualifications belong beside that number and must travel with it. First, the earlier
draft of this document reported a Holm corrected p of 0.021 and called the target
predictable. That figure came from a comparison in which the shallow network carried fixed
hyperparameters while the tree models were tuned, which T8 identified as an invalid
comparison; once every model is selected on the same purged inner splits, the advantage
shrinks and no longer survives correction. Second, the target is observed for only 34 of
74 events, and the missingness is structured rather than random: market wide volume is
unavailable for the 2000 archive year and for events after 2023, which is the first and
the last of the four folds, so the estimate rests on the middle of the sample period.

Safe sentence for the thesis:

> Forward abnormal trading volume shows the strongest evidence of out of sample
> predictability in this study, with a model explaining roughly a third of the variation
> in the five session volume response and an uncertainty interval excluding the no effect
> baseline. The evidence is suggestive rather than conclusive: it does not survive
> correction for multiplicity, it rests on 34 held out observations, and the missing
> labels are concentrated in the first and last validation folds.

What this does not license: a precise number. The typical absolute error is 0.38 in log
ratio units, which is a factor of about 1.5 either way, so a prediction of twenty per cent
above normal turnover is consistent with anything from roughly twenty per cent below to
eighty per cent above. Write "can estimate", not "can tell you".

## 2.2 ASPI return: direction yes, magnitude no

The magnitude result is a negative one and is reported as such. Across 465 configurations
and 1350 statistical comparisons, no confidence interval excluded zero, and within the 18
comparison confirmatory family the smallest Holm corrected p is 0.641. The best pooled R
squared in the notebook arm is 0.042.

The direction result is genuine and is the only finding in the study that survives a
family wise correction. At the ten session horizon, logistic regression on the combined
information set with ten selected features reaches a ROC AUC of 0.817, episode clustered
interval [0.657, 0.940], Holm corrected p below 0.001, with a balanced accuracy of 0.757
and a Matthews correlation of 0.530.

The direction model is not symmetric, and this must be stated rather than buried.
Sensitivity is 0.619 against a specificity of 0.895. In words: when the index did fall
over the following ten sessions the model caught it about three times in five, and when it
did not fall the model said so correctly about nine times in ten.

Safe sentences for the thesis:

> The size of the index response to a disaster is not predictable from pre event
> information at this sample size. This was tested systematically across a pre declared
> grid and is reported as a finding.

> At a ten session horizon the direction of the index response is predictable above
> chance. The model is a more reliable indicator that the index will not decline than a
> detector of declines.

This result holds at ten sessions only. At five sessions it fails outright, AUC 0.574. At
fifteen sessions the interval excludes chance on a single test, AUC 0.694, but does not
survive correction, and at twenty sessions it fails.

## 2.3 Market recovery duration: no result

Under the frozen target protocol this target returns nothing. Half of the events that fall
at all regain their pre event level within five sessions, and the mean duration is 14.3
because a few very slow recoveries pull the average out, so a single number prediction has
nothing to lock onto.

Neither does the ranking. The best of the fifteen survival configurations reaches a
Harrell concordance of 0.535 with an episode clustered interval of [0.371, 0.697], against
0.503 for a training fold Kaplan Meier baseline. Every interval in the family contains
0.5. On the point regression side every model loses to the training mean baseline on RMSE,
and the two stage hurdle loses to it too, at a mean absolute error of 27.5 sessions
against 18.3 for the training mean.

The sample is the reason. Forty pooled test points carry only fourteen observed
recoveries against twenty six censored ones, and twenty two of the seventy four events
never fell below their pre event level at all, so they pose no recovery question and are
excluded from the conditional stage by construction.

Safe sentence for the thesis:

> Recovery duration is not predictable from pre event information at this sample size,
> neither as a duration nor as a ranking. Fourteen observed recoveries in the pooled test
> set is too thin a base for a survival model with twenty covariates, and this is reported
> as a negative finding rather than worked around.

## 2.4 Summary table

| Target | What is predictable | Best validated model | Evidence | Statistically supported |
|---|---|---|---|---|
| Y1, return magnitude | nothing | none | 0 of 1350 comparisons with an interval excluding zero; 0 of the 18 confirmatory comparisons survive Holm | No |
| Y1, return direction at ten sessions | the sign of the return | logistic regression, combined features, ten features | AUC 0.817, interval [0.657, 0.940], Holm p below 0.001 | Yes |
| Y2, forward abnormal volume | possibly the size of the response | random forest, the strongest confirmatory model | pooled R squared 0.303 for the random forest and 0.334 for the ensemble; interval excludes zero against both baselines but Holm corrected p is 0.078; 34 held out points with structured missingness | Qualified |
| Y3, recovery duration | nothing | none | best concordance 0.535, interval [0.371, 0.697]; every model loses to the training mean on RMSE | No |

---

# Part 3. Claims that must not be made

1. That the model predicts how far the index will move after a disaster. It does not.
2. That disaster severity information improves return prediction beyond market state. Not
   demonstrated: the combined set beats the market only expected return baseline at three
   of four horizons and loses at the fourth, every interval spanning zero.
3. That the normal market plus disaster residual decomposition improved anything. It was
   the worst of the four information sets at three of the four horizons.
4. That recovery duration can be predicted, as a number of sessions or as a ranking. It
   cannot, on either measure.
5. That a positive pooled R squared is evidence of skill on its own. The best one in the
   return grid is 0.092 and carries a delta RMSE interval of [-0.388, +1.046].
6. That any feature is important on the strength of one fold's selection. Across four
   folds, 35.7 per cent of all selected features were selected in exactly one fold.
7. That the ten session direction result transfers to other horizons or to magnitude. It
   does not.
8. That the study found nothing. Two of the four questions asked return a supported
   positive answer, and the negatives are themselves findings about market efficiency at
   the event horizon.

---

# Part 4. Thesis revision guide, chapter by chapter

Written for the thesis author. The guidance below says what to change, what to add, what
to delete, and gives wording for the claims that are now defensible.

## 4.1 The one paragraph summary of what changed

All three targets were re specified on 2026-09-19 in
`docs/TARGET_DEFINITION_PROTOCOL.md`, before any performance under the new definitions was
observed, and every number in this repository now comes from that specification. Under it,
forward abnormal volume remains the supported continuous result and is stronger than
before. Return magnitude is a systematic negative finding rather than a weak positive one,
which is more defensible. Return direction at ten sessions is predictable above chance and
is the only finding that survives a family wise correction. Recovery duration, re founded
on right censored survival analysis, now returns no result at all: the concordance
interval contains chance and every point model loses to the training mean.

## 4.2 Chapter 3, methodology

Replace the two horizon description of the return target with the four pre declared
horizons:

> For each qualifying event the index return is measured over h in {5, 10, 15, 20} trading
> sessions as 100 ln(Ph divided by P0), where P0 is the last close fully observed before
> the disaster became known and Ph is the close of the hth complete trading session after
> it. Horizons are counted in trading sessions, never calendar days, and counting starts
> at one, so the five session horizon closes on the fifth complete session. Five sessions
> is the principal target; ten, fifteen and twenty are pre specified sensitivity analyses.
> All four were declared before any was scored.

Add a subsection on information sets. This is the methodological core of the revision:

> Three nested information sets are compared on identical folds and identical test events:
> market only, thirty three strictly pre event market, macro, exchange rate and global
> market state variables; disaster only, thirty five event identity, severity, exposure and
> hazard variables; and combined. A fourth architecture decomposes the target into a normal
> market component, estimated from the daily index series using only sessions whose own
> label settled strictly before the event, and a disaster residual. The primary comparison
> is combined against market only, which asks whether disaster specific information
> contributes anything beyond the market's existing condition.

Replace the point regression framing of recovery duration:

> Recovery duration is right censored by construction. Thirty eight of seventy four events
> carry no observed recovery: twenty two never fell below their pre event level and so pose
> no recovery question, ten were censored by a competing later qualifying disaster, and six
> reached the ninety session design cap. The primary analysis is therefore a two stage
> model, a drawdown classifier followed by an accelerated failure time fit on the drawdown
> events only. Synthetic oversampled rows are excluded from every survival fit,
> because interpolation can produce a duration but not a valid event indicator, so a
> synthetic row would assert a recovery that was never observed.

Add the inner cross validation degradation rule:

> Where purging leaves an inner split without a viable training side, the number of inner
> splits is reduced, and if no purged configuration remains, pre specified default
> hyperparameters are used. Reverting to an unpurged split is never permitted.

Add the survival metrics and demote pooled RMSE for the recovery target:

> Survival models are evaluated by Harrell concordance as the primary metric, an inverse
> probability of censoring weighted integrated Brier score, and calibration of the predicted
> recovery probabilities against a Kaplan Meier estimate. Point errors are reported only
> over genuinely observed recoveries and are treated as secondary. An RMSE computed as if
> censored observations were exact recovery times is not reported as a primary result.

## 4.3 Chapter 4, results

Rewrite the return magnitude section as a negative result, using the wording in Part 2.2.
Insert the sixteen row table from `docs/results.md` and put the full grid in an appendix
from `docs/thesis_materials/final_table_aspi.csv`. Do not select rows from it.

Give return direction its own subsection, clearly separated from magnitude, and follow it
immediately with the caveat that classification success is not regression success.

Rewrite the volume target section against the new definition. It is now a log ratio over
the five sessions after the prediction origin rather than a ratio minus one over the event
day window, so the reported units, the worked example and the conversion back to a
percentage all change. The result itself is stronger under the new definition, and it is
mechanically pinned by `tests/test_volume_target_frozen.py`.

Replace the recovery section entirely. Under the frozen protocol this target returns no
result: report the concordance and its interval, the fact that every point model loses to
the training mean, and the observed recovery count that explains why. Keep the calibration
table as evidence of what was attempted, not as a positive finding.

Add a feature stability paragraph:

> Across four folds, 35.7 per cent of all selected features were selected in exactly one
> fold and 20.0 per cent in every fold. No variable is described as important on the basis
> of a single fold's selection. At the ten session horizon on the combined information set
> the one session lagged return is selected in every fold, and the DesInventar log affected
> population in three of four.

## 4.4 Chapter 5, discussion and limitations

1. Frame the magnitude finding as informative about market efficiency at the event horizon,
   not as a failed experiment.
2. Discuss the asymmetry between direction and magnitude. It is consistent with a market
   that reprices reliably in sign but whose size is dominated by idiosyncratic noise at this
   sample size.
3. Explain concretely that thirty eight of seventy four events carry no observed recovery
   and that scoring them as observed inflates apparent recovery performance.
4. State that with 1350 return comparisons plus 16 survival models plus 10 direction
   comparisons, several nominally significant results are expected by chance, which is why
   the Holm correction is reported, and that only the ten session direction result
   survives it.
5. Keep the small sample limitations: no lockbox holdout, forty pooled test points, and
   only fourteen observed recoveries inside them, which is the most likely reason the
   recovery target returns nothing under any of the fifteen configurations tried.
6. State the pre declared omission: the direction analysis was declared on the combined
   information set only, so no incremental claim can be made for direction. Adding that arm
   after seeing the result would be a post hoc extension and is deliberately not done.

## 4.5 Figures and tables to regenerate

| Item | Source |
|---|---|
| Return results table, sixteen row extract | `docs/results.md` |
| Return full grid, appendix | `docs/thesis_materials/final_table_aspi.csv` |
| Direction table | `artifacts/tables/aspi_direction_metrics.parquet` |
| Feature stability table | `artifacts/tables/aspi_grid_feature_stability.parquet` |
| Recovery survival table | `docs/thesis_materials/final_table_recovery.csv` |
| Recovery calibration table | `artifacts/tables/recovery_probability_calibration.parquet` |
| Per event recovery probabilities | `artifacts/tables/recovery_grid_predictions.parquet` |
| Volume target results | `artifacts/tables/verdict_table.parquet` |

Rebuild the two final tables with `python scripts/build_final_tables.py`.
