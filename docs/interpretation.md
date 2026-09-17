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

When a qualifying disaster strikes Sri Lanka, this study can say three things about the
Colombo Stock Exchange. First, it can estimate how unusual trading volume will be on the
event day. That estimate is imprecise but better than chance, and it is the study's
strongest result. Second, at a ten session horizon it can indicate with reasonable
confidence that the index will not fall, although it detects actual falls less than half
the time. Third, it can state the probability that the market recovers within 10, 20, 30,
60 or 90 trading sessions, and those probabilities are well calibrated. It cannot say how
far the index will move, and it cannot say on which day recovery will occur. Both of those
were tested systematically and found to be unpredictable at this sample size.

The economically interesting version of the same sentence: the market's attention to
disasters is predictable, its repricing is not.

---

# Part 2. What can be said about each of the three targets

## 2.1 Volume crash magnitude: the size can be estimated

Evidence: pooled out of sample R squared of 0.27, RMSE 0.455, against a target whose own
standard deviation is 0.58. Gaussian process and support vector regression both beat the
zero baseline and the training fold mean with a paired episode clustered bootstrap
interval excluding zero. Support vector regression against the zero baseline also survives
a Holm family wise correction.

Safe sentence for the thesis:

> Abnormal trading volume is predictable out of sample. The model explains roughly 27 per
> cent of the variation in the event day volume response, and its advantage over a no
> effect baseline is statistically supported.

What this does not license: a precise number. The typical error is about 0.36 in ratio
units, so a prediction of twenty per cent above normal is consistent with anything from
fifteen per cent below to fifty five per cent above. Write "can estimate", not "can tell
you".

## 2.2 ASPI percentage change: direction yes, size no

The magnitude result is a negative one and is reported as such. Across 240 pre declared
configurations and 720 statistical comparisons, no confidence interval excluded zero. The
best pooled R squared anywhere in the grid is 0.075, with a delta RMSE interval of
[-0.560, +1.045].

The direction result is genuine and is the only finding in the study that survives a
family wise correction. At the ten session horizon, logistic regression on the combined
information set with ten features reaches a ROC AUC of 0.752, episode clustered interval
[0.567, 0.896], Holm corrected p of 0.032.

The direction model is not symmetric, and this must be stated rather than buried.
Sensitivity is 0.476 against a specificity of 0.842. In words: when the index did fall over
the following ten sessions the model caught it less than half the time, and when the index
did not fall the model said so correctly about five times in six.

Safe sentences for the thesis:

> The size of the index response to a disaster is not predictable from pre event
> information at this sample size. This was tested systematically across a pre declared
> grid and is reported as a finding.

> At a ten session horizon the model is a reliable indicator that the index will not
> decline. It is not a reliable detector of declines.

This result holds at ten sessions only. At five sessions it fails outright, AUC 0.552. At
fifteen and twenty sessions it is suggestive but does not survive correction.

## 2.3 Market recovery days: probabilities yes, the day no

Half the events recover within four trading sessions and the mean is 16.6, because a few
very slow recoveries pull the average out. That spread is exactly why a single number
prediction fails.

What works is ranking and probability. The two stage model, a drawdown occurrence
classifier followed by a Weibull accelerated failure time fit on the drawdown cases,
reaches a Harrell concordance of 0.657 with an episode clustered interval of
[0.522, 0.769], against 0.477 for a training fold Kaplan Meier baseline. Its recovery
probabilities are well calibrated, with a maximum deviation of 4.5 percentage points from
the observed Kaplan Meier value across five horizons.

Safe sentence for the thesis:

> Exact recovery duration cannot be predicted. Events can be ranked by likely recovery
> speed better than chance, and the model produces well calibrated probabilities of
> recovery within 10, 20, 30, 60 and 90 trading sessions. The ranking result does not
> survive a multiplicity correction and is reported as suggestive rather than established.

## 2.4 Summary table

| Target | What is predictable | Best validated model | Evidence | Statistically supported |
|---|---|---|---|---|
| ASPI percentage change, magnitude | nothing | none | 0 of 720 comparisons with an interval excluding zero | No |
| ASPI percentage change, direction at ten sessions | the sign of the return | logistic regression, combined features, ten features | AUC 0.752, interval [0.567, 0.896], Holm p 0.032 | Yes |
| Volume crash magnitude | the size of the response | support vector regression and Gaussian process | delta RMSE interval excludes zero against both baselines | Yes |
| Market recovery days | ranking and probabilities, not the day | two stage drawdown plus Weibull AFT | concordance 0.657, interval [0.522, 0.769], fails Holm | Suggestive |

---

# Part 3. Claims that must not be made

1. That the model predicts how far the index will move after a disaster. It does not.
2. That disaster severity information improves return prediction beyond market state. Not
   demonstrated: the combined set wins at three of four horizons and loses at the fourth,
   every interval spanning zero.
3. That the normal market plus disaster residual decomposition improved anything. It did
   not, and it was the worst information set at the fifteen and twenty session horizons.
4. That recovery duration can be predicted to a number of days. It cannot.
5. That the recovery concordance result is statistically established. It is suggestive and
   fails the Holm correction at p 0.165.
6. That a positive pooled R squared is evidence of skill on its own. The best one in the
   grid carries an interval of [-0.560, +1.045].
7. That any feature is important on the strength of one fold's selection. Across four
   folds, 35.6 per cent of all selected features were selected in exactly one fold.
8. That the ten session direction result transfers to other horizons or to magnitude. It
   does not.

---

# Part 4. Thesis revision guide, chapter by chapter

Written for the thesis author. The guidance below says what to change, what to add, what
to delete, and gives wording for the claims that are now defensible.

## 4.1 The one paragraph summary of what changed

The volume target is untouched and remains the statistically supported result. The return
magnitude result is now a systematic negative finding rather than a weak positive one,
which is stronger and more defensible. Return direction at ten sessions is newly shown to
be predictable above chance and is the only finding that survives a family wise
correction. Recovery duration is re founded on right censored survival analysis, which
replaces an inflated pooled R squared with an honest concordance index and, more usefully,
calibrated recovery probabilities.

## 4.2 Chapter 3, methodology

Replace the two horizon description of the return target with the four pre declared
horizons:

> For each qualifying event the index return is measured over h in {5, 10, 15, 20} trading
> sessions as 100 ln(P at t plus h divided by P at t minus 1), where t is the first valid
> trading session on or after the disaster date and P at t minus 1 is the last valid pre
> event close. Horizons are counted in trading sessions, never calendar days. Five sessions
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

> Recovery duration is right censored by construction. Seventeen of seventy four events are
> censored, eight at the ninety day design cap and nine by a competing later qualifying
> disaster. The primary analysis is therefore an accelerated failure time model fitted to
> genuine events only. Synthetic oversampled rows are excluded from every survival fit,
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

Leave the volume target section unchanged. It is now mechanically verified by
`tests/test_volume_target_frozen.py`.

Replace the recovery headline. Delete the pooled R squared of 0.155 as a headline number
and keep it only as a disclosed diagnostic of the distortion that point regression on
censored data introduces. Add the calibration table, which is the most practically useful
output the study has for this target.

Add a feature stability paragraph:

> Across four folds, 35.6 per cent of all selected features were selected in exactly one
> fold. No variable is described as important on the basis of a single fold's selection.
> The most consistently selected disaster variable is the DesInventar log affected
> population, and the most consistently selected market variable is the one session lagged
> return.

## 4.4 Chapter 5, discussion and limitations

1. Frame the magnitude finding as informative about market efficiency at the event horizon,
   not as a failed experiment.
2. Discuss the asymmetry between direction and magnitude. It is consistent with a market
   that reprices reliably in sign but whose size is dominated by idiosyncratic noise at this
   sample size.
3. Explain concretely that seventeen of seventy four events are censored and that scoring
   them as observed inflates apparent recovery performance.
4. State that with 720 plus 15 plus 8 comparisons, one nominally significant result is
   expected by chance, which is why the Holm correction is reported, and that the ten
   session direction result survives it.
5. Keep the small sample limitations: no lockbox holdout, forty pooled test points, thirty
   one observed recoveries, and a twenty covariate survival model against thirty one events,
   which is the most likely reason its corrected p value is 0.165.
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
| Volume target results | existing artifacts, do not regenerate |

Rebuild the two final tables with `python scripts/build_final_tables.py`.
