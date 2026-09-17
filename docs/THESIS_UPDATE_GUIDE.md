# Thesis update guide

Written for: the thesis author, revising the dissertation chapters after the 2026-09-17
Y1/Y3 improvement run.

What follows is chapter-by-chapter: what to change, what to add, what to delete, and the
exact wording to use for the claims that are now defensible — and the wording to avoid for
the ones that are not. Numbers cited here all trace to
`docs/Y1_Y3_FINAL_RESULTS.md` and `docs/thesis_materials/final_table_{y1,y3}.csv`.

---

## The one-paragraph summary of what changed

Y2 is untouched and remains the study's statistically supported result. Y1 return
**magnitude** is now shown to be unpredictable across a pre-declared 240-configuration grid
— a stronger and more publishable negative result than before. Y1 return **direction** at
the 10-session horizon is newly shown to be predictable above chance, and is the only
finding in the thesis that survives a family-wise correction. Y3 is re-founded on
right-censored survival analysis, which replaces an inflated pooled R2 with an honest
C-index of 0.657 and, more usefully, calibrated recovery probabilities.

---

## Chapter 3 — Methodology

**3.2.2 Target definitions.** Replace the two-horizon Y1 description with the four
pre-declared horizons:

> For each qualifying event, the ASPI return is measured over h in {5, 10, 15, 20} CSE
> trading sessions as `Y_h = 100 ln(P_{t+h} / P_{t-1})`, where `t` is the first valid
> trading session on or after the disaster date and `P_{t-1}` is the last valid pre-event
> close. Horizons are counted in trading sessions, never calendar days. h = 5 is the
> principal target; h = 10, 15 and 20 are pre-specified horizon-sensitivity analyses. All
> four were declared before any was scored (`docs/Y1_Y3_IMPROVEMENT_PREDECLARATION.md`).

**Add a new 3.2.3 — Information sets.** This is the methodological core of the revision:

> Three nested information sets are compared on identical folds and identical test events:
> market-only (33 strictly pre-event market, macro, FX and global-market state variables),
> disaster-only (35 event identity, severity, exposure and hazard variables), and combined.
> A fourth architecture decomposes the target into a normal-market component, estimated
> from the daily ASPI series using only sessions whose own h-day label settled strictly
> before the event, and a disaster residual. The primary scientific comparison is combined
> versus market-only: it asks whether disaster-specific information contributes
> incremental predictive value beyond the market's existing condition.

**3.2.4 Y3.** Replace the point-regression framing:

> Recovery duration is right-censored by construction: 17 of 74 events are censored (8 at
> the 90-day design cap, 9 by a competing later qualifying disaster). The primary analysis
> is therefore an accelerated failure time model fitted to genuine events only, with
> `event_observed = ~Y3_censored`. Synthetic (SMOGN) rows are excluded from every survival
> fit: interpolation can produce a duration, but not a valid event/censoring indicator, so
> a synthetic row would assert a recovery that was never observed. A two-stage architecture
> first estimates P(drawdown | X) and then models recovery duration for events that enter a
> recovery process.

**3.4 Validation.** Add the inner-CV degradation rule:

> Where purging leaves an inner split without a viable training side, the number of inner
> splits is reduced; if no purged configuration remains, pre-specified default
> hyperparameters are used. Reverting to an unpurged split is never permitted.

**3.5 Evaluation metrics.** Add the survival metrics and demote pooled RMSE for Y3:

> Survival models are evaluated by Harrell's C-index (primary), an inverse-probability-of-
> censoring-weighted integrated Brier score, and calibration of the predicted recovery
> probabilities against a Kaplan-Meier estimate. Point errors (MAE, median absolute error,
> RMSE) are reported only over genuinely observed recoveries and are treated as secondary;
> an RMSE computed as if censored observations were exact recovery times is not reported as
> a primary result.

---

## Chapter 4 — Results

### 4.1 Y1 magnitude — rewrite as a negative result

Delete any sentence implying that return magnitude is predictable. Use:

> Across 240 pre-declared configurations (4 horizons x 4 information sets x 3 feature
> capacities x 5 model families) and 720 paired, episode-clustered bootstrap comparisons
> against three baselines, **not one confidence interval excluded zero**. The best pooled
> out-of-sample R2 in the entire grid is +0.075 (h = 10, disaster-only, elastic net), with a
> delta-RMSE confidence interval of [-0.560, +1.045]. This study therefore finds no evidence
> that disaster-event ASPI return magnitude is predictable at N = 74.

Insert the Y1 table from `docs/Y1_Y3_FINAL_RESULTS.md` section 2.2 (16 rows: the best
configuration per horizon x information set). Put the full 240-row grid in an appendix from
`docs/thesis_materials/final_table_y1.csv` — do not cherry-pick rows.

State the primary comparison explicitly:

> Combined outperformed market-only at h = 5, 10 and 15 and underperformed it at h = 20, in
> every case with an interval spanning zero. The incremental value of disaster-specific
> information for return magnitude is therefore not demonstrated.

And report the failed decomposition rather than omitting it:

> The normal-market + disaster-residual decomposition did not improve prediction at any
> horizon and was the worst of the four information sets at h = 15 and h = 20. The Stage-A
> expected-return model itself carries very little skill (0.007 against a zero baseline at
> h = 5), which offers a coherent explanation: there is little ordinary market movement for
> the decomposition to remove.

### 4.2 Y1 direction — a NEW results subsection

This is new material and deserves its own subsection, clearly separated from magnitude:

> Return direction is a different question from return magnitude and is reported as one.
> Defining `negative_return = Y_h < 0` on identical folds and the same purge, logistic
> regression on the combined information set with K = 10 features achieves ROC-AUC 0.752 at
> the 10-session horizon (episode-clustered 95% CI [0.567, 0.896]; balanced accuracy 0.659;
> MCC 0.339; PR-AUC 0.771). This is the only result in the study that remains significant
> after a Holm family-wise correction across the eight direction comparisons
> (p_Holm = 0.032). Sensitivity is 0.476 against a specificity of 0.842: the model is more
> reliable at identifying events that will *not* be followed by a negative 10-session
> return than at flagging those that will.

Immediately follow it with the caveat, in the text, not a footnote:

> Classification success is not regression success. That the sign of the 10-session return
> is predictable above chance does not imply that its size is.

### 4.3 Y2 — unchanged

No edits. If the chapter states that Y2 is the strongest continuous result, that remains
correct and is now mechanically verified (`docs/Y2_FROZEN_VALIDATION_REPORT.md`,
`tests/test_y2_frozen.py`).

### 4.4 Y3 — replace the point-regression result with the survival result

Delete "Y3 pooled R2 = +0.155" as a headline. It was computed under a likelihood that
treats every censored row as an observed recovery. Keep it as a disclosed diagnostic:

> The previously reported Y3 pooled R2 of +0.155 is retained as a diagnostic of the
> distortion introduced by point-regression on censored data, not as a primary result.

New headline:

> Under right-censored survival analysis on genuine events only, a two-stage model —
> a drawdown-occurrence classifier followed by a Weibull AFT fit on drawdown cases — attains
> a Harrell C-index of 0.657 (episode-clustered 95% CI [0.522, 0.769]) against 0.477 for a
> training-fold Kaplan-Meier baseline. The interval excludes chance on a single comparison
> but does not survive Holm correction across the 15-model family (p_Holm = 0.165). The
> finding is therefore classified as suggestive rather than statistically supported.

Add the calibration table (section 3.2 of the final results) — it is the most practically
useful output the thesis has for Y3:

> The two-stage model's predicted recovery probabilities are well calibrated, with a
> maximum deviation of 4.5 percentage points from the Kaplan-Meier estimate across
> P(T <= 10), P(T <= 20), P(T <= 30), P(T <= 60) and P(T <= 90). The single-stage AFT
> drifts to +14.2 points at t = 90, over-predicting late recovery, because it must represent
> no-drawdown events (which "recover" at t = 0 by construction) inside one smooth duration
> distribution.

And state the negative category result:

> Of the two pre-specified recovery categories, `recovery <= 20 trading days` is not
> predictable (best AUC 0.612, CI [0.347, 0.854]); its 79% prevalence leaves little to
> discriminate. The existing slow-recovery classifier is unchanged (AUC 0.811, balanced
> accuracy 0.705).

### 4.5 Feature stability — new paragraph

> Across four folds, 35.6% of all selected features were selected in exactly one fold. No
> variable is described as important on the basis of a single fold's selection. The most
> consistently selected disaster variable is the DesInventar log affected population
> (`di_affected_log`, selection frequency 1.00 in the disaster-only set at h = 10); the most
> consistently selected market variable is the one-session lagged return
> (`lag_return_t-1`, selection frequency 1.00).

---

## Chapter 5 — Discussion and limitations

Add or strengthen:

1. **The negative result is a result.** Frame the Y1 magnitude finding as informative about
   CSE efficiency at the event horizon, not as a failed experiment.
2. **Direction vs magnitude.** The asymmetry (direction predictable at h = 10, magnitude
   not) is the discussion's most interesting hook: it is consistent with a market that
   reprices reliably in sign but with size dominated by idiosyncratic noise at N = 74.
3. **Censoring matters.** Explain concretely that 17 of 74 events are censored and that
   scoring them as observed inflates apparent Y3 performance.
4. **Multiplicity.** State that with 720 + 15 + 8 comparisons, one nominally significant
   result is expected by chance, which is why Holm is reported — and that the h = 10
   direction result survives it.
5. **Small-N limitations, unchanged and unresolved:** no lockbox holdout; 40 pooled test
   points; 31 observed recoveries; 35.6% single-fold feature selection; the Y3 K = 20 model
   carries 20 covariates against 31 events, which is the most likely reason its
   Holm-corrected p is 0.165.
6. **A pre-declared omission:** the direction analysis was declared on the combined
   information set only, so no incremental market-vs-combined claim can be made for
   direction. Adding that arm now would be a post-hoc extension and is deliberately not done.

---

## Claims to delete from any existing draft

* Any statement that the models predict the size of the post-disaster ASPI move.
* Any statement that disaster severity variables improve return prediction.
* Any statement that recovery duration can be predicted to a number of days.
* Any use of a positive pooled R2 as evidence of skill without its interval.
* Any description of a single-fold feature selection as a finding.
* Any presentation of the Y3 point-regression R2 as a primary result.

## Figures and tables to regenerate

| Item | Source |
|---|---|
| Y1 results table (16-row extract) | `docs/Y1_Y3_FINAL_RESULTS.md` section 2.2 |
| Y1 full grid (appendix) | `docs/thesis_materials/final_table_y1.csv` |
| Y1 direction table | `artifacts/y1_improve_direction.parquet` |
| Feature-stability table | `artifacts/y1_improve_stability.parquet` |
| Y3 survival table | `docs/thesis_materials/final_table_y3.csv` |
| Y3 calibration table | `artifacts/y3_improve_calibration.parquet` |
| Per-event recovery probabilities | `artifacts/y3_improve_oof.parquet` |
| Y2 (unchanged) | existing artifacts; do not regenerate |

Rebuild the tables with `python scripts/build_final_tables.py`.

## Reproduction

```
python scripts/freeze_baseline.py          # once, already done
python scripts/run_y1_improvement.py       # ~75 min
python scripts/run_y3_improvement.py       # ~1 min
python scripts/build_final_tables.py
pytest tests/ -q                           # must stay green, including the 20 Y2 freeze tests
```
