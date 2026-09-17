# Y1 / Y3 final results

Executed 2026-09-17 against the pre-declaration in
`docs/Y1_Y3_IMPROVEMENT_PREDECLARATION.md`, from baseline commit `1fbf6275`.
Every configuration listed in the pre-declaration was run, and every one of them is
reported here, in whichever direction it landed. Nothing was added after seeing a result.

**Headline, stated plainly:**

* **Y1 magnitude is not predictable at this sample size.** 0 of 240 configurations beat
  any of its three pre-declared baselines with a bootstrap CI excluding zero. Not one.
* **Y1 direction at the 10-session horizon is.** Logistic regression on the combined
  information set: ROC-AUC 0.752, episode-clustered 95% CI [0.567, 0.896], and it is the
  one result in this entire body of work that survives a Holm correction
  (p_holm = 0.032). This is a *different question* from magnitude and is reported as one.
* **Y3 recovery ranking is suggestive.** The two-stage (drawdown classifier + AFT) models
  at K=20 reach a C-index of 0.657 with a CI excluding 0.5 on a single test, but neither
  survives Holm across the 15-model family. Census-aware modelling clearly beats the
  Kaplan-Meier baseline's C-index of 0.477, but the evidence is **B**, not **A**.
* **Y2 is untouched.** See `docs/Y2_FROZEN_VALIDATION_REPORT.md`.

---

## 1. What was run

| | |
|---|---|
| Events | 74 (unchanged; threshold `total_affected >= 1000` unchanged) |
| Outer validation | chronological walk-forward, train=30, test=10, step=10, 4 folds |
| Pooled test points | 40 per configuration, identical events for every candidate and baseline |
| Y1 configurations | 4 horizons x 4 information sets x 3 capacities x 5 models = **240** |
| Y1 statistical comparisons | 240 x 3 baselines = **720** |
| Y3 models | 15 (13 survival + 2 baselines) |
| Seeds | `RANDOM_STATE = 42` throughout, unchanged |
| Bootstrap | 10,000 paired draws, clustered by 14-day disaster episode |

Artifacts: `y1_improve_{oof,metrics,verdicts,stability,direction}.parquet`,
`y3_improve_{oof,metrics,calibration,categories}.parquet`,
`docs/thesis_materials/final_table_y1.csv` (all 240 rows),
`docs/thesis_materials/final_table_y3.csv`.

## 2. Y1 — magnitude

### 2.1 Baselines (the numbers everything is measured against)

| Horizon | naive_zero RMSE | train_mean RMSE | market_only_expected RMSE |
|---|---|---|---|
| 5 | 3.3274 | 3.2622 | 3.3052 |
| 10 | 5.1824 | 5.1169 | 5.1367 |
| 15 | 6.9805 | 6.9117 | 6.8314 |
| 20 | 6.9742 | 6.9950 | 6.9833 |

The `market_only_expected` baseline is the Stage-A normal-market model of protocol 1.2:
a Ridge fit on the daily ASPI series using only sessions whose own h-day label had already
settled strictly before the event. It is available for 69 of 74 events (the first five
predate the 250-session minimum history) — and those five are all training-side rows, so
every one of the 40 pooled test points carries a genuine estimate.

### 2.2 Best configuration per horizon x information set

| horizon | info_set | K | model | n | RMSE | MAE | R2 | skill_vs_zero | skill_vs_train_mean | skill_vs_market_only | bootstrap_delta | CI_low | CI_high | Holm_p | sig. single | sig. adjusted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | combined | 20 | random_forest | 40 | 3.1084 | 2.2794 | +0.0621 | 0.0658 | 0.0471 | 0.0595 | +0.1968 | -0.2949 | 0.7190 | 1.0 | False | False |
| 5 | disaster_only | 10 | random_forest | 40 | 3.1926 | 2.3313 | +0.0107 | 0.0405 | 0.0213 | 0.0341 | +0.1127 | -0.5007 | 0.7435 | 1.0 | False | False |
| 5 | market_only | 10 | ridge | 40 | 3.2583 | 2.3231 | -0.0305 | 0.0208 | 0.0012 | 0.0142 | +0.0470 | -0.2111 | 0.4096 | 1.0 | False | False |
| 5 | normal+residual | 10 | random_forest | 40 | 3.2482 | 2.2932 | -0.0241 | 0.0238 | 0.0043 | 0.0173 | +0.0570 | -0.4316 | 0.4790 | 1.0 | False | False |
| 10 | combined | 10 | random_forest | 40 | 4.9931 | 3.9737 | +0.0216 | 0.0365 | 0.0242 | 0.0280 | +0.1436 | -0.6935 | 0.8766 | 1.0 | False | False |
| 10 | disaster_only | 10 | elastic_net | 40 | 4.8561 | 3.4852 | +0.0746 | 0.0630 | 0.0510 | 0.0546 | +0.2806 | -0.5602 | 1.0453 | 1.0 | False | False |
| 10 | market_only | 20 | random_forest | 40 | 5.0618 | 3.8576 | -0.0055 | 0.0233 | 0.0108 | 0.0146 | +0.0748 | -0.5629 | 0.6205 | 1.0 | False | False |
| 10 | normal+residual | 5 | elastic_net | 40 | 5.0240 | 3.6371 | +0.0095 | 0.0306 | 0.0182 | 0.0219 | +0.1126 | -0.5829 | 0.6684 | 1.0 | False | False |
| 15 | combined | 10 | elastic_net | 40 | 6.9234 | 4.8374 | -0.0547 | 0.0082 | -0.0017 | -0.0135 | -0.0920 | -1.3443 | 1.0491 | 1.0 | False | False |
| 15 | disaster_only | 5 | elastic_net | 40 | 7.0320 | 5.1868 | -0.0881 | -0.0074 | -0.0174 | -0.0294 | -0.2007 | -1.2711 | 0.8037 | 1.0 | False | False |
| 15 | market_only | 10 | elastic_net | 40 | 7.1048 | 5.1098 | -0.1107 | -0.0178 | -0.0279 | -0.0400 | -0.2735 | -1.1493 | 0.6800 | 1.0 | False | False |
| 15 | normal+residual | 5 | ridge | 40 | 7.2178 | 5.2900 | -0.1463 | -0.0340 | -0.0443 | -0.0566 | -0.3864 | -0.9755 | 0.2021 | 1.0 | False | False |
| 20 | combined | 10 | random_forest | 40 | 7.0560 | 5.3983 | -0.0739 | -0.0117 | -0.0087 | -0.0104 | -0.0727 | -1.4143 | 1.1860 | 1.0 | False | False |
| 20 | disaster_only | 5 | ridge | 40 | 7.3530 | 5.6529 | -0.1663 | -0.0543 | -0.0512 | -0.0529 | -0.3697 | -1.4105 | 0.6288 | 1.0 | False | False |
| 20 | market_only | 5 | random_forest | 40 | 6.6014 | 4.9054 | +0.0600 | 0.0535 | 0.0563 | 0.0547 | +0.3819 | -0.8686 | 1.5242 | 1.0 | False | False |
| 20 | normal+residual | 5 | ridge | 40 | 7.3739 | 5.4360 | -0.1729 | -0.0573 | -0.0542 | -0.0559 | -0.3906 | -1.0407 | 0.3207 | 1.0 | False | False |

`bootstrap_delta` / CI / Holm_p are against `market_only_expected`, the primary comparison.
All 720 comparisons (three baselines x 240 configurations) are in
`artifacts/y1_improve_verdicts.parquet`. Distribution of verdicts:

| Verdict | Count |
|---|---|
| A — statistically supported | **0** |
| B — suggestive but uncertain | 147 |
| C — unsupported | 573 |

**Every confidence interval in the Y1 magnitude grid contains zero.** The widest positive
point estimate (h=20, market-only, RF, +0.382 RMSE) carries a CI of [-0.869, +1.524]. At
40 pooled test points and event-clustered resampling, an effect this size is
indistinguishable from noise.

### 2.3 Does disaster information add anything beyond market state? (the primary question)

Comparing the best `combined` against the best `market_only` at each horizon:

| Horizon | market_only RMSE | combined RMSE | Difference | Reading |
|---|---|---|---|---|
| 5 | 3.2583 | 3.1084 | combined better by 0.150 | CI overlaps zero |
| 10 | 5.0618 | 4.9931 | combined better by 0.069 | CI overlaps zero |
| 15 | 7.1048 | 6.9234 | combined better by 0.181 | CI overlaps zero |
| 20 | 6.6014 | 7.0560 | market-only better by 0.455 | CI overlaps zero |

Direction is inconsistent and no interval excludes zero. **Answer: not demonstrated.**
Disaster-specific information does not measurably improve return-magnitude prediction
beyond the market's own pre-event state at this N.

### 2.4 The normal-market + disaster-residual architecture (A4)

The decomposition was implemented exactly as specified and is **not** an improvement: at
every horizon it lands between the market-only and combined models, and at h=15/20 it is
the worst of the four information sets. The hypothesis — that separating ordinary market
movement from disaster-associated movement would make the disaster component easier to
learn — is **not supported by this data**. The Stage-A model itself is only marginally
better than predicting zero (skill 0.007 at h=5), so there is very little "normal market"
signal for it to strip out, which is a coherent explanation of why the decomposition
neither helps nor hurts much.

This is a negative result and it is retained.

### 2.5 Feature capacity (A5/A6/A7)

No capacity dominates. Across the 16 (horizon, information set) cells, the best RMSE came
from K=5 six times, K=10 eight times, K=20 twice. The pre-declared compact capacities are
therefore at least as good as the frozen K=20, which is the useful finding: the study loses
nothing by reporting a 5- or 10-variable model, and gains interpretability.

### 2.6 Feature-selection stability

`artifacts/y1_improve_stability.parquet`. **35.6% of all selected features were selected in
exactly one of four folds** — a direct measure of how unstable variable selection is at this
sample size, and the reason no single-fold selection is described as a discovery anywhere in
this thesis.

The features that were stable (h=10, combined, K=10):

| Feature | selection_frequency | mean_rank | median_rank |
|---|---|---|---|
| lag_return_t-1 | 1.00 | 2.25 | 1.5 |
| di_affected_log | 0.75 | 3.67 | 2.0 |
| fx_vol_30 | 0.75 | 5.00 | 5.0 |
| log_return | 0.75 | 5.00 | 4.0 |
| lag_return_t-3 | 0.75 | 7.67 | 8.0 |

and (h=10, disaster-only, K=10): `di_affected_log` (1.00), `mag_area_km2`,
`total_deaths`, `di_records`, `hz_wind_max3d`, `disasters_trailing_365d`,
`di_districts_hit`, `hz_precip_anom` (all 0.75). The DesInventar affected-population
measure is the single most consistently selected disaster variable in the study.

### 2.7 Direction (protocol 1.6) — a different question, reported as one

`negative_return = Y_h < 0`, combined information set, K=10, same folds, same purge.

| Horizon | Model | n | prevalence | bal. acc. | ROC-AUC | PR-AUC | MCC | sens. | spec. | AUC CI | Holm p | Holm sig. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | logistic | 40 | 0.400 | 0.479 | 0.552 | 0.460 | -0.046 | 0.250 | 0.708 | [0.369, 0.736] | 0.611 | False |
| 5 | random_forest | 40 | 0.400 | 0.542 | 0.521 | 0.486 | 0.102 | 0.250 | 0.833 | [0.326, 0.710] | 0.611 | False |
| **10** | **logistic** | 40 | 0.525 | **0.659** | **0.752** | 0.771 | 0.339 | 0.476 | 0.842 | **[0.567, 0.896]** | **0.032** | **True** |
| 10 | random_forest | 40 | 0.525 | 0.617 | 0.702 | 0.714 | 0.306 | 0.286 | 0.947 | [0.517, 0.866] | 0.109 | False |
| 15 | logistic | 40 | 0.500 | 0.625 | 0.698 | 0.644 | 0.258 | 0.500 | 0.750 | [0.513, 0.865] | 0.123 | False |
| 15 | random_forest | 40 | 0.500 | 0.625 | 0.610 | 0.628 | 0.267 | 0.450 | 0.800 | [0.424, 0.791] | 0.470 | False |
| 20 | logistic | 40 | 0.525 | 0.654 | 0.662 | 0.654 | 0.311 | 0.571 | 0.737 | [0.480, 0.831] | 0.245 | False |
| 20 | random_forest | 40 | 0.525 | 0.576 | 0.576 | 0.638 | 0.173 | 0.429 | 0.737 | [0.396, 0.763] | 0.611 | False |

**This is the strongest genuinely new result in the improvement work.** It is also the only
one anywhere in this study that survives a family-wise correction. Note carefully what it
is and is not: the sign of the 10-session cumulative return is predictable above chance;
its size is not. Sensitivity is only 0.476 — the model is much better at identifying events
that will *not* produce a negative 10-day return (specificity 0.842) than at flagging the
ones that will. A classification success is not a regression success and is not reported
as one.

## 3. Y3 — recovery duration

74 events: 57 observed recoveries, 17 censored (9 by a competing later disaster, 8 at the
90-day cap), 51 with a genuine post-event drawdown. Pooled out-of-fold: 40 test rows, 31
observed, 9 censored.

### 3.1 Final Y3 table

| Model | n | recovered | censored | C-index | C-index CI | IBS | Median abs err (uncens.) | MAE (uncens.) | max abs calib. gap | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| two_stage_weibull_k20 | 40 | 31 | 9 | **0.657** | [0.522, 0.769] | 0.194 | 4.0 | 15.48 | 0.045 | 0.165 | **A (single test)** |
| two_stage_lognormal_k20 | 40 | 31 | 9 | **0.643** | [0.503, 0.758] | 0.196 | 3.0 | 14.77 | 0.045 | 0.336 | **A (single test)** |
| aft_lognormal_k20 | 40 | 31 | 9 | 0.598 | [0.453, 0.727] | 0.165 | 4.0 | 9.87 | 0.142 | 1.000 | B |
| two_stage_weibull_k5 | 40 | 31 | 9 | 0.589 | [0.456, 0.717] | 0.167 | 4.0 | 8.39 | 0.150 | 1.000 | B |
| aft_weibull_k20 | 40 | 31 | 9 | 0.581 | [0.427, 0.711] | 0.154 | 5.0 | 9.48 | 0.163 | 1.000 | B |
| aft_lognormal_k5 | 40 | 31 | 9 | 0.574 | [0.427, 0.721] | 0.163 | 3.0 | 5.77 | 0.153 | 1.000 | B |
| two_stage_lognormal_k5 | 40 | 31 | 9 | 0.574 | [0.437, 0.705] | 0.162 | 5.0 | 6.03 | 0.146 | 1.000 | B |
| two_stage_lognormal_k10 | 40 | 31 | 9 | 0.569 | [0.419, 0.713] | **0.147** | 3.0 | 5.84 | 0.156 | 1.000 | B |
| cox_k5 | 40 | 31 | 9 | 0.569 | [0.428, 0.710] | 0.169 | 3.0 | 7.19 | 0.097 | 1.000 | B |
| two_stage_weibull_k10 | 40 | 31 | 9 | 0.567 | [0.421, 0.706] | 0.149 | 4.0 | 7.65 | 0.149 | 1.000 | B |
| aft_lognormal_k10 | 40 | 31 | 9 | 0.563 | [0.397, 0.718] | 0.150 | **2.0** | 5.35 | 0.163 | 1.000 | B |
| aft_weibull_k5 | 40 | 31 | 9 | 0.560 | [0.421, 0.707] | 0.181 | 5.0 | 9.90 | 0.152 | 1.000 | B |
| aft_weibull_k10 | 40 | 31 | 9 | 0.539 | [0.369, 0.693] | 0.157 | 5.0 | 7.58 | 0.162 | 1.000 | B |
| km_train_baseline | 40 | 31 | 9 | 0.477 | [0.353, 0.615] | 0.158 | 3.0 | 4.87 | 0.049 | 1.000 | C |
| train_median_baseline | 40 | 31 | 9 | 0.477 | [0.353, 0.615] | 0.203 | 3.0 | 4.87 | 0.350 | 1.000 | C |

Read the last two columns together with the C-index, not instead of it. The Kaplan-Meier
baseline has the *best* uncensored MAE (4.87) precisely because a constant near-zero
prediction is hard to beat on a sample whose median recovery is short — which is exactly why
point error is a secondary metric here and rank concordance is the primary one.

### 3.2 Calibration of the recovery probabilities

`two_stage_weibull_k20`, predicted mean P(T <= t) vs the Kaplan-Meier observed value:

| t (trading days) | Predicted | Observed (KM) | Gap |
|---|---|---|---|
| 10 | 0.655 | 0.650 | +0.005 |
| 20 | 0.713 | 0.758 | -0.045 |
| 30 | 0.745 | 0.788 | -0.043 |
| 60 | 0.802 | 0.788 | +0.014 |
| 90 | 0.823 | 0.788 | +0.035 |

Well calibrated (worst gap 4.5 percentage points). By contrast the plain
`aft_lognormal_k20` drifts to +14.2 points at t=90 — it over-promises late recovery,
because without the drawdown stage it has to represent the "no drawdown, recovers at 0"
events inside a single smooth duration distribution. **This is the clearest evidence for
the two-stage architecture: it fixes calibration, not just ranking.**

Per-event output is no longer a single number. `artifacts/y3_improve_oof.parquet` carries,
for every test event, the predicted median and P(T <= 10/20/30/60/90).

### 3.3 Recovery categories (protocol 2.7 — exactly two, pre-specified)

`recovery <= 20 trading days`, scored from each survival model's own P(T <= 20), with rows
censored before day 20 excluded (status unknowable): n=38, prevalence 0.789. Best AUC 0.612
(`aft_weibull_k20`), CI [0.347, 0.854]. **No model beats chance.** The 79% prevalence
leaves very little to discriminate.

Category 2 is the existing, unchanged `C3b_slow_recovery` classifier, which stands at
AUC 0.811, balanced accuracy 0.705, CI [0.636, 0.947], `beats_baseline = True` in
`artifacts/classification_summary.parquet`. It was not re-run and is not re-tuned.

## 4. Ablation register — what each pre-declared ablation showed

| ID | Ablation | Result |
|---|---|---|
| A1 | Y1 market-only | Best RMSE 3.258 (h=5) / 6.601 (h=20). Beats nothing. Best-of-all at h=20. |
| A2 | Y1 disaster-only | Best RMSE 3.193 (h=5) / 4.856 (h=10, best single Y1 R2 at +0.075). CI overlaps zero. |
| A3 | Y1 combined | Best RMSE 3.108 (h=5), best at h=5 and h=15. CI overlaps zero. |
| A4 | Y1 normal-market + disaster-residual | **No improvement.** Mid-pack at h=5/10, worst at h=15/20. Negative result, retained. |
| A5 | Y1 K=5 | Best in 6 of 16 cells. Competitive with K=20. |
| A6 | Y1 K=10 | Best in 8 of 16 cells. |
| A7 | Y1 K=20 (robustness) | Best in 2 of 16 cells. No advantage from the extra capacity. |
| A8 | Y1 h=5 (principal) | Best R2 +0.062. 0 of 60 configurations significant. |
| A9 | Y1 h=10 | Best R2 +0.075 (highest in the grid). 0 of 60 significant. Direction IS significant here. |
| A10 | Y1 h=15 | Best R2 -0.055. Every configuration worse than the zero baseline on R2. |
| A11 | Y1 h=20 | Best R2 +0.060 (market-only). 0 of 60 significant. |
| B1 | Y3 point regression (existing) | Pooled R2 +0.155 (ensemble), but RMSE computed as if censored rows were observed. Retained as a diagnostic; superseded as the primary Y3 result. |
| B2 | Y3 AFT, real rows only | C-index 0.539-0.598. B. Beats the KM baseline (0.477) at every capacity. |
| B3 | Y3 two-stage drawdown + AFT | **Best**: C-index 0.657, CI excludes 0.5, best calibration (max gap 0.045). A on a single test, B after Holm. |
| B4 | Weibull vs Log-normal AFT | Near-identical ranking (0.657 vs 0.643 two-stage K=20). Log-normal has better IBS at K=10; Weibull better C-index at K=20. No decisive winner. |
| B5 | Y3 compact features | K=5 and K=10 give C-index 0.54-0.59 vs 0.58-0.66 at K=20. Here, unlike Y1, the larger capacity does help ranking. |
| B6 | SMOGN on/off (existing) | On Y3 point regression SMOGN helps (RMSE 21.55 vs 22.36; R2 +0.081 vs +0.011). Irrelevant to the primary survival result, which uses genuine rows only by design. |

No experiment outside this register was run. No post-hoc exploratory analyses were added.

## 5. Decision-rule classification (protocol Part 8)

| Finding | Class |
|---|---|
| Y1 return magnitude, any horizon, any information set, any model | **C — unsupported** |
| Disaster information adding value beyond market state (Y1 magnitude) | **C — unsupported** |
| Normal-market + disaster-residual decomposition (Y1) | **C — unsupported** |
| Y1 direction at h=10 (logistic, combined, K=10) | **A — statistically supported**, and the only finding here surviving Holm |
| Y1 direction at h=15 and h=20 | **B — suggestive but uncertain** |
| Y1 direction at h=5 | **C — unsupported** |
| Y3 recovery ranking, two-stage AFT, K=20 | **A on a single test / B after multiplicity adjustment** — reported as **B** |
| Y3 recovery-probability calibration (two-stage) | **B** — well calibrated, but calibration is descriptive, not a test against a null |
| Y3 `recovery <= 20 days` classification | **C — unsupported** |
| Y3 `C3b_slow_recovery` (existing, unchanged) | **A** on its own single-comparison criterion, as already reported |
| Y2 abnormal volume (frozen) | **A — statistically supported**, unchanged |

No **B** has been promoted to **A** anywhere in this document.

## 6. Thesis interpretation (the nine required questions)

**1. Can exact ASPI magnitude be predicted?**
No. Across 240 configurations and 720 statistical comparisons, not one confidence interval
excluded zero. The best pooled R2 in the entire grid is +0.075, on 40 test points, with a
delta-RMSE CI of [-0.560, +1.045]. The honest statement is that this study finds **no
evidence** that disaster-event ASPI return magnitude is predictable at N=74.

**2. At which pre-specified horizon is evidence strongest?**
For magnitude, nowhere: h=10 has the best point estimate (R2 = +0.075) and h=15 is
uniformly the worst, but none is distinguishable from its baseline. For direction, **h=10**
is unambiguously strongest and is the only horizon whose evidence survives correction.

**3. Does disaster-specific information improve prediction beyond pre-event market state?**
For magnitude: not demonstrated. Combined beats market-only at three of four horizons and
loses at the fourth, all with intervals spanning zero. For direction, the combined set is
what produces the h=10 result, but the study did not run a market-only direction arm, so
no incremental claim can be made there either — that is a limitation, stated as one, not
an omission to be filled in after the fact (protocol Part 9 forbids adding it now).

**4. Can exact recovery duration be predicted?**
No. Median absolute error among genuinely observed recoveries is 2-5 trading days for the
survival models, but the Kaplan-Meier baseline achieves a comparable 3 days by predicting
essentially the same short duration for everyone. Exact-day recovery estimation is not
demonstrated.

**5. Can recovery speed or recovery probabilities be predicted?**
Partly, and this is the useful Y3 finding. The two-stage model orders events by recovery
speed better than chance (C-index 0.657, CI [0.522, 0.769]) and produces
**well-calibrated** recovery probabilities (worst gap 4.5 percentage points across five
horizons). Both statements weaken to "suggestive" after Holm correction. The
already-existing `C3b_slow_recovery` classifier remains the strongest recovery-side result.

**6. Which findings remain significant after uncertainty and multiplicity correction?**
Exactly one from this work: **Y1 direction at h=10, logistic, ROC-AUC 0.752, Holm
p = 0.032**. Plus, from the frozen pipeline and unchanged: Y2 abnormal volume, SVR vs
naive_zero (Holm-significant). Everything else is single-comparison evidence at best.

**7. What limitations remain due to small N?**
74 events, 40 pooled out-of-fold test points, four folds. 35.6% of selected features appear
in exactly one fold. The narrowest delta-RMSE CI in the Y1 grid is still ±0.5 RMSE units on
a target whose standard deviation is ~3.3. There is no lockbox holdout (the sample cannot
spare one; this is disclosed in `docs/FINAL_ANALYSIS_PROTOCOL.md` section 8 and is not
resolved by this work). Y3 has 31 observed recoveries in the pooled test set, which is thin
for a survival model with 20 covariates — the K=20 result in particular should be read with
that in mind, and it is the reason its Holm-corrected p is 0.165 rather than significant.

**8. Which statements are safe to defend in a viva?**

* "Abnormal trading volume following a qualifying disaster is predictable out of sample;
  the effect survives an episode-clustered bootstrap and, for SVR against a zero baseline,
  a family-wise correction." (Y2, frozen, unchanged.)
* "The *direction* of the 10-session cumulative ASPI return is predictable above chance
  (AUC 0.752, CI [0.567, 0.896], Holm p = 0.032), while its *magnitude* is not."
* "Recovery duration should be modelled as right-censored survival data; doing so with a
  two-stage drawdown-then-AFT architecture ranks events better than a Kaplan-Meier
  baseline (C-index 0.657 vs 0.477) and produces calibrated recovery probabilities — but
  the ranking result does not survive multiplicity correction and is reported as
  suggestive."
* "A compact 5- or 10-variable model performs as well as a 20-variable one for Y1, so the
  thesis reports the compact model."
* "The study found no evidence for return-magnitude predictability, across a
  pre-declared grid of 240 configurations, and reports that as its finding."

**9. Which claims must NOT be made?**

* That the model predicts how far the ASPI will move after a disaster. It does not.
* That disaster severity information improves return prediction beyond market state. Not
  demonstrated.
* That the normal-market/disaster-residual decomposition improved anything. It did not.
* That recovery duration can be predicted to a number of days. It cannot.
* That the Y3 C-index result is statistically established. It is suggestive; it fails Holm.
* That a positive pooled R2 (e.g. +0.075 at h=10) is evidence of skill. With a CI of
  [-0.560, +1.045] it is not.
* That any feature is "important" on the strength of one fold's selection. 35.6% of
  selections occur in exactly one fold.
* That the h=10 direction result transfers to the other horizons, or to magnitude.

## 7. Stop condition

The pre-declared grid has been executed in full and the pipeline is frozen. No further
horizon, algorithm, threshold, target, feature count, observation removal, or variant
search follows from these results.
