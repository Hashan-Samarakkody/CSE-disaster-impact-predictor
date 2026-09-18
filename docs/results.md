# Results

Every number the executed repository produced, for all three research targets, in the
direction it landed. Disappointing results are reported here, not removed.

The machine readable versions live in `artifacts/tables/` and in
`docs/thesis_materials/final_table_aspi.csv` and
`docs/thesis_materials/final_table_recovery.csv`.

## Contents

1. Headline results for the three targets
2. The return and recovery improvement grid
3. Final tables
4. Full metric audit for every model and every target


---

# Part 1 and 2. The return and recovery improvement grid

## Y1 / Y3 final results

Executed 2026-09-17 against the pre-declaration in
`docs/audit.md Part 2`, from baseline commit `1fbf6275`.
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
* **Y2 is untouched.** See `docs/audit.md Part 4`.

---

### 1. What was run

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
`docs/thesis_materials/final_table_aspi.csv` (all 240 rows),
`docs/thesis_materials/final_table_recovery.csv`.

### 2. Y1,  magnitude

#### 2.1 Baselines (the numbers everything is measured against)

| Horizon | naive_zero RMSE | train_mean RMSE | market_only_expected RMSE |
|---|---|---|---|
| 5 | 3.3274 | 3.2622 | 3.3052 |
| 10 | 5.1824 | 5.1169 | 5.1367 |
| 15 | 6.9805 | 6.9117 | 6.8314 |
| 20 | 6.9742 | 6.9950 | 6.9833 |

The `market_only_expected` baseline is the Stage-A normal-market model of protocol 1.2:
a Ridge fit on the daily ASPI series using only sessions whose own h-day label had already
settled strictly before the event. It is available for 69 of 74 events (the first five
predate the 250-session minimum history),  and those five are all training-side rows, so
every one of the 40 pooled test points carries a genuine estimate.

#### 2.2 Best configuration per horizon x information set

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
`artifacts/aspi_grid_verdicts.parquet`. Distribution of verdicts:

| Verdict | Count |
|---|---|
| A,  statistically supported | **0** |
| B,  suggestive but uncertain | 147 |
| C,  unsupported | 573 |

**Every confidence interval in the Y1 magnitude grid contains zero.** The widest positive
point estimate (h=20, market-only, RF, +0.382 RMSE) carries a CI of [-0.869, +1.524]. At
40 pooled test points and event-clustered resampling, an effect this size is
indistinguishable from noise.

#### 2.3 Does disaster information add anything beyond market state? (the primary question)

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

#### 2.4 The normal-market + disaster-residual architecture (A4)

The decomposition was implemented exactly as specified and is **not** an improvement: at
every horizon it lands between the market-only and combined models, and at h=15/20 it is
the worst of the four information sets. The hypothesis,  that separating ordinary market
movement from disaster-associated movement would make the disaster component easier to
learn,  is **not supported by this data**. The Stage-A model itself is only marginally
better than predicting zero (skill 0.007 at h=5), so there is very little "normal market"
signal for it to strip out, which is a coherent explanation of why the decomposition
neither helps nor hurts much.

This is a negative result and it is retained.

#### 2.5 Feature capacity (A5/A6/A7)

No capacity dominates. Across the 16 (horizon, information set) cells, the best RMSE came
from K=5 six times, K=10 eight times, K=20 twice. The pre-declared compact capacities are
therefore at least as good as the frozen K=20, which is the useful finding: the study loses
nothing by reporting a 5- or 10-variable model, and gains interpretability.

#### 2.6 Feature-selection stability

`artifacts/aspi_grid_feature_stability.parquet`. **35.6% of all selected features were selected in
exactly one of four folds**,  a direct measure of how unstable variable selection is at this
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

#### 2.7 Direction (protocol 1.6),  a different question, reported as one

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
its size is not. Sensitivity is only 0.476,  the model is much better at identifying events
that will *not* produce a negative 10-day return (specificity 0.842) than at flagging the
ones that will. A classification success is not a regression success and is not reported
as one.

### 3. Y3,  recovery duration

74 events: 57 observed recoveries, 17 censored (9 by a competing later disaster, 8 at the
90-day cap), 51 with a genuine post-event drawdown. Pooled out-of-fold: 40 test rows, 31
observed, 9 censored.

#### 3.1 Final Y3 table

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
prediction is hard to beat on a sample whose median recovery is short,  which is exactly why
point error is a secondary metric here and rank concordance is the primary one.

#### 3.2 Calibration of the recovery probabilities

`two_stage_weibull_k20`, predicted mean P(T <= t) vs the Kaplan-Meier observed value:

| t (trading days) | Predicted | Observed (KM) | Gap |
|---|---|---|---|
| 10 | 0.655 | 0.650 | +0.005 |
| 20 | 0.713 | 0.758 | -0.045 |
| 30 | 0.745 | 0.788 | -0.043 |
| 60 | 0.802 | 0.788 | +0.014 |
| 90 | 0.823 | 0.788 | +0.035 |

Well calibrated (worst gap 4.5 percentage points). By contrast the plain
`aft_lognormal_k20` drifts to +14.2 points at t=90,  it over-promises late recovery,
because without the drawdown stage it has to represent the "no drawdown, recovers at 0"
events inside a single smooth duration distribution. **This is the clearest evidence for
the two-stage architecture: it fixes calibration, not just ranking.**

Per-event output is no longer a single number. `artifacts/recovery_grid_predictions.parquet` carries,
for every test event, the predicted median and P(T <= 10/20/30/60/90).

#### 3.3 Recovery categories (protocol 2.7,  exactly two, pre-specified)

`recovery <= 20 trading days`, scored from each survival model's own P(T <= 20), with rows
censored before day 20 excluded (status unknowable): n=38, prevalence 0.789. Best AUC 0.612
(`aft_weibull_k20`), CI [0.347, 0.854]. **No model beats chance.** The 79% prevalence
leaves very little to discriminate.

Category 2 is the existing, unchanged `C3b_slow_recovery` classifier, which stands at
AUC 0.811, balanced accuracy 0.705, CI [0.636, 0.947], `beats_baseline = True` in
`artifacts/classification_summary.parquet`. It was not re-run and is not re-tuned.

### 4. Ablation register,  what each pre-declared ablation showed

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

### 5. Decision-rule classification (protocol Part 8)

| Finding | Class |
|---|---|
| Y1 return magnitude, any horizon, any information set, any model | **C,  unsupported** |
| Disaster information adding value beyond market state (Y1 magnitude) | **C,  unsupported** |
| Normal-market + disaster-residual decomposition (Y1) | **C,  unsupported** |
| Y1 direction at h=10 (logistic, combined, K=10) | **A,  statistically supported**, and the only finding here surviving Holm |
| Y1 direction at h=15 and h=20 | **B,  suggestive but uncertain** |
| Y1 direction at h=5 | **C,  unsupported** |
| Y3 recovery ranking, two-stage AFT, K=20 | **A on a single test / B after multiplicity adjustment**,  reported as **B** |
| Y3 recovery-probability calibration (two-stage) | **B**,  well calibrated, but calibration is descriptive, not a test against a null |
| Y3 `recovery <= 20 days` classification | **C,  unsupported** |
| Y3 `C3b_slow_recovery` (existing, unchanged) | **A** on its own single-comparison criterion, as already reported |
| Y2 abnormal volume (frozen) | **A,  statistically supported**, unchanged |

No **B** has been promoted to **A** anywhere in this document.

### 6. Thesis interpretation (the nine required questions)

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
no incremental claim can be made there either,  that is a limitation, stated as one, not
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
spare one; this is disclosed in `docs/audit.md Part 1` section 8 and is not
resolved by this work). Y3 has 31 observed recoveries in the pooled test set, which is thin
for a survival model with 20 covariates,  the K=20 result in particular should be read with
that in mind, and it is the reason its Holm-corrected p is 0.165 rather than significant.

**8. Which statements are safe to defend in a viva?**

* "Abnormal trading volume following a qualifying disaster is predictable out of sample;
  the effect survives an episode-clustered bootstrap and, for SVR against a zero baseline,
  a family-wise correction." (Y2, frozen, unchanged.)
* "The *direction* of the 10-session cumulative ASPI return is predictable above chance
  (AUC 0.752, CI [0.567, 0.896], Holm p = 0.032), while its *magnitude* is not."
* "Recovery duration should be modelled as right-censored survival data; doing so with a
  two-stage drawdown-then-AFT architecture ranks events better than a Kaplan-Meier
  baseline (C-index 0.657 vs 0.477) and produces calibrated recovery probabilities,  but
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

### 7. Stop condition

The pre-declared grid has been executed in full and the pipeline is frozen. No further
horizon, algorithm, threshold, target, feature count, observation removal, or variant
search follows from these results.

---

# Part 3. Final tables

### Y1 final table (best configuration per horizon x information set)

| horizon | info_set | k | model | n | RMSE | MAE | R2 | skill_vs_zero | skill_vs_train_mean | skill_vs_market_only | bootstrap_delta | CI_low | CI_high | Holm_p | significant_single_test | significant_after_adjustment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | combined | 20 | random_forest | 40 | 3.1084 | 2.2794 | 0.0621 | 0.0658 | 0.0471 | 0.0595 | 0.1968 | -0.2949 | 0.719 | 1.0 | False | False |
| 5 | disaster_only | 10 | random_forest | 40 | 3.1926 | 2.3313 | 0.0107 | 0.0405 | 0.0213 | 0.0341 | 0.1127 | -0.5007 | 0.7435 | 1.0 | False | False |
| 5 | market_only | 10 | ridge | 40 | 3.2583 | 2.3231 | -0.0305 | 0.0208 | 0.0012 | 0.0142 | 0.047 | -0.2111 | 0.4096 | 1.0 | False | False |
| 5 | normal_plus_residual | 10 | random_forest | 40 | 3.2482 | 2.2932 | -0.0241 | 0.0238 | 0.0043 | 0.0173 | 0.057 | -0.4316 | 0.479 | 1.0 | False | False |
| 10 | combined | 10 | random_forest | 40 | 4.9931 | 3.9737 | 0.0216 | 0.0365 | 0.0242 | 0.028 | 0.1436 | -0.6935 | 0.8766 | 1.0 | False | False |
| 10 | disaster_only | 10 | elastic_net | 40 | 4.8561 | 3.4852 | 0.0746 | 0.063 | 0.051 | 0.0546 | 0.2806 | -0.5602 | 1.0453 | 1.0 | False | False |
| 10 | market_only | 20 | random_forest | 40 | 5.0618 | 3.8576 | -0.0055 | 0.0233 | 0.0108 | 0.0146 | 0.0748 | -0.5629 | 0.6205 | 1.0 | False | False |
| 10 | normal_plus_residual | 5 | elastic_net | 40 | 5.024 | 3.6371 | 0.0095 | 0.0306 | 0.0182 | 0.0219 | 0.1126 | -0.5829 | 0.6684 | 1.0 | False | False |
| 15 | combined | 10 | elastic_net | 40 | 6.9234 | 4.8374 | -0.0547 | 0.0082 | -0.0017 | -0.0135 | -0.092 | -1.3443 | 1.0491 | 1.0 | False | False |
| 15 | disaster_only | 5 | elastic_net | 40 | 7.032 | 5.1868 | -0.0881 | -0.0074 | -0.0174 | -0.0294 | -0.2007 | -1.2711 | 0.8037 | 1.0 | False | False |
| 15 | market_only | 10 | elastic_net | 40 | 7.1048 | 5.1098 | -0.1107 | -0.0178 | -0.0279 | -0.04 | -0.2735 | -1.1493 | 0.68 | 1.0 | False | False |
| 15 | normal_plus_residual | 5 | ridge | 40 | 7.2178 | 5.29 | -0.1463 | -0.034 | -0.0443 | -0.0566 | -0.3864 | -0.9755 | 0.2021 | 1.0 | False | False |
| 20 | combined | 10 | random_forest | 40 | 7.056 | 5.3983 | -0.0739 | -0.0117 | -0.0087 | -0.0104 | -0.0727 | -1.4143 | 1.186 | 1.0 | False | False |
| 20 | disaster_only | 5 | ridge | 40 | 7.353 | 5.6529 | -0.1663 | -0.0543 | -0.0512 | -0.0529 | -0.3697 | -1.4105 | 0.6288 | 1.0 | False | False |
| 20 | market_only | 5 | random_forest | 40 | 6.6014 | 4.9054 | 0.06 | 0.0535 | 0.0563 | 0.0547 | 0.3819 | -0.8686 | 1.5242 | 1.0 | False | False |
| 20 | normal_plus_residual | 5 | ridge | 40 | 7.3739 | 5.436 | -0.1729 | -0.0573 | -0.0542 | -0.0559 | -0.3906 | -1.0407 | 0.3207 | 1.0 | False | False |


Full 240-row grid: `docs/thesis_materials/final_table_aspi.csv`.


### Y1 direction analysis (secondary, a different question)

| horizon | model | n | n_negative | prevalence | balanced_accuracy | roc_auc | pr_auc | mcc | sensitivity | specificity | auc_ci_low | auc_ci_high | p_one_sided | auc_ci_excludes_chance | p_holm | holm_significant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | logistic | 40 | 16 | 0.4 | 0.4792 | 0.5521 | 0.4596 | -0.0457 | 0.25 | 0.7083 | 0.369 | 0.736 | 0.2905 | False | 0.6105 | False |
| 5 | random_forest | 40 | 16 | 0.4 | 0.5417 | 0.5208 | 0.486 | 0.1021 | 0.25 | 0.8333 | 0.326 | 0.7102 | 0.4115 | False | 0.6105 | False |
| 10 | logistic | 40 | 21 | 0.525 | 0.6591 | 0.7519 | 0.7709 | 0.3394 | 0.4762 | 0.8421 | 0.5666 | 0.8961 | 0.004 | True | 0.032 | True |
| 10 | random_forest | 40 | 21 | 0.525 | 0.6165 | 0.7018 | 0.7141 | 0.3063 | 0.2857 | 0.9474 | 0.5167 | 0.8658 | 0.0155 | True | 0.1085 | False |
| 15 | logistic | 40 | 20 | 0.5 | 0.625 | 0.6975 | 0.6437 | 0.2582 | 0.5 | 0.75 | 0.5126 | 0.8645 | 0.0205 | True | 0.123 | False |
| 15 | random_forest | 40 | 20 | 0.5 | 0.625 | 0.61 | 0.628 | 0.2669 | 0.45 | 0.8 | 0.4238 | 0.7905 | 0.1175 | False | 0.47 | False |
| 20 | logistic | 40 | 21 | 0.525 | 0.6541 | 0.6617 | 0.6542 | 0.3114 | 0.5714 | 0.7368 | 0.48 | 0.8307 | 0.049 | False | 0.245 | False |
| 20 | random_forest | 40 | 21 | 0.525 | 0.5827 | 0.5764 | 0.6384 | 0.1732 | 0.4286 | 0.7368 | 0.3959 | 0.7626 | 0.2035 | False | 0.6105 | False |


### Y3 final table

| model | n | n_recovered | n_censored | C_index | C_index_CI | Integrated_Brier_Score | Median_abs_error_uncensored | mae_uncensored | rmse_uncensored | max_abs_calibration_gap | p_holm | holm_significant | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| two_stage_weibull_k20 | 40 | 31 | 9 | 0.6571 | [0.522, 0.769] | 0.1943 | 4.0 | 15.4839 | 30.5962 | 0.0447 | 0.165 | False | A - statistically supported |
| two_stage_lognormal_k20 | 40 | 31 | 9 | 0.6432 | [0.503, 0.758] | 0.1961 | 3.0 | 14.7742 | 28.4752 | 0.0454 | 0.336 | False | A - statistically supported |
| aft_lognormal_k20 | 40 | 31 | 9 | 0.5981 | [0.453, 0.727] | 0.1647 | 4.0 | 9.871 | 18.2439 | 0.1419 | 1.0 | False | B - suggestive but uncertain |
| two_stage_weibull_k5 | 40 | 31 | 9 | 0.5885 | [0.456, 0.717] | 0.1668 | 4.0 | 8.3871 | 17.1408 | 0.1496 | 1.0 | False | B - suggestive but uncertain |
| aft_weibull_k20 | 40 | 31 | 9 | 0.5807 | [0.427, 0.711] | 0.1542 | 5.0 | 9.4839 | 17.4947 | 0.163 | 1.0 | False | B - suggestive but uncertain |
| aft_lognormal_k5 | 40 | 31 | 9 | 0.5738 | [0.427, 0.721] | 0.163 | 3.0 | 5.7742 | 8.0543 | 0.1534 | 1.0 | False | B - suggestive but uncertain |
| two_stage_lognormal_k5 | 40 | 31 | 9 | 0.5738 | [0.437, 0.705] | 0.162 | 5.0 | 6.0323 | 9.2055 | 0.1462 | 1.0 | False | B - suggestive but uncertain |
| two_stage_lognormal_k10 | 40 | 31 | 9 | 0.5686 | [0.419, 0.713] | 0.1471 | 3.0 | 5.8387 | 9.4169 | 0.1563 | 1.0 | False | B - suggestive but uncertain |
| cox_k5 | 40 | 31 | 9 | 0.5686 | [0.428, 0.710] | 0.1688 | 3.0 | 7.1935 | 15.7429 | 0.0969 | 1.0 | False | B - suggestive but uncertain |
| two_stage_weibull_k10 | 40 | 31 | 9 | 0.5668 | [0.421, 0.706] | 0.1485 | 4.0 | 7.6452 | 12.9677 | 0.1491 | 1.0 | False | B - suggestive but uncertain |
| aft_lognormal_k10 | 40 | 31 | 9 | 0.5634 | [0.397, 0.718] | 0.15 | 2.0 | 5.3548 | 7.7168 | 0.1634 | 1.0 | False | B - suggestive but uncertain |
| aft_weibull_k5 | 40 | 31 | 9 | 0.5599 | [0.421, 0.707] | 0.1806 | 5.0 | 9.9032 | 15.5055 | 0.1522 | 1.0 | False | B - suggestive but uncertain |
| aft_weibull_k10 | 40 | 31 | 9 | 0.5391 | [0.369, 0.693] | 0.1573 | 5.0 | 7.5806 | 11.0789 | 0.1617 | 1.0 | False | B - suggestive but uncertain |
| km_train_baseline | 40 | 31 | 9 | 0.4766 | [0.353, 0.615] | 0.1578 | 3.0 | 4.871 | 6.1879 | 0.0485 | 1.0 | False | C - unsupported |
| train_median_baseline | 40 | 31 | 9 | 0.4766 | [0.353, 0.615] | 0.2033 | 3.0 | 4.871 | 6.1879 | 0.35 | 1.0 | False | C - unsupported |


### Y3 recovery category `recovery <= 20 trading days`

| category | model | n | n_positive | prevalence | roc_auc | auc_ci_low | auc_ci_high | balanced_accuracy | beats_chance |
|---|---|---|---|---|---|---|---|---|---|
| recovery_le_20d | aft_weibull_k20 | 38 | 30 | 0.7895 | 0.6125 | 0.3472 | 0.8544 | 0.6042 | False |
| recovery_le_20d | aft_weibull_k10 | 38 | 30 | 0.7895 | 0.5917 | 0.3257 | 0.8519 | 0.6375 | False |
| recovery_le_20d | two_stage_weibull_k10 | 38 | 30 | 0.7895 | 0.5917 | 0.3179 | 0.8526 | 0.6208 | False |
| recovery_le_20d | aft_lognormal_k20 | 38 | 30 | 0.7895 | 0.5875 | 0.3279 | 0.8249 | 0.6208 | False |
| recovery_le_20d | aft_lognormal_k10 | 38 | 30 | 0.7895 | 0.5833 | 0.3073 | 0.8492 | 0.5625 | False |
| recovery_le_20d | two_stage_lognormal_k10 | 38 | 30 | 0.7895 | 0.5792 | 0.3118 | 0.8386 | 0.6083 | False |
| recovery_le_20d | two_stage_weibull_k20 | 38 | 30 | 0.7895 | 0.5458 | 0.2719 | 0.8243 | 0.5708 | False |
| recovery_le_20d | two_stage_weibull_k5 | 38 | 30 | 0.7895 | 0.5208 | 0.2857 | 0.7517 | 0.5125 | False |
| recovery_le_20d | km_train_baseline | 38 | 30 | 0.7895 | 0.5125 | 0.2893 | 0.7235 | 0.5 | False |
| recovery_le_20d | train_median_baseline | 38 | 30 | 0.7895 | 0.5 | 0.5 | 0.5 | 0.5 | False |
| recovery_le_20d | cox_k5 | 38 | 30 | 0.7895 | 0.5 | 0.2796 | 0.7238 | 0.4667 | False |
| recovery_le_20d | two_stage_lognormal_k20 | 38 | 30 | 0.7895 | 0.5 | 0.203 | 0.8 | 0.5708 | False |
| recovery_le_20d | two_stage_lognormal_k5 | 38 | 30 | 0.7895 | 0.4958 | 0.2704 | 0.7281 | 0.5292 | False |
| recovery_le_20d | aft_lognormal_k5 | 38 | 30 | 0.7895 | 0.4833 | 0.263 | 0.7136 | 0.4833 | False |
| recovery_le_20d | aft_weibull_k5 | 38 | 30 | 0.7895 | 0.4792 | 0.2672 | 0.6983 | 0.4167 | False |

---

# Part 4. Full metric audit for every model and every target

Regenerate this section with `python scripts/audit_results.py`.

```text
====================================================================================================
1. EVERY MODEL x EVERY TARGET -- regression metrics (pooled out-of-fold)
====================================================================================================

--- Y1_ASPI_5D_Forward_LogReturn_Pct (n=40 pooled test points) ---
           model  n    RMSE     MAE  pooled_R2  RMSE_null_zero  skill_vs_zero
             mlp 40 2.97125 2.32866    0.14309         3.32737        0.10703
        ensemble 40 3.01656 2.23442    0.11675         3.32737        0.09341
         stacked 30 3.47572 2.55991    0.06333         3.71400        0.06416
   random_forest 40 3.10701 2.25867    0.06299         3.32737        0.06623
             svr 40 3.17406 2.24411    0.02211         3.32737        0.04607
              gp 40 3.24030 2.32740   -0.01913         3.32737        0.02617
           ridge 40 3.24527 2.30919   -0.02226         3.32737        0.02467
         xgboost 40 3.24858 2.32114   -0.02434         3.32737        0.02368
naive_train_mean 40 3.26577 2.33504   -0.03521         3.32737        0.01851
      naive_zero 40 3.32737 2.33743   -0.07463         3.32737        0.00000
        quantile 40 4.74715 3.13076   -1.18738         3.32737       -0.42670

--- Y1_EventWindow_0_10_LogReturn_Pct (n=40 pooled test points) ---
           model  n    RMSE     MAE  pooled_R2  RMSE_null_zero  skill_vs_zero
             mlp 40 4.60099 3.65256    0.16926         5.18242        0.11219
         stacked 30 5.40859 4.21968    0.08153         5.83005        0.07229
           ridge 40 4.91162 3.68199    0.05330         5.18242        0.05225
        ensemble 40 5.02768 3.90713    0.00803         5.18242        0.02986
             svr 40 5.04151 3.72458    0.00257         5.18242        0.02719
   random_forest 40 5.07448 3.93122   -0.01052         5.18242        0.02083
naive_train_mean 40 5.13115 3.72347   -0.03322         5.18242        0.00989
      naive_zero 40 5.18242 3.63276   -0.05396         5.18242        0.00000
              gp 40 5.44254 4.15469   -0.16242         5.18242       -0.05019
         xgboost 40 5.89614 4.57570   -0.36426         5.18242       -0.13772
        quantile 40 6.10269 4.92944   -0.46151         5.18242       -0.17757

--- Y2_abnormal_volume (n=34 pooled test points) ---
           model  n    RMSE     MAE  pooled_R2  RMSE_null_zero  skill_vs_zero
              gp 34 0.45535 0.36114    0.27042         0.54219        0.16017
             svr 34 0.45790 0.36431    0.26222         0.54219        0.15546
   random_forest 34 0.48548 0.39716    0.17068         0.54219        0.10460
        ensemble 34 0.51955 0.40637    0.05017         0.54219        0.04175
      naive_zero 34 0.54219 0.44771   -0.03441         0.54219        0.00000
         xgboost 34 0.54343 0.44155   -0.03912         0.54219       -0.00228
           ridge 34 0.54557 0.45165   -0.04733         0.54219       -0.00623
naive_train_mean 34 0.54844 0.43425   -0.05838         0.54219       -0.01152
         stacked 24 0.68505 0.51963   -0.31802         0.59685       -0.14779
             mlp 34 0.62240 0.47767   -0.36308         0.54219       -0.14793
        quantile 34 0.89885 0.69610   -1.84292         0.54219       -0.65782

--- Y3_recovery_days (n=40 pooled test points) ---
           model  n     RMSE      MAE  pooled_R2  RMSE_null_zero  skill_vs_zero
        ensemble 40 22.42090 12.59637    0.15497        27.99911        0.19923
         xgboost 40 22.77473 12.51971    0.12809        27.99911        0.18659
             svr 40 22.98489 12.85510    0.11192        27.99911        0.17908
              gp 40 23.31799 13.27686    0.08600        27.99911        0.16719
   random_forest 40 23.38484 12.51536    0.08075        27.99911        0.16480
         stacked 30 18.34950 11.52613   -0.00642        21.17388        0.13339
           ridge 40 24.56030 14.00495   -0.01399        27.99911        0.12282
naive_train_mean 40 25.07263 20.37333   -0.05673        27.99911        0.10452
             mlp 40 26.36006 15.14429   -0.16804        27.99911        0.05854
      naive_zero 40 27.99911 13.75000   -0.31781        27.99911        0.00000
        quantile 40 33.71217 20.36220   -0.91046        27.99911       -0.20404

====================================================================================================
2. DID IT BEAT A NAIVE BASELINE? (paired event bootstrap + Diebold-Mariano)
====================================================================================================
                           target         model         baseline  n  delta_rmse    ci_low  ci_high    dm_p                     verdict
 Y1_ASPI_5D_Forward_LogReturn_Pct         ridge       naive_zero 40     0.08209  -0.13575  0.26875 0.44492 better, not distinguishable
 Y1_ASPI_5D_Forward_LogReturn_Pct         ridge naive_train_mean 40     0.02050  -0.06886  0.11467 0.65686 better, not distinguishable
 Y1_ASPI_5D_Forward_LogReturn_Pct random_forest       naive_zero 40     0.22036  -0.26570  0.66207 0.37726 better, not distinguishable
 Y1_ASPI_5D_Forward_LogReturn_Pct random_forest naive_train_mean 40     0.15876  -0.19766  0.49609 0.39118 better, not distinguishable
 Y1_ASPI_5D_Forward_LogReturn_Pct       xgboost       naive_zero 40     0.07879  -0.51848  0.66321 0.79592 better, not distinguishable
 Y1_ASPI_5D_Forward_LogReturn_Pct       xgboost naive_train_mean 40     0.01719  -0.46568  0.51785 0.94611 better, not distinguishable
 Y1_ASPI_5D_Forward_LogReturn_Pct            gp       naive_zero 40     0.08707  -0.20113  0.33541 0.55016 better, not distinguishable
 Y1_ASPI_5D_Forward_LogReturn_Pct            gp naive_train_mean 40     0.02547  -0.17034  0.19192 0.78596 better, not distinguishable
 Y1_ASPI_5D_Forward_LogReturn_Pct           svr       naive_zero 40     0.15330  -0.12979  0.39310 0.29557 better, not distinguishable
 Y1_ASPI_5D_Forward_LogReturn_Pct           svr naive_train_mean 40     0.09171  -0.06149  0.22383 0.24902 better, not distinguishable
 Y1_ASPI_5D_Forward_LogReturn_Pct      quantile       naive_zero 40    -1.41979  -2.56895 -0.22675 0.06198         worse than baseline
 Y1_ASPI_5D_Forward_LogReturn_Pct      quantile naive_train_mean 40    -1.48138  -2.64088 -0.27046 0.05672         worse than baseline
 Y1_ASPI_5D_Forward_LogReturn_Pct           mlp       naive_zero 40     0.35612  -0.34009  1.03379 0.34274 better, not distinguishable
 Y1_ASPI_5D_Forward_LogReturn_Pct           mlp naive_train_mean 40     0.29452  -0.30846  0.89740 0.36515 better, not distinguishable
 Y1_ASPI_5D_Forward_LogReturn_Pct      ensemble       naive_zero 40     0.31080  -0.23543  0.83709 0.29050 better, not distinguishable
 Y1_ASPI_5D_Forward_LogReturn_Pct      ensemble naive_train_mean 40     0.24921  -0.17351  0.68348 0.28764 better, not distinguishable
 Y1_ASPI_5D_Forward_LogReturn_Pct       stacked       naive_zero 30     0.23827  -0.04254  0.51248 0.14509 better, not distinguishable
 Y1_ASPI_5D_Forward_LogReturn_Pct       stacked naive_train_mean 30     0.16768  -0.00765  0.36695 0.11989 better, not distinguishable
               Y2_abnormal_volume         ridge       naive_zero 34    -0.00338  -0.09906  0.07457 0.94103         worse than baseline
               Y2_abnormal_volume         ridge naive_train_mean 34     0.00287  -0.11605  0.10130 0.95910 better, not distinguishable
               Y2_abnormal_volume random_forest       naive_zero 34     0.05671  -0.00948  0.12244 0.12000 better, not distinguishable
               Y2_abnormal_volume random_forest naive_train_mean 34     0.06296  -0.01850  0.13280 0.15984 better, not distinguishable
               Y2_abnormal_volume       xgboost       naive_zero 34    -0.00123  -0.09488  0.09718 0.98070         worse than baseline
               Y2_abnormal_volume       xgboost naive_train_mean 34     0.00501  -0.09408  0.09340 0.92214 better, not distinguishable
               Y2_abnormal_volume            gp       naive_zero 34     0.08684   0.02521  0.15398 0.01493              BEATS BASELINE
               Y2_abnormal_volume            gp naive_train_mean 34     0.09309   0.01229  0.16685 0.05199              BEATS BASELINE
               Y2_abnormal_volume           svr       naive_zero 34     0.08429   0.03435  0.14178 0.00477              BEATS BASELINE
               Y2_abnormal_volume           svr naive_train_mean 34     0.09054   0.02991  0.14331 0.02027              BEATS BASELINE
               Y2_abnormal_volume      quantile       naive_zero 34    -0.35666  -0.54618 -0.15812 0.00758         worse than baseline
               Y2_abnormal_volume      quantile naive_train_mean 34    -0.35042  -0.53595 -0.17300 0.00529         worse than baseline
               Y2_abnormal_volume           mlp       naive_zero 34    -0.08021  -0.20958  0.04571 0.24880         worse than baseline
               Y2_abnormal_volume           mlp naive_train_mean 34    -0.07396  -0.20169  0.03784 0.24087         worse than baseline
               Y2_abnormal_volume      ensemble       naive_zero 34     0.02264  -0.05570  0.10373 0.57737 better, not distinguishable
               Y2_abnormal_volume      ensemble naive_train_mean 34     0.02888  -0.04585  0.09827 0.46780 better, not distinguishable
               Y2_abnormal_volume       stacked       naive_zero 24    -0.08821  -0.24477  0.08639 0.32575         worse than baseline
               Y2_abnormal_volume       stacked naive_train_mean 24    -0.05510  -0.17502  0.06571 0.39836         worse than baseline
                 Y3_recovery_days         ridge       naive_zero 40     3.43880   1.49840  4.91188 0.03141              BEATS BASELINE
                 Y3_recovery_days         ridge naive_train_mean 40     0.51232  -2.65933  5.62035 0.78366 better, not distinguishable
                 Y3_recovery_days random_forest       naive_zero 40     4.61427   1.01494  9.76643 0.12292              BEATS BASELINE
                 Y3_recovery_days random_forest naive_train_mean 40     1.68779  -2.89177  7.86749 0.50670 better, not distinguishable
                 Y3_recovery_days       xgboost       naive_zero 40     5.22438  -2.76972 12.61894 0.23221 better, not distinguishable
                 Y3_recovery_days       xgboost naive_train_mean 40     2.29790  -3.45985  9.19337 0.44973 better, not distinguishable
                 Y3_recovery_days            gp       naive_zero 40     4.68112   0.58726 11.35506 0.17978              BEATS BASELINE
                 Y3_recovery_days            gp naive_train_mean 40     1.75464  -3.51090  9.30281 0.56995 better, not distinguishable
                 Y3_recovery_days           svr       naive_zero 40     5.01422   0.87593 11.26727 0.14144              BEATS BASELINE
                 Y3_recovery_days           svr naive_train_mean 40     2.08774  -2.78117  8.93152 0.45411 better, not distinguishable
                 Y3_recovery_days      quantile       naive_zero 40    -5.71306 -19.41656  6.20085 0.37619         worse than baseline
                 Y3_recovery_days      quantile naive_train_mean 40    -8.63954 -18.58197  1.64848 0.13388         worse than baseline
                 Y3_recovery_days           mlp       naive_zero 40     1.63905  -2.63249  5.35104 0.45164 better, not distinguishable
                 Y3_recovery_days           mlp naive_train_mean 40    -1.28743  -5.50195  4.06913 0.60835         worse than baseline
                 Y3_recovery_days      ensemble       naive_zero 40     5.57820   0.85811  9.89236 0.07934              BEATS BASELINE
                 Y3_recovery_days      ensemble naive_train_mean 40     2.65172  -1.32481  7.32878 0.17928 better, not distinguishable
                 Y3_recovery_days       stacked       naive_zero 30     2.82438 -12.63118 18.99051 0.73864 better, not distinguishable
                 Y3_recovery_days       stacked naive_train_mean 30     2.06614 -10.04231 14.37582 0.74637 better, not distinguishable
Y1_EventWindow_0_10_LogReturn_Pct         ridge       naive_zero 40     0.27080  -1.13479  1.54907 0.70629 better, not distinguishable
Y1_EventWindow_0_10_LogReturn_Pct         ridge naive_train_mean 40     0.21953  -1.00249  1.34307 0.72625 better, not distinguishable
Y1_EventWindow_0_10_LogReturn_Pct random_forest       naive_zero 40     0.10794  -0.76658  0.78619 0.78883 better, not distinguishable
Y1_EventWindow_0_10_LogReturn_Pct random_forest naive_train_mean 40     0.05667  -0.61359  0.61452 0.85851 better, not distinguishable
Y1_EventWindow_0_10_LogReturn_Pct       xgboost       naive_zero 40    -0.71372  -2.37237  0.65755 0.36637         worse than baseline
Y1_EventWindow_0_10_LogReturn_Pct       xgboost naive_train_mean 40    -0.76498  -2.23320  0.46251 0.28721         worse than baseline
Y1_EventWindow_0_10_LogReturn_Pct            gp       naive_zero 40    -0.26012  -1.22853  0.52323 0.55265         worse than baseline
Y1_EventWindow_0_10_LogReturn_Pct            gp naive_train_mean 40    -0.31138  -1.08205  0.31827 0.37278         worse than baseline
Y1_EventWindow_0_10_LogReturn_Pct           svr       naive_zero 40     0.14090  -0.35044  0.53531 0.56033 better, not distinguishable
Y1_EventWindow_0_10_LogReturn_Pct           svr naive_train_mean 40     0.08964  -0.19593  0.31220 0.51964 better, not distinguishable
Y1_EventWindow_0_10_LogReturn_Pct      quantile       naive_zero 40    -0.92027  -2.46836  0.51066 0.21891         worse than baseline
Y1_EventWindow_0_10_LogReturn_Pct      quantile naive_train_mean 40    -0.97153  -2.35026  0.37835 0.16301         worse than baseline
Y1_EventWindow_0_10_LogReturn_Pct           mlp       naive_zero 40     0.58143  -0.68521  1.64356 0.38088 better, not distinguishable
Y1_EventWindow_0_10_LogReturn_Pct           mlp naive_train_mean 40     0.53016  -0.53815  1.44004 0.34697 better, not distinguishable
Y1_EventWindow_0_10_LogReturn_Pct      ensemble       naive_zero 40     0.15473  -1.02951  1.10364 0.78298 better, not distinguishable
Y1_EventWindow_0_10_LogReturn_Pct      ensemble naive_train_mean 40     0.10347  -0.87578  0.90023 0.82377 better, not distinguishable
Y1_EventWindow_0_10_LogReturn_Pct       stacked       naive_zero 30     0.42146  -0.56284  1.32195 0.42523 better, not distinguishable
Y1_EventWindow_0_10_LogReturn_Pct       stacked naive_train_mean 30     0.36327  -0.36212  1.06532 0.36615 better, not distinguishable

>>> comparisons whose CI excludes zero: 9 of 72

====================================================================================================
3. CLASSIFICATION -- every label x every model
====================================================================================================
             label             model  n  n_pos  prevalence  accuracy  balanced_accuracy    mcc  precision  recall    f1  pr_auc   auc  auc_boot_lo  auc_boot_hi  beats_baseline
C1_negative_return          logistic 40     16       0.400     0.625              0.562  0.161      0.571   0.250 0.348   0.583 0.591        0.393        0.779           False
C1_negative_return majority_baseline 40     16       0.400     0.600              0.500  0.000      0.000   0.000 0.000   0.400 0.500        0.500        0.500           False
C1_negative_return            rf_clf 40     16       0.400     0.650              0.583  0.229      0.667   0.250 0.364   0.540 0.578        0.380        0.763           False
C1_negative_return           xgb_clf 40     16       0.400     0.625              0.573  0.171      0.556   0.312 0.400   0.575 0.672        0.495        0.828           False
  C1b_adverse_move          logistic 40     14       0.350     0.550              0.456 -0.105      0.250   0.143 0.182   0.322 0.338        0.151        0.541           False
  C1b_adverse_move majority_baseline 40     14       0.350     0.650              0.500  0.000      0.000   0.000 0.000   0.350 0.500        0.500        0.500           False
  C1b_adverse_move            rf_clf 40     14       0.350     0.600              0.495 -0.015      0.333   0.143 0.200   0.320 0.407        0.220        0.596           False
  C1b_adverse_move           xgb_clf 40     14       0.350     0.600              0.495 -0.015      0.333   0.143 0.200   0.396 0.409        0.212        0.613           False
   C2_volume_spike          logistic 34     14       0.412     0.588              0.564  0.132      0.500   0.429 0.462   0.713 0.711        0.507        0.889            True
   C2_volume_spike majority_baseline 34     14       0.412     0.588              0.500  0.000      0.000   0.000 0.000   0.412 0.500        0.500        0.500           False
   C2_volume_spike            rf_clf 34     14       0.412     0.676              0.629  0.313      0.714   0.357 0.476   0.665 0.696        0.500        0.868           False
   C2_volume_spike           xgb_clf 34     14       0.412     0.559              0.507  0.017      0.429   0.214 0.286   0.646 0.689        0.493        0.868           False
 C3_recovers_in_90          logistic 34     31       0.912     0.794              0.586  0.128      0.929   0.839 0.881   0.968 0.710        0.484        0.903           False
 C3_recovers_in_90 majority_baseline 34     31       0.912     0.912              0.500  0.000      0.912   1.000 0.954   0.912 0.500        0.500        0.500           False
 C3_recovers_in_90            rf_clf 34     31       0.912     0.882              0.484 -0.054      0.909   0.968 0.938   0.950 0.602        0.312        0.935           False
 C3_recovers_in_90           xgb_clf 34     31       0.912     0.882              0.484 -0.054      0.909   0.968 0.938   0.989 0.892        0.742        1.000           False
 C3b_slow_recovery          logistic 34     12       0.353     0.647              0.538  0.112      0.500   0.167 0.250   0.417 0.511        0.295        0.723           False
 C3b_slow_recovery majority_baseline 34     12       0.353     0.647              0.500  0.000      0.000   0.000 0.000   0.353 0.500        0.500        0.500           False
 C3b_slow_recovery            rf_clf 34     12       0.353     0.676              0.561  0.204      0.667   0.167 0.267   0.612 0.693        0.485        0.883           False
 C3b_slow_recovery           xgb_clf 34     12       0.353     0.765              0.705  0.461      0.750   0.500 0.600   0.767 0.811        0.636        0.947            True

>>> model/label pairs clearing BOTH the majority rule and chance: 2

====================================================================================================
4. BEFORE vs AFTER -- pooled R2 against the pre-change baseline (commit 85590a8)
====================================================================================================
baseline: N=64, 31 features, 30 test points | now: N=74, 54 features, 40 test points (34 for Y2)

                          target         model  R2_before  R2_after  change direction
Y1_ASPI_5D_Forward_LogReturn_Pct random_forest    -0.3151    0.0630  0.3781    better
Y1_ASPI_5D_Forward_LogReturn_Pct         ridge    -0.5338   -0.0223  0.5115    better
Y1_ASPI_5D_Forward_LogReturn_Pct       xgboost    -1.7660   -0.0243  1.7417    better
              Y2_abnormal_volume random_forest    -0.1221    0.1707  0.2928    better
              Y2_abnormal_volume         ridge     0.1307   -0.0473 -0.1780     worse
              Y2_abnormal_volume       xgboost    -0.4630   -0.0391  0.4239    better
                Y3_recovery_days random_forest    -0.1608    0.0807  0.2415    better
                Y3_recovery_days         ridge    -0.1853   -0.0140  0.1713    better
                Y3_recovery_days       xgboost    -0.1693    0.1281  0.2974    better

>>> improved: 8 of 9 model/target pairs

====================================================================================================
5. DECOMPOSITION -- was it the extra events, or the extra features?
====================================================================================================
The two changes landed together. The ablation separates them: `no_external` is
N=74 with the ORIGINAL 31 features, so comparing it to the N=64 baseline
isolates the sample extension, and comparing it to `full` isolates the data.

                          target         model  A_before (N=64, 31f)  B_more_events (N=74, 31f)  C_plus_external (N=74, 54f)  events_effect  external_effect
Y1_ASPI_5D_Forward_LogReturn_Pct random_forest               -0.3151                    -0.0147                       0.0630         0.3004           0.0777
Y1_ASPI_5D_Forward_LogReturn_Pct         ridge               -0.5338                    -0.0398                      -0.0223         0.4940           0.0175
Y1_ASPI_5D_Forward_LogReturn_Pct       xgboost               -1.7660                    -0.2436                      -0.0243         1.5224           0.2192
              Y2_abnormal_volume random_forest               -0.1221                    -0.1322                       0.1707        -0.0101           0.3028
              Y2_abnormal_volume         ridge                0.1307                    -0.0651                      -0.0473        -0.1958           0.0178
              Y2_abnormal_volume       xgboost               -0.4630                    -0.3587                      -0.0391         0.1043           0.3196
                Y3_recovery_days random_forest               -0.1608                     0.0714                       0.0807         0.2322           0.0093
                Y3_recovery_days         ridge               -0.1853                    -0.0153                      -0.0140         0.1700           0.0013
                Y3_recovery_days       xgboost               -0.1693                     0.2535                       0.1281         0.4228          -0.1254

>>> sample extension helped: 7/9
>>> external features helped: 8/9

CAVEAT, and it matters for how much weight the two columns carry:
  `external_effect` (C - B) is a clean comparison -- identical events, identical
  folds, identical test points, only the feature set differs. It is the column
  the block ablation puts confidence intervals on.
  `events_effect` (B - A) is NOT clean. A and B are scored on DIFFERENT test
  sets (30 vs 40 points, different events), so an R2 difference there confounds
  'the model got better' with 'the test set got easier or harder'. R2 is
  normalised by the target's own variance, which changes with the sample.
  Read it as directional only; no CI is computed for it and none should be.

====================================================================================================
6. WHICH EXTERNAL BLOCK EARNED ITS PLACE? (CI excluding zero)
====================================================================================================
model                                          random_forest    ridge  xgboost
target                            block                                       
Y1_ASPI_5D_Forward_LogReturn_Pct  desinventar        0.14665  0.02425  0.25165
                                  election          -0.01960 -0.00208 -0.01647
                                  external           0.12618  0.02765  0.33077
                                  fx                -0.01344 -0.02556  0.14442
                                  hazard            -0.03613  0.06658  0.11915
Y1_EventWindow_0_10_LogReturn_Pct desinventar        0.13251  0.25907 -0.36806
                                  election           0.00567  0.38738 -0.28289
                                  external           0.09687  0.26129 -0.56345
                                  fx                 0.08964  1.11386 -0.33227
                                  hazard            -0.09364  0.18881 -0.26039
Y2_abnormal_volume                desinventar        0.02142 -0.00412  0.02783
                                  election          -0.00013  0.00090 -0.00585
                                  external           0.08176  0.00462  0.07797
                                  fx                 0.05316 -0.00569  0.09798
                                  hazard            -0.00022 -0.00230  0.01444
Y3_recovery_days                  desinventar        0.17467  0.23578 -0.28519
                                  election           0.29532 -0.05269 -0.72087
                                  external           0.11832  0.01545 -1.70169
                                  fx                 0.54550 -0.01479 -0.39405
                                  hazard             0.63552 -0.20250  0.02450

>>> significant: 7 of 60
                           target         model    block  delta_rmse      lo      hi
 Y1_ASPI_5D_Forward_LogReturn_Pct       xgboost external     0.33077 0.08272 0.55988
               Y2_abnormal_volume random_forest       fx     0.05316 0.00609 0.10340
               Y2_abnormal_volume random_forest external     0.08176 0.01357 0.15117
               Y2_abnormal_volume       xgboost       fx     0.09798 0.02319 0.18118
Y1_EventWindow_0_10_LogReturn_Pct         ridge       fx     1.11386 0.31072 1.90541
Y1_EventWindow_0_10_LogReturn_Pct         ridge election     0.38738 0.04479 0.85196
Y1_EventWindow_0_10_LogReturn_Pct random_forest       fx     0.08964 0.01125 0.18194

====================================================================================================
7. CLASSIFICATION BEFORE vs AFTER -- C2_volume_spike, the one that works
====================================================================================================
BEFORE (N=64, 30 test points -- and 0 of them had missing volume):
   model  n  balanced_accuracy  precision  recall    f1   auc  auc_boot_lo  auc_boot_hi  beats_baseline
logistic 30              0.699      0.529   0.818 0.643 0.794        0.593        0.957            True
  rf_clf 30              0.727      1.000   0.454 0.625 0.790        0.593        0.943            True
 xgb_clf 30              0.538      0.500   0.182 0.267 0.708        0.478        0.919           False

AFTER (N=74, 34 test points; 6 events with no volume data now correctly dropped):
   model  n  balanced_accuracy  precision  recall    f1   auc  auc_boot_lo  auc_boot_hi  beats_baseline
logistic 34              0.564      0.500   0.429 0.462 0.711        0.507        0.889            True
  rf_clf 34              0.629      0.714   0.357 0.476 0.696        0.500        0.868           False
 xgb_clf 34              0.507      0.429   0.214 0.286 0.689        0.493        0.868           False

====================================================================================================
8. SECTOR PANEL
====================================================================================================
                 target         model         baseline  delta_rmse   ci_low  ci_high  n_events  n_rows  significant
   Y1_sector_log_return         ridge       naive_zero     -0.0835  -0.2346   0.0398        30     594        False
   Y1_sector_log_return         ridge naive_train_mean     -0.0937  -0.2015  -0.0045        30     594        False
   Y1_sector_log_return random_forest       naive_zero     -0.3725  -0.7098  -0.1112        30     594        False
   Y1_sector_log_return random_forest naive_train_mean     -0.3826  -0.7087  -0.1272        30     594        False
Y3_sector_recovery_days         ridge       naive_zero     -3.4724 -11.7982   3.4667        30     594        False
Y3_sector_recovery_days         ridge naive_train_mean     -7.4232 -12.8524  -2.7977        30     594        False
Y3_sector_recovery_days random_forest       naive_zero     -7.6078 -16.6699  -0.2868        30     594        False
Y3_sector_recovery_days random_forest naive_train_mean    -11.5585 -17.8631  -5.9332        30     594        False

>>> sector comparisons beating their null: 0 of 8

====================================================================================================
9. Y3 HURDLE
====================================================================================================
           model  n  MAE_trading_days   RMSE
 single-stage RF 40            13.303 24.813
      naive_zero 40            13.750 27.999
naive_train_mean 40            20.279 25.003
          hurdle 40            20.766 29.612

====================================================================================================
10. DATA COVERAGE -- what the external sources actually bought
====================================================================================================
events: 74 | features: 102 | external features: 18
external feature coverage: 100.0%-100.0%
financial_damage real (the variable they were added to replace): 18/74
DesInventar matched: 52/74
  Y1_ASPI_5D_Forward_LogReturn_Pct: 74/74 observed
  Y2_abnormal_volume: 61/74 observed
  Y3_recovery_days: 74/74 observed

====================================================================================================
11. Y1/Y3 IMPROVEMENT RUN (2026-09-17) -- pre-declared grid, appended, nothing above overwritten
====================================================================================================

Pre-declaration: docs/audit.md Part 2 (written before execution).
Full results:    docs/results.md
Y2 freeze proof: docs/audit.md Part 4  (no Y2 number moved; 20 assertions)
Baseline commit: 1fbf6275, frozen in artifacts/frozen_baseline.json

Sections 1-10 above describe the FROZEN pipeline and are unchanged. This section reports a
SEPARATE, additive grid that does not modify any artifact those sections are computed from.


====================================================================================================
11.1 Y1 BASELINES (40 pooled test points per horizon, identical events)
====================================================================================================

 horizon                model  n    rmse     mae  pooled_r2
       5           naive_zero 40 3.32737 2.33743   -0.07463
       5     naive_train_mean 40 3.26220 2.32954   -0.03295
       5 market_only_expected 40 3.30523 2.32029   -0.06038
      10           naive_zero 40 5.18242 3.63276   -0.05396
      10     naive_train_mean 40 5.11694 3.71088   -0.02750
      10 market_only_expected 40 5.13667 3.71780   -0.03544
      15           naive_zero 40 6.98045 4.76094   -0.07216
      15     naive_train_mean 40 6.91169 4.98606   -0.05114
      15 market_only_expected 40 6.83137 4.90860   -0.02685
      20           naive_zero 40 6.97425 4.92857   -0.04920
      20     naive_train_mean 40 6.99496 5.16534   -0.05544
      20 market_only_expected 40 6.98328 5.19065   -0.05192

====================================================================================================
11.2 Y1 BEST CONFIGURATION PER HORIZON x INFORMATION SET
====================================================================================================

 horizon             info_set  k         model  n    RMSE     MAE       R2  skill_vs_zero  skill_vs_train_mean  skill_vs_market_only  bootstrap_delta   CI_low  CI_high  Holm_p  significant_single_test  significant_after_adjustment
       5             combined 20 random_forest 40 3.10842 2.27939  0.06214        0.06580              0.04714               0.05955          0.19682 -0.29490  0.71900     1.0                    False                         False
       5        disaster_only 10 random_forest 40 3.19257 2.33126  0.01067        0.04051              0.02135               0.03409          0.11266 -0.50068  0.74345     1.0                    False                         False
       5          market_only 10         ridge 40 3.25828 2.32305 -0.03047        0.02076              0.00120               0.01421          0.04696 -0.21109  0.40958     1.0                    False                         False
       5 normal_plus_residual 10 random_forest 40 3.24820 2.29321 -0.02410        0.02379              0.00429               0.01726          0.05704 -0.43160  0.47905     1.0                    False                         False
      10             combined 10 random_forest 40 4.99310 3.97367  0.02163        0.03653              0.02420               0.02795          0.14357 -0.69347  0.87658     1.0                    False                         False
      10        disaster_only 10   elastic_net 40 4.85606 3.48515  0.07460        0.06297              0.05098               0.05463          0.28062 -0.56018  1.04527     1.0                    False                         False
      10          market_only 20 random_forest 40 5.06185 3.85762 -0.00549        0.02327              0.01077               0.01457          0.07482 -0.56288  0.62051     1.0                    False                         False
      10 normal_plus_residual  5   elastic_net 40 5.02403 3.63710  0.00947        0.03056              0.01816               0.02193          0.11264 -0.58290  0.66844     1.0                    False                         False
      15             combined 10   elastic_net 40 6.92339 4.83743 -0.05470        0.00817             -0.00169              -0.01347         -0.09202 -1.34425  1.04910     1.0                    False                         False
      15        disaster_only  5   elastic_net 40 7.03204 5.18683 -0.08807       -0.00739             -0.01741              -0.02937         -0.20067 -1.27109  0.80373     1.0                    False                         False
      15          market_only 10   elastic_net 40 7.10482 5.10976 -0.11071       -0.01782             -0.02794              -0.04003         -0.27346 -1.14930  0.68002     1.0                    False                         False
      15 normal_plus_residual  5         ridge 40 7.21778 5.28997 -0.14631       -0.03400             -0.04429              -0.05656         -0.38641 -0.97553  0.20206     1.0                    False                         False
      20             combined 10 random_forest 40 7.05602 5.39834 -0.07394       -0.01172             -0.00873              -0.01042         -0.07273 -1.41433  1.18601     1.0                    False                         False
      20        disaster_only  5         ridge 40 7.35302 5.65291 -0.16625       -0.05431             -0.05119              -0.05295         -0.36973 -1.41054  0.62884     1.0                    False                         False
      20          market_only  5 random_forest 40 6.60140 4.90537  0.05999        0.05346              0.05626               0.05469          0.38189 -0.86863  1.52424     1.0                    False                         False
      20 normal_plus_residual  5         ridge 40 7.37391 5.43602 -0.17289       -0.05730             -0.05417              -0.05594         -0.39062 -1.04067  0.32066     1.0                    False                         False


Full 240-configuration grid: docs/thesis_materials/final_table_aspi.csv
Verdict distribution over all 720 comparisons (240 configs x 3 baselines):

verdict
C - unsupported                 573
B - suggestive but uncertain    147


>>> Y1 magnitude comparisons whose CI excludes zero: 0 of 720.
    Every interval contains zero. This is the finding, not a failure to report one.


====================================================================================================
11.3 Y1 DIRECTION (secondary analysis -- a DIFFERENT question from magnitude)
====================================================================================================

 horizon         model  n  n_negative  prevalence  balanced_accuracy  roc_auc  pr_auc      mcc  sensitivity  specificity  auc_ci_low  auc_ci_high  p_one_sided  auc_ci_excludes_chance  p_holm  holm_significant
       5      logistic 40          16       0.400            0.47917  0.55208 0.45961 -0.04572      0.25000      0.70833     0.36901      0.73602       0.2905                   False  0.6105             False
       5 random_forest 40          16       0.400            0.54167  0.52083 0.48599  0.10206      0.25000      0.83333     0.32604      0.71019       0.4115                   False  0.6105             False
      10      logistic 40          21       0.525            0.65915  0.75188 0.77088  0.33936      0.47619      0.84211     0.56660      0.89614       0.0040                    True  0.0320              True
      10 random_forest 40          21       0.525            0.61654  0.70175 0.71406  0.30633      0.28571      0.94737     0.51667      0.86575       0.0155                    True  0.1085             False
      15      logistic 40          20       0.500            0.62500  0.69750 0.64368  0.25820      0.50000      0.75000     0.51260      0.86446       0.0205                    True  0.1230             False
      15 random_forest 40          20       0.500            0.62500  0.61000 0.62801  0.26688      0.45000      0.80000     0.42381      0.79053       0.1175                   False  0.4700             False
      20      logistic 40          21       0.525            0.65414  0.66165 0.65420  0.31141      0.57143      0.73684     0.48000      0.83070       0.0490                   False  0.2450             False
      20 random_forest 40          21       0.525            0.58271  0.57644 0.63843  0.17318      0.42857      0.73684     0.39586      0.76257       0.2035                   False  0.6105             False


>>> h=10 logistic is the ONLY comparison anywhere in this improvement run that survives a
    Holm family-wise correction (p_holm = 0.032).


====================================================================================================
11.4 Y1 FEATURE-SELECTION STABILITY
====================================================================================================

features selected in exactly one of four folds: 0.356 of all selections


--- h=5 combined K=10

         feature  selection_count  selection_frequency  mean_rank  median_rank
       fx_vol_30                4                 1.00      6.250          6.0
     fx_logret_5                4                 1.00      6.500          6.5
 di_affected_log                3                 0.75      1.667          1.0
      di_records                3                 0.75      5.333          6.0
      log_return                2                 0.50      1.000          1.0
  lag_return_t-1                2                 0.50      2.500          2.5
log_vol_change_1                2                 0.50      5.000          5.0
     fx_logret_1                2                 0.50      6.000          6.0


--- h=10 combined K=10

        feature  selection_count  selection_frequency  mean_rank  median_rank
 lag_return_t-1                4                 1.00      2.250          1.5
di_affected_log                3                 0.75      3.667          2.0
      fx_vol_30                3                 0.75      5.000          5.0
     log_return                3                 0.75      5.000          4.0
 lag_return_t-3                3                 0.75      7.667          8.0
     di_records                2                 0.50      3.500          3.5
    fx_logret_5                2                 0.50      4.000          4.0
   mag_area_km2                2                 0.50      8.500          8.5


--- h=10 disaster_only K=10

                feature  selection_count  selection_frequency  mean_rank  median_rank
        di_affected_log                4                 1.00      3.500          2.0
           mag_area_km2                3                 0.75      2.667          2.0
           total_deaths                3                 0.75      3.667          3.0
             di_records                3                 0.75      4.333          3.0
          hz_wind_max3d                3                 0.75      6.000          4.0
disasters_trailing_365d                3                 0.75      6.333          6.0
       di_districts_hit                3                 0.75      6.667          7.0
         hz_precip_anom                3                 0.75      6.667          7.0



====================================================================================================
11.5 Y3 CENSORING-AWARE SURVIVAL MODELS (40 pooled test rows, 31 observed, 9 censored)
====================================================================================================

                  model  n  n_recovered  n_censored  C_index     C_index_CI  Integrated_Brier_Score  Median_abs_error_uncensored  mae_uncensored  rmse_uncensored  max_abs_calibration_gap  p_holm  holm_significant                      verdict
  two_stage_weibull_k20 40           31           9  0.65712 [0.522, 0.769]                 0.19430                          4.0        15.48387         30.59623                  0.04472   0.165             False  A - statistically supported
two_stage_lognormal_k20 40           31           9  0.64323 [0.503, 0.758]                 0.19612                          3.0        14.77419         28.47523                  0.04537   0.336             False  A - statistically supported
      aft_lognormal_k20 40           31           9  0.59809 [0.453, 0.727]                 0.16466                          4.0         9.87097         18.24387                  0.14192   1.000             False B - suggestive but uncertain
   two_stage_weibull_k5 40           31           9  0.58854 [0.456, 0.717]                 0.16678                          4.0         8.38710         17.14078                  0.14964   1.000             False B - suggestive but uncertain
        aft_weibull_k20 40           31           9  0.58073 [0.427, 0.711]                 0.15417                          5.0         9.48387         17.49470                  0.16298   1.000             False B - suggestive but uncertain
       aft_lognormal_k5 40           31           9  0.57378 [0.427, 0.721]                 0.16303                          3.0         5.77419          8.05425                  0.15343   1.000             False B - suggestive but uncertain
 two_stage_lognormal_k5 40           31           9  0.57378 [0.437, 0.705]                 0.16204                          5.0         6.03226          9.20554                  0.14623   1.000             False B - suggestive but uncertain
two_stage_lognormal_k10 40           31           9  0.56858 [0.419, 0.713]                 0.14711                          3.0         5.83871          9.41687                  0.15628   1.000             False B - suggestive but uncertain
                 cox_k5 40           31           9  0.56858 [0.428, 0.710]                 0.16879                          3.0         7.19355         15.74289                  0.09686   1.000             False B - suggestive but uncertain
  two_stage_weibull_k10 40           31           9  0.56684 [0.421, 0.706]                 0.14847                          4.0         7.64516         12.96770                  0.14914   1.000             False B - suggestive but uncertain
      aft_lognormal_k10 40           31           9  0.56337 [0.397, 0.718]                 0.15005                          2.0         5.35484          7.71676                  0.16336   1.000             False B - suggestive but uncertain
         aft_weibull_k5 40           31           9  0.55990 [0.421, 0.707]                 0.18058                          5.0         9.90323         15.50546                  0.15221   1.000             False B - suggestive but uncertain
        aft_weibull_k10 40           31           9  0.53906 [0.369, 0.693]                 0.15730                          5.0         7.58065         11.07890                  0.16170   1.000             False B - suggestive but uncertain
      km_train_baseline 40           31           9  0.47656 [0.353, 0.615]                 0.15783                          3.0         4.87097          6.18792                  0.04850   1.000             False              C - unsupported
  train_median_baseline 40           31           9  0.47656 [0.353, 0.615]                 0.20327                          3.0         4.87097          6.18792                  0.35000   1.000             False              C - unsupported

====================================================================================================
11.6 Y3 RECOVERY-PROBABILITY CALIBRATION (predicted vs Kaplan-Meier)
====================================================================================================

--- two_stage_weibull_k20

   t  predicted_P_T_le_t  observed_KM_P_T_le_t  calibration_gap
10.0              0.6547                0.6500           0.0047
20.0              0.7130                0.7577          -0.0447
30.0              0.7447                0.7880          -0.0433
60.0              0.8023                0.7880           0.0143
90.0              0.8231                0.7880           0.0351


--- aft_lognormal_k20

   t  predicted_P_T_le_t  observed_KM_P_T_le_t  calibration_gap
10.0              0.6904                0.6500           0.0404
20.0              0.7903                0.7577           0.0327
30.0              0.8381                0.7880           0.0501
60.0              0.9020                0.7880           0.1140
90.0              0.9299                0.7880           0.1419


--- km_train_baseline

   t  predicted_P_T_le_t  observed_KM_P_T_le_t  calibration_gap
10.0              0.6305                0.6500          -0.0195
20.0              0.7092                0.7577          -0.0485
30.0              0.7594                0.7880          -0.0286
60.0              0.7794                0.7880          -0.0086
90.0              0.7794                0.7880          -0.0086



====================================================================================================
11.7 Y3 RECOVERY CATEGORY: recovery <= 20 trading days (pre-specified)
====================================================================================================

       category                   model  n  n_positive  prevalence  roc_auc  auc_ci_low  auc_ci_high  balanced_accuracy  beats_chance
recovery_le_20d       km_train_baseline 38          30      0.7895   0.5125      0.2893       0.7235             0.5000         False
recovery_le_20d   train_median_baseline 38          30      0.7895   0.5000      0.5000       0.5000             0.5000         False
recovery_le_20d          aft_weibull_k5 38          30      0.7895   0.4792      0.2672       0.6983             0.4167         False
recovery_le_20d    two_stage_weibull_k5 38          30      0.7895   0.5208      0.2857       0.7517             0.5125         False
recovery_le_20d        aft_lognormal_k5 38          30      0.7895   0.4833      0.2630       0.7136             0.4833         False
recovery_le_20d  two_stage_lognormal_k5 38          30      0.7895   0.4958      0.2704       0.7281             0.5292         False
recovery_le_20d                  cox_k5 38          30      0.7895   0.5000      0.2796       0.7238             0.4667         False
recovery_le_20d         aft_weibull_k10 38          30      0.7895   0.5917      0.3257       0.8519             0.6375         False
recovery_le_20d   two_stage_weibull_k10 38          30      0.7895   0.5917      0.3179       0.8526             0.6208         False
recovery_le_20d       aft_lognormal_k10 38          30      0.7895   0.5833      0.3073       0.8492             0.5625         False
recovery_le_20d two_stage_lognormal_k10 38          30      0.7895   0.5792      0.3118       0.8386             0.6083         False
recovery_le_20d         aft_weibull_k20 38          30      0.7895   0.6125      0.3472       0.8544             0.6042         False
recovery_le_20d   two_stage_weibull_k20 38          30      0.7895   0.5458      0.2719       0.8243             0.5708         False
recovery_le_20d       aft_lognormal_k20 38          30      0.7895   0.5875      0.3279       0.8249             0.6208         False
recovery_le_20d two_stage_lognormal_k20 38          30      0.7895   0.5000      0.2030       0.8000             0.5708         False


>>> models beating chance: 0 of 15. Prevalence is 0.789; there is little to discriminate.
    The second pre-specified category (C3b_slow_recovery) is unchanged -- see section 3.


====================================================================================================
11.8 DECISION-RULE CLASSIFICATION (protocol Part 8)
====================================================================================================

A = statistically supported  |  B = suggestive but uncertain  |  C = unsupported

  Y1 return magnitude (any horizon / information set / model)          C
  Disaster info beyond market state, for magnitude                     C
  Normal-market + disaster-residual decomposition                      C
  Y1 direction, h=10, logistic, combined, K=10                         A   (survives Holm)
  Y1 direction, h=15 and h=20                                          B
  Y1 direction, h=5                                                    C
  Y3 recovery ranking, two-stage AFT, K=20                             A on a single test,
                                                                       B after Holm -> reported as B
  Y3 recovery-probability calibration (two-stage)                      B   (descriptive)
  Y3 recovery <= 20 days classification                                C
  Y3 C3b_slow_recovery (existing, unchanged)                           A   (as already reported)
  Y2 abnormal volume (frozen, unchanged)                               A

No B was promoted to A. No negative experiment was removed.
```
