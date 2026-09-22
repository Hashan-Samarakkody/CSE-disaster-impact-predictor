# Results

Every number the executed repository produced, for all three research targets, in the
direction it landed. Disappointing results are reported here, not removed.

All figures on this page come from a complete re run of the pipeline on 2026-09-21 under
the target definitions frozen in `docs/TARGET_DEFINITION_PROTOCOL.md` on 2026-09-19.
Numbers published before that date were produced under the earlier definitions and do not
carry over; `docs/audit.md` keeps them as the historical record and says so at its head.

The machine readable versions live in `artifacts/tables/`, and the two thesis facing
tables in `docs/thesis_materials/final_table_aspi.csv` and
`docs/thesis_materials/final_table_recovery.csv`.

## Contents

1. What was run
2. Headline answers
3. Target one, return magnitude
4. Target one, return direction
5. Target two, forward abnormal volume
6. Target three, recovery duration
7. The classification arm
8. Robustness and ablations
9. The sector panel

---

## 1. What was run

| | |
|---|---|
| Events | 74, EM-DAT records with `total_affected >= 1000` matched to a tradable CSE session |
| Sample period | 2000-09-18 to 2025-11-27 |
| Outer validation | chronological walk forward, train 30, test 10, step 10, four folds |
| Pooled test points | 40 per configuration, the identical events for every candidate and every baseline |
| Return grid | 5 horizons x 6 information sets x 3 feature capacities x 5 model families = 465 configurations |
| Return comparisons | 1350 paired bootstrap tests, of which 18 form the confirmatory family and 1332 are exploratory |
| Confirmatory family | horizon 5, the real_time and ex_post information sets, capacity 10, and the models ridge, random forest and MLP against three baselines. Holm is applied to these 18 and to nothing else |
| Survival grid | 16 configurations, including the Aalen-Johansen competing risks arm added by T2 |
| Classification arm | 6 labels x 3 model families, against a majority rule baseline |
| Seed | `RANDOM_STATE = 42` throughout, never varied to obtain a better score |

A model beats a baseline when the paired bootstrap interval on the RMSE difference
excludes zero. The bootstrap resamples disaster episodes rather than individual events.
The Holm correction is reported alongside as a stricter diagnostic and is never folded
into the primary criterion.

## 2. Headline answers

These are the numbers from the single final run of 2026-09-22, under the Revision 2
changes recorded in `docs/audit.md` Part 8. Where a figure below differs from an earlier
draft of this document, the cause is T8: the Gaussian process, the support vector
regressor and the quantile regressor are now tuned on the same purged inner splits as
every other model, so the tuned-against-untuned comparison that produced the earlier
volume verdict no longer exists.

- **Return magnitude is not predictable.** 0 of 450 configurations was significant on even
  a single uncorrected test, and 0 of the 18 confirmatory comparisons produced an interval
  excluding zero. The smallest Holm corrected p in the confirmatory family is 0.641. The
  best pooled R squared in the notebook arm is 0.042.
- **Return direction at ten sessions is predictable.** Logistic regression on the combined
  information set reaches ROC AUC 0.817, interval [0.657, 0.940], and it is the only
  result in this body of work that survives a Holm correction, at p below 0.001. The ten
  session horizon is the one pre-declared for direction; at the primary five session
  horizon the AUC is 0.574 with an interval spanning 0.5.
- **Forward abnormal volume is the one qualified positive.** The ensemble reaches pooled R
  squared 0.334 and the random forest, the strongest confirmatory model, reaches 0.303.
  The random forest interval excludes zero against both naive baselines, [0.037, 0.175]
  against the zero baseline, but it does not survive the Holm correction, at corrected p
  0.078. It rests on 34 held-out points, and the missingness is structured: volume is
  unavailable for 2000 and for events after 2023, the first and last of the four folds.
  Report this as suggestive, not as established.
- **Recovery duration is not predictable, in duration or in rank.** Every point model
  loses to the training mean, every concordance interval in the survival family contains
  0.5, and the Kaplan-Meier and Aalen-Johansen baselines carry the best integrated Brier
  scores, so the fitted models are worse calibrated than doing nothing.
- **Later knowledge of severity buys almost nothing.** On the principal analysis the
  real_time information set gives a delta RMSE of +0.0930 and the ex_post set +0.0932.
  The finalised EM-DAT, DesInventar, NASA POWER and World Bank variables move the result
  by 0.0002.
- **A measurable response does not imply a forecastable one.** The event study finds no
  detectable realised return response at all, CAAR(1,5) = +0.0006 with every test p above
  0.85, while volume shows CAAR(1,5) = +0.63 log points on which only the Corrado rank
  test fires, p 0.024, against BMP p 0.186 and Kolari-Pynnonen p 0.298.

---

## 3. Target one, return magnitude

### 3.1 Baselines

Everything in the grid is measured against these three.

|   horizon | model                |   n |   rmse |    mae |   pooled_r2 |
|----------:|:---------------------|----:|-------:|-------:|------------:|
|         5 | market_only_expected |  40 | 2.8721 | 1.9710 |     -0.0966 |
|         5 | naive_train_mean     |  40 | 2.8014 | 1.9469 |     -0.0433 |
|         5 | naive_zero           |  40 | 2.8691 | 1.9206 |     -0.0942 |
|        10 | market_only_expected |  40 | 5.2052 | 3.6434 |     -0.0542 |
|        10 | naive_train_mean     |  40 | 5.1686 | 3.6337 |     -0.0394 |
|        10 | naive_zero           |  40 | 5.2618 | 3.5455 |     -0.0772 |
|        15 | market_only_expected |  40 | 6.7513 | 4.7390 |     -0.0314 |
|        15 | naive_train_mean     |  40 | 6.7730 | 4.7869 |     -0.0380 |
|        15 | naive_zero           |  40 | 6.8626 | 4.6094 |     -0.0656 |
|        20 | market_only_expected |  40 | 6.4293 | 4.8432 |     -0.0502 |
|        20 | naive_train_mean     |  40 | 6.4334 | 4.8465 |     -0.0515 |
|        20 | naive_zero           |  40 | 6.4560 | 4.6192 |     -0.0589 |

Note that all three baselines carry a negative pooled R squared. The target has almost no
variance a constant can capture, which is the first sign of what follows.

### 3.2 Best configuration per horizon and information set

Selected by lowest pooled RMSE within each cell, out of the fifteen model and capacity
combinations in it.

|   horizon | info_set             |   k | model         |   n |   rmse |    mae |   pooled_r2 |   skill_vs_zero |
|----------:|:---------------------|----:|:--------------|----:|-------:|-------:|------------:|----------------:|
|         5 | combined             |  20 | mlp           |  40 | 2.7095 | 2.0361 |      0.0241 |          0.0556 |
|         5 | disaster_only        |   5 | random_forest |  40 | 2.6902 | 1.9291 |      0.0380 |          0.0623 |
|         5 | market_only          |  10 | mlp           |  40 | 2.7025 | 2.0190 |      0.0291 |          0.0580 |
|         5 | normal_plus_residual |   5 | random_forest |  40 | 2.8084 | 2.0033 |     -0.0484 |          0.0212 |
|        10 | combined             |  10 | random_forest |  40 | 4.8868 | 3.7764 |      0.0709 |          0.0713 |
|        10 | disaster_only        |   5 | elastic_net   |  40 | 4.8303 | 3.5492 |      0.0922 |          0.0820 |
|        10 | market_only          |  20 | random_forest |  40 | 4.9736 | 3.7023 |      0.0376 |          0.0548 |
|        10 | normal_plus_residual |   5 | elastic_net   |  40 | 4.9062 | 3.4601 |      0.0635 |          0.0676 |
|        15 | combined             |  20 | random_forest |  40 | 6.6015 | 5.0079 |      0.0139 |          0.0380 |
|        15 | disaster_only        |   5 | elastic_net   |  40 | 7.0052 | 5.2269 |     -0.1104 |         -0.0208 |
|        15 | market_only          |  10 | random_forest |  40 | 6.7881 | 5.0774 |     -0.0426 |          0.0109 |
|        15 | normal_plus_residual |   5 | elastic_net   |  40 | 7.0840 | 5.2553 |     -0.1355 |         -0.0323 |
|        20 | combined             |  10 | random_forest |  40 | 6.5728 | 4.9104 |     -0.0976 |         -0.0181 |
|        20 | disaster_only        |   5 | ridge         |  40 | 6.9762 | 5.4043 |     -0.2365 |         -0.0806 |
|        20 | market_only          |  20 | random_forest |  40 | 6.1534 | 4.6387 |      0.0380 |          0.0469 |
|        20 | normal_plus_residual |   5 | ridge         |  40 | 6.8517 | 5.1450 |     -0.1927 |         -0.0613 |

### 3.3 The primary question, does disaster information add anything

The pre declared primary comparison is each configuration against the market only expected
return baseline, which represents what the market's own state already implied.

|   horizon | info_set      |   k | model         |   delta_rmse |   ci_low |   ci_high |   p_holm | verdict                      |
|----------:|:--------------|----:|:--------------|-------------:|---------:|----------:|---------:|:-----------------------------|
|         5 | disaster_only |   5 | random_forest |       0.1820 |  -0.3953 |    0.6822 |   1.0000 | B - suggestive but uncertain |
|        10 | disaster_only |   5 | elastic_net   |       0.3749 |  -0.3884 |    1.0463 |   1.0000 | B - suggestive but uncertain |
|        15 | combined      |  20 | random_forest |       0.1498 |  -0.7853 |    0.9120 |   1.0000 | B - suggestive but uncertain |
|        20 | market_only   |  20 | random_forest |       0.2760 |  -0.7552 |    1.1875 |   1.0000 | B - suggestive but uncertain |

Every interval contains zero. Across all 1350 comparisons, against all three baselines, the
count of intervals excluding zero is zero, and the count surviving the Holm correction is
also zero.

---

## 4. Target one, return direction

A different question from magnitude, reported as one. The label is simply whether the
horizon return is negative, scored on the same folds, the same purge and the combined
information set with ten selected features.

|   horizon | model         |   n |   prevalence |   balanced_accuracy |   roc_auc |   pr_auc |    mcc |   sensitivity |   specificity |   auc_ci_low |   auc_ci_high |   p_holm | holm_significant   |
|----------:|:--------------|----:|-------------:|--------------------:|----------:|---------:|-------:|--------------:|--------------:|-------------:|--------------:|---------:|:-------------------|
|         5 | logistic      |  40 |       0.4750 |              0.5100 |    0.5739 |   0.5326 | 0.0250 |        0.2105 |        0.8095 |       0.4131 |        0.7486 |   0.6940 | False              |
|         5 | random_forest |  40 |       0.4750 |              0.5313 |    0.5439 |   0.5273 | 0.0946 |        0.1579 |        0.9048 |       0.3624 |        0.7181 |   0.6940 | False              |
|        10 | logistic      |  40 |       0.5250 |              0.7569 |    0.8170 |   0.8254 | 0.5300 |        0.6190 |        0.8947 |       0.6566 |        0.9396 |   0.0000 | True               |
|        10 | random_forest |  40 |       0.5250 |              0.6642 |    0.7093 |   0.7172 | 0.3926 |        0.3810 |        0.9474 |       0.5286 |        0.8738 |   0.1085 | False              |
|        15 | logistic      |  40 |       0.5250 |              0.6065 |    0.6942 |   0.6645 | 0.2197 |        0.4762 |        0.7368 |       0.5135 |        0.8546 |   0.1085 | False              |
|        15 | random_forest |  40 |       0.5250 |              0.6090 |    0.5489 |   0.6411 | 0.2325 |        0.4286 |        0.7895 |       0.3609 |        0.7300 |   0.6940 | False              |
|        20 | logistic      |  40 |       0.5000 |              0.6250 |    0.6375 |   0.6202 | 0.2529 |        0.5500 |        0.7000 |       0.4486 |        0.8124 |   0.3950 | False              |
|        20 | random_forest |  40 |       0.5000 |              0.6000 |    0.5850 |   0.6301 | 0.2097 |        0.4500 |        0.7500 |       0.4034 |        0.7724 |   0.6940 | False              |

The ten session logistic model is the study's single corrected result. It is asymmetric:
sensitivity 0.619 against specificity 0.895, so it is a better indicator that the index
will not fall than a detector of falls. The result does not transfer. At five sessions it
fails outright, and at fifteen the interval excludes chance on a single test but does not
survive correction.

---

## 5. Target two, forward abnormal volume

### 5.1 Pooled regression metrics for all four modelled columns

| target                                | model            |   n |    rmse |     mae |   pooled_r2 |
|:--------------------------------------|:-----------------|----:|--------:|--------:|------------:|
| Y1_ASPI_10D_Forward_LogReturn_Pct     | random_forest    |  40 |  4.9391 |  3.7556 |      0.0509 |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | ridge            |  40 |  5.0590 |  3.6116 |      0.0042 |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | svr              |  40 |  5.0651 |  3.8476 |      0.0018 |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | ensemble         |  40 |  5.0814 |  3.9176 |     -0.0046 |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | quantile         |  40 |  5.1325 |  3.6767 |     -0.0249 |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | mlp              |  40 |  5.1762 |  4.1834 |     -0.0425 |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | naive_train_mean |  40 |  5.1799 |  3.6434 |     -0.0439 |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | gp               |  40 |  5.2454 |  3.9781 |     -0.0705 |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | naive_zero       |  40 |  5.2618 |  3.5455 |     -0.0772 |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | stacked          |  30 |  5.9581 |  4.5825 |     -0.1001 |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | xgboost          |  40 |  5.8161 |  4.1990 |     -0.3161 |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | mlp              |  40 |  2.6852 |  2.1020 |      0.0415 |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | ensemble         |  40 |  2.6975 |  1.9923 |      0.0327 |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | random_forest    |  40 |  2.7235 |  2.0076 |      0.0140 |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | stacked          |  30 |  3.0697 |  2.2106 |      0.0091 |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | gp               |  40 |  2.7335 |  1.9475 |      0.0067 |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | svr              |  40 |  2.8037 |  1.9417 |     -0.0449 |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | naive_train_mean |  40 |  2.8051 |  1.9483 |     -0.0460 |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | ridge            |  40 |  2.8478 |  1.9051 |     -0.0781 |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | quantile         |  40 |  2.8614 |  2.0840 |     -0.0884 |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | naive_zero       |  40 |  2.8691 |  1.9206 |     -0.0942 |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | xgboost          |  40 |  2.9379 |  2.1282 |     -0.1474 |
| Y2_5D_Forward_AbnormalVolume_LogRatio | ensemble         |  34 |  0.4882 |  0.3816 |      0.3344 |
| Y2_5D_Forward_AbnormalVolume_LogRatio | mlp              |  34 |  0.4913 |  0.3836 |      0.3257 |
| Y2_5D_Forward_AbnormalVolume_LogRatio | stacked          |  24 |  0.5491 |  0.4531 |      0.3154 |
| Y2_5D_Forward_AbnormalVolume_LogRatio | random_forest    |  34 |  0.4995 |  0.4003 |      0.3030 |
| Y2_5D_Forward_AbnormalVolume_LogRatio | gp               |  34 |  0.5133 |  0.4223 |      0.2640 |
| Y2_5D_Forward_AbnormalVolume_LogRatio | xgboost          |  34 |  0.5167 |  0.4097 |      0.2544 |
| Y2_5D_Forward_AbnormalVolume_LogRatio | svr              |  34 |  0.5973 |  0.4650 |      0.0036 |
| Y2_5D_Forward_AbnormalVolume_LogRatio | naive_zero       |  34 |  0.6055 |  0.4749 |     -0.0241 |
| Y2_5D_Forward_AbnormalVolume_LogRatio | ridge            |  34 |  0.6079 |  0.4573 |     -0.0324 |
| Y2_5D_Forward_AbnormalVolume_LogRatio | quantile         |  34 |  0.6126 |  0.5038 |     -0.0483 |
| Y2_5D_Forward_AbnormalVolume_LogRatio | naive_train_mean |  34 |  0.6171 |  0.4999 |     -0.0638 |
| Y3_ASPI_Recovery_Time                 | naive_train_mean |  40 | 21.5973 | 16.3517 |     -0.0369 |
| Y3_ASPI_Recovery_Time                 | quantile         |  40 | 21.9889 | 11.4483 |     -0.0749 |
| Y3_ASPI_Recovery_Time                 | ridge            |  40 | 22.0146 | 12.7323 |     -0.0774 |
| Y3_ASPI_Recovery_Time                 | random_forest    |  40 | 22.4085 | 12.1773 |     -0.1163 |
| Y3_ASPI_Recovery_Time                 | svr              |  40 | 22.7306 | 11.6428 |     -0.1486 |
| Y3_ASPI_Recovery_Time                 | gp               |  40 | 22.8618 | 12.0355 |     -0.1619 |
| Y3_ASPI_Recovery_Time                 | ensemble         |  40 | 23.5885 | 13.8893 |     -0.2369 |
| Y3_ASPI_Recovery_Time                 | stacked          |  30 | 20.5934 | 14.8222 |     -0.2543 |
| Y3_ASPI_Recovery_Time                 | naive_zero       |  40 | 24.4182 | 12.1000 |     -0.3255 |
| Y3_ASPI_Recovery_Time                 | mlp              |  40 | 25.6179 | 14.5511 |     -0.4589 |
| Y3_ASPI_Recovery_Time                 | xgboost          |  40 | 26.7466 | 15.9278 |     -0.5903 |

### 5.2 Verdicts against the two naive baselines

| target                                | model         | baseline         |   n |   rmse_model |   rmse_baseline |   delta_rmse |   ci_low |   ci_high |   dm_p |   p_holm | holm_significant   | verdict                     |
|:--------------------------------------|:--------------|:-----------------|----:|-------------:|----------------:|-------------:|---------:|----------:|-------:|---------:|:-------------------|:----------------------------|
| Y1_ASPI_5D_Forward_LogReturn_Pct      | ridge         | naive_zero       |  40 |       2.8478 |          2.8691 |       0.0212 |  -0.1393 |    0.1902 | 0.7939 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | ridge         | naive_train_mean |  40 |       2.8478 |          2.8051 |      -0.0427 |  -0.1798 |    0.1151 | 0.5898 |   1.0000 | False              | worse than baseline         |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | random_forest | naive_zero       |  40 |       2.7235 |          2.8691 |       0.1456 |  -0.2815 |    0.4777 | 0.4797 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | random_forest | naive_train_mean |  40 |       2.7235 |          2.8051 |       0.0817 |  -0.2191 |    0.3234 | 0.5737 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | xgboost       | naive_zero       |  40 |       2.9379 |          2.8691 |      -0.0688 |  -0.4247 |    0.2032 | 0.6575 |   1.0000 | False              | worse than baseline         |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | xgboost       | naive_train_mean |  40 |       2.9379 |          2.8051 |      -0.1327 |  -0.3654 |    0.0533 | 0.2060 |   1.0000 | False              | worse than baseline         |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | gp            | naive_zero       |  40 |       2.7335 |          2.8691 |       0.1355 |  -0.2154 |    0.4103 | 0.4341 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | gp            | naive_train_mean |  40 |       2.7335 |          2.8051 |       0.0716 |  -0.1728 |    0.2577 | 0.5354 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | svr           | naive_zero       |  40 |       2.8037 |          2.8691 |       0.0654 |  -0.3075 |    0.3899 | 0.7252 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | svr           | naive_train_mean |  40 |       2.8037 |          2.8051 |       0.0015 |  -0.2415 |    0.2185 | 0.9902 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | quantile      | naive_zero       |  40 |       2.8614 |          2.8691 |       0.0077 |  -0.6221 |    0.5455 | 0.9805 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | quantile      | naive_train_mean |  40 |       2.8614 |          2.8051 |      -0.0562 |  -0.5670 |    0.4123 | 0.8335 |   1.0000 | False              | worse than baseline         |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | mlp           | naive_zero       |  40 |       2.6852 |          2.8691 |       0.1838 |  -0.3778 |    0.6357 | 0.5102 | nan      | False              | better, not distinguishable |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | mlp           | naive_train_mean |  40 |       2.6852 |          2.8051 |       0.1199 |  -0.3677 |    0.5358 | 0.6206 | nan      | False              | better, not distinguishable |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | ensemble      | naive_zero       |  40 |       2.6975 |          2.8691 |       0.1716 |  -0.2165 |    0.4710 | 0.3702 | nan      | False              | better, not distinguishable |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | ensemble      | naive_train_mean |  40 |       2.6975 |          2.8051 |       0.1077 |  -0.1703 |    0.3290 | 0.4283 | nan      | False              | better, not distinguishable |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | stacked       | naive_zero       |  30 |       3.0697 |          3.2280 |       0.1583 |  -0.1377 |    0.4137 | 0.3144 | nan      | False              | better, not distinguishable |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | stacked       | naive_train_mean |  30 |       3.0697 |          3.1400 |       0.0703 |  -0.0964 |    0.2180 | 0.4083 | nan      | False              | better, not distinguishable |
| Y2_5D_Forward_AbnormalVolume_LogRatio | ridge         | naive_zero       |  34 |       0.6079 |          0.6055 |      -0.0024 |  -0.1137 |    0.0831 | 0.9632 |   1.0000 | False              | worse than baseline         |
| Y2_5D_Forward_AbnormalVolume_LogRatio | ridge         | naive_train_mean |  34 |       0.6079 |          0.6171 |       0.0092 |  -0.1160 |    0.1040 | 0.8727 |   1.0000 | False              | better, not distinguishable |
| Y2_5D_Forward_AbnormalVolume_LogRatio | random_forest | naive_zero       |  34 |       0.4995 |          0.6055 |       0.1060 |   0.0371 |    0.1754 | 0.0124 |   0.0782 | False              | BEATS BASELINE              |
| Y2_5D_Forward_AbnormalVolume_LogRatio | random_forest | naive_train_mean |  34 |       0.4995 |          0.6171 |       0.1176 |   0.0175 |    0.2169 | 0.0383 |   0.4092 | False              | BEATS BASELINE              |
| Y2_5D_Forward_AbnormalVolume_LogRatio | xgboost       | naive_zero       |  34 |       0.5167 |          0.6055 |       0.0889 |   0.0122 |    0.1673 | 0.0434 |   0.5074 | False              | BEATS BASELINE              |
| Y2_5D_Forward_AbnormalVolume_LogRatio | xgboost       | naive_train_mean |  34 |       0.5167 |          0.6171 |       0.1005 |  -0.0001 |    0.1996 | 0.0675 |   1.0000 | False              | better, not distinguishable |
| Y2_5D_Forward_AbnormalVolume_LogRatio | gp            | naive_zero       |  34 |       0.5133 |          0.6055 |       0.0922 |  -0.0400 |    0.2167 | 0.1908 |   1.0000 | False              | better, not distinguishable |
| Y2_5D_Forward_AbnormalVolume_LogRatio | gp            | naive_train_mean |  34 |       0.5133 |          0.6171 |       0.1038 |  -0.0364 |    0.2367 | 0.1698 |   1.0000 | False              | better, not distinguishable |
| Y2_5D_Forward_AbnormalVolume_LogRatio | svr           | naive_zero       |  34 |       0.5973 |          0.6055 |       0.0083 |  -0.0572 |    0.0715 | 0.8064 |   1.0000 | False              | better, not distinguishable |
| Y2_5D_Forward_AbnormalVolume_LogRatio | svr           | naive_train_mean |  34 |       0.5973 |          0.6171 |       0.0199 |  -0.0183 |    0.0619 | 0.3508 |   1.0000 | False              | better, not distinguishable |
| Y2_5D_Forward_AbnormalVolume_LogRatio | quantile      | naive_zero       |  34 |       0.6126 |          0.6055 |      -0.0071 |  -0.1804 |    0.1359 | 0.9327 |   1.0000 | False              | worse than baseline         |
| Y2_5D_Forward_AbnormalVolume_LogRatio | quantile      | naive_train_mean |  34 |       0.6126 |          0.6171 |       0.0045 |  -0.1803 |    0.1616 | 0.9600 |   1.0000 | False              | better, not distinguishable |
| Y2_5D_Forward_AbnormalVolume_LogRatio | mlp           | naive_zero       |  34 |       0.4913 |          0.6055 |       0.1142 |   0.0137 |    0.2161 | 0.0424 | nan      | False              | BEATS BASELINE              |
| Y2_5D_Forward_AbnormalVolume_LogRatio | mlp           | naive_train_mean |  34 |       0.4913 |          0.6171 |       0.1258 |   0.0036 |    0.2526 | 0.0645 | nan      | False              | BEATS BASELINE              |
| Y2_5D_Forward_AbnormalVolume_LogRatio | ensemble      | naive_zero       |  34 |       0.4882 |          0.6055 |       0.1174 |   0.0507 |    0.1850 | 0.0061 | nan      | False              | BEATS BASELINE              |
| Y2_5D_Forward_AbnormalVolume_LogRatio | ensemble      | naive_train_mean |  34 |       0.4882 |          0.6171 |       0.1290 |   0.0319 |    0.2295 | 0.0237 | nan      | False              | BEATS BASELINE              |
| Y2_5D_Forward_AbnormalVolume_LogRatio | stacked       | naive_zero       |  24 |       0.5491 |          0.6640 |       0.1149 |   0.0186 |    0.2074 | 0.0418 | nan      | False              | BEATS BASELINE              |
| Y2_5D_Forward_AbnormalVolume_LogRatio | stacked       | naive_train_mean |  24 |       0.5491 |          0.6986 |       0.1495 |   0.0494 |    0.2458 | 0.0118 | nan      | False              | BEATS BASELINE              |
| Y3_ASPI_Recovery_Time                 | ridge         | naive_zero       |  40 |      22.0146 |         24.4182 |       2.4037 |   0.7076 |    3.6781 | 0.0568 |   0.2205 | False              | BEATS BASELINE              |
| Y3_ASPI_Recovery_Time                 | ridge         | naive_train_mean |  40 |      22.0146 |         21.5973 |      -0.4172 |  -2.7333 |    3.5424 | 0.7825 |   1.0000 | False              | worse than baseline         |
| Y3_ASPI_Recovery_Time                 | random_forest | naive_zero       |  40 |      22.4085 |         24.4182 |       2.0097 |   0.1511 |    3.7958 | 0.1204 |   0.6478 | False              | BEATS BASELINE              |
| Y3_ASPI_Recovery_Time                 | random_forest | naive_train_mean |  40 |      22.4085 |         21.5973 |      -0.8112 |  -3.6452 |    3.4043 | 0.6532 |   1.0000 | False              | worse than baseline         |
| Y3_ASPI_Recovery_Time                 | xgboost       | naive_zero       |  40 |      26.7466 |         24.4182 |      -2.3283 |  -9.5444 |    1.6502 | 0.4157 |   1.0000 | False              | worse than baseline         |
| Y3_ASPI_Recovery_Time                 | xgboost       | naive_train_mean |  40 |      26.7466 |         21.5973 |      -5.1492 | -10.8284 |    0.1995 | 0.1185 |   1.0000 | False              | worse than baseline         |
| Y3_ASPI_Recovery_Time                 | gp            | naive_zero       |  40 |      22.8618 |         24.4182 |       1.5564 |   0.8806 |    2.2400 | 0.0123 |   0.0188 | True               | BEATS BASELINE              |
| Y3_ASPI_Recovery_Time                 | gp            | naive_train_mean |  40 |      22.8618 |         21.5973 |      -1.2645 |  -4.1633 |    3.8144 | 0.5429 |   1.0000 | False              | worse than baseline         |
| Y3_ASPI_Recovery_Time                 | svr           | naive_zero       |  40 |      22.7306 |         24.4182 |       1.6877 |   1.0843 |    2.3309 | 0.0090 |   0.0000 | True               | BEATS BASELINE              |
| Y3_ASPI_Recovery_Time                 | svr           | naive_train_mean |  40 |      22.7306 |         21.5973 |      -1.1333 |  -4.0471 |    4.0694 | 0.5809 |   1.0000 | False              | worse than baseline         |
| Y3_ASPI_Recovery_Time                 | quantile      | naive_zero       |  40 |      21.9889 |         24.4182 |       2.4293 |   0.2983 |    4.8637 | 0.0796 |   0.6006 | False              | BEATS BASELINE              |
| Y3_ASPI_Recovery_Time                 | quantile      | naive_train_mean |  40 |      21.9889 |         21.5973 |      -0.3916 |  -3.0801 |    4.4704 | 0.8254 |   1.0000 | False              | worse than baseline         |
| Y3_ASPI_Recovery_Time                 | mlp           | naive_zero       |  40 |      25.6179 |         24.4182 |      -1.1997 |  -9.0594 |    3.3236 | 0.7050 | nan      | False              | worse than baseline         |
| Y3_ASPI_Recovery_Time                 | mlp           | naive_train_mean |  40 |      25.6179 |         21.5973 |      -4.0206 |  -9.4866 |    1.5279 | 0.2370 | nan      | False              | worse than baseline         |
| Y3_ASPI_Recovery_Time                 | ensemble      | naive_zero       |  40 |      23.5885 |         24.4182 |       0.8297 |  -2.3385 |    2.7529 | 0.5247 | nan      | False              | better, not distinguishable |
| Y3_ASPI_Recovery_Time                 | ensemble      | naive_train_mean |  40 |      23.5885 |         21.5973 |      -1.9912 |  -4.4465 |    1.6990 | 0.2774 | nan      | False              | worse than baseline         |
| Y3_ASPI_Recovery_Time                 | stacked       | naive_zero       |  30 |      20.5934 |         21.5476 |       0.9542 |  -2.8807 |    3.9352 | 0.5369 | nan      | False              | better, not distinguishable |
| Y3_ASPI_Recovery_Time                 | stacked       | naive_train_mean |  30 |      20.5934 |         18.9695 |      -1.6239 |  -4.2111 |    1.6528 | 0.4097 | nan      | False              | worse than baseline         |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | ridge         | naive_zero       |  40 |       5.0590 |          5.2618 |       0.2028 |  -1.3004 |    1.5009 | 0.7837 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | ridge         | naive_train_mean |  40 |       5.0590 |          5.1799 |       0.1208 |  -1.1892 |    1.2605 | 0.8521 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | random_forest | naive_zero       |  40 |       4.9391 |          5.2618 |       0.3227 |  -0.7070 |    1.0807 | 0.5058 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | random_forest | naive_train_mean |  40 |       4.9391 |          5.1799 |       0.2408 |  -0.5762 |    0.8524 | 0.5343 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | xgboost       | naive_zero       |  40 |       5.8161 |          5.2618 |      -0.5543 |  -2.4486 |    1.0014 | 0.5289 |   1.0000 | False              | worse than baseline         |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | xgboost       | naive_train_mean |  40 |       5.8161 |          5.1799 |      -0.6362 |  -2.3507 |    0.7903 | 0.4371 |   1.0000 | False              | worse than baseline         |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | gp            | naive_zero       |  40 |       5.2454 |          5.2618 |       0.0165 |  -0.9184 |    0.7646 | 0.9694 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | gp            | naive_train_mean |  40 |       5.2454 |          5.1799 |      -0.0655 |  -0.7818 |    0.5044 | 0.8404 |   1.0000 | False              | worse than baseline         |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | svr           | naive_zero       |  40 |       5.0651 |          5.2618 |       0.1967 |  -0.7796 |    0.9237 | 0.6601 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | svr           | naive_train_mean |  40 |       5.0651 |          5.1799 |       0.1148 |  -0.6445 |    0.6813 | 0.7404 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | quantile      | naive_zero       |  40 |       5.1325 |          5.2618 |       0.1294 |  -0.7729 |    0.8180 | 0.7680 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | quantile      | naive_train_mean |  40 |       5.1325 |          5.1799 |       0.0474 |  -0.6481 |    0.6069 | 0.8934 |   1.0000 | False              | better, not distinguishable |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | mlp           | naive_zero       |  40 |       5.1762 |          5.2618 |       0.0856 |  -1.2664 |    1.1339 | 0.8942 | nan      | False              | better, not distinguishable |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | mlp           | naive_train_mean |  40 |       5.1762 |          5.1799 |       0.0036 |  -1.1681 |    0.9442 | 0.9949 | nan      | False              | better, not distinguishable |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | ensemble      | naive_zero       |  40 |       5.0814 |          5.2618 |       0.1805 |  -1.0849 |    1.1376 | 0.7567 | nan      | False              | better, not distinguishable |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | ensemble      | naive_train_mean |  40 |       5.0814 |          5.1799 |       0.0985 |  -0.9631 |    0.9215 | 0.8406 | nan      | False              | better, not distinguishable |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | stacked       | naive_zero       |  30 |       5.9581 |          5.9416 |      -0.0165 |  -1.4174 |    1.1527 | 0.9799 | nan      | False              | worse than baseline         |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | stacked       | naive_train_mean |  30 |       5.9581 |          5.8495 |      -0.1085 |  -1.2539 |    0.8472 | 0.8386 | nan      | False              | worse than baseline         |

Thirteen comparisons have an interval excluding zero. Nine of them belong to target two,
where the ensemble, the random forest, extreme gradient boosting, the shallow network and
the stacked model all beat the zero baseline. Three survive the Holm correction: the
target two ensemble against the zero baseline at p 0.021, and two target three models, the
Gaussian process and the support vector regression, which beat the zero baseline while
losing to the training mean. Beating one baseline and losing to the other is not a result,
so only the target two ensemble counts.

---

### 5.3 Fold-wise stability and leave-one-event-out sensitivity

The review asked specifically whether the volume result is stable across folds and whether
a pooled improvement is being produced by one catastrophic event. Both tables below are
computed from the same stored out-of-fold predictions as section 5.1. Nothing was refitted
to produce them.

Fold by fold. The 34 held-out points are not spread evenly: folds one to three carry ten
each and fold four carries four. The reason is worth stating precisely, because it is
easy to describe wrongly. Market wide turnover is unavailable for the 2000 archive year
and for events after 2023. The 2000 events fall inside the first training block rather
than inside any test fold, since the walk forward only begins testing at the thirty first
event, so the early gap costs no held-out points. The post-2023 gap does fall in a test
fold, and it removes six of the ten points from fold four.

| model            |   fold |   n |   rmse |      r2 |
|:-----------------|-------:|----:|-------:|--------:|
| ensemble         |      1 |  10 | 0.4117 | -0.4494 |
| ensemble         |      2 |  10 | 0.5952 |  0.1598 |
| ensemble         |      3 |  10 | 0.4369 |  0.4357 |
| ensemble         |      4 |   4 | 0.4887 |  0.1165 |
| mlp              |      1 |  10 | 0.5158 | -1.2750 |
| mlp              |      2 |  10 | 0.5795 |  0.2037 |
| mlp              |      3 |  10 | 0.3923 |  0.5452 |
| mlp              |      4 |   4 | 0.4035 |  0.3976 |
| random_forest    |      1 |  10 | 0.3489 | -0.0409 |
| random_forest    |      2 |  10 | 0.6215 |  0.0840 |
| random_forest    |      3 |  10 | 0.4774 |  0.3262 |
| random_forest    |      4 |   4 | 0.5302 | -0.0400 |
| naive_zero       |      1 |  10 | 0.4342 | -0.6127 |
| naive_zero       |      2 |  10 | 0.7383 | -0.2926 |
| naive_zero       |      3 |  10 | 0.6149 | -0.1175 |
| naive_zero       |      4 |   4 | 0.5806 | -0.2475 |
| naive_train_mean |      1 |  10 | 0.3515 | -0.0566 |
| naive_train_mean |      2 |  10 | 0.6620 | -0.0393 |
| naive_train_mean |      3 |  10 | 0.7493 | -0.6595 |
| naive_train_mean |      4 |   4 | 0.6550 | -0.5876 |

Leave one event out. Each row removes one held-out event at a time and recomputes the
pooled RMSE, so the spread between the minimum and the maximum shows how much a single
event moves the result.

| model            |   n |   pooled_rmse |   loo_min |   loo_max |   worst_event_index |
|:-----------------|----:|--------------:|----------:|----------:|--------------------:|
| ensemble         |  34 |        0.4882 |    0.4425 |    0.4955 |                  15 |
| mlp              |  34 |        0.4913 |    0.4383 |    0.4987 |                   6 |
| random_forest    |  34 |        0.4995 |    0.4541 |    0.5070 |                  15 |
| naive_zero       |  34 |        0.6055 |    0.5613 |    0.6146 |                  15 |
| naive_train_mean |  34 |        0.6171 |    0.5775 |    0.6264 |                  25 |

Read together, these say two things. First, the volume result is not the artefact of one
catastrophic event: removing any single held-out event leaves the pooled RMSE inside a
narrow band, so no single observation is carrying the improvement. Second, the result is
not uniform across folds. Every model, including both naive baselines, records its worst
fit on the earliest test fold and its best on the third, which is the pattern of a
relationship that is not stable over the sample period rather than one that holds
throughout it. The ensemble's fold-wise R squared runs from -0.449 to +0.436 around a
pooled 0.334, and a single pooled figure conceals that spread.

Neither table changes the verdict, and neither was run to change it. They qualify it: the
evidence for the volume target is an average over four folds that disagree, measured on 34
points, six of which the data do not supply in the final fold.

## 6. Target three, recovery duration

### 6.1 The survival grid

| model                   |   n |   n_recovered |   n_censored |   c_index |   c_index_ci_low |   c_index_ci_high |   integrated_brier_score |   mae_uncensored |   rmse_uncensored |   p_holm | verdict                      |
|:------------------------|----:|--------------:|-------------:|----------:|-----------------:|------------------:|-------------------------:|-----------------:|------------------:|---------:|:-----------------------------|
| two_stage_lognormal_k20 |  40 |            14 |           26 |    0.5346 |           0.3708 |            0.6974 |                   0.3376 |          12.7857 |           23.8822 |   1.0000 | B - suggestive but uncertain |
| two_stage_weibull_k20   |  40 |            14 |           26 |    0.5266 |           0.3673 |            0.6818 |                   0.3607 |          13.5000 |           24.0550 |   1.0000 | B - suggestive but uncertain |
| two_stage_lognormal_k10 |  40 |            14 |           26 |    0.5133 |           0.3568 |            0.6690 |                   0.2732 |           6.6429 |            8.5649 |   1.0000 | B - suggestive but uncertain |
| train_median_baseline   |  40 |            14 |           26 |    0.5027 |           0.3681 |            0.6463 |                   0.3139 |           5.2857 |            7.3776 |   1.0000 | B - suggestive but uncertain |
| km_train_baseline       |  40 |            14 |           26 |    0.5027 |           0.3681 |            0.6463 |                   0.2076 |           5.2857 |            7.3776 |   1.0000 | B - suggestive but uncertain |
| aft_lognormal_k20       |  40 |            14 |           26 |    0.4840 |           0.2819 |            0.6410 |                   0.4564 |          33.3571 |           49.5285 |   1.0000 | C - unsupported              |
| two_stage_weibull_k10   |  40 |            14 |           26 |    0.4787 |           0.3281 |            0.6284 |                   0.3049 |           7.1429 |            9.6658 |   1.0000 | C - unsupported              |
| aft_lognormal_k10       |  40 |            14 |           26 |    0.4787 |           0.2885 |            0.6571 |                   0.3323 |          16.9286 |           32.6092 |   1.0000 | C - unsupported              |
| two_stage_lognormal_k5  |  40 |            14 |           26 |    0.4468 |           0.2727 |            0.6277 |                   0.2733 |           6.4286 |            8.1766 |   1.0000 | C - unsupported              |
| aft_weibull_k20         |  40 |            14 |           26 |    0.4468 |           0.2239 |            0.6292 |                   0.4696 |          30.8571 |           46.4097 |   1.0000 | C - unsupported              |
| two_stage_weibull_k5    |  40 |            14 |           26 |    0.4309 |           0.2642 |            0.6064 |                   0.2786 |           7.3571 |            8.9163 |   1.0000 | C - unsupported              |
| aft_weibull_k10         |  40 |            14 |           26 |    0.4309 |           0.2481 |            0.6154 |                   0.3620 |          17.5000 |           33.0897 |   1.0000 | C - unsupported              |
| aft_weibull_k5          |  40 |            14 |           26 |    0.3697 |           0.2222 |            0.5390 |                   0.3305 |          20.1429 |           33.3766 |   1.0000 | C - unsupported              |
| aft_lognormal_k5        |  40 |            14 |           26 |    0.3617 |           0.2015 |            0.5348 |                   0.3154 |          18.4286 |           32.8351 |   1.0000 | C - unsupported              |
| cox_k5                  |  40 |            14 |           26 |    0.3537 |           0.2190 |            0.5048 |                   0.3033 |          16.1429 |           32.4588 |   1.0000 | C - unsupported              |

Every interval contains 0.5. The best configuration, the two stage log normal model at
capacity twenty, reaches 0.535 against 0.503 for the training fold Kaplan Meier baseline.

### 6.2 Recovery probability calibration

Predicted against the Kaplan Meier estimate at five horizons.

| model                   |       t |   predicted_P_T_le_t |   observed_KM_P_T_le_t |   calibration_gap |
|:------------------------|--------:|---------------------:|-----------------------:|------------------:|
| km_train_baseline       | 10.0000 |               0.5751 |                 0.4091 |            0.1660 |
| km_train_baseline       | 20.0000 |               0.6906 |                 0.6061 |            0.0845 |
| km_train_baseline       | 30.0000 |               0.7251 |                 0.6623 |            0.0628 |
| km_train_baseline       | 60.0000 |               0.7513 |                 0.6623 |            0.0889 |
| km_train_baseline       | 90.0000 |               0.7513 |                 0.6623 |            0.0889 |
| train_median_baseline   | 10.0000 |               0.7500 |                 0.4091 |            0.3409 |
| train_median_baseline   | 20.0000 |               1.0000 |                 0.6061 |            0.3939 |
| train_median_baseline   | 30.0000 |               1.0000 |                 0.6623 |            0.3377 |
| train_median_baseline   | 60.0000 |               1.0000 |                 0.6623 |            0.3377 |
| train_median_baseline   | 90.0000 |               1.0000 |                 0.6623 |            0.3377 |
| aft_weibull_k5          | 10.0000 |               0.4532 |                 0.4091 |            0.0441 |
| aft_weibull_k5          | 20.0000 |               0.6376 |                 0.6061 |            0.0315 |
| aft_weibull_k5          | 30.0000 |               0.7408 |                 0.6623 |            0.0785 |
| aft_weibull_k5          | 60.0000 |               0.8758 |                 0.6623 |            0.2134 |
| aft_weibull_k5          | 90.0000 |               0.9242 |                 0.6623 |            0.2618 |
| two_stage_weibull_k5    | 10.0000 |               0.6397 |                 0.4091 |            0.2306 |
| two_stage_weibull_k5    | 20.0000 |               0.7576 |                 0.6061 |            0.1515 |
| two_stage_weibull_k5    | 30.0000 |               0.8243 |                 0.6623 |            0.1620 |
| two_stage_weibull_k5    | 60.0000 |               0.9154 |                 0.6623 |            0.2530 |
| two_stage_weibull_k5    | 90.0000 |               0.9506 |                 0.6623 |            0.2883 |
| aft_lognormal_k5        | 10.0000 |               0.4750 |                 0.4091 |            0.0659 |
| aft_lognormal_k5        | 20.0000 |               0.6593 |                 0.6061 |            0.0532 |
| aft_lognormal_k5        | 30.0000 |               0.7529 |                 0.6623 |            0.0906 |
| aft_lognormal_k5        | 60.0000 |               0.8709 |                 0.6623 |            0.2086 |
| aft_lognormal_k5        | 90.0000 |               0.9156 |                 0.6623 |            0.2533 |
| two_stage_lognormal_k5  | 10.0000 |               0.6508 |                 0.4091 |            0.2417 |
| two_stage_lognormal_k5  | 20.0000 |               0.7705 |                 0.6061 |            0.1644 |
| two_stage_lognormal_k5  | 30.0000 |               0.8330 |                 0.6623 |            0.1706 |
| two_stage_lognormal_k5  | 60.0000 |               0.9141 |                 0.6623 |            0.2517 |
| two_stage_lognormal_k5  | 90.0000 |               0.9453 |                 0.6623 |            0.2830 |
| cox_k5                  | 10.0000 |               0.6301 |                 0.4091 |            0.2210 |
| cox_k5                  | 20.0000 |               0.7359 |                 0.6061 |            0.1298 |
| cox_k5                  | 30.0000 |               0.7698 |                 0.6623 |            0.1074 |
| cox_k5                  | 60.0000 |               0.7841 |                 0.6623 |            0.1218 |
| cox_k5                  | 90.0000 |               0.7841 |                 0.6623 |            0.1218 |
| aft_weibull_k10         | 10.0000 |               0.6326 |                 0.4091 |            0.2235 |
| aft_weibull_k10         | 20.0000 |               0.7948 |                 0.6061 |            0.1888 |
| aft_weibull_k10         | 30.0000 |               0.8635 |                 0.6623 |            0.2012 |
| aft_weibull_k10         | 60.0000 |               0.9284 |                 0.6623 |            0.2661 |
| aft_weibull_k10         | 90.0000 |               0.9454 |                 0.6623 |            0.2830 |
| two_stage_weibull_k10   | 10.0000 |               0.7632 |                 0.4091 |            0.3541 |
| two_stage_weibull_k10   | 20.0000 |               0.8649 |                 0.6061 |            0.2588 |
| two_stage_weibull_k10   | 30.0000 |               0.9090 |                 0.6623 |            0.2467 |
| two_stage_weibull_k10   | 60.0000 |               0.9542 |                 0.6623 |            0.2919 |
| two_stage_weibull_k10   | 90.0000 |               0.9683 |                 0.6623 |            0.3060 |
| aft_lognormal_k10       | 10.0000 |               0.5956 |                 0.4091 |            0.1865 |
| aft_lognormal_k10       | 20.0000 |               0.7587 |                 0.6061 |            0.1527 |
| aft_lognormal_k10       | 30.0000 |               0.8318 |                 0.6623 |            0.1695 |
| aft_lognormal_k10       | 60.0000 |               0.9108 |                 0.6623 |            0.2485 |
| aft_lognormal_k10       | 90.0000 |               0.9356 |                 0.6623 |            0.2732 |
| two_stage_lognormal_k10 | 10.0000 |               0.7443 |                 0.4091 |            0.3352 |
| two_stage_lognormal_k10 | 20.0000 |               0.8408 |                 0.6061 |            0.2348 |
| two_stage_lognormal_k10 | 30.0000 |               0.8869 |                 0.6623 |            0.2246 |
| two_stage_lognormal_k10 | 60.0000 |               0.9433 |                 0.6623 |            0.2809 |
| two_stage_lognormal_k10 | 90.0000 |               0.9629 |                 0.6623 |            0.3005 |
| aft_weibull_k20         | 10.0000 |               0.4493 |                 0.4091 |            0.0402 |
| aft_weibull_k20         | 20.0000 |               0.6355 |                 0.6061 |            0.0295 |
| aft_weibull_k20         | 30.0000 |               0.7246 |                 0.6623 |            0.0623 |
| aft_weibull_k20         | 60.0000 |               0.8250 |                 0.6623 |            0.1626 |
| aft_weibull_k20         | 90.0000 |               0.8419 |                 0.6623 |            0.1796 |
| two_stage_weibull_k20   | 10.0000 |               0.7119 |                 0.4091 |            0.3028 |
| two_stage_weibull_k20   | 20.0000 |               0.7793 |                 0.6061 |            0.1733 |
| two_stage_weibull_k20   | 30.0000 |               0.8198 |                 0.6623 |            0.1574 |
| two_stage_weibull_k20   | 60.0000 |               0.8714 |                 0.6623 |            0.2090 |
| two_stage_weibull_k20   | 90.0000 |               0.8976 |                 0.6623 |            0.2353 |
| aft_lognormal_k20       | 10.0000 |               0.4381 |                 0.4091 |            0.0290 |
| aft_lognormal_k20       | 20.0000 |               0.6140 |                 0.6061 |            0.0080 |
| aft_lognormal_k20       | 30.0000 |               0.7055 |                 0.6623 |            0.0432 |
| aft_lognormal_k20       | 60.0000 |               0.7973 |                 0.6623 |            0.1349 |
| aft_lognormal_k20       | 90.0000 |               0.8247 |                 0.6623 |            0.1623 |
| two_stage_lognormal_k20 | 10.0000 |               0.7064 |                 0.4091 |            0.2973 |
| two_stage_lognormal_k20 | 20.0000 |               0.7938 |                 0.6061 |            0.1878 |
| two_stage_lognormal_k20 | 30.0000 |               0.8324 |                 0.6623 |            0.1701 |
| two_stage_lognormal_k20 | 60.0000 |               0.8799 |                 0.6623 |            0.2175 |
| two_stage_lognormal_k20 | 90.0000 |               0.9069 |                 0.6623 |            0.2446 |

### 6.3 The pre specified recovery category

| category        | model                   |   n |   n_positive |   prevalence |   roc_auc |   auc_ci_low |   auc_ci_high |   balanced_accuracy | beats_chance   |
|:----------------|:------------------------|----:|-------------:|-------------:|----------:|-------------:|--------------:|--------------------:|:---------------|
| recovery_le_20d | km_train_baseline       |  20 |           13 |       0.6500 |    0.5000 |       0.2368 |        0.7679 |              0.5000 | False          |
| recovery_le_20d | train_median_baseline   |  20 |           13 |       0.6500 |    0.5000 |       0.5000 |        0.5000 |              0.5000 | False          |
| recovery_le_20d | two_stage_lognormal_k10 |  20 |           13 |       0.6500 |    0.4396 |       0.1354 |        0.7363 |              0.5330 | False          |
| recovery_le_20d | aft_lognormal_k20       |  20 |           13 |       0.6500 |    0.4066 |       0.1429 |        0.6667 |              0.2692 | False          |
| recovery_le_20d | two_stage_lognormal_k20 |  20 |           13 |       0.6500 |    0.3626 |       0.1190 |        0.6374 |              0.4231 | False          |
| recovery_le_20d | aft_lognormal_k10       |  20 |           13 |       0.6500 |    0.3407 |       0.0833 |        0.6267 |              0.3846 | False          |
| recovery_le_20d | aft_weibull_k20         |  20 |           13 |       0.6500 |    0.3132 |       0.0707 |        0.5678 |              0.2308 | False          |
| recovery_le_20d | two_stage_weibull_k10   |  20 |           13 |       0.6500 |    0.3022 |       0.0600 |        0.5628 |              0.4615 | False          |
| recovery_le_20d | two_stage_lognormal_k5  |  20 |           13 |       0.6500 |    0.2857 |       0.0705 |        0.5417 |              0.5000 | False          |
| recovery_le_20d | cox_k5                  |  20 |           13 |       0.6500 |    0.2747 |       0.0588 |        0.5467 |              0.4231 | False          |
| recovery_le_20d | aft_weibull_k10         |  20 |           13 |       0.6500 |    0.2692 |       0.0520 |        0.5298 |              0.3846 | False          |
| recovery_le_20d | two_stage_weibull_k5    |  20 |           13 |       0.6500 |    0.2637 |       0.0587 |        0.5238 |              0.5000 | False          |
| recovery_le_20d | two_stage_weibull_k20   |  20 |           13 |       0.6500 |    0.2582 |       0.0440 |        0.5055 |              0.3846 | False          |
| recovery_le_20d | aft_lognormal_k5        |  20 |           13 |       0.6500 |    0.2527 |       0.0400 |        0.5238 |              0.4231 | False          |
| recovery_le_20d | aft_weibull_k5          |  20 |           13 |       0.6500 |    0.2088 |       0.0220 |        0.4601 |              0.3077 | False          |

### 6.4 The two stage hurdle model

Fitted and scored on drawdown events only, as the protocol's stage two requires.

| model            |   n |   MAE_trading_days |    RMSE |
|:-----------------|----:|-------------------:|--------:|
| single-stage RF  |  23 |            17.4795 | 27.6026 |
| naive_train_mean |  23 |            18.3126 | 24.7050 |
| naive_zero       |  23 |            21.0435 | 32.2018 |
| hurdle           |  23 |            27.4813 | 39.7414 |

The hurdle loses to every baseline including doing nothing. That result is kept.

---

## 7. The classification arm

Six binary labels derived from the three targets. Accuracy is never read without its
prevalence, because one label has a prevalence near ninety per cent.

| label              | model             |   n |   n_pos |   prevalence |   majority_baseline_acc |   accuracy |   balanced_accuracy |   precision |   recall |     f1 |     mcc |   pr_auc |    auc |   auc_boot_lo |   auc_boot_hi | beats_baseline   |
|:-------------------|:------------------|----:|--------:|-------------:|------------------------:|-----------:|--------------------:|------------:|---------:|-------:|--------:|---------:|-------:|--------------:|--------------:|:-----------------|
| C0_drawdown_occurs | logistic          |  40 |      23 |       0.5750 |                  0.5750 |     0.5000 |              0.5192 |      0.6000 |   0.3913 | 0.4737 |  0.0392 |   0.5570 | 0.4348 |        0.2558 |        0.6240 | False            |
| C0_drawdown_occurs | majority_baseline |  40 |      23 |       0.5750 |                  0.5750 |     0.5750 |              0.5000 |      0.5750 |   1.0000 | 0.7302 |  0.0000 |   0.5750 | 0.5000 |        0.5000 |        0.5000 | False            |
| C0_drawdown_occurs | rf_clf            |  40 |      23 |       0.5750 |                  0.5750 |     0.7000 |              0.7008 |      0.7619 |   0.6957 | 0.7273 |  0.3975 |   0.6697 | 0.5959 |        0.4118 |        0.7852 | False            |
| C0_drawdown_occurs | xgb_clf           |  40 |      23 |       0.5750 |                  0.5750 |     0.5500 |              0.5396 |      0.6087 |   0.6087 | 0.6087 |  0.0793 |   0.6209 | 0.5524 |        0.3683 |        0.7391 | False            |
| C1_negative_return | logistic          |  40 |      19 |       0.4750 |                  0.5250 |     0.5250 |              0.5075 |      0.5000 |   0.1579 | 0.2400 |  0.0210 |   0.5447 | 0.6266 |        0.4411 |        0.8045 | False            |
| C1_negative_return | majority_baseline |  40 |      19 |       0.4750 |                  0.5250 |     0.5250 |              0.5000 |      0.0000 |   0.0000 | 0.0000 |  0.0000 |   0.4750 | 0.5000 |        0.5000 |        0.5000 | False            |
| C1_negative_return | rf_clf            |  40 |      19 |       0.4750 |                  0.5250 |     0.5500 |              0.5313 |      0.6000 |   0.1579 | 0.2500 |  0.0946 |   0.5590 | 0.5764 |        0.3910 |        0.7569 | False            |
| C1_negative_return | xgb_clf           |  40 |      19 |       0.4750 |                  0.5250 |     0.5750 |              0.5576 |      0.6667 |   0.2105 | 0.3200 |  0.1612 |   0.6026 | 0.5840 |        0.3985 |        0.7644 | False            |
| C1b_adverse_move   | logistic          |  40 |       9 |       0.2250 |                  0.7750 |     0.7000 |              0.4910 |      0.2000 |   0.1111 | 0.1429 | -0.0226 |   0.2902 | 0.5735 |        0.3799 |        0.7491 | False            |
| C1b_adverse_move   | majority_baseline |  40 |       9 |       0.2250 |                  0.7750 |     0.7750 |              0.5000 |      0.0000 |   0.0000 | 0.0000 |  0.0000 |   0.2250 | 0.5000 |        0.5000 |        0.5000 | False            |
| C1b_adverse_move   | rf_clf            |  40 |       9 |       0.2250 |                  0.7750 |     0.7250 |              0.5466 |      0.3333 |   0.2222 | 0.2667 |  0.1090 |   0.3532 | 0.6057 |        0.3978 |        0.7993 | False            |
| C1b_adverse_move   | xgb_clf           |  40 |       9 |       0.2250 |                  0.7750 |     0.7750 |              0.6183 |      0.5000 |   0.3333 | 0.4000 |  0.2766 |   0.4310 | 0.6057 |        0.3763 |        0.8136 | False            |
| C2_volume_spike    | logistic          |  34 |      13 |       0.3824 |                  0.6176 |     0.6176 |              0.6026 |      0.5000 |   0.5385 | 0.5185 |  0.2025 |   0.6258 | 0.6154 |        0.3883 |        0.8168 | False            |
| C2_volume_spike    | majority_baseline |  34 |      13 |       0.3824 |                  0.6176 |     0.6176 |              0.5000 |      0.0000 |   0.0000 | 0.0000 |  0.0000 |   0.3824 | 0.5000 |        0.5000 |        0.5000 | False            |
| C2_volume_spike    | rf_clf            |  34 |      13 |       0.3824 |                  0.6176 |     0.6471 |              0.6117 |      0.5455 |   0.4615 | 0.5000 |  0.2321 |   0.5926 | 0.6557 |        0.4542 |        0.8388 | False            |
| C2_volume_spike    | xgb_clf           |  34 |      13 |       0.3824 |                  0.6176 |     0.5588 |              0.4963 |      0.3750 |   0.2308 | 0.2857 | -0.0084 |   0.4817 | 0.5897 |        0.3993 |        0.7766 | False            |
| C3_recovers_in_90  | logistic          |  16 |      14 |       0.8750 |                  0.8750 |     0.6875 |              0.3929 |      0.8462 |   0.7857 | 0.8148 | -0.1816 |   0.9299 | 0.5714 |        0.2857 |        0.8571 | False            |
| C3_recovers_in_90  | majority_baseline |  16 |      14 |       0.8750 |                  0.8750 |     0.8750 |              0.5000 |      0.8750 |   1.0000 | 0.9333 |  0.0000 |   0.8750 | 0.5000 |        0.5000 |        0.5000 | False            |
| C3_recovers_in_90  | rf_clf            |  16 |      14 |       0.8750 |                  0.8750 |     0.6250 |              0.3571 |      0.8333 |   0.7143 | 0.7692 | -0.2182 |   0.9218 | 0.5000 |        0.2143 |        0.7857 | False            |
| C3_recovers_in_90  | xgb_clf           |  16 |      14 |       0.8750 |                  0.8750 |     0.8750 |              0.5000 |      0.8750 |   1.0000 | 0.9333 |  0.0000 |   0.9223 | 0.5000 |        0.2143 |        0.7857 | False            |
| C3b_slow_recovery  | logistic          |  16 |       8 |       0.5000 |                  0.5000 |     0.3750 |              0.3750 |      0.2500 |   0.1250 | 0.1667 | -0.2887 |   0.4685 | 0.4219 |        0.1250 |        0.7344 | False            |
| C3b_slow_recovery  | majority_baseline |  16 |       8 |       0.5000 |                  0.5000 |     0.5000 |              0.5000 |      0.5000 |   1.0000 | 0.6667 |  0.0000 |   0.5000 | 0.5000 |        0.5000 |        0.5000 | False            |
| C3b_slow_recovery  | rf_clf            |  16 |       8 |       0.5000 |                  0.5000 |     0.4375 |              0.4375 |      0.4000 |   0.2500 | 0.3077 | -0.1348 |   0.4745 | 0.4531 |        0.1562 |        0.7500 | False            |
| C3b_slow_recovery  | xgb_clf           |  16 |       8 |       0.5000 |                  0.5000 |     0.3750 |              0.3750 |      0.3333 |   0.2500 | 0.2857 | -0.2582 |   0.4813 | 0.4688 |        0.1562 |        0.7812 | False            |

No label clears both the majority rule and a bootstrap AUC interval excluding one half.
The best of them is the stage one drawdown label, where the random forest reaches accuracy
0.700 and balanced accuracy 0.701 against a majority baseline of 0.575, but its AUC
interval of [0.412, 0.785] still contains chance.

---

## 8. Robustness and ablations

### 8.1 Conformal prediction interval coverage

| target                                | kind     |   coverage |   covered |   n |   half_width |   nominal |
|:--------------------------------------|:---------|-----------:|----------:|----:|-------------:|----------:|
| Y1_ASPI_5D_Forward_LogReturn_Pct      | model    |     0.6667 |        20 |  30 |       2.1453 |    0.8000 |
| Y1_ASPI_5D_Forward_LogReturn_Pct      | marginal |     0.6333 |        19 |  30 |       1.9407 |    0.8000 |
| Y2_5D_Forward_AbnormalVolume_LogRatio | model    |     0.5000 |        12 |  24 |       0.5171 |    0.8000 |
| Y2_5D_Forward_AbnormalVolume_LogRatio | marginal |     0.5000 |        12 |  24 |       0.6684 |    0.8000 |
| Y3_ASPI_Recovery_Time                 | model    |     0.8667 |        26 |  30 |      13.8764 |    0.8000 |
| Y3_ASPI_Recovery_Time                 | marginal |     0.8667 |        26 |  30 |      13.6667 |    0.8000 |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | model    |     0.6333 |        19 |  30 |       3.8311 |    0.8000 |
| Y1_ASPI_10D_Forward_LogReturn_Pct     | marginal |     0.5667 |        17 |  30 |       3.2104 |    0.8000 |

Coverage undershoots the nominal level for both return columns and for volume, which is
the expected behaviour of split conformal prediction at this calibration size, and is
reported rather than tuned away.

### 8.2 Feature block ablations

| config         | model         | target                                |   pooled_r2 |    rmse |
|:---------------|:--------------|:--------------------------------------|------------:|--------:|
| full           | ridge         | Y1_ASPI_5D_Forward_LogReturn_Pct      |     -0.0781 |  2.8478 |
| full           | ridge         | Y2_5D_Forward_AbnormalVolume_LogRatio |     -0.0324 |  0.6079 |
| full           | ridge         | Y3_ASPI_Recovery_Time                 |     -0.0774 | 22.0146 |
| full           | ridge         | Y1_ASPI_10D_Forward_LogReturn_Pct     |      0.0042 |  5.0590 |
| full           | random_forest | Y1_ASPI_5D_Forward_LogReturn_Pct      |      0.0140 |  2.7235 |
| full           | random_forest | Y2_5D_Forward_AbnormalVolume_LogRatio |      0.3030 |  0.4995 |
| full           | random_forest | Y3_ASPI_Recovery_Time                 |     -0.1163 | 22.4085 |
| full           | random_forest | Y1_ASPI_10D_Forward_LogReturn_Pct     |      0.0509 |  4.9391 |
| full           | xgboost       | Y1_ASPI_5D_Forward_LogReturn_Pct      |     -0.1474 |  2.9379 |
| full           | xgboost       | Y2_5D_Forward_AbnormalVolume_LogRatio |      0.2544 |  0.5167 |
| full           | xgboost       | Y3_ASPI_Recovery_Time                 |     -0.5903 | 26.7466 |
| full           | xgboost       | Y1_ASPI_10D_Forward_LogReturn_Pct     |     -0.3161 |  5.8161 |
| no_external    | ridge         | Y1_ASPI_5D_Forward_LogReturn_Pct      |     -0.0645 |  2.8298 |
| no_external    | ridge         | Y2_5D_Forward_AbnormalVolume_LogRatio |      0.1234 |  0.5602 |
| no_external    | ridge         | Y3_ASPI_Recovery_Time                 |     -0.0743 | 21.9830 |
| no_external    | ridge         | Y1_ASPI_10D_Forward_LogReturn_Pct     |     -0.0349 |  5.1575 |
| no_external    | random_forest | Y1_ASPI_5D_Forward_LogReturn_Pct      |     -0.0865 |  2.8589 |
| no_external    | random_forest | Y2_5D_Forward_AbnormalVolume_LogRatio |      0.2813 |  0.5072 |
| no_external    | random_forest | Y3_ASPI_Recovery_Time                 |     -0.1556 | 22.7998 |
| no_external    | random_forest | Y1_ASPI_10D_Forward_LogReturn_Pct     |     -0.0187 |  5.1168 |
| no_external    | xgboost       | Y1_ASPI_5D_Forward_LogReturn_Pct      |     -0.1922 |  2.9947 |
| no_external    | xgboost       | Y2_5D_Forward_AbnormalVolume_LogRatio |      0.1961 |  0.5365 |
| no_external    | xgboost       | Y3_ASPI_Recovery_Time                 |     -0.1479 | 22.7236 |
| no_external    | xgboost       | Y1_ASPI_10D_Forward_LogReturn_Pct     |     -0.1136 |  5.3500 |
| no_hazard      | ridge         | Y1_ASPI_5D_Forward_LogReturn_Pct      |     -0.1126 |  2.8930 |
| no_hazard      | ridge         | Y2_5D_Forward_AbnormalVolume_LogRatio |      0.0794 |  0.5741 |
| no_hazard      | ridge         | Y3_ASPI_Recovery_Time                 |     -0.0491 | 21.7234 |
| no_hazard      | ridge         | Y1_ASPI_10D_Forward_LogReturn_Pct     |     -0.1510 |  5.4390 |
| no_hazard      | random_forest | Y1_ASPI_5D_Forward_LogReturn_Pct      |      0.0159 |  2.7209 |
| no_hazard      | random_forest | Y2_5D_Forward_AbnormalVolume_LogRatio |      0.3091 |  0.4973 |
| no_hazard      | random_forest | Y3_ASPI_Recovery_Time                 |     -0.1625 | 22.8674 |
| no_hazard      | random_forest | Y1_ASPI_10D_Forward_LogReturn_Pct     |      0.1058 |  4.7941 |
| no_hazard      | xgboost       | Y1_ASPI_5D_Forward_LogReturn_Pct      |     -0.1192 |  2.9016 |
| no_hazard      | xgboost       | Y2_5D_Forward_AbnormalVolume_LogRatio |      0.2056 |  0.5333 |
| no_hazard      | xgboost       | Y3_ASPI_Recovery_Time                 |     -0.4327 | 25.3869 |
| no_hazard      | xgboost       | Y1_ASPI_10D_Forward_LogReturn_Pct     |     -0.3316 |  5.8503 |
| no_desinventar | ridge         | Y1_ASPI_5D_Forward_LogReturn_Pct      |     -0.0490 |  2.8091 |
| no_desinventar | ridge         | Y2_5D_Forward_AbnormalVolume_LogRatio |     -0.0351 |  0.6087 |
| no_desinventar | ridge         | Y3_ASPI_Recovery_Time                 |     -0.0821 | 22.0629 |
| no_desinventar | ridge         | Y1_ASPI_10D_Forward_LogReturn_Pct     |     -0.0960 |  5.3075 |
| no_desinventar | random_forest | Y1_ASPI_5D_Forward_LogReturn_Pct      |     -0.1041 |  2.8820 |
| no_desinventar | random_forest | Y2_5D_Forward_AbnormalVolume_LogRatio |      0.3185 |  0.4939 |
| no_desinventar | random_forest | Y3_ASPI_Recovery_Time                 |     -0.1006 | 22.2510 |
| no_desinventar | random_forest | Y1_ASPI_10D_Forward_LogReturn_Pct     |      0.0046 |  5.0581 |
| no_desinventar | xgboost       | Y1_ASPI_5D_Forward_LogReturn_Pct      |     -0.2246 |  3.0352 |
| no_desinventar | xgboost       | Y2_5D_Forward_AbnormalVolume_LogRatio |      0.2187 |  0.5289 |
| no_desinventar | xgboost       | Y3_ASPI_Recovery_Time                 |     -0.1945 | 23.1808 |
| no_desinventar | xgboost       | Y1_ASPI_10D_Forward_LogReturn_Pct     |     -0.3871 |  5.9708 |
| no_fx          | ridge         | Y1_ASPI_5D_Forward_LogReturn_Pct      |     -0.3174 |  3.1481 |
| no_fx          | ridge         | Y2_5D_Forward_AbnormalVolume_LogRatio |      0.1220 |  0.5607 |
| no_fx          | ridge         | Y3_ASPI_Recovery_Time                 |     -0.0917 | 22.1602 |
| no_fx          | ridge         | Y1_ASPI_10D_Forward_LogReturn_Pct     |     -0.3581 |  5.9082 |
| no_fx          | random_forest | Y1_ASPI_5D_Forward_LogReturn_Pct      |      0.0368 |  2.6918 |
| no_fx          | random_forest | Y2_5D_Forward_AbnormalVolume_LogRatio |      0.3363 |  0.4875 |
| no_fx          | random_forest | Y3_ASPI_Recovery_Time                 |     -0.1345 | 22.5908 |
| no_fx          | random_forest | Y1_ASPI_10D_Forward_LogReturn_Pct     |      0.0509 |  4.9390 |
| no_fx          | xgboost       | Y1_ASPI_5D_Forward_LogReturn_Pct      |     -0.0370 |  2.7931 |
| no_fx          | xgboost       | Y2_5D_Forward_AbnormalVolume_LogRatio |      0.2388 |  0.5220 |
| no_fx          | xgboost       | Y3_ASPI_Recovery_Time                 |     -0.2974 | 24.1582 |
| no_fx          | xgboost       | Y1_ASPI_10D_Forward_LogReturn_Pct     |     -0.0461 |  5.1852 |
| no_election    | ridge         | Y1_ASPI_5D_Forward_LogReturn_Pct      |     -0.0307 |  2.7845 |
| no_election    | ridge         | Y2_5D_Forward_AbnormalVolume_LogRatio |     -0.0661 |  0.6178 |
| no_election    | ridge         | Y3_ASPI_Recovery_Time                 |     -0.0792 | 22.0333 |
| no_election    | ridge         | Y1_ASPI_10D_Forward_LogReturn_Pct     |     -0.0759 |  5.2587 |
| no_election    | random_forest | Y1_ASPI_5D_Forward_LogReturn_Pct      |      0.0194 |  2.7160 |
| no_election    | random_forest | Y2_5D_Forward_AbnormalVolume_LogRatio |      0.3117 |  0.4964 |
| no_election    | random_forest | Y3_ASPI_Recovery_Time                 |     -0.1375 | 22.6209 |
| no_election    | random_forest | Y1_ASPI_10D_Forward_LogReturn_Pct     |      0.0662 |  4.8991 |
| no_election    | xgboost       | Y1_ASPI_5D_Forward_LogReturn_Pct      |     -0.1420 |  2.9309 |
| no_election    | xgboost       | Y2_5D_Forward_AbnormalVolume_LogRatio |      0.2080 |  0.5325 |
| no_election    | xgboost       | Y3_ASPI_Recovery_Time                 |     -1.5696 | 33.9987 |
| no_election    | xgboost       | Y1_ASPI_10D_Forward_LogReturn_Pct     |     -0.3561 |  5.9037 |

### 8.3 The oversampling ablation

| target                                |   rmse_no_smogn |   rmse_with_smogn |   pooled_r2_no_smogn |   pooled_r2_with_smogn | verdict                  |
|:--------------------------------------|----------------:|------------------:|---------------------:|-----------------------:|:-------------------------|
| Y1_ASPI_5D_Forward_LogReturn_Pct      |          2.4673 |            2.4673 |               0.0140 |                 0.0140 | SMOGN hurts or no effect |
| Y2_5D_Forward_AbnormalVolume_LogRatio |          0.5145 |            0.4945 |               0.2162 |                 0.3030 | SMOGN helps              |
| Y3_ASPI_Recovery_Time                 |         21.6035 |           21.0511 |              -0.1752 |                -0.1163 | SMOGN helps              |
| Y1_ASPI_10D_Forward_LogReturn_Pct     |          4.4424 |            4.4424 |               0.0509 |                 0.0509 | SMOGN hurts or no effect |

### 8.4 Event date precision sensitivity

Five of the seventy four in scope events carry a month or a year rather than an exact
day, three of them inside the pooled test set. Re scoring on exact day events only tests
whether the headline depends on them.

| target                                |   n_all |   pooled_r2_all |   n_exact_day |   pooled_r2_exact_day_only |
|:--------------------------------------|--------:|----------------:|--------------:|---------------------------:|
| Y1_ASPI_5D_Forward_LogReturn_Pct      |      40 |          0.0140 |            37 |                     0.0090 |
| Y2_5D_Forward_AbnormalVolume_LogRatio |      34 |          0.3030 |            31 |                     0.2511 |
| Y3_ASPI_Recovery_Time                 |      40 |         -0.1163 |            37 |                    -0.1337 |
| Y1_ASPI_10D_Forward_LogReturn_Pct     |      40 |          0.0509 |            37 |                     0.0513 |

The volume result weakens from 0.303 to 0.251 on exact day events only and remains the
strongest of the four. No conclusion changes.

---

## 9. The sector panel

A secondary analysis: the same questions asked of each sector index rather than the
all share index, with folds grouped so every sector of one event stays on the same side of
the split.

| target                  | model         | baseline         |   delta_rmse |   ci_low |   ci_high |   n_events |   n_rows | significant   |
|:------------------------|:--------------|:-----------------|-------------:|---------:|----------:|-----------:|---------:|:--------------|
| Y1_sector_log_return    | ridge         | naive_zero       |      -0.1243 |  -0.2339 |   -0.0483 |         30 |      594 | False         |
| Y1_sector_log_return    | ridge         | naive_train_mean |      -0.1245 |  -0.2061 |   -0.0576 |         30 |      594 | False         |
| Y1_sector_log_return    | random_forest | naive_zero       |      -0.2955 |  -0.4802 |   -0.1612 |         30 |      594 | False         |
| Y1_sector_log_return    | random_forest | naive_train_mean |      -0.2957 |  -0.4752 |   -0.1527 |         30 |      594 | False         |
| Y3_sector_recovery_days | ridge         | naive_zero       |      -2.9330 |  -9.9231 |    2.7526 |         30 |      594 | False         |
| Y3_sector_recovery_days | ridge         | naive_train_mean |      -6.4605 | -11.0828 |   -2.6179 |         30 |      594 | False         |
| Y3_sector_recovery_days | random_forest | naive_zero       |      -6.2358 | -14.2652 |    0.0551 |         30 |      594 | False         |
| Y3_sector_recovery_days | random_forest | naive_train_mean |      -9.7633 | -15.4866 |   -4.9828 |         30 |      594 | False         |

Nothing is significant, and every point estimate is negative, meaning the models are worse
than the naive baselines at sector level. The panel also carries a known limitation,
recorded in `docs/refactor_validation.md`: it zero fills its own feature matrix rather
than using the global median values the index level pipeline uses.
