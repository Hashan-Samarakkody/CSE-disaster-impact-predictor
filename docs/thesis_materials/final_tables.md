## Y1 final table (best configuration per horizon x information set)

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


Full 240-row grid: `docs/thesis_materials/final_table_y1.csv`.


## Y1 direction analysis (secondary, a different question)

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


## Y3 final table

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


## Y3 recovery category `recovery <= 20 trading days`

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