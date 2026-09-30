# Statistical Testing Guide: Step-by-Step for All Targets

This document explains how statistical tests were performed for each prediction target in the CSE disaster impact study. It covers the equations used, the bootstrap procedure, and the decision rules applied.

---

## 1. Common Framework: Episode-Clustered Paired Bootstrap

All primary comparisons use the same bootstrap engine. The unit of resampling is a **disaster episode** (events within 14 calendar days are grouped into one episode), not individual events. This accounts for serial dependence between clustered disasters.

### Step-by-step procedure (applies to all targets):

1. **Collect paired predictions**: For each held-out event *i*, record both the model's prediction and the baseline's prediction alongside the realized outcome.

2. **Compute the pooled statistic** on the original sample:
   - For regression targets (Y1 magnitude, Y2 volume): compute RMSE for both model and baseline, then take the difference:
     ```
     Δ RMSE = RMSE_baseline - RMSE_model
     ```
     Positive Δ means the model is better.

   - For classification (Y1 direction): compute AUC directly.

   - For survival (Y3 recovery): compute Harrell's Concordance Index (C-index).

3. **Group events into episodes**: Any events whose prediction origins fall within 14 calendar days of each other form one episode. The 40 held-out events yield ~38 episodes.

4. **Resample episodes with replacement** (not individual events):
   - Draw B = 10,000 bootstrap replicates (B = 5,000 for concordance due to tied-risk filtering).
   - In each replicate *r*, sample episodes with replacement, unpack all events within each drawn episode, then recompute the statistic.

5. **Build the 95% confidence interval** from the bootstrap distribution:
   ```
   CI_95 = [Q_0.025({Θ*_r}), Q_0.975({Θ*_r})]
   ```
   where Q denotes the percentile of the bootstrap distribution.

6. **Decision rule**: If the CI excludes zero (for Δ RMSE) or excludes 0.50 (for AUC/C-index), the comparison is flagged as significant at the 95% level *before* multiple-comparison correction.

---

## 2. Holm Familywise Correction

Because multiple models and configurations are compared, a Holm-Bonferroni correction is applied to control the familywise error rate.

### Steps:

1. **Count the comparisons in the family**: For Y1 five-session return magnitude, there are m = 18 confirmatory comparisons (3 models x 2 information sets x 3 feature capacities).

2. **Compute a bootstrap tail probability** for each comparison:
   ```
   p = (number of bootstrap replicates where Δ ≤ 0) / B
   ```
   This is the one-sided p-value testing whether the model improves on the baseline.

3. **Sort the p-values** from smallest to largest:
   ```
   p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
   ```

4. **Apply the Holm step-down rule**: Reject H_(j) in order while:
   ```
   p_(j) ≤ α / (m - j + 1)
   ```
   where α = 0.05. For m = 18, the first comparison must beat 0.05/18 ≈ 0.00278.

5. **Compute adjusted p-values**:
   ```
   p_adj(j) = min[1, max_{i≤j} {(m - i + 1) × p_(i)}]
   ```

6. **Decision**: Reject H0 only if the Holm-adjusted p < 0.05.

---

## 3. Target 1: ASPI Five-Session Return Magnitude

### What is being tested
Whether any ML model predicts five-session forward ASPI log return better than the market-only expected-return baseline.

### Metric
RMSE (Root Mean Squared Error) on 40 held-out events across 4 walk-forward folds.

### Baseline
Market-only expected return: `E(R) = α + β × R_SP500`, estimated from the training fold's 250-session pre-event window. This baseline uses no disaster information.

### Test statistic
```
Δ RMSE = RMSE_baseline - RMSE_model
```
Positive = model better.

### Procedure
1. Pool all 40 out-of-fold predictions.
2. Compute RMSE for the model and RMSE for the baseline.
3. Compute Δ RMSE = baseline RMSE - model RMSE.
4. Run 10,000 episode-clustered bootstrap replicates (seed = 20260923).
5. Extract the 2.5th and 97.5th percentiles for the 95% CI.
6. Apply Holm correction across the 18-comparison family.

### Result
All 18 intervals included zero. Smallest Holm-adjusted p = 0.641. **H0 not rejected.**

### Example from the thesis (Table A5)
| Model | RMSE_model | RMSE_baseline | Δ RMSE | 95% CI |
|-------|-----------|--------------|--------|--------|
| Ridge (real-time) | 2.776 | 2.872 | +0.096 | [-0.143, +0.389] |

---

## 4. Target 1 (Secondary): Ten-Session Return Direction

### What is being tested
Whether a logistic classifier can predict the *direction* (negative vs. non-negative) of the 10-session forward return better than chance (AUC > 0.50).

### Metric
- **AUC** (Area Under the ROC Curve) - primary
- **Balanced Accuracy** and **Matthews Correlation Coefficient (MCC)** - secondary

### Baseline
Random chance: AUC = 0.50.

### Test statistic
AUC computed on the 40 held-out event predictions, with the label defined as:
```
negative = 1  if  Y1_10D_Forward_LogReturn ≤ 0.00
negative = 0  otherwise
```

### Procedure
1. Pool all 40 out-of-fold predicted probabilities from the logistic classifier.
2. Compute AUC against the binary negative-return label.
3. Run 10,000 episode-clustered bootstrap replicates.
4. Check if the 95% CI excludes 0.50.
5. For Balanced Accuracy and MCC: 3,000 episode bootstrap replicates (seed = 42).
6. Apply Holm correction within the directional comparison family.

### Result
| Model | AUC | 95% CI | Holm p |
|-------|-----|--------|--------|
| Logistic | 0.817 | [0.657, 0.940] | 0.000 |
| Random Forest | 0.709 | [0.529, 0.874] | 0.140 |

Only logistic regression survived Holm correction. **H0 rejected for logistic direction only.**

---

## 5. Target 2: Five-Session Abnormal Trading Volume

### What is being tested
Whether the Random Forest predicts five-session abnormal volume (log ratio) better than a zero baseline.

### Metric
RMSE on 34 held-out events (13 of 74 events lack volume labels).

### Baselines
- **Zero baseline**: predicts Δ volume = 0 (no change from pre-event baseline)
- **Training-fold mean**: predicts the average volume ratio from the training data

### Test statistic
```
Δ RMSE = RMSE_zero - RMSE_model
```

### Procedure
1. Pool all 34 scored out-of-fold predictions (Fold 4 has only 4 labeled events).
2. Compute RMSE for Random Forest and for the zero baseline.
3. Compute Δ RMSE.
4. Run 10,000 episode-clustered bootstrap replicates.
5. Only compute fold-level intervals for folds with n ≥ 8.
6. Apply Holm correction.

### Result
| Model | n | Δ RMSE | 95% CI | Holm p |
|-------|---|--------|--------|--------|
| Random Forest | 34 | +0.106 | [+0.037, +0.175] | 0.078 |

The interval excludes zero, but Holm-adjusted p = 0.078 > 0.05. **Suggestive only; H0 not rejected.**

### Influence sensitivity
Removing the most influential event (2019-9389-LKA, 2019-08-01) reduces Δ from +0.106 to +0.083; recalculated CI [+0.023, +0.135] still excludes zero on 33 events.

---

## 6. Target 3: Market Recovery Duration

### What is being tested
Whether survival models (Weibull AFT, Cox PH) predict recovery duration better than a nonparametric Kaplan-Meier baseline.

### Key complication
Recovery is a **right-censored time-to-event outcome**. Standard RMSE cannot be used directly because:
- 22 of 74 events had no drawdown (structural zeros, not recoveries)
- 16 of 52 drawdown events were censored (next disaster arrived or 90-session cap hit)

### Two-stage structure
```
P(D=d, T≤t | x) = {
    1 - π(x),                      if d=0 (no drawdown)
    π(x) × f_{T|D=1,x}(t) dt,     if d=1 (drawdown occurred)
}
```
Stage 1 predicts drawdown occurrence (classification), Stage 2 predicts duration conditional on drawdown (survival).

### Metrics
1. **Harrell's Concordance Index (C-index)**: proportion of comparable event pairs correctly ranked by the model. C = 0.50 means no better than chance.
   ```
   C = Σ_{i<j} I(comparable) × [I(concordant) + 0.5 × I(tied)] / Σ_{i<j} I(comparable)
   ```

2. **Integrated Brier Score (IBS)**: average calibration error across five horizons (10, 20, 30, 60, 90 sessions), weighted by inverse probability of censoring (IPCW).

3. **AUC for Stage 1** (drawdown classifier): same episode-bootstrap as direction.

### Procedure
1. Fit Weibull AFT and Cox PH on training folds (52 drawdown events, ~36 observed recoveries).
2. Generate survival curves for the 40 held-out events (23 with drawdown).
3. Compute out-of-fold C-index using the joint survival function:
   ```
   S_joint(t|x) = π(x) × S(t|D=1, x)
   ```
4. Run 2,000 episode-clustered bootstrap replicates for C-index intervals.
5. Compute 5-point IPCW Brier score at t = {10, 20, 30, 60, 90} sessions.
6. Compare model C-index and Brier against the training-fold KM baseline.

### Result
| Model | n | C-index | 95% CI | 5-pt Brier |
|-------|---|---------|--------|------------|
| Weibull AFT k5 | 40 | 0.370 | [0.222, 0.539] | 0.331 |
| Cox PH k5 | 40 | 0.354 | [0.219, 0.505] | 0.303 |
| KM baseline | 40 | 0.503 | [0.368, 0.646] | 0.208 |

All C-index intervals contain 0.50. KM baseline has lowest Brier. **H0 not rejected.**

### Point-regression diagnostic (not primary)
A parallel analysis treated recovery duration as an ordinary regression target (ignoring censoring). All models had higher RMSE than the training-fold mean. These numbers are reported as diagnostics only because censored observations were scored as if they were exact recovery times.

---

## 7. Summary of Decision Rules

| Target | Test | Null Reference | Decision Rule | Outcome |
|--------|------|----------------|---------------|---------|
| Y1 magnitude (5-session) | Δ RMSE bootstrap | 0 | CI excludes 0 AND Holm p < 0.05 | Not rejected |
| Y1 direction (10-session) | AUC bootstrap | 0.50 | CI excludes 0.50 AND Holm p < 0.05 | **Rejected** (logistic only) |
| Y2 volume (5-session) | Δ RMSE bootstrap | 0 | CI excludes 0 AND Holm p < 0.05 | Suggestive (CI excludes 0, Holm p = 0.078) |
| Y3 recovery | C-index bootstrap | 0.50 | CI excludes 0.50 | Not rejected |

---

## 8. Equations Reference

| Eq. | Name | Formula | Used For |
|-----|------|---------|----------|
| 15a | Holm ordering | p_(1) ≤ ... ≤ p_(m), m = 18 | Multiple comparison correction |
| 15b | Holm adjusted p | p_adj(j) = min[1, max_{i≤j}{(m-i+1)×p_(i)}] | Adjusted significance |
| 16a | Pooled statistic | Θ* = T({(x_ai, y_ai)}) | Bootstrap point estimate |
| 16b | Bootstrap replicate | Θ*(r) = T(U_r(sample_r(C, [C]))) | Episode-clustered resampling |
| 16c | Confidence interval | CI_95 = [Q_0.025, Q_0.975] over R=10,000 | Uncertainty quantification |
| 17 | MDE approximation | MDE = (t_{1-α/2} + t_{1-β}) × sqrt(2σ²/N) | Power/sensitivity planning |
| 23 | Harrell C-index | C = concordant pairs / comparable pairs | Survival model discrimination |

---

## 9. Key Implementation Details

- **Random seed**: 20260923 for return/volume bootstrap; 42 for direction metrics and information-set comparisons
- **Bootstrap draws**: 10,000 for RMSE comparisons; 5,000 valid draws for concordance; 3,000 for balanced accuracy/MCC
- **Episode definition**: events within 14 calendar days grouped together
- **Fold structure**: 4 outer folds, each with 30 training / 10 test events, sliding forward by 10
- **Purging**: training events whose label end-date ≥ first test origin are removed (Equation 14)
- **Imputation/scaling**: fitted on training fold only, never on test events
- **SMOGN synthetic rows**: training folds only, never in test folds
