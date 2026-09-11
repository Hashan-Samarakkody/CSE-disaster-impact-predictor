# A Machine Learning Approach to Predicting the Impact of Natural Disasters on the Colombo Stock Exchange

A multi-target machine learning study of natural disaster impacts on Sri Lanka's Colombo Stock Exchange. It jointly models ASPI log return, abnormal trading volume, and market recovery time across 64 real EM-DAT-qualifying disasters (2000 – Jun 2023), with SHAP-based explainability, for a data-scarce frontier market.

**Scope, stated precisely.** This is an **ex-post impact-attribution** study, not an early-warning system. Several of its features are EM-DAT damage assessments finalised weeks to months after an event, so the model cannot be run prospectively on the day before a disaster. It answers *given a disaster of known severity, what was the market's response* — not *what will tomorrow's disaster do*.

## Headline result

**Magnitude is not predictable. Direction, on one target, is.**

Every regression comparison was tested against two naive baselines — a constant-zero
"no measurable effect" null and a training-fold mean — using a paired event-level
bootstrap and a Diebold–Mariano test. On all three targets the best model's advantage
sits inside its own confidence interval:

| Target | Best model | ΔRMSE vs best null | 95% CI | Verdict |
|---|---|---|---|---|
| Y1 ASPI log return | random forest | +0.00098 | [−0.00013, +0.00258] | not distinguishable |
| Y2 abnormal volume | ridge | +0.0615 | [−0.0275, +0.1480] | not distinguishable |
| Y3 recovery days | random forest | +0.232 | [−0.641, +0.820] | not distinguishable |

Ridge reaches pooled R² = +0.131 on abnormal volume, the best value in the study — and
30 out-of-fold test points cannot separate it from noise. It is reported as measured and
**not** claimed as a result.

The same data answers a binary question that it cannot answer as a regression. Asked
whether trading volume will exceed its own 30-day baseline, three independent model
families clear both the majority rule and chance:

| `C2_volume_spike` | AUC [95% CI] | balanced acc. | precision | recall | F1 |
|---|---|---|---|---|---|
| logistic | 0.794 [0.615, 0.973] | 0.699 | 0.529 | 0.818 | 0.643 |
| random forest | 0.789 [0.609, 0.970] | 0.727 | 1.000 | 0.455 | 0.625 |
| majority rule | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |

Direction on Y1 remains indistinguishable from chance (best AUC 0.655, CI includes 0.5),
and recovery-within-90-days scores *below* chance on balanced accuracy.

**The finding: at N = 64 there is enough information to call whether volume responds, but
not enough to predict by how much, and none at all for index-level price response.** For a
market the thesis itself characterises as thin and semi-strong-inefficient, that is an
interpretable result — index-level aggregation and low liquidity absorb localised physical
shocks.

## What makes the negative result credible

- 100% real data: no synthetic events, no relaxed inclusion criteria, no gap-filling
- Leakage-free chronological walk-forward throughout; no k-fold anywhere
- Every threshold and capacity bound fixed a priori, never chosen on held-out scores
- Two naive baselines on every comparison, with confidence intervals on every claim
- Bugs found during development documented rather than concealed — including a test-set
  leak in the ensemble blend and a missing volume feature block, both found by audit and
  both fixed here

## Layout

```
disaster_finance_predictor/
  notebooks/     01 acquire -> 02 features -> 03 EDA -> 04 regression -> 05 classification
                 -> 06 evaluation -> 07 SHAP -> 08 sector panel -> 09 synthesis
  src/           data_pipeline, sampling, models, training, evaluation
  tests/         34 tests, no data files or notebook run required
docs/
  METHODOLOGY_AUDIT.md    full methodology audit, P0-P3 register, experiment matrix
  THESIS_AMENDMENTS.md    20 required corrections to the thesis text
  figures/                exported at 300 dpi for citation
```

See [notebooks/README.md](disaster_finance_predictor/notebooks/README.md) for the stage
order and the artifact cache that makes re-running cheap.
