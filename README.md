# A Machine Learning Approach to Predicting the Impact of Natural Disasters on the Colombo Stock Exchange

A multi-target machine learning study of natural disaster impacts on Sri Lanka's Colombo
Stock Exchange. It jointly models ASPI log return, abnormal trading volume, and market
recovery time across **74 real EM-DAT-qualifying disasters (Sep 2000 – Nov 2025)**, with
SHAP-based explainability, for a data-scarce frontier market.

**Scope, stated precisely.** This is an **ex-post impact-attribution** study, not an
early-warning system. Several of its features are EM-DAT damage assessments finalised
weeks to months after an event, so the model cannot be run prospectively on the day
before a disaster. It answers *given a disaster of known severity, what was the market's
response* — not *what will tomorrow's disaster do*.

## Headline result

**Magnitude is not predictable. Direction, on one target, is.**

Every regression comparison was tested against two naive baselines — a constant-zero
"no measurable effect" null and a training-fold mean — using a paired event-level
bootstrap and a Diebold–Mariano test, on 40 pooled out-of-fold points. On all three
targets the best model's advantage sits inside its own confidence interval:

| Target | Best model | ΔRMSE vs best null | 95% CI | Verdict |
|---|---|---|---|---|
| Y1 ASPI log return | mlp | +0.00057 | [−0.00140, +0.00186] | not distinguishable |
| Y2 abnormal volume | ensemble | +0.0651 | [−0.0101, +0.1333] | not distinguishable |
| Y3 recovery days | random forest | +0.227 | [−0.223, +0.629] | not distinguishable |

The same data answers a binary question that it cannot answer as a regression. Asked
whether trading volume will exceed its own 30-day baseline, a random forest clears both
the majority rule and chance:

| `C2_volume_spike` (n=34) | AUC [95% CI] | balanced acc. | precision | recall | F1 |
|---|---|---|---|---|---|
| **random forest** | **0.764 [0.578, 0.929]** | 0.629 | 0.714 | 0.357 | 0.476 |
| logistic | 0.664 [0.454, 0.857] | 0.682 | 0.588 | 0.714 | 0.645 |
| majority rule | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |

Direction on Y1 remains indistinguishable from chance, and recovery-within-90-days
scores *below* chance on AUC.

**The finding: at N = 74 there is enough information to call whether volume responds, but
not enough to predict by how much, and none at all for index-level price response.** For a
market the thesis itself characterises as thin and semi-strong-inefficient, that is an
interpretable result — index-level aggregation and low liquidity absorb localised physical
shocks.

## External data, and one prediction that failed

Five sources were added after a missing-data audit found `financial_damage` real for only
17 of 64 events: DesInventar Sendai (district-level physical severity), NASA POWER (daily
precipitation and wind at 7 district points), FRED `DEXSLUS` (daily LKR/USD),
countryeconomy.com (ASPI beyond the archive's 2023 cutoff, taking N from 64 to 74), and
Wikidata election dates. Every feature was specified in
[`docs/EXTERNAL_DATA_PRE_DECLARATION.md`](docs/EXTERNAL_DATA_PRE_DECLARATION.md) **before
any of them was scored**, including five written-down expectations.

The block ablation says the external data measurably improves Y2 — pooled R² moves from
−0.089 to +0.191 (random forest) — with the whole-block contribution's CI excluding zero
on two of three models. **The block doing the work is the exchange rate**, which the
pre-declaration had expected to "do little on its own". That prediction is recorded as
wrong rather than quietly rewritten.

This does not make Y2 predictable. Adding FX significantly improves the model *relative
to a model without it*; the model still does not beat a constant. Both statements are
reported together.

## What makes the negative result credible

- 100% real data: no synthetic events, no relaxed inclusion criteria, no gap-filling
- External features pre-declared in writing before the run that scored them, with the
  expectations graded afterwards including the one that was wrong
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
  tests/         56 tests, no data files or notebook run required
docs/
  METHODOLOGY_AUDIT.md    full methodology audit, P0-P3 register, experiment matrix
  EXTERNAL_DATA_PRE_DECLARATION.md  the 5 external sources, 18 features, ablation outcome
  THESIS_AMENDMENTS.md    29 required corrections to the thesis text
  figures/                exported at 300 dpi for citation
```

See [notebooks/README.md](disaster_finance_predictor/notebooks/README.md) for the stage
order and the artifact cache that makes re-running cheap.
