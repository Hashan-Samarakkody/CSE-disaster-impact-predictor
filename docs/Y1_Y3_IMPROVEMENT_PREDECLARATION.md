# Y1 / Y3 improvement pre-declaration

**Written 2026-09-17, BEFORE any experiment below was executed.**
Baseline commit: `1fbf6275` (`git rev-parse HEAD` at freeze time).
Frozen baseline artifact: `artifacts/frozen_baseline.json` (`scripts/freeze_baseline.py`).
Test suite at freeze: **99 passed** (`pytest tests/ -q`), plus the 20 new Y2-freeze
assertions in `tests/test_y2_frozen.py`.

This document exists so that the experiment list cannot grow after a disappointing
result. Everything that will be run is listed here first. Anything added later is
explicitly labelled **post-hoc exploratory** in `docs/Y1_Y3_FINAL_RESULTS.md` and is not
allowed to carry a confirmatory claim.

---

## 0. What is frozen and may not move

* **Y2 in full** — target, event sample, prediction origin, folds, features, models,
  statistics, results. Enforced mechanically by `tests/test_y2_frozen.py`
  (target values, event sample, fold definitions, pooled OOF predictions, per-fold
  RMSE/MAE/R2, pooled R2, bootstrap verdict + CI + Holm flag, and the C2 classification
  AUC / balanced accuracy). A failure is a stop-and-diagnose signal.
* Inclusion threshold `total_affected >= 1000` (`FeatureEngineeringConfig.min_affected`).
* The 74-event sample. No event is added, removed, or re-dated for performance reasons.
* `RANDOM_STATE = 42` everywhere. Seeds are not re-rolled.
* The outer walk-forward geometry: `train=30, test=10, step=10`, chronological.
* Y3's 5-session adverse-response gate and its competing-event censoring rule
  (`T_obs = min(T_recovery, T_next_disaster, 90)`, `Y3_censored`, `Y3_censor_reason`).
* The verdict rule: paired episode-clustered bootstrap (10,000 draws), CI excluding zero,
  Holm-corrected p reported as a secondary strict diagnostic.

## 1. Baseline numbers being improved on (from `docs/RESULTS_AUDIT.txt`)

| Target | Best pooled R2 | Model | Bootstrap verdict |
|---|---|---|---|
| Y1 (5-day) | +0.143 | mlp | no comparison's CI excludes zero |
| Y1 (10-day) | +0.169 | mlp | no comparison's CI excludes zero |
| Y2 | +0.270 | gp | **BEATS BASELINE** (gp, svr, both baselines) |
| Y3 (point regression) | +0.155 | ensemble | 5 rows beat `naive_zero` only |

## 2. Y1 — pre-declared experiment grid

### 2.1 Horizons (exactly four; no fifth will be added)

`h` in {5, 10, 15, 20} CSE **trading sessions**, defined as
`Y_h = 100 * ln(P_{t+h} / P_{t-1})` where `t` is the first valid trading session on/after
the disaster date and `P_{t-1}` is the last valid pre-event close.
Columns: `Y1_ASPI_EventWindow_0_{h}_LogReturn_Pct` plus
`Y1_ASPI_EventWindow_0_{h}_horizon_end_date` for target-specific purging.
**h=5 is the principal target.** h=10/15/20 are pre-specified horizon-sensitivity
analyses, reported whatever they show.

### 2.2 Information sets (identical folds, identical test events)

* **A — market-only**: strictly pre-event market-state columns only.
* **B — disaster-only**: disaster / severity / hazard / exposure columns only.
* **C — combined**: A union B.
* **D — normal-market + disaster-residual**: a Stage-A expected-return model
  `R_hat_normal,h` estimated on the **daily market series using only rows strictly before
  the event**, then a Stage-B model of `R_residual,h = R_obs,h - R_hat_normal,h` on
  disaster features, with `R_hat_final,h = R_hat_normal,h + R_hat_residual,h`.
  The Stage-A model for an outer test event sees no data dated on/after that event.

The **primary scientific comparison is C vs A** — does disaster-specific information add
anything beyond the market's own pre-event state.

### 2.3 Feature capacity

`K` in {5, 10} pre-specified; `K = 20` retained only as the already-frozen robustness
comparison. Feature-selection stability is reported per (horizon, information set) as
`selection_frequency`, `mean_rank`, `median_rank` across folds. A feature selected in one
fold only is reported as such and is not described as a finding.

### 2.4 Models (closed list)

Ridge / ElasticNet (`StandardScaler` + `RidgeCV`/`ElasticNetCV` inside the purged inner
temporal CV), Random Forest, XGBoost, shallow MLP. **No algorithm will be added.**
Existing repository models (GP, SVR, quantile regression, ensemble, stacked) stay in the
appendix exactly as already reported.

### 2.5 Direction, as a separate question

`negative_return = Y_h < 0`, per horizon: balanced accuracy, ROC-AUC, PR-AUC, MCC,
sensitivity, specificity, bootstrap AUC CI. Reported as a **different question** from
magnitude regression; a classification success is never presented as a regression success.

### 2.6 Augmentation: SMOGN held OFF, uniformly

The new Y1 grid runs with **SMOGN off for every configuration**. This is a
held-constant factor, not a result-driven change: the grid's whole purpose is the
paired A-vs-C and horizon comparisons, and an augmentation whose minority mask is
defined from each fold's own target distribution would vary across the 4 horizons and
3 information sets, confounding exactly the contrasts being measured. SMOGN's effect is
already measured separately and reported (`artifacts/smogn_ablation.parquet`, ablation
B6, protocol section 5), and the existing frozen Y1/Y2/Y3 pipeline keeps it ON and is
untouched by this grid.

### 2.7 Validation

Outer: chronological walk-forward `train=30, test=10, step=10`.
Per horizon, `training label end < first test prediction origin` — so h=20 carries a
20-session target embargo, enforced through that horizon's own
`*_horizon_end_date` column. Inner: purged temporal CV (`purged_inner_cv`).
If purging makes an inner split impossible: **reduce the number of inner splits**; if
still impossible, fall back to pre-specified default hyperparameters. Leakage protection
is never removed.

## 3. Y3 — pre-declared experiment grid

* **Primary analysis is censoring-aware and uses genuine rows only.** No SMOGN synthetic
  rows enter the survival likelihood: SMOGN interpolates a duration but cannot generate a
  valid event/censoring indicator, so a synthetic `(38, observed)` pair asserts a
  recovery that was never seen.
* **Two-stage architecture**: Stage 1 estimates `P(Drawdown = 1 | X)` from the existing
  `Y3_drawdown_occurred` gate; Stage 2 models `T_recovery` for drawdown cases with a
  censoring-aware AFT fit.
* **Survival models**: Weibull AFT, Log-normal AFT, and penalized Cox PH *only if* the
  effective event count supports it without instability. No Random Survival Forest.
* **Outputs**: predicted median recovery `T_hat_0.50` **and** recovery probabilities
  `P(T <= 10), P(T <= 20), P(T <= 30), P(T <= 60), P(T <= 90)`.
* **Metrics**: Harrell C-index (primary), Integrated Brier Score where feasible,
  recovery-probability calibration; secondary — MAE / median absolute error / RMSE
  computed **among uncensored recoveries only**. Ordinary RMSE over censored rows treated
  as point values is not a primary metric.
* **Recovery categories: exactly two, pre-specified** — `recovery <= 20 trading days`
  (an interpretable "recovers within a trading month") and the **existing**
  `C3b_slow_recovery` classifier, kept unchanged. No threshold sweep.

## 4. Statistics (Y1 and Y3)

* Paired evaluation: every candidate and baseline predicts **exactly the same** test rows.
* Episode-cluster bootstrap retained (`build_episode_ids`, 14-day chaining).
* Baselines — Y1: naive zero return, training-fold mean, **market-only expected-return
  model**. Y3: training-fold survival baseline (Kaplan-Meier), training-fold median
  recovery.
* Holm family-wise correction across the whole Y1/Y3 results family, reported as a
  **secondary strict diagnostic** alongside the single-comparison CI, never folded into it.

## 5. Pre-declared ablation register

| ID | Ablation |
|---|---|
| A1 | Y1 market-only |
| A2 | Y1 disaster-only |
| A3 | Y1 combined |
| A4 | Y1 normal-market + disaster-residual |
| A5 | Y1 K=5 features |
| A6 | Y1 K=10 features |
| A7 | Y1 K=20 (robustness only) |
| A8 | Y1 5-day horizon |
| A9 | Y1 10-day horizon |
| A10 | Y1 15-day horizon |
| A11 | Y1 20-day horizon |
| B1 | Y3 current point regression (existing result, unchanged) |
| B2 | Y3 AFT survival, real rows only |
| B3 | Y3 two-stage drawdown + AFT |
| B4 | Y3 Weibull vs Log-normal AFT |
| B5 | Y3 survival with compact features |
| B6 | Y3 SMOGN vs no-SMOGN diagnostic (existing `smogn_ablation.parquet`) |

## 6. Decision rule, declared before any result is read

* **A — statistically supported**: out-of-sample performance beats its pre-specified
  baseline AND the uncertainty interval excludes the null.
* **B — suggestive but uncertain**: point estimate improves, interval overlaps the null.
* **C — unsupported**: no consistent improvement over baseline.

A **B** is never re-labelled **A** because an R2 looks attractive. A non-significant
final result is retained and reported.

## 7. Stop condition

After this grid executes: no further horizon, no further algorithm, no threshold change,
no target change, no feature-count increase, no observation removal, no additional
variant search. The pipeline freezes.
