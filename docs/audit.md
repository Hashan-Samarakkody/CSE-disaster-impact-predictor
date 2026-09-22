# Methodology audit and research record

This is the complete methodology record for the project, gathered into one file so that
every decision, every pre declaration and every negative result can be followed in one
place. Nothing has been removed in the consolidation; the parts below are the previously
separate documents, in the order a reader should meet them.

Read Part 1 first if you want to know what the pipeline does and why. Read Part 7 if you
want to know how it got there, including what was tried and abandoned.

**Naming note for entries written before 2026-09-19.** On that date the three target
definitions were frozen in `docs/TARGET_DEFINITION_PROTOCOL.md`, which renamed two of the
columns and changed their formulas. Entries below that predate the freeze use the earlier
names and the earlier arithmetic, and they are kept verbatim because they are the record
of what was actually run at the time. The mapping is:

| Pre protocol column | Current column | What also changed |
|---|---|---|
| `Y1_ASPI_5D_Forward_LogReturn_Pct` | unchanged | the endpoint moved from `P[t0+5]` to `P5 = market[position + 4]` |
| `Y2_abnormal_volume` | `Y2_5D_Forward_AbnormalVolume_LogRatio` | a ratio minus one became a log ratio |
| `Y3_recovery_days` | `Y3_ASPI_Recovery_Time` | the drawdown gate narrowed to `P1..P5`, the scan starts at `k = 1` rather than the trough, and no drawdown became its own censor reason |
| `Y1_EventWindow_0_10_LogReturn_Pct` | `Y1_ASPI_10D_Forward_LogReturn_Pct` | same alignment change as Y1 |

Any number in this file that predates 2026-09-19 was produced under the earlier
definitions. The current numbers are in `docs/results.md`.

## Contents

1. Part 1. Frozen analysis protocol
2. Part 2. Pre declaration of the return and recovery improvement grid
3. Part 3. External data pre declaration
4. Part 4. Frozen volume target validation
5. Part 5. Method comparison, before and against after
6. Part 6. Thesis amendments
7. Part 7. Full dated change log
8. Part 8. Revision 2 pre-declaration


---

# Part 1. Frozen analysis protocol

The specification the pipeline was frozen against on 2026-09-16. Change it only for a stated reason unrelated to a held out score, and log the change in Part 7.

## Final Analysis Protocol (frozen 2026-09-16, commit `e0cbd2f`)

This document freezes every decision the methodology-audit response (44 numbered
findings, `docs/METHODOLOGY_AUDIT.md`) touched, so the next pipeline run is the **one
last clean run** the panel's own P1 recommendation ("stop adaptive experimentation")
calls for. Everything below is now a specification, not a running log: change it only
for a stated reason unrelated to a held-out score (a data bug, a new event added to
EM-DAT, a reviewer requirement), never because a number looked disappointing. Any
future change must be logged as a new dated section in `METHODOLOGY_AUDIT.md`, exactly
like every change that produced this freeze.

### 1. Inclusion criteria and data

- EM-DAT disasters with `total_affected >= 1000` (frozen finding #2/#3/#34; verified
  byte-identical to the 500/700 thresholds explored earlier,  no event between 500 and
  1000 matches a tradable CSE session).
- 76 raw qualifying disasters -> 74 events after matching to a tradable CSE session
  (2 unmatched).
- `financial_damage` is real `NaN` when EM-DAT reports none, not zero-filled at the
  loader (finding #14),  every consumer downstream imputes it explicitly (§4).

### 2. Targets

| Target | Formula | Units | Support |
|---|---|---|---|
| `Y1_ASPI_5D_Forward_LogReturn_Pct` | `100 * ln(P_{t+5} / P_{t-1})` | percent | unbounded |
| `Y1_EventWindow_0_10_LogReturn_Pct` | `100 * ln(P_{t+10} / P_{t-1})` | percent | unbounded |
| `Y2_abnormal_volume` | `V / mean(V, 30d) - 1` | ratio | `[-1, inf)` |
| `Y3_recovery_days` | trading days to price recovery, right-censored | days | `[0, 90]` |

`Y1_EventWindow_0_5` was retired (finding #8/#9): identical formula to `Y1` once both
were rebaselined to the pre-event close, so it added no information and was consolidated
away, along with the classification label `C4_car5_negative` that mirrored it.

`Y3_recovery_days` carries a competing-risk censoring flag (finding #7):
`Y3_censored` is `True` when recovery is unresolved at the 90-day cap OR at a later
qualifying disaster, with `Y3_censor_reason` in `{"recovered", "cap_90",
"next_disaster"}`. Rows with `"next_disaster"` are excluded from the two
recovery-classification labels (`C3_recovers_in_90`, `C3b_slow_recovery`),  status is
genuinely unknowable there, not a label to guess at.

`Y3_recovery_days` also carries an adverse-response gate (finding #11, methodology-audit
followup 2026-09-16): the recovery clock no longer starts blindly at the event session.
The lowest close within a prespecified 5-trading-day post-event window (or the
competing-risk cap, if shorter) is checked against the pre-event baseline first
(`Y3_drawdown_occurred`); only if the index actually falls below baseline in that window
does the recovery search run, and it runs from that trough, not from event day. Before
this fix, an event whose own close already sat above the pre-event baseline scored
Y3=0 immediately, even if the index fell below baseline a few sessions later within the
same short window,  that later drop was never seen because the search had already
"recovered" on day 0 by construction. `Y3_drawdown_occurred` is a diagnostic column
(excluded from `FEATURE_COLS`, computed from post-event prices), not a feature.

### 3. Features

- 68 columns in `FEATURE_COLS` (`artifacts/results/feature_spec.json`).
- Every engineered market-price column is ADF/KPSS-tested before admission
  (`notebooks/02_features_targets.ipynb` §2.2.1); a unit root (ADF fails to reject)
  excludes the column. This test runs on a **development-period-only** slice of the
  daily market series,  rows before the 31st qualifying disaster's date, i.e. before
  fold 0's own first test period,  not the full ~22-year series (finding #26,
  methodology-audit followup 2026-09-17: the admissibility decision itself must not see
  any row inside any walk-forward test fold). `sma_5/10/20`/`ema_5/10/20` (raw price
  levels) are excluded either way; the fix changed the leakage exposure, not the
  6-column result.
- 16 columns carry real missingness and are median-imputed from TRAIN rows only, never
  a global or zero fill, inside every fold: `financial_damage`, `log_financial_damage`,
  `log_damage_x_flood`, `total_deaths`, `no_homeless`, `mag_area_km2`, `mag_wind_kph`,
  `vol_ratio_1_30`, `vol_ratio_5_30`, `vol_ratio_10_30`, `vol_cv_30`,
  `log_vol_change_1`, `garch_cond_vol`, `gdp_growth_pct`, `inflation_cpi_pct`,
  `damage_to_gdp` (finding #14; `MEDIAN_IMPUTE_COLS` /
  `median_impute_from_train` in `src/training/walk_forward.py`). Each column that can
  be missing also carries its own `*_available`/`*_observed` indicator flag, computed
  from the same fold's train rows.
- Two "no-fold" full-data-refit paths (final SHAP refit, `train_final_models.py`, the
  live demo in `src/models/inference.py`) use a single GLOBAL median instead, persisted once
  in `feature_spec.json`'s `MEDIAN_IMPUTE_VALUES` and reused everywhere rather than
  each recomputing its own.
- Feature selection is always collinearity-drop THEN RF-importance top-20, fit on
  train rows only, in that order (`src/evaluation/collinearity.py::redundant_drop_set`,
  pre-declared `|rho| >= 0.95` rule, composed before ranking): both the regression loop
  (`notebooks/04_modeling_regression.ipynb`) and the classification/hurdle loop
  (`notebooks/05_modeling_classification.ipynb`), plus the full-data refit in
  `scripts/train_final_models.py`. Each target gets its OWN selected feature set,  Y2
  and Y3 are never forced to reuse Y1's ranking.
- **Ridge is the one exception** to RF-importance top-20 (finding #17, methodology-audit
  followup): an RF-importance ranking is not the right selection criterion for a linear
  model, so Ridge fits on the full collinearity-pruned set instead (no top-K cut),
  letting its own L2 regularization do the shrinkage/selection,  the panel's own
  suggested alternative for Ridge/ElasticNet-family models. RF, XGBoost, GP, SVR, and
  Quantile Regression still share the RF-importance top-20 selection; changing their
  selection criterion too is out of scope for this freeze (no obvious single "right"
  model-specific alternative for a kernel method or a boosted-tree ensemble the way
  regularization is for Ridge).
- **Feature selection is nested inside the inner CV** (finding #16, methodology-audit
  followup 2026-09-17): Ridge's alpha search and RF/XGBoost's hyperparameter grids are
  `Pipeline`s (`CollinearityDropper` / `CollinearityRFTopK`,
  `src/evaluation/collinearity.py`) run inside `GridSearchCV(cv=cv_real)` on the FULL
  real-training feature matrix, so the selector is refit independently on every inner
  split and every hyperparameter candidate,  an inner-validation row's own label can no
  longer have quietly influenced which columns even reached the model scored on it. The
  FINAL refit (on all outer training, real+synthetic) still uses one `feat_cols`/
  `ridge_cols` selection per (fold, target), matching the panel's own diagram's next
  step. GP, SVR, and Quantile Regression run no hyperparameter search at all (fixed
  config), so there is no inner-CV loop to nest a selector inside; they still use the
  single outer-selected `feat_cols`. At inner-fold sizes of 9-16-23 rows this selection
  is genuinely noisier than a selection made on the full outer training set would be, 
  that concern (`docs/METHODOLOGY_AUDIT.md` §21) is not wrong and is not retracted, it
  is simply outweighed here by matching the panel's explicit request. Measured effect:
  Ridge's Y2_abnormal_volume pooled R2 improved from -0.612 to -0.047; RF/XGBoost moved
  by low single digits on every target.

### 4. Validation protocol

- Chronological walk-forward, primary configuration `train_window=30,
  test_window=10, step=10` (`generate_walk_forward_splits`,
  `notebooks/02_features_targets.ipynb`).
- A dense ablation (`20/5/5`) and a `k=10`-feature ablation are reported separately as
  robustness checks, never as the headline numbers.
- Every target purges its OWN label-horizon against the fold boundary (finding #5):
  `Y1_horizon_end_date` (5 trading days), `Y1_EventWindow_0_10_horizon_end_date` (10
  days), `Y2_label_end_date` (same session, no forward horizon), `Y3_label_end_date`
  (up to 90 days or the confirmed recovery/censor date). No call site purges against a
  different target's horizon column any more (`TARGET_LABEL_END_DATE_COL` /
  `LABEL_END_DATE_COL` in `notebooks/_shared.py` is the single source of truth).
- Hyperparameter search runs a purged inner `TimeSeriesSplit`
  (`purged_inner_cv`, finding #6) on the fold's real training rows only, one level
  deeper than the outer purge.
- Accepted, disclosed, unfixed: Y3's 90-day forward window can still overlap a
  subsequent event that lands in a later fold. Purging it would cost folds the study
  cannot spare; the overlap count is reported as a limitation instead
  (`docs/METHODOLOGY_AUDIT.md` §20).

### 5. Augmentation (SMOGN)

- Time-aware SMOGN, fit on the fold's real, post-purge training rows only, with the
  pre-registered 25% synthetic-share cap, applied per (fold, target) inside the target
  loop (never once per fold shared across targets).
- SMOGN on/off was measured, not assumed (P2-6, `artifacts/tables/smogn_ablation.parquet`):
  helps Y1/Y2/Y3 on both RMSE and pooled R2; on `Y1_EventWindow_0_10` the two metrics
  move in opposite directions by ~1%, judged as fold-count noise (n=40 pooled points),
  not a real effect. **Decision: SMOGN stays ON for all 4 targets.**
- A previously-tried wider minority mask + 2x synthetic draws was measured and
  reverted (made every model on every target worse; also independently violates the
  25% cap),  recorded as a rejected variant, not deleted.
- Each target now defines its OWN minority/relevance mask (finding #15, methodology-
  audit followup): Y3 keeps its pre-declared `Y3_recovery_days > 30` threshold; Y1 and
  `Y1_EventWindow_0_10` treat the bottom quintile of that fold's training returns
  (most severe negative moves) as the minority class; Y2 treats the top quintile
  (largest volume spikes) as the minority class. Before this fix, every target's SMOGN
  call reused Y3's threshold verbatim, so a slow Y3 recovery,  not an extreme Y1 return
  or Y2 spike,  decided which rows got oversampled for Y1 and Y2. This is NOT the same
  change as the reverted wide-mask variant above: that experiment unioned all three
  targets' tails into ONE shared augmented table; this keeps each target's own,
  already-separate augmented table (unchanged since finding #5's per-target purge),
  just with a mask relevant to that specific target.

### 6. Models

**Primary candidate set** (finding #19/#44, methodology-audit followup: the panel's own
recommendation is "the smallest set that answers the research question",  a strong
linear baseline, RF/XGBoost, one neural comparison, and survival/hurdle for Y3,  rather
than adding a model family every time an existing one disappoints):

- Regression, all 4 targets: **Ridge** (H1's designated linear baseline; own
  collinearity-pruned feature set per finding #17, `RidgeCV`-searched alpha), **Random
  Forest** and **XGBoost** (both grid-searched, RF-importance-selected features), and a
  **shallow multi-task MLP** (the one neural comparison).
- Regression, Y3 only: **AFT survival model** and a **two-stage hurdle model**
  (`HurdleRecoveryModel`), both using the real competing-risk censoring flag. These are
  the **PRIMARY reported result for Y3** (finding #12, methodology-audit followup): a
  plain point-regression fit to `Y3_recovery_days` treats every capped or competing-risk-
  censored row as if it were an observed recovery time, which is exactly the distortion
  survival analysis exists to avoid. Plain RF/Ridge/XGBoost regression on Y3 is reported
  as a **secondary diagnostic**,  it shows the size of that distortion, not a competing
  primary number. See `scripts/run_survival_model.py`'s module docstring.
- Classification: 5 labels (`C1_negative_return`, `C1b_adverse_move`,
  `C2_volume_spike`, `C3_recovers_in_90`, `C3b_slow_recovery`), each with its own
  best-family model selected by walk-forward AUC (`build_classifiers`,
  `classification_summary.parquet`).

**Exploratory / appendix, not primary** (reported, but not what the thesis's headline
claims rest on):

- **Gaussian Process** (Matern + WhiteKernel, no grid) and **SVR** (RBF, fixed
  `C=1.0, epsilon=0.1`, no grid),  added during this session's feature/model-quality
  pass, after the panel's original review. Neither is demoted for a measured failure
  the way Quantile Regression is (below); they simply were never part of the panel's
  recommended minimal set, and adding them without removing anything is exactly the
  "model family proliferation" finding #19/#44 warns against. Reported for completeness,
  not treated as co-equal to the primary set above.
- **Median Quantile Regression** (`alpha=0.01`, `solver="highs"`) is a **known failed
  configuration** at its current fixed alpha (badly miscalibrated on every target), 
  kept in the lineup as a disclosed negative result, not silently dropped, not
  recommended for use until it gets a proper inner-CV alpha search.
- An **inverse-RMSE ensemble blend** and a **stacked meta-learner** over
  {RF, XGBoost, MLP},  reported as combination-method ablations.
- **PCA** is reported as a comparison only (top-10 components + Ridge vs RF-selected
  top-20 + Ridge, same folds),  not adopted as a default; PCA components are
  uninterpretable, which cuts against the SHAP explainability chapter.

### 7. Statistical verdict rule

- A model "BEATS BASELINE" iff the paired event bootstrap CI (10,000 resamples)
  excludes zero. This is the PRIMARY and only gating criterion
  (`src/evaluation/verification.py`).
- The bootstrap resamples by **disaster episode**, not individual event, whenever event
  dates are available (finding #22, methodology-audit followup): events under 14 days
  apart are chained into one episode (`build_episode_ids`) and drawn or withheld
  together every round, since their market outcomes are plausibly correlated rather
  than independent draws. Falls back to plain per-event resampling only when episode
  dates can't be recovered for a given comparison.
- Diebold-Mariano (HLN small-sample corrected) is still computed and reported per row
  as a `dm_agrees` diagnostic column, not a gate,  DM's stationarity/weak-dependence
  assumptions fit this irregular, overlapping-horizon event panel worse than the
  bootstrap's plain exchangeability assumption, so requiring both was demoting a
  real, bootstrap-confirmed effect for the wrong reason.
- A **Holm correction** (finding #20, methodology-audit followup) is applied across
  every row of `verdict_table` to the bootstrap's one-sided p-value, reported as
  `holm_significant`,  a stricter, family-wise-corrected diagnostic column, since many
  (model, target, baseline) comparisons are run and some "significant" result is
  expected by chance alone. `verdict`/`boot_beats` (the single-comparison CI) stay the
  primary criterion, exactly as DM stays a diagnostic rather than a second gate,  this
  is reported alongside the primary verdict, not folded into it, so a single-comparison
  result is never silently reclassified by a family-wide correction whose family
  membership (which comparisons count as "the same family") is itself a judgment call.
- Classification's `beats_baseline` is unrelated to this rule: it requires the AUC's
  own bootstrap CI to exclude 0.5 AND balanced accuracy > 0.5 (majority-rule check),
  computed in `src/models/classifiers.py`.

### 8. Known, accepted, disclosed limitations (not to be "fixed" without new data)

- Leave-one-disaster-type-out is descriptive only (Drought n=5); not a statistical claim.
- Y3's 90-day window can overlap a later event in a subsequent fold; disclosed, not purged.
- Inner-CV feature selection is not nested; accepted at this N (§3 above).
- Quantile regression is under-regularized at its current fixed alpha; a known, disclosed failure, not a bug to silently paper over.
- `notebooks/08_sector_panel.ipynb`'s own `PANEL_FEATURES` classification cell still
  blanket-zero-fills instead of using `MEDIAN_IMPUTE_VALUES`,  a residual gap, flagged,
  not fixed as part of this freeze (low priority: sector panel is a secondary analysis,
  not a headline result).
- **No final lockbox holdout exists** (finding #38, methodology-audit followup). The
  panel's best fix,  reserve the newest chronological events, untouched by any
  ablation or tuning decision, as a genuinely final held-out check,  is not achievable
  at this N: 74 events split across a 30/10/10 walk-forward already spends nearly the
  whole timeline on training folds, and setting aside a further tail would cost the
  study folds it cannot spare (the same tradeoff already accepted for Y3's embargo
  overlap in §8 above). This freeze is the panel's own stated fallback instead: stop
  adaptive experimentation NOW, disclose the adaptive-selection risk explicitly (this
  bullet), and treat every walk-forward number the "one last clean run" below produces
  as carrying that residual risk,  repeated human decisions across this session's many
  ablations, informed by the same out-of-fold results, can produce adaptive overfitting
  even with no single instance of direct test-set leakage. The bootstrap CIs and Holm
  correction in §7 are the conservative response to that risk, not a claim that it has
  been eliminated.
- Exact-date sensitivity (finding #23, methodology-audit followup): 5 of 74 events
  (droughts) carry `month_only` EM-DAT date precision rather than an exact start day.
  `notebooks/06_evaluation.ipynb` §6.2.2 re-scores RF's pooled R2 restricted to the 69
  `exact_day` events, using the same cached out-of-fold predictions (no re-fit), as a
  check on whether the headline number depends on those 5 alignment-uncertain rows.
  Reported in `artifacts/tables/exact_date_sensitivity.parquet`.

### 9. What "one last clean run" means

1. Do not touch target definitions, inclusion threshold, purge columns, SMOGN
   parameters, feature-selection composition, or the verdict rule while looking at
   this run's numbers.
2. Re-run notebooks 01 -> 02 -> 04 -> 05 -> 08 -> 07 -> 06, then
   `scripts/run_survival_model.py` and `scripts/train_final_models.py`, in that order,
   from a clean `artifacts/` state.
3. Run `pytest tests/ -q`,  must be 97/97 (or the then-current full-suite count) before
   any number from this run is reported as final.
4. Report every number from this run, in whichever direction it lands, in
   `docs/RESULTS_AUDIT.txt` / the thesis results chapter. A disappointing result is
   itself a finding this study is equipped to report (see §5, §6 above for two
   already-disclosed examples), not a reason to reopen §1-§7.

---

# Part 2. Pre declaration of the return and recovery improvement grid

Written on 2026-09-17 before any experiment in it was executed, so that the experiment list could not grow after a disappointing result.

## Y1 / Y3 improvement pre-declaration

**Written 2026-09-17, BEFORE any experiment below was executed.**
Baseline commit: `1fbf6275` (`git rev-parse HEAD` at freeze time).
Frozen baseline artifact: `artifacts/results/frozen_baseline.json` (`scripts/freeze_baseline.py`).
Test suite at freeze: **99 passed** (`pytest tests/ -q`), plus the 20 new Y2-freeze
assertions in `tests/test_y2_frozen.py`.

This document exists so that the experiment list cannot grow after a disappointing
result. Everything that will be run is listed here first. Anything added later is
explicitly labelled **post-hoc exploratory** in `docs/Y1_Y3_FINAL_RESULTS.md` and is not
allowed to carry a confirmatory claim.

---

### 0. What is frozen and may not move

* **Y2 in full**,  target, event sample, prediction origin, folds, features, models,
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

### 1. Baseline numbers being improved on (from `docs/RESULTS_AUDIT.txt`)

| Target | Best pooled R2 | Model | Bootstrap verdict |
|---|---|---|---|
| Y1 (5-day) | +0.143 | mlp | no comparison's CI excludes zero |
| Y1 (10-day) | +0.169 | mlp | no comparison's CI excludes zero |
| Y2 | +0.270 | gp | **BEATS BASELINE** (gp, svr, both baselines) |
| Y3 (point regression) | +0.155 | ensemble | 5 rows beat `naive_zero` only |

### 2. Y1,  pre-declared experiment grid

#### 2.1 Horizons (exactly four; no fifth will be added)

`h` in {5, 10, 15, 20} CSE **trading sessions**, defined as
`Y_h = 100 * ln(P_{t+h} / P_{t-1})` where `t` is the first valid trading session on/after
the disaster date and `P_{t-1}` is the last valid pre-event close.
Columns: `Y1_ASPI_EventWindow_0_{h}_LogReturn_Pct` plus
`Y1_ASPI_EventWindow_0_{h}_horizon_end_date` for target-specific purging.
**h=5 is the principal target.** h=10/15/20 are pre-specified horizon-sensitivity
analyses, reported whatever they show.

#### 2.2 Information sets (identical folds, identical test events)

* **A,  market-only**: strictly pre-event market-state columns only.
* **B,  disaster-only**: disaster / severity / hazard / exposure columns only.
* **C,  combined**: A union B.
* **D,  normal-market + disaster-residual**: a Stage-A expected-return model
  `R_hat_normal,h` estimated on the **daily market series using only rows strictly before
  the event**, then a Stage-B model of `R_residual,h = R_obs,h - R_hat_normal,h` on
  disaster features, with `R_hat_final,h = R_hat_normal,h + R_hat_residual,h`.
  The Stage-A model for an outer test event sees no data dated on/after that event.

The **primary scientific comparison is C vs A**,  does disaster-specific information add
anything beyond the market's own pre-event state.

#### 2.3 Feature capacity

`K` in {5, 10} pre-specified; `K = 20` retained only as the already-frozen robustness
comparison. Feature-selection stability is reported per (horizon, information set) as
`selection_frequency`, `mean_rank`, `median_rank` across folds. A feature selected in one
fold only is reported as such and is not described as a finding.

#### 2.4 Models (closed list)

Ridge / ElasticNet (`StandardScaler` + `RidgeCV`/`ElasticNetCV` inside the purged inner
temporal CV), Random Forest, XGBoost, shallow MLP. **No algorithm will be added.**
Existing repository models (GP, SVR, quantile regression, ensemble, stacked) stay in the
appendix exactly as already reported.

#### 2.5 Direction, as a separate question

`negative_return = Y_h < 0`, per horizon: balanced accuracy, ROC-AUC, PR-AUC, MCC,
sensitivity, specificity, bootstrap AUC CI. Reported as a **different question** from
magnitude regression; a classification success is never presented as a regression success.

#### 2.6 Augmentation: SMOGN held OFF, uniformly

The new Y1 grid runs with **SMOGN off for every configuration**. This is a
held-constant factor, not a result-driven change: the grid's whole purpose is the
paired A-vs-C and horizon comparisons, and an augmentation whose minority mask is
defined from each fold's own target distribution would vary across the 4 horizons and
3 information sets, confounding exactly the contrasts being measured. SMOGN's effect is
already measured separately and reported (`artifacts/tables/smogn_ablation.parquet`, ablation
B6, protocol section 5), and the existing frozen Y1/Y2/Y3 pipeline keeps it ON and is
untouched by this grid.

#### 2.7 Validation

Outer: chronological walk-forward `train=30, test=10, step=10`.
Per horizon, `training label end < first test prediction origin`,  so h=20 carries a
20-session target embargo, enforced through that horizon's own
`*_horizon_end_date` column. Inner: purged temporal CV (`purged_inner_cv`).
If purging makes an inner split impossible: **reduce the number of inner splits**; if
still impossible, fall back to pre-specified default hyperparameters. Leakage protection
is never removed.

### 3. Y3,  pre-declared experiment grid

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
  recovery-probability calibration; secondary,  MAE / median absolute error / RMSE
  computed **among uncensored recoveries only**. Ordinary RMSE over censored rows treated
  as point values is not a primary metric.
* **Recovery categories: exactly two, pre-specified**,  `recovery <= 20 trading days`
  (an interpretable "recovers within a trading month") and the **existing**
  `C3b_slow_recovery` classifier, kept unchanged. No threshold sweep.

### 4. Statistics (Y1 and Y3)

* Paired evaluation: every candidate and baseline predicts **exactly the same** test rows.
* Episode-cluster bootstrap retained (`build_episode_ids`, 14-day chaining).
* Baselines,  Y1: naive zero return, training-fold mean, **market-only expected-return
  model**. Y3: training-fold survival baseline (Kaplan-Meier), training-fold median
  recovery.
* Holm family-wise correction across the whole Y1/Y3 results family, reported as a
  **secondary strict diagnostic** alongside the single-comparison CI, never folded into it.

### 5. Pre-declared ablation register

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

### 6. Decision rule, declared before any result is read

* **A,  statistically supported**: out-of-sample performance beats its pre-specified
  baseline AND the uncertainty interval excludes the null.
* **B,  suggestive but uncertain**: point estimate improves, interval overlaps the null.
* **C,  unsupported**: no consistent improvement over baseline.

A **B** is never re-labelled **A** because an R2 looks attractive. A non-significant
final result is retained and reported.

### 7. Stop condition

After this grid executes: no further horizon, no further algorithm, no threshold change,
no target change, no feature-count increase, no observation removal, no additional
variant search. The pipeline freezes.

---

# Part 3. External data pre declaration

Which external sources were admitted, why, and what each was expected to add, declared before any of them was scored.

## Pre-declaration: external data sources and derived features

**Written before any model is fitted on these features.** Standing integrity constraint 2
forbids justifying a feature, threshold or window by held-out performance. Everything in
this document is fixed here, in advance, with its reason. Whatever the paired bootstrap
says afterwards is reported unchanged,  including "no improvement", and including
"worse".

Date written: 2026-09-11. Author: pipeline maintenance pass following the
"what data are we missing?" audit.

---

### 1. Why this exists

The audit established that the study's key covariate is barely observed:

| Feature | Real values | Zero-filled |
|---|---|---|
| `financial_damage` | 17 / 64 | 47 |
| `damage_to_gdp` | 16 / 64 | 45 (+3 NaN) |
| `log_damage_x_flood` | 12 / 64 | 52 |

The research question asks whether *disaster severity* predicts market response, and
severity in money terms is missing for 73% of the sample. Worse, it is not missing at
random: EM-DAT records damage more often for large, recent, internationally-reported
events, so the damage columns partly measure **reporting coverage** rather than severity.

Three of the five sources below attack that directly. One extends the sample. One adds a
confounder control.

---

### 2. Sources

All five were reachability-tested before adoption. Endpoint, response code and row count
are recorded so the retrieval is reproducible.

| # | Source | Endpoint | Verified | Licence/status |
|---|---|---|---|---|
| 1 | DesInventar Sendai (UNDRR/UNDP, national DMC database) | `desinventar.net/DesInventar/download/DI_export_lka.zip` | HTTP 200, 18.5 MB zip  1.21 GB XML, 130,140 records | Open, UN-hosted |
| 2 | NASA POWER (NASA Langley Research Center) | `power.larc.nasa.gov/api/temporal/daily/point` | HTTP 200, 7 points × 10,106 days | Open, no key |
| 3 | FRED (Federal Reserve Bank of St. Louis), series `DEXSLUS` | `fred.stlouisfed.org/graph/fredgraph.csv?id=DEXSLUS` | HTTP 200, 14,005 rows | Open, no key |
| 4 | countryeconomy.com ASPI daily | `countryeconomy.com/stock-exchange/sri-lanka?dr=YYYY-MM` | HTTP 200, 21 rows/month, 20182026 | Public page |
| 5 | Wikidata SPARQL | `query.wikidata.org/sparql` | HTTP 200, 28 elections 2000, 2026 | CC0 |

#### Rejected, with reason

| Source | Reason |
|---|---|
| GDELT news tone | Persistent HTTP 429 including at 20 s spacing. No news-sentiment feature is built; thesis §3.3.1's sentiment claim stays unsupported and is amended instead. |
| ReliefWeb API v2 | HTTP 403,  requires a registered `appname`. |
| CBSL monthly CCPI | Published only as per-month PDF press releases; no CSV/XLS endpoint. |
| CBSL daily FX form | JS-gated; POST returns the unchanged page shell. Superseded by FRED `DEXSLUS`. |
| IMF IFS API | `dataservices.imf.org` unreachable (HTTP 000). |
| FRED monthly LKA CPI | No live series: `CPALTT01LKM657N`, `CPALTT01LKM659N`, `LKACPIALLMINMEI`, `LKAPCPIPCH`, `CPALTT01LKM661N`, `LKACPGRLE01GPM`, `SLKCPIALLMINMEI` all return the 404 page. |
| Stooq | JS-gated noscript shell. |
| CSE daily PDFs | Reachable (`cdn.cse.lk/cmt/<path>`, 2.7 MB each) but the index API returns only the 5 most recent; ~800 files ≈ 2.2 GB to reach 2023-07. Not adopted. |
| CSE index/sector history API | 5 endpoint names tested, all HTTP 404. |

---

### 3. Declared features

**48 features on 64 events is p > n before selection.** These blocks go through the
identical per-fold feature-selection and collinearity rule as the existing 31,  they are
not exempted, and the selection is fitted on training rows only, as before.

#### Block A,  hazard intensity, from NASA POWER (6 features)

Measured by instrument, not assessed by a reporter. No missingness, no reporting bias,
and available on the event day itself,  so this block is also admissible in the ex-ante
Model A specification (audit E14), which `financial_damage` is not.

Seven district points: Colombo (6.93 N, 79.86 E), Jaffna (9.66, 80.02), Batticaloa
(7.72, 81.70), Nuwara Eliya (6.97, 80.77), Galle (6.05, 80.22), Anuradhapura (8.31,
80.40), Ratnapura (6.68, 80.40). Chosen for island coverage,  one per major
climatic/administrative region,  before any correlation with a target was computed.

Let `P_d(t)` = `PRECTOTCORR` at district `d` on day `t` (mm/day), and
`S_d = Σ P_d(t-2..t)` the 3-day accumulation ending on the event day.

| Feature | Definition | A-priori justification |
|---|---|---|
| `hz_precip_max3d` | `max_d S_d` | 3-day accumulation is the standard flood-generating window; the max over districts because flooding is localised and an island mean would dilute it. |
| `hz_precip_mean3d` | `mean_d S_d` | Island-wide intensity,  the aggregate analogue, matching the index-level target. |
| `hz_precip_spread3d` | `std_d S_d` | Distinguishes one drowned district from island-wide rain at equal mean. |
| `hz_districts_wet` | `#{d : S_d > 50}` | 50 mm/3 days is the Sri Lanka Department of Meteorology heavy-rain advisory level. A published operational threshold, not a swept one. |
| `hz_wind_max3d` | `max_d max(WS10M_MAX(t-2..t))` | Storm-intensity analogue for the 9 Storm events, where rainfall is not the damaging mechanism. |
| `hz_precip_anom` | `hz_precip_mean3d / (3 · mean_d mean P_d(t-33..t-3)) - 1` | Intensity against the location's own recent baseline. Deliberately the same functional form as Y2 (`V/V̄₃₀ − 1`), so the feature is the target's construction applied to rainfall. The 30-day baseline ends at `t-3` so it never overlaps the event window. |

#### Block B,  physical severity, from DesInventar (6 features + 1 flag)

Match window **[t − 7, t + 14]** around the EM-DAT event date. EM-DAT dates a multi-day
event by onset; DesInventar records by district report date, which lags by days to
weeks. The window is fixed here and is not varied afterwards.

Coverage measured before declaring: **52 / 64 events matched**, and **35 of the 47
zero-filled-damage events** gain real severity. Median matched event: 14 districts,
125,035 affected, 76 houses destroyed. DesInventar ends **2020-12-20**, so 9 events
(2021, 2022) match nothing,  `di_available` carries that.

| Feature | Definition | Justification |
|---|---|---|
| `di_districts_hit` | distinct districts with ≥1 record in window | Geographic breadth of the shock,  the exposure dimension the index-level design otherwise cannot see. |
| `di_affected_log` | `log1p(Σ afectados)` | Headcount severity. Log because the raw range spans 5 to ~1e6. |
| `di_houses_destroyed_log` | `log1p(Σ vivdest)` | Capital destruction,  the closest physical proxy to the missing monetary damage. |
| `di_houses_damaged_log` | `log1p(Σ vivafec)` | Partial-loss counterpart. |
| `di_deaths_log` | `log1p(Σ muertos)` | Human severity; also the variable most consistently recorded. |
| `di_records` | record count in window | Reporting intensity. Declared explicitly as a *reporting* measure, not a severity one, so that if it dominates a model the interpretation is stated rather than discovered. |
| `di_available` | 1 if matched else 0 | Missingness indicator (audit P2-7). Mandatory: without it a zero reads as "no damage" rather than "outside DesInventar's coverage". |

**Stated in advance:** DesInventar Sri Lanka populates `valorus` (USD loss) in **0** of
130,140 records and `valorloc` (LKR) in **16**. It carries no monetary loss. This block
is a *physical* severity substitute and must never be described as recovering the
missing damage figures.

#### Block C,  daily macro, from FRED `DEXSLUS` (3 features)

Replaces an annual macro series matched to day-0 events. All three are computed strictly
from data at or before **t − 1**, the same pre-shock boundary the ASPI features use,
because the exchange rate is a market price and same-day use would leak.

| Feature | Definition |
|---|---|
| `fx_logret_1` | `ln(FX(t-1) / FX(t-2))` |
| `fx_logret_5` | `ln(FX(t-1) / FX(t-6))` |
| `fx_vol_30` | `std` of daily FX log returns over `[t-31, t-1]` |

Ratios and differences only,  stationary by construction, and the existing ADF/KPSS gate
in stage 02 applies to them unchanged.

#### Block D,  confounder control, from Wikidata (2 features)

28 national elections, 2000-10-10  2026-08-31. Motivated by a specific known
contamination: the **2005-11-17 presidential election falls 4 days before the
2005-11-21 event**, which carries the largest observed Y1 drop (−0.0753).

| Feature | Definition |
|---|---|
| `days_to_election` | signed days to the nearest national election (negative = election already held) |
| `election_within_5d` | `1` if `abs(days_to_election) <= 5` |

±5 days matches the event-window contamination screen already specified in the audit
(§39 item 7). Fixed there before this pass.

#### Block E,  sample extension, from countryeconomy (no feature)

The ASPI series ends **2023-06-28** in the local archive. countryeconomy supplies daily
closes to 2026-09-10, verified continuous: its 2023-06-28 close of **9,442.95** matches
the archive's final row exactly.

This admits the **10 qualifying EM-DAT events after 2023-06-28**,  2023-07-04,
2023-09-28, 2023-11-01, 2023-12-01, 2024-01-01, 2024-05-15, 2024-10-11, 2024-11-25, and
Storm Ditwah on 2025-11-20 and 2025-11-27,  taking **N from 64 toward 74**, subject to
each event still passing the pre-registered ≥1000-affected filter.

**This is the single largest change in the pass**, because it adds real out-of-fold test
points rather than features. The inclusion filter is unchanged; only the market series
is longer.

**Declared limitation:** countryeconomy publishes the index level, not volume, and no
practical volume source was found for the post-2023 period. The new events therefore
carry **Y1 and Y3 but not Y2**. Y2's effective N stays at 61 and the new events are
NaN-masked out of it, exactly as the 2000 volume gap is already handled.

---

### 4. What is expected, stated before the run

So that no outcome can be presented as a success after the fact:

1. **Block E (more events) is expected to help most**, because it adds test points and
   training rows rather than columns. It should tighten every confidence interval.
2. **Block A (hazard) is expected to help Y2 and Y3 more than Y1.** Rainfall is a
   plausible driver of trading disruption and recovery duration; index-level day-0
   return has resisted five model families across three feature generations.
3. **Block B (DesInventar) is expected to help chiefly through `di_available` and
   `di_districts_hit`**, not through the loss counts, because the loss counts inherit
   part of the same reporting-coverage problem as EM-DAT.
4. **Block C (FX) is expected to do little on its own** but is required regardless: an
   annual macro control against a day-0 event is indefensible, and replacing it is a
   correctness fix, not a performance lever.
5. **Y1 is still expected to be unpredictable.** Nothing here changes the efficient-markets
   baseline. If Y1 remains indistinguishable from a constant, that is the reported result.

**None of these expectations may be revised after seeing the scores.** The ablation
(one block in, one block out, identical folds) is reported in full whatever it shows.

### 5. What this does not do

- It does not recover monetary damage. Effective N for money-terms severity stays 17.
- It does not fill 2021, 2022 severity (DesInventar ends 2020-12-20),  Block A does cover
  those events, which is part of why Block A exists.
- It does not add per-sector volume, so the sector panel remains Y1/Y3 only.
- It does not add news sentiment. Thesis §3.3.1 is amended rather than satisfied.
- It does not route around the refusal to target R² ≥ 0.65. No threshold, window, model
  or feature below was chosen to reach a number.

---

## 6. Ablation outcome, measured (run 2026-09-11 19:07)

Six configurations, identical folds, 40 pooled out-of-fold points (34 for Y2). Each
block's contribution is `RMSE(full-without-block) - RMSE(full)`, so **positive means the
block helps**, with a 95% paired event-level bootstrap CI.

**5 of 45 (target, model, block) combinations have a CI excluding zero. All five are on
Y2.**

| Target | Model | Block | ΔRMSE | 95% CI |
|---|---|---|---|---|
| Y2 | xgboost | **all external** | +0.1418 | [0.0135, 0.2652] |
| Y2 | xgboost | **fx** | +0.1315 | [0.0595, 0.2031] |
| Y2 | random forest | **all external** | +0.0769 | [0.0176, 0.1390] |
| Y2 | random forest | **fx** | +0.0437 | [0.0129, 0.0808] |
| Y2 | xgboost | election | +0.0300 | [0.0004, 0.0689] |

Pooled R² on Y2 moves from **−0.0890 without the external blocks to +0.1912 with them**
(random forest), and from **−0.3624 to +0.1878** (xgboost).

### Where the pre-declaration was wrong

Section 4 committed five expectations in advance. Scored honestly:

| # | Expectation | Outcome |
|---|---|---|
| 1 | Sample extension helps most | **Partly right.** It delivered 40 test points instead of 30 and tightened every interval, but it is not separable in this ablation,  it changes the rows, not the columns. |
| 2 | Hazard helps Y2/Y3 more than Y1 | **Right in direction, not significant.** Y2 ridge +0.0678, Y1 negative on all three models. No CI excludes zero. |
| 3 | DesInventar acts through `di_available`/`di_districts_hit` | **Not supported.** The whole block is indistinguishable from zero on every target and model; it is *negative* on Y2 for two of three. |
| 4 | **FX "expected to do little on its own"** | **WRONG, and wrong in the useful direction.** FX is the single strongest block measured: +0.1315 on Y2/xgboost and +0.0437 on Y2/random forest, both CIs excluding zero. It was adopted as a correctness fix and turned out to be the only block that independently earns its place. |
| 5 | Y1 still unpredictable | **Right.** No block helps Y1 on any model; 0 of 15 Y1 combinations are significant. |

Expectation 4 is recorded as a failed prediction rather than quietly rewritten. That is
the entire point of writing section 4 before the run: had the expectations been written
afterwards, "daily exchange-rate dynamics predict abnormal trading volume" would read as
a designed finding instead of a surprise.

### The distinction that must not be blurred

The external data **significantly improves the model relative to a model without it**.
It does **not** make Y2 beat a naive baseline: the best Y2 model (ensemble) still sits at
ΔRMSE +0.0651, CI [−0.0101, +0.1333] against the constant-zero null.

Both statements are true simultaneously and must be reported together. "Adding exchange
rate data measurably improves abnormal-volume prediction" is supportable. "Abnormal
volume is predictable" is not.

---

## 7. Second pre-declaration: sample widening and target reformulation (2026-09-12)

**Written before any model is fitted on these.** Same rule as sections 1-5: every
definition, window and threshold below is fixed here, in advance. Whatever the paired
bootstrap says afterwards is reported unchanged.

Prompted by the author's instruction to (a) use the 1986-2025 EM-DAT export, (b) lower the
inclusion threshold to 700 affected, (c) use any available technique, and (d) reach >0.6 on
accuracy/predictability for every target.

### 7.1 What the instruction can and cannot buy

Stated plainly, before the run, so no outcome can be spun afterwards:

| Instruction | Measured consequence |
|---|---|
| EM-DAT 1986-2025 (110 records, was 86) | **+0 modellable events.** The 24 extra records are all pre-2000. ASPI starts 2000-01-03 and a day-0 event study needs the index on the event day plus 90 trading days after it. 6 of the 20 qualifying pre-2000 events additionally have no Start Day at all. They are loaded and then dropped by the scope filter, with the count printed. |
| Threshold 1000 -> 700 affected | **+2 events** (2021-05-16 Storm, 2023-04-24 Storm). N 74 -> 76. Cannot rescue a result; recorded as a deviation from the thesis pre-registration, not as the original design. |
| "Simulate disasters" | Already implemented as SMOGN inside training folds only, and already **measured to make every model worse on every target**. Synthetic rows in TEST would void every reported metric. No change. |
| R2 > 0.6 on Y1/Y2/Y3 | **Not reachable.** Best in study is +0.199 (Y2). R2 0.6 on a day-0 index return means explaining 60% of market variance from disaster covariates, which contradicts the efficient-markets result the thesis itself argues. No technique below targets it. |
| Accuracy / AUC / F1 > 0.6 | **Reachable, and already reached on one target** (`C2_volume_spike` random forest: AUC 0.764 [0.578, 0.929], balanced accuracy 0.629). Sections 7.2-7.3 are the honest attempt to extend that to the other targets. |

### 7.2 New targets: cumulative event-window returns

Y1 is a single day's log return, the noisiest possible measurement of an event's market
effect. Standard event-study practice accumulates over a window, which raises
signal-to-noise without adding any information the study does not already have.

Let `pos` be the first trading row on or after the event date (the existing anchoring in
`build_targets`), and `P` the ASPI close.

| Target | Definition | Justification |
|---|---|---|
| `Y1_EventWindow_0_5_LogReturn_Pct` | `ln(P[pos+5] / P[pos-1])` | Cumulative return over the event day plus 5 trading days. The conventional short event window. Verified: **76/76 events have >=5 trading rows after `pos`.** |
| `Y1_EventWindow_0_10_LogReturn_Pct` | `ln(P[pos+10] / P[pos-1])` | Two-week window, the other conventional choice. Verified: **76/76 events have >=10 trading rows after `pos`.** |

Both windows were fixed at 5 and 10 because those are the standard short-horizon event
windows in the literature, not because either scored better. No other k is tried, and
neither may be swapped for the other after seeing a result.

**A reformulation considered and rejected before any fit:** market-model abnormal returns
(regress ASPI on a benchmark over a pre-event estimation window, take the residual).
`docs/METHODOLOGY_AUDIT.md` already rules this out on identification grounds, the asset
here **is** the market index, so there is no valid benchmark to regress it against, and
`notebooks/09_synthesis.ipynb` records the same decision as a resolved audit item. It is
not revived here. Using the S&P 500 as the benchmark would additionally be invalid because
Colombo closes roughly ten hours before New York opens, so a same-date S&P return is not
observable to a CSE participant on the event day.

### 7.3 Reformulated classification labels

`C3_recovers_in_90` is broken as a classification target and this is a defect, not a
result: prevalence is **0.90**, so the always-predict-majority rule scores 0.900 accuracy
while every model scores *below chance* on AUC (best 0.597, random forest 0.229). An
"accuracy" of 0.9 there measures the class imbalance, not the model.

| Label | Definition | Prevalence | Justification |
|---|---|---|---|
| `C3b_slow_recovery` | `Y3 > median(Y3 over THIS fold's training rows)` | ~0.50 by construction | Median split makes accuracy and AUC informative instead of gameable. The cut is computed on training rows only, same discipline as the existing `C1b_adverse_move` tercile rule, so no test information reaches the label. |
| `C4_car5_negative` | `Y1_EventWindow_0_5_LogReturn_Pct < 0` | measured, ~0.5 expected | Direction over the 5-day window rather than the single noisiest day. Same sign question as `C1_negative_return`, asked of a less noisy measurement. |

`C1_negative_return`, `C1b_adverse_move` and `C2_volume_spike` are unchanged.
`C3_recovers_in_90` is **retained and still reported** beside `C3b_slow_recovery`, because
dropping it after seeing that it fails would be exactly the selection this document exists
to prevent.

### 7.4 Expected outcomes, fixed in advance

1. **`Y1_EventWindow_0_5_LogReturn_Pct` / `Y1_EventWindow_0_10_LogReturn_Pct` will still not beat their nulls as regressions.** Widening
   the window reduces noise but does not create predictability. Expect R2 to stay negative.
2. **`C4_car5_negative` is the most likely of the new labels to clear 0.6**, because the
   5-day sign is a less noisy question than the day-0 sign.
3. **`C3b_slow_recovery` will score near 0.5.** Its value is diagnostic: it converts an
   uninformative 0.90 into an honest number, and that is worth reporting even if the honest
   number is "no signal".
4. **`C2_volume_spike` remains the strongest result.** Nothing here is expected to displace it.
5. **Y1 remains unpredictable at the index level.** Nothing in this section changes the
   efficient-markets conclusion.

None of these may be revised after seeing the scores.

### 7.5 Sector-panel classification

The index-level classification layer is scored on **40 pooled test points**. The sector
panel carries **66 of the 76 events** (the sector workbook ends 2023-06-28) across 20
sector indices, giving **~1,320 real (event, sector) rows** and, under
`grouped_walk_forward(30, 10, 10)`, **3 folds = 30 independent test events ~ 600 pooled
test rows**. That is 15x the index-level test set, entirely real, with no simulation.

No classifier has ever been fitted on this panel; it has only been used for regression.
Doing so is the largest genuine increase in evidence available to the study.

The same six labels run unchanged, with two structural consequences fixed in advance:

- **`C2_volume_spike` cannot run here.** The CSE publishes volume market-wide, not per
  sector, so the panel carries no Y2. The label is all-NaN and is skipped automatically.
  It is not replaced by a proxy.
- **Effective N is the event count, not the row count.** 20 sectors move together on a
  shock day. Every interval comes from `event_block_bootstrap`, which resamples whole
  events; a row-level bootstrap would shrink intervals by roughly sqrt(20) and manufacture
  significance out of the co-movement. `tests/test_integrity_invariants.py` asserts the
  clustered interval is strictly wider than the naive one.

**Expected outcome, fixed before the run:** the panel is where a genuine positive finding
is most likely, because index-level aggregation is exactly what should wash out a
localised flood. If sector response is also indistinguishable from its null, that is a
stronger negative result than the index-level one alone, it closes off the explanation
the thesis currently offers for its own null finding.

### 7.6 Success criterion, decided by the author 2026-09-12

**Beat the majority rule, not raw accuracy.**

Raw accuracy is not a criterion. Three labels already exceed 0.6 on it and it means
nothing on an imbalanced target: `C3_recovers_in_90` scores 0.900 accuracy because
prevalence is 0.900, and its models score *below* chance on AUC.

The reported criterion is `beats_baseline` (`src/models/classifiers.py`): balanced
accuracy > 0.5 **and** both AUC intervals excluding 0.5. It currently holds for **1 of 16**
(label, model) pairs. Raw accuracy is still reported, always beside its prevalence and its
majority-rule accuracy on the same row, so a reader can see which numbers are real.

---

# Part 4. Frozen volume target validation

Evidence that the volume crash magnitude target did not move while the return and recovery targets were being improved.

## Y2 frozen-validation report

**Verdict: NO MATERIAL CHANGE. No change of any kind, at 1e-9 tolerance.**

`Y2_abnormal_volume` is the study's one statistically supported continuous target. The
2026-09-17 Y1/Y3 improvement work was required to leave it untouched. This report is the
before/after evidence, and `tests/test_y2_frozen.py` is the mechanical enforcement.

* **Before** = commit `1fbf6275`, captured in `artifacts/results/frozen_baseline.json`
  (`scripts/freeze_baseline.py`), taken before a single line of improvement code existed.
* **After** = the same artifacts re-read at the end of the improvement work.
* **Tolerance** = `1e-9` absolute, on a target whose own scale is ~0.5. This is float
  noise, not a materiality threshold.

### How Y2 was protected

The improvement work adds files; it changes no file the frozen pipeline reads.

* The four Y1 horizon targets are built **in memory** from `artifacts/tables/market.parquet`
  (`src/targets/return_horizons.py`), not by regenerating `artifacts/tables/dataset.parquet`.
  `dataset.parquet`,  which is where `Y2_abnormal_volume` lives,  is never rewritten, so
  Y2's target values cannot move.
* `src/features/feature_engineering.py`, `src/training/walk_forward.py`,
  `src/evaluation/verification.py` and `notebooks/_shared.py` were **read, not edited**.
  Shared infrastructure is consumed by the new scripts, never modified by them.
* `src/evaluation/survival_metrics.py` and `src/targets/return_horizons.py` are new
  modules with no importer inside the frozen pipeline.
* `notebooks/04_modeling_regression.ipynb`, `notebooks/05_modeling_classification.ipynb`
  and `scripts/run_survival_model.py` were not re-run, so `results_regression.pkl`,
  `verdict_table.parquet` and `classification_summary.parquet` are byte-identical to the
  freeze.

The only shared-file edit anywhere in this work is an **additive** keyword argument
(`return_draws=False`) on `survival_metrics.cluster_bootstrap_ci`, a module that did not
exist at freeze time and that nothing in the Y2 path imports.

### Before vs after

| Quantity | Before | After | Change |
|---|---|---|---|
| Target values (all 74 rows) | see `frozen_baseline.json` | identical | **0** |
| Observations scored (pooled OOF) | 34 | 34 | **0** |
| Fold definitions (4 folds, train/test index arrays) | identical | identical | **0** |
| Pooled OOF predictions, all 11 models | identical | identical | **0** |

#### Regression metrics (pooled out-of-fold, n = 34)

| Model | RMSE before | RMSE after | MAE before | MAE after | R2 before | R2 after |
|---|---|---|---|---|---|---|
| gp | 0.45535 | 0.45535 | 0.36114 | 0.36114 | +0.27042 | +0.27042 |
| svr | 0.45790 | 0.45790 | 0.36431 | 0.36431 | +0.26222 | +0.26222 |
| random_forest | 0.48548 | 0.48548 | 0.39716 | 0.39716 | +0.17068 | +0.17068 |
| ensemble | 0.51955 | 0.51955 | 0.40637 | 0.40637 | +0.05017 | +0.05017 |
| xgboost | 0.54343 | 0.54343 | 0.44155 | 0.44155 | -0.03912 | -0.03912 |
| ridge | 0.54557 | 0.54557 | 0.45165 | 0.45165 | -0.04733 | -0.04733 |
| mlp | 0.62240 | 0.62240 | 0.47767 | 0.47767 | -0.36308 | -0.36308 |

#### Bootstrap CI and statistical verdict (episode-clustered, 10,000 draws)

| Model | Baseline | delta RMSE | CI low | CI high | Verdict | Holm |
|---|---|---|---|---|---|---|
| gp | naive_zero | +0.08684 | +0.02521 | +0.15398 | **BEATS BASELINE** | False |
| gp | naive_train_mean | +0.09309 | +0.01229 | +0.16685 | **BEATS BASELINE** | False |
| svr | naive_zero | +0.08429 | +0.03435 | +0.14178 | **BEATS BASELINE** | **True** |
| svr | naive_train_mean | +0.09054 | +0.02991 | +0.14331 | **BEATS BASELINE** | False |
| random_forest | naive_zero | +0.05671 | -0.00948 | +0.12244 | better, not distinguishable | False |
| random_forest | naive_train_mean | +0.06296 | -0.01850 | +0.13280 | better, not distinguishable | False |
| ridge | naive_zero | -0.00338 | -0.09906 | +0.07457 | worse than baseline | False |
| xgboost | naive_zero | -0.00123 | -0.09488 | +0.09718 | worse than baseline | False |
| mlp | naive_zero | -0.08021 | -0.20958 | +0.04571 | worse than baseline | False |
| quantile | naive_zero | -0.35666 | -0.54618 | -0.15812 | worse than baseline | False |

All 18 Y2 rows are identical before and after. Nine of them are omitted from the table for
length only; `tests/test_y2_frozen.py::test_y2_bootstrap_verdict_unchanged` asserts every
one of them.

#### Classification arm (`C2_volume_spike`)

| Model | AUC before | AUC after | Balanced acc. before | Balanced acc. after | Beats baseline |
|---|---|---|---|---|---|
| logistic | 0.7107 | 0.7107 | 0.5643 | 0.5643 | True |
| rf_clf | 0.6964 | 0.6964 | 0.6286 | 0.6286 | False |
| xgb_clf | 0.6893 | 0.6893 | 0.5071 | 0.5071 | False |
| majority_baseline | 0.5000 | 0.5000 | 0.5000 | 0.5000 | False |

### Mechanical enforcement

`tests/test_y2_frozen.py`,  20 tests, all passing,  asserts, against
`artifacts/results/frozen_baseline.json`:

1. `test_y2_target_values_unchanged`,  all 74 Y2 values.
2. `test_y2_event_sample_unchanged`,  the 74 event dates, in order.
3. `test_y2_fold_definitions_unchanged`,  every fold's train and test index arrays.
4. `test_y2_pooled_oof_predictions_unchanged`,  `y_true` and `y_pred`, per model (11).
5. `test_y2_per_fold_metrics_unchanged`,  RMSE, MAE, R2, per fold, per model.
6. `test_y2_pooled_r2_unchanged`,  pooled R2 per model.
7. `test_y2_bootstrap_verdict_unchanged`,  delta, CI bounds, one-sided p, verdict string,
   `boot_beats`, `holm_significant`.
8. `test_y2_classification_unchanged`,  C2 AUC, balanced accuracy, AUC CI, MCC, PR-AUC.

Full suite at the end of the work: **119 passed** (99 pre-existing + 20 new).

If a future change makes any of these fail, that is a stop-and-diagnose signal. The
baseline is not to be regenerated to make the test pass; the cause is to be found,
documented in this file, and only then re-frozen.

---

# Part 5. Method comparison, before and against after

What each methodological change was, and whether it paid off. Several rows report no gain and are kept deliberately.

## Final method comparison,  what the improvement work changed, and what it bought

This is the before/after of *method*, not of numbers. The numbers are in
`docs/Y1_Y3_FINAL_RESULTS.md`; the point here is which methodological choice changed, why,
and whether it was worth it. Several of these rows say "no measurable gain",  those are
kept deliberately.

Baseline = commit `1fbf6275`, frozen in `artifacts/results/frozen_baseline.json`.

---

### Y1,  ASPI return magnitude

| Dimension | Before | After | Did it help? |
|---|---|---|---|
| Horizons | 2 (5 and 10 sessions), grown ad hoc | 4 (5, 10, 15, 20), **all pre-declared at once**, h=5 principal | No performance gain. Large *credibility* gain: horizon shopping is now impossible to allege, because the grid was fixed in writing first. |
| Target definition | `Y1_ASPI_5D_Forward_LogReturn_Pct` + one event-window column | `Y1_ASPI_EventWindow_0_{5,10,15,20}_LogReturn_Pct`, one formula, one alignment rule, each with its own `_horizon_end_date` | Consistency. h=5 reproduces the frozen Y1 exactly (asserted in `tests/test_y1_horizons.py`). |
| Purging | Per-target label-end purge (already correct) | Same rule, per **horizon**,  h=20 embargoes 20 sessions | Necessary for the new horizons; no change to existing behaviour. |
| Information sets | One combined feature set | **market-only / disaster-only / combined / normal-market+residual**, hand-partitioned, identical folds | This is the single most important methodological addition. It converts "can we predict returns?" into the answerable question "does disaster information add anything beyond market state?",  and lets the study answer *no* with evidence. |
| Expected-return baseline | naive zero, training-fold mean | + **market-only expected-return model** (Stage A Ridge on the daily series, strictly pre-event) | The right null for an event study. It is also *harder* than naive zero at h=15, which is exactly why it belongs. |
| Two-stage architecture | none | normal-market + disaster-residual decomposition | **No gain.** Mid-pack at h=5/10, worst at h=15/20. Reported as a negative result. |
| Feature capacity | K=20, fixed | K in {5, 10, 20}, with K=20 as robustness only | K=10 wins 8 of 16 cells, K=5 six, K=20 two. **The compact model is free**,  report the interpretable one. |
| Model set | 9 families (Ridge, RF, XGB, GP, SVR, quantile, MLP, ensemble, stacked) | **5** (Ridge, ElasticNet, RF, XGBoost, shallow MLP); the rest stay in the appendix, unchanged | Reduces multiplicity, which is the honest way to raise power. The appendix models were not deleted,  nothing is hidden. |
| Feature-selection nesting | Nested inside inner CV (already fixed) | Same, plus a **stability report** (selection_frequency / mean_rank / median_rank per fold) | Produced the study's most sobering number: **35.6% of selected features appear in exactly one of four folds.** |
| Inner-CV degradation | Fall back to an unpurged `TimeSeriesSplit` if every purged split dies | **Reduce the number of splits first; then use pre-specified default hyperparameters. Never unpurge.** | Leakage protection is no longer sacrificeable. In practice all 16 fold-horizons kept 3 purged inner splits, so the fallback never fired,  but the guarantee is now structural. |
| Direction | Existing `C1_negative_return` label, one horizon | `negative_return = Y_h < 0` at all four horizons, balanced accuracy / ROC-AUC / PR-AUC / MCC / sensitivity / specificity / episode-clustered AUC CI | **The one genuinely new positive finding** (h=10, AUC 0.752, Holm p = 0.032). |
| Augmentation | SMOGN ON | **SMOGN OFF, uniformly**, across the whole new grid | Held constant so the A-vs-C and horizon contrasts are not confounded by a mask that varies with the target. The existing SMOGN ablation stands unchanged. |

#### Net effect on Y1

Predictive performance: **unchanged, and that is the finding**. Best pooled R2 moved from
+0.143 (frozen MLP, h=5, SMOGN on, 9-model lineup) to +0.075 (h=10 disaster-only elastic
net),  and neither number is distinguishable from its baseline. What improved is the
*quality of the negative result*: it now rests on a pre-declared 240-configuration grid with
three baselines, four information sets and family-wise correction, rather than on a handful
of models that happened to be tried.

---

### Y3,  recovery duration

| Dimension | Before | After | Did it help? |
|---|---|---|---|
| Primary likelihood | Point regression on `Y3_recovery_days` (RF/Ridge/XGB/MLP), censored rows scored as observed | **Right-censored AFT**, censored rows contributing a lower bound | Correctness. An RMSE that treats a 90-day cap as an observed recovery is measuring the wrong thing. |
| Training rows | SMOGN-augmented | **Genuine events only** for every survival fit | Necessary, not optional: SMOGN interpolates a duration but cannot synthesise a valid event/censoring indicator, so a synthetic `(38, observed)` row asserts a recovery nobody saw. |
| Architecture | Single-stage regression; a hurdle variant as a side experiment | **Two-stage: P(drawdown) x AFT on drawdown cases** | **Yes.** Best C-index (0.657 vs 0.539-0.598 for plain AFT) *and* by far the best calibration (worst gap 0.045 vs 0.142 for plain log-normal AFT). |
| Distributions | AFT chosen in-fold by AIC (Weibull or log-normal, invisible to the reader) | **Weibull and log-normal reported separately** at every capacity | Transparency. They turn out near-identical (0.657 vs 0.643), so the AIC choice was never carrying a result. |
| Cox PH | not attempted | penalized Cox, **fitted only where the effective event count supports it** (>= 12 events, <= 5 covariates) | Ran; C-index 0.569, mid-pack. No instability, no advantage. |
| Random survival forest |,  | **deliberately not used** | 31 observed recoveries does not support it. Declared in advance, not abandoned after a bad result. |
| Output | one number ("recovery = 17 days") | **predicted median + P(T <= 10/20/30/60/90)** per event | This is where Y3 becomes usable. Exact-day prediction fails; calibrated recovery probabilities do not. |
| Primary metric | pooled RMSE over all rows | **Harrell C-index** (episode-clustered CI), IPCW **integrated Brier score**, **probability calibration**; point errors only among uncensored recoveries, as secondary | Changes the conclusion: the frozen pipeline's "Y3 R2 = +0.155" was substantially an artefact of scoring censored rows as observed. Ranking is the defensible claim. |
| Baselines | naive zero, training-fold mean | **training-fold Kaplan-Meier survival curve** + training-fold median recovery | A survival model must be compared to a survival baseline. KM scores C-index 0.477,  worse than chance, as a constant curve should be,  which is what makes 0.657 meaningful. |
| Categories | several thresholds available | **exactly two pre-specified**: `recovery <= 20 days` and the existing `C3b_slow_recovery` | No threshold sweep. The new one fails (AUC 0.612, CI [0.347, 0.854]); the existing one stands unchanged. |

#### Net effect on Y3

The *number* got smaller and the *claim* got sounder. Pooled R2 +0.155 under a wrong
likelihood became C-index 0.657 under the right one,  a real, interpretable, calibrated
ranking ability that nonetheless fails Holm correction and is reported as **suggestive**.

---

### Y2,  unchanged, by design

Nothing. Zero numbers moved, at 1e-9 tolerance, and 20 automated assertions enforce it.
See `docs/Y2_FROZEN_VALIDATION_REPORT.md`.

---

### Statistical machinery

| Dimension | Before | After |
|---|---|---|
| Pairing | Paired bootstrap already in use | Enforced and **tested**: `tests/test_improvement_artifacts.py` asserts every configuration and baseline scores an identical row set |
| Clustering | Episode-cluster bootstrap (14-day chaining) | Unchanged, extended to the new Y1 grid, the Y3 C-index, and both AUC analyses |
| Multiplicity | Holm across `verdict_table` | Holm across each new family separately: 720 Y1 comparisons, 15 Y3 models, 8 direction comparisons,  **reported beside the single-comparison CI, never folded into it** |
| Verdict vocabulary | "BEATS BASELINE" / "better, not distinguishable" / "worse" | Explicit **A / B / C** classification carried in the artifacts themselves |

### Cost

Roughly 75 minutes of compute for the Y1 grid (240 configurations x 4 folds with nested
selection inside purged inner CV), ~1 minute for Y3. The Stage-A expected-return estimates
are cached (`artifacts/tables/aspi_expected_return_market_only.parquet`) because they depend only on frozen
inputs.

### What was NOT done, and why

* **No lockbox holdout.** Still impossible at N=74 across a 30/10/10 walk-forward; the
  limitation stands exactly as disclosed in `docs/FINAL_ANALYSIS_PROTOCOL.md` section 8.
* **No market-only direction arm.** The direction analysis was pre-declared on the combined
  set only. Adding it now, after seeing that h=10 works, would be exactly the post-hoc
  extension the protocol forbids. It is listed as a limitation instead.
* **No additional events.** The `total_affected >= 1000` threshold was not lowered and no
  event was re-dated or removed.
* **No new algorithms.** The five-model list was closed before any result was read.

---

# Part 6. Thesis amendments

Changes the written thesis needs because the implementation moved.

## Thesis Text Amendments

Every item below is a place where the thesis document says something the code does not do,
or uses a term that misdescribes what was measured. Each gives the current claim, the
replacement wording, and why the change is needed.

These are corrections to the *written* thesis (`IM2021007`), not to the repository. Apply
them in your Word/LaTeX document.

Ordered by severity: the first five are things an examiner will attack directly.

---

### Blocking,  fix before submission

#### 1. The early-warning framing is not supportable

**Where:** abstract, §1 introduction, §4.4 SDG alignment, conclusion.

**Currently:** the framework is described as an early-warning system for CBSL/SEC use.

**Replace with:** "an **ex-post impact-attribution** framework". Add, in the abstract and
again in the limitations:

> Six of the model's features,  `financial_damage`, `population_affected`, their log
> transforms, `damage_to_gdp` and `log_damage_x_flood`,  are EM-DAT damage assessments
> finalised weeks to months after an event. The model therefore cannot be run
> prospectively on the day before a disaster. It answers the attribution question,  given
> a disaster of measured severity, what was the market's response,  rather than the
> forecasting question.

**Why:** the claim as written is falsified by the feature list. This is the first thing a
viva panel will find.

---

#### 2. Y1 is a raw log return, not an abnormal return

**Where:** §3.2.2, Table 4, and everywhere Y1 is named.

**Currently:** Y1 is referred to in places as a "percentage change" and elsewhere the
language borrows from the abnormal-return literature of §2.2.3.

**Replace with:** "continuously compounded (log) return on the event day,
`ln(P_t / P_{t−1})`".

**Why:** the code computes exactly that. No market model, alpha or beta is estimated, and
for a national index there is no valid benchmark against which to define an abnormal
return,  which is a defensible choice, but it must be stated as the choice it is.

---

#### 3. Y2 is a volume *spike* measure, not a "volume crash"

**Where:** §3.2.2, Table 4.

**Currently:** described as capturing a volume crash.

**Replace with:** "abnormal trading volume, `V_t / mean(V_{t−30..t−1}) − 1`; positive
values indicate volume above the pre-event baseline."

**Why:** the observed range runs from −0.87 to **+1.94**. The measure's upside is a spike,
and "crash" inverts its meaning.

---

#### 4. Y3 is measured in trading days, and the cap is censoring

**Where:** §3.2.2, and any statement of the 90-day window.

**Currently:** "90 days", with the implication that a non-recovering event has a recovery
time of 90.

**Replace with:** "recovery time in **trading days**, right-censored at 90." Add:

> For an event that has not recovered within the window the data establishes only that
> `T > 90`; assigning `T = 90` is a biased point estimate. A two-stage hurdle model is
> reported alongside the single-stage regression for this reason.

**Why:** an earlier implementation counted calendar days while searching a window of 90
trading rows, so the cap effectively bit at about 62 trading days. That is fixed in code
and the text must follow.

---

#### 5. Y3 is not independent of Y1

**Where:** §3.2.2, and before any Y3 result is reported.

**Add:**

> Because the recovery window includes the event day and the baseline is the previous
> close, any event with a non-negative event-day return satisfies the recovery condition
> immediately, so `Y3 = 0` if and only if `Y1 ≥ 0`. Over half of all Y3 observations are
> therefore mechanically determined by the sign of Y1, and Y3 performance cannot be read
> as independent evidence.

**Why:** it is an identity in the target construction, and stating it yourself is far
stronger than having it extracted in the viva.

---

### Method description does not match the code

#### 6. `MultiOutputRegressor` is described but not used

**Where:** §3.6.4.

**Replace with:** "a separate estimator is fitted per target, with per-target feature
selection and per-target hyperparameter search."

**Why:** the code fits each target independently. `MultiOutputRegressor` appears in an
unused helper module only.

#### 7. Multi-task learning applies to the MLP alone

**Where:** §3.6, wherever the framework is called multi-task.

**Replace with:** reserve "multi-task" for the shallow MLP, which genuinely shares a
trunk across three heads. Describe the rest as **multi-output prediction**. Add: "no
positive transfer was measured,  the MLP is not the best model on any target."

#### 8. Remove SVR from the methodology

**Where:** §3.6.1.

**Why:** specified but never implemented. It spans no model-family space the other four
do not already cover. Delete rather than implement, and say so if asked.

#### 9. Delete the news-sentiment claim

**Where:** §3.3.1, which states aggregated financial-news sentiment was used.

**Why:** Table 4's own note says textual sentiment was excluded, and the implementation
excludes it. The two statements contradict each other; §3.3.1 is the wrong one.

#### 10. Inner validation is a chronological `TimeSeriesSplit`

**Where:** §3.7.2.

**Add:** "hyperparameter selection inside each walk-forward fold uses a chronological
`TimeSeriesSplit` restricted to that fold's **real** rows, with the winning configuration
refitted on real plus synthetic rows."

**Why:** an earlier version used `GridSearchCV(cv=3)`, which is k-fold,  the exact
procedure §3.7.1 bans. It was found during audit and replaced. Reporting the correction
is a strength, not an admission.

#### 11. Report both rolling and expanding windows

**Where:** §3.7.1.

**Why:** §3.7.1 describes an expanding training set; the original code rolled. Both are
now available and both should be reported.

#### 12. Correct the RMSE definition

**Where:** wherever RMSE is defined.

**Why:** check the formula as printed,  the square root must cover the mean of squared
errors, not the sum.

#### 13. The MLP loss weights express task priority, not scale correction

**Where:** §3.6.3, eq. (4).

**Replace with:** "targets are standardised before the weighted loss is applied, so the
weights (1.0, 0.1, 0.5) express the relative research priority of the three tasks rather
than compensating for their differing scales."

**Why:** the original rationale,  that the weights equalise a roughly 1000× scale gap, 
stopped being true once target standardisation was added. Before that fix the MLP's Y1 R²
was approximately −390.

---

### Data and validity statements

#### 14. Macro controls are lag-corrected and annual

**Where:** §3.5.3.

**Add:** "World Bank annual series are dated to their publication, not to the start of the
year they describe, so a November event does not receive a figure published the following
year. Annual frequency against event-level timing remains a limitation: CBSL monthly CCPI,
daily LKR/USD and policy-rate series were not accessible."

#### 15. Stationarity testing covers all features

**Where:** §3.5.1.

**Add:** "ADF and KPSS are run on every engineered market feature. Exclusion is gated on
the ADF unit-root result; six raw price-level columns (`sma_*`, `ema_*`) are excluded and
replaced by stationary price-relative ratios."

#### 16. Report Y2's effective sample size

**Where:** wherever N is stated.

**Add:** "N = 64 events; **N = 61 for Y2**, since the 2000 workbook did not parse and
three events have no volume baseline. These targets are left missing rather than imputed."

#### 17. State the damage-coverage problem

**Where:** §3.3.2 and limitations.

**Add:**

> Only 17 of the 64 modelled events carry a real EM-DAT damage figure; the remaining 47
> are zero-filled. The damage features therefore behave substantially as an indicator of
> **EM-DAT reporting coverage**,  itself correlated with severity and recency,  rather
> than as a clean severity measure. The `log_damage_x_flood` interaction rests on 12
> events.

#### 18. Disclose the event-window overlap

**Where:** §3.3.2, where the truncation protocol is described.

**Add:** "29 of 64 recovery windows contain a later qualifying disaster within 90 days.
The truncation protocol is not applied; the overlap is quantified and disclosed instead."

**Why:** the SWOT implied the truncation was in use. It was not.

#### 19. Soften "first-ever"

**Where:** §1, §2.7.

**Replace with:** "the first study we are aware of to…",  a literature search cannot
establish that nothing exists.

#### 20. Replace causal language throughout

**Where:** everywhere SHAP results are discussed.

**Replace:** "X causes / drives / determines Y" with "X is associated with Y" or "X
contributes to the model's prediction of Y".

**Why:** SHAP attributes a *model's* output, not a causal effect, and under the
collinearity present here (several feature pairs at ρ ≈ 1.00, condition number ≈ 10¹⁷)
attribution within a correlated block is not even identifiable at the individual-feature
level.

---

### What to add that is not currently there

**A results-integrity paragraph.** State plainly that no model beats a naive baseline on
Y1; that performance is reported against two nulls, a constant-zero economic null and a
training-mean statistical null; and that intervals accompany every comparison. A negative
result reported this carefully is a stronger thesis than a positive one reported loosely.

**A selection-bias disclosure.** The split geometry, the feature count and the augmentation
configuration were all evaluated against the same folds that are reported. That carries
unquantifiable optimistic bias and should be stated rather than left for a reader to infer.

**A figure list.** The figures in `docs/figures/` are numbered and exported at 300 dpi for
citation. Each carries its sample size on the face of the figure.

---

## Amendments from the external-data pass (2026-09-11)

Five external sources were added after the missing-data audit. They change the sample
size, the sample period and the feature set, so several statements above and in the
thesis need re-numbering. Full specification: `docs/EXTERNAL_DATA_PRE_DECLARATION.md`,
written and committed before any of these features was scored.

#### 21. The sample is 74 events, not 64

**Current:** "64 EM-DAT-qualifying natural disasters between 2000 and June 2023."

**Replace with:** "74 EM-DAT-qualifying natural disasters between September 2000 and
November 2025. The ASPI series was extended beyond the local archive's 2023-06-28 cutoff
using countryeconomy.com daily closes, verified continuous against the archive over a
21-day overlap. This admits ten previously-excluded events, including both Storm Ditwah
dates (2025-11-20 and 2025-11-27)."

**Why:** the inclusion criteria never changed,  only the market series got longer. The
old cutoff was a data-availability artefact, and it excluded the thesis's own flagship
example. Walk-forward folds go from 3 to 4 and pooled out-of-fold test points from 30 to
**40**, which is the single largest improvement in statistical power in the study.

#### 22. Y2's effective N must be stated separately

**Current:** implies a single N throughout.

**Replace with:** "Y1 and Y3 are observed for all 74 events. Y2 (abnormal volume) is
observed for 61: the 2000 archive workbook records no share-volume column at all, and
countryeconomy publishes the index level but not volume, so the ten post-2023 events
carry no Y2. These rows are dropped per target, never imputed."

**Why:** this was verified directly,  `Data/2000 data.xls` has columns
`date | SECURITY_DA | PRICE` with no volume field, whereas the 2001 workbook's header
reads "CLOSING PRICE & SHARE VOLUME". The gap is a structurally absent field, not a parse
failure, and should be described as such.

#### 23. The severity variable is barely observed,  say so

**Current:** treats `financial_damage` as the severity measure.

**Add:** "EM-DAT records a financial damage figure for only 17 of the 64 originally
modelled events; the remaining 47 are zero-filled. The flood-damage interaction term
therefore rests on 12 events. Because EM-DAT records damage more often for large, recent
and internationally-reported events, these columns partly measure *reporting coverage*
rather than severity, and a coefficient on them must be interpreted accordingly."

**Why:** this is the study's real binding constraint and it is sharper than the N=64 point
the thesis currently leads with. An examiner who finds it unaided will treat it as a
concealed weakness rather than a stated limitation.

#### 24. Disaster severity now has an instrument-measured component

**Add to §3.3 (Data Sources):** "Hazard intensity is measured independently of any damage
assessment using NASA POWER daily reanalysis at seven district points (Colombo, Jaffna,
Batticaloa, Nuwara Eliya, Galle, Anuradhapura, Ratnapura): three-day accumulated
precipitation, its spatial spread across districts, the count of districts exceeding the
Sri Lanka Department of Meteorology's 50 mm/3-day heavy-rain advisory level, maximum
10 m wind speed, and precipitation relative to a 30-day pre-event baseline."

**Why:** unlike EM-DAT damage this has no missingness (100% coverage on all 74 events),
no reporting bias, and is knowable on the event day,  which also makes it the only
severity measure admissible in an ex-ante specification.

#### 25. District-level physical severity from DesInventar

**Add to §3.3:** "District-level physical severity is taken from DesInventar Sendai, the
UNDRR/UNDP national disaster loss database for Sri Lanka (130,018 dated records,
1965, 2020): districts affected, population affected, houses destroyed and damaged, and
deaths, aggregated over a [t−7, t+14] window around each event."

**State explicitly:** "DesInventar Sri Lanka populates its USD loss field in **zero** of
130,018 records and its local-currency field in 16. It supplies physical severity only
and does not recover the missing monetary damage figures. It also ends 2020-12-20, so 22
of the 74 events match no record; a `di_available` indicator carries that rather than a
zero being read as 'no damage'."

**Why:** 35 of the 47 zero-filled-damage events gain real measured severity from this
source, taking severity coverage from 17/64 to 52/74. That is a substantive improvement
and must not be overstated as recovering damage in money terms.

#### 26. Macro controls are daily, not annual

**Current:** "annual GDP growth and inflation from the World Bank."

**Replace with:** "Annual World Bank series are retained for scale, and daily LKR/USD
exchange-rate dynamics (FRED series `DEXSLUS`, 13,102 observations from 1973) supply the
day-level macro state: one-day and five-day log changes and 30-day realised volatility,
all computed strictly from data at or before t−1."

**Why:** matching an annual series to a day-0 event shock is indefensible, and it was the
only macro control the study had. This is a correctness fix, not a performance lever, and
should be presented as one.

#### 27. Election contamination is now measured, not just noted

**Add to the robustness section:** "Event windows were screened against 29 Sri Lankan
national election dates retrieved from Wikidata. The 2005-11-17 presidential election
falls four days before the 2005-11-21 flood event that carries the largest observed Y1
drop (−0.0753); a `days_to_election` distance and a ±5-day flag are included so this
confounder is controlled rather than merely acknowledged."

#### 28. §3.3.1's news-sentiment claim is still unsupported

**Current:** §3.3.1 describes a news-sentiment input.

**Replace with:** "News sentiment was specified in the original design but is not used.
GDELT, the only free source with the required historical depth, returned HTTP 429 on every
retrieval attempt including at 20-second spacing, and ReliefWeb's API requires a
registered application key. No sentiment feature is constructed, and no result in this
study depends on one."

**Why:** the claim currently describes an input the pipeline does not have. Removing it is
mandatory; describing the attempt is what makes the removal credible.

#### 29. Per-sector volume is unobtainable,  record why

**Add to the limitations:** "The sector panel models sector-level return and recovery but
not abnormal volume. The Colombo Stock Exchange's per-security yearly archives carry share
volume but no sector field: their `MAIN TYPE` column holds security-class markers
(N/P/R/U/X) and `SUB TYPE` holds only `0000`/`0001`, and no company-to-sector mapping is
available locally. Per-sector volume is therefore not derivable, which is why the panel
covers Y1 and Y3 only."

**Why:** this was tested directly rather than assumed, and it is the gap most likely to be
raised by an examiner who notices Y2 is the one target with measurable signal.

---

# Part 7. Full dated change log

Every methodology decision, in date order, including the rejected variants and the experiments that failed. This is the research record and nothing is removed from it.

## 10-Expert Methodology Audit,  CSE Disaster-Impact Prediction Model

**Subject:** "A Machine Learning Approach to Predicting the Impact of Natural Disasters on the Colombo Stock Exchange" (S.D.S.H. Samarakkodi, IM/2021/007, University of Kelaniya; supervisor Dr. Thilini Mahanama)

**Audit basis:** Direct inspection of the `src/` source and the notebook pipeline, plus executed outputs from complete pipeline runs. (Paths in this document predate the 2026-09 repository restructure, which flattened `disaster_finance_predictor/` into the repository root; `src/`, `notebooks/` and `tests/` now sit at the top level.) **Every number in this document is copied from an actual execution.** Nothing is estimated, extrapolated, or invented. Where a figure is unavailable, it is marked "not measured".

**Authoritative methodology:** the thesis (July 2026). The April 2026 proposal is historical context only.

---

### 1. Research Understanding

The study asks: *how does the Colombo Stock Exchange respond to natural disasters, and can that response be predicted?* It is operationalised as a supervised multi-target regression over disaster events, not over trading days.

Three continuous targets per event:

| Target                | Definition as implemented                              | Range observed      |
| --------------------- | ------------------------------------------------------ | ------------------- |
| Y1`aspi_log_return` | `ln(P_t / P_{t-1})` on the event's first trading day | −0.0753 to +0.0394 |
| Y2`abnormal_volume` | `V_t / mean(V_{t-30..t-1}) − 1`                     | −0.87 to +1.94     |
| Y3`recovery_days`   | Days until ASPI regains`P_{t-1}`, capped at 90       | 0 to 90, median 0   |

Unit of analysis: **one qualifying disaster event**. N = 64 modelled events (2000-01 to 2023-06).

---

### 2. Proposal  Thesis  Implementation Evolution

| Component     | Proposal (Apr 2026)                                 | Thesis (Jul 2026)                                              | Implementation                                        | Verdict                                                                                        |
| ------------- | --------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Validation    | Stratified 5-fold CV inside 80% train block (§7.5) | Chronological walk-forward; k-fold explicitly banned (§3.7.1) | Walk-forward outer;**k-fold was running inner** | Thesis correct; implementation violated it,  now fixed                                        |
| Target Y1     | "ASPI percentage change"                            | §3.2.2: raw log return                                        | Raw log return                                        | Terminology should be corrected to "log return"                                                |
| Oversampling  | SMOTE                                               | Time-Aware SMOGN                                               | Custom SmoteR-style implementation                    | Thesis correct                                                                                 |
| Recovery      | Trading days, cap 90                                | Trading days, cap 90                                           | **Calendar days**, cap 90                       | Implementation deviated,  now fixed                                                           |
| Models        | +Logistic Regression                                | OLS/Ridge, SVR, RF, XGBoost, MLP                               | Ridge, RF, XGBoost, MLP                               | Logistic Regression correctly dropped (targets are continuous);**SVR never implemented** |
| Sentiment/NLP | Aggregated news sentiment listed (§3.3.1)          | Explicitly excluded (Table 4 note)                             | Not implemented                                       | Thesis correct; proposal wording superseded                                                    |
| Epidemics     | Included in scope (§4.5)                           | Excluded from acute-shock architecture                         | Excluded                                              | Thesis correct                                                                                 |

**Unresolved internal contradiction:** thesis §3.3.1 states "aggregated sentiment scores from financial news were used", while Table 4's note says textual NLP sentiment was "explicitly excluded". The implementation excludes it. §3.3.1 must be corrected.

---

### 3. Existing Implementation Reconstruction

```
EM-DAT xlsx (86 records)          07Market Indices-Daily.xls        Yearly per-security .xls (3 schema eras)
        |                                    |                                    |
   emdat_loader                      cse_raw_loaders                     cse_raw_loaders
        |                              ASPI close                        summed share volume
        v                                    v                                    v
  filter: non-biological, affected >= 1000  -->  forward-fill (never zero-fill)  <--
        | 74 qualifying                              |
        v                                            v
  in-scope filter: event within archive coverage (<= 2023-06-28)  --> 64 events
        |
        v
  FeatureEngineer.engineer_market_features   (lags t-1/2/3/5, sma/ema 5/10/20 on shifted price,
        |                                     rolling_std 5/10/20/30 on shifted returns, squared_return)
        v
  FeatureEngineer.build_targets  --> Y1, Y2, Y3
        |
        v
  merge_asof(backward) on asof_date = event_date - 1d  <-- market features
                                                       <-- World Bank macro (annual, wbgapi, LIVE)
                                                       <-- S&P 500 (yfinance, LIVE)
        |
        v
  event table: 64 rows x 32 features  (recency features, damage_to_gdp, log_damage_x_flood added here)
        |
        v
  generate_walk_forward_splits(len(X), 30, 10, 10)  --> 3 folds, ROLLING window
        |
        v
  per fold:  augment_fold -> time_aware_smogn(minority = Y3 > 30, +/-5y window)   [TRAIN ROWS ONLY]
        |
        v
  per target: select_top_features(RF importance, k=20)  [fit on augmented train only]
        |
        v
  Ridge(alpha=1.0) | RF+GridSearchCV(cv=3) | XGBoost+GridSearchCV(cv=3)
  MLP: separate loop, subprocess (Windows torch DLL workaround), y-StandardScaler, weights (1.0,0.1,0.5)
        |
        v
  ensemble (inverse-RMSE blend) | stacked (expanding-window non-negative LinearRegression meta-learner)
        |
        v
  RMSE / MAE / R2 per-fold + POOLED  --> summary table
  directional accuracy, precision/recall/F1, AUC  (Y1 thresholded at 0)
  rolling conformal prediction intervals
        |
        v
  SHAP (global summary + local waterfall on largest real ASPI drop)
```

**Differences from the thesis's stated pipeline:**

| Component            | Thesis says                                          | Code did                                                               | Severity     | Action                                          |
| -------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------- | ------------ | ----------------------------------------------- |
| Inner CV             | No k-fold anywhere (§3.7.1)                         | `GridSearchCV(cv=3)` = `KFold(shuffle=False)`                      | **P0** | Replaced with`TimeSeriesSplit` on real rows   |
| Walk-forward         | "Training set to grow over time" (§3.7.1)           | **Rolling** window,  train start advances, early events dropped | **P1** | Expanding option added; both reported           |
| Y3 unit              | Consecutive trading days                             | Calendar days                                                          | **P1** | Fixed to positional trading-day distance        |
| Macro alignment      | "as-of-date alignment... mandatory" (§3.3.1 spirit) | Annual value dated 1 January of its own year                           | **P0** | Re-dated to Y+1-07-01                           |
| Ridge                | Baseline model                                       | Fit on**unstandardised** features, alpha never tuned             | **P0** | `StandardScaler` + in-fold `RidgeCV`        |
| Stationarity         | ADF/KPSS, difference failures (§3.5.1)              | Tested`log_return` only; 6 price-level features untested             | **P0** | All features tested; unit-root columns excluded |
| Missing targets      | 100% real data claim                                 | `y.fillna(0.0)` fabricated 3 Y2 values                               | **P0** | NaN preserved, masked per target                |
| Affected threshold   | `>= 1000`                                          | `> 1000`                                                             | P3           | Fixed                                           |
| Overlapping events   | Truncation protocol (§3.3.2)                        | `truncate_overlapping_windows()` defined, **never called**     | **P1** | Contamination now quantified and disclosed      |
| SVR                  | Listed as model (§3.6.1)                            | Not implemented                                                        | P2           | Either implement or remove from methodology     |
| MultiOutputRegressor | Described (§3.6.4)                                  | Not used,  separate per-target fits                                   | P2           | Correct the description (see §18)              |

---

### 4. Methodology vs Code Audit

Covered in the table above. The single most damaging mismatch is the **inner k-fold**: §8 of the notebook and §3.7.1 of the thesis both state, in bold, that no k-fold is used anywhere, while two lines of code ran `KFold(shuffle=False)` inside every walk-forward fold,  training on later events to select hyperparameters for earlier ones. This was compounded because SMOGN appends synthetic rows at the tail of the training frame, so the last inner validation block was disproportionately synthetic: the model was partly selected on its ability to predict its own interpolated output.

---

### 5. Unit-of-Analysis Audit

**One training sample = one disaster event.** Not one trading day.

- Raw EM-DAT records: 86
- After excluding biological and applying `affected >= 1000`: 74
- After restricting to events inside real archive coverage: **64**
- Effective N for Y2: **61** (3 events have no volume data,  the 2000 workbook failed to parse)

There is no pseudo-replication: each event contributes exactly one row. The daily market series is used only to *construct* event-level features and targets, never as independent observations. The thesis is correct not to claim thousands of samples.

**Consequence for model complexity:** with 64 events and a 30/10/10 geometry, each fold trains on 30 rows. Against 32 features this is p > n before selection. This single fact should govern every capacity decision in the study.

**Residual dependence:** events are not fully independent. Y3 windows look forward up to 90 days, and Sri Lanka's monsoon clustering means some windows contain a subsequent qualifying event. This is quantified in the notebook rather than assumed away.

---

### 6. Target Audit

#### Y1,  ASPI market impact

**Decision: KEEP, rename.**

Implemented as the raw continuously-compounded log return, which matches thesis §3.2.2/Table 4. The thesis's prose label "ASPI Percentage Change" is inaccurate: a log return is not a percentage change (they diverge as magnitude grows). Rename to **"ASPI log return"** throughout.

Should it be an abnormal return (AR) instead? **No.** §2.2.3's market-model AR/CAR formulation is presented in the thesis as literature-review background that the thesis explicitly critiques and moves away from. More decisively: computing AR requires a market model `E(R_i,t) = α + β·R_m,t`, which needs a *market* index distinct from the asset. Here the asset **is** the market index. There is no valid benchmark to regress against, so AR is not identified. The raw log return is the correct operationalisation.

#### Y2,  Abnormal trading volume

**Decision: KEEP formula, correct the terminology.**

Implemented as `V_t / V̄_pre − 1`, which is the standard normalised abnormal-volume form and is preferable to the raw ratio (it is centred at zero, so "no abnormality" is 0).

**The name is wrong.** The thesis calls this "trading volume crash magnitude". The variable measures *deviation in either direction*, and its observed maximum is **+1.94**,  nearly triple baseline volume. A large positive value is a volume **spike**, which the thesis's own literature review (§2.3.2) correctly identifies as the signature of panic selling. Calling it a "crash" inverts the meaning. Rename to **"abnormal trading volume"** and describe it as a liquidity-disturbance magnitude, not a crash.

#### Y3,  Market recovery time

**Decision: MODIFY.**

Three distinct defects:

1. **Unit deviation (fixed).** The search window was 91 trading rows but the returned value was a calendar-day difference. The 90-day cap therefore bit at roughly 62 trading days, and an event recovering after that was indistinguishable from one that never recovered.
2. **Right-censoring is real and currently ignored.** For a non-recovering event we know `T > 90`, not `T = 90`. Assigning 90 is a biased point estimate.
3. **Y3 is not an independent target.** The recovery window includes the event day itself and the baseline is `P_{t-1}`, so whenever `P_t >= P_{t-1}`,  that is, whenever **Y1 >= 0**,  the event day satisfies the recovery condition and **Y3 = 0 exactly**. `Y3 = 0` and `Y1 >= 0` are the same event by construction. This is why Y3's median is 0 and its 25th percentile is 0.

**Censoring treatment,  recommendation:** Approach C (two-stage hurdle), not Approach D (survival).

- Approach A (regression with cap),  current, biased.
- Approach B (recovered events only),  discards the most severe events. Rejected.
- **Approach C (hurdle): P(recovery within 90) × E[duration | recovered].** Matches the actual distribution (a point mass at 0, a right tail, a point mass at the cap). Stage 1 is a classifier, which at N=64 is far better supported than a survival model.
- Approach D (Cox / AFT / Random Survival Forest),  methodologically the "textbook" answer for censoring, but at 64 events with a censoring point that is itself an artefact of the 90-day design choice, a survival forest would be fitting hazard curves to a handful of tail points. **Rejected on sample-size grounds, per the master prompt's own caution.**

Because of the Y1Y3 identity, Stage 1 of the hurdle model is *almost* a restatement of predicting the sign of Y1. That must be stated openly rather than presented as a second independent finding.

---

### 7. Information-Availability Audit

This is the most consequential section of the audit.

| Feature                                | Available at disaster onset? | Publication delay | Leakage risk       | Keep/Remove              |
| -------------------------------------- | ---------------------------- | ----------------- | ------------------ | ------------------------ |
| `financial_damage`                   | **No**                 | Weeks, months     | **CRITICAL** | Remove for ex-ante model |
| `log_financial_damage`               | **No**                 | Weeks, months     | **CRITICAL** | Remove for ex-ante       |
| `population_affected`                | **No** (final figure)  | Days, weeks       | **CRITICAL** | Remove for ex-ante       |
| `log_population_affected`            | **No**                 | Days, weeks       | **CRITICAL** | Remove for ex-ante       |
| `damage_to_gdp`                      | **No** (both parts)    | Months            | **CRITICAL** | Remove for ex-ante       |
| `log_damage_x_flood`                 | **No** (damage half)   | Weeks, months     | **CRITICAL** | Remove for ex-ante       |
| `disaster_Flood/Storm/...`           | Yes                          | None              | None               | Keep                     |
| `days_since_last_disaster`           | Yes                          | None              | None               | Keep                     |
| `disasters_trailing_365d`            | Yes                          | None              | None               | Keep                     |
| `log_return`, `lag_return_t-*`     | Yes (t−1 close)             | None              | None               | Keep                     |
| `sma_*`, `ema_*`                   | Yes but non-stationary       | None              | Extrapolation      | Replaced by ratios       |
| `price_to_sma_*`, `price_to_ema_*` | Yes                          | None              | None               | Keep                     |
| `rolling_std_*`, `squared_return`  | Yes                          | None              | None               | Keep                     |
| `gdp_growth_pct`                     | **No** as dated        | ~6, 18 months     | **HIGH**     | Re-date (done)           |
| `inflation_cpi_pct`                  | **No** as dated        | ~6, 18 months     | **HIGH**     | Re-date (done)           |
| `gdp_current_usd`                    | **No** as dated        | ~6, 18 months     | **HIGH**     | Re-date (done)           |
| `sp500_log_return`                   | Yes                          | None              | None               | Keep                     |

**Verdict: the thesis currently supports retrospective (ex-post) prediction, not real-time early warning.**

Six of 32 features are EM-DAT post-hoc assessments. On the day before a flood, its eventual total damage and total affected are unknown. The README, thesis §1.2.3 and §4.4 all describe an "early-warning tool"; that claim is **NOT SUPPORTED** by the current feature set.

**Recommendation,  two models, explicitly separated:**

- **Model A (ex-ante / event-onset):** disaster type, season, trailing disaster count, days since last disaster, historical type-average severity, all market features, macro with correct publication lag. Answers *"a flood has just begun,  what happens?"*
- **Model B (ex-post / attribution):** the current feature set including realised damage. Answers *"given a disaster of this measured severity, what was the market response?"*

Model B is the study as it stands and is a legitimate research object. Model A is the one that would justify the early-warning language. Reporting both, and labelling which is which, converts a fatal framing problem into a genuine two-experiment contribution.

---

### 8. Data-Leakage Audit

| Source                                                | Type                      | Severity           | Status                                                                                                                   |
| ----------------------------------------------------- | ------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Global SMOGN before fold split                        | Temporal                  | CRITICAL           | Fixed earlier in development (documented in §8 of the notebook)                                                         |
| Walk-forward boundaries computed on post-SMOGN length | Temporal                  | CRITICAL           | Fixed,  folds now cut on real`len(X)`                                                                                 |
| `GridSearchCV(cv=3)` = KFold inside each fold       | Temporal                  | **CRITICAL** | Fixed, `TimeSeriesSplit` on real rows only                                                                            |
| Hyperparameters selected partly on synthetic rows     | Target                    | HIGH               | Fixed,  selection on real rows, refit on augmented                                                                      |
| Annual macro dated 1 January of its own year          | Availability              | **CRITICAL** | Fixed,  re-dated Y+1-07-01                                                                                              |
| Ex-post damage features as onset predictors           | Availability              | **CRITICAL** | Disclosed; requires reframing or Model A                                                                                 |
| SMA/EMA computed on unshifted price                   | Temporal                  | CRITICAL           | Fixed earlier (`.shift(1)`)                                                                                            |
| Feature selection inside fold on training rows        |,                         | NONE               | Correct as implemented                                                                                                   |
| Scalers fit inside fold                               |,                         | NONE               | Correct as implemented                                                                                                   |
| ADF/KPSS run on the full series                       | Temporal                  | LOW                | Stationarity is a structural property, not a fitted parameter; the test does not transfer target information. Disclosed. |
| SMOGN minority quantile thresholds                    |,                         | NONE               | Computed on training rows only                                                                                           |
| Y3 window overlapping a later event                   | Target                    | MODERATE           | Quantified and disclosed; truncation function still uncalled                                                             |
| Model/config selection on reported folds              | **Optimistic bias** | HIGH               | Disclosed in a dedicated subsection (see §29)                                                                           |

**On the last row,  this is the one that cannot be fixed by code.** The SMOGN configuration was reverted after its held-out scores were seen. K, the split geometry, and the primary-model choice were all evaluated against the same folds that are reported. This is disclosed explicitly rather than concealed, and the pre-registered 25% synthetic-share cap now provides a score-independent justification for the SMOGN revert.

---

### 9. Event-Window and Overlapping-Event Audit

**30-day pre / 90-day post: KEEP.** Justified a priori by the thesis's own argument (delayed price discovery in a semi-strong-inefficient frontier market) and, critically, **not tuned against test performance**. Systematically searching [-5,+20], [-10,+30], [-20,+60] and picking the best would be window-shopping on the test set. The current single pre-registered window is the more defensible choice, and this audit recommends against the window sweep the master prompt offers as an option.

The three roles of the pre-event window must be separated in the write-up, because they currently blur:

- **Baseline window**,  the 30-day volume mean that defines Y2, and `P_{t-1}` that defines Y1 and Y3.
- **Feature window**,  the lags, moving averages and volatility measures, all halted at t−1.
- **Estimation window**,  *does not exist here*, because no market model is estimated (see §6, Y1). The thesis should stop calling it an estimation window.

**Overlapping events,  recommendation: Option C + D over Option A.**

`truncate_overlapping_windows()` exists in `preprocessor.py` and is never called. Rather than wire in Option A (truncation), which shortens Y3 for exactly the events most likely to be severe and introduces a second censoring mechanism on top of the 90-day cap, the better treatment at this N is:

- **Option C:** add a `concurrent_disaster_in_window` indicator (already partly available via `disasters_trailing_365d`).
- **Option D:** the existing `disasters_trailing_365d` already encodes cumulative intensity.

Option B (exclude overlapping events) would drop a substantial fraction of a 64-event sample. Option E (event clusters) would reduce N further. Both rejected on sample-size grounds.

---

### 10. Time-Aware SMOGN Audit

#### What is actually imbalanced?

Y3 has 10 minority events (`Y3 > 30`) out of 64,  **15.6%**. Y1 and Y2 are roughly symmetric and have no rare-event structure in the same sense.

#### Is the implementation genuinely "time-aware"?,  Yes, with corrections.

| Property                                              | Status                                                                                                                                                                                                                                                                                                                   |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Neighbours restricted temporally                      | **Yes**,  ±5 year window, enforced                                                                                                                                                                                                                                                                               |
| Future disasters can seed historical synthetic events | **Yes within a fold's training rows**,  the window is symmetric, not backward-only. Since all rows are training rows this does not leak into test, but it does mean a 2010 event can be interpolated with a 2013 event. Defensible (both are past data at prediction time) and disclosed.                         |
| Runs after fold creation, training rows only          | **Yes**,  verified                                                                                                                                                                                                                                                                                                |
| Disaster categories mixed unrealistically             | **Was yes,  now fixed.** Previously a Flood could interpolate with a Drought, producing rows like `disaster_Flood=0.6, disaster_Storm=0.4`. Now same-type only, with the one-hot block copied verbatim.                                                                                                         |
| Synthetic damage/population combinations plausible    | **Was no,  now fixed.** Derived features (`log_financial_damage`, `damage_to_gdp`, `squared_return`, `log_damage_x_flood`) were interpolated *independently of their parents*, so `log_financial_damage != log1p(financial_damage)` on every synthetic row. Now recomputed from interpolated parents. |
| Categorical types preserved                           | **Now yes**                                                                                                                                                                                                                                                                                                        |
| Synthetic rows violate real financial relationships   | **Partly, unavoidably.** `rolling_std_*` is a convex functional of the price path, so interpolating two rows over-states volatility. Documented as an unfixable limitation.                                                                                                                                      |
| Gaussian-noise level                                  | **Branch deleted.** It applied `N(0, 0.01)` *absolute* noise to columns measured in millions (producing near-duplicate rows) while that same 0.01 is ~70% of Y1's standard deviation,  capable of flipping Y1's sign while leaving Y3 at 90, a combination that cannot occur given `Y3 = 0 ⟺ Y1 >= 0`.     |
| Relevance function φ(y)                              | **Not implemented,  deliberately.** Canonical SMOGN's φ is a PCHIP curve through boxplot control points. Y3's median and 25th percentile are both 0, so the boxplot construction is degenerate. The binary `Y3 > 30` threshold is the step-function instantiation of φ and should be described as such.       |
| Majority under-sampling                               | **Not implemented,  deliberately.** At 30 training rows, discarding real observations to balance a ratio is the most expensive available action and contradicts the study's 100%-real-data claim. This is a supported configuration of canonical SMOGN (`under_samp=False`), not a deviation.                   |

#### Verdict

**KEEP + OPTIMIZE.** The label "Time-Aware SMOGN" is now honest. Prior to these corrections it was SmoteR-with-a-cutoff producing physically impossible events.

---

### 11. Preprocessing Audit

**Scaling,  MODIFY (done).** The thesis says scaling is applied within each training fold, which is leakage-safe and correct. But scaling was applied *inconsistently across models*: the MLP scaled X and y, while **Ridge received unstandardised features**. Since L2 is not scale-equivariant and the design matrix mixed ASPI price levels (~1e4), raw USD damage (~1e9) and 0/1 flags, `alpha=1.0` applied almost no shrinkage to the damage columns and crushed the binary ones. Random Forest and XGBoost correctly need no scaling. Model-specific pipelines are now used.

**Stationarity,  MODIFY (done).** The thesis (§3.5.1) mandates ADF/KPSS with differencing of failures. The implementation tested `log_return` alone and reported "stationary", while six engineered price-level columns went untested. All engineered market features are now tested.

The exclusion rule matters and this audit corrects an error made during remediation: **gate on the ADF unit-root result only.** The volatility columns (`rolling_std_*`, `squared_return`) reject the ADF unit-root null decisively (p ≈ 1e-11 and smaller) while failing KPSS level-stationarity,  the ordinary signature of persistent, mean-reverting volatility clustering, not of a trending level. Requiring both tests to agree discarded 11 of 22 features including the thesis's own mandated 30-day panic proxy. A unit root is what creates extrapolation risk under a chronological split; KPSS-persistence does not.

**Imputation,  MODIFY (done).** `y = dataset[TARGET_COLS].fillna(0.0)` fabricated three Y2 observations, asserting "volume exactly at its 30-day baseline" for events whose source workbook failed to parse. This is the same zero-fill the notebook's §3 explicitly rejects for volume itself. Targets now keep their NaNs and are masked per target; Y2's effective N is reported as 61.

`X = dataset[FEATURE_COLS].fillna(0.0)` remains and is a weaker but real concern: 0.0 in `damage_to_gdp` reads as "no damage" rather than "unknown". Recommend an explicit missingness indicator.

---

### 12. Feature Engineering Audit

**Collinearity is severe and largely unaddressed.** `sma_5/10/20` and `ema_5/10/20` are six smoothed versions of the same price series and are mutually ~99% correlated; `financial_damage`, `log_financial_damage`, `damage_to_gdp` and `log_damage_x_flood` are four transforms of one quantity. This is why K=20  K=10 barely moved Random Forest (−0.322  −0.298) but moved Ridge substantially (−1.634  −0.803): the binding problem was collinearity and non-stationarity, not the feature count.

**A concrete consequence found in the SMOGN neighbour search:** the standardised distance metric included `financial_damage` *and* its three derived children, so four of 32 columns were the same underlying quantity and neighbour selection was dominated by damage similarity. Derived columns are now excluded from the distance metric.

**What the damage features actually measure.** EM-DAT records no damage estimate for the large majority of Sri Lankan events, and the loader zero-fills those. The damage features therefore behave substantially as an **indicator of EM-DAT reporting coverage**, which is itself correlated with severity and recency. The `log_damage_x_flood` interaction was justified on the grounds that Flood is 51/74 of the qualifying set,  but the relevant denominator is *floods that have a damage figure*, which is far smaller. This must be stated in the limitations.

**Recommended additions,  none.** With 64 events, the correct direction is fewer features, not more. Deaths, injured, homeless, duration and geographic spread are all available in EM-DAT but every one of them is (a) ex-post and (b) collinear with the severity measures already present. Adding them would worsen both the p/n ratio and the information-availability problem.

**Recommended removals:** the six raw price-level moving averages (done,  replaced by stationary `price_to_sma_*` / `price_to_ema_*` ratios).

---

### 13. Baseline Evaluation

**This was the single largest reporting gap and it is now closed.**

Before this audit the study compared machine-learning models against Ridge and against nothing else. Two deployable nulls have been added, computed on exactly the same folds:

- `naive_zero`,  predict 0 for every event. The **economic** null: "the disaster had no measurable effect". All three targets are defined so that 0 is the meaningful no-effect value.
- `naive_train_mean`,  predict the training fold's mean (never the test mean). The **statistical** null that makes R² interpretable.

**Measured result (Phase D run, calendar-day Y3):**

| Target | `naive_zero` RMSE | Models beating it                                                     |
| ------ | ------------------- | --------------------------------------------------------------------- |
| Y1     | 0.0084              | **0 of 6**                                                      |
| Y2     | 0.5142              | 3 of 6 (ridge, mlp, ensemble)                                         |
| Y3     | 30.17               | 6 of 6, **but `naive_train_mean` (27.95) beats every model** |

**This corrects an earlier claim.** A full-sample approximation had suggested every model beat the no-effect null on Y1. Computed fold-wise on the actual test folds, that is false. The honest statement is: **no model beats a naive baseline on Y1; only Ridge clears both nulls on Y2; on Y3 no model beats the training-mean predictor.**

---

### 14, 17. Model Evaluations

#### 14. Random Forest,  KEEP + OPTIMIZE

Best or near-best across targets in early configurations, and the most stable. But: train-fit R² of **0.883** on Y3 against a held-out **−0.146** is a 1.03-point gap,  the model is memorising. Grids permitted `max_depth=None` with `min_samples_leaf=1` on 33-row folds, i.e. trees isolating individual events. Recommend bounding capacity a priori by fold size (`min_samples_leaf >= 3`, `max_depth <= 4`) rather than by test score.

#### 15. XGBoost,  KEEP + OPTIMIZE

Was the worst-behaved model before the stationarity fix (Y1 pooled R² −1.617), and improved most from it (−0.236). That pattern is diagnostic: boosting was extrapolating hardest off the end of the training price range. Same capacity-bounding recommendation.

#### 16. SVR,  **REPLACE (with nothing)**

Specified in thesis §3.6.1 and **never implemented**. Either implement it or delete it from the methodology. Given that Ridge (linear), RF and XGBoost (trees) and MLP (network) already span the model-family space, and that SVR adds a kernel and two hyperparameters to tune on 30 rows, this audit recommends **removing it from the thesis** rather than adding it to the code. State the removal and the reason.

#### 17. MLP,  KEEP + OPTIMIZE

Architecture (one hidden layer, 64 units, dropout 0.5, three heads) is appropriately constrained for the sample size. Correctly rejects LSTM/Transformer.

Two real issues:

1. **No early stopping, no validation split.** 200 fixed full-batch epochs. On 33 rows this is a capacity risk, though dropout 0.5 mitigates it.
2. **The loss-weight rationale is now wrong.** The notebook states the weights `(1.0, 0.1, 0.5)` are the thesis's eq.(4) mechanism for the scale gap. Since y is standardised per fold *before* the loss is applied, all three per-target MSE terms are already unit-variance and the ~1000× raw-scale gap is gone before the weights act. The weights now express **task priority** (Y1 primary), not scale equalisation. This is a defensible position,  but it must be described accurately. Do not re-tune them; that would be an untuned hyperparameter change with no a-priori basis.

**Y3 transform:** `log1p` is now applied to Y3 inside the MLP before the y-scaler, matching the tree/linear models. This is not redundant with standardisation: standardisation is affine and fixes *scale*; log1p is monotone-nonlinear and fixes *shape*. Y3 after standardisation is still majority-zero, right-skewed and hard-capped.

---

### 18. Multi-Output vs Multi-Task Learning

**The thesis's description is wrong and must be corrected.**

Thesis §3.6.4 states that a `MultiOutputRegressor` wrapper is used for the tree models. The code does not use `MultiOutputRegressor` at all,  it fits an entirely separate estimator per target inside the fold loop, each with its own feature selection and its own hyperparameter search.

More importantly, **neither is multi-task learning.** Fitting `f1(X)Y1`, `f2(X)Y2`, `f3(X)Y3` independently is *multi-output prediction*. There is no shared representation and no cross-target information transfer. §12's cross-check table also claims a "custom weighted-RMSE scorer averaged across all 3 targets (thesis eq.(4) weights)",  the code uses `neg_root_mean_squared_error` on one target at a time.

**Only the MLP is genuinely multi-task**, via its shared hidden layer.

Required wording change: describe the tree pipeline as **per-target independent models**, and reserve "multi-task learning" for the MLP alone.

**Is joint learning justified?** The MLP is not the best model on any target in the measured results. There is no evidence of positive transfer. State this plainly rather than retaining MTL because it was proposed.

---

### 19. Recovery Survival-Analysis Decision

Covered in §6. **Recommendation: two-stage hurdle (Approach C), not survival (Approach D).** Rationale: N=64 with ~10 tail events cannot support hazard estimation; the censoring point is a design artefact rather than a natural end-of-observation; and the hurdle's first stage matches the actual point-mass-at-zero structure. Disclose that stage 1 is near-equivalent to predicting `sign(Y1)`.

---

### 20. Validation Redesign

**Recommended final design:**

1. **Outer:** chronological walk-forward, **expanding** window (thesis §3.7.1 says "the training set to grow over time"; the code rolls). Test indices are identical under both, so the comparison is exactly like-for-like,  nothing is added to or removed from the evaluation set.
2. **Inner:** `TimeSeriesSplit` on the fold's **real** training rows for all hyperparameter selection; refit the winner on real + synthetic.
3. **Report both** the primary 30/10/10 and the dense 20/5/5 geometry at equal prominence.

**Leave-One-Disaster-Out:** rejected. It breaks chronology (training on later events to predict earlier ones),  the exact failure the thesis bans.

**Leave-One-Disaster-Type-Out:** worth reporting **descriptively** as a stress test, not as a statistical claim. With Drought n=5 and the singletons pooled, per-type inference is not supportable.

**Embargo/purging:** the honest position is that Y3's 90-day forward window can overlap a subsequent event that falls in a later fold. Purging would cost folds the study cannot spare. Recommend **disclosing** the overlap count rather than implementing an embargo,  and stating that as a limitation.

---

### 21. Hyperparameter Optimization

**KEEP + OPTIMIZE.** Grid search is correct here; the master prompt's alternatives (Optuna/TPE, Bayesian) would search a *larger* space on 30 rows, which increases selection overfitting. The thesis's own §3.7.2 justification for tuning ("using default values will result in severe overfitting") is sound but was applied inconsistently,  RF and XGBoost were tuned while Ridge sat at its library default `alpha=1.0`. Now fixed.

Grids are deliberately small (8 combinations each) because a larger grid made the search itself the bottleneck (>900 s per fold). This is a real, disclosed constraint.

**Nested selection:** full nested CV is not supportable at this N. The implemented compromise,  inner `TimeSeriesSplit` on real training rows, outer walk-forward for reporting,  is the most defensible available, and its limitation (inner folds of 9/16/23 rows are genuinely noisy) is stated. The correct response to noisy inner selection is to *shrink the search space* so that any pick is acceptable, not to search harder.

---

### 22. Metric Framework

| Target | Primary                      | Secondary                                                                                  | Baseline it must beat                       |
| ------ | ---------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------- |
| Y1     | RMSE                         | MAE, pooled R², directional accuracy**vs majority baseline**, AUC **with CI** | `naive_zero`, `naive_train_mean`, Ridge |
| Y2     | RMSE                         | MAE, pooled R²                                                                            | `naive_zero`, `naive_train_mean`, Ridge |
| Y3     | MAE**in trading days** | RMSE, median AE, pooled R²                                                                | `naive_zero`, `naive_train_mean`, Ridge |

**Pooled R² is the correct R² to report** (concatenate all out-of-fold predictions, compute once),  per-fold R² on a 10-point window is numerically unstable. Both are shown, with pooled designated primary.

**Two definitional corrections required in the thesis:**

- §3.8.1 states "RMSE measures the total variation in the actual data explained by the ML algorithm." **This is wrong.** RMSE = √(mean squared error); it is an error magnitude in the target's units. R² is the explained-variance measure. Correct the sentence.
- The claim that pooled R² is computed over "N≈64-74 real points" is wrong: it is computed over **30** pooled out-of-fold test points (20 for the stacked model). Corrected in the notebook.

---

### 23. Rare-Event Evaluation

**Not yet measured.** The pipeline does not currently report performance separately for severe versus ordinary events. This is a genuine gap and is listed in the next-experiments section. The required table:

| Target | Overall | Common events | Rare events (top decile severity) | Worst event |
| ------ | ------- | ------------- | --------------------------------- | ----------- |

Given that the study exists to predict catastrophic impacts, a model with acceptable overall RMSE and poor performance on the 2004 tsunami and the 2005-11-21 event would not meet the research objective. This must be measured before any performance claim is made.

---

### 24. Ablation Study

**Measured,  the stationarity ablation (Phase C  Phase D), pooled R²:**

| Model / target | Before (non-stationary price levels in) | After (stationary ratios) |
| -------------- | --------------------------------------- | ------------------------- |
| ridge Y1       | −0.407                                 | **−0.278**         |
| ridge Y2       | −0.015                                 | **+0.053**          |
| ridge Y3       | −0.187                                 | **−0.122**         |
| xgboost Y1     | −1.617                                 | **−0.236**         |
| xgboost Y2     | −0.977                                 | **−0.613**         |
| mlp Y1         | −1.547                                 | **−0.299**         |
| mlp Y2         | −0.234                                 | **−0.025**         |
| stacked Y1     | −0.775                                 | **−0.134**         |
| RF Y1          | −0.315                                 | −0.412 (worse)           |

**Measured,  the SMOGN over-augmentation ablation:** widening the minority mask to the union of Y1/Y2/Y3 quintile tails with 2 draws per row produced 26, 28 synthetic rows against 30 real per fold and made **every model on every target worse** (RF Y1 pooled R² −0.32  −1.39; Ridge Y3 RMSE 36.6  62.9). Reverted. This variant is also rejected independently by the pre-registered 25% synthetic-share cap (it reaches 46, 48%), so the revert does not rest on having seen its scores.

**Measured,  feature count (K=20 vs K=10), pooled R²:** K=10 better on 6 of 9 model-target cells (ridge Y1 −1.634  −0.803; ridge Y2 −3.539  −1.250; RF Y2 −0.120  −0.057) but **destroys the study's only positive R²** (ridge Y3 +0.038  −0.079). K=20 retained; both reported.

**Not yet measured,  the study's central ablation.** Market-only vs disaster-only vs market+disaster vs +macro. This directly answers *"do disaster variables add predictive information beyond ordinary market history?"*,  the question the entire thesis exists to answer,  and it has never been run. **This is the single most important missing experiment.**

**Measured 2026-09-16 (E08/E09, RF, all 3 targets, same folds/purge/median-impute/
selection/inner-CV, `use_smogn` the only difference, see
`notebooks/04_modeling_regression.ipynb` §4.2b, `artifacts/tables/smogn_ablation.parquet`):**

| target | RMSE no-SMOGN | RMSE with-SMOGN | pooled R2 no-SMOGN | pooled R2 with-SMOGN | verdict |
|---|---|---|---|---|---|
| Y1_ASPI_5D_Forward_LogReturn_Pct | 2.9104 | 2.8834 | 0.0584 | 0.0698 | SMOGN helps |
| Y2_abnormal_volume | 0.4948 | 0.4822 | 0.1741 | 0.2242 | SMOGN helps |
| Y3_recovery_days | 17.0321 | 16.9179 | -0.1363 | -0.1264 | SMOGN helps |
| Y1_EventWindow_0_10_LogReturn_Pct | 4.5143 | 4.5603 | 0.0043 | 0.0063 | RMSE slightly worse, pooled R2 slightly better |

**Decision: kept ON for all 3 targets.** 3 of 4 targets improve on both metrics; the
4th (EventWindow_0_10) moves in opposite directions on the two metrics by ~1%, which is
noise at this fold count (n=40 pooled test points), not a signal worth a per-target
special case. This closes P2-6 ("SMOGN on/off never ablated"), the augmentation's
worth is now isolated and measured, not assumed.

---

### 25. Robustness Analysis

Measured: split geometry (30/10/10 vs 20/5/5), feature count (20 vs 10), augmentation ratio (1× vs 2× with widened mask), stationarity treatment, Y3 response transform (raw vs log1p).

Not measured: random seed sensitivity, recovery-definition sensitivity (exact vs 1, 2% tolerance vs stable-for-k-sessions), removal of the two extreme events (2004-12-26, 2005-11-21).

**Note on split sensitivity,  a result that must be reported.** The dense 20/5/5 configuration produces the best number anywhere in the study (RF Y1 pooled R² **+0.151** against −0.322 primary) *and* the worst (Ridge Y3 pooled R² −229 before prediction clipping). Both directions are the same finding: **at this N, results are highly sensitive to split geometry.** That is itself the most important robustness conclusion.

---

### 26. Error Analysis

**Why pooled R² is negative,  the quantitative explanation.**

R² = 1 − SSE/SST, where SST is computed against the *pooled test* mean. But every model can only ever centre on its *training-fold* mean. The chronological split compounds this: the two largest shocks in the dataset,  **2004-12-26 (Earthquake/tsunami, Y1 = −0.0443)** and **2005-11-21 (Flood, Y1 = −0.0753)**,  fall permanently inside fold 0's **training** window. The held-out folds therefore contain the calmer events and carry only a fraction of full-sample variance:

| Target | Full-sample σ | σ of pooled test folds | Ratio |
| ------ | -------------- | ----------------------- | ----- |
| Y1     | 0.01403        | 0.00880                 | 0.63  |
| Y2     | 0.5799         | 0.4954                  | 0.85  |
| Y3     | 30.64          | 28.54                   | 0.93  |

A negative pooled R² here means "worse than an oracle that already knows the test set's mean",  which is **not** the same as "worse than a usable baseline". The `beats_null_rmse` column answers the second, more meaningful question. This explanation is now in the notebook, along with the `n` and `sigma_y_test` columns that let a reader verify it.

**A caution on the 2005-11-21 event:** Sri Lanka's presidential election was 17 November 2005, four days before the largest ASPI drop in the dataset. There is **no event-window contamination screen anywhere in the pipeline**,  no filter for concurrent elections, policy announcements or macro shocks. This is the core identification threat in any event study and the thesis does not address it.

---

### 27. Explainability / SHAP

Implemented: global summary plot and a local waterfall for the largest real ASPI drop, on real data only.

**Two requirements:**

1. **SHAP under correlated predictors is unstable.** With six ~99%-correlated moving averages and four damage transforms, attribution splits arbitrarily among collinear columns. Any SHAP ranking must be reported as *model attribution under collinearity*, not as a stable importance ordering. Recommend re-running SHAP after the collinearity reduction and reporting whether the ranking changed.
2. **SHAP is not causal.** The thesis must not describe high-SHAP features as drivers of market impact. Replace any causal phrasing with "predictive association".

---

### 28. Ten-Expert Initial Scores

Scoring the framework **as it stood before this audit's remediation**, 0, 10 per criterion, 12 criteria, /120.

| Expert                                | Score /120 | Main strength                                                            | Main weakness                                                                                 | Highest-priority fix               |
| ------------------------------------- | ---------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | ---------------------------------- |
| 1. Financial Econometrician           | 58         | Correct rejection of market-model AR (no valid benchmark for an index)   | Y1 mislabelled "percentage change"; no contamination screen for concurrent events             | Event-window contamination screen  |
| 2. Frontier-Market Specialist         | 64         | Genuine understanding of CSE illiquidity and delayed price discovery     | Y2 mislabelled "volume crash" when max is +1.94 (a spike)                                     | Correct Y2 terminology             |
| 3. Time-Series ML                     | 41         | Walk-forward outer loop, per-fold scaling and selection                  | **Inner k-fold contradicting the stated method**; rolling instead of expanding window   | Replace inner`cv=3`              |
| 4. Rare-Event / Imbalanced Regression | 55         | Real SmoteR interpolation with temporal constraint                       | Fractional one-hots, incoherent derived features, scale-blind noise fallback                  | SMOGN fidelity rewrite             |
| 5. Tree Ensembles                     | 62         | Appropriate family for small tabular data; grids kept small deliberately | `max_depth=None`/`min_samples_leaf=1` on 33 rows; train-fit R² 0.883 vs held-out −0.146 | Bound capacity by fold size        |
| 6. Neural Networks                    | 66         | Correctly constrained shallow MTL; LSTM rightly rejected                 | No early stopping; loss-weight rationale invalidated by y-standardisation                     | Correct the weight rationale       |
| 7. Disaster-Risk / Climate Finance    | 44         | Correct EM-DAT filtering and epidemic exclusion                          | Damage features are largely a reporting-coverage indicator; used as onset predictors          | Information-availability reframing |
| 8. Statistical Validation             | 38         | Pooled R² correctly preferred over per-fold mean                        | **No naive baselines at all**; directional accuracy reported against nothing; no CIs    | Add naive baselines                |
| 9. XAI / Feature Engineering          | 57         | SHAP global + local implemented on real data                             | Severe collinearity unaddressed; SHAP instability unacknowledged                              | Collinearity reduction             |
| 10. Critical Thesis Examiner          | 35         | Unusually honest documentation of found-and-fixed bugs                   | Early-warning claim unsupported; several §-level claims contradicted by the code             | Resolve early-warning framing      |

**Mean: 52/120.** The framework was conceptually well-designed and substantially mis-implemented.

---

### 29. Critical Problems Ranked P0, P3

#### P0,  Could invalidate results

| #    | Problem                                                                           | Status                                                       |
| ---- | --------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| P0-1 | `GridSearchCV(cv=3)` = k-fold inside walk-forward, contradicting §3.7.1        | **Fixed**                                              |
| P0-2 | Ex-post damage features used as onset predictors; early-warning claim unsupported | **Disclosed; requires Model A or permanent reframing** |
| P0-3 | Annual macro dated 1 Jan of its own year  6, 18 month look-ahead on 4 features  | **Fixed**                                              |
| P0-4 | Ridge (the H1 baseline) fit unstandardised  H1 comparison meaningless           | **Fixed**                                              |
| P0-5 | `y.fillna(0.0)` fabricated 3 Y2 observations                                    | **Fixed**                                              |
| P0-6 | Non-stationary price levels in a chronological split                              | **Fixed**                                              |
| P0-7 | Model/config choices made on the reported folds                                   | **Disclosed,  cannot be undone**                      |

#### P1,  Major methodological improvement

| #    | Problem                                                                                     | Status                                             |
| ---- | ------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| P1-1 | Y3 in calendar days while thesis specifies trading days                                     | **Fixed**                                    |
| P1-2 | Y3 right-censoring ignored (cap treated as observed)                                        | **Open,  hurdle model recommended**         |
| P1-3 | `Y3 = 0 ⟺ Y1 >= 0` by construction; targets not independent                              | **Documented**                               |
| P1-4 | Directional accuracy reported without its majority baseline                                 | **Fixed**                                    |
| P1-5 | No naive baselines                                                                          | **Fixed**                                    |
| P1-6 | Rolling rather than expanding walk-forward, contradicting §3.7.1                           | **Option added; both reported**              |
| P1-7 | Central market-vs-disaster ablation never run                                               | **Open,  highest-value missing experiment** |
| P1-8 | `truncate_overlapping_windows()` defined and never called while SWOT implies it is in use | **Quantified and disclosed**                 |
| P1-9 | Predictions violate definitional bounds (Ridge Y3 RMSE 157 on dense config)                 | **Fixed by clipping**                        |

#### P2,  Performance / robustness

P2-1 SVR specified but never implemented · P2-2 RF/XGB capacity unbounded relative to fold size · P2-3 Severe collinearity unaddressed · P2-4 SHAP instability under collinearity unacknowledged · P2-5 Rare-event performance never measured separately · P2-6 SMOGN on/off never ablated · P2-7 `X.fillna(0.0)` conflates "zero damage" with "unknown" · P2-8 Live unpinned API pulls make Table 10 non-reproducible

#### P3,  Optional

P3-1 `> 1000` vs `>= 1000` (fixed) · P3-2 Singleton disaster types as memorisation keys (fixed by pooling) · P3-3 `forward_fill_missing_values` also ffills provenance columns · P3-4 EM-DAT missing month/day filled to 1 January, count unreported

---

### 30. KEEP / OPTIMIZE / MODIFY / REPLACE

| Component                      | Decision                    | Reason                                                                                  |
| ------------------------------ | --------------------------- | --------------------------------------------------------------------------------------- |
| 30/90 event window             | **KEEP**              | Justified a priori by frontier-market price discovery; not tuned on test                |
| ASPI log-return target         | **KEEP** (rename)     | Correct operationalisation; AR not identified for an index. Label is wrong              |
| ATV target                     | **KEEP** (rename)     | `V/V̄ − 1` is the right form; "volume crash" inverts its meaning                    |
| 90-day recovery cap            | **MODIFY**            | Right-censoring must be modelled, not assigned. Hurdle model                            |
| `>= 1000` affected threshold | **KEEP**              | Thesis-mandated; code now matches                                                       |
| Lag structure (t−1,2,3,5)     | **KEEP**              | Stationary, available at prediction time, low-dimensional                               |
| SMA/EMA                        | **REPLACE**           | Non-stationary levels  stationary price-relative ratios                               |
| ADF/KPSS                       | **MODIFY**            | Must cover all features; gate exclusion on ADF unit root, not on both tests             |
| Macroeconomic controls         | **MODIFY**            | Publication lag mandatory; annual frequency against daily events is a stated limitation |
| S&P 500 control                | **KEEP**              | Available same-day, genuinely exogenous, separates global from local shocks             |
| Time-Aware SMOGN               | **KEEP + OPTIMIZE**   | Concept sound and thesis-mandated; implementation required fidelity corrections         |
| OLS/Ridge                      | **KEEP + OPTIMIZE**   | Correct H1 baseline; needed standardisation and in-fold alpha                           |
| SVR                            | **REPLACE (remove)**  | Never implemented; adds tuning burden without spanning new model-family space           |
| Random Forest                  | **KEEP + OPTIMIZE**   | Right family; capacity must be bounded by fold size                                     |
| XGBoost                        | **KEEP + OPTIMIZE**   | Right family; same capacity bound                                                       |
| Shallow MLP                    | **KEEP + OPTIMIZE**   | Appropriately constrained; needs early stopping and corrected weight rationale          |
| MultiOutputRegressor           | **REPLACE (in text)** | Not used by the code; description must be corrected                                     |
| MTL MLP                        | **KEEP** (reframe)    | Only genuine MTL component; no measured positive transfer,  say so                     |
| Weighted MSE                   | **KEEP** (reframe)    | Weights now express task priority, not scale equalisation                               |
| Walk-forward CV                | **KEEP + OPTIMIZE**   | Correct choice; make expanding, fix inner CV                                            |
| Grid/Bayesian tuning           | **KEEP + OPTIMIZE**   | Grid is right at this N; must run on real rows chronologically                          |
| RMSE/MAE/R²                   | **KEEP + OPTIMIZE**   | Correct core; add naive baselines, CIs, and per-target secondary metrics                |
| SHAP                           | **KEEP + OPTIMIZE**   | Correct tool; must acknowledge collinearity instability and drop causal language        |

---

### 31, 33. Recommended Target-Specific Pipelines

#### Pipeline A,  Y1 (ASPI log return)

```
Ex-ante features (Model A) OR ex-post features (Model B),  labelled, never mixed silently
   -> per-fold: SMOGN (train rows only, same-type, derived recomputed)
   -> per-fold: RF-importance top-K feature selection (train rows only)
   -> StandardScaler (Ridge/MLP only; not for trees)
   -> Ridge(alpha via in-fold TimeSeriesSplit RidgeCV) | RF | XGBoost | MLP
   -> expanding walk-forward, outer
   -> RMSE (primary), MAE, pooled R², directional accuracy vs majority baseline, AUC + CI
   -> vs naive_zero, naive_train_mean, Ridge
   -> SHAP global + local
```

#### Pipeline B,  Y2 (abnormal trading volume)

Identical, with two differences: **N = 61** (three events have no volume data,  reported, not imputed), and predictions clipped to `>= −1` (volume cannot be negative).

#### Pipeline C,  Y3 (recovery, trading days)

```
... same preprocessing ...
   -> STAGE 1: classifier, P(recovery within 90 trading days)
        [disclose: near-equivalent to predicting sign(Y1) by construction]
   -> STAGE 2: regressor on log1p(duration | recovered), expm1-inverted, clipped to [0, 90]
   -> combined expectation = P(recover) * E[duration | recover] + (1 - P(recover)) * 90
   -> MAE in trading days (primary), median AE, RMSE, pooled R²
   -> vs naive_zero, naive_train_mean
```

---

### 34. Experiment Matrix

| ID            | Research question                          | Features           | Rare treatment   | Model                         | Validation                                 | Target             |
| ------------- | ------------------------------------------ | ------------------ | ---------------- | ----------------------------- | ------------------------------------------ | ------------------ |
| E01           | Economic null                              |,                  | None             | predict 0                     | Walk-forward                               | All 3              |
| E02           | Statistical null                           |,                  | None             | train mean                    | Walk-forward                               | All 3              |
| E03           | Linear baseline                            | Market             | None             | Ridge (scaled, tuned)         | Walk-forward                               | All 3              |
| **E04** | **Do disaster variables add value?** | Market only        | None             | RF                            | Walk-forward                               | All 3              |
| **E05** | **Do disaster variables add value?** | Disaster only      | None             | RF                            | Walk-forward                               | All 3              |
| **E06** | **Do disaster variables add value?** | Market + disaster  | None             | RF                            | Walk-forward                               | All 3              |
| E07           | Do macro controls add value?               | + macro (lagged)   | None             | RF                            | Walk-forward                               | All 3              |
| E08           | Does SMOGN add value?                      | Full               | **None**   | RF                            | Walk-forward                               | All 3              |
| E09           | Does SMOGN add value?                      | Full               | SMOGN            | RF                            | Walk-forward                               | All 3              |
| E10           | Does sample weighting beat resampling?     | Full               | Sample weights   | RF                            | Walk-forward                               | All 3              |
| E11           | Which model class wins?                    | Full               | Best of E08, E10 | Ridge/RF/XGB/MLP              | Walk-forward                               | All 3              |
| E12           | Does MTL add value?                        | Full               | "                | MLP-MTL vs 3 single-task MLPs | Walk-forward                               | All 3              |
| E13           | Does ensembling add value?                 | Full               | "                | blend + stack                 | Walk-forward                               | All 3              |
| E14           | Ex-ante vs ex-post                         | Model A vs Model B | "                | Best of E11                   | Walk-forward                               | All 3              |
| E15           | Does Y3 need a hurdle model?               | Full               | "                | Regression vs hurdle          | Walk-forward                               | Y3                 |
| E16           | Are severe events predictable?             | Full               | "                | Best of E11                   | Walk-forward, stratified report            | All 3              |
| E17           | Is performance stable?                     | Full               | "                | Best of E11                   | 30/10/10 and 20/5/5, rolling and expanding | All 3              |
| E18           | Recovery-definition robustness             | Full               | "                | Best of E11                   | Walk-forward                               | Y3 (3 definitions) |

E04, E06 are the study's central experiment and have never been run.

---

### 35. Expected Performance Improvements

**No numerical predictions are given**,  the master prompt forbids fabricating them, and this audit has already produced one case where an estimate (the naive-baseline comparison) turned out to be wrong when actually computed.

Qualitative expectations, ordered by confidence:

- **Prediction clipping to definitional bounds:** *provably* non-worsening. Clipping is Euclidean projection onto the target's support, and every true value already lies inside it, so absolute error cannot increase on any point. RMSE and MAE non-increasing, pooled R² non-decreasing, conformal coverage exactly unchanged with non-increasing width. This is the only change in the entire audit with a guarantee attached.
- **Ridge standardisation:** large effect on Ridge, already measured. Direction on other models: none (they are scale-invariant).
- **Stationary feature replacement:** already measured; largest single improvement observed, concentrated in the models that extrapolate (XGBoost, MLP, Ridge).
- **Capacity bounding (RF/XGB):** expected to move R² *toward* zero from below,  "less wrong", not "predictive". Say so.
- **Hurdle model for Y3:** unknown. Better matched to the distribution, but adds a second model to fit on 30 rows.
- **Macro publication-lag fix:** expected to make results *slightly worse*. Do it anyway and report it.
- **Ex-ante feature set (Model A):** expected to be substantially worse than Model B. That gap is itself the finding.

---

### 36. Recommended Final Architecture

```
EM-DAT (86) --> non-biological, affected >= 1000 --> 74 --> within archive coverage --> 64 events
                                                                    |
CSE ASPI + volume --> forward-fill --> stationary feature engineering               |
   (lags, price_to_sma/ema ratios, rolling_std, squared_return)                     |
                                                                    |               |
World Bank macro --> re-dated to Y+1-07-01 (publication lag) -------+               |
S&P 500 (same-day) -------------------------------------------------+               |
                                                                    v               v
                                    merge_asof(backward, asof = event_date - 1 day)
                                                    |
                        +---------------------------+---------------------------+
                        |                                                       |
                 MODEL A (ex-ante)                                      MODEL B (ex-post)
        type, season, trailing counts, market,                  Model A features + realised
        macro, historical type-average severity                 damage / affected / damage_to_gdp
                        |                                                       |
                        +---------------------------+---------------------------+
                                                    |
                              EXPANDING chronological walk-forward (outer)
                                                    |
                        per fold, per target:  SMOGN on TRAIN REAL rows only
                                               (same-type, derived recomputed, <=25% synthetic)
                                                    |
                                       RF-importance top-K on train rows
                                                    |
                              inner TimeSeriesSplit on REAL rows -> hyperparameters
                                          refit winner on real + synthetic
                                                    |
              +------------------+------------------+------------------+
              |                  |                  |                  |
        Ridge (scaled,     Random Forest        XGBoost           MLP (MTL,
        RidgeCV alpha)     (depth<=4,           (depth<=3)        log1p Y3,
                            leaf>=3)                              early stopping)
              |                  |                  |                  |
              +------------------+--------+---------+------------------+
                                          |
                          Y3 only: two-stage hurdle wrapper
                                          |
                        clip predictions to definitional bounds
                                          |
              +---------------------------+---------------------------+
              |                                                       |
   POINT METRICS vs naive_zero / naive_train_mean / Ridge      ROLLING CONFORMAL INTERVALS
   RMSE (primary), MAE, pooled R² (+ n, sigma_y_test)          finite-sample quantile,
   Y1: directional accuracy vs majority baseline, AUC + CI     Wilson CI, vs marginal interval
   stratified: common vs rare vs worst events
              |
              v
        SHAP (post-collinearity-reduction), attribution not causation
```

---

### 37. Methodology Amendments Required in the Thesis

| #  | Current statement                                  | Problem                                                                       | Replacement                                                                    | Section                            |
| -- | -------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------- |
| 1  | "ASPI Percentage Change"                           | A log return is not a percentage change                                       | "ASPI log return, ln(P_t/P_{t−1})"                                            | §3.2.2, Table 4, abstract, RQ1, 4 |
| 2  | "Trading volume crash magnitude"                   | Variable measures deviation in both directions; observed max +1.94 is a spike | "Abnormal trading volume (ATV)"                                                | §3.2.2, Table 4, abstract, RQ1    |
| 3  | "Consecutive trading days" (Y3)                    | Code returned calendar days                                                   | Retain the trading-day definition; state that the implementation was corrected | §3.2.2                            |
| 4  | Recovery capped at 90 with no censoring discussion | `T > 90` is not `T = 90`                                                  | Add right-censoring discussion + hurdle model                                  | §3.2.2                            |
| 5  | "No k-fold anywhere"                               | Inner`cv=3` was a k-fold                                                    | Retain the principle; document the inner`TimeSeriesSplit`                    | §3.7.1                            |
| 6  | "The training set to grow over time"               | Implementation used a rolling window                                          | Report both; state which is primary                                            | §3.7.1                            |
| 7  | `MultiOutputRegressor` wrapper described         | Not used; separate per-target fits                                            | "Independent per-target estimators"                                            | §3.6.4                            |
| 8  | "Multi-task learning" for tree models              | Multi-output ≠ multi-task                                                    | Reserve MTL for the MLP                                                        | §3.6.4, abstract                  |
| 9  | "RMSE measures the total variation explained"      | Wrong,  that is R²                                                          | "RMSE is the root of the mean squared error, in the target's own units"        | §3.8.1                            |
| 10 | "Explainable early-warning system"                 | Six features are ex-post                                                      | "Ex-post impact-attribution framework"; add Model A as future work             | Abstract, §1.2.3, §4.4, README   |
| 11 | "Aggregated sentiment scores... were used"         | Contradicts Table 4 and the code                                              | Delete; state sentiment was excluded and why                                   | §3.3.1                            |
| 12 | SVR listed as a model                              | Never implemented                                                             | Remove, with a stated reason                                                   | §3.6.1                            |
| 13 | Loss weights as eq.(4) scale mechanism             | y is standardised first, so scale gap is already gone                         | Weights express task priority                                                  | §3.6.4                            |
| 14 | Macro variables with no publication-lag treatment  | Look-ahead                                                                    | Add as-of-date alignment                                                       | §3.3.1                            |
| 15 | ADF/KPSS "confirm log_return is stationary"        | Only one feature tested                                                       | All features tested; unit-root columns excluded                                | §3.5.1                            |
| 16 | Overlapping-window truncation described            | Function never called                                                         | Either wire it in or state it was not applied and quantify overlap             | §3.3.2                            |
| 17 | No statement of Y1Y3 dependence                  | `Y3 = 0 ⟺ Y1 >= 0` by construction                                         | Add explicitly before any Y3 result                                            | §3.2.2                            |
| 18 | 100% real data                                     | 3 Y2 values were zero-filled                                                  | Now true; report Y2 N = 61                                                     | §3.4.1                            |
| 19 | "First-ever" framework                             | Unverifiable superlative                                                      | "To our knowledge, the first..."                                               | §1.2.3                            |
| 20 | Feature-importance language implying causation     | SHAP is attribution                                                           | "Predictive association"                                                       | §3.8.2                            |

---

### 38. Research Contribution After Improvement

Stated honestly, the contribution is **not** a working predictor. It is four things, and they are real:

1. **A well-evidenced null result.** Across five model families, two validation geometries, two feature counts and with/without augmentation, index-level CSE response to qualifying natural disasters is not predictable at this sample size. Median Y1 is +0.000003 and median Y3 is 0,  more than half of qualifying disasters produce no measurable day-0 index reaction. For a market the thesis itself characterises as thin and semi-strong-inefficient, that is a genuine, interpretable finding: index-level aggregation and low liquidity absorb localised physical shocks. Consistency across specifications is what makes a null credible, and most undergraduate theses cannot offer a robustness surface at all.
2. **A methodological contribution on multi-task regression with heterogeneously-scaled targets.** The finding that loss weights of (1.0, 0.1, 0.5) cannot compensate a ~1000× raw-scale gap, and that target standardisation is a necessary complement, is transferable and was not documented anywhere the student found it. (Before the fix, MLP Y1 R² was ≈ −390.)
3. **A negative-results register that saves the next researcher real time.** Yahoo's `^CSE` feed verified dead three ways; the market-cap CSV rejected for having no historical join key; the widened-SMOGN variant tried, measured and reverted; SMOGN's own defects found and corrected. In frontier-market data work this is the scarcest material in the literature.
4. **Reusable data infrastructure.** A parser handling three distinct CSE report layouts across 23 years via header-text detection, and a validated 64-event CSE-disaster panel that does not exist elsewhere.

**One-sentence statement of contribution:** *the first end-to-end, leakage-audited multi-target ML framework for disaster impact on a frontier exchange, applied to a complete real 64-event panel, which establishes,  robustly across five model families and two validation geometries,  that index-level CSE response to qualifying natural disasters is not predictable at this sample size, and which documents the specific data, leakage and target-construction obstacles any future attempt must overcome.*

---

### 39. Limitations Remaining

1. **N = 64.** The binding constraint. No technique overcomes it.
2. **Ex-post features.** Until Model A is built, the framework cannot support prospective claims.
3. **Annual macro against event-level timing.** CBSL monthly CCPI, daily LKR/USD and policy rates were not accessible; World Bank annual series are a coarse proxy even after the lag correction.
4. **Archive ends Jun-2023 (price) / Mar-2023 (volume).** Storm Ditwah (Nov 2025), the thesis's own flagship example, is outside the modelled set.
5. **EM-DAT reporting bias.** Most Sri Lankan records carry no damage estimate; damage features partly measure reporting coverage.
6. **Y3 censoring and its dependence on Y1.**
7. **No event-window contamination screen.** Concurrent elections and policy shocks are not controlled,  note the 2005 presidential election four days before the largest observed drop.
8. **Optimistic bias from configuration choices made on the reported folds.** Disclosed, unquantifiable.
9. **Interpolated volatility is convexity-biased high** in synthetic rows and cannot be corrected without the price path.
10. **Live unpinned API pulls** make the exact table non-reproducible until cached.
11. **Inner-CV noise.** Chronological inner folds of 9/16/23 rows are close to uninformative; the response is a deliberately small search space, not a better search.

---

### 40. Viva Defence

**"Nine of your twelve R² values are negative. Why should we believe anything here?"**
Because a negative pooled R² is a statement about the pooled test mean, not about usefulness. R² penalises against an oracle that already knows the test set's mean; our chronological split places the two largest shocks permanently in fold 0's training window, leaving the held-out folds with 63% of full-sample variance on Y1. The question that matters is whether the model beats a *deployable* baseline, so we computed two,  a constant-zero economic null and a training-mean statistical null,  on identical folds. The honest answer is that on Y1 no model beats them, on Y2 only the corrected Ridge does, and on Y3 the training-mean predictor wins. We report that rather than the R² framing because it is the more meaningful comparison, and because it is less flattering.

**"On the day before the flood, how do you know its total damage?"**
We do not. Six of our features are EM-DAT post-hoc assessments, so this framework answers the attribution question,  given a disaster of measured severity, what was the market response,  and not the forecasting question. We corrected the early-warning language in the README and thesis when we found this, and specified an ex-ante feature set as the separate experiment it actually requires.

**"Section 3.7.1 bans k-fold. What was `cv=3`?"**
A k-fold, inside every walk-forward fold, selecting hyperparameters by training on later events to validate earlier ones. We found it during audit and replaced it with a chronological `TimeSeriesSplit` restricted to each fold's real rows, with the winner refit on real plus synthetic. We report it because the contradiction between our stated method and our code is exactly the kind of thing an audit exists to catch.

**"Your best directional accuracy is 65%. What does 'always predict no crash' score?"**
Also 65%. No model beats the trivial rule, and several score below it. That is why the notebook now prints the majority baseline beside every accuracy figure and reports AUC with a Hanley, McNeil interval that includes 0.5 in every case.

**"Why should a Random Forest work on 64 events?"**
It largely does not, and we quantify it: train-fit R² of 0.883 on Y3 against a held-out −0.146. We report that 1.03-point gap as an overfitting diagnostic rather than as a performance figure.

**"Is Y3 = 0 the same as Y1 ≥ 0?"**
Yes, by construction,  the recovery window includes the event day and the baseline is the prior close. Over half our Y3 values are therefore mechanically determined by the sign of Y1. We state this before reporting any Y3 result, and it is why we recommend a hurdle model whose first stage is acknowledged to be close to a sign classifier.

**"You chose 30/10/10. What happens at 20/5/5?"**
Random Forest Y1 pooled R² goes from −0.322 to +0.151,  the best number anywhere in our study. We do not promote it, because it was evaluated on the same folds we report and adopting it would be selection on the test set. We report it at equal prominence as an upper bound on what this pipeline can be made to appear to achieve.

---

### 41. Final Ten-Expert Verdict

Assessed against the **remediated** framework, assuming the open items in §29 are completed.

| Expert                        | Verdict                         | Condition                                                                     |
| ----------------------------- | ------------------------------- | ----------------------------------------------------------------------------- |
| 1. Financial Econometrician   | PASS WITH CORRECTIONS           | Add event-window contamination screen; rename Y1                              |
| 2. Frontier-Market Specialist | PASS WITH CORRECTIONS           | Rename Y2                                                                     |
| 3. Time-Series ML             | PASS                            | Inner k-fold removed; expanding window reported                               |
| 4. Rare-Event Specialist      | PASS WITH CORRECTIONS           | Run the SMOGN on/off ablation (E08/E09)                                       |
| 5. Tree Ensembles             | PASS WITH CORRECTIONS           | Bound capacity by fold size                                                   |
| 6. Neural Networks            | PASS WITH CORRECTIONS           | Add early stopping; correct the loss-weight rationale                         |
| 7. Disaster-Risk Specialist   | PASS WITH CORRECTIONS           | Report in-scope damage-provenance breakdown                                   |
| 8. Statistical Validation     | PASS WITH CORRECTIONS           | Add bootstrap CIs and the rare-event stratified table                         |
| 9. XAI / Feature Engineering  | PASS WITH CORRECTIONS           | Re-run SHAP post-collinearity-reduction; remove causal language               |
| 10. Critical Thesis Examiner  | **PASS WITH CORRECTIONS** | **Blocking: resolve the early-warning framing (P0-2) and run E04, E06** |

**No unresolved P0 remains in the code.** Two P0-class items are documentation/experiment obligations rather than defects: the early-warning reframing (P0-2) and the disclosure of configuration selection on reported folds (P0-7, disclosed and unquantifiable).

**Overall: PASS WITH CORRECTIONS.** The architecture may be labelled recommended once P0-2 is resolved in the text and E04, E06 have been run.

---

## TOP 10 CHANGES TO IMPLEMENT FIRST

Ranked by (research value × defensibility) / (complexity × overfitting risk). Status reflects work already completed during this audit.

#### 1. Run the market-vs-disaster ablation (E04, E06),  **NOT YET DONE**

- **Current problem:** the thesis exists to answer "do disaster characteristics add predictive information beyond ordinary market history?" and that experiment has never been run.
- **Exact change:** three additional `run_walk_forward` calls with `FEATURE_COLS` restricted to (a) market-only, (b) disaster-only, (c) market+disaster; report `ΔRMSE = RMSE_market_only − RMSE_market+disaster` per target.
- **Why:** without it the study cannot claim disaster data contributes anything. It is the central research question.
- **Affected target:** all three.
- **Files/cells:** notebook cell `15d59212` (feature subsets), new cell after `c8445d58`.
- **Experiment:** E04, E05, E06.
- **Success metric:** ΔRMSE with a paired bootstrap CI over pooled out-of-fold squared errors.
- **Reject if:** ΔRMSE CI straddles zero,  in which case report honestly that disaster features add nothing measurable, which is itself a publishable finding.

#### 2. Resolve the early-warning framing (P0-2),  **PARTLY DONE**

- **Current problem:** six features are EM-DAT post-hoc assessments; the README and thesis claim an early-warning system.
- **Exact change:** README and notebook §15 corrected to "ex-post impact attribution" (done). Remaining: build Model A (ex-ante feature set) and report both.
- **Why:** the claim as written is unsupportable and is the first thing an examiner will attack.
- **Affected target:** all three.
- **Files/cells:** `README.md` (done), notebook `e62f629a` (done), `c868f959` (done); Model A requires a new feature-subset cell.
- **Experiment:** E14.
- **Success metric:** two labelled models reported; no prospective claim attached to Model B.
- **Reject if:** never,  the reframing is mandatory regardless of results.

#### 3. Fix the inner k-fold,  **DONE**

- **Problem:** `GridSearchCV(cv=3)` = `KFold(shuffle=False)` inside every walk-forward fold, contradicting thesis §3.7.1; SMOGN's tail-appended synthetic rows made the last inner block up to ~45% synthetic.
- **Change:** `TimeSeriesSplit(n_splits=3)` on each fold's **real** rows, `refit=False`, winner refit on real+synthetic.
- **Files:** notebook `c5f83fcb`.
- **Metric:** no k-fold present anywhere; the results change is incidental, correctness is the point.
- **Reject if:** never.

#### 4. Standardise Ridge and tune alpha in-fold,  **DONE**

- **Problem:** the H1 baseline was fit on unstandardised features mixing 1e4 price levels, 1e9 USD damage and 0/1 flags, with `alpha` left at its library default.
- **Change:** `make_pipeline(StandardScaler(), Ridge(alpha=RidgeCV-selected on real rows))`.
- **Measured effect:** Ridge Y1 −1.634  −0.407  −0.278 (after stationarity fix); Y2 −3.539  −0.015  **+0.053**.
- **Reject if:** never,  a mis-specified baseline invalidates the H1 comparison in either direction.

#### 5. Replace non-stationary price levels,  **DONE**

- **Problem:** `sma_*`/`ema_*` are raw ASPI levels (574  10,000+); under a chronological split the test fold sits outside training support.
- **Change:** stationary `price_to_sma_w = P_{t−1}/sma_w − 1` ratios; ADF/KPSS on all features; exclude on ADF unit root only.
- **Measured effect:** largest single improvement in the study,  XGBoost Y1 −1.617  −0.236, MLP Y1 −1.547  −0.299, stacked Y1 −0.775  −0.134.
- **Reject if:** never,  thesis §3.5.1 mandates stationarity.

#### 6. Add naive baselines and the majority baseline,  **DONE**

- **Problem:** no null to compare against; directional accuracy reported against nothing (65% against a 65% majority baseline).
- **Change:** `naive_zero` and `naive_train_mean` rows in the summary table with a `beats_null_rmse` column; majority baseline printed beside every accuracy; Hanley, McNeil CI on AUC.
- **Measured effect:** revealed that **0 of 6 models beat the no-effect null on Y1**.
- **Reject if:** never,  this is a reporting-integrity fix, not a performance change.

#### 7. Clip predictions to definitional bounds,  **DONE**

- **Problem:** Ridge produced Y3 RMSE 157 on the dense config (target capped at 90) and a 182-day conformal interval on a 90-day-capped target.
- **Change:** `clip_to_bounds`,  Y3 to [0, 90], Y2 to ≥ −1,  applied at every prediction site including ensemble, stack and interval endpoints.
- **Why:** uses only target definitions from §3.2.2; provably cannot increase absolute error on any point.
- **Reject if:** never,  it carries a mathematical guarantee.

#### 8. Stop imputing missing targets,  **DONE**

- **Problem:** `y.fillna(0.0)` asserted "volume exactly at baseline" for 3 events, in a study claiming 100% real data, using the same zero-fill §3 explicitly rejects for volume.
- **Change:** NaNs preserved, masked per target; Y2 effective N reported as 61.
- **Reject if:** never.

#### 9. Bound RF/XGBoost capacity by fold size,  **NOT YET DONE**

- **Current problem:** grids allow `max_depth=None`, `min_samples_leaf=1` on 33-row folds; RF train-fit R² 0.883 vs held-out −0.146 on Y3.
- **Exact change:** `RF_PARAM_GRID = {"n_estimators":[300], "max_depth":[2,3,4], "min_samples_leaf":[3,5]}`; `XGB_PARAM_GRID = {"n_estimators":[100,300], "max_depth":[2,3], "learning_rate":[0.03,0.1], "subsample":[0.8], "colsample_bytree":[0.8]}`.
- **Why:** bounded a priori by fold size (leaf ≥ 3 means every leaf rests on ~10% of the fold), not by test score.
- **Affected target:** all three.
- **Files/cells:** notebook `c5f83fcb` grid definitions.
- **Metric:** the train-fit-to-held-out R² gap should narrow; held-out R² expected to move toward 0 from below.
- **Reject if:** held-out RMSE worsens materially on two or more targets.

#### 10. Implement the Y3 hurdle model,  **NOT YET DONE**

- **Current problem:** the 90-day cap is treated as an observed value; `T > 90` is not `T = 90`.
- **Exact change:** stage 1 classifier for P(recovery ≤ 90 trading days); stage 2 regressor on `log1p(duration | recovered)`; combine as `P·E[d|recover] + (1−P)·90`.
- **Why:** matches the zero-inflated, right-censored structure; avoids a survival model the sample cannot support.
- **Affected target:** Y3 only.
- **Files/cells:** notebook `c5f83fcb`, Y3 branch.
- **Experiment:** E15.
- **Metric:** MAE in trading days vs the current single-stage regressor and vs `naive_train_mean`.
- **Reject if:** MAE does not improve over single-stage,  report the negative result and keep the simpler model (parsimony).

---

## NEXT 5 EXPERIMENTS, IN ORDER

#### Experiment 1,  Do disaster variables add predictive value? (E04, E06)

- **Hypothesis:** adding EM-DAT disaster severity features to a market-history baseline reduces out-of-sample RMSE on at least one target.
- **Current configuration:** all 27 features together; no decomposition.
- **Modified configuration:** three runs,  market-only, disaster-only, market+disaster,  identical folds, identical model (RF), identical SMOGN.
- **Validation:** expanding chronological walk-forward, 30/10/10.
- **Metrics:** ΔRMSE, ΔMAE, Δpooled R² per target, with paired bootstrap CI on pooled squared errors.
- **Expected interpretation:** if the ΔRMSE CI straddles zero on all three targets, the study's central premise is not supported at this N,  report that as the primary finding. If disaster-only beats market-only on Y3, that is the strongest positive result available.

#### Experiment 2,  Does SMOGN add value? (E08, E10)

- **Hypothesis:** time-aware SMOGN improves prediction of rare high-impact events without degrading overall performance.
- **Current configuration:** SMOGN always on; never ablated.
- **Modified configuration:** (a) no oversampling, (b) current SMOGN, (c) sample weighting by target rarity instead of resampling.
- **Validation:** as above.
- **Metrics:** overall RMSE **and** RMSE restricted to the rare subset (`Y3 > 30`), reported separately.
- **Expected interpretation:** with 3 synthetic rows against 30 real, the honest expectation is no measurable difference. Publishing that is a legitimate contribution,  the thesis proposes SMOGN, so it owes the reader evidence either way.

#### Experiment 3,  Rare-event stratified performance (E16)

- **Hypothesis:** models that perform acceptably overall fail on the catastrophic events the study exists to predict.
- **Current configuration:** aggregate metrics only.
- **Modified configuration:** partition pooled out-of-fold predictions into common / rare (top-decile severity) / worst single event; report per stratum.
- **Validation:** no re-run required,  computed from the stored `results` arrays.
- **Metrics:** RMSE and MAE per stratum; the 2004-12-26 and 2005-11-21 events named individually.
- **Expected interpretation:** if rare-event error is much worse than overall error, the research objective is not met regardless of aggregate numbers. This must be known before any performance claim is made.

#### Experiment 4,  Ex-ante vs ex-post (E14)

- **Hypothesis:** a feature set restricted to information available at disaster onset performs materially worse than one including realised damage.
- **Current configuration:** ex-post only (Model B).
- **Modified configuration:** Model A,  disaster type, month/season, `days_since_last_disaster`, `disasters_trailing_365d`, historical type-average severity computed from prior events only, all market features, lag-corrected macro.
- **Validation:** as above, identical folds.
- **Metrics:** RMSE per target for A vs B.
- **Expected interpretation:** B is expected to beat A. **The size of that gap is the quantified value of post-event information**,  the number that tells a policymaker how much better an assessment-based tool is than a forecast. This converts the framing problem into a finding.

#### Experiment 5,  Recovery-definition robustness (E18)

- **Hypothesis:** conclusions about Y3 do not depend on the specific recovery definition.
- **Current configuration:** exact recovery (ASPI ≥ pre-event baseline), single definition.
- **Modified configuration:** (a) exact, (b) tolerance,  within 1% of baseline, (c) stable,  at or above baseline for 3 consecutive sessions.
- **Validation:** as above.
- **Metrics:** MAE in trading days per definition; correlation between the three Y3 vectors; whether model ranking changes.
- **Expected interpretation:** if conclusions flip across definitions, Y3 results are definition-artefacts and must be heavily qualified. Do **not** adopt whichever definition scores best,  report all three and keep the pre-registered exact definition as primary.

---

### Rules Observed in This Audit

No performance value in this document was fabricated, estimated or extrapolated. Every figure was copied from an executed run; unmeasured quantities are marked "not measured". No modification is described as working before experimental evidence demonstrated it,  where a change was measured, the measurement is shown, including the cases where it made results worse (the widened SMOGN variant) or destroyed a favourable number (K=10 removing the only positive R²). Where an earlier estimate was contradicted by later measurement, the correction is stated explicitly (the naive-baseline comparison).

---

### 42. Visualization and Diagnostics

This section did not exist in the original audit, and its absence was itself a defect: the
pipeline produced **two images across 59 notebook cells**, both SHAP, neither exported to
disk, so nothing in the study could be cited as a figure. More importantly, three findings
recorded here in prose had no visual evidence at all, and two of them are exactly the kind
of claim a reader will not accept on assertion.

Figures now live in `src/visualization/result_figures.py` and export to `docs/figures/` at 300 dpi
under stable filenames. Two conventions are enforced in code rather than left to the
caller, and both exist because the underlying results are weak:

- **Every evaluation figure carries its sample size on its face** (`stamp()`). A ROC curve
  drawn from 30 points looks identical to one drawn from 30,000. Printing "n = 30 pooled
  out-of-fold points" in the corner is what stops the figure being over-read.
- **Nothing is encoded by hue alone.** Every categorical series carries colour *and*
  marker *and* linestyle; magnitude heatmaps use a monotonic-lightness colormap. The
  figures survive greyscale printing.

Where a figure could imply performance the numbers do not support, the honest reference is
drawn in: chance diagonals on ROC, prevalence lines on precision-recall, naive-baseline
rules across every metric bar, and the marginal-interval comparison on every conformal
plot.

#### Which audit finding each figure evidences

| Figure | Evidences |
|---|---|
| `fig_02_target_distributions` | Y3's zero-inflation and censoring (P1-2),  the ECDF panel shows the point masses at 0 and 90 that a histogram hides |
| `fig_03_target_dependence_y1_y3` | **P1-3**, `Y3 = 0 ⟺ Y1 ≥ 0`. The contingency inset shows the *Y1 ≥ 0 and Y3 > 0* cell is empty |
| `fig_04_feature_correlation_heatmap` | **P2-3**, severe collinearity. Spearman, clustered, with cells below the n=64 significance threshold greyed out |
| `fig_06_collinearity_vif` | **P2-3** quantified,  VIF on a log axis plus the condition-index panel |
| `fig_09_walk_forward_folds` | §26, the variance-compression artefact: both extreme events sit permanently in fold 0's *training* window |
| `fig_11_conformal_coverage` | The model-vs-marginal interval comparison,  coverage is meaningless without width |
| `fig_14_roc_with_ci` | Directional performance, with the chance diagonal and an automatic "includes 0.5" flag on any CI covering chance |
| `fig_17` / `fig_18` | §13, the baseline comparison. The skill forest is the decisive one: a whisker crossing zero is not a win |
| `fig_19_pred_vs_actual` | What a negative R² actually looks like,  predictions collapsing into a narrow band off the identity line |
| `fig_24_overfitting_gap` | **P2-2**, the memorisation gap between in-sample refit and held-out R² |
| `fig_31_sector_response_and_skill` | The sector extension,  response and predictability per sector |

#### Measured by the new diagnostics

The collinearity figures immediately produced numbers the audit had only characterised
qualitatively:

- **Condition number ≈ 2.3 × 10¹⁷.** The usual threshold for concern is 30.
- **Three feature pairs at Spearman exactly 1.000**: `financial_damage` /
  `log_financial_damage`, `population_affected` / `log_population_affected`, and
  `vol_ratio_10_30` / `vol_trend_10_30`.
- The pre-declared redundancy rule (within any group correlated above |ρ| = 0.95, keep the
  least-derived member) removes **8 of 37** features.

The third pair was a defect in the new volume block introduced during this remediation, 
a 10-day mean over a 30-day mean is by definition the 10/30 ratio, and the first version
shipped both. The diagnostic caught it before it reached a model, which is the argument
for building the diagnostic.

#### Deliberately not built

- **A classification reliability diagram over the regression outputs.** The regression
  reframing scores with `−ŷ`, which is not a probability, and fitting a calibration map on
  the same 30 out-of-fold points used to evaluate it would be test-set fitting. A
  regression-calibration slope is the defensible substitute.
- **Boxplots of per-fold metrics.** Quartiles from three observations. A slope plot across
  folds shows the same dispersion without implying quantiles the sample cannot support.
- **Impurity-based `feature_importances_` charts.** Maximally unstable under exactly the
  ρ ≈ 1.00 collinearity documented above, and they would visually contradict the SHAP
  figures. Grouped SHAP supersedes them.
- **Per-fold ROC curves.** Ten test points per fold is a ten-step staircase; overplotting
  three of them is noise.

---

### Improvement attempts (Y1 / Y3, targets held fixed)

Following the results in `docs/RESULTS_AUDIT.txt` (Y1_aspi_log_return R2=-0.15,
Y3_recovery_days R2=-0.02, only Y2 beats baseline), a literature/repo search was run
before writing any code, restricted to techniques that improve predictability of the
SAME target definitions (no target reframing). Findings actually implemented below;
everything traces to a cited source.

#### Sources consulted

- Stock price recovery after a market shock: a survival analysis approach (ResearchGate,
  2026) and "A survival analysis method for stock market prediction" (ResearchGate), 
  both model firm-level post-shock recovery time as a right-censored duration (firms not
  recovered by the observation cutoff are censored, not coded as "recovered at the cutoff"),
  fit with a Cox / AFT model rather than point regression on the censored value.
- "Measuring and forecasting financial system resilience under multiple shocks: a survival
  analysis approach" (ScienceDirect),  Cox proportional-hazards treatment of financial
  recovery under repeated shocks, same censoring logic.
- Davidescu et al. (2025), "Evaluating Sectoral Vulnerability to Natural Disasters in the US
  Stock Market... DCC-GARCH Models" (already in the thesis reference list) and
  "GARCH-Informed Neural Networks for Volatility Prediction in Financial Markets" (ACM,
  2024),  conditional (GARCH) volatility as a feature carries information a realized
  rolling standard deviation does not (it is a forecast, not a backward-looking average),
  and is the standard volatility-modeling companion to ML regressors in this literature.
- General small-N financial ML literature (permutation importance / RFE feature pruning to
  fight collinearity),  already implemented in this repo's `select_top_features` /
  `_select_top_features` (per-fold RF-importance top-K), confirmed present in
  `src/training/walk_forward.py` and `scripts/train_final_models.py`; no further action
  needed there, it was already the state of the art for this sample size.

#### What was implemented

1. **`src/models/survival_recovery.py`**,  a right-censored AFT model (via `lifelines`,
   `WeibullAFTFitter`/`LogNormalAFTFitter`, selected in-fold by AIC) for Y3. The existing
   `Y3_recovery_days` target is used unchanged; the only change is that observations at the
   90-day cap are marked `event_observed=False` (censored) instead of being treated as a
   literal observed value of 90, which is the textbook-correct likelihood for exactly this
   data shape (right-censored at a fixed follow-up window). Reported alongside the existing
   RMSE/MAE table plus a concordance index (the standard metric for censored time-to-event
   predictions, not meaningful for the other point-regression models).
2. **`garch_cond_vol` feature** in `src/features/feature_engineering.py`,  GARCH(1,1)
   conditional-volatility forecast on the ASPI log-return series (via the `arch` package),
   refit on an expanding window with an annual refresh cadence so no fold ever sees a
   volatility estimate whose GARCH parameters were fit on data beyond that fold's own
   history point,  same causal discipline as the existing `shift(1)`-guarded rolling
   features, documented inline. Added as an extra Y1 feature candidate; it flows through
   the existing per-fold `select_top_features` step like any other column, so it is kept
   only if it earns its place, not force-included.

#### What was NOT implemented, and why

- **Bayesian hyperparameter optimization**,  considered and rejected per the existing
  audit note (§21): would search a larger space on ~30-40 training rows, increasing
  selection overfitting risk for no evidenced benefit over grid search at this N.
- **Sector-level reframing of Y1**,  explicitly out of scope per author instruction: Y1
  must remain the ASPI-aggregate log return, not a sector sub-index.
- **Deep sequence models (LSTM/Transformer)**,  ruled out per existing audit reasoning;
  N=40-76 is far below what these architectures need to generalize rather than memorize.

#### Measured results (targeted ablations, real data, same folds as the main table)

**Y3, AFT survival model** (`scripts/run_survival_model.py`, n=40 pooled test points,
identical walk-forward folds/SMOGN/feature-selection as the main table):

| model | RMSE | MAE | pooled R2 |
|---|---|---|---|
| naive_zero | 32.254 | 13.725 | -0.221 |
| naive_train_mean | 29.469 | 19.922 | -0.019 |
| **aft_survival** | 34.971 | 17.281 | -0.435 |

Point-prediction RMSE is worse than both naive baselines, the AFT median does not, on
this sample, out-predict a flat training mean any more than the hurdle model did. BUT
the model was fit to optimise a survival likelihood, not RMSE, and on the metric that
likelihood actually targets, Harrell's concordance index, i.e. "does it rank which
events recover faster than which others correctly more often than chance", it scores
0.696 / 0.448 / 0.529 / 0.600 across the four folds, mean **0.568** (0.5 = chance). That
is a real, if modest, ranking signal invisible to a plain RMSE comparison: correctly
treating the 90-day cap as censoring, rather than as an observed value of 90, recovers
some genuine ordinal information about recovery speed that the point-regression models
(including the hurdle model) do not surface. This is the honest result to report: a
positive methodological finding (censoring matters, ranking signal exists) alongside a
still-negative one (point RMSE does not beat naive at N=40).

**Y1, GARCH(1,1) conditional volatility feature** (`scripts/run_garch_ablation.py`,
Ridge, n=40, with vs without `garch_cond_vol` in the candidate pool, everything else
identical):

| variant | RMSE | MAE | pooled R2 |
|---|---|---|---|
| without_garch | 0.01555 | 0.00981 | -0.301 |
| with_garch | 0.01609 | 0.01017 | -0.393 |

delta_rmse = -0.00054, 95% CI [-0.00434, +0.00163] -- **does not exclude zero**: adding
a GARCH conditional-volatility forecast to the feature pool makes no statistically
distinguishable difference to Y1 prediction, and the point estimate is if anything
slightly worse (consistent with the audit's own collinearity finding, one more
correlated volatility-family column adds selection noise more readily than it adds
signal at N=40). This null result is itself consistent with the lit review already in
the thesis (Kengatharan & Jeyan Suganya, 2019; Priyadarshani & Perera, 2023): the
aggregate ASPI index appears to be genuinely difficult to predict at the daily-return
level regardless of which volatility feature feeds it, which is the diversification
story those papers already tell, feature engineering does not manufacture a signal
the aggregate index may not carry.

#### Bottom line for the thesis

Two real techniques were implemented, both correctly, both tested on real data through
the actual pipeline (not simulated): a right-censored AFT survival model for Y3, and a
GARCH conditional-volatility feature for Y1. Report both outcomes as findings, not as
failures to hide: Y3 point-RMSE is still not beaten, but a genuine, citable, positive
result exists in the concordance index; Y1 remains a defensible null result strengthened,
not weakened, by having tried a targeted, literature-backed feature and shown it does
not move the needle. This is a stronger, more honest thesis than either silently omitting
the attempt or overstating what it achieved.

#### Full official re-run confirms it (not just the targeted ablation script)

`notebooks/02_features_targets.ipynb` -> `04_modeling_regression.ipynb` ->
`06_evaluation.ipynb` -> `scripts/audit_results.py` were re-executed end to end with
`garch_cond_vol` live in the candidate feature pool (92 features instead of 91). The
resulting `docs/RESULTS_AUDIT.txt` is **byte-identical to the pre-change version in
every single model/target row** except the feature count in the data-coverage section
(91 -> 92). That means the per-fold RF-importance top-20 selector never once picked
`garch_cond_vol` over the other 91 candidates, for any target, in any fold, the
strongest possible version of the null result: it is not just statistically
indistinguishable when forced in, the pipeline's own feature-selection step
independently agreed it does not carry enough signal to make the cut. The AFT survival
model's c-index result (0.568) stands as reported above, unaffected by this (Y3's
feature pool is separately selected and the AFT script above was run against the exact
same real dataset).

### Y1 redefinition to 5-trading-day cumulative return, and a feature/model pass

Following the improvement work above, the author redefined Y1 (target definition, not a
technique) from a single-day log return to `ASPI_5D_Log_Return_Pct = 100 * ln(ASPI_(t+5)
/ ASPI_t)`, `t+5` counted in actual CSE trading sessions (verified by hand against 5 real
events; see the manual-verification note kept alongside `feature_eng.py`). This produced
a positive pooled R2 for the first time on any Y1 model (ensemble R2=+0.085 at the point
this redefinition was scored), though not statistically distinguishable from naive_zero
(bootstrap CI on delta_rmse included zero). A fold-boundary purge
(`src/training/walk_forward.py::purge_horizon_overlap`) was added at the same time,
because a 5-day-forward label can otherwise leak across a walk-forward fold boundary, 3 real event pairs in this dataset are closer together than the 5-trading-day horizon.

A further feature/model pass was then run, cited and reasoned before implementation:

- **Collinearity pruning composed into feature selection.** `src/evaluation/collinearity.py`
  already implemented the pre-declared |rho|>=0.95 redundancy rule (see the collinearity
  section earlier in this document) but it had never actually been wired into
  `notebooks/04_modeling_regression.ipynb`'s `select_top_features` -- only RF-importance
  ranking ran there. Composed as: redundancy-drop (reads no target, so not selection on
  the test set) -> RF-importance top-k, on each fold's real training rows only. Standard
  filter-method combination for small-N tabular data (VIF/correlation pruning + importance
  ranking; a >0.75 correlation-pair diagnostic table is also generated and saved to
  `artifacts/tables/y1_feature_stability.csv`, but the ACTUAL drop threshold stays at the
  pre-registered 0.95, per author decision, not the harder 0.75 that would count as
  revising a pre-declared rule after seeing results).
- **PCA**,  added as a reported ablation only (`notebooks/04_modeling_regression.ipynb`,
  section 4.8b): PCA(10 components)+Ridge vs RF-selected-features+Ridge, same folds. Not
  adopted as a default (PCA components are uninterpretable, which cuts against the
  explainability/SHAP chapter) unless it clearly wins.
- **Gaussian Process regression, SVR, median quantile regression** added to the model
  lineup (literature-standard choices for small-N nonlinear regression with calibrated
  uncertainty; SVR specifically closes this document's own earlier-flagged open item,
  "SVR specified but never implemented").

#### Measured effect (full official pipeline re-run, real data)

Composing collinearity-drop into `select_top_features` changed EVERY model's Y1 result,
not just the ones that use it directly (MLP inherits its feature set from the same
per-fold selection). Before vs after, same folds, same pipeline:

| model | RMSE before | RMSE after | R2 before | R2 after |
|---|---|---|---|---|
| ridge | 2.608 | 2.667 | -0.004 | -0.050 |
| random_forest | 2.594 | 2.598 | +0.007 | +0.004 |
| xgboost | 2.564 | 2.851 | +0.030 | -0.200 |
| mlp | 2.766 | 2.926 | -0.130 | -0.264 |
| **ensemble** | **2.489** | **2.652** | **+0.085** | **-0.039** |

This is an honest negative result for Y1: composing the pre-declared collinearity rule
into the actual scored pipeline made the ensemble (and every individual model except RF,
roughly flat) measurably WORSE on point-prediction metrics, flipping the one positive R2
result of the day back negative. The likely mechanism: RF-importance ranking already
implicitly down-weights redundant correlated features on its own, so the extra hard
pre-filter mostly removes features that, despite correlation, still carried fold-specific
marginal signal at this N, collinearity pruning is a correctness/interpretability
argument (it directly fixes a documented problem: `sma_5/10/20` etc. at ~99% mutual
correlation), not a guaranteed RMSE improvement, especially with only ~30 training rows
per fold where any feature-set change is high-variance.

Y2 moved the other way (RF R2 0.247->0.268, and SVR/GP/XGBoost/ensemble all newly beat
naive_zero where previously only RF/ensemble/XGBoost did), the same change helped one
target and hurt another, which is itself informative: it is evidence AGAINST a single
fixed feature-selection recipe being optimal for all three targets simultaneously, not
evidence that collinearity-pruning is simply "good" or "bad" in general.

**SVR and Gaussian Process regression**: SVR is now the only Y1 model with both a
positive pooled R2 (+0.018) and a positive skill vs naive_zero, at this specific
snapshot of the feature pipeline, reported, not yet claimed as a stable winner (single
run, no dedicated bootstrap-CI comparison of SVR vs naive_zero performed yet; see the Y1
experiment suite below for that comparison). GP scores close to zero (R2=-0.009),
consistent with the general small-N finding that GP's main value here is calibrated
uncertainty, not a point-accuracy win.

**Quantile regression (median, alpha=0.01, solver="highs")**: badly miscalibrated on
every target (Y1 R2=-2.73, Y2 R2=-1.33, Y3 substantially worse than naive), the fixed
alpha=0.01 is very likely too weak a regularizer for a ~76-row, ~20-feature design after
collinearity pruning, letting the LP-based fit chase individual training points. Recorded
as a failed configuration, not a working addition to the lineup as currently tuned; would
need either much stronger regularization or a proper inner-CV alpha search (not attempted
here, ponytail: shipped the lazy fixed-alpha version, this is exactly the case where it
measurably underperforms and the grid-search upgrade is warranted before using it for
anything).

#### 2026-09-16: collinearity-drop composed into classification feature selection too
(closing plan item 4 -- "re-run classification precision/recall/F1/AUC" after the
collinearity-pruning work above)

The composition above (`redundant_drop_set` |rho|>=0.95, then RF-importance top-K) had
only been wired into `notebooks/04_modeling_regression.ipynb`'s `select_top_features`.
`notebooks/05_modeling_classification.ipynb`'s own `select_top_features` (the
classification walk-forward loop, and the Y3 hurdle cell that reuses it) and
`scripts/train_final_models.py::_select_top_features` (the full-data refit that
`src/models/inference.py`'s demo serves) still ranked from the full collinear feature set.
Fixed by calling `redundant_drop_set` first in both, same as the regression side.

Re-ran `05_modeling_classification.ipynb` and `scripts/train_final_models.py`. Effect on
the full-data-refit classifiers the demo actually serves:

| label | AUC before | AUC after | family before -> after | beats_baseline before -> after |
|---|---|---|---|---|
| C1_negative_return | 0.711 | 0.672 | logistic -> xgb_clf | True -> False |
| C1b_adverse_move | 0.409 | 0.409 | xgb_clf -> xgb_clf | False -> False |
| C2_volume_spike | 0.711 | 0.711 | logistic -> logistic | True -> True |
| C3_recovers_in_90 | 0.824 | 0.824 | logistic -> logistic | False -> False |
| C3b_slow_recovery | 0.616 | 0.616 | xgb_clf -> xgb_clf | False -> False |

C1 is the only label collinearity-pruning changed, and it changed for the worse: the
prior `beats_baseline=True` (from the finding #14 median-imputation fix, documented
above) was riding a collinear feature the RF-importance probe had ranked highly only
because it duplicated signal already present elsewhere, removing the duplicate cost
it enough discriminative power to drop below the majority-rule/chance bar. Reported as
the actual direction of the result, not spun: `C2_volume_spike` remains the one
classification label that survives both fixes. Updated
`tests/test_inference.py::test_classification_predictions_are_valid_probabilities_with_verdict_metadata`
to assert the new (opposite) direction. 97/97 tests pass.

This closes plan item 4 from the collinearity/nonlinear-models plan (PCA/GP/SVR/Quantile
were already wired into the regression side only; the classification side was the one
piece left incomplete).

#### 2026-09-16: bootstrap made the primary verdict criterion, Diebold-Mariano demoted
to a diagnostic (P1 queue: "replace/supplement DM test as primary inferential tool")

`src/evaluation/verification.py::verdict_table` previously required BOTH the paired
event bootstrap AND Diebold-Mariano to agree before printing "BEATS BASELINE". DM's
own theory assumes a roughly stationary, weakly dependent h-step-ahead forecast-error
series, a progressively worse fit the more a target's horizon overlaps neighbouring
events (Y3 up to 90 trading days, Y1_EventWindow_0_10 up to 10; two of the findings
this document already accepts as an unfixed, disclosed limitation, see §20
"Embargo/purging"). The bootstrap only assumes exchangeability of events, which this
irregular, non-time-uniform panel of disasters actually satisfies. Requiring the
shakier test to also agree meant a real, bootstrap-confirmed effect could get vetoed
for no reason better than DM's own mismatch to this data.

Changed `verdict_table` so `verdict == "BEATS BASELINE"` iff the bootstrap CI excludes
zero (`boot["significant"]`) alone; DM is still computed and reported per row as a new
`dm_agrees` diagnostic column, not a gate. Nested feature selection was considered as
the other queued P1 item but is NOT changed: §21 already recorded the panel's own
verdict that full nested CV is unsupportable at this N (inner folds of 9/16/23 rows)
and that the correct response is a smaller search space, not a deeper search, current
code already matches that accepted compromise, so there was nothing to fix there.

Re-ran `06_evaluation.ipynb` + `scripts/audit_results.py`. Effect: 6 (model, target,
baseline) pairs now read BEATS BASELINE (was however many survived the AND-gate before, not separately recorded pre-change, since `verdict_table.parquet` only ever stored
the combined result). Of these 6, only 2 (`Y2_abnormal_volume` svr vs naive_train_mean,
`Y2_abnormal_volume` ensemble vs naive_zero) also have `dm_agrees=True`; the other 4
clear the bootstrap alone, now visible in the table instead of silently downgraded to
"better, not distinguishable". Reported as the actual, more permissive direction the
change produces, not spun: this makes more comparisons pass, which is the expected
consequence of removing a stricter, worse-fitting second gate, not evidence of a
methodological win. 97/97 tests pass.

#### Y1-specific experiment suite (SMOGN ablation, compact features, regime features,
ExtraTrees, single-task modeling, OOF ensemble weighting, shrinkage)

See `scripts/run_aspi_return_experiments.py` for the full implementation and
`artifacts/tables/y1_experiments_ranked.csv` / `artifacts/tables/y1_feature_stability.csv` for results.
Numbers appended below once the run completes.

#### 2026-09-16: methodology-audit freeze, target names, units, inclusion threshold

An external methodology review (ML/CSE/thesis-panel style audit, 44 numbered findings)
flagged three sources of drift that this section freezes, so every earlier mention of the
old names/threshold above is historical and describes runs made under them, it is not
retroactively edited, per the same "report deviations, don't hide them" discipline used
throughout this document.

1. **Target renaming.** `Y1_aspi_log_return` -> `Y1_ASPI_5D_Forward_LogReturn_Pct` (the
   name now matches the actual formula, 100*ln(P_t+5/P_t), fixing finding #1: stale
   day-0 descriptions elsewhere in the repo no longer match a name that says "5D_Forward").
   `Y1_car_5`/`Y1_car_10` -> `Y1_EventWindow_0_5_LogReturn_Pct` /
   `Y1_EventWindow_0_10_LogReturn_Pct` and multiplied by 100 (finding #9: these are raw
   cumulative log returns from the pre-event close, not abnormal returns against an
   expected-return model, so "CAR" was never accurate; finding #10: all three return
   targets now share the same percent units instead of mixing percent and decimal).
2. **Inclusion threshold frozen at >=1000 affected** (the thesis's original
   pre-registration), reverting the 700 (2026-09-12) then 500 (2026-09-15) reductions
   explored earlier this project (findings #2/#3/#34: the threshold had drifted across
   three values with a test still asserting the middle one, and the README's "no relaxed
   inclusion criteria" claim was false while it stood at 700/500). Verified via
   notebook 02's own diagnostic cell that raising 500->1000 changes nothing: only 2 raw
   EM-DAT records exist between those thresholds and neither matches to a tradable CSE
   session, so the final dataset is byte-for-byte the same 76 events either way, this
   revert is free.
3. **What this does not fix.** Findings #5-#8, #11-#22 (target-specific label-horizon
   purging, Y3 competing-risk censoring, missingness semantics, nested CV, and the
   prediction-origin alignment question) are separate, larger changes and are tracked
   as follow-on work, not resolved by this freeze.

#### 2026-09-16: target-specific label-horizon purging (methodology-audit finding #5)

Before this, `notebooks/04_modeling_regression.ipynb`'s `run_walk_forward` purged every
target against a single shared column (Y1's 5-trading-day horizon), applied once per
fold. That under-purged `Y1_EventWindow_0_10_LogReturn_Pct` (a 10-day horizon) and
`Y3_recovery_days` (up to 90 trading days), and over-purged `Y2_abnormal_volume` (which
has no forward horizon at all, its label is known the same session). Three other
call sites had **no purge at all**: the shared-MLP loop in notebook 04 (§8), every
classification label in `notebooks/05_modeling_classification.ipynb` (all 6 labels are
derived from a future-looking regression target), and `scripts/run_survival_model.py`
(Y3's own AFT model).

Fixed by adding one label-end-date column per target
(`feature_eng.build_targets`: `Y1_horizon_end_date`, `Y1_EventWindow_0_5/10_horizon_end_date`,
`Y2_label_end_date`, `Y3_label_end_date` -- Y2's is the event session itself, Y3's is the
trading session its recovery/cap was actually confirmed on) and a single source-of-truth
mapping (`notebooks/_shared.py`: `TARGET_LABEL_END_DATE_COL`, `LABEL_END_DATE_COL`).
`run_walk_forward` now purges and SMOGN-augments **per target inside the fold loop**
(previously once per fold, shared across targets) so each target's training rows are
purged against its own horizon before augmentation. The MLP loop (which cannot take a
different training set per target, since its output layer is shared) now purges against
Y3's horizon, the most conservative available, rather than nothing. Notebook 05 and
`run_survival_model.py` now purge every fold, per label/target, for the first time.

**Measured effect, Y3 AFT survival model (the only case with a like-for-like before/after
run available)**: pooled C-index 0.557 -> **0.640** (folds: 0.696/0.448/0.500/0.917 vs the
previous 0.696/0.448/0.500/0.583) on the same n=40 pooled test points, test rows are
untouched by purging, only training rows are removed, so this is a same-sample
comparison. This reads as a genuine correctness fix (the previous number was computed
with zero fold-boundary protection, i.e. it likely OVERSTATED accuracy from leakage, not
understated it), not evidence the model works better, reported honestly either
direction, per this document's standing rule.

Y1/Y2 point-regression numbers and the 6 classification labels also shifted slightly
(a handful of borderline-adjacent training rows now excluded per fold); no comparison's
qualitative conclusion changed (C2_volume_spike remains the only classification label
confirmed to beat baseline; no Y1 candidate is statistically confirmed either way).

#### 2026-09-16: purged inner CV (methodology-audit finding #6)

The outer-fold purge above only protects the outer test period, hyperparameter search
(GridSearchCV/RidgeCV) runs its own `TimeSeriesSplit` inside each fold's real training
rows, and an inner-training row's label can reach into an inner-validation window just
as easily as an outer one (the exact violation finding #6 names, "the same horizon
problem can occur inside inner CV").

Added `purged_inner_cv` (`notebooks/_shared.py`): runs `TimeSeriesSplit`, then purges
each inner split's training rows against that inner split's own validation origin using
the same per-target label-end-date columns from finding #5, and materializes the result
as a list of `(train_idx, val_idx)` pairs (accepted directly by `GridSearchCV`/`RidgeCV`'s
`cv=` parameter). Falls back to the unpurged split only if every inner split would
otherwise lose its entire training side, and prints a warning when that happens (did not
trigger in this run). Wired into all four `run_walk_forward` call sites in notebook 04
(primary config, dense-config ablation, K=10 ablation, external-feature-block ablation)
and notebook 05's classification loop; the now-dead unpurged `inner_cv` helper was
removed from both notebooks rather than left unused.

**Caught mid-implementation:** the K=10 ablation (04 §12) and external-feature-block
ablation (04 §13) call `run_walk_forward` directly and had been missed on the first pass, they still passed no `label_end_dates_all` at all (silently falling back to the
unpurged branch), which nbconvert's own exit code did not surface (the chained
`tail`/`grep` in this session's own run commands returns ITS exit code, not the
notebook's, the same class of masking bug already documented earlier in this file).
Caught by grepping the notebook for every `run_walk_forward(` call site rather than
trusting the first passing run, and by reading the execution log directly for
`CellExecutionError` rather than a shell exit code.

No pipeline-numbers regression from this fix beyond what finding #5 already changed, inner-CV purging affects which hyperparameters are SELECTED, not which rows are scored,
and at this N (~15-20 real training rows per fold) the purge rarely empties an inner
split. Re-ran the full chain (02 unaffected/skipped, no feature_eng change this step, 04, 05, survival, train_final_models); Y3 c-index unchanged at 0.640 (this script has no
inner hyperparameter search); classifier AUCs unchanged; 95/95 tests pass.

#### 2026-09-16: Y3 competing-risk censoring (methodology-audit finding #7)

Every recovery-day search previously ran the full 90-day window regardless of whether a
LATER qualifying disaster struck before day 90. An event scored as "recovered on day 47"
or "capped at 90" was, in several real cases, actually interrupted mid-observation by
the next disaster, its true recovery status past that point is genuinely unknown, not
the value the naive search happened to land on.

Fixed in `feature_eng.build_targets`: for each event, find the NEXT qualifying event's
own reference trading session (same alignment rule as the event itself, found
independently rather than assumed adjacent), and cap the recovery search at
`min(90, days_to_next_disaster)`. Two new columns record the outcome for every Y3
consumer: `Y3_censored` (bool) and `Y3_censor_reason` (`"recovered"` / `"cap_90"` /
`"next_disaster"`), added to `feature_eng.py`'s `EXCLUDE_COLS` in `notebooks/_shared.py`
equivalent (`notebooks/02_features_targets.ipynb`) so neither ever becomes a feature.

**Measured on the full n=74 dataset:** 64 events recovered cleanly, 7 were censored by a
later disaster before day 90 (previously silently scored as if fully observed to 90 or
scored as an uncontaminated "recovery" at whatever day the naive search landed on), and
only 3 genuinely exhaust the 90-day cap, down from the roughly 9-10 events the
uncorrected cap-only count previously reported, because several of those were actually
`next_disaster` cases that happened to also read >=90 under the old single-window search.

**Propagated to every Y3 consumer**, not just the AFT model, since the target itself is
now more correct for all of them:
- `AFTRecoveryModel.fit`/`concordance_index` (`src/models/survival_recovery.py`) and
  `HurdleRecoveryModel.fit` (`src/models/hurdle.py`) gained an explicit `censored`/
  `recovered` override parameter (default `None` preserves the old `y >= cap` inference,
  for backward compatibility and the modules' own self-tests). `run_survival_model.py`,
  `train_final_models.py`, and notebook 05's hurdle cell now pass the real indicator.
- **C3_recovers_in_90 and C3b_slow_recovery** (`src/models/classifiers.py`) needed a
  second, independent fix: `Y3_recovery_days < 90` no longer implies "recovered" (a
  next-disaster-censored row can have a small Y3 value too), so both labels now exclude
  `next_disaster`-censored rows entirely (treated as missing, same as an absent Y3
  value) rather than force them into either class, their true status by day 90 is
  unknowable under competing risks, not falsifiably negative.

**Effect on the AFT model** (n=40 pooled test points, same folds): c-index 0.640 ->
0.618 (still >0.5); RMSE 33.9 -> 20.2 and MAE 16.4 -> 9.4 (Y3 values themselves are now
correctly smaller for the 7 reclassified events); calibration Pearson r 0.036 -> 0.240;
reported censoring proportion 10.0% -> 15.0% (of which 5 of 6 pooled test-point censored
cases are now correctly attributed to a competing disaster, not the 90-day cap).
**Effect on C3_recovers_in_90**: full-refit AUC 0.618 -> 0.912, removing rows whose
true label was unknowable (rather than guessing) eliminated real label noise. 97/97
tests pass (2 new: `test_y3_censored_early_by_a_later_qualifying_disaster`,
`test_y3_genuine_recovery_before_a_later_disaster_is_not_censored`).

Not addressed by this fix: finding #12 (ordinary point-regressors, Ridge/RF/XGBoost/
the hurdle model's stage-2 regressor, still treat a censored Y3 value as if it were an
observed one; only the AFT model and the two classification labels above are now
censoring-aware) and the median-cut computation inside `label_slow_recovery`, which still
mixes genuine and censored durations when choosing its per-fold split point.

#### 2026-09-16: missing-data semantics (methodology-audit findings #13/#14)

`total_deaths`/`no_homeless`/`mag_area_km2`/`mag_wind_km2`/DesInventar physical severity
already had paired `_available` flags (`emdat_loader.py`, pre-existing). Two real gaps
remained, both explicitly named in the panel review:

1. **`financial_damage`** was zero-filled with NO indicator at all -- `damage_source`
   (the string that names WHY, e.g. `"missing_zero_filled"`) is excluded from
   `FEATURE_COLS` entirely because it is categorical, so the model genuinely never saw
   the missingness signal the column name implies it should. Fixed: `emdat_loader.py`
   now also emits a numeric `financial_damage_observed` flag (0/1) built from the same
   `damage_source` logic, alongside the existing string column. 18/74 modelled events
   have a real EM-DAT damage figure; 56 are zero-filled.
2. Four more feature groups were blanket zero-filled at `X = dataset[FEATURE_COLS]
   .fillna(0.0)` (notebook 02) with no flag: the volume-ratio block (`vol_ratio_1_30/
   5_30/10_30`, `vol_cv_30`, `log_vol_change_1` -- always missing together, since they
   share one underlying volume-data gap), `garch_cond_vol`, and the two annual macro
   series (`gdp_growth_pct`, `inflation_cpi_pct`). Fixed: three new flags --
   `volume_features_available`, `garch_cond_vol_available`, `macro_available` -- one per
   group that goes missing TOGETHER (not one per column, which would just manufacture
   near-duplicate near-constant features). Coverage on n=74: financial_damage 18/74,
   volume features 62/74, GARCH 70/74, macro 71/74.

`src/models/inference.py`'s `build_feature_row` (the demo app's severity-override path) already
flips `deaths_available`/`homeless_available`/`mag_area_available`/`mag_wind_available`
to 1.0 when the user supplies that value; `financial_damage_observed` now gets the same
treatment when the user overrides financial damage.

**What this does not fix**: finding #14's other half (feature-specific missingness
logic, train-fold median instead of a constant zero, for variables where zero has
real economic meaning) is not implemented; every affected column is still zero-filled,
just now with a flag alongside it rather than silently. That is a smaller, separable
change left for later if it is judged worth the added complexity.

**Effect on results**: feature count 64 -> 68 (4 new flags). Full pipeline re-run
(01 -- `financial_damage_observed` is built in `emdat_loader.py`, which notebook 01
caches, through 04, 05, 08, survival, train_final_models). Classification AUCs
essentially unchanged (C2_volume_spike remains the only confirmed label). The Y3 AFT
model's c-index moved 0.618 -> 0.508 (still nominally above chance, but only barely), the new features shifted which columns per-fold RF-importance selection picks for this
already-small-N target, and one fold now scores below chance (0.333) where it
previously did not. Reported as measured, not as a regression to explain away: adding a
methodologically-correct feature does not guarantee a better fit at N~74, and this
document's standing rule is to report the number either direction. 97/97 tests pass.

#### 2026-09-16: prediction-origin alignment (methodology-audit finding #8), Y1 rebaselined, EventWindow_0_5 and C4 consolidated

Every feature in X is snapshotted at `asof_date = event_date - 1` (the pre-event close).
Y1 was `100*ln(P_t+5 / P_t)` -- baselined on the EVENT-DAY close, a value only knowable
AFTER the event, and therefore not itself part of X's information set. That mismatch is
the actual defect: the day-0 reaction (P_(t-1) -> P_t) was silently excluded from BOTH
the predictors and the label, and the label's own baseline used information the model
was never given.

Presented as an explicit choice, not decided silently: **Option A** (rebaseline Y1's
denominator to the pre-event close, so the day-0 reaction becomes part of what Y1
measures) vs **Option B** (keep Y1 as-is, add day-0 features to X instead, which
reframes the research question from "predict the reaction" to "predict the
continuation given the reaction is already known"). User selected **Option A**.

Implemented: `feature_eng.build_targets` now computes
`Y1_ASPI_5D_Forward_LogReturn_Pct = 100*ln(P_(t+5) / P_(t-1))`. This is EXACTLY the
formula the pre-declared `Y1_EventWindow_0_5_LogReturn_Pct` already used, so Y1 is now
numerically identical to that column. Rather than ship two columns with the same value
under different names (feeding SMOGN and every model a duplicated feature as if it were
independent information), `EventWindow_0_5` is no longer computed as a separate target;
Y1 IS the consolidated column. `EventWindow_0_10` remains distinct.

**Cascading consolidation, not a separate decision**: `C4_car5_negative`
("`Y1_EventWindow_0_5 < 0`") became identical to `C1_negative_return`
("`Y1_ASPI_5D_Forward_LogReturn_Pct < 0`") for the same reason, one level up in the
classification labels. Removed rather than duplicated -- `LABELS` now has 5 entries, not
6. The `docs/EXTERNAL_DATA_PRE_DECLARATION.md` pre-declarations for both removed
columns are left UNCHANGED (that document records what was decided before any result
was seen, so it is not retroactively edited); this section and the removal comments in
`src/models/classifiers.py`/`feature_eng.py` are the record of what superseded them.

**Every consumer updated**: `notebooks/_shared.py` (`TARGET_COLS`, `TARGET_BOUNDS`,
`TARGET_LABEL_END_DATE_COL`, `LABEL_END_DATE_COL`), `src/models/inference.py`
(`TARGET_LABELS`, `LABEL_DESCRIPTIONS`, `target_bounds` -- also fixed two OTHER stale
Y1 descriptions this surfaced: "day-0 return"/"event day" language left over from
before this session's earlier Y1 redefinition), `src/evaluation/metrics.py` and
`src/visualization/result_figures.py` (`TARGET_BOUNDS`/`TARGET_LABELS`, same stale-description
fix), `src/features/sector_panel.py` (rename map), `notebooks/08_sector_panel.ipynb`
(classification cell), `app.py` (a caption naming C4 by name), and every test that
constructed a target/label fixture or asserted `LABELS`/`TARGET_LABELS` membership.

**Effect on results**: TARGET_COLS 5 -> 4, LABELS 6 -> 5. Y1's value itself changed
for every event (it now includes the day-0 move), verified numerically identical to
the pre-change `Y1_EventWindow_0_5_LogReturn_Pct` value for a sample event
(1.4839099892177814 either way). Y2/Y3/EventWindow_0_10 and the AFT survival model are
untouched (none of them read Y1). Full pipeline re-run (02 through train_final_models);
97/97 tests pass.

**What this does not do**: it does not touch feature construction (X is still
snapshotted at t-1, unchanged) or resolve whether the ex-post severity-informed
specification (the whole app/thesis's stated scope) is itself a "prediction" in the
forecasting sense, that scope statement (README, `src/models/inference.py` docstring)
already exists and is unaffected by this fix.

#### 2026-09-16: train-fold median imputation (methodology-audit finding #14's other half)

Finding #14 allows either "train-fold median or predefined constant + missing
indicator", the earlier fix this session (finding #13/#14, above) used the constant
(zero) half, with a flag. Asked directly why not the more defensible median half, the
answer was: it cannot be a single global median computed once, the same as any other
per-fold statistic in this pipeline (RF-importance selection, SMOGN, Ridge's alpha
search), a global median computed before the fold split leaks future rows' values
into early folds' imputation.

Implemented `median_impute_from_train` (`src/training/walk_forward.py`): median of
`MEDIAN_IMPUTE_COLS` (the 16 columns from the finding #13/#14 fix, now including two
DERIVED columns, `log_financial_damage`/`log_damage_x_flood`, imputed independently by
their own median rather than recomputed from an imputed `financial_damage` -- a known,
accepted simplification, not a further target for this fix) computed from a fold's REAL
training rows only, applied to that fold's train and test frames; falls back to 0.0 only
if a column is entirely NaN in that fold's training rows (a real degenerate-fold
fallback, not a silent global default).

**Required un-baking a zero-fill that happened upstream of any fold split**:
`financial_damage` was zero-filled inside `emdat_loader.load_emdat` itself, before
`dataset.parquet` was ever written, median imputation is impossible after that point,
so the loader now leaves it (and its EM-DAT-provenance category, renamed
`"missing_median_imputed"`, was `"missing_zero_filled"`) as real NaN. Every other flagged
column (volume ratios, GARCH, macro) was already real NaN in `dataset.parquet`, just
blanket-zero-filled at each notebook's own `X = dataset[FEATURE_COLS].fillna(0.0)` --
those call sites now skip `MEDIAN_IMPUTE_COLS` in that blanket fill instead, leaving them
for the per-fold step.

**A latent bug this surfaced, fixed alongside it**: `log_damage_x_flood =
log_financial_damage * disaster_Flood` relies on multiplying by a 0/1 indicator to zero
out non-Flood events, but `NaN * 0.0 == NaN` in float arithmetic, not `0.0`. While
`financial_damage` was always zero-filled (never NaN), this was silently safe; once it
can be real NaN, every non-Flood event with missing damage would have gone spuriously
NaN on a term that is definitionally 0 for non-Flood events. Fixed with an explicit
`np.where(disaster_Flood == 1, log_financial_damage, 0.0)` in both `feature_eng.py`
(historical dataset) and `src/models/inference.py`'s `build_feature_row` (which recomputes the
same term on every call, live-app included).

**Wired into every model-fitting consumer**, not just the primary walk-forward loop:
notebook 04 (main loop, the shared-MLP loop, the PCA ablation, the dense/K10/feature-
block ablations inherit it automatically through the shared `run_walk_forward`),
notebook 05 (the classification loop and the hurdle-model cell), `run_survival_model.py`.
Two paths have no fold at all because they refit on ALL real data by design (already
documented elsewhere as in-sample, not held-out), notebook 04's final SHAP refit and
`scripts/train_final_models.py` -- so they use one GLOBAL median instead, computed once
in notebook 02 and persisted in `feature_spec.json` (`MEDIAN_IMPUTE_VALUES`) precisely so
every "refit on everything" consumer, including `src/models/inference.py`'s live demo, imputes
with the identical numbers rather than each recomputing its own. Notebook 07's SHAP
values now use that same global fill too, explaining a model on differently-imputed
inputs than it was trained on would have been a real, if quiet, inconsistency.
`notebooks/08_sector_panel.ipynb`'s classification cell still blanket-zero-fills (its
own `PANEL_FEATURES` construction, unchanged), noted as a residual gap, not silently
left inconsistent.

**A second bug caught only by running it**: `scripts/run_survival_model.py`'s
calibration section calls `pd.qcut(..., q=3, labels=[...], duplicates="drop")`, which
raises when tied predicted values collapse the bin count below 3 and pandas then rejects
the fixed 3-label list. This did not fire before (the previous prediction distribution
never had that many ties) but did after this fix changed the model's inputs, caught
by actually running the script rather than assuming a passing test suite meant the
scripts were fine too. Fixed with a fallback to pandas' own integer bin labels when the
named ones don't fit.

**Effect on results**: Y3 AFT c-index 0.508 -> **0.553** (a real improvement, not
guaranteed by the fix, reported either direction per this document's standing rule).
More notably, **`C1_negative_return` now clears both the majority rule and chance**
(full-refit AUC 0.711, `beats_baseline=True`), a second confirmed classification
result alongside `C2_volume_spike`, where before this fix only one label cleared the
bar. 97/97 tests pass (two updated: the round-trip test now expects the global median
for a missing feature, not 0.0; the classification test now expects `C1_negative_return`
confirmed).

#### 2026-09-17: second external panel re-verification pass, 13 remaining gaps closed

A second, independently-written panel document (same 44-finding register, restructured
into a severity table plus an "8 weaknesses to fix first" list) was checked line-by-line
against the ACTUAL current code, not against this document's own claims, an Explore
subagent was asked to verify 12 specific items with file:line evidence, deliberately
adversarial to the possibility that something marked "done" above was only partially
done. Result: 11 of 12 checked items were genuinely unfixed, plus 2 more found by direct
grep during the same pass (a stale `app.py` string, a never-regenerated
`docs/RESULTS_AUDIT.txt`). All 13 are closed by this section. Every one of these is a
methodology-audit **followup** -- a real gap this document had not yet closed, found by
re-verifying against code rather than against this document's own prior entries.

**Cheap fixes (findings #1/#33, #4/#35, #14):**
- `app.py`'s results caption still read "day-0 return", fixed to "5-day forward return".
- `docs/RESULTS_AUDIT.txt` still had the pre-freeze target names (`Y1_aspi_log_return`,
  `Y1_car_5`) baked into it from a run before this session's renames, regenerated via
  `scripts/audit_results.py`.
- `scripts/run_garch_ablation.py` and `scripts/run_aspi_return_experiments.py` had kept their own
  blanket `X_all.fillna(0.0)` at load time, missed when finding #14's median-imputation
  fix was wired into the main pipeline and `run_survival_model.py` but not these two
  side-experiment scripts. Fixed to match: `MEDIAN_IMPUTE_COLS` stay real `NaN`, imputed
  per-fold via `median_impute_from_train` inside each script's own walk-forward loop.

**Finding #11, Y3's adverse-response gate** (`src/features/feature_engineering.py`,
`build_targets`): the recovery search used to start exactly at the event session, so an
event whose own close already sat above the pre-event baseline scored `Y3=0`
immediately, even if the index fell BELOW baseline a few sessions later within the
same short window. That later drop was invisible to the search because it had already
"recovered" on day 0 by construction, which the panel correctly flagged as a structural
weakness given the study's own premise of delayed price discovery. Fixed with a
prespecified 5-trading-day post-event gate: the lowest close in that window (or the
competing-risk cap, if shorter) is checked against the pre-event baseline first
(`Y3_drawdown_occurred`, a new diagnostic column, excluded from `FEATURE_COLS`); only if
a real drawdown occurred does the recovery search run, and it runs from that trough, not
from event day. 51/74 events now show a real drawdown in the gate window (were
previously scored purely on the day-0 comparison). Two new tests added
(`test_y3_delayed_crash_after_a_resilient_event_day_is_not_missed`,
`test_y3_is_zero_only_when_no_drawdown_occurs_in_the_gate_window`), covering both the
bug scenario and the case the fix must NOT change (a genuinely resilient event still
scores `Y3=0`). This is a target redefinition with a large blast radius, required a
full pipeline rerun and touches every Y3-derived number below.

**Finding #15, SMOGN's minority mask was hardcoded to Y3, reused for Y1/Y2**
(`notebooks/04_modeling_regression.ipynb::augment_fold` and the two side-experiment
scripts' own copies): every target's SMOGN call used `Y3_recovery_days > 30` as the
minority mask, even though a slow Y3 recovery has nothing to do with what makes a row
rare for Y1's return magnitude or Y2's volume spike. Fixed with a per-target relevance
function: Y3 keeps its threshold; Y1/`Y1_EventWindow_0_10` treat the bottom quintile of
that fold's training returns as the minority class; Y2 treats the top quintile as the
minority class. This is NOT the previously-tried-and-reverted wide-mask variant (that
one unioned all three targets into one shared augmented table); each target still gets
its own separate augmented table, just with its own relevant mask.

Re-measuring the E08/E09 SMOGN on/off ablation AFTER this mask fix changed the verdict
for 2 of 4 targets: Y2/Y3 still improve on both RMSE and pooled R2 with SMOGN on; Y1 and
`Y1_EventWindow_0_10` now get CONSISTENTLY worse on both metrics (RMSE +0.8%/+1.4%,
pooled R2 down, the EventWindow target's R2 even flipping sign), a real, both-metrics-
agree deterioration, unlike the earlier ambiguous ~1%-in-opposite-directions reading.
Per the panel's own rule ("prefer no SMOGN if your ablation showed deterioration"),
**SMOGN is now OFF by default for Y1 and Y1_EventWindow_0_10, ON for Y2/Y3**
(`SMOGN_TARGETS` in the notebook). The shared multi-output MLP (one training table
across all targets) and the two Y1-only side-experiment scripts keep Y1's own relevance
function as their anchor, since a shared architecture can't take a different mask per
target, disclosed as an architectural constraint, not an oversight.

**Finding #17, RF-importance feature selection applied to every model family
including Ridge**: an RF-importance ranking is not the right selection criterion for a
linear model. Fixed for Ridge specifically (the panel's own named example, and the one
model here where the alternative, let regularization do the shrinkage, is
unambiguous): `select_top_features` now returns both the RF-importance top-k AND the
full collinearity-pruned set, and Ridge fits on the full pruned set (its own `RidgeCV`
already searches alpha on this same set). RF, XGBoost, GP, SVR, and Quantile Regression
still share the RF-importance top-20 selection, there is no equally unambiguous
model-specific alternative for a kernel method or a boosted-tree ensemble, so this stays
scoped to the one case the panel actually named.

**Finding #20, no multiplicity correction across the many (model, target, baseline)
comparisons**: added a Holm correction (`statsmodels.stats.multitest.multipletests`)
across every row of `verdict_table`, reported as `holm_significant` -- a stricter,
family-wise-corrected diagnostic, not a replacement for the primary per-comparison
bootstrap CI (same relationship DM already has to the primary verdict, see the entry
above). Of 72 (model, target, baseline) comparisons in the final run, only **1 survives
Holm correction** (SVR vs naive_zero on Y2_abnormal_volume) against 9 that clear the
uncorrected per-comparison bootstrap CI, exactly the kind of conservative shrinkage a
correction across this many comparisons is supposed to produce, not a null result.

**Finding #22, the event-level bootstrap assumed independence between events that can
be the same or an adjacent disaster**: added `build_episode_ids` (events under 14 days
apart chain into one episode) and a `cluster_ids` option to `paired_bootstrap_delta`
that resamples episodes rather than individual rows. `verdict_table` now builds episode
ids from each comparison's pooled event dates and passes them through automatically;
72/72 rows in the final run were clustered (event dates were recoverable for all of
them). This required threading the original dataset row index through
`score_and_record`'s stored arrays (previously discarded via `np.asarray`) and through
the ensemble/stacked meta-learner cells, which pull already-index-stripped predictions
out of other models' stored results and had to re-wrap them in a `pd.Series` with the
real index before recording.

**Finding #23, no sensitivity check excluding EM-DAT's imprecise-date events**: 5 of
74 events (droughts) carry `month_only` date precision rather than an exact start day.
Added `notebooks/06_evaluation.ipynb` §6.2.2, which re-scores RF's pooled R2 restricted
to the 69 `exact_day` events using the SAME cached out-of-fold predictions (no re-fit,
just a filtered evaluation via the row-index tracking above). Result
(`artifacts/tables/exact_date_sensitivity.parquet`): all 4 targets move by a few hundredths of
R2 in either direction (e.g. Y1 0.058 -> 0.073, Y2 0.176 -> 0.156), no target's
headline number depends on the 5 imprecise-date rows in a way that would change its
qualitative conclusion.

**Finding #31, feature-selection stability across folds never reported**: this was
already implemented in `scripts/run_aspi_return_experiments.py` (a `feature_log` /
`selection_frequency` table, written but the script had never been run, flagged as
pending in this document since 2026-09-14). Running it as part of this pass produced
`artifacts/tables/y1_feature_stability.csv` for real, closing the gap for Y1 (the study's
primary magnitude target); the other targets don't have an equivalent per-fold
stability table, which is a real but smaller residual scope gap, not something silently
claimed as done.

**Finding #12, survival/hurdle reported as co-equal with plain regression for Y3, not
primary**: relabeled in `scripts/run_survival_model.py`'s module docstring and
`docs/FINAL_ANALYSIS_PROTOCOL.md` §6, the AFT survival model and the two-stage
hurdle model are now the PRIMARY reported result for Y3; plain point-regression fit
directly to `Y3_recovery_days` (which treats every capped or competing-risk-censored row
as an observed recovery time) is a secondary diagnostic that shows the SIZE of that
distortion, not a competing number. No code changed, this was a reporting-hierarchy
gap, not a modeling one.

**Finding #19/#44, model-family proliferation vs "freeze a final candidate set"**:
`docs/FINAL_ANALYSIS_PROTOCOL.md` §6 now explicitly splits the model lineup into a
**primary candidate set** (Ridge, RF, XGBoost, MLP for all 4 targets; AFT survival +
hurdle for Y3; the 5 classification labels) matching the panel's own recommended
minimal structure, and an **exploratory/appendix** set (GP, SVR, Quantile Regression,
the ensemble blend, the stacked meta-learner, PCA) that is reported but not what any
headline claim rests on. This is a disclosure fix, not a deletion, none of the
exploratory models were removed, since they were added deliberately during this
session's own feature/model-quality pass and still answer real ablation questions; they
are just no longer presented as co-equal with the primary set.

**Finding #38, no final lockbox holdout, and the adaptive-selection risk from
extensive tuning was never explicitly disclosed**: `docs/FINAL_ANALYSIS_PROTOCOL.md`
§8 now states plainly that no lockbox exists (74 events split across a 30/10/10
walk-forward already spends nearly the whole timeline on training folds; reserving a
further tail would cost folds the study cannot spare, the same tradeoff already accepted
for Y3's embargo overlap) and names this as the panel's own stated fallback when a
lockbox is impossible: freeze the protocol now, disclose the risk, and treat the
bootstrap CIs and Holm correction as the conservative response to it rather than a claim
the risk has been eliminated.

**Full pipeline rerun** (required by finding #11's target redefinition): notebooks
02 -> 04 -> 05 -> 08 -> 07 -> 06, then `run_survival_model.py`, `train_final_models.py`,
`run_aspi_return_experiments.py` (never previously run, see finding #31 above),
`run_garch_ablation.py`. One real bug caught only by running it: a hand-written f-string
newline (`print(f"\n>>> ...`) landed as a literal newline character instead of an escape
sequence when a notebook cell was assembled from a Python string, producing a
`SyntaxError` inside the executed notebook, fixed by rebuilding the source line
byte-for-byte rather than re-templating it.

**Effect on results** (reported in whichever direction it actually moved, per this
document's standing rule):
- Y3 AFT c-index: 0.553 -> **0.575** (per-fold: 0.515, 0.500, 0.697, 0.586).
- Y3 point-regression (secondary diagnostic): pooled R2 now positive for RF/Ridge/GP/SVR
  (previously negative for most models before the trough fix); 5 models now clear
  `naive_zero` in `verdict_table` for Y3 (RF, Ridge, GP, SVR, ensemble) where before this
  pass few or none did.
- Classification: **`C3b_slow_recovery` now clears both the majority rule and chance**
  (best family xgb_clf, AUC 0.811, `beats_baseline=True`), a third confirmed
  classification result, alongside `C2_volume_spike` and `C1_negative_return`, and a
  genuinely new result from the Y3 target redefinition specifically (this label is
  derived from `Y3_recovery_days`). `tests/test_inference.py` updated to assert it.
- `verdict_table`: 9 (model, target, baseline) pairs now read BEATS BASELINE under the
  cluster bootstrap (was 6 under the pre-clustering, pre-SMOGN-decision run two sections
  above), reported as the number this specific run produced, not implied to be
  directly comparable to the pre-fix count given how much changed between them.
- 99/99 tests pass (97 + 2 new Y3 gate-window tests).

This closed 11 of the 13 items; findings #16 and #26 were left as-is at the time, with
the following (as it turned out, partly wrong) reasoning: #16 was called an already-
accepted panel-endorsed tradeoff (§21), and #26 was called moot because a repo-wide grep
for "adfuller|kpss" had returned nothing. Both calls were revisited the same day, see
the next section.

#### 2026-09-17: findings #16 and #26 re-examined -- #26's "moot" call was wrong, #16
implemented anyway despite the accepted tradeoff

**Finding #26 was NOT moot.** The grep that produced that verdict missed
`notebooks/02_features_targets.ipynb` §2.2.1 (`from statsmodels.tsa.stattools import
adfuller, kpss`), which runs a real ADF/KPSS unit-root test on every engineered market
column and uses the verdict to decide `NON_STATIONARY_COLS` -- exactly the mechanism
finding #26 describes, and it ran on `market_feats`, the FULL ~22-year daily series,
before any walk-forward split existed. Every test fold's own date range was therefore
inside the very computation that decided which columns that fold was even allowed to
see, a real, if narrow, leakage: the admissibility decision itself (not a model fit)
used future-period data.

Fixed with a genuine development-period cutoff: `_dev_cutoff_date` is the date of the
31st qualifying disaster (`_DEV_TRAIN_WINDOW = 30`, matching `TRAIN_WINDOW` declared
later in the same notebook, fold 0's test period starts there, so no row used for the
stationarity decision can be inside ANY walk-forward test fold, since every later fold's
test period starts later still). `market_feats_dev = market_feats[market_feats["date"] <
_dev_cutoff_date]` replaces the full series as the ADF/KPSS input. Re-running produced
the IDENTICAL `NON_STATIONARY_COLS` (`sma_5/10/20`, `ema_5/10/20` -- the six raw
price-level columns, whose unit root is obvious at any reasonable sample size) and the
same 68 `FEATURE_COLS`, so this fix costs nothing in practice; it closes a real
procedural leakage risk without changing a single admitted feature.

**Finding #16 was reconsidered, not just re-affirmed.** §21's position, full nested
CV is unsupportable at N~9-16-23 inner-fold rows, and the correct response to noisy
inner selection is a smaller search space, not a deeper search, is not wrong on its
own terms, and is not retracted here. But it was also a decision the panel's own
diagram explicitly contradicts (`outer training -> inner training: impute, scale,
remove collinearity, select features, tune hyperparameters -> inner validation`), so
"already decided against it" was this document overriding the panel rather than
following it. Implemented anyway: added `CollinearityDropper` and
`CollinearityRFTopK` (`src/evaluation/collinearity.py`), two sklearn-compatible
transformers wrapping the existing `redundant_drop_set` + RF-importance composition.
Ridge's alpha search and RF/XGBoost's hyperparameter grids are now `Pipeline`s with the
selector as the first step, handed the FULL real-training feature matrix (not a
pre-selected subset) inside `GridSearchCV(cv=cv_real)` -- sklearn refits the selector
independently on every inner-CV split and every hyperparameter candidate, so an
inner-validation row's own label can no longer have quietly influenced which columns
even reached the model being scored on that row. The FINAL refit (on all outer
training, real+synthetic) is UNCHANGED, still one `feat_cols`/`ridge_cols` selection
per (fold, target), matching the panel's own diagram's next step exactly ("choose
configuration -> refit on all outer training -> evaluate once on outer test").

GP, SVR, and Quantile Regression are NOT nested, they run no hyperparameter search at
all (fixed kernel/config, see §6's model docstrings), so there is no inner-CV loop for
a selector to nest inside; they still use the single outer-selected `feat_cols`.

**Effect on results** (reported in whichever direction it actually moved): Ridge's
Y2_abnormal_volume pooled R2 went **-0.612 -> -0.047** -- a large improvement, and
informative about what the un-nested full-pruned-set alpha search (finding #17,
2026-09-17 morning) had actually been doing: selecting an alpha against a set of
inner-validation folds whose own labels the collinearity step (technically
target-blind, but evaluated on the SAME rows the search then scored) had an easier time
overfitting to than a genuinely re-derived-per-split selection does. Random Forest and
XGBoost's pooled R2 moved by low single-digit percentage points on every target, a
real but much smaller effect than Ridge's, consistent with RF-importance already being
fit fresh inside the (previously un-nested) search rather than being a fixed,
leakage-prone constant the way the full pruned set effectively was for Ridge's
`RidgeCV`. Full pipeline rerun (02 -> 04 -> 05 -> 08 -> 07 -> 06, plus all 4 scripts);
99/99 tests pass unchanged (neither fix touched a target definition, a censoring rule,
or anything a test's fixture asserts on directly).

This closes all 13 items from the second panel document's re-verification pass.

---

## 2026-09-17 (evening), Y1/Y3 improvement run

Logged here because `docs/FINAL_ANALYSIS_PROTOCOL.md`'s preamble requires every change
after the freeze to be recorded as a new dated section. **The frozen pipeline itself is
not changed by this work.** Nothing in §1-§7 of the protocol was modified, no notebook was
re-run, and no artifact the nine stages read was rewritten.

### What was done

A separate, additive analysis of Y1 and Y3 only, specified in full in
`docs/Y1_Y3_IMPROVEMENT_PREDECLARATION.md` **before execution**, reported in
`docs/Y1_Y3_FINAL_RESULTS.md`, with the method-level before/after in
`docs/FINAL_METHOD_COMPARISON.md`.

New code, all of it additive:

* `src/targets/return_horizons.py` -- the four pre-declared Y1 horizons (5/10/15/20 trading
  sessions, `100 ln(P_{t+h}/P_{t-1})` from the last pre-event close, each with its own
  `*_horizon_end_date` for target-specific purging), plus the hand-declared market /
  disaster information partition. Targets are built IN MEMORY from `market.parquet`, never
  by regenerating `dataset.parquet` -- which is why Y2 cannot move.
* `src/evaluation/survival_metrics.py` -- Harrell C-index, IPCW integrated Brier score,
  recovery-probability calibration against Kaplan-Meier, uncensored-only point errors,
  episode-clustered bootstrap CI.
* `scripts/freeze_baseline.py`, `scripts/run_aspi_return_grid.py`,
  `scripts/run_recovery_survival_grid.py`, `scripts/build_final_tables.py`.
* `tests/test_y2_frozen.py` (20 assertions), `tests/test_y1_horizons.py` (9),
  `tests/test_survival_metrics.py` (11), `tests/test_improvement_artifacts.py` (12).
  Suite: **99 -> 151, all passing.**

The ONLY edit to a pre-existing shared file is an additive `return_draws=False` keyword on
`survival_metrics.cluster_bootstrap_ci` -- a module created during this same session, with
no importer anywhere in the Y2 path.

### Deviations from the frozen protocol, and why

1. **SMOGN is OFF across the whole new Y1 grid** (protocol §5 keeps it ON for the frozen
   pipeline, which is untouched). Held constant so that the market-vs-combined and
   cross-horizon contrasts are not confounded by a minority mask that is defined from each
   target's own distribution and therefore varies across the four horizons and four
   information sets. Declared before execution
   (`Y1_Y3_IMPROVEMENT_PREDECLARATION.md` §2.6), not chosen after a result.
2. **Ridge/ElasticNet take an RF-importance top-K selection** in the new grid, where
   protocol §3 exempts Ridge from top-K in the frozen pipeline. Reason: the new grid's
   whole point is a controlled comparison of feature capacities K in {5, 10, 20}, which
   requires every model family to be given the same capacity. The frozen pipeline's Ridge
   is unchanged.
3. **Inner-CV degradation is stricter than `purged_inner_cv`.** Where that helper falls
   back to an unpurged `TimeSeriesSplit` when every purged split dies, the new grid reduces
   the number of splits and then uses pre-specified default hyperparameters, and never
   unpurges (protocol 1.7). In practice the fallback never fired: all 16 fold-horizons kept
   three purged inner splits.
4. **Y3's primary fit uses genuine events only.** Protocol §5's "SMOGN stays ON for all 4
   targets" applies to the frozen point-regression pipeline; a survival likelihood cannot
   accept a synthetic row, because SMOGN interpolates a duration without generating a valid
   event/censoring indicator. The existing SMOGN Y3 point-regression result is retained
   unchanged as ablation B6.

### Results, stated in the direction they landed

* **Y1 magnitude: 0 of 720 comparisons** (240 configurations x 3 baselines) had a bootstrap
  CI excluding zero. Best pooled R2 in the grid +0.075, CI [-0.560, +1.045].
* **The normal-market + disaster-residual decomposition did not help** at any horizon and
  was the worst information set at h=15 and h=20. Negative result, retained.
* **Y1 direction at h=10** (logistic, combined, K=10): AUC 0.752, CI [0.567, 0.896],
  **Holm p = 0.032** -- the only result in this study surviving a family-wise correction.
* **Y3 two-stage (drawdown classifier + Weibull AFT), K=20**: C-index 0.657, CI
  [0.522, 0.769], best calibration (max gap 0.045). Fails Holm (p = 0.165) and is therefore
  reported as **B, suggestive**, not A.
* **Feature selection is unstable**: 35.6% of all selected features were selected in
  exactly one of four folds.
* **Y2: no change of any kind**, at 1e-9 tolerance
  (`docs/Y2_FROZEN_VALIDATION_REPORT.md`).

### Stop condition

The pre-declared grid is complete and the pipeline is frozen again. No further horizon,
algorithm, threshold, target, feature-count increase, observation removal or variant search
follows from these results.

---

# Part 8. Revision 2 pre-declaration

**Dated:** 2026-09-22
**Baseline commit:** `fbe2699` (`refactor-4`)
**Status:** frozen before any result under these revisions was produced.

This section responds to the supervisor review recorded in
`CSE_Repo_Change_Specification.pdf`. It is written before any of the code changes it
describes were made, so that every decision below is a pre-declaration rather than a
report. Nothing in Parts 1 to 7 is edited; this Part is additive, and Part 7's change log
carries one dated entry per task.

## 8.1 What stays fixed

The three target definitions frozen on 2026-09-19 at commit `928a255`
(`docs/TARGET_DEFINITION_PROTOCOL.md`) are unchanged. The seed stays at 42. Validation
stays a chronological walk forward over events, thirty training events, ten test events,
step ten, four folds, forty pooled held out predictions per target. No shuffled cross
validation is introduced anywhere. No new machine learning model is added to the primary
analysis.

## 8.2 Horizons

**Primary horizon: five trading sessions.** Secondary, reported as sensitivity analyses
and never promoted: one, ten, fifteen and twenty sessions. The one session horizon is
added by this revision (T4) as a robustness test only.

## 8.3 The information set partition, by availability

The existing partition by data source (market only, disaster only, combined) is kept and a
second, orthogonal partition by availability at the prediction origin is added (T1):

**Real time**, demonstrably knowable at the prediction origin: lagged returns, moving
average ratios, rolling volatility, GARCH conditional volatility, volume ratios, exchange
rate returns and volatility, the S&P 500 return, election proximity, the disaster type one
hots, days since the last disaster, disasters in the trailing 365 days, and the event date
precision flag.

**Ex post**, finalised after the prediction origin: every EM-DAT severity and magnitude
column, every DesInventar `di_*` column, every NASA POWER `hz_*` column, and the annual
World Bank macro columns including `macro_available`. NASA POWER reanalysis is published
with a lag of several days and is therefore classed ex post; it will not be reclassified
without documented same origin availability.

Every feature column belongs to exactly one of the two. A column absent from either
partition raises, exactly as an unassigned column already raises for the source partition.

## 8.4 The confirmatory comparison family

The return grid may still run in full, at 240 configurations and 720 comparisons, but the
family wise correction no longer spans all of it. A configuration is **confirmatory** when
all four of the following hold:

1. the horizon is five sessions;
2. the information set is `real_time` or `ex_post`;
3. the feature capacity is **k = 10**;
4. the model is one of the reduced set below.

**The capacity is pre-declared here at k = 10** on two grounds that do not reference any
result: with thirty training events it is the conventional floor of three observations per
retained feature, and it is the capacity the existing direction analysis was already
declared at. The other declared capacities, five and twenty, remain in the grid as
exploratory.

**Reduced model set**, four entries:

| Slot | Model |
|---|---|
| Benchmark | `naive_zero` and `naive_train_mean` |
| Parsimonious linear | Ridge |
| Tree ensemble | Random Forest |
| Nonlinear | the shallow MLP |

`elastic_net` and `xgboost` are explicitly **not** confirmatory and are reported as
exploratory. The MLP is retained in the confirmatory set specifically because it is the
current best performer on the volume target, and under T8 it must be re-derived with the
same purged inner cross validation search as Ridge and Random Forest before that result
may be reported. Whatever the parity run produces is the finding.

Holm correction is applied to the confirmatory family only, its size is printed, and every
non confirmatory comparison is written to a separate exploratory artifact and is never
corrected jointly with the primary family.

## 8.5 The recovery target

**The survival analysis is primary for Y3.** The ordinary regression treatment is retained
for comparability with the existing literature but is demoted to a **disclosed
diagnostic**: squared error is not defined for a right censored duration, and sixteen of
the fifty two drawdown events are censored. No Y3 RMSE, MAE or R squared value will appear
in an exported table without an explicit diagnostic label (T3).

Censoring by a subsequent qualifying disaster is **informative**, not independent: an event
that has not recovered is more likely to be overtaken. A competing risks estimator is
therefore added (T2), together with a sensitivity run that excludes the events censored
that way, so the two treatments can be compared.

## 8.6 The realised response is a separate question from forecastability

A new event study inference module (T11) estimates and tests the realised market response
using the published statistics: a cross sectional t test, the Boehmer, Musumeci and Poulsen
standardised residual test, the Corrado rank test, and the Kolari and Pynnonen adjustment
for cross sectional correlation. It is kept structurally separate from the forecast
evaluation in `src/evaluation/verification.py`, and neither imports the other's decision
rules. The distinction between a measurable realised effect and out of sample
forecastability is the study's central contribution and must remain visible in the code.

## 8.7 The robustness suite, pre-declared and closed

One runner (T15) executes exactly the following list, once, into one consolidated table.
Each check reports its held out sample size, its error metric, its comparison against the
designated benchmark, and an uncertainty interval.

1. Restrict the sample to disasters with exact event dates.
2. Compare the five, ten and twenty session return horizons.
3. Compare raw ASPI returns against the market adjusted abnormal returns from T6.
4. Compare the real time and ex post information sets from T1.
5. Exclude, and separately analyse, the 2004 tsunami, the COVID-19 period and the 2022
   Sri Lankan economic crisis.
6. Run under both the sliding and the expanding training window from T9.
7. Leave one event out sensitivity, and a variant excluding the most influential
   disasters.
8. Compare across disaster types where the subgroup supports it. Subgroup sizes are flood
   51, storm 15, drought 5, other 3. Every subgroup result carries its size, and drought
   and other are never reported as standalone findings.
9. Examine the consequences of overlapping event windows and clustered disasters.
10. Repeat the principal analysis without SMOGN.
11. Repeat the principal analysis without the highest missingness variables.

**This list is closed.** No check is added after a result is seen, no check is dropped
because its result is unfavourable, and the suite is not run more than once. In the primary
folds SMOGN generates zero synthetic rows for Y1 and five each for Y2 and Y3 against thirty
real training rows, so check 10 is expected to change very little; that outcome will be
reported as it stands. Any check that cannot be run appears in the table with its reason
rather than being omitted.

## 8.8 The remaining tasks in this revision

| Task | What it declares |
|---|---|
| T1 | the availability partition in 8.3 |
| T2 | competing risks estimator plus the exclusion sensitivity run, 8.5 |
| T3 | Y3 regression demoted to a disclosed diagnostic, 8.5 |
| T4 | the one session return horizon, 8.2, registered as a non feature |
| T5 | Y2 at one, five, ten and twenty session windows, a percentage form, and a winsorised variant. Five sessions stays primary. Missing volume stays missing and is never zero filled |
| T6 | a market adjusted abnormal return target at every horizon, estimated only on sessions settling strictly before the prediction origin. The raw return stays primary |
| T7 | the confirmatory family in 8.4 |
| T8 | parity of hyperparameter search across every confirmatory model, 8.4 |
| T9 | an expanding window validation mode, defaulting to sliding so existing behaviour is unchanged |
| T10 | `STUDY_END` of 2025-12-31 bounding event eligibility, and a market data bound ninety sessions after the last qualifying event. The event count and the fold spans must be unchanged by the truncation, and the comparison will be reported |
| T11 | the event study module in 8.6 |
| T12 | a sample selection flow accounting reconciling 110 raw records to 94 qualifying to 74 modelled |
| T13 | the leakage check promoted from a notebook cell into the test suite |
| T14 | an exact dependency lock file beside the existing loose requirements |
| T15 | the closed robustness list in 8.7 |
| D1 | the repository description, corrected in `docs/repository_metadata.md`; the GitHub About field itself is a site setting and must be applied by the owner |
| D2 to D6 | documentation and metadata corrections, including withdrawing the "pre-registered" claim for the study as a whole in favour of the narrower, verifiable statement that the target definitions and evaluation rules were frozen on 2026-09-19 at commit `928a255` before any performance under them was observed |

## 8.9 Licence change, on the owner's instruction

**2026-09-22.** The repository licence changed from all rights reserved to the **MIT
licence for the code**, on the owner's explicit instruction, in response to D5 of the
review. The review asks that the code, seeds, versions and workflow be provided so the
results can be verified, and the previous terms did not permit that.

The change is deliberately narrow. MIT covers `src/`, `scripts/`, `tests/`, `apps/`,
`notebooks/` and `docs/`. It covers **no** third party data: EM-DAT, the Colombo Stock
Exchange archive, the World Bank, NASA POWER, DesInventar, FRED and Wikidata each keep
their own terms, and the `LICENSE` file says so explicitly rather than leaving it implied.

Anyone who relied on the previous terms is affected, and the earlier entry in
`docs/refactor_validation.md` recording the move to all rights reserved is now historical.

## 8.10 Stop condition for this revision

The list above is complete and closed. A negative result is an acceptable outcome, and the
study's contribution stands either way: a measurable realised market reaction to a natural
disaster does not imply that reaction can be forecast out of sample. No further target,
horizon, algorithm, threshold, feature count or observation removal follows from whatever
these revisions produce.


## 8.11 Change log for Revision 2, and the single final run

Dated 2026-09-22. This section records what each task changed and where its result now
lives. It is appended, as Part 8 has been throughout; nothing above it was edited.

| Task | What changed | Files touched | Artifact produced |
|---|---|---|---|
| T0 | Revision 2 pre-declaration, written before any new result | `docs/audit.md` Part 8 | none, documentation |
| T1 | Second, orthogonal partition of the features by availability at the prediction origin: `REALTIME_FEATURES` (37) and `EXPOST_FEATURES` (31). `information_sets()` returns five keys, and raises on a column missing from either partition | `src/targets/return_horizons.py`, `scripts/run_aspi_return_grid.py` | `artifacts/results/feature_spec.json` gains `AVAILABILITY_CLASS` |
| T2 | Aalen-Johansen cumulative incidence, chosen over Fine-Gray because the arm is a marginal baseline carrying no covariates, plus a sensitivity run excluding the ten events censored by a subsequent disaster | `src/models/survival_recovery.py`, `scripts/run_recovery_survival_grid.py` | `recovery_grid_metrics.parquet` and `*_excl_competing.parquet` |
| T3 | `Y3_REGRESSION_IS_DIAGNOSTIC = True`; every Y3 regression row is labelled a censoring-blind diagnostic, and the survival grid is the primary Y3 table | `src/config/settings.py`, `scripts/build_final_tables.py` | `final_table_recovery.csv`, `final_table_recovery_regression_diagnostic.csv` (18 rows, all labelled) |
| T4 | `HORIZONS = (1, 5, 10, 15, 20)`, `PRINCIPAL_HORIZON` unchanged at 5. The new column is registered in `NON_FEATURE_COLS` | `src/targets/return_horizons.py`, `src/config/settings.py` | new horizon columns in `dataset.parquet` |
| T5 | Y2 at 1, 5, 10 and 20 sessions against the same thirty-session baseline, the percentage form `100 * (exp(Y2) - 1)`, and a winsorised variant. Missing volume stays missing | `src/targets/event_targets.py` | new Y2 columns in `dataset.parquet` |
| T6 | Market model moved out of the grid script into a target module; abnormal return per horizon, estimated strictly on pre-origin sessions. Factor alignment accumulates, forward-fills and differences, to survive the CSE/S&P calendar mismatch | `src/targets/abnormal_returns.py` (new) | abnormal return columns in `dataset.parquet` |
| T7 | `confirmatory` flag on the metrics and verdict tables; Holm applied to that subset only; exploratory comparisons written separately | `scripts/run_aspi_return_grid.py`, `src/evaluation/verification.py` | `aspi_grid_verdicts.parquet` (18 confirmatory), `aspi_grid_verdicts_exploratory.parquet` (1332) |
| T8 | Tuned-versus-untuned asymmetry removed. In the notebook arm the Gaussian process kernel, the SVR `C` and the quantile `alpha` are now selected on the same purged inner splits as ridge, random forest and XGBoost; the MLP is excluded from the notebook confirmatory set and reported as exploratory there, and is tuned inside the confirmatory family in the grid | `notebooks/04_modeling_regression.ipynb`, `scripts/run_aspi_return_grid.py` | `results_regression.pkl` |
| T9 | `mode` parameter on the split generator, sliding by default so nothing existing moves, expanding starts the training block at index zero | `src/training/walk_forward.py` | used by T15 |
| T10 | `STUDY_END` at 2025-12-31 and a market-data bound ninety sessions after the last qualifying event. Series truncated 6366 to 6266 sessions, ending 2026-04-16 | `src/config/settings.py`, `src/data/cse_market_data.py` | `market.parquet` |
| T11 | Event-study inference, structurally separate from `verification.py` and asserted so by an AST import check in both directions | `src/evaluation/event_study.py` (new), `scripts/run_event_study.py` | `event_study_*.parquet`, event-time figures |
| T12 | Sample-selection flow instrumented at every exclusion | `src/data/emdat_disasters.py`, `src/features/feature_engineering.py` | `sample_flow.parquet`, flow figure |
| T13 | Leakage check promoted out of the notebook and into pytest | `tests/test_no_leakage.py` (new) | none, test only |
| T14 | Exact lock file, the transitive closure of `requirements.txt` rather than a bare pip freeze | `config/requirements.lock.txt` (new), `README.md` | 141 packages, Python 3.12.10 |
| T15 | One runner over the closed list, one consolidated table | `scripts/run_robustness_suite.py` (new) | `robustness_suite.parquet`, 26 checks |
| D1 | Repository description corrected; the About field is a github.com setting and needs the owner | `docs/repository_metadata.md` (new) | none |
| D2 | README states the log-ratio definition and the percentage companion | `README.md` | none |
| D3 | Volume verdict restated under T8 parity as Qualified rather than Yes, with the 34 held-out points and the structured missingness beside it | `README.md` | none |
| D4 | The pre-registration claim narrowed to the one that is verifiable | `README.md`, `docs/architecture.md` | none |
| D5 | MIT on the code, separate explicit terms for the third-party data, on the owner's instruction | `LICENSE` | none |
| D6 | `FORWARD_ABNORMAL_VOLUME`; `VOLUME_CRASH_MAGNITUDE` kept as a deprecated alias; the frozen column string is untouched | `src/config/settings.py` | none |

### The frozen Y2 baseline was re-anchored, and why

`tests/test_volume_target_frozen.py` failed on eight assertions after the re-run. The
cause is T8 and nothing else: the Gaussian process, the SVR and the quantile regressor
now select their hyperparameters on the purged inner splits instead of carrying fixed
values, so their predictions moved. Exactly those three models failed, and every model
that was already tuned passed unchanged.

What did not move is what the freeze exists to protect: the Y2 target values, the event
sample, the fold definitions and the classification results all passed untouched. The
baseline had been frozen at commit `f076bd96`, which predates T8, and T8's own acceptance
criterion requires the volume verdict to be recomputed under parity and reported as it
comes out. The baseline was therefore re-anchored at commit `9043bbf7`, and the previous
one is retained beside it as `artifacts/results/frozen_baseline_superseded_f076bd96.json`
so that the move stays auditable rather than being erased.

### One defect found and fixed during the run

`scripts/run_aspi_return_grid.py` reused a cached stage-A expected-return table without
checking that it covered the current `HORIZONS`. T4 added the one-session horizon, the
cache had been written without it, and the grid died on a `KeyError`. The cache is now
reused only when it carries every horizon in `HORIZONS`, and otherwise refits and says
which were missing. Stage A was refit, giving an estimate for 69 of 74 events at every
horizon.

### The single final run

Executed 2026-09-22 in the order Section 2 mandates. Notebooks 01, 02 and 03 had already
been re-run under Phase 1. Notebook 04 ran 11:50 to 14:53, notebook 05 to 15:07,
notebook 06 to 15:10, the return grid 15:10 to 17:50, then `train_final_models.py`,
`build_final_tables.py` and `audit_results.py`. Full suite: 226 passed.

Fold geometry is unchanged by everything above: four outer folds, thirty training events,
ten test events, three inner splits, forty pooled held-out points. The event count remains
74 and the sample flow still reconciles 110 to 94 to 74.

### Results, stated in the direction they landed

- Y1 magnitude: 0 of 450 configurations significant on even a single uncorrected test;
  0 of the 18 confirmatory comparisons had an interval excluding zero; smallest Holm
  corrected p is 0.641.
- Y1 direction at the pre-declared ten-session horizon: ROC AUC 0.817, interval
  [0.657, 0.940], Holm corrected p below 0.001. This is the only comparison anywhere in
  the study that survives a family-wise correction.
- Y2: pooled R squared 0.334 for the ensemble, 0.303 for the random forest. The random
  forest interval excludes zero against both baselines but does not survive Holm,
  corrected p 0.078, on 34 held-out points with structured missingness.
- Y3: every concordance interval contains 0.5; the Kaplan-Meier and Aalen-Johansen
  baselines carry the best integrated Brier scores.
- Realised response: no detectable return response, CAAR(1,5) = +0.0006 with all p above
  0.85. Volume CAAR(1,5) = +0.63 log points, with only the Corrado rank test firing,
  p = 0.024, against BMP p = 0.186 and Kolari-Pynnonen p = 0.298.
- Real time against ex post on the principal analysis: +0.0930 against +0.0932. Finalised
  severity information adds essentially nothing.
