# Final Analysis Protocol (frozen 2026-09-16, commit `e0cbd2f`)

This document freezes every decision the methodology-audit response (44 numbered
findings, `docs/METHODOLOGY_AUDIT.md`) touched, so the next pipeline run is the **one
last clean run** the panel's own P1 recommendation ("stop adaptive experimentation")
calls for. Everything below is now a specification, not a running log: change it only
for a stated reason unrelated to a held-out score (a data bug, a new event added to
EM-DAT, a reviewer requirement), never because a number looked disappointing. Any
future change must be logged as a new dated section in `METHODOLOGY_AUDIT.md`, exactly
like every change that produced this freeze.

## 1. Inclusion criteria and data

- EM-DAT disasters with `total_affected >= 1000` (frozen finding #2/#3/#34; verified
  byte-identical to the 500/700 thresholds explored earlier — no event between 500 and
  1000 matches a tradable CSE session).
- 76 raw qualifying disasters -> 74 events after matching to a tradable CSE session
  (2 unmatched).
- `financial_damage` is real `NaN` when EM-DAT reports none, not zero-filled at the
  loader (finding #14) — every consumer downstream imputes it explicitly (§4).

## 2. Targets

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
recovery-classification labels (`C3_recovers_in_90`, `C3b_slow_recovery`) — status is
genuinely unknowable there, not a label to guess at.

`Y3_recovery_days` also carries an adverse-response gate (finding #11, methodology-audit
followup 2026-09-16): the recovery clock no longer starts blindly at the event session.
The lowest close within a prespecified 5-trading-day post-event window (or the
competing-risk cap, if shorter) is checked against the pre-event baseline first
(`Y3_drawdown_occurred`); only if the index actually falls below baseline in that window
does the recovery search run, and it runs from that trough, not from event day. Before
this fix, an event whose own close already sat above the pre-event baseline scored
Y3=0 immediately, even if the index fell below baseline a few sessions later within the
same short window — that later drop was never seen because the search had already
"recovered" on day 0 by construction. `Y3_drawdown_occurred` is a diagnostic column
(excluded from `FEATURE_COLS`, computed from post-event prices), not a feature.

## 3. Features

- 68 columns in `FEATURE_COLS` (`artifacts/feature_spec.json`).
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
  live demo in `src/inference.py`) use a single GLOBAL median instead, persisted once
  in `feature_spec.json`'s `MEDIAN_IMPUTE_VALUES` and reused everywhere rather than
  each recomputing its own.
- Feature selection is always collinearity-drop THEN RF-importance top-20, fit on
  train rows only, in that order (`src/evaluation/collinearity.py::redundant_drop_set`,
  pre-declared `|rho| >= 0.95` rule, composed before ranking): both the regression loop
  (`notebooks/04_modeling_regression.ipynb`) and the classification/hurdle loop
  (`notebooks/05_modeling_classification.ipynb`), plus the full-data refit in
  `scripts/train_final_models.py`. Each target gets its OWN selected feature set — Y2
  and Y3 are never forced to reuse Y1's ranking.
- **Ridge is the one exception** to RF-importance top-20 (finding #17, methodology-audit
  followup): an RF-importance ranking is not the right selection criterion for a linear
  model, so Ridge fits on the full collinearity-pruned set instead (no top-K cut),
  letting its own L2 regularization do the shrinkage/selection — the panel's own
  suggested alternative for Ridge/ElasticNet-family models. RF, XGBoost, GP, SVR, and
  Quantile Regression still share the RF-importance top-20 selection; changing their
  selection criterion too is out of scope for this freeze (no obvious single "right"
  model-specific alternative for a kernel method or a boosted-tree ensemble the way
  regularization is for Ridge).
- Full nested CV (redoing feature selection per inner-CV split) is **not** used, and
  this is an accepted panel verdict, not an open gap: at inner-fold sizes of 9-16-23
  rows, feature selection inside the inner split would be selecting on noise. The
  correct response to a noisy inner search is a smaller search space, not a deeper one
  (`docs/METHODOLOGY_AUDIT.md` §21).

## 4. Validation protocol

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

## 5. Augmentation (SMOGN)

- Time-aware SMOGN, fit on the fold's real, post-purge training rows only, with the
  pre-registered 25% synthetic-share cap, applied per (fold, target) inside the target
  loop (never once per fold shared across targets).
- SMOGN on/off was measured, not assumed (P2-6, `artifacts/smogn_ablation.parquet`):
  helps Y1/Y2/Y3 on both RMSE and pooled R2; on `Y1_EventWindow_0_10` the two metrics
  move in opposite directions by ~1%, judged as fold-count noise (n=40 pooled points),
  not a real effect. **Decision: SMOGN stays ON for all 4 targets.**
- A previously-tried wider minority mask + 2x synthetic draws was measured and
  reverted (made every model on every target worse; also independently violates the
  25% cap) — recorded as a rejected variant, not deleted.
- Each target now defines its OWN minority/relevance mask (finding #15, methodology-
  audit followup): Y3 keeps its pre-declared `Y3_recovery_days > 30` threshold; Y1 and
  `Y1_EventWindow_0_10` treat the bottom quintile of that fold's training returns
  (most severe negative moves) as the minority class; Y2 treats the top quintile
  (largest volume spikes) as the minority class. Before this fix, every target's SMOGN
  call reused Y3's threshold verbatim, so a slow Y3 recovery — not an extreme Y1 return
  or Y2 spike — decided which rows got oversampled for Y1 and Y2. This is NOT the same
  change as the reverted wide-mask variant above: that experiment unioned all three
  targets' tails into ONE shared augmented table; this keeps each target's own,
  already-separate augmented table (unchanged since finding #5's per-target purge),
  just with a mask relevant to that specific target.

## 6. Models

**Primary candidate set** (finding #19/#44, methodology-audit followup: the panel's own
recommendation is "the smallest set that answers the research question" — a strong
linear baseline, RF/XGBoost, one neural comparison, and survival/hurdle for Y3 — rather
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
  as a **secondary diagnostic** — it shows the size of that distortion, not a competing
  primary number. See `scripts/run_survival_model.py`'s module docstring.
- Classification: 5 labels (`C1_negative_return`, `C1b_adverse_move`,
  `C2_volume_spike`, `C3_recovers_in_90`, `C3b_slow_recovery`), each with its own
  best-family model selected by walk-forward AUC (`build_classifiers`,
  `classification_summary.parquet`).

**Exploratory / appendix, not primary** (reported, but not what the thesis's headline
claims rest on):

- **Gaussian Process** (Matern + WhiteKernel, no grid) and **SVR** (RBF, fixed
  `C=1.0, epsilon=0.1`, no grid) — added during this session's feature/model-quality
  pass, after the panel's original review. Neither is demoted for a measured failure
  the way Quantile Regression is (below); they simply were never part of the panel's
  recommended minimal set, and adding them without removing anything is exactly the
  "model family proliferation" finding #19/#44 warns against. Reported for completeness,
  not treated as co-equal to the primary set above.
- **Median Quantile Regression** (`alpha=0.01`, `solver="highs"`) is a **known failed
  configuration** at its current fixed alpha (badly miscalibrated on every target) —
  kept in the lineup as a disclosed negative result, not silently dropped, not
  recommended for use until it gets a proper inner-CV alpha search.
- An **inverse-RMSE ensemble blend** and a **stacked meta-learner** over
  {RF, XGBoost, MLP} — reported as combination-method ablations.
- **PCA** is reported as a comparison only (top-10 components + Ridge vs RF-selected
  top-20 + Ridge, same folds) — not adopted as a default; PCA components are
  uninterpretable, which cuts against the SHAP explainability chapter.

## 7. Statistical verdict rule

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
  as a `dm_agrees` diagnostic column, not a gate — DM's stationarity/weak-dependence
  assumptions fit this irregular, overlapping-horizon event panel worse than the
  bootstrap's plain exchangeability assumption, so requiring both was demoting a
  real, bootstrap-confirmed effect for the wrong reason.
- A **Holm correction** (finding #20, methodology-audit followup) is applied across
  every row of `verdict_table` to the bootstrap's one-sided p-value, reported as
  `holm_significant` — a stricter, family-wise-corrected diagnostic column, since many
  (model, target, baseline) comparisons are run and some "significant" result is
  expected by chance alone. `verdict`/`boot_beats` (the single-comparison CI) stay the
  primary criterion, exactly as DM stays a diagnostic rather than a second gate — this
  is reported alongside the primary verdict, not folded into it, so a single-comparison
  result is never silently reclassified by a family-wide correction whose family
  membership (which comparisons count as "the same family") is itself a judgment call.
- Classification's `beats_baseline` is unrelated to this rule: it requires the AUC's
  own bootstrap CI to exclude 0.5 AND balanced accuracy > 0.5 (majority-rule check),
  computed in `src/models/classifiers.py`.

## 8. Known, accepted, disclosed limitations (not to be "fixed" without new data)

- Leave-one-disaster-type-out is descriptive only (Drought n=5); not a statistical claim.
- Y3's 90-day window can overlap a later event in a subsequent fold; disclosed, not purged.
- Inner-CV feature selection is not nested; accepted at this N (§3 above).
- Quantile regression is under-regularized at its current fixed alpha; a known, disclosed failure, not a bug to silently paper over.
- `notebooks/08_sector_panel.ipynb`'s own `PANEL_FEATURES` classification cell still
  blanket-zero-fills instead of using `MEDIAN_IMPUTE_VALUES` — a residual gap, flagged,
  not fixed as part of this freeze (low priority: sector panel is a secondary analysis,
  not a headline result).
- **No final lockbox holdout exists** (finding #38, methodology-audit followup). The
  panel's best fix — reserve the newest chronological events, untouched by any
  ablation or tuning decision, as a genuinely final held-out check — is not achievable
  at this N: 74 events split across a 30/10/10 walk-forward already spends nearly the
  whole timeline on training folds, and setting aside a further tail would cost the
  study folds it cannot spare (the same tradeoff already accepted for Y3's embargo
  overlap in §8 above). This freeze is the panel's own stated fallback instead: stop
  adaptive experimentation NOW, disclose the adaptive-selection risk explicitly (this
  bullet), and treat every walk-forward number the "one last clean run" below produces
  as carrying that residual risk — repeated human decisions across this session's many
  ablations, informed by the same out-of-fold results, can produce adaptive overfitting
  even with no single instance of direct test-set leakage. The bootstrap CIs and Holm
  correction in §7 are the conservative response to that risk, not a claim that it has
  been eliminated.
- Exact-date sensitivity (finding #23, methodology-audit followup): 5 of 74 events
  (droughts) carry `month_only` EM-DAT date precision rather than an exact start day.
  `notebooks/06_evaluation.ipynb` §6.2.2 re-scores RF's pooled R2 restricted to the 69
  `exact_day` events, using the same cached out-of-fold predictions (no re-fit), as a
  check on whether the headline number depends on those 5 alignment-uncertain rows.
  Reported in `artifacts/exact_date_sensitivity.parquet`.

## 9. What "one last clean run" means

1. Do not touch target definitions, inclusion threshold, purge columns, SMOGN
   parameters, feature-selection composition, or the verdict rule while looking at
   this run's numbers.
2. Re-run notebooks 01 -> 02 -> 04 -> 05 -> 08 -> 07 -> 06, then
   `scripts/run_survival_model.py` and `scripts/train_final_models.py`, in that order,
   from a clean `artifacts/` state.
3. Run `pytest tests/ -q` — must be 97/97 (or the then-current full-suite count) before
   any number from this run is reported as final.
4. Report every number from this run, in whichever direction it lands, in
   `docs/RESULTS_AUDIT.txt` / the thesis results chapter. A disappointing result is
   itself a finding this study is equipped to report (see §5, §6 above for two
   already-disclosed examples), not a reason to reopen §1-§7.
