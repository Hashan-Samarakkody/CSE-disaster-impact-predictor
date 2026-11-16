# Leakage Fix Commits

Tracing each leakage/data-quality item from the manuscript (Appendix H, Table 16) to the
repository commits that resolved it. Items are numbered as they appear in the manuscript
table. Commits are from this repository's `main` branch; run `git show <hash>` to inspect
the diff.

---

## Item 1 — Global SMOGN applied before fold split (Temporal, CRITICAL)

Synthetic oversampling was run on the whole dataset before splitting into folds, so test
events' target distribution influenced the synthetic neighbours seen in training.

**Fix commit:** `ca0b809` — REFACTOR THE CODE (2026-09-17)
- Walk-forward splitting moved to precede any SMOGN call; oversampling now applied inside
  each training fold only.

**Verification:** `47dbb8c` — IMPROVEMENT (2026-09-18) confirmed fold geometry correct.

---

## Item 2 — Walk-forward fold boundaries computed on post-SMOGN row count (Temporal, CRITICAL)

Fold indices were computed on `len(X)` after synthetic rows were appended, so split
positions drifted relative to the real event chronology.

**Fix commit:** `ca0b809` — REFACTOR THE CODE (2026-09-17)
- Splits now computed on `len(real_X)` before augmentation; synthetic rows appended inside
  training blocks only.

---

## Item 3 — `GridSearchCV(cv=3)` = random KFold inside each walk-forward fold (Temporal, CRITICAL)

Inner cross-validation for hyperparameter search used sklearn's default `KFold`, which
shuffles and may assign test-window rows to inner training.

**Fix commit:** `ca0b809` — REFACTOR THE CODE (2026-09-17)
- Inner CV replaced with `TimeSeriesSplit` on real (non-synthetic) rows; purged inner
  splits added in `src/training/walk_forward.py`.

**Reinforcing commit:** `4cfdef3` — P0 - 2 target-specific label-horizon purging (2026-09-16)
- `purge_horizon_overlap` applied to inner CV splits as well as outer folds.

---

## Item 4 — Hyperparameters selected partly on synthetic rows (Target, HIGH)

After SMOGN, hyperparameter search saw augmented training rows, so the selected
hyperparameters were partly tuned on fabricated events.

**Fix commit:** `ca0b809` — REFACTOR THE CODE (2026-09-17)
- Hyperparameter selection now runs on real rows only; model is refit on augmented
  training block after selection.

**Reinforcing commit:** `e9ff454` — Phases 3-5 code (2026-09-22)
- T8 parity: MLP hyperparameter grid declared and searched with the same purged inner CV
  as ridge and random forest.

---

## Item 5 — Annual World Bank macro features dated 1 January of their own year (Availability, CRITICAL)

`gdp_growth_pct`, `inflation_cpi_pct`, `macro_available` were assigned the date
`YYYY-01-01`, making 2010 annual figures available to events on 2010-01-15 — before the
year had elapsed.

**Fix commit:** `a6b4a0c` — Both P0 #8 and the median-imputation follow-up (2026-09-16)
- Macro features re-dated to `YYYY+1-07-01` (publication lag); events before that date
  receive the previous year's value.

**Examiner reinforcing commit:** `7a5cc97` — Address examiner corrections (2026-10-25)
- `macro_available` flag reclassified ex_post; feature availability parquet regenerated.

---

## Item 6 — Ex-post damage severity features used as real-time predictors (Availability, CRITICAL)

`financial_damage`, `total_deaths`, `population_affected`, `di_affected_log`, hazard
intensity columns (`hz_precip_max3d`, `mag_wind_kph`) require EM-DAT or NASA POWER data
finalised weeks after the event, yet were included in the real-time feature set.

**Fix commit:** `f7dc800` — Phase 1: data, targets and information sets (2026-09-21)
- `REALTIME_FEATURES` / `EXPOST_FEATURES` partition defined in
  `src/targets/return_horizons.py`; all damage/severity features assigned ex_post.
- Results now reported separately for each information set.

**Reinforcing commit:** `7a5cc97` — Address examiner corrections (2026-10-25)
- Three additional features (`date_is_exact`, `days_to_election`, `election_within_5d`)
  moved from `REALTIME_FEATURES` to `EXPOST_FEATURES` after examiner review.

---

## Item 7 — SMA/EMA computed on current-day price instead of lagged price (Temporal, CRITICAL)

Moving-average and EMA features used `close` without a `.shift(1)`, so the prediction
origin price leaked into the input features.

**Fix commit:** `a6b4a0c` — Both P0 #8 and the median-imputation follow-up (2026-09-16)
- All market features computed with `.shift(1)` to halt at P0 (last known close before
  the disaster onset).

**Verification commit:** `f7dc800` — Phase 1 (2026-09-21)
- `test_no_leakage.py` added to the test suite as T13; checks that no feature has
  look-ahead relative to prediction origin.

---

## Item 8 — SMOGN interpolated disaster type one-hot flags across different disaster categories (SMOGN quality)

Interpolating between a Flood row and a Drought row produced synthetic events with
fractional disaster-type flags (e.g., `disaster_Flood=0.6, disaster_Storm=0.4`), which
are physically impossible.

**Fix commit:** `e9ff454` — Phases 3-5 code (2026-09-22)
- SMOGN neighbour search restricted to same-disaster-type pairs; one-hot block copied
  verbatim from the seed row rather than linearly interpolated.

---

## Item 9 — Synthetic rows had derived features re-interpolated independently of parents (SMOGN quality)

SMOGN interpolated `log_financial_damage`, `damage_to_gdp`, `squared_return` and
`log_damage_x_flood` as independent columns, so `log_financial_damage ≠ log1p(financial_damage)`
on every synthetic row — a physically impossible combination.

**Fix commit:** `e9ff454` — Phases 3-5 code (2026-09-22)
- Derived features excluded from SMOGN interpolation; recomputed from their parent
  columns after interpolation.

---

## Item 10 — Ridge regression received un-standardised features (Preprocessing, HIGH)

L2 regularisation is not scale-equivariant. The feature matrix mixed ASPI price levels
(~1e4), raw USD damage (~1e9), and 0/1 binary flags; `alpha=1.0` applied near-zero
shrinkage to damage columns while crushing binary features.

**Fix commit:** `e9ff454` — Phases 3-5 code (2026-09-22)
- Model-specific pipelines: Ridge wrapped in `StandardScaler → Ridge`; Random Forest and
  XGBoost receive raw features (scale-invariant by design).

---

## Item 11 — Y2 target zero-filled for three missing observations (Target quality, HIGH)

`y = dataset[TARGET_COLS].fillna(0.0)` fabricated three Y2 (abnormal volume) labels,
asserting "volume exactly at its 30-day baseline" for events whose source workbook failed
to parse — the same zero-fill the notebook's §3 explicitly rejects for volume.

**Fix commit:** `77264c5` — P0 #5 (missing-data semantics) (2026-09-16)
- Targets retain `NaN`; each fold masks non-finite labels before fitting; Y2 effective
  sample size reported as 61 (not 74).

---

## Notes

- Items 1–4 were fixed together in the major refactor (`ca0b809`); no earlier standalone
  fix commit exists for these because the original pre-refactor code was not modular.
- Item 6 is additionally *disclosed* as a study limitation (ex-post features cannot
  be avoided in an impact attribution study); the fix separates real-time from ex-post
  reporting rather than removing those features.
- The leakage audit in `docs/audit.md §8` documents two items marked NONE (feature
  selection inside fold; scalers fit inside fold) — those were always correct and are
  excluded from this table.
