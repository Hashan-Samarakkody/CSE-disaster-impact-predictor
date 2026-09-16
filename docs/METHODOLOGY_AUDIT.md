# 10-Expert Methodology Audit — CSE Disaster-Impact Prediction Model

**Subject:** "A Machine Learning Approach to Predicting the Impact of Natural Disasters on the Colombo Stock Exchange" (S.D.S.H. Samarakkodi, IM/2021/007, University of Kelaniya; supervisor Dr. Thilini Mahanama)

**Audit basis:** Direct inspection of the `src/` source and the notebook pipeline, plus executed outputs from complete pipeline runs. (Paths in this document predate the 2026-09 repository restructure, which flattened `disaster_finance_predictor/` into the repository root; `src/`, `notebooks/` and `tests/` now sit at the top level.) **Every number in this document is copied from an actual execution.** Nothing is estimated, extrapolated, or invented. Where a figure is unavailable, it is marked "not measured".

**Authoritative methodology:** the thesis (July 2026). The April 2026 proposal is historical context only.

---

## 1. Research Understanding

The study asks: *how does the Colombo Stock Exchange respond to natural disasters, and can that response be predicted?* It is operationalised as a supervised multi-target regression over disaster events, not over trading days.

Three continuous targets per event:

| Target                | Definition as implemented                              | Range observed      |
| --------------------- | ------------------------------------------------------ | ------------------- |
| Y1`aspi_log_return` | `ln(P_t / P_{t-1})` on the event's first trading day | −0.0753 to +0.0394 |
| Y2`abnormal_volume` | `V_t / mean(V_{t-30..t-1}) − 1`                     | −0.87 to +1.94     |
| Y3`recovery_days`   | Days until ASPI regains`P_{t-1}`, capped at 90       | 0 to 90, median 0   |

Unit of analysis: **one qualifying disaster event**. N = 64 modelled events (2000-01 to 2023-06).

---

## 2. Proposal → Thesis → Implementation Evolution

| Component     | Proposal (Apr 2026)                                 | Thesis (Jul 2026)                                              | Implementation                                        | Verdict                                                                                        |
| ------------- | --------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Validation    | Stratified 5-fold CV inside 80% train block (§7.5) | Chronological walk-forward; k-fold explicitly banned (§3.7.1) | Walk-forward outer;**k-fold was running inner** | Thesis correct; implementation violated it — now fixed                                        |
| Target Y1     | "ASPI percentage change"                            | §3.2.2: raw log return                                        | Raw log return                                        | Terminology should be corrected to "log return"                                                |
| Oversampling  | SMOTE                                               | Time-Aware SMOGN                                               | Custom SmoteR-style implementation                    | Thesis correct                                                                                 |
| Recovery      | Trading days, cap 90                                | Trading days, cap 90                                           | **Calendar days**, cap 90                       | Implementation deviated — now fixed                                                           |
| Models        | +Logistic Regression                                | OLS/Ridge, SVR, RF, XGBoost, MLP                               | Ridge, RF, XGBoost, MLP                               | Logistic Regression correctly dropped (targets are continuous);**SVR never implemented** |
| Sentiment/NLP | Aggregated news sentiment listed (§3.3.1)          | Explicitly excluded (Table 4 note)                             | Not implemented                                       | Thesis correct; proposal wording superseded                                                    |
| Epidemics     | Included in scope (§4.5)                           | Excluded from acute-shock architecture                         | Excluded                                              | Thesis correct                                                                                 |

**Unresolved internal contradiction:** thesis §3.3.1 states "aggregated sentiment scores from financial news were used", while Table 4's note says textual NLP sentiment was "explicitly excluded". The implementation excludes it. §3.3.1 must be corrected.

---

## 3. Existing Implementation Reconstruction

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
| Walk-forward         | "Training set to grow over time" (§3.7.1)           | **Rolling** window — train start advances, early events dropped | **P1** | Expanding option added; both reported           |
| Y3 unit              | Consecutive trading days                             | Calendar days                                                          | **P1** | Fixed to positional trading-day distance        |
| Macro alignment      | "as-of-date alignment... mandatory" (§3.3.1 spirit) | Annual value dated 1 January of its own year                           | **P0** | Re-dated to Y+1-07-01                           |
| Ridge                | Baseline model                                       | Fit on**unstandardised** features, alpha never tuned             | **P0** | `StandardScaler` + in-fold `RidgeCV`        |
| Stationarity         | ADF/KPSS, difference failures (§3.5.1)              | Tested`log_return` only; 6 price-level features untested             | **P0** | All features tested; unit-root columns excluded |
| Missing targets      | 100% real data claim                                 | `y.fillna(0.0)` fabricated 3 Y2 values                               | **P0** | NaN preserved, masked per target                |
| Affected threshold   | `>= 1000`                                          | `> 1000`                                                             | P3           | Fixed                                           |
| Overlapping events   | Truncation protocol (§3.3.2)                        | `truncate_overlapping_windows()` defined, **never called**     | **P1** | Contamination now quantified and disclosed      |
| SVR                  | Listed as model (§3.6.1)                            | Not implemented                                                        | P2           | Either implement or remove from methodology     |
| MultiOutputRegressor | Described (§3.6.4)                                  | Not used — separate per-target fits                                   | P2           | Correct the description (see §18)              |

---

## 4. Methodology vs Code Audit

Covered in the table above. The single most damaging mismatch is the **inner k-fold**: §8 of the notebook and §3.7.1 of the thesis both state, in bold, that no k-fold is used anywhere, while two lines of code ran `KFold(shuffle=False)` inside every walk-forward fold — training on later events to select hyperparameters for earlier ones. This was compounded because SMOGN appends synthetic rows at the tail of the training frame, so the last inner validation block was disproportionately synthetic: the model was partly selected on its ability to predict its own interpolated output.

---

## 5. Unit-of-Analysis Audit

**One training sample = one disaster event.** Not one trading day.

- Raw EM-DAT records: 86
- After excluding biological and applying `affected >= 1000`: 74
- After restricting to events inside real archive coverage: **64**
- Effective N for Y2: **61** (3 events have no volume data — the 2000 workbook failed to parse)

There is no pseudo-replication: each event contributes exactly one row. The daily market series is used only to *construct* event-level features and targets, never as independent observations. The thesis is correct not to claim thousands of samples.

**Consequence for model complexity:** with 64 events and a 30/10/10 geometry, each fold trains on 30 rows. Against 32 features this is p > n before selection. This single fact should govern every capacity decision in the study.

**Residual dependence:** events are not fully independent. Y3 windows look forward up to 90 days, and Sri Lanka's monsoon clustering means some windows contain a subsequent qualifying event. This is quantified in the notebook rather than assumed away.

---

## 6. Target Audit

### Y1 — ASPI market impact

**Decision: KEEP, rename.**

Implemented as the raw continuously-compounded log return, which matches thesis §3.2.2/Table 4. The thesis's prose label "ASPI Percentage Change" is inaccurate: a log return is not a percentage change (they diverge as magnitude grows). Rename to **"ASPI log return"** throughout.

Should it be an abnormal return (AR) instead? **No.** §2.2.3's market-model AR/CAR formulation is presented in the thesis as literature-review background that the thesis explicitly critiques and moves away from. More decisively: computing AR requires a market model `E(R_i,t) = α + β·R_m,t`, which needs a *market* index distinct from the asset. Here the asset **is** the market index. There is no valid benchmark to regress against, so AR is not identified. The raw log return is the correct operationalisation.

### Y2 — Abnormal trading volume

**Decision: KEEP formula, correct the terminology.**

Implemented as `V_t / V̄_pre − 1`, which is the standard normalised abnormal-volume form and is preferable to the raw ratio (it is centred at zero, so "no abnormality" is 0).

**The name is wrong.** The thesis calls this "trading volume crash magnitude". The variable measures *deviation in either direction*, and its observed maximum is **+1.94** — nearly triple baseline volume. A large positive value is a volume **spike**, which the thesis's own literature review (§2.3.2) correctly identifies as the signature of panic selling. Calling it a "crash" inverts the meaning. Rename to **"abnormal trading volume"** and describe it as a liquidity-disturbance magnitude, not a crash.

### Y3 — Market recovery time

**Decision: MODIFY.**

Three distinct defects:

1. **Unit deviation (fixed).** The search window was 91 trading rows but the returned value was a calendar-day difference. The 90-day cap therefore bit at roughly 62 trading days, and an event recovering after that was indistinguishable from one that never recovered.
2. **Right-censoring is real and currently ignored.** For a non-recovering event we know `T > 90`, not `T = 90`. Assigning 90 is a biased point estimate.
3. **Y3 is not an independent target.** The recovery window includes the event day itself and the baseline is `P_{t-1}`, so whenever `P_t >= P_{t-1}` — that is, whenever **Y1 >= 0** — the event day satisfies the recovery condition and **Y3 = 0 exactly**. `Y3 = 0` and `Y1 >= 0` are the same event by construction. This is why Y3's median is 0 and its 25th percentile is 0.

**Censoring treatment — recommendation:** Approach C (two-stage hurdle), not Approach D (survival).

- Approach A (regression with cap) — current, biased.
- Approach B (recovered events only) — discards the most severe events. Rejected.
- **Approach C (hurdle): P(recovery within 90) × E[duration | recovered].** Matches the actual distribution (a point mass at 0, a right tail, a point mass at the cap). Stage 1 is a classifier, which at N=64 is far better supported than a survival model.
- Approach D (Cox / AFT / Random Survival Forest) — methodologically the "textbook" answer for censoring, but at 64 events with a censoring point that is itself an artefact of the 90-day design choice, a survival forest would be fitting hazard curves to a handful of tail points. **Rejected on sample-size grounds, per the master prompt's own caution.**

Because of the Y1↔Y3 identity, Stage 1 of the hurdle model is *almost* a restatement of predicting the sign of Y1. That must be stated openly rather than presented as a second independent finding.

---

## 7. Information-Availability Audit

This is the most consequential section of the audit.

| Feature                                | Available at disaster onset? | Publication delay | Leakage risk       | Keep/Remove              |
| -------------------------------------- | ---------------------------- | ----------------- | ------------------ | ------------------------ |
| `financial_damage`                   | **No**                 | Weeks–months     | **CRITICAL** | Remove for ex-ante model |
| `log_financial_damage`               | **No**                 | Weeks–months     | **CRITICAL** | Remove for ex-ante       |
| `population_affected`                | **No** (final figure)  | Days–weeks       | **CRITICAL** | Remove for ex-ante       |
| `log_population_affected`            | **No**                 | Days–weeks       | **CRITICAL** | Remove for ex-ante       |
| `damage_to_gdp`                      | **No** (both parts)    | Months            | **CRITICAL** | Remove for ex-ante       |
| `log_damage_x_flood`                 | **No** (damage half)   | Weeks–months     | **CRITICAL** | Remove for ex-ante       |
| `disaster_Flood/Storm/...`           | Yes                          | None              | None               | Keep                     |
| `days_since_last_disaster`           | Yes                          | None              | None               | Keep                     |
| `disasters_trailing_365d`            | Yes                          | None              | None               | Keep                     |
| `log_return`, `lag_return_t-*`     | Yes (t−1 close)             | None              | None               | Keep                     |
| `sma_*`, `ema_*`                   | Yes but non-stationary       | None              | Extrapolation      | Replaced by ratios       |
| `price_to_sma_*`, `price_to_ema_*` | Yes                          | None              | None               | Keep                     |
| `rolling_std_*`, `squared_return`  | Yes                          | None              | None               | Keep                     |
| `gdp_growth_pct`                     | **No** as dated        | ~6–18 months     | **HIGH**     | Re-date (done)           |
| `inflation_cpi_pct`                  | **No** as dated        | ~6–18 months     | **HIGH**     | Re-date (done)           |
| `gdp_current_usd`                    | **No** as dated        | ~6–18 months     | **HIGH**     | Re-date (done)           |
| `sp500_log_return`                   | Yes                          | None              | None               | Keep                     |

**Verdict: the thesis currently supports retrospective (ex-post) prediction, not real-time early warning.**

Six of 32 features are EM-DAT post-hoc assessments. On the day before a flood, its eventual total damage and total affected are unknown. The README, thesis §1.2.3 and §4.4 all describe an "early-warning tool"; that claim is **NOT SUPPORTED** by the current feature set.

**Recommendation — two models, explicitly separated:**

- **Model A (ex-ante / event-onset):** disaster type, season, trailing disaster count, days since last disaster, historical type-average severity, all market features, macro with correct publication lag. Answers *"a flood has just begun — what happens?"*
- **Model B (ex-post / attribution):** the current feature set including realised damage. Answers *"given a disaster of this measured severity, what was the market response?"*

Model B is the study as it stands and is a legitimate research object. Model A is the one that would justify the early-warning language. Reporting both, and labelling which is which, converts a fatal framing problem into a genuine two-experiment contribution.

---

## 8. Data-Leakage Audit

| Source                                                | Type                      | Severity           | Status                                                                                                                   |
| ----------------------------------------------------- | ------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Global SMOGN before fold split                        | Temporal                  | CRITICAL           | Fixed earlier in development (documented in §8 of the notebook)                                                         |
| Walk-forward boundaries computed on post-SMOGN length | Temporal                  | CRITICAL           | Fixed — folds now cut on real`len(X)`                                                                                 |
| `GridSearchCV(cv=3)` = KFold inside each fold       | Temporal                  | **CRITICAL** | Fixed —`TimeSeriesSplit` on real rows only                                                                            |
| Hyperparameters selected partly on synthetic rows     | Target                    | HIGH               | Fixed — selection on real rows, refit on augmented                                                                      |
| Annual macro dated 1 January of its own year          | Availability              | **CRITICAL** | Fixed — re-dated Y+1-07-01                                                                                              |
| Ex-post damage features as onset predictors           | Availability              | **CRITICAL** | Disclosed; requires reframing or Model A                                                                                 |
| SMA/EMA computed on unshifted price                   | Temporal                  | CRITICAL           | Fixed earlier (`.shift(1)`)                                                                                            |
| Feature selection inside fold on training rows        | —                        | NONE               | Correct as implemented                                                                                                   |
| Scalers fit inside fold                               | —                        | NONE               | Correct as implemented                                                                                                   |
| ADF/KPSS run on the full series                       | Temporal                  | LOW                | Stationarity is a structural property, not a fitted parameter; the test does not transfer target information. Disclosed. |
| SMOGN minority quantile thresholds                    | —                        | NONE               | Computed on training rows only                                                                                           |
| Y3 window overlapping a later event                   | Target                    | MODERATE           | Quantified and disclosed; truncation function still uncalled                                                             |
| Model/config selection on reported folds              | **Optimistic bias** | HIGH               | Disclosed in a dedicated subsection (see §29)                                                                           |

**On the last row — this is the one that cannot be fixed by code.** The SMOGN configuration was reverted after its held-out scores were seen. K, the split geometry, and the primary-model choice were all evaluated against the same folds that are reported. This is disclosed explicitly rather than concealed, and the pre-registered 25% synthetic-share cap now provides a score-independent justification for the SMOGN revert.

---

## 9. Event-Window and Overlapping-Event Audit

**30-day pre / 90-day post: KEEP.** Justified a priori by the thesis's own argument (delayed price discovery in a semi-strong-inefficient frontier market) and, critically, **not tuned against test performance**. Systematically searching [-5,+20], [-10,+30], [-20,+60] and picking the best would be window-shopping on the test set. The current single pre-registered window is the more defensible choice, and this audit recommends against the window sweep the master prompt offers as an option.

The three roles of the pre-event window must be separated in the write-up, because they currently blur:

- **Baseline window** — the 30-day volume mean that defines Y2, and `P_{t-1}` that defines Y1 and Y3.
- **Feature window** — the lags, moving averages and volatility measures, all halted at t−1.
- **Estimation window** — *does not exist here*, because no market model is estimated (see §6, Y1). The thesis should stop calling it an estimation window.

**Overlapping events — recommendation: Option C + D over Option A.**

`truncate_overlapping_windows()` exists in `preprocessor.py` and is never called. Rather than wire in Option A (truncation), which shortens Y3 for exactly the events most likely to be severe and introduces a second censoring mechanism on top of the 90-day cap, the better treatment at this N is:

- **Option C:** add a `concurrent_disaster_in_window` indicator (already partly available via `disasters_trailing_365d`).
- **Option D:** the existing `disasters_trailing_365d` already encodes cumulative intensity.

Option B (exclude overlapping events) would drop a substantial fraction of a 64-event sample. Option E (event clusters) would reduce N further. Both rejected on sample-size grounds.

---

## 10. Time-Aware SMOGN Audit

### What is actually imbalanced?

Y3 has 10 minority events (`Y3 > 30`) out of 64 — **15.6%**. Y1 and Y2 are roughly symmetric and have no rare-event structure in the same sense.

### Is the implementation genuinely "time-aware"? — Yes, with corrections.

| Property                                              | Status                                                                                                                                                                                                                                                                                                                   |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Neighbours restricted temporally                      | **Yes** — ±5 year window, enforced                                                                                                                                                                                                                                                                               |
| Future disasters can seed historical synthetic events | **Yes within a fold's training rows** — the window is symmetric, not backward-only. Since all rows are training rows this does not leak into test, but it does mean a 2010 event can be interpolated with a 2013 event. Defensible (both are past data at prediction time) and disclosed.                         |
| Runs after fold creation, training rows only          | **Yes** — verified                                                                                                                                                                                                                                                                                                |
| Disaster categories mixed unrealistically             | **Was yes — now fixed.** Previously a Flood could interpolate with a Drought, producing rows like `disaster_Flood=0.6, disaster_Storm=0.4`. Now same-type only, with the one-hot block copied verbatim.                                                                                                         |
| Synthetic damage/population combinations plausible    | **Was no — now fixed.** Derived features (`log_financial_damage`, `damage_to_gdp`, `squared_return`, `log_damage_x_flood`) were interpolated *independently of their parents*, so `log_financial_damage != log1p(financial_damage)` on every synthetic row. Now recomputed from interpolated parents. |
| Categorical types preserved                           | **Now yes**                                                                                                                                                                                                                                                                                                        |
| Synthetic rows violate real financial relationships   | **Partly, unavoidably.** `rolling_std_*` is a convex functional of the price path, so interpolating two rows over-states volatility. Documented as an unfixable limitation.                                                                                                                                      |
| Gaussian-noise level                                  | **Branch deleted.** It applied `N(0, 0.01)` *absolute* noise to columns measured in millions (producing near-duplicate rows) while that same 0.01 is ~70% of Y1's standard deviation — capable of flipping Y1's sign while leaving Y3 at 90, a combination that cannot occur given `Y3 = 0 ⟺ Y1 >= 0`.     |
| Relevance function φ(y)                              | **Not implemented — deliberately.** Canonical SMOGN's φ is a PCHIP curve through boxplot control points. Y3's median and 25th percentile are both 0, so the boxplot construction is degenerate. The binary `Y3 > 30` threshold is the step-function instantiation of φ and should be described as such.       |
| Majority under-sampling                               | **Not implemented — deliberately.** At 30 training rows, discarding real observations to balance a ratio is the most expensive available action and contradicts the study's 100%-real-data claim. This is a supported configuration of canonical SMOGN (`under_samp=False`), not a deviation.                   |

### Verdict

**KEEP + OPTIMIZE.** The label "Time-Aware SMOGN" is now honest. Prior to these corrections it was SmoteR-with-a-cutoff producing physically impossible events.

---

## 11. Preprocessing Audit

**Scaling — MODIFY (done).** The thesis says scaling is applied within each training fold, which is leakage-safe and correct. But scaling was applied *inconsistently across models*: the MLP scaled X and y, while **Ridge received unstandardised features**. Since L2 is not scale-equivariant and the design matrix mixed ASPI price levels (~1e4), raw USD damage (~1e9) and 0/1 flags, `alpha=1.0` applied almost no shrinkage to the damage columns and crushed the binary ones. Random Forest and XGBoost correctly need no scaling. Model-specific pipelines are now used.

**Stationarity — MODIFY (done).** The thesis (§3.5.1) mandates ADF/KPSS with differencing of failures. The implementation tested `log_return` alone and reported "stationary", while six engineered price-level columns went untested. All engineered market features are now tested.

The exclusion rule matters and this audit corrects an error made during remediation: **gate on the ADF unit-root result only.** The volatility columns (`rolling_std_*`, `squared_return`) reject the ADF unit-root null decisively (p ≈ 1e-11 and smaller) while failing KPSS level-stationarity — the ordinary signature of persistent, mean-reverting volatility clustering, not of a trending level. Requiring both tests to agree discarded 11 of 22 features including the thesis's own mandated 30-day panic proxy. A unit root is what creates extrapolation risk under a chronological split; KPSS-persistence does not.

**Imputation — MODIFY (done).** `y = dataset[TARGET_COLS].fillna(0.0)` fabricated three Y2 observations, asserting "volume exactly at its 30-day baseline" for events whose source workbook failed to parse. This is the same zero-fill the notebook's §3 explicitly rejects for volume itself. Targets now keep their NaNs and are masked per target; Y2's effective N is reported as 61.

`X = dataset[FEATURE_COLS].fillna(0.0)` remains and is a weaker but real concern: 0.0 in `damage_to_gdp` reads as "no damage" rather than "unknown". Recommend an explicit missingness indicator.

---

## 12. Feature Engineering Audit

**Collinearity is severe and largely unaddressed.** `sma_5/10/20` and `ema_5/10/20` are six smoothed versions of the same price series and are mutually ~99% correlated; `financial_damage`, `log_financial_damage`, `damage_to_gdp` and `log_damage_x_flood` are four transforms of one quantity. This is why K=20 → K=10 barely moved Random Forest (−0.322 → −0.298) but moved Ridge substantially (−1.634 → −0.803): the binding problem was collinearity and non-stationarity, not the feature count.

**A concrete consequence found in the SMOGN neighbour search:** the standardised distance metric included `financial_damage` *and* its three derived children, so four of 32 columns were the same underlying quantity and neighbour selection was dominated by damage similarity. Derived columns are now excluded from the distance metric.

**What the damage features actually measure.** EM-DAT records no damage estimate for the large majority of Sri Lankan events, and the loader zero-fills those. The damage features therefore behave substantially as an **indicator of EM-DAT reporting coverage**, which is itself correlated with severity and recency. The `log_damage_x_flood` interaction was justified on the grounds that Flood is 51/74 of the qualifying set — but the relevant denominator is *floods that have a damage figure*, which is far smaller. This must be stated in the limitations.

**Recommended additions — none.** With 64 events, the correct direction is fewer features, not more. Deaths, injured, homeless, duration and geographic spread are all available in EM-DAT but every one of them is (a) ex-post and (b) collinear with the severity measures already present. Adding them would worsen both the p/n ratio and the information-availability problem.

**Recommended removals:** the six raw price-level moving averages (done — replaced by stationary `price_to_sma_*` / `price_to_ema_*` ratios).

---

## 13. Baseline Evaluation

**This was the single largest reporting gap and it is now closed.**

Before this audit the study compared machine-learning models against Ridge and against nothing else. Two deployable nulls have been added, computed on exactly the same folds:

- `naive_zero` — predict 0 for every event. The **economic** null: "the disaster had no measurable effect". All three targets are defined so that 0 is the meaningful no-effect value.
- `naive_train_mean` — predict the training fold's mean (never the test mean). The **statistical** null that makes R² interpretable.

**Measured result (Phase D run, calendar-day Y3):**

| Target | `naive_zero` RMSE | Models beating it                                                     |
| ------ | ------------------- | --------------------------------------------------------------------- |
| Y1     | 0.0084              | **0 of 6**                                                      |
| Y2     | 0.5142              | 3 of 6 (ridge, mlp, ensemble)                                         |
| Y3     | 30.17               | 6 of 6 —**but `naive_train_mean` (27.95) beats every model** |

**This corrects an earlier claim.** A full-sample approximation had suggested every model beat the no-effect null on Y1. Computed fold-wise on the actual test folds, that is false. The honest statement is: **no model beats a naive baseline on Y1; only Ridge clears both nulls on Y2; on Y3 no model beats the training-mean predictor.**

---

## 14–17. Model Evaluations

### 14. Random Forest — KEEP + OPTIMIZE

Best or near-best across targets in early configurations, and the most stable. But: train-fit R² of **0.883** on Y3 against a held-out **−0.146** is a 1.03-point gap — the model is memorising. Grids permitted `max_depth=None` with `min_samples_leaf=1` on 33-row folds, i.e. trees isolating individual events. Recommend bounding capacity a priori by fold size (`min_samples_leaf >= 3`, `max_depth <= 4`) rather than by test score.

### 15. XGBoost — KEEP + OPTIMIZE

Was the worst-behaved model before the stationarity fix (Y1 pooled R² −1.617), and improved most from it (−0.236). That pattern is diagnostic: boosting was extrapolating hardest off the end of the training price range. Same capacity-bounding recommendation.

### 16. SVR — **REPLACE (with nothing)**

Specified in thesis §3.6.1 and **never implemented**. Either implement it or delete it from the methodology. Given that Ridge (linear), RF and XGBoost (trees) and MLP (network) already span the model-family space, and that SVR adds a kernel and two hyperparameters to tune on 30 rows, this audit recommends **removing it from the thesis** rather than adding it to the code. State the removal and the reason.

### 17. MLP — KEEP + OPTIMIZE

Architecture (one hidden layer, 64 units, dropout 0.5, three heads) is appropriately constrained for the sample size. Correctly rejects LSTM/Transformer.

Two real issues:

1. **No early stopping, no validation split.** 200 fixed full-batch epochs. On 33 rows this is a capacity risk, though dropout 0.5 mitigates it.
2. **The loss-weight rationale is now wrong.** The notebook states the weights `(1.0, 0.1, 0.5)` are the thesis's eq.(4) mechanism for the scale gap. Since y is standardised per fold *before* the loss is applied, all three per-target MSE terms are already unit-variance and the ~1000× raw-scale gap is gone before the weights act. The weights now express **task priority** (Y1 primary), not scale equalisation. This is a defensible position — but it must be described accurately. Do not re-tune them; that would be an untuned hyperparameter change with no a-priori basis.

**Y3 transform:** `log1p` is now applied to Y3 inside the MLP before the y-scaler, matching the tree/linear models. This is not redundant with standardisation: standardisation is affine and fixes *scale*; log1p is monotone-nonlinear and fixes *shape*. Y3 after standardisation is still majority-zero, right-skewed and hard-capped.

---

## 18. Multi-Output vs Multi-Task Learning

**The thesis's description is wrong and must be corrected.**

Thesis §3.6.4 states that a `MultiOutputRegressor` wrapper is used for the tree models. The code does not use `MultiOutputRegressor` at all — it fits an entirely separate estimator per target inside the fold loop, each with its own feature selection and its own hyperparameter search.

More importantly, **neither is multi-task learning.** Fitting `f1(X)→Y1`, `f2(X)→Y2`, `f3(X)→Y3` independently is *multi-output prediction*. There is no shared representation and no cross-target information transfer. §12's cross-check table also claims a "custom weighted-RMSE scorer averaged across all 3 targets (thesis eq.(4) weights)" — the code uses `neg_root_mean_squared_error` on one target at a time.

**Only the MLP is genuinely multi-task**, via its shared hidden layer.

Required wording change: describe the tree pipeline as **per-target independent models**, and reserve "multi-task learning" for the MLP alone.

**Is joint learning justified?** The MLP is not the best model on any target in the measured results. There is no evidence of positive transfer. State this plainly rather than retaining MTL because it was proposed.

---

## 19. Recovery Survival-Analysis Decision

Covered in §6. **Recommendation: two-stage hurdle (Approach C), not survival (Approach D).** Rationale: N=64 with ~10 tail events cannot support hazard estimation; the censoring point is a design artefact rather than a natural end-of-observation; and the hurdle's first stage matches the actual point-mass-at-zero structure. Disclose that stage 1 is near-equivalent to predicting `sign(Y1)`.

---

## 20. Validation Redesign

**Recommended final design:**

1. **Outer:** chronological walk-forward, **expanding** window (thesis §3.7.1 says "the training set to grow over time"; the code rolls). Test indices are identical under both, so the comparison is exactly like-for-like — nothing is added to or removed from the evaluation set.
2. **Inner:** `TimeSeriesSplit` on the fold's **real** training rows for all hyperparameter selection; refit the winner on real + synthetic.
3. **Report both** the primary 30/10/10 and the dense 20/5/5 geometry at equal prominence.

**Leave-One-Disaster-Out:** rejected. It breaks chronology (training on later events to predict earlier ones) — the exact failure the thesis bans.

**Leave-One-Disaster-Type-Out:** worth reporting **descriptively** as a stress test, not as a statistical claim. With Drought n=5 and the singletons pooled, per-type inference is not supportable.

**Embargo/purging:** the honest position is that Y3's 90-day forward window can overlap a subsequent event that falls in a later fold. Purging would cost folds the study cannot spare. Recommend **disclosing** the overlap count rather than implementing an embargo — and stating that as a limitation.

---

## 21. Hyperparameter Optimization

**KEEP + OPTIMIZE.** Grid search is correct here; the master prompt's alternatives (Optuna/TPE, Bayesian) would search a *larger* space on 30 rows, which increases selection overfitting. The thesis's own §3.7.2 justification for tuning ("using default values will result in severe overfitting") is sound but was applied inconsistently — RF and XGBoost were tuned while Ridge sat at its library default `alpha=1.0`. Now fixed.

Grids are deliberately small (8 combinations each) because a larger grid made the search itself the bottleneck (>900 s per fold). This is a real, disclosed constraint.

**Nested selection:** full nested CV is not supportable at this N. The implemented compromise — inner `TimeSeriesSplit` on real training rows, outer walk-forward for reporting — is the most defensible available, and its limitation (inner folds of 9/16/23 rows are genuinely noisy) is stated. The correct response to noisy inner selection is to *shrink the search space* so that any pick is acceptable, not to search harder.

---

## 22. Metric Framework

| Target | Primary                      | Secondary                                                                                  | Baseline it must beat                       |
| ------ | ---------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------- |
| Y1     | RMSE                         | MAE, pooled R², directional accuracy**vs majority baseline**, AUC **with CI** | `naive_zero`, `naive_train_mean`, Ridge |
| Y2     | RMSE                         | MAE, pooled R²                                                                            | `naive_zero`, `naive_train_mean`, Ridge |
| Y3     | MAE**in trading days** | RMSE, median AE, pooled R²                                                                | `naive_zero`, `naive_train_mean`, Ridge |

**Pooled R² is the correct R² to report** (concatenate all out-of-fold predictions, compute once) — per-fold R² on a 10-point window is numerically unstable. Both are shown, with pooled designated primary.

**Two definitional corrections required in the thesis:**

- §3.8.1 states "RMSE measures the total variation in the actual data explained by the ML algorithm." **This is wrong.** RMSE = √(mean squared error); it is an error magnitude in the target's units. R² is the explained-variance measure. Correct the sentence.
- The claim that pooled R² is computed over "N≈64-74 real points" is wrong: it is computed over **30** pooled out-of-fold test points (20 for the stacked model). Corrected in the notebook.

---

## 23. Rare-Event Evaluation

**Not yet measured.** The pipeline does not currently report performance separately for severe versus ordinary events. This is a genuine gap and is listed in the next-experiments section. The required table:

| Target | Overall | Common events | Rare events (top decile severity) | Worst event |
| ------ | ------- | ------------- | --------------------------------- | ----------- |

Given that the study exists to predict catastrophic impacts, a model with acceptable overall RMSE and poor performance on the 2004 tsunami and the 2005-11-21 event would not meet the research objective. This must be measured before any performance claim is made.

---

## 24. Ablation Study

**Measured — the stationarity ablation (Phase C → Phase D), pooled R²:**

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

**Measured — the SMOGN over-augmentation ablation:** widening the minority mask to the union of Y1/Y2/Y3 quintile tails with 2 draws per row produced 26–28 synthetic rows against 30 real per fold and made **every model on every target worse** (RF Y1 pooled R² −0.32 → −1.39; Ridge Y3 RMSE 36.6 → 62.9). Reverted. This variant is also rejected independently by the pre-registered 25% synthetic-share cap (it reaches 46–48%), so the revert does not rest on having seen its scores.

**Measured — feature count (K=20 vs K=10), pooled R²:** K=10 better on 6 of 9 model-target cells (ridge Y1 −1.634 → −0.803; ridge Y2 −3.539 → −1.250; RF Y2 −0.120 → −0.057) but **destroys the study's only positive R²** (ridge Y3 +0.038 → −0.079). K=20 retained; both reported.

**Not yet measured — the study's central ablation.** Market-only vs disaster-only vs market+disaster vs +macro. This directly answers *"do disaster variables add predictive information beyond ordinary market history?"* — the question the entire thesis exists to answer — and it has never been run. **This is the single most important missing experiment.**

**Not yet measured:** SMOGN on/off. The augmentation's worth has never been isolated.

---

## 25. Robustness Analysis

Measured: split geometry (30/10/10 vs 20/5/5), feature count (20 vs 10), augmentation ratio (1× vs 2× with widened mask), stationarity treatment, Y3 response transform (raw vs log1p).

Not measured: random seed sensitivity, recovery-definition sensitivity (exact vs 1–2% tolerance vs stable-for-k-sessions), removal of the two extreme events (2004-12-26, 2005-11-21).

**Note on split sensitivity — a result that must be reported.** The dense 20/5/5 configuration produces the best number anywhere in the study (RF Y1 pooled R² **+0.151** against −0.322 primary) *and* the worst (Ridge Y3 pooled R² −229 before prediction clipping). Both directions are the same finding: **at this N, results are highly sensitive to split geometry.** That is itself the most important robustness conclusion.

---

## 26. Error Analysis

**Why pooled R² is negative — the quantitative explanation.**

R² = 1 − SSE/SST, where SST is computed against the *pooled test* mean. But every model can only ever centre on its *training-fold* mean. The chronological split compounds this: the two largest shocks in the dataset — **2004-12-26 (Earthquake/tsunami, Y1 = −0.0443)** and **2005-11-21 (Flood, Y1 = −0.0753)** — fall permanently inside fold 0's **training** window. The held-out folds therefore contain the calmer events and carry only a fraction of full-sample variance:

| Target | Full-sample σ | σ of pooled test folds | Ratio |
| ------ | -------------- | ----------------------- | ----- |
| Y1     | 0.01403        | 0.00880                 | 0.63  |
| Y2     | 0.5799         | 0.4954                  | 0.85  |
| Y3     | 30.64          | 28.54                   | 0.93  |

A negative pooled R² here means "worse than an oracle that already knows the test set's mean" — which is **not** the same as "worse than a usable baseline". The `beats_null_rmse` column answers the second, more meaningful question. This explanation is now in the notebook, along with the `n` and `sigma_y_test` columns that let a reader verify it.

**A caution on the 2005-11-21 event:** Sri Lanka's presidential election was 17 November 2005, four days before the largest ASPI drop in the dataset. There is **no event-window contamination screen anywhere in the pipeline** — no filter for concurrent elections, policy announcements or macro shocks. This is the core identification threat in any event study and the thesis does not address it.

---

## 27. Explainability / SHAP

Implemented: global summary plot and a local waterfall for the largest real ASPI drop, on real data only.

**Two requirements:**

1. **SHAP under correlated predictors is unstable.** With six ~99%-correlated moving averages and four damage transforms, attribution splits arbitrarily among collinear columns. Any SHAP ranking must be reported as *model attribution under collinearity*, not as a stable importance ordering. Recommend re-running SHAP after the collinearity reduction and reporting whether the ranking changed.
2. **SHAP is not causal.** The thesis must not describe high-SHAP features as drivers of market impact. Replace any causal phrasing with "predictive association".

---

## 28. Ten-Expert Initial Scores

Scoring the framework **as it stood before this audit's remediation**, 0–10 per criterion, 12 criteria, /120.

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

## 29. Critical Problems Ranked P0–P3

### P0 — Could invalidate results

| #    | Problem                                                                           | Status                                                       |
| ---- | --------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| P0-1 | `GridSearchCV(cv=3)` = k-fold inside walk-forward, contradicting §3.7.1        | **Fixed**                                              |
| P0-2 | Ex-post damage features used as onset predictors; early-warning claim unsupported | **Disclosed; requires Model A or permanent reframing** |
| P0-3 | Annual macro dated 1 Jan of its own year → 6–18 month look-ahead on 4 features  | **Fixed**                                              |
| P0-4 | Ridge (the H1 baseline) fit unstandardised → H1 comparison meaningless           | **Fixed**                                              |
| P0-5 | `y.fillna(0.0)` fabricated 3 Y2 observations                                    | **Fixed**                                              |
| P0-6 | Non-stationary price levels in a chronological split                              | **Fixed**                                              |
| P0-7 | Model/config choices made on the reported folds                                   | **Disclosed — cannot be undone**                      |

### P1 — Major methodological improvement

| #    | Problem                                                                                     | Status                                             |
| ---- | ------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| P1-1 | Y3 in calendar days while thesis specifies trading days                                     | **Fixed**                                    |
| P1-2 | Y3 right-censoring ignored (cap treated as observed)                                        | **Open — hurdle model recommended**         |
| P1-3 | `Y3 = 0 ⟺ Y1 >= 0` by construction; targets not independent                              | **Documented**                               |
| P1-4 | Directional accuracy reported without its majority baseline                                 | **Fixed**                                    |
| P1-5 | No naive baselines                                                                          | **Fixed**                                    |
| P1-6 | Rolling rather than expanding walk-forward, contradicting §3.7.1                           | **Option added; both reported**              |
| P1-7 | Central market-vs-disaster ablation never run                                               | **Open — highest-value missing experiment** |
| P1-8 | `truncate_overlapping_windows()` defined and never called while SWOT implies it is in use | **Quantified and disclosed**                 |
| P1-9 | Predictions violate definitional bounds (Ridge Y3 RMSE 157 on dense config)                 | **Fixed by clipping**                        |

### P2 — Performance / robustness

P2-1 SVR specified but never implemented · P2-2 RF/XGB capacity unbounded relative to fold size · P2-3 Severe collinearity unaddressed · P2-4 SHAP instability under collinearity unacknowledged · P2-5 Rare-event performance never measured separately · P2-6 SMOGN on/off never ablated · P2-7 `X.fillna(0.0)` conflates "zero damage" with "unknown" · P2-8 Live unpinned API pulls make Table 10 non-reproducible

### P3 — Optional

P3-1 `> 1000` vs `>= 1000` (fixed) · P3-2 Singleton disaster types as memorisation keys (fixed by pooling) · P3-3 `forward_fill_missing_values` also ffills provenance columns · P3-4 EM-DAT missing month/day filled to 1 January, count unreported

---

## 30. KEEP / OPTIMIZE / MODIFY / REPLACE

| Component                      | Decision                    | Reason                                                                                  |
| ------------------------------ | --------------------------- | --------------------------------------------------------------------------------------- |
| 30/90 event window             | **KEEP**              | Justified a priori by frontier-market price discovery; not tuned on test                |
| ASPI log-return target         | **KEEP** (rename)     | Correct operationalisation; AR not identified for an index. Label is wrong              |
| ATV target                     | **KEEP** (rename)     | `V/V̄ − 1` is the right form; "volume crash" inverts its meaning                    |
| 90-day recovery cap            | **MODIFY**            | Right-censoring must be modelled, not assigned. Hurdle model                            |
| `>= 1000` affected threshold | **KEEP**              | Thesis-mandated; code now matches                                                       |
| Lag structure (t−1,2,3,5)     | **KEEP**              | Stationary, available at prediction time, low-dimensional                               |
| SMA/EMA                        | **REPLACE**           | Non-stationary levels → stationary price-relative ratios                               |
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
| MTL MLP                        | **KEEP** (reframe)    | Only genuine MTL component; no measured positive transfer — say so                     |
| Weighted MSE                   | **KEEP** (reframe)    | Weights now express task priority, not scale equalisation                               |
| Walk-forward CV                | **KEEP + OPTIMIZE**   | Correct choice; make expanding, fix inner CV                                            |
| Grid/Bayesian tuning           | **KEEP + OPTIMIZE**   | Grid is right at this N; must run on real rows chronologically                          |
| RMSE/MAE/R²                   | **KEEP + OPTIMIZE**   | Correct core; add naive baselines, CIs, and per-target secondary metrics                |
| SHAP                           | **KEEP + OPTIMIZE**   | Correct tool; must acknowledge collinearity instability and drop causal language        |

---

## 31–33. Recommended Target-Specific Pipelines

### Pipeline A — Y1 (ASPI log return)

```
Ex-ante features (Model A) OR ex-post features (Model B) — labelled, never mixed silently
   -> per-fold: SMOGN (train rows only, same-type, derived recomputed)
   -> per-fold: RF-importance top-K feature selection (train rows only)
   -> StandardScaler (Ridge/MLP only; not for trees)
   -> Ridge(alpha via in-fold TimeSeriesSplit RidgeCV) | RF | XGBoost | MLP
   -> expanding walk-forward, outer
   -> RMSE (primary), MAE, pooled R², directional accuracy vs majority baseline, AUC + CI
   -> vs naive_zero, naive_train_mean, Ridge
   -> SHAP global + local
```

### Pipeline B — Y2 (abnormal trading volume)

Identical, with two differences: **N = 61** (three events have no volume data — reported, not imputed), and predictions clipped to `>= −1` (volume cannot be negative).

### Pipeline C — Y3 (recovery, trading days)

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

## 34. Experiment Matrix

| ID            | Research question                          | Features           | Rare treatment   | Model                         | Validation                                 | Target             |
| ------------- | ------------------------------------------ | ------------------ | ---------------- | ----------------------------- | ------------------------------------------ | ------------------ |
| E01           | Economic null                              | —                 | None             | predict 0                     | Walk-forward                               | All 3              |
| E02           | Statistical null                           | —                 | None             | train mean                    | Walk-forward                               | All 3              |
| E03           | Linear baseline                            | Market             | None             | Ridge (scaled, tuned)         | Walk-forward                               | All 3              |
| **E04** | **Do disaster variables add value?** | Market only        | None             | RF                            | Walk-forward                               | All 3              |
| **E05** | **Do disaster variables add value?** | Disaster only      | None             | RF                            | Walk-forward                               | All 3              |
| **E06** | **Do disaster variables add value?** | Market + disaster  | None             | RF                            | Walk-forward                               | All 3              |
| E07           | Do macro controls add value?               | + macro (lagged)   | None             | RF                            | Walk-forward                               | All 3              |
| E08           | Does SMOGN add value?                      | Full               | **None**   | RF                            | Walk-forward                               | All 3              |
| E09           | Does SMOGN add value?                      | Full               | SMOGN            | RF                            | Walk-forward                               | All 3              |
| E10           | Does sample weighting beat resampling?     | Full               | Sample weights   | RF                            | Walk-forward                               | All 3              |
| E11           | Which model class wins?                    | Full               | Best of E08–E10 | Ridge/RF/XGB/MLP              | Walk-forward                               | All 3              |
| E12           | Does MTL add value?                        | Full               | "                | MLP-MTL vs 3 single-task MLPs | Walk-forward                               | All 3              |
| E13           | Does ensembling add value?                 | Full               | "                | blend + stack                 | Walk-forward                               | All 3              |
| E14           | Ex-ante vs ex-post                         | Model A vs Model B | "                | Best of E11                   | Walk-forward                               | All 3              |
| E15           | Does Y3 need a hurdle model?               | Full               | "                | Regression vs hurdle          | Walk-forward                               | Y3                 |
| E16           | Are severe events predictable?             | Full               | "                | Best of E11                   | Walk-forward, stratified report            | All 3              |
| E17           | Is performance stable?                     | Full               | "                | Best of E11                   | 30/10/10 and 20/5/5, rolling and expanding | All 3              |
| E18           | Recovery-definition robustness             | Full               | "                | Best of E11                   | Walk-forward                               | Y3 (3 definitions) |

E04–E06 are the study's central experiment and have never been run.

---

## 35. Expected Performance Improvements

**No numerical predictions are given** — the master prompt forbids fabricating them, and this audit has already produced one case where an estimate (the naive-baseline comparison) turned out to be wrong when actually computed.

Qualitative expectations, ordered by confidence:

- **Prediction clipping to definitional bounds:** *provably* non-worsening. Clipping is Euclidean projection onto the target's support, and every true value already lies inside it, so absolute error cannot increase on any point. RMSE and MAE non-increasing, pooled R² non-decreasing, conformal coverage exactly unchanged with non-increasing width. This is the only change in the entire audit with a guarantee attached.
- **Ridge standardisation:** large effect on Ridge, already measured. Direction on other models: none (they are scale-invariant).
- **Stationary feature replacement:** already measured; largest single improvement observed, concentrated in the models that extrapolate (XGBoost, MLP, Ridge).
- **Capacity bounding (RF/XGB):** expected to move R² *toward* zero from below — "less wrong", not "predictive". Say so.
- **Hurdle model for Y3:** unknown. Better matched to the distribution, but adds a second model to fit on 30 rows.
- **Macro publication-lag fix:** expected to make results *slightly worse*. Do it anyway and report it.
- **Ex-ante feature set (Model A):** expected to be substantially worse than Model B. That gap is itself the finding.

---

## 36. Recommended Final Architecture

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

## 37. Methodology Amendments Required in the Thesis

| #  | Current statement                                  | Problem                                                                       | Replacement                                                                    | Section                            |
| -- | -------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------- |
| 1  | "ASPI Percentage Change"                           | A log return is not a percentage change                                       | "ASPI log return, ln(P_t/P_{t−1})"                                            | §3.2.2, Table 4, abstract, RQ1–4 |
| 2  | "Trading volume crash magnitude"                   | Variable measures deviation in both directions; observed max +1.94 is a spike | "Abnormal trading volume (ATV)"                                                | §3.2.2, Table 4, abstract, RQ1    |
| 3  | "Consecutive trading days" (Y3)                    | Code returned calendar days                                                   | Retain the trading-day definition; state that the implementation was corrected | §3.2.2                            |
| 4  | Recovery capped at 90 with no censoring discussion | `T > 90` is not `T = 90`                                                  | Add right-censoring discussion + hurdle model                                  | §3.2.2                            |
| 5  | "No k-fold anywhere"                               | Inner`cv=3` was a k-fold                                                    | Retain the principle; document the inner`TimeSeriesSplit`                    | §3.7.1                            |
| 6  | "The training set to grow over time"               | Implementation used a rolling window                                          | Report both; state which is primary                                            | §3.7.1                            |
| 7  | `MultiOutputRegressor` wrapper described         | Not used; separate per-target fits                                            | "Independent per-target estimators"                                            | §3.6.4                            |
| 8  | "Multi-task learning" for tree models              | Multi-output ≠ multi-task                                                    | Reserve MTL for the MLP                                                        | §3.6.4, abstract                  |
| 9  | "RMSE measures the total variation explained"      | Wrong — that is R²                                                          | "RMSE is the root of the mean squared error, in the target's own units"        | §3.8.1                            |
| 10 | "Explainable early-warning system"                 | Six features are ex-post                                                      | "Ex-post impact-attribution framework"; add Model A as future work             | Abstract, §1.2.3, §4.4, README   |
| 11 | "Aggregated sentiment scores... were used"         | Contradicts Table 4 and the code                                              | Delete; state sentiment was excluded and why                                   | §3.3.1                            |
| 12 | SVR listed as a model                              | Never implemented                                                             | Remove, with a stated reason                                                   | §3.6.1                            |
| 13 | Loss weights as eq.(4) scale mechanism             | y is standardised first, so scale gap is already gone                         | Weights express task priority                                                  | §3.6.4                            |
| 14 | Macro variables with no publication-lag treatment  | Look-ahead                                                                    | Add as-of-date alignment                                                       | §3.3.1                            |
| 15 | ADF/KPSS "confirm log_return is stationary"        | Only one feature tested                                                       | All features tested; unit-root columns excluded                                | §3.5.1                            |
| 16 | Overlapping-window truncation described            | Function never called                                                         | Either wire it in or state it was not applied and quantify overlap             | §3.3.2                            |
| 17 | No statement of Y1↔Y3 dependence                  | `Y3 = 0 ⟺ Y1 >= 0` by construction                                         | Add explicitly before any Y3 result                                            | §3.2.2                            |
| 18 | 100% real data                                     | 3 Y2 values were zero-filled                                                  | Now true; report Y2 N = 61                                                     | §3.4.1                            |
| 19 | "First-ever" framework                             | Unverifiable superlative                                                      | "To our knowledge, the first..."                                               | §1.2.3                            |
| 20 | Feature-importance language implying causation     | SHAP is attribution                                                           | "Predictive association"                                                       | §3.8.2                            |

---

## 38. Research Contribution After Improvement

Stated honestly, the contribution is **not** a working predictor. It is four things, and they are real:

1. **A well-evidenced null result.** Across five model families, two validation geometries, two feature counts and with/without augmentation, index-level CSE response to qualifying natural disasters is not predictable at this sample size. Median Y1 is +0.000003 and median Y3 is 0 — more than half of qualifying disasters produce no measurable day-0 index reaction. For a market the thesis itself characterises as thin and semi-strong-inefficient, that is a genuine, interpretable finding: index-level aggregation and low liquidity absorb localised physical shocks. Consistency across specifications is what makes a null credible, and most undergraduate theses cannot offer a robustness surface at all.
2. **A methodological contribution on multi-task regression with heterogeneously-scaled targets.** The finding that loss weights of (1.0, 0.1, 0.5) cannot compensate a ~1000× raw-scale gap, and that target standardisation is a necessary complement, is transferable and was not documented anywhere the student found it. (Before the fix, MLP Y1 R² was ≈ −390.)
3. **A negative-results register that saves the next researcher real time.** Yahoo's `^CSE` feed verified dead three ways; the market-cap CSV rejected for having no historical join key; the widened-SMOGN variant tried, measured and reverted; SMOGN's own defects found and corrected. In frontier-market data work this is the scarcest material in the literature.
4. **Reusable data infrastructure.** A parser handling three distinct CSE report layouts across 23 years via header-text detection, and a validated 64-event CSE-disaster panel that does not exist elsewhere.

**One-sentence statement of contribution:** *the first end-to-end, leakage-audited multi-target ML framework for disaster impact on a frontier exchange, applied to a complete real 64-event panel, which establishes — robustly across five model families and two validation geometries — that index-level CSE response to qualifying natural disasters is not predictable at this sample size, and which documents the specific data, leakage and target-construction obstacles any future attempt must overcome.*

---

## 39. Limitations Remaining

1. **N = 64.** The binding constraint. No technique overcomes it.
2. **Ex-post features.** Until Model A is built, the framework cannot support prospective claims.
3. **Annual macro against event-level timing.** CBSL monthly CCPI, daily LKR/USD and policy rates were not accessible; World Bank annual series are a coarse proxy even after the lag correction.
4. **Archive ends Jun-2023 (price) / Mar-2023 (volume).** Storm Ditwah (Nov 2025), the thesis's own flagship example, is outside the modelled set.
5. **EM-DAT reporting bias.** Most Sri Lankan records carry no damage estimate; damage features partly measure reporting coverage.
6. **Y3 censoring and its dependence on Y1.**
7. **No event-window contamination screen.** Concurrent elections and policy shocks are not controlled — note the 2005 presidential election four days before the largest observed drop.
8. **Optimistic bias from configuration choices made on the reported folds.** Disclosed, unquantifiable.
9. **Interpolated volatility is convexity-biased high** in synthetic rows and cannot be corrected without the price path.
10. **Live unpinned API pulls** make the exact table non-reproducible until cached.
11. **Inner-CV noise.** Chronological inner folds of 9/16/23 rows are close to uninformative; the response is a deliberately small search space, not a better search.

---

## 40. Viva Defence

**"Nine of your twelve R² values are negative. Why should we believe anything here?"**
Because a negative pooled R² is a statement about the pooled test mean, not about usefulness. R² penalises against an oracle that already knows the test set's mean; our chronological split places the two largest shocks permanently in fold 0's training window, leaving the held-out folds with 63% of full-sample variance on Y1. The question that matters is whether the model beats a *deployable* baseline, so we computed two — a constant-zero economic null and a training-mean statistical null — on identical folds. The honest answer is that on Y1 no model beats them, on Y2 only the corrected Ridge does, and on Y3 the training-mean predictor wins. We report that rather than the R² framing because it is the more meaningful comparison, and because it is less flattering.

**"On the day before the flood, how do you know its total damage?"**
We do not. Six of our features are EM-DAT post-hoc assessments, so this framework answers the attribution question — given a disaster of measured severity, what was the market response — and not the forecasting question. We corrected the early-warning language in the README and thesis when we found this, and specified an ex-ante feature set as the separate experiment it actually requires.

**"Section 3.7.1 bans k-fold. What was `cv=3`?"**
A k-fold, inside every walk-forward fold, selecting hyperparameters by training on later events to validate earlier ones. We found it during audit and replaced it with a chronological `TimeSeriesSplit` restricted to each fold's real rows, with the winner refit on real plus synthetic. We report it because the contradiction between our stated method and our code is exactly the kind of thing an audit exists to catch.

**"Your best directional accuracy is 65%. What does 'always predict no crash' score?"**
Also 65%. No model beats the trivial rule, and several score below it. That is why the notebook now prints the majority baseline beside every accuracy figure and reports AUC with a Hanley–McNeil interval that includes 0.5 in every case.

**"Why should a Random Forest work on 64 events?"**
It largely does not, and we quantify it: train-fit R² of 0.883 on Y3 against a held-out −0.146. We report that 1.03-point gap as an overfitting diagnostic rather than as a performance figure.

**"Is Y3 = 0 the same as Y1 ≥ 0?"**
Yes, by construction — the recovery window includes the event day and the baseline is the prior close. Over half our Y3 values are therefore mechanically determined by the sign of Y1. We state this before reporting any Y3 result, and it is why we recommend a hurdle model whose first stage is acknowledged to be close to a sign classifier.

**"You chose 30/10/10. What happens at 20/5/5?"**
Random Forest Y1 pooled R² goes from −0.322 to +0.151 — the best number anywhere in our study. We do not promote it, because it was evaluated on the same folds we report and adopting it would be selection on the test set. We report it at equal prominence as an upper bound on what this pipeline can be made to appear to achieve.

---

## 41. Final Ten-Expert Verdict

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
| 10. Critical Thesis Examiner  | **PASS WITH CORRECTIONS** | **Blocking: resolve the early-warning framing (P0-2) and run E04–E06** |

**No unresolved P0 remains in the code.** Two P0-class items are documentation/experiment obligations rather than defects: the early-warning reframing (P0-2) and the disclosure of configuration selection on reported folds (P0-7, disclosed and unquantifiable).

**Overall: PASS WITH CORRECTIONS.** The architecture may be labelled recommended once P0-2 is resolved in the text and E04–E06 have been run.

---

# TOP 10 CHANGES TO IMPLEMENT FIRST

Ranked by (research value × defensibility) / (complexity × overfitting risk). Status reflects work already completed during this audit.

### 1. Run the market-vs-disaster ablation (E04–E06) — **NOT YET DONE**

- **Current problem:** the thesis exists to answer "do disaster characteristics add predictive information beyond ordinary market history?" and that experiment has never been run.
- **Exact change:** three additional `run_walk_forward` calls with `FEATURE_COLS` restricted to (a) market-only, (b) disaster-only, (c) market+disaster; report `ΔRMSE = RMSE_market_only − RMSE_market+disaster` per target.
- **Why:** without it the study cannot claim disaster data contributes anything. It is the central research question.
- **Affected target:** all three.
- **Files/cells:** notebook cell `15d59212` (feature subsets), new cell after `c8445d58`.
- **Experiment:** E04, E05, E06.
- **Success metric:** ΔRMSE with a paired bootstrap CI over pooled out-of-fold squared errors.
- **Reject if:** ΔRMSE CI straddles zero — in which case report honestly that disaster features add nothing measurable, which is itself a publishable finding.

### 2. Resolve the early-warning framing (P0-2) — **PARTLY DONE**

- **Current problem:** six features are EM-DAT post-hoc assessments; the README and thesis claim an early-warning system.
- **Exact change:** README and notebook §15 corrected to "ex-post impact attribution" (done). Remaining: build Model A (ex-ante feature set) and report both.
- **Why:** the claim as written is unsupportable and is the first thing an examiner will attack.
- **Affected target:** all three.
- **Files/cells:** `README.md` (done), notebook `e62f629a` (done), `c868f959` (done); Model A requires a new feature-subset cell.
- **Experiment:** E14.
- **Success metric:** two labelled models reported; no prospective claim attached to Model B.
- **Reject if:** never — the reframing is mandatory regardless of results.

### 3. Fix the inner k-fold — **DONE**

- **Problem:** `GridSearchCV(cv=3)` = `KFold(shuffle=False)` inside every walk-forward fold, contradicting thesis §3.7.1; SMOGN's tail-appended synthetic rows made the last inner block up to ~45% synthetic.
- **Change:** `TimeSeriesSplit(n_splits=3)` on each fold's **real** rows, `refit=False`, winner refit on real+synthetic.
- **Files:** notebook `c5f83fcb`.
- **Metric:** no k-fold present anywhere; the results change is incidental, correctness is the point.
- **Reject if:** never.

### 4. Standardise Ridge and tune alpha in-fold — **DONE**

- **Problem:** the H1 baseline was fit on unstandardised features mixing 1e4 price levels, 1e9 USD damage and 0/1 flags, with `alpha` left at its library default.
- **Change:** `make_pipeline(StandardScaler(), Ridge(alpha=RidgeCV-selected on real rows))`.
- **Measured effect:** Ridge Y1 −1.634 → −0.407 → −0.278 (after stationarity fix); Y2 −3.539 → −0.015 → **+0.053**.
- **Reject if:** never — a mis-specified baseline invalidates the H1 comparison in either direction.

### 5. Replace non-stationary price levels — **DONE**

- **Problem:** `sma_*`/`ema_*` are raw ASPI levels (574 → 10,000+); under a chronological split the test fold sits outside training support.
- **Change:** stationary `price_to_sma_w = P_{t−1}/sma_w − 1` ratios; ADF/KPSS on all features; exclude on ADF unit root only.
- **Measured effect:** largest single improvement in the study — XGBoost Y1 −1.617 → −0.236, MLP Y1 −1.547 → −0.299, stacked Y1 −0.775 → −0.134.
- **Reject if:** never — thesis §3.5.1 mandates stationarity.

### 6. Add naive baselines and the majority baseline — **DONE**

- **Problem:** no null to compare against; directional accuracy reported against nothing (65% against a 65% majority baseline).
- **Change:** `naive_zero` and `naive_train_mean` rows in the summary table with a `beats_null_rmse` column; majority baseline printed beside every accuracy; Hanley–McNeil CI on AUC.
- **Measured effect:** revealed that **0 of 6 models beat the no-effect null on Y1**.
- **Reject if:** never — this is a reporting-integrity fix, not a performance change.

### 7. Clip predictions to definitional bounds — **DONE**

- **Problem:** Ridge produced Y3 RMSE 157 on the dense config (target capped at 90) and a 182-day conformal interval on a 90-day-capped target.
- **Change:** `clip_to_bounds` — Y3 to [0, 90], Y2 to ≥ −1 — applied at every prediction site including ensemble, stack and interval endpoints.
- **Why:** uses only target definitions from §3.2.2; provably cannot increase absolute error on any point.
- **Reject if:** never — it carries a mathematical guarantee.

### 8. Stop imputing missing targets — **DONE**

- **Problem:** `y.fillna(0.0)` asserted "volume exactly at baseline" for 3 events, in a study claiming 100% real data, using the same zero-fill §3 explicitly rejects for volume.
- **Change:** NaNs preserved, masked per target; Y2 effective N reported as 61.
- **Reject if:** never.

### 9. Bound RF/XGBoost capacity by fold size — **NOT YET DONE**

- **Current problem:** grids allow `max_depth=None`, `min_samples_leaf=1` on 33-row folds; RF train-fit R² 0.883 vs held-out −0.146 on Y3.
- **Exact change:** `RF_PARAM_GRID = {"n_estimators":[300], "max_depth":[2,3,4], "min_samples_leaf":[3,5]}`; `XGB_PARAM_GRID = {"n_estimators":[100,300], "max_depth":[2,3], "learning_rate":[0.03,0.1], "subsample":[0.8], "colsample_bytree":[0.8]}`.
- **Why:** bounded a priori by fold size (leaf ≥ 3 means every leaf rests on ~10% of the fold), not by test score.
- **Affected target:** all three.
- **Files/cells:** notebook `c5f83fcb` grid definitions.
- **Metric:** the train-fit-to-held-out R² gap should narrow; held-out R² expected to move toward 0 from below.
- **Reject if:** held-out RMSE worsens materially on two or more targets.

### 10. Implement the Y3 hurdle model — **NOT YET DONE**

- **Current problem:** the 90-day cap is treated as an observed value; `T > 90` is not `T = 90`.
- **Exact change:** stage 1 classifier for P(recovery ≤ 90 trading days); stage 2 regressor on `log1p(duration | recovered)`; combine as `P·E[d|recover] + (1−P)·90`.
- **Why:** matches the zero-inflated, right-censored structure; avoids a survival model the sample cannot support.
- **Affected target:** Y3 only.
- **Files/cells:** notebook `c5f83fcb`, Y3 branch.
- **Experiment:** E15.
- **Metric:** MAE in trading days vs the current single-stage regressor and vs `naive_train_mean`.
- **Reject if:** MAE does not improve over single-stage — report the negative result and keep the simpler model (parsimony).

---

# NEXT 5 EXPERIMENTS, IN ORDER

### Experiment 1 — Do disaster variables add predictive value? (E04–E06)

- **Hypothesis:** adding EM-DAT disaster severity features to a market-history baseline reduces out-of-sample RMSE on at least one target.
- **Current configuration:** all 27 features together; no decomposition.
- **Modified configuration:** three runs — market-only, disaster-only, market+disaster — identical folds, identical model (RF), identical SMOGN.
- **Validation:** expanding chronological walk-forward, 30/10/10.
- **Metrics:** ΔRMSE, ΔMAE, Δpooled R² per target, with paired bootstrap CI on pooled squared errors.
- **Expected interpretation:** if the ΔRMSE CI straddles zero on all three targets, the study's central premise is not supported at this N — report that as the primary finding. If disaster-only beats market-only on Y3, that is the strongest positive result available.

### Experiment 2 — Does SMOGN add value? (E08–E10)

- **Hypothesis:** time-aware SMOGN improves prediction of rare high-impact events without degrading overall performance.
- **Current configuration:** SMOGN always on; never ablated.
- **Modified configuration:** (a) no oversampling, (b) current SMOGN, (c) sample weighting by target rarity instead of resampling.
- **Validation:** as above.
- **Metrics:** overall RMSE **and** RMSE restricted to the rare subset (`Y3 > 30`), reported separately.
- **Expected interpretation:** with 3 synthetic rows against 30 real, the honest expectation is no measurable difference. Publishing that is a legitimate contribution — the thesis proposes SMOGN, so it owes the reader evidence either way.

### Experiment 3 — Rare-event stratified performance (E16)

- **Hypothesis:** models that perform acceptably overall fail on the catastrophic events the study exists to predict.
- **Current configuration:** aggregate metrics only.
- **Modified configuration:** partition pooled out-of-fold predictions into common / rare (top-decile severity) / worst single event; report per stratum.
- **Validation:** no re-run required — computed from the stored `results` arrays.
- **Metrics:** RMSE and MAE per stratum; the 2004-12-26 and 2005-11-21 events named individually.
- **Expected interpretation:** if rare-event error is much worse than overall error, the research objective is not met regardless of aggregate numbers. This must be known before any performance claim is made.

### Experiment 4 — Ex-ante vs ex-post (E14)

- **Hypothesis:** a feature set restricted to information available at disaster onset performs materially worse than one including realised damage.
- **Current configuration:** ex-post only (Model B).
- **Modified configuration:** Model A — disaster type, month/season, `days_since_last_disaster`, `disasters_trailing_365d`, historical type-average severity computed from prior events only, all market features, lag-corrected macro.
- **Validation:** as above, identical folds.
- **Metrics:** RMSE per target for A vs B.
- **Expected interpretation:** B is expected to beat A. **The size of that gap is the quantified value of post-event information** — the number that tells a policymaker how much better an assessment-based tool is than a forecast. This converts the framing problem into a finding.

### Experiment 5 — Recovery-definition robustness (E18)

- **Hypothesis:** conclusions about Y3 do not depend on the specific recovery definition.
- **Current configuration:** exact recovery (ASPI ≥ pre-event baseline), single definition.
- **Modified configuration:** (a) exact, (b) tolerance — within 1% of baseline, (c) stable — at or above baseline for 3 consecutive sessions.
- **Validation:** as above.
- **Metrics:** MAE in trading days per definition; correlation between the three Y3 vectors; whether model ranking changes.
- **Expected interpretation:** if conclusions flip across definitions, Y3 results are definition-artefacts and must be heavily qualified. Do **not** adopt whichever definition scores best — report all three and keep the pre-registered exact definition as primary.

---

## Rules Observed in This Audit

No performance value in this document was fabricated, estimated or extrapolated. Every figure was copied from an executed run; unmeasured quantities are marked "not measured". No modification is described as working before experimental evidence demonstrated it — where a change was measured, the measurement is shown, including the cases where it made results worse (the widened SMOGN variant) or destroyed a favourable number (K=10 removing the only positive R²). Where an earlier estimate was contradicted by later measurement, the correction is stated explicitly (the naive-baseline comparison).

---

## 42. Visualization and Diagnostics

This section did not exist in the original audit, and its absence was itself a defect: the
pipeline produced **two images across 59 notebook cells**, both SHAP, neither exported to
disk, so nothing in the study could be cited as a figure. More importantly, three findings
recorded here in prose had no visual evidence at all, and two of them are exactly the kind
of claim a reader will not accept on assertion.

Figures now live in `src/evaluation/figures.py` and export to `docs/figures/` at 300 dpi
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

### Which audit finding each figure evidences

| Figure | Evidences |
|---|---|
| `fig_02_target_distributions` | Y3's zero-inflation and censoring (P1-2) — the ECDF panel shows the point masses at 0 and 90 that a histogram hides |
| `fig_03_target_dependence_y1_y3` | **P1-3**, `Y3 = 0 ⟺ Y1 ≥ 0`. The contingency inset shows the *Y1 ≥ 0 and Y3 > 0* cell is empty |
| `fig_04_feature_correlation_heatmap` | **P2-3**, severe collinearity. Spearman, clustered, with cells below the n=64 significance threshold greyed out |
| `fig_06_collinearity_vif` | **P2-3** quantified — VIF on a log axis plus the condition-index panel |
| `fig_09_walk_forward_folds` | §26, the variance-compression artefact: both extreme events sit permanently in fold 0's *training* window |
| `fig_11_conformal_coverage` | The model-vs-marginal interval comparison — coverage is meaningless without width |
| `fig_14_roc_with_ci` | Directional performance, with the chance diagonal and an automatic "includes 0.5" flag on any CI covering chance |
| `fig_17` / `fig_18` | §13, the baseline comparison. The skill forest is the decisive one: a whisker crossing zero is not a win |
| `fig_19_pred_vs_actual` | What a negative R² actually looks like — predictions collapsing into a narrow band off the identity line |
| `fig_24_overfitting_gap` | **P2-2**, the memorisation gap between in-sample refit and held-out R² |
| `fig_31_sector_response_and_skill` | The sector extension — response and predictability per sector |

### Measured by the new diagnostics

The collinearity figures immediately produced numbers the audit had only characterised
qualitatively:

- **Condition number ≈ 2.3 × 10¹⁷.** The usual threshold for concern is 30.
- **Three feature pairs at Spearman exactly 1.000**: `financial_damage` /
  `log_financial_damage`, `population_affected` / `log_population_affected`, and
  `vol_ratio_10_30` / `vol_trend_10_30`.
- The pre-declared redundancy rule (within any group correlated above |ρ| = 0.95, keep the
  least-derived member) removes **8 of 37** features.

The third pair was a defect in the new volume block introduced during this remediation —
a 10-day mean over a 30-day mean is by definition the 10/30 ratio, and the first version
shipped both. The diagnostic caught it before it reached a model, which is the argument
for building the diagnostic.

### Deliberately not built

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

## Improvement attempts (Y1 / Y3, targets held fixed)

Following the results in `docs/RESULTS_AUDIT.txt` (Y1_aspi_log_return R2=-0.15,
Y3_recovery_days R2=-0.02, only Y2 beats baseline), a literature/repo search was run
before writing any code, restricted to techniques that improve predictability of the
SAME target definitions (no target reframing). Findings actually implemented below;
everything traces to a cited source.

### Sources consulted

- Stock price recovery after a market shock: a survival analysis approach (ResearchGate,
  2026) and "A survival analysis method for stock market prediction" (ResearchGate) —
  both model firm-level post-shock recovery time as a right-censored duration (firms not
  recovered by the observation cutoff are censored, not coded as "recovered at the cutoff"),
  fit with a Cox / AFT model rather than point regression on the censored value.
- "Measuring and forecasting financial system resilience under multiple shocks: a survival
  analysis approach" (ScienceDirect) — Cox proportional-hazards treatment of financial
  recovery under repeated shocks, same censoring logic.
- Davidescu et al. (2025), "Evaluating Sectoral Vulnerability to Natural Disasters in the US
  Stock Market... DCC-GARCH Models" (already in the thesis reference list) and
  "GARCH-Informed Neural Networks for Volatility Prediction in Financial Markets" (ACM,
  2024) — conditional (GARCH) volatility as a feature carries information a realized
  rolling standard deviation does not (it is a forecast, not a backward-looking average),
  and is the standard volatility-modeling companion to ML regressors in this literature.
- General small-N financial ML literature (permutation importance / RFE feature pruning to
  fight collinearity) — already implemented in this repo's `select_top_features` /
  `_select_top_features` (per-fold RF-importance top-K), confirmed present in
  `src/training/walk_forward.py` and `scripts/train_final_models.py`; no further action
  needed there, it was already the state of the art for this sample size.

### What was implemented

1. **`src/models/survival_recovery.py`** — a right-censored AFT model (via `lifelines`,
   `WeibullAFTFitter`/`LogNormalAFTFitter`, selected in-fold by AIC) for Y3. The existing
   `Y3_recovery_days` target is used unchanged; the only change is that observations at the
   90-day cap are marked `event_observed=False` (censored) instead of being treated as a
   literal observed value of 90, which is the textbook-correct likelihood for exactly this
   data shape (right-censored at a fixed follow-up window). Reported alongside the existing
   RMSE/MAE table plus a concordance index (the standard metric for censored time-to-event
   predictions, not meaningful for the other point-regression models).
2. **`garch_cond_vol` feature** in `src/data_pipeline/feature_eng.py` — GARCH(1,1)
   conditional-volatility forecast on the ASPI log-return series (via the `arch` package),
   refit on an expanding window with an annual refresh cadence so no fold ever sees a
   volatility estimate whose GARCH parameters were fit on data beyond that fold's own
   history point — same causal discipline as the existing `shift(1)`-guarded rolling
   features, documented inline. Added as an extra Y1 feature candidate; it flows through
   the existing per-fold `select_top_features` step like any other column, so it is kept
   only if it earns its place, not force-included.

### What was NOT implemented, and why

- **Bayesian hyperparameter optimization** — considered and rejected per the existing
  audit note (§21): would search a larger space on ~30-40 training rows, increasing
  selection overfitting risk for no evidenced benefit over grid search at this N.
- **Sector-level reframing of Y1** — explicitly out of scope per author instruction: Y1
  must remain the ASPI-aggregate log return, not a sector sub-index.
- **Deep sequence models (LSTM/Transformer)** — ruled out per existing audit reasoning;
  N=40-76 is far below what these architectures need to generalize rather than memorize.

### Measured results (targeted ablations, real data, same folds as the main table)

**Y3 -- AFT survival model** (`scripts/run_survival_model.py`, n=40 pooled test points,
identical walk-forward folds/SMOGN/feature-selection as the main table):

| model | RMSE | MAE | pooled R2 |
|---|---|---|---|
| naive_zero | 32.254 | 13.725 | -0.221 |
| naive_train_mean | 29.469 | 19.922 | -0.019 |
| **aft_survival** | 34.971 | 17.281 | -0.435 |

Point-prediction RMSE is worse than both naive baselines -- the AFT median does not, on
this sample, out-predict a flat training mean any more than the hurdle model did. BUT
the model was fit to optimise a survival likelihood, not RMSE, and on the metric that
likelihood actually targets -- Harrell's concordance index, i.e. "does it rank which
events recover faster than which others correctly more often than chance" -- it scores
0.696 / 0.448 / 0.529 / 0.600 across the four folds, mean **0.568** (0.5 = chance). That
is a real, if modest, ranking signal invisible to a plain RMSE comparison: correctly
treating the 90-day cap as censoring, rather than as an observed value of 90, recovers
some genuine ordinal information about recovery speed that the point-regression models
(including the hurdle model) do not surface. This is the honest result to report: a
positive methodological finding (censoring matters, ranking signal exists) alongside a
still-negative one (point RMSE does not beat naive at N=40).

**Y1 -- GARCH(1,1) conditional volatility feature** (`scripts/run_garch_ablation.py`,
Ridge, n=40, with vs without `garch_cond_vol` in the candidate pool, everything else
identical):

| variant | RMSE | MAE | pooled R2 |
|---|---|---|---|
| without_garch | 0.01555 | 0.00981 | -0.301 |
| with_garch | 0.01609 | 0.01017 | -0.393 |

delta_rmse = -0.00054, 95% CI [-0.00434, +0.00163] -- **does not exclude zero**: adding
a GARCH conditional-volatility forecast to the feature pool makes no statistically
distinguishable difference to Y1 prediction, and the point estimate is if anything
slightly worse (consistent with the audit's own collinearity finding -- one more
correlated volatility-family column adds selection noise more readily than it adds
signal at N=40). This null result is itself consistent with the lit review already in
the thesis (Kengatharan & Jeyan Suganya, 2019; Priyadarshani & Perera, 2023): the
aggregate ASPI index appears to be genuinely difficult to predict at the daily-return
level regardless of which volatility feature feeds it, which is the diversification
story those papers already tell -- feature engineering does not manufacture a signal
the aggregate index may not carry.

### Bottom line for the thesis

Two real techniques were implemented, both correctly, both tested on real data through
the actual pipeline (not simulated): a right-censored AFT survival model for Y3, and a
GARCH conditional-volatility feature for Y1. Report both outcomes as findings, not as
failures to hide: Y3 point-RMSE is still not beaten, but a genuine, citable, positive
result exists in the concordance index; Y1 remains a defensible null result strengthened,
not weakened, by having tried a targeted, literature-backed feature and shown it does
not move the needle. This is a stronger, more honest thesis than either silently omitting
the attempt or overstating what it achieved.

### Full official re-run confirms it (not just the targeted ablation script)

`notebooks/02_features_targets.ipynb` -> `04_modeling_regression.ipynb` ->
`06_evaluation.ipynb` -> `scripts/audit_results.py` were re-executed end to end with
`garch_cond_vol` live in the candidate feature pool (92 features instead of 91). The
resulting `docs/RESULTS_AUDIT.txt` is **byte-identical to the pre-change version in
every single model/target row** except the feature count in the data-coverage section
(91 -> 92). That means the per-fold RF-importance top-20 selector never once picked
`garch_cond_vol` over the other 91 candidates, for any target, in any fold -- the
strongest possible version of the null result: it is not just statistically
indistinguishable when forced in, the pipeline's own feature-selection step
independently agreed it does not carry enough signal to make the cut. The AFT survival
model's c-index result (0.568) stands as reported above, unaffected by this (Y3's
feature pool is separately selected and the AFT script above was run against the exact
same real dataset).

## Y1 redefinition to 5-trading-day cumulative return, and a feature/model pass

Following the improvement work above, the author redefined Y1 (target definition, not a
technique) from a single-day log return to `ASPI_5D_Log_Return_Pct = 100 * ln(ASPI_(t+5)
/ ASPI_t)`, `t+5` counted in actual CSE trading sessions (verified by hand against 5 real
events; see the manual-verification note kept alongside `feature_eng.py`). This produced
a positive pooled R2 for the first time on any Y1 model (ensemble R2=+0.085 at the point
this redefinition was scored), though not statistically distinguishable from naive_zero
(bootstrap CI on delta_rmse included zero). A fold-boundary purge
(`src/training/walk_forward.py::purge_horizon_overlap`) was added at the same time,
because a 5-day-forward label can otherwise leak across a walk-forward fold boundary --
3 real event pairs in this dataset are closer together than the 5-trading-day horizon.

A further feature/model pass was then run, cited and reasoned before implementation:

- **Collinearity pruning composed into feature selection.** `src/evaluation/collinearity.py`
  already implemented the pre-declared |rho|>=0.95 redundancy rule (see the collinearity
  section earlier in this document) but it had never actually been wired into
  `notebooks/04_modeling_regression.ipynb`'s `select_top_features` -- only RF-importance
  ranking ran there. Composed as: redundancy-drop (reads no target, so not selection on
  the test set) -> RF-importance top-k, on each fold's real training rows only. Standard
  filter-method combination for small-N tabular data (VIF/correlation pruning + importance
  ranking; a >0.75 correlation-pair diagnostic table is also generated and saved to
  `artifacts/y1_feature_stability.csv`, but the ACTUAL drop threshold stays at the
  pre-registered 0.95, per author decision, not the harder 0.75 that would count as
  revising a pre-declared rule after seeing results).
- **PCA** — added as a reported ablation only (`notebooks/04_modeling_regression.ipynb`,
  section 4.8b): PCA(10 components)+Ridge vs RF-selected-features+Ridge, same folds. Not
  adopted as a default (PCA components are uninterpretable, which cuts against the
  explainability/SHAP chapter) unless it clearly wins.
- **Gaussian Process regression, SVR, median quantile regression** added to the model
  lineup (literature-standard choices for small-N nonlinear regression with calibrated
  uncertainty; SVR specifically closes this document's own earlier-flagged open item,
  "SVR specified but never implemented").

### Measured effect (full official pipeline re-run, real data)

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
marginal signal at this N -- collinearity pruning is a correctness/interpretability
argument (it directly fixes a documented problem: `sma_5/10/20` etc. at ~99% mutual
correlation), not a guaranteed RMSE improvement, especially with only ~30 training rows
per fold where any feature-set change is high-variance.

Y2 moved the other way (RF R2 0.247->0.268, and SVR/GP/XGBoost/ensemble all newly beat
naive_zero where previously only RF/ensemble/XGBoost did) -- the same change helped one
target and hurt another, which is itself informative: it is evidence AGAINST a single
fixed feature-selection recipe being optimal for all three targets simultaneously, not
evidence that collinearity-pruning is simply "good" or "bad" in general.

**SVR and Gaussian Process regression**: SVR is now the only Y1 model with both a
positive pooled R2 (+0.018) and a positive skill vs naive_zero, at this specific
snapshot of the feature pipeline -- reported, not yet claimed as a stable winner (single
run, no dedicated bootstrap-CI comparison of SVR vs naive_zero performed yet; see the Y1
experiment suite below for that comparison). GP scores close to zero (R2=-0.009),
consistent with the general small-N finding that GP's main value here is calibrated
uncertainty, not a point-accuracy win.

**Quantile regression (median, alpha=0.01, solver="highs")**: badly miscalibrated on
every target (Y1 R2=-2.73, Y2 R2=-1.33, Y3 substantially worse than naive) -- the fixed
alpha=0.01 is very likely too weak a regularizer for a ~76-row, ~20-feature design after
collinearity pruning, letting the LP-based fit chase individual training points. Recorded
as a failed configuration, not a working addition to the lineup as currently tuned; would
need either much stronger regularization or a proper inner-CV alpha search (not attempted
here, ponytail: shipped the lazy fixed-alpha version, this is exactly the case where it
measurably underperforms and the grid-search upgrade is warranted before using it for
anything).

### Y1-specific experiment suite (SMOGN ablation, compact features, regime features,
ExtraTrees, single-task modeling, OOF ensemble weighting, shrinkage)

See `scripts/run_y1_experiments.py` for the full implementation and
`artifacts/y1_experiments_ranked.csv` / `artifacts/y1_feature_stability.csv` for results.
Numbers appended below once the run completes.

### 2026-09-16: methodology-audit freeze -- target names, units, inclusion threshold

An external methodology review (ML/CSE/thesis-panel style audit, 44 numbered findings)
flagged three sources of drift that this section freezes, so every earlier mention of the
old names/threshold above is historical and describes runs made under them -- it is not
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
   session, so the final dataset is byte-for-byte the same 76 events either way -- this
   revert is free.
3. **What this does not fix.** Findings #5-#8, #11-#22 (target-specific label-horizon
   purging, Y3 competing-risk censoring, missingness semantics, nested CV, and the
   prediction-origin alignment question) are separate, larger changes and are tracked
   as follow-on work, not resolved by this freeze.

### 2026-09-16: target-specific label-horizon purging (methodology-audit finding #5)

Before this, `notebooks/04_modeling_regression.ipynb`'s `run_walk_forward` purged every
target against a single shared column (Y1's 5-trading-day horizon), applied once per
fold. That under-purged `Y1_EventWindow_0_10_LogReturn_Pct` (a 10-day horizon) and
`Y3_recovery_days` (up to 90 trading days), and over-purged `Y2_abnormal_volume` (which
has no forward horizon at all -- its label is known the same session). Three other
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
Y3's horizon -- the most conservative available -- rather than nothing. Notebook 05 and
`run_survival_model.py` now purge every fold, per label/target, for the first time.

**Measured effect, Y3 AFT survival model (the only case with a like-for-like before/after
run available)**: pooled C-index 0.557 -> **0.640** (folds: 0.696/0.448/0.500/0.917 vs the
previous 0.696/0.448/0.500/0.583) on the same n=40 pooled test points -- test rows are
untouched by purging, only training rows are removed, so this is a same-sample
comparison. This reads as a genuine correctness fix (the previous number was computed
with zero fold-boundary protection, i.e. it likely OVERSTATED accuracy from leakage, not
understated it), not evidence the model works better -- reported honestly either
direction, per this document's standing rule.

Y1/Y2 point-regression numbers and the 6 classification labels also shifted slightly
(a handful of borderline-adjacent training rows now excluded per fold); no comparison's
qualitative conclusion changed (C2_volume_spike remains the only classification label
confirmed to beat baseline; no Y1 candidate is statistically confirmed either way).

### 2026-09-16: purged inner CV (methodology-audit finding #6)

The outer-fold purge above only protects the outer test period -- hyperparameter search
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
ablation (04 §13) call `run_walk_forward` directly and had been missed on the first pass
-- they still passed no `label_end_dates_all` at all (silently falling back to the
unpurged branch), which nbconvert's own exit code did not surface (the chained
`tail`/`grep` in this session's own run commands returns ITS exit code, not the
notebook's -- the same class of masking bug already documented earlier in this file).
Caught by grepping the notebook for every `run_walk_forward(` call site rather than
trusting the first passing run, and by reading the execution log directly for
`CellExecutionError` rather than a shell exit code.

No pipeline-numbers regression from this fix beyond what finding #5 already changed --
inner-CV purging affects which hyperparameters are SELECTED, not which rows are scored,
and at this N (~15-20 real training rows per fold) the purge rarely empties an inner
split. Re-ran the full chain (02 unaffected/skipped -- no feature_eng change this step --
04, 05, survival, train_final_models); Y3 c-index unchanged at 0.640 (this script has no
inner hyperparameter search); classifier AUCs unchanged; 95/95 tests pass.

### 2026-09-16: Y3 competing-risk censoring (methodology-audit finding #7)

Every recovery-day search previously ran the full 90-day window regardless of whether a
LATER qualifying disaster struck before day 90. An event scored as "recovered on day 47"
or "capped at 90" was, in several real cases, actually interrupted mid-observation by
the next disaster -- its true recovery status past that point is genuinely unknown, not
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
only 3 genuinely exhaust the 90-day cap -- down from the roughly 9-10 events the
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
  value) rather than force them into either class -- their true status by day 90 is
  unknowable under competing risks, not falsifiably negative.

**Effect on the AFT model** (n=40 pooled test points, same folds): c-index 0.640 ->
0.618 (still >0.5); RMSE 33.9 -> 20.2 and MAE 16.4 -> 9.4 (Y3 values themselves are now
correctly smaller for the 7 reclassified events); calibration Pearson r 0.036 -> 0.240;
reported censoring proportion 10.0% -> 15.0% (of which 5 of 6 pooled test-point censored
cases are now correctly attributed to a competing disaster, not the 90-day cap).
**Effect on C3_recovers_in_90**: full-refit AUC 0.618 -> 0.912 -- removing rows whose
true label was unknowable (rather than guessing) eliminated real label noise. 97/97
tests pass (2 new: `test_y3_censored_early_by_a_later_qualifying_disaster`,
`test_y3_genuine_recovery_before_a_later_disaster_is_not_censored`).

Not addressed by this fix: finding #12 (ordinary point-regressors -- Ridge/RF/XGBoost/
the hurdle model's stage-2 regressor -- still treat a censored Y3 value as if it were an
observed one; only the AFT model and the two classification labels above are now
censoring-aware) and the median-cut computation inside `label_slow_recovery`, which still
mixes genuine and censored durations when choosing its per-fold split point.

### 2026-09-16: missing-data semantics (methodology-audit findings #13/#14)

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

`src/inference.py`'s `build_feature_row` (the demo app's severity-override path) already
flips `deaths_available`/`homeless_available`/`mag_area_available`/`mag_wind_available`
to 1.0 when the user supplies that value; `financial_damage_observed` now gets the same
treatment when the user overrides financial damage.

**What this does not fix**: finding #14's other half (feature-specific missingness
logic -- train-fold median instead of a constant zero, for variables where zero has
real economic meaning) is not implemented; every affected column is still zero-filled,
just now with a flag alongside it rather than silently. That is a smaller, separable
change left for later if it is judged worth the added complexity.

**Effect on results**: feature count 64 -> 68 (4 new flags). Full pipeline re-run
(01 -- `financial_damage_observed` is built in `emdat_loader.py`, which notebook 01
caches -- through 04, 05, 08, survival, train_final_models). Classification AUCs
essentially unchanged (C2_volume_spike remains the only confirmed label). The Y3 AFT
model's c-index moved 0.618 -> 0.508 (still nominally above chance, but only barely) --
the new features shifted which columns per-fold RF-importance selection picks for this
already-small-N target, and one fold now scores below chance (0.333) where it
previously did not. Reported as measured, not as a regression to explain away: adding a
methodologically-correct feature does not guarantee a better fit at N~74, and this
document's standing rule is to report the number either direction. 97/97 tests pass.

### 2026-09-16: prediction-origin alignment (methodology-audit finding #8) -- Y1 rebaselined, EventWindow_0_5 and C4 consolidated

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
formula the pre-declared `Y1_EventWindow_0_5_LogReturn_Pct` already used -- so Y1 is now
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
`TARGET_LABEL_END_DATE_COL`, `LABEL_END_DATE_COL`), `src/inference.py`
(`TARGET_LABELS`, `LABEL_DESCRIPTIONS`, `target_bounds` -- also fixed two OTHER stale
Y1 descriptions this surfaced: "day-0 return"/"event day" language left over from
before this session's earlier Y1 redefinition), `src/evaluation/metrics.py` and
`src/evaluation/figures.py` (`TARGET_BOUNDS`/`TARGET_LABELS`, same stale-description
fix), `src/data_pipeline/sector_panel.py` (rename map), `notebooks/08_sector_panel.ipynb`
(classification cell), `app.py` (a caption naming C4 by name), and every test that
constructed a target/label fixture or asserted `LABELS`/`TARGET_LABELS` membership.

**Effect on results**: TARGET_COLS 5 -> 4, LABELS 6 -> 5. Y1's value itself changed
for every event (it now includes the day-0 move) -- verified numerically identical to
the pre-change `Y1_EventWindow_0_5_LogReturn_Pct` value for a sample event
(1.4839099892177814 either way). Y2/Y3/EventWindow_0_10 and the AFT survival model are
untouched (none of them read Y1). Full pipeline re-run (02 through train_final_models);
97/97 tests pass.

**What this does not do**: it does not touch feature construction (X is still
snapshotted at t-1, unchanged) or resolve whether the ex-post severity-informed
specification (the whole app/thesis's stated scope) is itself a "prediction" in the
forecasting sense -- that scope statement (README, `src/inference.py` docstring)
already exists and is unaffected by this fix.

### 2026-09-16: train-fold median imputation (methodology-audit finding #14's other half)

Finding #14 allows either "train-fold median or predefined constant + missing
indicator" -- the earlier fix this session (finding #13/#14, above) used the constant
(zero) half, with a flag. Asked directly why not the more defensible median half, the
answer was: it cannot be a single global median computed once, the same as any other
per-fold statistic in this pipeline (RF-importance selection, SMOGN, Ridge's alpha
search) -- a global median computed before the fold split leaks future rows' values
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
`dataset.parquet` was ever written -- median imputation is impossible after that point,
so the loader now leaves it (and its EM-DAT-provenance category, renamed
`"missing_median_imputed"`, was `"missing_zero_filled"`) as real NaN. Every other flagged
column (volume ratios, GARCH, macro) was already real NaN in `dataset.parquet`, just
blanket-zero-filled at each notebook's own `X = dataset[FEATURE_COLS].fillna(0.0)` --
those call sites now skip `MEDIAN_IMPUTE_COLS` in that blanket fill instead, leaving them
for the per-fold step.

**A latent bug this surfaced, fixed alongside it**: `log_damage_x_flood =
log_financial_damage * disaster_Flood` relies on multiplying by a 0/1 indicator to zero
out non-Flood events -- but `NaN * 0.0 == NaN` in float arithmetic, not `0.0`. While
`financial_damage` was always zero-filled (never NaN), this was silently safe; once it
can be real NaN, every non-Flood event with missing damage would have gone spuriously
NaN on a term that is definitionally 0 for non-Flood events. Fixed with an explicit
`np.where(disaster_Flood == 1, log_financial_damage, 0.0)` in both `feature_eng.py`
(historical dataset) and `src/inference.py`'s `build_feature_row` (which recomputes the
same term on every call, live-app included).

**Wired into every model-fitting consumer**, not just the primary walk-forward loop:
notebook 04 (main loop, the shared-MLP loop, the PCA ablation -- the dense/K10/feature-
block ablations inherit it automatically through the shared `run_walk_forward`),
notebook 05 (the classification loop and the hurdle-model cell), `run_survival_model.py`.
Two paths have no fold at all because they refit on ALL real data by design (already
documented elsewhere as in-sample, not held-out) -- notebook 04's final SHAP refit and
`scripts/train_final_models.py` -- so they use one GLOBAL median instead, computed once
in notebook 02 and persisted in `feature_spec.json` (`MEDIAN_IMPUTE_VALUES`) precisely so
every "refit on everything" consumer, including `src/inference.py`'s live demo, imputes
with the identical numbers rather than each recomputing its own. Notebook 07's SHAP
values now use that same global fill too -- explaining a model on differently-imputed
inputs than it was trained on would have been a real, if quiet, inconsistency.
`notebooks/08_sector_panel.ipynb`'s classification cell still blanket-zero-fills (its
own `PANEL_FEATURES` construction, unchanged) -- noted as a residual gap, not silently
left inconsistent.

**A second bug caught only by running it**: `scripts/run_survival_model.py`'s
calibration section calls `pd.qcut(..., q=3, labels=[...], duplicates="drop")`, which
raises when tied predicted values collapse the bin count below 3 and pandas then rejects
the fixed 3-label list. This did not fire before (the previous prediction distribution
never had that many ties) but did after this fix changed the model's inputs -- caught
by actually running the script rather than assuming a passing test suite meant the
scripts were fine too. Fixed with a fallback to pandas' own integer bin labels when the
named ones don't fit.

**Effect on results**: Y3 AFT c-index 0.508 -> **0.553** (a real improvement, not
guaranteed by the fix -- reported either direction per this document's standing rule).
More notably, **`C1_negative_return` now clears both the majority rule and chance**
(full-refit AUC 0.711, `beats_baseline=True`) -- a second confirmed classification
result alongside `C2_volume_spike`, where before this fix only one label cleared the
bar. 97/97 tests pass (two updated: the round-trip test now expects the global median
for a missing feature, not 0.0; the classification test now expects `C1_negative_return`
confirmed).
