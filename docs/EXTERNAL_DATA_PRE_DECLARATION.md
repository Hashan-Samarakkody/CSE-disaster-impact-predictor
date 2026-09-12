# Pre-declaration: external data sources and derived features

**Written before any model is fitted on these features.** Standing integrity constraint 2
forbids justifying a feature, threshold or window by held-out performance. Everything in
this document is fixed here, in advance, with its reason. Whatever the paired bootstrap
says afterwards is reported unchanged — including "no improvement", and including
"worse".

Date written: 2026-09-11. Author: pipeline maintenance pass following the
"what data are we missing?" audit.

---

## 1. Why this exists

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

## 2. Sources

All five were reachability-tested before adoption. Endpoint, response code and row count
are recorded so the retrieval is reproducible.

| # | Source | Endpoint | Verified | Licence/status |
|---|---|---|---|---|
| 1 | DesInventar Sendai (UNDRR/UNDP, national DMC database) | `desinventar.net/DesInventar/download/DI_export_lka.zip` | HTTP 200, 18.5 MB zip → 1.21 GB XML, 130,140 records | Open, UN-hosted |
| 2 | NASA POWER (NASA Langley Research Center) | `power.larc.nasa.gov/api/temporal/daily/point` | HTTP 200, 7 points × 10,106 days | Open, no key |
| 3 | FRED (Federal Reserve Bank of St. Louis), series `DEXSLUS` | `fred.stlouisfed.org/graph/fredgraph.csv?id=DEXSLUS` | HTTP 200, 14,005 rows | Open, no key |
| 4 | countryeconomy.com ASPI daily | `countryeconomy.com/stock-exchange/sri-lanka?dr=YYYY-MM` | HTTP 200, 21 rows/month, 2018→2026 | Public page |
| 5 | Wikidata SPARQL | `query.wikidata.org/sparql` | HTTP 200, 28 elections 2000–2026 | CC0 |

### Rejected, with reason

| Source | Reason |
|---|---|
| GDELT news tone | Persistent HTTP 429 including at 20 s spacing. No news-sentiment feature is built; thesis §3.3.1's sentiment claim stays unsupported and is amended instead. |
| ReliefWeb API v2 | HTTP 403 — requires a registered `appname`. |
| CBSL monthly CCPI | Published only as per-month PDF press releases; no CSV/XLS endpoint. |
| CBSL daily FX form | JS-gated; POST returns the unchanged page shell. Superseded by FRED `DEXSLUS`. |
| IMF IFS API | `dataservices.imf.org` unreachable (HTTP 000). |
| FRED monthly LKA CPI | No live series: `CPALTT01LKM657N`, `CPALTT01LKM659N`, `LKACPIALLMINMEI`, `LKAPCPIPCH`, `CPALTT01LKM661N`, `LKACPGRLE01GPM`, `SLKCPIALLMINMEI` all return the 404 page. |
| Stooq | JS-gated noscript shell. |
| CSE daily PDFs | Reachable (`cdn.cse.lk/cmt/<path>`, 2.7 MB each) but the index API returns only the 5 most recent; ~800 files ≈ 2.2 GB to reach 2023-07. Not adopted. |
| CSE index/sector history API | 5 endpoint names tested, all HTTP 404. |

---

## 3. Declared features

**48 features on 64 events is p > n before selection.** These blocks go through the
identical per-fold feature-selection and collinearity rule as the existing 31 — they are
not exempted, and the selection is fitted on training rows only, as before.

### Block A — hazard intensity, from NASA POWER (6 features)

Measured by instrument, not assessed by a reporter. No missingness, no reporting bias,
and available on the event day itself — so this block is also admissible in the ex-ante
Model A specification (audit E14), which `financial_damage` is not.

Seven district points: Colombo (6.93 N, 79.86 E), Jaffna (9.66, 80.02), Batticaloa
(7.72, 81.70), Nuwara Eliya (6.97, 80.77), Galle (6.05, 80.22), Anuradhapura (8.31,
80.40), Ratnapura (6.68, 80.40). Chosen for island coverage — one per major
climatic/administrative region — before any correlation with a target was computed.

Let `P_d(t)` = `PRECTOTCORR` at district `d` on day `t` (mm/day), and
`S_d = Σ P_d(t-2..t)` the 3-day accumulation ending on the event day.

| Feature | Definition | A-priori justification |
|---|---|---|
| `hz_precip_max3d` | `max_d S_d` | 3-day accumulation is the standard flood-generating window; the max over districts because flooding is localised and an island mean would dilute it. |
| `hz_precip_mean3d` | `mean_d S_d` | Island-wide intensity — the aggregate analogue, matching the index-level target. |
| `hz_precip_spread3d` | `std_d S_d` | Distinguishes one drowned district from island-wide rain at equal mean. |
| `hz_districts_wet` | `#{d : S_d > 50}` | 50 mm/3 days is the Sri Lanka Department of Meteorology heavy-rain advisory level. A published operational threshold, not a swept one. |
| `hz_wind_max3d` | `max_d max(WS10M_MAX(t-2..t))` | Storm-intensity analogue for the 9 Storm events, where rainfall is not the damaging mechanism. |
| `hz_precip_anom` | `hz_precip_mean3d / (3 · mean_d mean P_d(t-33..t-3)) - 1` | Intensity against the location's own recent baseline. Deliberately the same functional form as Y2 (`V/V̄₃₀ − 1`), so the feature is the target's construction applied to rainfall. The 30-day baseline ends at `t-3` so it never overlaps the event window. |

### Block B — physical severity, from DesInventar (6 features + 1 flag)

Match window **[t − 7, t + 14]** around the EM-DAT event date. EM-DAT dates a multi-day
event by onset; DesInventar records by district report date, which lags by days to
weeks. The window is fixed here and is not varied afterwards.

Coverage measured before declaring: **52 / 64 events matched**, and **35 of the 47
zero-filled-damage events** gain real severity. Median matched event: 14 districts,
125,035 affected, 76 houses destroyed. DesInventar ends **2020-12-20**, so 9 events
(2021–2022) match nothing — `di_available` carries that.

| Feature | Definition | Justification |
|---|---|---|
| `di_districts_hit` | distinct districts with ≥1 record in window | Geographic breadth of the shock — the exposure dimension the index-level design otherwise cannot see. |
| `di_affected_log` | `log1p(Σ afectados)` | Headcount severity. Log because the raw range spans 5 to ~1e6. |
| `di_houses_destroyed_log` | `log1p(Σ vivdest)` | Capital destruction — the closest physical proxy to the missing monetary damage. |
| `di_houses_damaged_log` | `log1p(Σ vivafec)` | Partial-loss counterpart. |
| `di_deaths_log` | `log1p(Σ muertos)` | Human severity; also the variable most consistently recorded. |
| `di_records` | record count in window | Reporting intensity. Declared explicitly as a *reporting* measure, not a severity one, so that if it dominates a model the interpretation is stated rather than discovered. |
| `di_available` | 1 if matched else 0 | Missingness indicator (audit P2-7). Mandatory: without it a zero reads as "no damage" rather than "outside DesInventar's coverage". |

**Stated in advance:** DesInventar Sri Lanka populates `valorus` (USD loss) in **0** of
130,140 records and `valorloc` (LKR) in **16**. It carries no monetary loss. This block
is a *physical* severity substitute and must never be described as recovering the
missing damage figures.

### Block C — daily macro, from FRED `DEXSLUS` (3 features)

Replaces an annual macro series matched to day-0 events. All three are computed strictly
from data at or before **t − 1**, the same pre-shock boundary the ASPI features use,
because the exchange rate is a market price and same-day use would leak.

| Feature | Definition |
|---|---|
| `fx_logret_1` | `ln(FX(t-1) / FX(t-2))` |
| `fx_logret_5` | `ln(FX(t-1) / FX(t-6))` |
| `fx_vol_30` | `std` of daily FX log returns over `[t-31, t-1]` |

Ratios and differences only — stationary by construction, and the existing ADF/KPSS gate
in stage 02 applies to them unchanged.

### Block D — confounder control, from Wikidata (2 features)

28 national elections, 2000-10-10 → 2026-08-31. Motivated by a specific known
contamination: the **2005-11-17 presidential election falls 4 days before the
2005-11-21 event**, which carries the largest observed Y1 drop (−0.0753).

| Feature | Definition |
|---|---|
| `days_to_election` | signed days to the nearest national election (negative = election already held) |
| `election_within_5d` | `1` if `abs(days_to_election) <= 5` |

±5 days matches the event-window contamination screen already specified in the audit
(§39 item 7). Fixed there before this pass.

### Block E — sample extension, from countryeconomy (no feature)

The ASPI series ends **2023-06-28** in the local archive. countryeconomy supplies daily
closes to 2026-09-10, verified continuous: its 2023-06-28 close of **9,442.95** matches
the archive's final row exactly.

This admits the **10 qualifying EM-DAT events after 2023-06-28** — 2023-07-04,
2023-09-28, 2023-11-01, 2023-12-01, 2024-01-01, 2024-05-15, 2024-10-11, 2024-11-25, and
Storm Ditwah on 2025-11-20 and 2025-11-27 — taking **N from 64 toward 74**, subject to
each event still passing the pre-registered ≥1000-affected filter.

**This is the single largest change in the pass**, because it adds real out-of-fold test
points rather than features. The inclusion filter is unchanged; only the market series
is longer.

**Declared limitation:** countryeconomy publishes the index level, not volume, and no
practical volume source was found for the post-2023 period. The new events therefore
carry **Y1 and Y3 but not Y2**. Y2's effective N stays at 61 and the new events are
NaN-masked out of it, exactly as the 2000 volume gap is already handled.

---

## 4. What is expected, stated before the run

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

## 5. What this does not do

- It does not recover monetary damage. Effective N for money-terms severity stays 17.
- It does not fill 2021–2022 severity (DesInventar ends 2020-12-20) — Block A does cover
  those events, which is part of why Block A exists.
- It does not add per-sector volume, so the sector panel remains Y1/Y3 only.
- It does not add news sentiment. Thesis §3.3.1 is amended rather than satisfied.
- It does not route around the refusal to target R² ≥ 0.65. No threshold, window, model
  or feature below was chosen to reach a number.

---

# 6. Ablation outcome, measured (run 2026-09-11 19:07)

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

## Where the pre-declaration was wrong

Section 4 committed five expectations in advance. Scored honestly:

| # | Expectation | Outcome |
|---|---|---|
| 1 | Sample extension helps most | **Partly right.** It delivered 40 test points instead of 30 and tightened every interval, but it is not separable in this ablation — it changes the rows, not the columns. |
| 2 | Hazard helps Y2/Y3 more than Y1 | **Right in direction, not significant.** Y2 ridge +0.0678, Y1 negative on all three models. No CI excludes zero. |
| 3 | DesInventar acts through `di_available`/`di_districts_hit` | **Not supported.** The whole block is indistinguishable from zero on every target and model; it is *negative* on Y2 for two of three. |
| 4 | **FX "expected to do little on its own"** | **WRONG, and wrong in the useful direction.** FX is the single strongest block measured: +0.1315 on Y2/xgboost and +0.0437 on Y2/random forest, both CIs excluding zero. It was adopted as a correctness fix and turned out to be the only block that independently earns its place. |
| 5 | Y1 still unpredictable | **Right.** No block helps Y1 on any model; 0 of 15 Y1 combinations are significant. |

Expectation 4 is recorded as a failed prediction rather than quietly rewritten. That is
the entire point of writing section 4 before the run: had the expectations been written
afterwards, "daily exchange-rate dynamics predict abnormal trading volume" would read as
a designed finding instead of a surprise.

## The distinction that must not be blurred

The external data **significantly improves the model relative to a model without it**.
It does **not** make Y2 beat a naive baseline: the best Y2 model (ensemble) still sits at
ΔRMSE +0.0651, CI [−0.0101, +0.1333] against the constant-zero null.

Both statements are true simultaneously and must be reported together. "Adding exchange
rate data measurably improves abnormal-volume prediction" is supportable. "Abnormal
volume is predictable" is not.

---

# 7. Second pre-declaration: sample widening and target reformulation (2026-09-12)

**Written before any model is fitted on these.** Same rule as sections 1-5: every
definition, window and threshold below is fixed here, in advance. Whatever the paired
bootstrap says afterwards is reported unchanged.

Prompted by the author's instruction to (a) use the 1986-2025 EM-DAT export, (b) lower the
inclusion threshold to 700 affected, (c) use any available technique, and (d) reach >0.6 on
accuracy/predictability for every target.

## 7.1 What the instruction can and cannot buy

Stated plainly, before the run, so no outcome can be spun afterwards:

| Instruction | Measured consequence |
|---|---|
| EM-DAT 1986-2025 (110 records, was 86) | **+0 modellable events.** The 24 extra records are all pre-2000. ASPI starts 2000-01-03 and a day-0 event study needs the index on the event day plus 90 trading days after it. 6 of the 20 qualifying pre-2000 events additionally have no Start Day at all. They are loaded and then dropped by the scope filter, with the count printed. |
| Threshold 1000 -> 700 affected | **+2 events** (2021-05-16 Storm, 2023-04-24 Storm). N 74 -> 76. Cannot rescue a result; recorded as a deviation from the thesis pre-registration, not as the original design. |
| "Simulate disasters" | Already implemented as SMOGN inside training folds only, and already **measured to make every model worse on every target**. Synthetic rows in TEST would void every reported metric. No change. |
| R2 > 0.6 on Y1/Y2/Y3 | **Not reachable.** Best in study is +0.199 (Y2). R2 0.6 on a day-0 index return means explaining 60% of market variance from disaster covariates, which contradicts the efficient-markets result the thesis itself argues. No technique below targets it. |
| Accuracy / AUC / F1 > 0.6 | **Reachable, and already reached on one target** (`C2_volume_spike` random forest: AUC 0.764 [0.578, 0.929], balanced accuracy 0.629). Sections 7.2-7.3 are the honest attempt to extend that to the other targets. |

## 7.2 New targets: cumulative event-window returns

Y1 is a single day's log return -- the noisiest possible measurement of an event's market
effect. Standard event-study practice accumulates over a window, which raises
signal-to-noise without adding any information the study does not already have.

Let `pos` be the first trading row on or after the event date (the existing anchoring in
`build_targets`), and `P` the ASPI close.

| Target | Definition | Justification |
|---|---|---|
| `Y1_car_5` | `ln(P[pos+5] / P[pos-1])` | Cumulative return over the event day plus 5 trading days. The conventional short event window. Verified: **76/76 events have >=5 trading rows after `pos`.** |
| `Y1_car_10` | `ln(P[pos+10] / P[pos-1])` | Two-week window, the other conventional choice. Verified: **76/76 events have >=10 trading rows after `pos`.** |

Both windows were fixed at 5 and 10 because those are the standard short-horizon event
windows in the literature, not because either scored better. No other k is tried, and
neither may be swapped for the other after seeing a result.

**A reformulation considered and rejected before any fit:** market-model abnormal returns
(regress ASPI on a benchmark over a pre-event estimation window, take the residual).
`docs/METHODOLOGY_AUDIT.md` already rules this out on identification grounds -- the asset
here **is** the market index, so there is no valid benchmark to regress it against, and
`notebooks/09_synthesis.ipynb` records the same decision as a resolved audit item. It is
not revived here. Using the S&P 500 as the benchmark would additionally be invalid because
Colombo closes roughly ten hours before New York opens, so a same-date S&P return is not
observable to a CSE participant on the event day.

## 7.3 Reformulated classification labels

`C3_recovers_in_90` is broken as a classification target and this is a defect, not a
result: prevalence is **0.90**, so the always-predict-majority rule scores 0.900 accuracy
while every model scores *below chance* on AUC (best 0.597, random forest 0.229). An
"accuracy" of 0.9 there measures the class imbalance, not the model.

| Label | Definition | Prevalence | Justification |
|---|---|---|---|
| `C3b_slow_recovery` | `Y3 > median(Y3 over THIS fold's training rows)` | ~0.50 by construction | Median split makes accuracy and AUC informative instead of gameable. The cut is computed on training rows only -- same discipline as the existing `C1b_adverse_move` tercile rule -- so no test information reaches the label. |
| `C4_car5_negative` | `Y1_car_5 < 0` | measured, ~0.5 expected | Direction over the 5-day window rather than the single noisiest day. Same sign question as `C1_negative_return`, asked of a less noisy measurement. |

`C1_negative_return`, `C1b_adverse_move` and `C2_volume_spike` are unchanged.
`C3_recovers_in_90` is **retained and still reported** beside `C3b_slow_recovery`, because
dropping it after seeing that it fails would be exactly the selection this document exists
to prevent.

## 7.4 Expected outcomes, fixed in advance

1. **`Y1_car_5` / `Y1_car_10` will still not beat their nulls as regressions.** Widening
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

## 7.5 Sector-panel classification

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
stronger negative result than the index-level one alone -- it closes off the explanation
the thesis currently offers for its own null finding.

## 7.6 Success criterion, decided by the author 2026-09-12

**Beat the majority rule, not raw accuracy.**

Raw accuracy is not a criterion. Three labels already exceed 0.6 on it and it means
nothing on an imbalanced target: `C3_recovers_in_90` scores 0.900 accuracy because
prevalence is 0.900, and its models score *below* chance on AUC.

The reported criterion is `beats_baseline` (`src/models/classifiers.py`): balanced
accuracy > 0.5 **and** both AUC intervals excluding 0.5. It currently holds for **1 of 16**
(label, model) pairs. Raw accuracy is still reported, always beside its prevalence and its
majority-rule accuracy on the same row, so a reader can see which numbers are real.
