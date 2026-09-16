# Features and targets (stage 02)

Stage `02_features_targets.ipynb` turns the four cached tables into one event-level
modelling table: **one row per qualifying disaster**, with everything the model may
know before the event on the left and the outcomes to predict on the right.

Output artifacts: `dataset.parquet`, `market_feats.parquet`, `in_scope.parquet`,
`feature_spec.json`, `splits.pkl`.

## 1. Which events qualify

Three filters run in order, and each prints what it removed:

1. **Type.** Epidemics and other biological disasters are excluded
   (`EXCLUDED_DISASTER_TYPES`) — a pandemic is not a dated physical shock to a
   district.
2. **Severity.** `population_affected >= min_affected`. The thesis pre-registered
   `1000`; it was lowered to **700** on the author's instruction, which admits two
   further Storm events. The threshold lives in `FeatureEngineeringConfig` rather than
   at a call site, so the deviation from the pre-registration is visible in the config
   every stage reads.
3. **Coverage.** The event must have an ASPI observation on its day and 90 trading
   days after it. Events outside the market series are loaded, counted, and dropped —
   they are reported as an explicit limitation, not silently absent.

What survives is **N = 76 events**, which is the number every sample-size caveat in
this project refers to.

## 2. Market features — the one-day rule

`FeatureEngineer.engineer_market_features` builds lagged returns, moving averages, a
30-day volatility "panic proxy" and a pre-event volume block. Every one of them obeys
the same rule: **a feature attached to day `t` may only use data up to day `t-1`.**

Mechanically, the price series is shifted by one day *before* rolling, not after. This
is easy to get subtly wrong: a rolling mean that includes day `t` and is then shifted
is not the same thing as a rolling mean of the already-shifted series, and only the
second is honest.

Two design decisions are worth naming.

**Levels are replaced by ratios.** The raw ASPI runs from about 574 in 2000 to over
10,000 by 2022. Under a chronological split, a later test fold therefore occupies a
feature range disjoint from anything in training: tree models pin at the training
boundary and linear models extrapolate off the end. The price-relative ratio
(`price / sma_w`) carries the same momentum information with no trend in the level.

**Y2 gets its own driving series.** The pre-event volume block exists because the
feature table originally contained no volume-derived column at all — the one target on
which a model beats its baselines was being predicted with no information about
volume. These features are not the product of a search: Y2 is *defined* as
`V_t / mean(V_{t-30..t-1}) - 1`, so the same functional evaluated one trading day
earlier is its natural autoregressive predictor.

## 3. Disaster features

Severity (`population_affected`, `financial_damage`, deaths, homeless), the hazard
magnitude columns, a one-hot disaster type, and derived terms: `log1p` transforms of
the two heavy-tailed severity measures, `damage_to_gdp`, and a
`log_damage x flood` interaction.

Disaster types with too few events to support their own indicator are pooled, because
a one-hot column with a single positive row is a row identifier, not a feature.

## 4. Targets

Five, all built by `FeatureEngineer.build_targets`. Let `pos` be the event's position
in the trading-day index and `P` the ASPI close.

| Target | Definition | Notes |
|---|---|---|
| `Y1_ASPI_5D_Forward_LogReturn_Pct` | `ln(P[pos] / P[pos-1])` | day-0 response; the noisiest possible measurement |
| `Y2_abnormal_volume` | `V[pos] / mean(V[pos-30..pos-1]) - 1` | `NaN` where volume was never recorded |
| `Y3_recovery_days` | trading days until `P` regains its pre-event level, **capped at 90** | censored by construction |
| `Y1_EventWindow_0_5_LogReturn_Pct` | `ln(P[pos+5] / P[pos-1])` | cumulative 5-day event window |
| `Y1_EventWindow_0_10_LogReturn_Pct` | `ln(P[pos+10] / P[pos-1])` | cumulative 10-day event window |

The two cumulative returns follow standard event-study practice: accumulating over a
window raises signal-to-noise relative to a single day. The windows 5 and 10 are the
conventional short ones and were fixed in the pre-declaration **before** anything
scored them; neither may be swapped for the other afterwards.

Where the series runs out, a target is `NaN` rather than a truncated window, so a
partial accumulation is never reported as a full one.

Y3 is capped, which makes it censored rather than merely bounded: roughly one event in
ten sits exactly on 90. That point mass is why Y3 gets its own two-stage model — see
[`modeling.md`](modeling.md).

## 5. Missing values have three states, not two

This is the study's central data problem, and the pipeline keeps it visible end to end:

- **observed** — a real measurement;
- **zero-filled** — the variable was not measured, and the loader wrote `0.0` so the
  row would survive;
- **missing** — `NaN`.

Only 18 of 76 events have a real `financial_damage` figure. A conventional missingness
check would report that column as 100% complete, because it contains no `NaN` at all.
A model reads those zeros as *no damage*. Every stage that touches damage therefore
also reads `damage_source`, and the exploratory figures in stage 03 draw the three
states in three different colours.

## 6. Chronological splits

`generate_walk_forward_splits(n, train_window=30, test_window=10, step=10)` produces
rolling windows in which training rows always precede test rows. At N = 76 that is
**4 folds and 40 pooled out-of-fold predictions** (fewer for Y2, which loses events to
missing volume).

The splits are computed on the **real** event count. Computing them on the
post-oversampling length was a live bug: synthetic rows sit at the tail of the
augmented table, so a boundary cut on the inflated length could place fabricated rows
inside a "held-out" fold and score the model against its own synthetic output. The
notebook asserts `train_index.max() < test_index.min()` on every fold.

There is **no k-fold anywhere in this project**. Shuffled k-fold would train on later
events to validate earlier ones, which for a time-ordered market series is
look-ahead — the thesis bans it in section 3.7.1, and
`tests/test_integrity_invariants.py` enforces it.

## 7. Synthetic oversampling, and where it is allowed

`src/sampling/time_aware_smogn.py` implements SMOGN-style oversampling for the rare
high-severity events. **Synthetic rows may exist only inside a training fold. A test
fold is 100% real, always.** A single synthetic row in a test fold would void every
metric in the study.

The implementation is neighbour-pair interpolation, not noise added to a single point:
for each minority row it finds the nearest same-type neighbour inside a bounded time
window and synthesises a row on the segment between two real events. Three constraints
keep synthetic rows physically possible:

1. **Categorical integrity** — the one-hot disaster-type block is copied verbatim from
   a parent, never blended. An earlier version produced rows that were 60% Flood and
   40% Storm.
2. **Derived-feature consistency** — columns that are functions of other columns (the
   log terms, `damage_to_gdp`, the interaction) are *recomputed* from the interpolated
   parents. Interpolating a child independently of its parent breaks the identity
   between them on every synthetic row, teaching a relationship that holds nowhere in
   the real data.
3. **Same-type pairing** — a flood is interpolated only with another flood. A
   flood-drought blend is not an approximation, it is a physical impossibility. A
   hazard type with a single event in a fold produces no synthetic rows at all, and
   deliberately does not fall back to adding noise to that lone point.

Interpolation happens in raw units, because that is what EM-DAT measures and what
`damage_to_gdp` needs as a numerator; the log terms are derived afterwards.

`max_synthetic_share` is a pre-registered ceiling on the synthetic fraction, and
exceeding it **raises** rather than truncating — over-augmentation fails loudly
instead of degrading quality quietly.

One residual limitation is known and unfixed: `rolling_std_*` columns are convex
functionals of the price path, so a convex combination of two rows overstates
volatility slightly relative to the same functional applied to the combined path. It
cannot be corrected without the underlying price series.

Measured outcome: oversampling made every model worse on every target. That result is
reported rather than the setting being quietly dropped.

## 8. Collinearity, decided by a rule fixed in advance

`src/evaluation/collinearity.py` computes the correlation matrix, VIFs and condition
indices, and `redundant_drop_set` applies the pre-declared rule: within any group whose
pairwise |rho| exceeds the threshold, keep the most primitive member (lowest derivation
depth, ties broken alphabetically) and drop the rest.

The rule reads no target, no fold and no score, which is precisely what makes applying
it something other than selection on the test set. Whatever it does to the metrics is
reported either way.
