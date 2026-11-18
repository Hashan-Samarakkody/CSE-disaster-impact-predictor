# Notebook 02, Feature Engineering and Target Construction

Documents `notebooks/02_features_targets.ipynb`.

## Purpose

Build the single event level table that every modelling stage reads: one row per qualifying
disaster, holding the features that were knowable before the market reacted and the three
research targets that describe what it did afterwards. It also fixes the chronological
folds, once, so that every later stage scores exactly the same events.

## Inputs

`market.parquet`, `disasters.parquet`, `macro.parquet` and `sp500.parquet` from stage 01,
plus the cached external downloads.

## Main processing stages

Section 2.2 builds the market features and tests them for stationarity. Section 2.3 builds
the disaster features and applies the inclusion filter. Section 2.4 determines which events
the market series can actually carry. Section 2.5 constructs the three targets. Section 2.6
assembles the event level table. Section 2.7 creates the walk forward folds and illustrates
the oversampling. Section 2.8 caches everything. Section 2.9 states what was produced.

## Important functions called

| Function | Module | What it does |
|---|---|---|
| `FeatureEngineer.engineer_market_features` | `src/features/feature_engineering.py` | lagged returns, price to moving average ratios, rolling volatility, conditional volatility, the pre event volume block |
| `FeatureEngineer.engineer_disaster_features` | `src/features/feature_engineering.py` | severity transforms, the inclusion filter, rare type pooling, one hot encoding |
| `build_event_targets` | `src/targets/event_targets.py` | the three targets and their label settlement dates |
| `calculate_aspi_forward_log_return` | `src/targets/event_targets.py` | target one, and the ten, fifteen and twenty session sensitivity variants |
| `calculate_forward_abnormal_volume` | `src/targets/event_targets.py` | target two, the log volume ratio and its two constituent means |
| `calculate_recovery_time` | `src/targets/event_targets.py` | target three, with the drawdown gate, the ninety session cap and competing event censoring |
| `generate_walk_forward_splits` | `src/training/walk_forward.py` | the chronological folds |

## Important outputs

`dataset.parquet`, the modelling table. `market_feats.parquet`, the daily feature series.
`in_scope.parquet`, the events that survived the filters. `feature_spec.json`, the frozen
feature list and the global impute values. `splits.pkl`, the fold definitions.

## Relationship with other notebooks

Reads stage 01. Feeds stage 03 for description, stages 04 and 05 for fitting, and stage 08
for the sector panel.

## Methodological decisions made here

1. **Every feature is snapshotted before the event.** Rolling windows are shifted by one
   session before being computed, so a twenty day moving average at the event never
   includes the event day itself. This is a look ahead bug that was found in review and
   fixed, and the shift is now part of the definition.
2. **Non stationary columns are excluded by test, not by opinion.** Every engineered price
   column is tested for a unit root before admission. Raw moving average levels run from
   about 574 to over 10,000 across the sample, so a later chronological fold sits outside
   any training range. Their price relative ratios carry the same momentum information with
   no trend in the level, and those are kept instead.
3. **The admissibility test runs on development period rows only.** The decision about
   which columns may enter is itself a decision that must not see any row inside any test
   fold, so the test is restricted to sessions before the first fold's own test period.
4. **The targets are baselined on the pre event close.** Every feature is known as of the
   session before the event, so a target measured from the event day close would have been
   inconsistent with what the model is given to predict from. Baselining on the previous
   close makes the day zero reaction part of what the target measures.
5. **The recovery clock is gated on a real drawdown, and the scan starts at the first
   session.** If the index never falls below its pre event level inside the five session
   gate window the event is genuinely resilient: duration zero, no recovery event
   observed, censor reason `no_drawdown`. If it does fall, the scan runs forward from the
   first session after the origin and stops at the first close that regains the baseline
   after an earlier dip. Anchoring the scan on the trough instead would skip a genuine
   recovery that precedes a later, deeper dip.
6. **A later disaster censors an earlier recovery.** If another qualifying disaster arrives
   before recovery, the observation is right censored at that date with the reason recorded,
   rather than pretending recovery occurred then.
7. **Folds are created once and cached.** Every later stage loads the same `splits.pkl`, so
   no stage can quietly score a different set of events.

## Justification

The chronological discipline follows standard practice for financial event studies: with
overlapping outcome windows, an ordinary shuffled cross validation lets a training row's
outcome be measured from prices that a test row has already seen. Testing stationarity
before admission follows the same logic as any time series regression, since a regressor
with a trend in its level cannot be extrapolated to a later period. Treating the recovery
target as right censored is the standard survival analysis treatment of a duration that the
follow up window did not reach, and the reason stage 05 and the recovery grid fit it under
a censored likelihood rather than a squared error loss.

## Execution requirements

Stage 01 must have run. This stage is fast, a minute or two, and it is the one to re run
first after any change to feature or target code.
