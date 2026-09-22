# Testing

What the test suite checks, why each group exists, and how to run it.

```bash
python -m pytest tests/ -q
```

The suite needs no network access. Most of it needs no data file either: the tests build
small synthetic frames and assert on behaviour. The exceptions are the frozen target tests
and the experiment artifact tests, which compare the live cache against a recorded
snapshot and skip with a clear message when that cache is absent.

## 1. Why these tests exist

A research repository fails in two ways that ordinary software does not. It can keep
running while quietly computing the wrong number, and it can produce a different number
tomorrow than it did today. The suite is aimed at both.

Every test below either pins a definition that the thesis quotes, or pins a property the
statistics depend on. There are no tests written to raise the count.

## 2. The groups

### 2.1 Target construction

`tests/test_feature_engineering.py` checks the three research targets against hand
computed values on synthetic price series. It covers the alignment rule, that an event on
a non trading day maps to the next session, that horizons are counted in trading sessions
rather than calendar days, that a window which runs off the end of the series yields a
missing value rather than a truncated one, and that the recovery clock respects both the
adverse response gate and the competing event cap.

`tests/test_return_horizons.py` covers the four pre declared return horizons. It asserts
that the horizon column names are the ones the frozen protocol defines, that every one of
them exists in the built dataset, and that their values follow the protocol alignment when
recomputed directly from the raw market series rather than from the code that wrote them.

It also asserts that the market and disaster information sets partition every feature
column with no overlap, and that no severity or hazard column has leaked into the market
set. Without that check the phrase "market only baseline" could silently change meaning
between runs.

### 2.2 The frozen volume target

`tests/test_volume_target_frozen.py` is the most important file in the suite. Forward
abnormal volume is the study's one statistically supported continuous result, and nothing
done to the other two targets is allowed to move it.

Twenty assertions compare the live artifacts against
`artifacts/results/frozen_baseline.json`:

1. all seventy four target values
2. the event sample and its dates, in order
3. every fold's training and test index arrays
4. pooled out of fold predictions and true values, for all eleven models
5. per fold RMSE, MAE and R squared, for every model
6. pooled R squared per model
7. the bootstrap verdict rows, including the interval bounds, the one sided p value, the
   verdict string and the Holm flag
8. the classification arm: AUC, balanced accuracy, the AUC interval, Matthews correlation
   and the precision recall AUC

Tolerance is 1e-9 absolute on a target whose own scale is about 0.6. That is float noise,
not a materiality threshold.

The snapshot was re taken on 2026-09-21, at commit `f076bd9`, after the target definitions
were re frozen in `docs/TARGET_DEFINITION_PROTOCOL.md` on 2026-09-19. The previous
snapshot, taken at commit `1fbf6275`, pinned the earlier volume definition, a ratio minus
one over the event day window, which the protocol replaced with a log ratio over the five
sessions after the prediction origin. Those assertions could not pass and were not meant
to: the definition they guarded no longer exists. The re freeze is recorded here and in
`docs/refactor_validation.md` rather than done silently.

A failure here is a stop and diagnose signal, not a number to re freeze. If one of these
ever fails, find the cause, write it down in `docs/audit.md`, and only then consider
regenerating the baseline. Regenerating it is justified only by a deliberate, pre declared
change to the target definition itself.

### 2.3 Leakage and chronology

`tests/test_integrity_invariants.py` pins the properties the validation design rests on:
that walk forward splits never place a test event before a training event, that the purge
removes exactly the training rows whose label reaches into the test period, that the
pre declared redundancy threshold is applied as written, and that the sector panel's event
grouped folds keep every sector of one event on the same side of the split.

### 2.4 Statistics

`tests/test_survival_metrics.py` checks that each censoring aware metric behaves
differently from its censoring blind counterpart, which is the whole reason for adopting
them. The sharpest test corrupts a censored row's prediction by a thousand days and asserts
that the uncensored point error does not move.

`tests/test_classifiers.py` covers the six label definitions, the metric calculations and
the two stage hurdle model, including its degenerate fold fallback. It also checks that the
stage one drawdown label is defined for every event while the two duration labels drop the
events that never fell, which is what the protocol requires of a conditional stage.

### 2.5 Experiment artifacts

`tests/test_experiment_artifacts.py` asserts the properties the improvement grid's
statistics depend on. Chiefly, that every configuration and every baseline scored exactly
the same test events, because a paired bootstrap over unpaired vectors is meaningless. It
also checks that predicted recovery probabilities form a valid cumulative distribution,
non decreasing in time and inside the unit interval, and that the verdict columns agree
with the intervals they were derived from.

### 2.6 Figures and the demo

`tests/test_result_figures.py` and `tests/test_eda_figures.py` render every figure function
on a small synthetic frame and assert that a file lands on disk. The point is not pixel
correctness. It is that a figure function cannot silently stop producing output.

`tests/test_inference.py` checks the demo bundle the Streamlit app loads, including that it
refuses to start when the fitted models are absent rather than showing stale numbers.

`tests/test_external_sources.py` covers the external loaders against recorded fixtures, so
the suite never depends on a live API.

`tests/test_time_aware_smogn.py` asserts the augmentation constraints: synthetic rows are
drawn only from training rows, only within the declared time window, and never exceed the
pre registered share.

### 2.7 Leakage, promoted out of the notebooks

`tests/test_no_leakage.py` runs the guard that used to live only in a notebook cell, so it
now fires under pytest without anyone opening a notebook. It asserts that the persisted
feature specification shares no column with the declared non feature set, that every
feature is dated at or before the prediction origin, and that the label end purge really
does drop a training event whose horizon reaches into the test period.

Two of its tests deliberately reintroduce the defect they guard against, confirm that the
guard raises, and leave the correct state behind. A guard nobody has seen fail is not known
to work.

### 2.8 The realised response

`tests/test_event_study.py` validates every event study statistic twice over: against a
synthetic series with a known planted effect, where each must fire, and against a null
series, where none may. It also pins the properties that make each statistic worth having,
including that the Corrado statistic is invariant to a strictly increasing transform of the
abnormal values while a mean based statistic is not, and that the Kolari and Pynnonen
correction can only ever shrink a result.

One test asserts the structural separation the design depends on: neither
`src/evaluation/event_study.py` nor `src/evaluation/verification.py` imports the other.

### 2.9 Informative censoring and the competing risks arm

`tests/test_competing_risks.py` checks that the Aalen-Johansen estimator does what
independent censoring cannot: when the competing event strikes early, its recovery
incidence must be strictly lower than the Kaplan-Meier one, and when no competing event
occurs at all the two must agree. It also holds the T3 demotion in place, asserting that no
exported Y3 error column appears without a diagnostic label.

### 2.10 The closed robustness list

`tests/test_robustness_suite.py` compares what ran against the list pre declared in
`docs/audit.md` Part 8.7, in both directions: nothing missing and nothing added. It also
asserts that every unrunnable check carries a stated reason rather than being omitted, and
that the small disaster type subgroups are never reported as standalone findings.

### 2.11 The new targets

`tests/test_volume_targets.py` holds the constraint that matters for target two: volume is
unavailable for the 2000 archive year and for events after 2023, and a missing label must
stay missing. Zero filling would assert that turnover sat exactly at its baseline when in
fact it is unknown.

`tests/test_abnormal_returns.py` holds the constraint that matters for the market adjusted
return: the estimation window must close before the prediction origin. One test corrupts
every session from the origin onward and asserts the fitted coefficients do not move.

`tests/test_sample_flow.py` asserts that the sample accounting closes, and that recomputing
it from the same inputs reproduces the cached table, so the figure quoted in the thesis can
never drift from the pipeline.

## 3. What the suite does not check

It does not verify that the research conclusions are correct. It verifies that the code
computes what the documentation says it computes, and that the frozen results have not
moved. Judging the conclusions is what `docs/results.md` and `docs/interpretation.md` are
for.

It also does not execute the notebooks. Notebook execution is a separate manual step,
recorded in `docs/refactor_validation.md`.
