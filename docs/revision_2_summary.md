# Revision 2 — what changed and where each result lives

Written for the supervisor reviewing the second revision, and for anyone re-running the
repository afterwards.

This is the summary document the change specification requires. It maps every task to the
code that implements it, the artifact it writes, and the place a reader should look for
the corresponding result. The dated per-task change log is `docs/audit.md` Part 8.11; the
pre-declaration written before any of these results existed is Part 8.1 to 8.10.

Every number quoted anywhere in the repository comes from one run, executed 2026-09-22 in
the order the specification mandates. The seed stayed at 42. Nothing was re-run in search
of a better result.

## The four claims a reader should take away

1. **The realised return response is not detectable.** Cumulative average abnormal return
   over sessions 1 to 5 is +0.0006, and every test statistic returns p above 0.85. This is
   not a weak effect that the models failed to find; there is no effect in the returns to
   find.
2. **The volume response is detectable only by the test that does not assume normality.**
   CAAR(1,5) is +0.63 log points. The Corrado rank test fires at p 0.024 while the
   Boehmer-Musumeci-Poulsen standardised residual test gives p 0.186 and the
   Kolari-Pynnonen cross-correlation adjustment gives p 0.298. The disagreement is itself
   the finding, and it is what event-induced variance in a 74-event sample looks like.
3. **Almost nothing is forecastable out of sample.** One comparison in the entire study
   survives a family-wise correction: the direction of the return at the pre-declared ten
   session horizon, ROC AUC 0.817. Return magnitude, recovery duration and every
   classification label fail. Forward abnormal volume is the one qualified positive.
4. **Later knowledge of severity buys essentially nothing.** The real-time and ex-post
   information sets differ by 0.0002 in delta RMSE on the principal analysis. The
   finalised EM-DAT damage assessments, DesInventar counts, NASA POWER reanalysis and
   annual World Bank series — 31 of the 68 features — do not improve on what the market's
   own prior state already implies.

Taken together these separate two questions that the earlier draft ran together: whether
disasters move the market, and whether those moves can be predicted. The answer to the
first is mostly no and, for volume, only under a rank test. The answer to the second is no
almost everywhere.

## Where each result lives

| Question | Artifact | Document |
|---|---|---|
| Did the market react at all | `artifacts/tables/event_study_*.parquet`, `docs/figures/` event-time plots | `docs/results.md` §2 |
| Is return magnitude forecastable | `aspi_grid_verdicts.parquet` (18 confirmatory), `aspi_grid_verdicts_exploratory.parquet` (1332) | `docs/results.md` §3, `docs/interpretation.md` §2.2 |
| Is return direction forecastable | `aspi_direction_metrics.parquet` | `docs/results.md` §4 |
| Is volume forecastable | `verdict_table.parquet`, `results_regression.pkl` | `docs/results.md` §5, `docs/interpretation.md` §2.1 |
| Is recovery forecastable | `recovery_grid_metrics.parquet`, `final_table_recovery.csv` | `docs/results.md` §6 |
| Does the regression treatment of recovery distort it | `final_table_recovery_regression_diagnostic.csv`, every row labelled | `docs/results.md` §6 |
| Classification metrics | `classification_summary.parquet` | `docs/results.md` §7 |
| Does the result survive perturbation | `robustness_suite.parquet`, 26 checks | `docs/results.md` §8 |
| How the sample was built | `sample_flow.parquet`, flow figure | `docs/results.md` §1 |
| What is available when | `feature_spec.json`, key `AVAILABILITY_CLASS` | `docs/audit.md` §8.3 |

## Tasks, by what they were for

**Making the question answerable.** T1 split the features by whether a forecaster could
actually have had them, which is what makes claim 4 above possible to state at all. T6
built a market-adjusted abnormal return so that a global move in March 2020 or during the
2022 crisis is not read as a disaster effect. T10 bounded the study period and removed the
five unexplained months of 2026 data, without changing the event count or the fold spans.

**Making the inference honest.** T11 added the event-study tests, so the claim that
disasters are followed by measurable effects now rests on a statistic rather than on a
descriptive mean. T7 cut the Holm family from everything to the 18 comparisons that
correspond to stated hypotheses. T8 removed the tuned-against-untuned asymmetry, which is
what changed the volume verdict. T2 and T3 stopped treating an informatively censored
duration as an ordinary regression target.

**Making it checkable.** T12 made the 110 to 94 to 74 progression an artifact that a test
asserts against the pipeline. T13 moved the leakage check into pytest. T14 pinned the
environment. T15 collapsed the scattered robustness work into one closed, pre-declared
list, run once.

**Correcting what the repository said about itself.** D1 to D6. The repository had
described itself as an early-warning system for volume crashes; it is an ex-post
attribution study of a two-sided volume response, and 23 of the 61 observed values are
positive.

## Two things a reader should be told plainly

**The volume verdict moved, and the reason is in the method, not the data.** An earlier
draft reported Holm corrected p 0.021 for the volume target and called it supported. That
comparison ranked a tuned tree model against an untuned neural network. T8 required both
to be selected on the same purged inner splits; once they are, the advantage falls to
corrected p 0.078 and the verdict becomes qualified. The frozen Y2 regression baseline was
re-anchored for the same reason, with the previous baseline retained beside it as
`artifacts/results/frozen_baseline_superseded_f076bd96.json`.

**One outstanding item needs the repository owner.** The GitHub About field is a setting on
github.com rather than a file, so it cannot be corrected by a commit. The replacement text
is in `docs/repository_metadata.md`.

## Reproducing the run

```
pip install -r config/requirements.lock.txt      # Python 3.12.10
python -m pytest tests/ -q                       # 226 passed
```

Then notebooks 01 to 06 in order, followed by `scripts/run_aspi_return_grid.py`,
`scripts/run_recovery_survival_grid.py`, `scripts/run_event_study.py`,
`scripts/run_robustness_suite.py`, and finally `scripts/train_final_models.py`,
`scripts/build_final_tables.py` and `scripts/audit_results.py`. The return grid takes
roughly two and three quarter hours and the regression notebook roughly three.
