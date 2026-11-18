# Notebook 05, Classification and the Recovery Hurdle Model

Documents `notebooks/05_modeling_classification.ipynb`.

## Purpose

Answer the binary version of the same three research questions. A market response can be
hard to predict in size and still be predictable in direction, and this stage tests that
separately rather than presenting it as a regression success.

## Inputs

`dataset.parquet`, `splits.pkl` and `feature_spec.json` from stage 02.

## Main processing stages

Section 5.2 runs the walk forward classification loop over the six pre registered labels.
Section 5.3 summarises the results and states a verdict per label. Section 5.4 fits the two
stage hurdle model for recovery duration and compares it against the naive baselines.
Section 5.5 states what was produced.

## Important functions called

| Function | Module | What it does |
|---|---|---|
| `LABELS` | `src/models/classifiers.py` | the six label definitions, derived from the three targets |
| `build_classifiers` | `src/models/classifiers.py` | the candidate families and the per label selection |
| `classification_metrics` | `src/models/classifiers.py` | balanced accuracy, precision, recall, F1, Matthews correlation, precision recall AUC, ROC AUC with a bootstrap interval |
| `HurdleRecoveryModel` | `src/models/hurdle.py` | classify recovery within the window, then regress duration on the recovered rows only |
| `paired_bootstrap_delta`, `diebold_mariano` | `src/evaluation/verification.py` | the paired comparison against the baselines |

## Important outputs

`results_classification.pkl`, `classification_summary.parquet` and `hurdle_table.parquet`.

## Relationship with other notebooks

Reads stage 02. Feeds stage 06 and the thesis results chapter. The volume spike label in
this stage is the classification arm of the frozen volume target, and it is one of the
things `tests/test_volume_target_frozen.py` holds fixed.

## Methodological decisions made here

1. **Accuracy is never reported alone.** One label has a prevalence near ninety per cent,
   where always predicting the majority class scores ninety per cent for free. The success
   criterion throughout is balanced accuracy above one half together with an AUC interval
   that excludes one half, and every accuracy figure is printed beside its prevalence.
2. **Labels are purged against their own target's horizon.** Each label inherits the
   settlement date of the target it is derived from, so a label built on a ninety session
   recovery window is embargoed by ninety sessions, not by five.
3. **Competing risk rows are excluded from the recovery labels, not guessed.** An event
   whose recovery was interrupted by a later disaster has a genuinely unknowable recovery
   status, so it is dropped from those two labels rather than labelled either way.
4. **The hurdle model exists because the target has a point mass.** Some events sit
   exactly on the ninety session cap and a single regressor smears that mass across the
   range. Stage one classifies recovery within the window and stage two regresses the log
   duration on the recovered rows only.
5. **The hurdle is fitted and scored on drawdown events only.** Under the frozen target
   protocol an event with no drawdown carries duration zero as a recorded state rather
   than as an instant recovery, so including it would ask the model a question that event
   does not pose and would force a ninety session prediction against a true zero.
6. **The hurdle model is reported even though it loses.** It scores worse than doing nothing
   on this sample. That result is kept, and it is part of the reason the recovery target was
   later re founded on a censored survival likelihood rather than a point regression.

## Justification

Separating direction from magnitude is standard in forecast evaluation, where a model can
carry directional information while failing a squared error test. Balanced accuracy with a
bootstrap interval on the AUC is the appropriate criterion under class imbalance, because
raw accuracy rewards the majority rule. The hurdle structure is the textbook treatment of a
semi continuous outcome with a mass at a boundary, and reporting its failure here is what
motivates the censoring aware treatment documented in `docs/experiments.md`.

## Execution requirements

Stage 02 must have run. This stage takes a few minutes.
