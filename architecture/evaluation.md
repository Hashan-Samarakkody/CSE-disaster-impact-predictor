# Evaluation and explainability (stages 03, 06, 07, 09)

This project's headline result is largely negative, which raises the bar on
evaluation: a negative result is only worth reporting if the machinery that produced it
would have detected a positive one. Everything below exists to make that claim
checkable.

## Exploratory data analysis (stage 03)

`03_eda_diagnostics.ipynb` fits nothing. It runs before any model and produces the
evidence for decisions made later. `src/evaluation/eda_figures.py` exports twelve
figures to `docs/figures/`:

| Figure | Question it answers |
|---|---|
| `eda_08_event_timeline` | what period does the market series cover, and where do the events fall in it |
| `eda_01_missingness_matrix` | what is observed, what is zero-filled, and what is absent |
| `eda_02_missingness_ranked` | which variables are least complete |
| `eda_03_outliers` | which events are extreme, by two different rules |
| `eda_04_feature_distributions` | how skewed is each variable |
| `eda_05_qq_targets` | are the targets anywhere near normal |
| `eda_06_scatter_matrix` | does any severity measure track any target |
| `eda_07_corr_*` (x3) | which predictors correlate with each target |
| `eda_09_class_balance` | what does the majority rule score for free on each label |
| `eda_10_sector_coverage` | which sector indices span the study window |

Four conventions run through all of them.

**A missing value and a zero-filled value are drawn differently.** The loader replaces
absent EM-DAT damage figures with `0.0`, so `financial_damage` contains no `NaN` and a
conventional missingness plot would report it as complete. It is not: for most events
that zero means *unknown*, and a model reads it as *no damage*. Measured across the
table, 586 cells are present-but-not-measured against 95 outright `NaN`.

**Outliers are flagged two ways and retained, never trimmed.** The IQR fence is
conventional; the modified z-score (median absolute deviation) is reported beside it
because IQR fences are themselves computed from quantiles that a few extreme events can
move. The two disagreeing is itself informative: Y3 has 14 IQR outliers and 0 MAD
outliers, which says the "outliers" are the point mass at the 90-day cap rather than
stray values. Extreme events are **named** on the figure — an outlier that turns out to
be the 2004 Indian Ocean tsunami is a real observation to discuss, not an error to
remove.

**A trend line is drawn only where the correlation clears the critical value at this
sample size.** A fitted line through noise is the most misleading mark that can appear
on a scatter plot. At n = 76 the Fisher-z critical value is about 0.23, so most panels
correctly carry no line at all.

**Class balance is plotted before any model is fitted**, because prevalence determines
what an accuracy figure can possibly mean. A label at prevalence 0.90 hands the
always-predict-majority rule 0.90 accuracy for free.

## The verdict machinery (stage 06)

`06_evaluation.ipynb` answers one question first, before any results table:
**does anything beat its baseline?**

`src/evaluation/verification.py` runs two tests on every (model, target, baseline)
triple:

- **Paired event-level bootstrap** — resamples *events* in pairs, so the model and the
  baseline are always compared on the same resampled set, and reports a confidence
  interval for `(baseline error − model error)`.
- **Diebold-Mariano** — a test on the loss differential, with the
  Harvey-Leybourne-Newbold small-sample correction, which matters at 40 pooled points.

A verdict counts only when both agree. Anything else is reported as "not
distinguishable", which at this sample size is the honest answer more often than not.

### Metrics, and what each is for

`src/evaluation/metrics.py`:

- `evaluate_regression` — RMSE, MAE, R². RMSE is the primary metric (thesis 3.8.1):
  it squares errors, so it is the one that penalises the large misses that matter for
  a market application.
- `pooled_arrays` / `pooled_frame` — concatenate per-fold out-of-fold predictions into
  one paired vector, and map each prediction back to the event that produced it. This
  is fiddlier than it looks: predictions were recorded *after* a per-target `NaN` mask,
  the stacked model stores one fewer fold, and an all-`NaN` fold is skipped entirely.
  Fold indices are therefore read from the store rather than inferred by offset
  arithmetic, which a skipped middle fold breaks silently. A closing length assertion
  turns any remaining mismatch into a hard failure instead of a misaligned scatter plot.
- `skill_score` — `1 − model/reference`; positive means an improvement on the
  reference.
- `bootstrap_metric_ci`, `bootstrap_auc_ci`, `hanley_mcneil_ci`, `wilson_ci` — interval
  estimates. Every reported number carries one.
- `conformal_q` — the finite-sample conformal quantile behind the prediction intervals.

### Classification metrics, and the criterion that actually matters

`classification_metrics` in `src/models/classifiers.py` reports accuracy, balanced
accuracy, precision, recall, F1, MCC and AUC **always beside the majority-rule
baseline**.

The success criterion for this study is `beats_baseline`, which requires:

- balanced accuracy strictly above 0.5, **and**
- an AUC whose lower bound excludes 0.5 under *both* the Hanley-McNeil interval and
  the bootstrap interval.

Raw accuracy is explicitly **not** the criterion. On `C3_recovers_in_90` the majority
rule alone scores 0.900 because prevalence is 0.900; a model matching that number has
demonstrated nothing. `tests/test_models_classification.py` asserts the property that
makes this the right choice — the trivial always-majority classifier scores exactly 0.5
balanced accuracy and cannot game it.

### Prediction intervals

Rolling split conformal prediction at a target coverage of 80%. Achieved coverage is
reported with Wilson intervals beside the mean half-width, because a prediction
interval that achieves its coverage by being uselessly wide is not a success.

### Figures

`src/evaluation/figures.py` exports the evaluation figures at 300 dpi to
`docs/figures/`, all in the colourblind-safe Okabe-Ito palette, none encoding
information by hue alone, each stamped with its sample size. The ones that carry
argument rather than decoration:

- `fig_17_model_vs_naive_baseline` — both naive baselines drawn as reference rules, so
  the comparison is visible rather than tabulated;
- `fig_18_skill_forest_*` — skill scores with bootstrap intervals, which is where the
  "not distinguishable" verdicts become obvious;
- `fig_14_roc_with_ci` — ROC with a bootstrap band and an automatic flag when the band
  covers the chance diagonal;
- `fig_24_overfitting_gap` — in-sample R² down to held-out pooled R², per target.

## Explainability (stage 07)

`07_explainability.ipynb` runs SHAP on the fitted random forests: global importance
over the pooled out-of-fold predictions, and a local waterfall for one real modelled
event (the largest ASPI drop in the training set).

The scope limit stated in the README applies here too. This is **ex-post impact
attribution**, so a SHAP attribution explains what the model used to describe a market
response after the fact — not a forecast anyone could have acted on the day before.

## Synthesis (stage 09)

`09_synthesis.ipynb` contains no code. It is the written record: the cross-check
against the thesis blueprint, SWOT, the SMART statement, the ethics and reliability
discussion, and the conclusion. It reads the artifacts the earlier stages cached and
cites them.

## The standing audit

`scripts/audit_results.py` is the one place that answers "every metric, per model, per
target" in a single run, and it also compares the current results against the recorded
pre-change baseline so that an improvement claim is measured rather than asserted. Run
it after the pipeline:

```bash
python scripts/audit_results.py
```

Its output is kept at [`../docs/RESULTS_AUDIT.txt`](../docs/RESULTS_AUDIT.txt).
