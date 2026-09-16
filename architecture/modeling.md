# Modelling (stages 04, 05, 08)

Three modelling stages share one discipline: **every fold trains on events that
happened before the events it is scored on, and no choice anywhere is made by looking
at a held-out score.**

## 1. The walk-forward loop (stage 04)

`04_modeling_regression.ipynb` runs the same loop for every model family and every
target. For each of the 4 chronological folds:

1. Slice `X_train, y_train` from the training window and `X_test, y_test` from the
   test window that follows it.
2. Drop rows where *this* target is missing. The mask is per target, which is why Y2
   has fewer pooled points than Y1.
3. Fit the imputer and the scaler **on the training rows only**, then apply them to the
   test rows. Fitting a scaler on the full column would leak the test distribution's
   mean into training.
4. Select the top `k = 20` features by random-forest importance, ranked against *this*
   target, fit on *this* fold's training rows only. Each target gets its own feature
   set; Y2 and Y3 are not forced to reuse Y1's ranking.
5. Tune hyperparameters with an inner `TimeSeriesSplit` **inside** the training window.
   (The earlier `cv=3` expanded to `KFold(shuffle=False)`, which validates earlier
   events using later ones — the exact look-ahead the design forbids.)
6. Predict the test fold, clip to the target's definitional support, and store the
   out-of-fold predictions.

Clipping uses only the target definitions from the thesis (`Y3` in `[0, 90]`, `Y2`
above `-1`), never held-out data, so it is projection onto a known support rather than
test-set fitting. Because every true value already lies inside those bounds, clipping
can never increase the absolute error of any point. Without it, an under-regularised
model in log space emitted recovery times in the hundreds of days for a target capped
at 90.

### Model families

| Family | Implementation | Why it is here |
|---|---|---|
| Ridge | `StandardScaler` + `RidgeCV`, alphas `logspace(-3, 3, 13)` | the penalised linear reference |
| Random forest | `RandomForestRegressor`, 8-combination grid | non-linear, handles small N without extrapolating |
| XGBoost | `XGBRegressor`, 8-combination grid | gradient boosting, the usual tabular benchmark |
| Shallow MLP | `src/models/shallow_mlp.py`, run in a subprocess | the thesis's multi-task architecture |
| Ensemble blend | equal-weight average of RF, XGB and MLP | a blend that cannot consult held-out error |
| Stacked | `LinearRegression` meta-learner over the three | learned combination; forfeits fold 0 to train the meta-learner |

The grids are deliberately small. At N around 76, with an inner CV inside every
walk-forward fold, a larger grid made the search itself the bottleneck — over 900
seconds per fold from fit-call overhead alone, not from data size.

The blend is **equal-weight for a reason**. An earlier version weighted each member by
its RMSE on the very fold it was predicting, which is selection on test performance.
That bug is documented rather than quietly patched, and the archived pre-split
notebook's numbers should not be quoted because of it.

### The shallow multi-task MLP

`ShallowMultiTaskMLP` is one shared hidden layer (64 units, dropout 0.5) feeding one
linear head per target, trained with `WeightedMultiTaskMSELoss`. The head count is a
constructor argument, not a constant — hardcoding three heads broke the moment the two
cumulative-return targets were added.

Two implementation details matter:

**Targets are standardised, not just features.** Y3 spans 0-90 and Y1 spans roughly
±0.07 — three orders of magnitude apart in raw scale. The loss weights `(1.0, 0.1,
0.5)` rebalance by a factor of 2-10x, nowhere near enough, so the shared hidden layer
collapsed onto minimising Y3's dominant squared-error term and produced near-random
output on Y1. Standardising *y* per fold and inverse-transforming predictions back to
raw scale before evaluating is the necessary complement to loss weighting here.

**It runs in its own process.** `src/models/mlp_subprocess_runner.py` exists because on
Windows, `import torch` after scikit-learn, XGBoost and SHAP in the same process
raises `OSError: [WinError 1114] ... c10.dll`, while importing it first in a clean
process never does. The caller writes `X_train, y_train, X_test` to an `.npz`, spawns
the runner, and reads `pred` back. The IPC cost is negligible at this sample size.

### Naive baselines

Two, and every model is compared against both:

- **constant zero** — "the event had no measurable effect";
- **training-fold mean** — "the event was average".

A model that cannot beat both of these has demonstrated nothing, regardless of its
R². This is enforced in [`evaluation.md`](evaluation.md), not left to the reader.

### Ablations

Run in the same notebook, each isolating one decision: walk-forward density (a denser
20/5/5 configuration), feature count (`k = 20` against `k = 10`), and an
**external-data block ablation** that removes each pre-declared block (hazard,
DesInventar, FX, elections) in turn and reports the whole-block contribution with a
confidence interval.

A fold whose target is entirely missing is skipped and *recorded as skipped*, because
a skipped fold in the middle of the sequence silently breaks any offset arithmetic
that maps predictions back to events.

## 2. Classification (stage 05)

Regression asks *by how much*. `05_modeling_classification.ipynb` asks *which way*,
which is a strictly easier question and the one this data can sometimes answer.

`src/models/classifiers.py` defines six pre-registered labels:

| Label | Definition | Prevalence issue |
|---|---|---|
| `C1_negative_return` | `Y1 < 0` | mild |
| `C1b_adverse_move` | `Y1` below the bottom tercile of **this fold's training window** | ~33% in training, 10% in test |
| `C2_volume_spike` | `Y2 > 0` | balanced |
| `C3_recovers_in_90` | `Y3 < 90` | 0.90 — barely informative |
| `C3b_slow_recovery` | `Y3` above the median of **this fold's training window** | ~50% by construction |
| `C4_car5_negative` | `Y1_EventWindow_0_5_LogReturn_Pct < 0` | balanced |

Cut points that move per fold (`C1b`, `C3b`) are computed on training rows only. The
*rule* is fixed in advance even though the *number* it produces differs by fold — that
is what keeps them leak-free.

`C3_recovers_in_90` is retained and still reported even though it fails, because
dropping a label after seeing it fail is exactly the selection that the
pre-declaration exists to prevent. `C3b` was added beside it, not in place of it.

### A bug worth knowing about

`(series > 0).astype(float)` maps `NaN` to `0.0`. For `C2_volume_spike` that asserted
"no volume spike" for six events whose volume was never recorded, and because the
caller masks on `notna()`, those fabricated zeros passed straight through and were
scored as true negatives. Every label now routes through `_binarise`, which propagates
missingness as `NaN`. Fixing it moved C2's AUC from 0.530 to 0.764 — the defect had
been burying the study's one working classifier.

### The Y3 hurdle model

`src/models/hurdle.py`. Y3 is censored at 90, so roughly one event in ten sits exactly
on the cap and a single regressor smears that point mass across the range. The hurdle
model splits the question:

1. a classifier predicts whether recovery happens inside the window at all;
2. a regressor fits `log1p(duration)` on the **uncensored rows only**.

Predictions are recombined and stay inside `[0, 90]` by construction. When a fold
contains no censored events the model detects the degenerate case and falls back
rather than fitting a classifier with one class.

## 3. The sector panel (stage 08)

The index-level result is largely negative, and the study's own explanation for that is
testable: an index aggregates 20 sectors, so a localised flood that hits plantations
can be invisible in the ASPI. `08_sector_panel.ipynb` tests it directly.

`src/data_pipeline/sector_panel.py` builds **one row per (event, sector)** — about 594
test rows across 30 events, against 40 at index level. Disaster covariates are constant
across an event's sectors; the market features are each sector's own.

That structure demands two changes to the validation machinery, both already built and
tested:

- **`grouped_walk_forward`** cuts fold boundaries *between* events, never inside one.
  An ordinary row-level split would put the same event's Plantations row in training
  and its Hotels row in test.
- **`event_block_bootstrap`** resamples **events**, carrying all 20 of an event's
  sector rows together. The 20 sectors of one event move together on a shock day, so
  the effective sample size is much closer to the event count than to the row count.
  A row-level bootstrap would report intervals several times too narrow.

Both regression and classification run on the panel. The measured outcome is negative:
0 of 15 sector-level comparisons beat their baselines, which closes off the
"aggregation hides the effect" explanation rather than leaving it as an untested
excuse.
