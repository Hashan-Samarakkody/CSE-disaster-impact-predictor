# Final method comparison — what the improvement work changed, and what it bought

This is the before/after of *method*, not of numbers. The numbers are in
`docs/Y1_Y3_FINAL_RESULTS.md`; the point here is which methodological choice changed, why,
and whether it was worth it. Several of these rows say "no measurable gain" — those are
kept deliberately.

Baseline = commit `1fbf6275`, frozen in `artifacts/frozen_baseline.json`.

---

## Y1 — ASPI return magnitude

| Dimension | Before | After | Did it help? |
|---|---|---|---|
| Horizons | 2 (5 and 10 sessions), grown ad hoc | 4 (5, 10, 15, 20), **all pre-declared at once**, h=5 principal | No performance gain. Large *credibility* gain: horizon shopping is now impossible to allege, because the grid was fixed in writing first. |
| Target definition | `Y1_ASPI_5D_Forward_LogReturn_Pct` + one event-window column | `Y1_ASPI_EventWindow_0_{5,10,15,20}_LogReturn_Pct`, one formula, one alignment rule, each with its own `_horizon_end_date` | Consistency. h=5 reproduces the frozen Y1 exactly (asserted in `tests/test_y1_horizons.py`). |
| Purging | Per-target label-end purge (already correct) | Same rule, per **horizon** — h=20 embargoes 20 sessions | Necessary for the new horizons; no change to existing behaviour. |
| Information sets | One combined feature set | **market-only / disaster-only / combined / normal-market+residual**, hand-partitioned, identical folds | This is the single most important methodological addition. It converts "can we predict returns?" into the answerable question "does disaster information add anything beyond market state?" — and lets the study answer *no* with evidence. |
| Expected-return baseline | naive zero, training-fold mean | + **market-only expected-return model** (Stage A Ridge on the daily series, strictly pre-event) | The right null for an event study. It is also *harder* than naive zero at h=15, which is exactly why it belongs. |
| Two-stage architecture | none | normal-market + disaster-residual decomposition | **No gain.** Mid-pack at h=5/10, worst at h=15/20. Reported as a negative result. |
| Feature capacity | K=20, fixed | K in {5, 10, 20}, with K=20 as robustness only | K=10 wins 8 of 16 cells, K=5 six, K=20 two. **The compact model is free** — report the interpretable one. |
| Model set | 9 families (Ridge, RF, XGB, GP, SVR, quantile, MLP, ensemble, stacked) | **5** (Ridge, ElasticNet, RF, XGBoost, shallow MLP); the rest stay in the appendix, unchanged | Reduces multiplicity, which is the honest way to raise power. The appendix models were not deleted — nothing is hidden. |
| Feature-selection nesting | Nested inside inner CV (already fixed) | Same, plus a **stability report** (selection_frequency / mean_rank / median_rank per fold) | Produced the study's most sobering number: **35.6% of selected features appear in exactly one of four folds.** |
| Inner-CV degradation | Fall back to an unpurged `TimeSeriesSplit` if every purged split dies | **Reduce the number of splits first; then use pre-specified default hyperparameters. Never unpurge.** | Leakage protection is no longer sacrificeable. In practice all 16 fold-horizons kept 3 purged inner splits, so the fallback never fired — but the guarantee is now structural. |
| Direction | Existing `C1_negative_return` label, one horizon | `negative_return = Y_h < 0` at all four horizons, balanced accuracy / ROC-AUC / PR-AUC / MCC / sensitivity / specificity / episode-clustered AUC CI | **The one genuinely new positive finding** (h=10, AUC 0.752, Holm p = 0.032). |
| Augmentation | SMOGN ON | **SMOGN OFF, uniformly**, across the whole new grid | Held constant so the A-vs-C and horizon contrasts are not confounded by a mask that varies with the target. The existing SMOGN ablation stands unchanged. |

### Net effect on Y1

Predictive performance: **unchanged, and that is the finding**. Best pooled R2 moved from
+0.143 (frozen MLP, h=5, SMOGN on, 9-model lineup) to +0.075 (h=10 disaster-only elastic
net) — and neither number is distinguishable from its baseline. What improved is the
*quality of the negative result*: it now rests on a pre-declared 240-configuration grid with
three baselines, four information sets and family-wise correction, rather than on a handful
of models that happened to be tried.

---

## Y3 — recovery duration

| Dimension | Before | After | Did it help? |
|---|---|---|---|
| Primary likelihood | Point regression on `Y3_recovery_days` (RF/Ridge/XGB/MLP), censored rows scored as observed | **Right-censored AFT**, censored rows contributing a lower bound | Correctness. An RMSE that treats a 90-day cap as an observed recovery is measuring the wrong thing. |
| Training rows | SMOGN-augmented | **Genuine events only** for every survival fit | Necessary, not optional: SMOGN interpolates a duration but cannot synthesise a valid event/censoring indicator, so a synthetic `(38, observed)` row asserts a recovery nobody saw. |
| Architecture | Single-stage regression; a hurdle variant as a side experiment | **Two-stage: P(drawdown) x AFT on drawdown cases** | **Yes.** Best C-index (0.657 vs 0.539-0.598 for plain AFT) *and* by far the best calibration (worst gap 0.045 vs 0.142 for plain log-normal AFT). |
| Distributions | AFT chosen in-fold by AIC (Weibull or log-normal, invisible to the reader) | **Weibull and log-normal reported separately** at every capacity | Transparency. They turn out near-identical (0.657 vs 0.643), so the AIC choice was never carrying a result. |
| Cox PH | not attempted | penalized Cox, **fitted only where the effective event count supports it** (>= 12 events, <= 5 covariates) | Ran; C-index 0.569, mid-pack. No instability, no advantage. |
| Random survival forest | — | **deliberately not used** | 31 observed recoveries does not support it. Declared in advance, not abandoned after a bad result. |
| Output | one number ("recovery = 17 days") | **predicted median + P(T <= 10/20/30/60/90)** per event | This is where Y3 becomes usable. Exact-day prediction fails; calibrated recovery probabilities do not. |
| Primary metric | pooled RMSE over all rows | **Harrell C-index** (episode-clustered CI), IPCW **integrated Brier score**, **probability calibration**; point errors only among uncensored recoveries, as secondary | Changes the conclusion: the frozen pipeline's "Y3 R2 = +0.155" was substantially an artefact of scoring censored rows as observed. Ranking is the defensible claim. |
| Baselines | naive zero, training-fold mean | **training-fold Kaplan-Meier survival curve** + training-fold median recovery | A survival model must be compared to a survival baseline. KM scores C-index 0.477 — worse than chance, as a constant curve should be — which is what makes 0.657 meaningful. |
| Categories | several thresholds available | **exactly two pre-specified**: `recovery <= 20 days` and the existing `C3b_slow_recovery` | No threshold sweep. The new one fails (AUC 0.612, CI [0.347, 0.854]); the existing one stands unchanged. |

### Net effect on Y3

The *number* got smaller and the *claim* got sounder. Pooled R2 +0.155 under a wrong
likelihood became C-index 0.657 under the right one — a real, interpretable, calibrated
ranking ability that nonetheless fails Holm correction and is reported as **suggestive**.

---

## Y2 — unchanged, by design

Nothing. Zero numbers moved, at 1e-9 tolerance, and 20 automated assertions enforce it.
See `docs/Y2_FROZEN_VALIDATION_REPORT.md`.

---

## Statistical machinery

| Dimension | Before | After |
|---|---|---|
| Pairing | Paired bootstrap already in use | Enforced and **tested**: `tests/test_improvement_artifacts.py` asserts every configuration and baseline scores an identical row set |
| Clustering | Episode-cluster bootstrap (14-day chaining) | Unchanged, extended to the new Y1 grid, the Y3 C-index, and both AUC analyses |
| Multiplicity | Holm across `verdict_table` | Holm across each new family separately: 720 Y1 comparisons, 15 Y3 models, 8 direction comparisons — **reported beside the single-comparison CI, never folded into it** |
| Verdict vocabulary | "BEATS BASELINE" / "better, not distinguishable" / "worse" | Explicit **A / B / C** classification carried in the artifacts themselves |

## Cost

Roughly 75 minutes of compute for the Y1 grid (240 configurations x 4 folds with nested
selection inside purged inner CV), ~1 minute for Y3. The Stage-A expected-return estimates
are cached (`artifacts/y1_stage_a_expected.parquet`) because they depend only on frozen
inputs.

## What was NOT done, and why

* **No lockbox holdout.** Still impossible at N=74 across a 30/10/10 walk-forward; the
  limitation stands exactly as disclosed in `docs/FINAL_ANALYSIS_PROTOCOL.md` section 8.
* **No market-only direction arm.** The direction analysis was pre-declared on the combined
  set only. Adding it now, after seeing that h=10 works, would be exactly the post-hoc
  extension the protocol forbids. It is listed as a limitation instead.
* **No additional events.** The `total_affected >= 1000` threshold was not lowered and no
  event was re-dated or removed.
* **No new algorithms.** The five-model list was closed before any result was read.
