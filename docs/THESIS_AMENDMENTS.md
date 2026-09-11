# Thesis Text Amendments

Every item below is a place where the thesis document says something the code does not do,
or uses a term that misdescribes what was measured. Each gives the current claim, the
replacement wording, and why the change is needed.

These are corrections to the *written* thesis (`IM2021007`), not to the repository. Apply
them in your Word/LaTeX document.

Ordered by severity: the first five are things an examiner will attack directly.

---

## Blocking — fix before submission

### 1. The early-warning framing is not supportable

**Where:** abstract, §1 introduction, §4.4 SDG alignment, conclusion.

**Currently:** the framework is described as an early-warning system for CBSL/SEC use.

**Replace with:** "an **ex-post impact-attribution** framework". Add, in the abstract and
again in the limitations:

> Six of the model's features — `financial_damage`, `population_affected`, their log
> transforms, `damage_to_gdp` and `log_damage_x_flood` — are EM-DAT damage assessments
> finalised weeks to months after an event. The model therefore cannot be run
> prospectively on the day before a disaster. It answers the attribution question — given
> a disaster of measured severity, what was the market's response — rather than the
> forecasting question.

**Why:** the claim as written is falsified by the feature list. This is the first thing a
viva panel will find.

---

### 2. Y1 is a raw log return, not an abnormal return

**Where:** §3.2.2, Table 4, and everywhere Y1 is named.

**Currently:** Y1 is referred to in places as a "percentage change" and elsewhere the
language borrows from the abnormal-return literature of §2.2.3.

**Replace with:** "continuously compounded (log) return on the event day,
`ln(P_t / P_{t−1})`".

**Why:** the code computes exactly that. No market model, alpha or beta is estimated, and
for a national index there is no valid benchmark against which to define an abnormal
return — which is a defensible choice, but it must be stated as the choice it is.

---

### 3. Y2 is a volume *spike* measure, not a "volume crash"

**Where:** §3.2.2, Table 4.

**Currently:** described as capturing a volume crash.

**Replace with:** "abnormal trading volume, `V_t / mean(V_{t−30..t−1}) − 1`; positive
values indicate volume above the pre-event baseline."

**Why:** the observed range runs from −0.87 to **+1.94**. The measure's upside is a spike,
and "crash" inverts its meaning.

---

### 4. Y3 is measured in trading days, and the cap is censoring

**Where:** §3.2.2, and any statement of the 90-day window.

**Currently:** "90 days", with the implication that a non-recovering event has a recovery
time of 90.

**Replace with:** "recovery time in **trading days**, right-censored at 90." Add:

> For an event that has not recovered within the window the data establishes only that
> `T > 90`; assigning `T = 90` is a biased point estimate. A two-stage hurdle model is
> reported alongside the single-stage regression for this reason.

**Why:** an earlier implementation counted calendar days while searching a window of 90
trading rows, so the cap effectively bit at about 62 trading days. That is fixed in code
and the text must follow.

---

### 5. Y3 is not independent of Y1

**Where:** §3.2.2, and before any Y3 result is reported.

**Add:**

> Because the recovery window includes the event day and the baseline is the previous
> close, any event with a non-negative event-day return satisfies the recovery condition
> immediately, so `Y3 = 0` if and only if `Y1 ≥ 0`. Over half of all Y3 observations are
> therefore mechanically determined by the sign of Y1, and Y3 performance cannot be read
> as independent evidence.

**Why:** it is an identity in the target construction, and stating it yourself is far
stronger than having it extracted in the viva.

---

## Method description does not match the code

### 6. `MultiOutputRegressor` is described but not used

**Where:** §3.6.4.

**Replace with:** "a separate estimator is fitted per target, with per-target feature
selection and per-target hyperparameter search."

**Why:** the code fits each target independently. `MultiOutputRegressor` appears in an
unused helper module only.

### 7. Multi-task learning applies to the MLP alone

**Where:** §3.6, wherever the framework is called multi-task.

**Replace with:** reserve "multi-task" for the shallow MLP, which genuinely shares a
trunk across three heads. Describe the rest as **multi-output prediction**. Add: "no
positive transfer was measured — the MLP is not the best model on any target."

### 8. Remove SVR from the methodology

**Where:** §3.6.1.

**Why:** specified but never implemented. It spans no model-family space the other four
do not already cover. Delete rather than implement, and say so if asked.

### 9. Delete the news-sentiment claim

**Where:** §3.3.1, which states aggregated financial-news sentiment was used.

**Why:** Table 4's own note says textual sentiment was excluded, and the implementation
excludes it. The two statements contradict each other; §3.3.1 is the wrong one.

### 10. Inner validation is a chronological `TimeSeriesSplit`

**Where:** §3.7.2.

**Add:** "hyperparameter selection inside each walk-forward fold uses a chronological
`TimeSeriesSplit` restricted to that fold's **real** rows, with the winning configuration
refitted on real plus synthetic rows."

**Why:** an earlier version used `GridSearchCV(cv=3)`, which is k-fold — the exact
procedure §3.7.1 bans. It was found during audit and replaced. Reporting the correction
is a strength, not an admission.

### 11. Report both rolling and expanding windows

**Where:** §3.7.1.

**Why:** §3.7.1 describes an expanding training set; the original code rolled. Both are
now available and both should be reported.

### 12. Correct the RMSE definition

**Where:** wherever RMSE is defined.

**Why:** check the formula as printed — the square root must cover the mean of squared
errors, not the sum.

### 13. The MLP loss weights express task priority, not scale correction

**Where:** §3.6.3, eq. (4).

**Replace with:** "targets are standardised before the weighted loss is applied, so the
weights (1.0, 0.1, 0.5) express the relative research priority of the three tasks rather
than compensating for their differing scales."

**Why:** the original rationale — that the weights equalise a roughly 1000× scale gap —
stopped being true once target standardisation was added. Before that fix the MLP's Y1 R²
was approximately −390.

---

## Data and validity statements

### 14. Macro controls are lag-corrected and annual

**Where:** §3.5.3.

**Add:** "World Bank annual series are dated to their publication, not to the start of the
year they describe, so a November event does not receive a figure published the following
year. Annual frequency against event-level timing remains a limitation: CBSL monthly CCPI,
daily LKR/USD and policy-rate series were not accessible."

### 15. Stationarity testing covers all features

**Where:** §3.5.1.

**Add:** "ADF and KPSS are run on every engineered market feature. Exclusion is gated on
the ADF unit-root result; six raw price-level columns (`sma_*`, `ema_*`) are excluded and
replaced by stationary price-relative ratios."

### 16. Report Y2's effective sample size

**Where:** wherever N is stated.

**Add:** "N = 64 events; **N = 61 for Y2**, since the 2000 workbook did not parse and
three events have no volume baseline. These targets are left missing rather than imputed."

### 17. State the damage-coverage problem

**Where:** §3.3.2 and limitations.

**Add:**

> Only 17 of the 64 modelled events carry a real EM-DAT damage figure; the remaining 47
> are zero-filled. The damage features therefore behave substantially as an indicator of
> **EM-DAT reporting coverage** — itself correlated with severity and recency — rather
> than as a clean severity measure. The `log_damage_x_flood` interaction rests on 12
> events.

### 18. Disclose the event-window overlap

**Where:** §3.3.2, where the truncation protocol is described.

**Add:** "29 of 64 recovery windows contain a later qualifying disaster within 90 days.
The truncation protocol is not applied; the overlap is quantified and disclosed instead."

**Why:** the SWOT implied the truncation was in use. It was not.

### 19. Soften "first-ever"

**Where:** §1, §2.7.

**Replace with:** "the first study we are aware of to…" — a literature search cannot
establish that nothing exists.

### 20. Replace causal language throughout

**Where:** everywhere SHAP results are discussed.

**Replace:** "X causes / drives / determines Y" with "X is associated with Y" or "X
contributes to the model's prediction of Y".

**Why:** SHAP attributes a *model's* output, not a causal effect, and under the
collinearity present here (several feature pairs at ρ ≈ 1.00, condition number ≈ 10¹⁷)
attribution within a correlated block is not even identifiable at the individual-feature
level.

---

## What to add that is not currently there

**A results-integrity paragraph.** State plainly that no model beats a naive baseline on
Y1; that performance is reported against two nulls, a constant-zero economic null and a
training-mean statistical null; and that intervals accompany every comparison. A negative
result reported this carefully is a stronger thesis than a positive one reported loosely.

**A selection-bias disclosure.** The split geometry, the feature count and the augmentation
configuration were all evaluated against the same folds that are reported. That carries
unquantifiable optimistic bias and should be stated rather than left for a reader to infer.

**A figure list.** The figures in `docs/figures/` are numbered and exported at 300 dpi for
citation. Each carries its sample size on the face of the figure.
