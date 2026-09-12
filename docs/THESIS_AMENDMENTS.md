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

---

# Amendments from the external-data pass (2026-09-11)

Five external sources were added after the missing-data audit. They change the sample
size, the sample period and the feature set, so several statements above and in the
thesis need re-numbering. Full specification: `docs/EXTERNAL_DATA_PRE_DECLARATION.md`,
written and committed before any of these features was scored.

### 21. The sample is 74 events, not 64

**Current:** "64 EM-DAT-qualifying natural disasters between 2000 and June 2023."

**Replace with:** "74 EM-DAT-qualifying natural disasters between September 2000 and
November 2025. The ASPI series was extended beyond the local archive's 2023-06-28 cutoff
using countryeconomy.com daily closes, verified continuous against the archive over a
21-day overlap. This admits ten previously-excluded events, including both Storm Ditwah
dates (2025-11-20 and 2025-11-27)."

**Why:** the inclusion criteria never changed — only the market series got longer. The
old cutoff was a data-availability artefact, and it excluded the thesis's own flagship
example. Walk-forward folds go from 3 to 4 and pooled out-of-fold test points from 30 to
**40**, which is the single largest improvement in statistical power in the study.

### 22. Y2's effective N must be stated separately

**Current:** implies a single N throughout.

**Replace with:** "Y1 and Y3 are observed for all 74 events. Y2 (abnormal volume) is
observed for 61: the 2000 archive workbook records no share-volume column at all, and
countryeconomy publishes the index level but not volume, so the ten post-2023 events
carry no Y2. These rows are dropped per target, never imputed."

**Why:** this was verified directly — `Data/2000 data.xls` has columns
`date | SECURITY_DA | PRICE` with no volume field, whereas the 2001 workbook's header
reads "CLOSING PRICE & SHARE VOLUME". The gap is a structurally absent field, not a parse
failure, and should be described as such.

### 23. The severity variable is barely observed — say so

**Current:** treats `financial_damage` as the severity measure.

**Add:** "EM-DAT records a financial damage figure for only 17 of the 64 originally
modelled events; the remaining 47 are zero-filled. The flood-damage interaction term
therefore rests on 12 events. Because EM-DAT records damage more often for large, recent
and internationally-reported events, these columns partly measure *reporting coverage*
rather than severity, and a coefficient on them must be interpreted accordingly."

**Why:** this is the study's real binding constraint and it is sharper than the N=64 point
the thesis currently leads with. An examiner who finds it unaided will treat it as a
concealed weakness rather than a stated limitation.

### 24. Disaster severity now has an instrument-measured component

**Add to §3.3 (Data Sources):** "Hazard intensity is measured independently of any damage
assessment using NASA POWER daily reanalysis at seven district points (Colombo, Jaffna,
Batticaloa, Nuwara Eliya, Galle, Anuradhapura, Ratnapura): three-day accumulated
precipitation, its spatial spread across districts, the count of districts exceeding the
Sri Lanka Department of Meteorology's 50 mm/3-day heavy-rain advisory level, maximum
10 m wind speed, and precipitation relative to a 30-day pre-event baseline."

**Why:** unlike EM-DAT damage this has no missingness (100% coverage on all 74 events),
no reporting bias, and is knowable on the event day — which also makes it the only
severity measure admissible in an ex-ante specification.

### 25. District-level physical severity from DesInventar

**Add to §3.3:** "District-level physical severity is taken from DesInventar Sendai, the
UNDRR/UNDP national disaster loss database for Sri Lanka (130,018 dated records,
1965–2020): districts affected, population affected, houses destroyed and damaged, and
deaths, aggregated over a [t−7, t+14] window around each event."

**State explicitly:** "DesInventar Sri Lanka populates its USD loss field in **zero** of
130,018 records and its local-currency field in 16. It supplies physical severity only
and does not recover the missing monetary damage figures. It also ends 2020-12-20, so 22
of the 74 events match no record; a `di_available` indicator carries that rather than a
zero being read as 'no damage'."

**Why:** 35 of the 47 zero-filled-damage events gain real measured severity from this
source, taking severity coverage from 17/64 to 52/74. That is a substantive improvement
and must not be overstated as recovering damage in money terms.

### 26. Macro controls are daily, not annual

**Current:** "annual GDP growth and inflation from the World Bank."

**Replace with:** "Annual World Bank series are retained for scale, and daily LKR/USD
exchange-rate dynamics (FRED series `DEXSLUS`, 13,102 observations from 1973) supply the
day-level macro state: one-day and five-day log changes and 30-day realised volatility,
all computed strictly from data at or before t−1."

**Why:** matching an annual series to a day-0 event shock is indefensible, and it was the
only macro control the study had. This is a correctness fix, not a performance lever, and
should be presented as one.

### 27. Election contamination is now measured, not just noted

**Add to the robustness section:** "Event windows were screened against 29 Sri Lankan
national election dates retrieved from Wikidata. The 2005-11-17 presidential election
falls four days before the 2005-11-21 flood event that carries the largest observed Y1
drop (−0.0753); a `days_to_election` distance and a ±5-day flag are included so this
confounder is controlled rather than merely acknowledged."

### 28. §3.3.1's news-sentiment claim is still unsupported

**Current:** §3.3.1 describes a news-sentiment input.

**Replace with:** "News sentiment was specified in the original design but is not used.
GDELT, the only free source with the required historical depth, returned HTTP 429 on every
retrieval attempt including at 20-second spacing, and ReliefWeb's API requires a
registered application key. No sentiment feature is constructed, and no result in this
study depends on one."

**Why:** the claim currently describes an input the pipeline does not have. Removing it is
mandatory; describing the attempt is what makes the removal credible.

### 29. Per-sector volume is unobtainable — record why

**Add to the limitations:** "The sector panel models sector-level return and recovery but
not abnormal volume. The Colombo Stock Exchange's per-security yearly archives carry share
volume but no sector field: their `MAIN TYPE` column holds security-class markers
(N/P/R/U/X) and `SUB TYPE` holds only `0000`/`0001`, and no company-to-sector mapping is
available locally. Per-sector volume is therefore not derivable, which is why the panel
covers Y1 and Y3 only."

**Why:** this was tested directly rather than assumed, and it is the gap most likely to be
raised by an examiner who notices Y2 is the one target with measurable signal.
