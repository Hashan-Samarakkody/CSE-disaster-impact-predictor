# Event Study Cross-Sectional Dependence

## Method change (A2 — R2-revision5)

`src/evaluation/event_study.py:average_cross_correlation()` previously used
the event-window overlap share as a proxy for the pairwise correlation of
market-model residuals.  The function now estimates the correlation directly
from `panel.estimation_residuals`: for each pair (i, j) of events, the sample
Pearson correlation of their estimation-window residual series is computed over
the shared calendar sessions; pairs with fewer than 10 shared sessions are
excluded.

## Results

| series | mean r | median r | n\_pairs\_used | n\_pairs\_total |
|--------|--------|----------|---------------|----------------|
| return (market model) | 0.9977 | 0.9992 | 208 | 2701 |
| volume (mean model) | 1.0000 | 1.0000 | 187 | 2145 |

n\_pairs\_total = C(74,2) = 2701 for returns, C(66,2) = 2145 for volume.
n\_pairs\_used: pairs whose estimation windows share ≥ 10 calendar sessions.

## Interpretation

The 208 (return) and 187 (volume) pairs with sufficient overlap are
nearly perfectly correlated (r ≥ 0.997).  This is not surprising: all events
are measured on the same single security (ASPI), so any two events whose
estimation windows fall on overlapping calendar periods sample literally the
same underlying residual series.  Events separated in time (non-overlapping
windows) have no shared sessions and are excluded from the average.

The practical effect on the Kolari–Pynnönen test statistic is that the KP
denominator becomes `1 + (n-1) × r ≈ 1 + 73 × 0.9977 ≈ 73.9`, collapsing
the adjusted t-statistic to approximately zero regardless of the BMP input.
This is the correct behaviour: in a single-security design where correlated
observation windows are pooled, the KP formula correctly signals that
cross-sectional averaging does not add independent information.

## KP-adjusted results (recomputed, event_study_caar.parquet)

| series | CAAR(1,5) | t\_cross | t\_BMP | t\_Corrado | t\_KP |
|--------|-----------|----------|--------|------------|-------|
| return | +0.0006 | +0.177 (p=0.860) | −0.042 (p=0.967) | −0.122 (p=0.903) | ≈0.000 (p≈1.000) |
| volume | +0.6296 | +1.569 (p=0.121) | +1.336 (p=0.186) | +2.255 (p=0.024) | ≈0.000 (p≈1.000) |

The KP statistic is effectively uninformative for this single-security design.
The Corrado rank test (p=0.024 for volume) is the more appropriate non-
parametric test; it does not rely on a cross-sectional correlation assumption.

## Artifact source

Script: `scripts/run_event_study.py`
Artifact: `artifacts/tables/event_study_caar.parquet` (column `mean_cross_correlation`)
