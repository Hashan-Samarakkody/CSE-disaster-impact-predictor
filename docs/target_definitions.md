# Target definitions, as implemented

Written for: the researcher, to adapt into thesis section 3.2.2 and Table 4.

This document states exactly what the code computes, in a form precise enough that a
second person could reproduce the dataset from this page alone. The binding specification
is `docs/TARGET_DEFINITION_PROTOCOL.md`, frozen at commit `928a255` on 2026-09-19 before
any model performance under these definitions was observed. This page is the same content
written for a thesis reader, plus the observed distributions.

Implementation: `src/targets/event_targets.py`. Dataset: 74 events, 2000-09-18 to
2025-11-27, in `artifacts/tables/dataset.parquet`.

## 1. Event alignment

Let `tau` be the moment the disaster becomes known, and let `position` be the first
**complete** CSE trading session on or after the recorded disaster date. A disaster
falling on a Saturday, Sunday or market holiday advances `position` to the next real
trading session. No observation is fabricated for a non trading day.

The protocol indexes prices and volumes from `tau`, not from the session:

```
P0  = market[position - 1]        the last close fully observed before tau
Pk  = market[position + k - 1]    the close of the kth complete session after tau
V-i = market[position - i]        the ith session of volume before tau
Vi  = market[position + i - 1]    the volume of the ith complete session after tau
```

So `P1` is the close of the session that carries `position`, and `P5` is
`market[position + 4]`. Counting runs from 1, not from 0. This is the single most common
source of irreproducibility in event studies and the thesis should state it explicitly.

An event is excluded when `position` cannot be located in the market series, or when
`position` is the first session of the series, because no `P0` would exist.

## 2. Y1, ASPI return magnitude

Column `Y1_ASPI_5D_Forward_LogReturn_Pct`.

```
Y1 = 100 * ln( P5 / P0 )
```

The continuously compounded return, in percent, from the last close known before the
disaster to the close of the fifth complete session after it.

| | |
|---|---|
| units | percent (log points) |
| support | unbounded |
| sessions spanned | five post event sessions, `P1` through `P5` |
| label end date | `date[position + 4]`, the session of `P5` |
| missing when | fewer than five complete sessions remain after `position` |

Pre registered horizon sensitivity analyses, never promoted to primary:
`Y1_ASPI_10D_Forward_LogReturn_Pct`, `Y1_ASPI_15D_Forward_LogReturn_Pct` and
`Y1_ASPI_20D_Forward_LogReturn_Pct`, each `100 * ln(Ph / P0)` with `Ph = market[position + h - 1]`.

### Observed values

| | n | mean | sd | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|---|---|
| Y1 5 sessions | 74 | +0.339 | 2.680 | -5.397 | -0.966 | -0.121 | +1.197 | +8.797 |
| Y1 10 sessions | 74 | +0.937 | 4.744 | -10.705 | -1.813 | -0.010 | +3.247 | +17.302 |

51.4 percent of events are negative at five sessions, 50.0 percent at ten. The
distribution is close to symmetric around zero: disasters do not, on average, move the
CSE in a fixed direction.

## 3. Y2, forward abnormal trading volume

Column `Y2_5D_Forward_AbnormalVolume_LogRatio`.

```
V_base    = mean( V-30 ... V-1 )     30 sessions before tau
V_future5 = mean( V1 ... V5 )        the first five complete sessions after tau

Y2 = ln( V_future5 / V_base )
```

| | |
|---|---|
| units | log ratio |
| support | unbounded above and below |
| response window | `V1` to `V5`, exactly five sessions |
| baseline window | the thirty sessions before `tau`, exactly thirty, `V1` excluded |
| label end date | `date[position + 4]`, the session of `V5` |
| missing when | the volume series is unavailable, or fewer than five sessions remain |

Reader facing conversions: `100 * (exp(Y2) - 1)` is the percentage above or below normal
turnover, and `V_base * exp(Y2)` is the implied future volume. Both constituent means are
stored, as `Y2_V_base` and `Y2_V_future5`, so any reader can recompute the target by hand.

**Label end date.** `date[position + 4]`, the final session of the response window, not
the prediction origin. The walk forward purge reads this column to drop training events
whose label reaches into a test period. Dating it at the origin would under purge by four
sessions and let test period information into training. This is a correctness
requirement, not a convention.

**Naming.** "Volume crash magnitude" presupposes a negative sign, and 23 of the 61
observed values are positive. The neutral name, used throughout the code and the
protocol, is forward abnormal trading volume.

### Observed values

| n | mean | sd | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|---|
| 61 | -0.134 | 0.594 | -1.464 | -0.494 | -0.083 | +0.321 | +1.120 |

Observed for 61 of 74 events. The 13 missing are the 2000 archive year and post 2023
events, where the exchange publishes the index level but not market wide volume. They
must stay missing and must never be zero filled, which would assert that volume sat
exactly at baseline.

## 4. Y3, market recovery duration

Column `Y3_ASPI_Recovery_Time`. A time to event outcome, not ordinary regression.

The recovery baseline is `B = P0`, the last close known before the disaster.

1. **Drawdown condition.** `D = 1` when `min(P1 ... P5) < B`, else `D = 0`.
2. **No drawdown.** The market never fell below its pre event level, so no recovery
   process exists. The row carries duration 0, `Y3_event_observed = 0` and
   `Y3_censor_reason = "no_drawdown"`. It is a recorded state, not a zero length
   recovery, and the protocol forbids treating it as an ordinary duration observation.
3. **Drawdown occurred.** `T_recovery = min{ k >= 1 : Pk >= B and some j < k has Pj < B }`.
   The scan runs forward from `k = 1`, not from the trough, because a genuine recovery
   that precedes a later and deeper dip must not be skipped.
4. **Censoring.** `T_observed = min(T_recovery, T_next, 90)`, where `T_next` is the number
   of sessions from the origin to the next qualifying disaster. A recovery counts as
   observed only when `T_recovery < T_next` and `T_recovery <= 90`.

| | |
|---|---|
| units | CSE trading sessions |
| support | `[0, 90]` |
| label end date | `date[position + T_observed - 1]`, the recovery or censoring session |

Stored companion columns: `Y3_event_observed`, `Y3_censored` (its complement),
`Y3_censor_reason` and `Y3_drawdown_occurred`. The drawdown flag is derived from
`P1 ... P5`, so it is future information: it is retained as an outcome and as the stage 1
label, and is never a predictor.

**Two stage architecture.** Stage 1 predicts `D`. Stage 2 predicts duration conditional on
`D = 1`. The preferred model output is not a point duration but a predicted median
recovery time and the probabilities `P(T <= 10)`, `P(T <= 20)`, `P(T <= 30)`,
`P(T <= 60)` and `P(T <= 90)`.

### Censoring counts

| `Y3_censor_reason` | meaning | `Y3_censored` | n |
|---|---|---|---|
| `recovered` | recovery observed before the cap and before any competing event | False | 36 |
| `no_drawdown` | the index never fell below `P0` inside `P1 ... P5` | True | 22 |
| `next_disaster` | a further qualifying disaster intervened first | True | 10 |
| `90_day_cap` | still below `P0` after ninety sessions | True | 6 |

Any censoring aware model, survival or hurdle, must read `Y3_event_observed`, never infer
censoring from `Y3 >= 90`.

### Observed values

| n | mean | sd | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|---|
| 74 | 14.257 | 24.787 | 0 | 0 | 4.0 | 14.5 | 90 |

- 52 of 74 events (70.3 percent) show a qualifying drawdown.
- 22 events (29.7 percent) record duration 0 with no drawdown.
- 38 events (51.4 percent) are censored in total: 22 with no drawdown, 10 by a competing
  disaster, 6 at the cap.
- Among the 36 events with an observed recovery: median 5 sessions, p75 8.5, max 35.

The distribution is right skewed with a point mass at zero. Any model whose prediction
carries a floor, for example a hurdle whose expectation includes a `(1 - P) * 90` term, is
structurally disadvantaged against it.

## 5. Label end dates

Every target carries the session on which its value becomes fully known. The walk forward
purge reads these to drop training events whose label reaches into a test period.

| target | column | value |
|---|---|---|
| Y1, 5 sessions | `Y1_horizon_end_date` | `date[position + 4]` |
| Y1, h sessions | `Y1_{h}D_horizon_end_date` | `date[position + h - 1]` |
| Y2 | `Y2_horizon_end_date` | `date[position + 4]` |
| Y3 | `Y3_label_end_date` | `date[position + T_observed - 1]` |

If any target's window changes, its label end date must change with it. This is not
cosmetic: a stale label end date is a leakage defect.

## 6. Worked example, the 2004 Indian Ocean tsunami

Disaster date 2004-12-26, a Sunday. 2004-12-27 was a market holiday, so the first complete
session is 2004-12-28.

```
P0  = 1,572.55   (2004-12-23)      V_base    = 7,810,033 shares
P1  = 1,504.41   (2004-12-28)      V_future5 = 4,992,180 shares
P5  = 1,509.25   (2005-01-03)
P10 = 1,568.11   (2005-01-10)

Y1_5D  = 100 * ln(1509.25 / 1572.55) = -4.1086
Y1_10D = 100 * ln(1568.11 / 1572.55) = -0.2827
Y2     = ln(4,992,180 / 7,810,033)   = -0.4475,  i.e. 36.1 percent below normal turnover
Y3     = 13 sessions, drawdown occurred, recovery observed on 2005-01-13
```

Read the numbers. Five sessions after the tsunami the ASPI stood 4.11 percent below the
close last known before the disaster; ten sessions after, 0.28 percent below. Turnover
over the five post event sessions ran 36 percent below its thirty session norm. The index
regained its pre event level on the thirteenth session after the disaster.

The event session itself shows the shock: the ASPI fell from 1,572.55 to 1,504.41, a 4.3
percent single session drop, and turnover roughly halved. The index then recovered most of
that within three weeks. This is the pattern the targets are built to measure.

## 7. Constants

```
pre event volume baseline      30 trading sessions
Y2 response window              5 sessions, V1 to V5
Y1 primary horizon              5 sessions, endpoint P5
Y1 sensitivity horizons        10, 15 and 20 sessions
drawdown gate                   5 sessions, P1 to P5
maximum recovery follow up     90 trading sessions
```

These values live in `src/targets/event_targets.py` as module constants and in
`src/config/settings.py` as the target names, bounds and label end date mapping.

## 8. Checklist before the thesis goes out

- [ ] Section 3.2.2 Y1 no longer says `ln(P_t / P_{t-1})`, which is a one day return
- [ ] Y1 described as a five session forward return from the last pre event close, with
      the endpoint stated as `P5 = market[position + 4]`
- [ ] Y2 described as a log ratio, not a ratio minus one, and not a monetary value;
      volume is measured in shares
- [ ] Y3's baseline named explicitly as `P0`
- [ ] Y3's scan origin stated: the scan starts at `k = 1`, not at the trough
- [ ] Y3's no drawdown convention stated: duration 0, no event observed, reason
      `no_drawdown`
- [ ] Censoring distinguished from capping, and the four censor reasons listed
- [ ] "Trading sessions" used throughout, never "days"
- [ ] Table 4 target rows replaced with the three definitions above
- [ ] Figure output labels updated to the current column names
