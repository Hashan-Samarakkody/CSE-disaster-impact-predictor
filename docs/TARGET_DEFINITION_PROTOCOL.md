# Target definition protocol — FROZEN

**Frozen at commit:** `928a255d0c23429a20ff062c7f09bf06f0186d31`
**Branch:** `update`
**Frozen on:** 2026-09-19, **before** any model performance under these definitions was observed.

These three formulas are final. They are not to be modified after seeing model
performance, and no additional target definition is to be searched because one of these
produces a poor R² or an insignificant result.

---

## 0. Common time definitions

`τ` = the time at which the disaster is known and a prediction can realistically be made.

`t` = the first **complete** CSE trading session occurring after `τ`.

- disaster after market close → `t` = next session
- disaster before market open → `t` = that upcoming complete session
- disaster during an active session → that unfinished session is **not** used; `t` = next complete session

`P0` = the latest fully observed ASPI close available **before** `τ`.

`P1 … Pk` = ASPI closes of the 1st … kth complete sessions after `τ`. So `P1` is the close
of session `t`.

`V_{-1} … V_{-30}` = market-wide traded volumes of the 30 complete sessions immediately
preceding `τ`.

`V1 … V5` = market-wide traded volumes of the first five complete sessions after `τ`.

All target observations occur strictly after the prediction origin. No predictor feature
may contain information finalised after it.

### Implementation mapping

The repository indexes the market series by `position` = the first session on or after the
recorded disaster date. That session is the first complete session after `τ`. Therefore:

```
P0 = market[position - 1]        Pk = market[position + k - 1]
V_{-i} = market[position - i]    Vi = market[position + i - 1]
```

So `P5 = market[position + 4]` and `V1…V5 = market[position … position + 4]`.

---

## TARGET 1 — ASPI return magnitude

**Name:** `Y1_ASPI_5D_Forward_LogReturn_Pct`

```
Y1 = 100 × ln( P5 / P0 )
```

Units: percent (log points). Support: unbounded.
Label-end date: the date of `P5`.

Answers: *"From the market level currently known, how much will ASPI move during the next
trading week after the disaster?"*

The previous 30-session average ASPI is **not** the denominator. The denominator is the
market level that actually existed immediately before prediction.

**Horizon sensitivity analyses** (clearly labelled as such, never promoted to primary):

```
Y1_10 = 100 × ln( P10 / P0 )
Y1_15 = 100 × ln( P15 / P0 )
Y1_20 = 100 × ln( P20 / P0 )
```

**Worked example from the specification.** `P0 = 10,000`, `P5 = 9,700`:
`Y1 = 100 × ln(9700/10000) = −3.0459`.

---

## TARGET 2 — Forward abnormal trading volume

**Name:** `Y2_5D_Forward_AbnormalVolume_LogRatio`

```
V_base    = (1/30) × Σ V_{-i}        i = 1 … 30
V_future5 = (1/5)  × Σ V_i           i = 1 … 5

Y2 = ln( V_future5 / V_base )
```

Units: log ratio. Support: unbounded below and above (a log ratio, not a ratio minus one).
Label-end date: the date of `V5`.

Answers: *"How abnormal will average trading activity be during the next five complete
trading sessions?"*

Reader-facing conversions:

```
abnormal volume %        = 100 × [exp(Y2) − 1]
predicted future volume  = V_base × exp(Y2)
```

**Worked example from the specification.** `V_base = 20e6`, `V_future5 = 30e6`:
`Y2 = ln(30/20) = 0.4055`, i.e. `100 × (exp(0.4055) − 1) = 50%` above normal.

`V1 … V5` are target-only. None of them may appear as an input feature.

---

## TARGET 3 — Market recovery duration

**Name:** `Y3_ASPI_Recovery_Time`

A time-to-event / survival target, not ordinary regression.

### 3.1 Baseline

```
B = P0
```

The last fully observed close before prediction origin. The 30-session average ASPI is
**not** the recovery level.

### 3.2 Drawdown condition

```
D = 1  if  min(P1, P2, P3, P4, P5) < B
D = 0  otherwise
```

`D = 0` means "no observed post-disaster drawdown / no recovery process required". Such
events are **not** forced to behave like ordinary recovery-duration observations.

Model architecture, two stages:

- **Stage 1** — predict whether a drawdown occurs (`D`).
- **Stage 2** — conditional on `D = 1`, predict recovery duration.

### 3.3 Recovery event

```
T_recovery = min{ k ≥ 1 :  Pk ≥ B  AND  ∃ j < k with Pj < B }
```

The scan runs forward from `k = 1`. Recovery requires that the index has previously fallen
below `B`. The scan origin is **not** the trough.

### 3.4 Maximum observation window

```
C_max = 90 trading sessions
```

Recovery not observed within 90 sessions → `T_observed = 90`, `event_observed = 0`,
right-censored. This states "recovery took longer than 90 sessions", not "recovery took
exactly 90 sessions".

### 3.5 Competing disaster censoring

```
T_next     = sessions from prediction origin to the next qualifying disaster
T_observed = min( T_recovery, T_next, 90 )

event_observed = 1  only if  T_recovery < T_next  AND  T_recovery ≤ 90
                 0  otherwise
```

`censor_reason` ∈ {`recovered`, `next_disaster`, `90_day_cap`, `no_drawdown`}.

### 3.6 Stored columns

```
Y3_ASPI_Recovery_Time      duration = T_observed
Y3_event_observed          1 if a genuine recovery was observed, else 0
Y3_censor_reason
Y3_drawdown_occurred       D
```

`Y3_drawdown_occurred` is derived from `P1…P5` and is therefore future information. It is
retained as an outcome and as the stage-1 label. It is **never** a predictor.

### 3.7 Primary model output

Not a point duration. Preferred outputs:

- predicted median recovery time, in trading sessions
- `P(T ≤ 10)`, `P(T ≤ 20)`, `P(T ≤ 30)`, `P(T ≤ 60)`, `P(T ≤ 90)`

**Worked example from the specification.** `B = 10,000`; post-event closes
`9850, 9600, 9500, 9700, 9850, …, 10020` at k = 12 → `T_recovery = 12`.

---

## Prediction-time rule, all three targets

```
feature_timestamp ≤ prediction_origin
target_timestamp  >  prediction_origin
```

No feature may contain `P1…P5`, `V1…V5`, future disaster severity, future macroeconomic
values, future market values, or any value finalised after the prediction origin.

**Open caveat, recorded rather than hidden.** EM-DAT severity fields (total affected,
deaths, financial damage) are finalised days to months after the event, so a genuine
real-time forecaster could not use them. They remain in the current feature set. Any
result that depends on them is an ex-post attribution study, not a real-time forecast, and
must be labelled as such. A prediction-time-only variant would need estimates available at
`τ` instead.

---

## Change log

| date | commit | change |
|---|---|---|
| 2026-09-19 | `928a255` | Protocol frozen. Y1 endpoint moves from `P[t0+5]` to `P5 = P[t0+4]`; Y2 becomes a log ratio over `V1…V5`; Y3 baseline stays `P0` but the drawdown gate narrows to `P1…P5` and the recovery scan starts at `k = 1` rather than the trough. |
