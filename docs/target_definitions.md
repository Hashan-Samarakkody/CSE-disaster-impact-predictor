# Target definitions — as implemented

Written for: the researcher, to adapt into thesis §3.2.2 and Table 4.

This document states exactly what the code computes, in a form precise enough that a
second person could reproduce the dataset from this page alone. It replaces the
underspecified wording currently in §3.2.2, which defines Y1 as a daily return
(`ln(P_t / P_{t−1})`) and leaves Y2's window and Y3's baseline unstated.

Implementation: `src/targets/event_targets.py`. Dataset: 74 events, 2000-09-18 to
2025-11-27.

---

## 1. Event alignment

Let `t0` be the **first valid CSE trading session on or after the recorded disaster date**.

A disaster falling on a Saturday, Sunday or market holiday advances `t0` to the next real
trading session. No observation is fabricated for a non-trading day.

`P[k]` denotes the ASPI close at session offset `k` from `t0`, and `V[k]` the market-wide
traded volume (the sum of every listed security's share volume) at that session. So `P[−1]`
is the last pre-event close and `P[0]` is the close on `t0` itself.

**Counting convention.** Offsets are relative to `t0`, which carries offset 0. `t0+5` is
therefore the sixth post-event session. State this explicitly in the thesis — it is the
single most common source of irreproducibility in event studies.

An event is excluded when `t0` cannot be located in the market series, or when `t0` is the
first session of the series (no `P[−1]` exists to baseline against).

---

## 2. Y1 — ASPI percentage change

```
Y1_5D  = 100 · ln( P[+5] / P[−1] )
Y1_10D = 100 · ln( P[+10] / P[−1] )
```

The continuously compounded return, in percent, from the last pre-event close to the
close five (respectively ten) sessions after the disaster session.

|                  |                                                                               |
| ---------------- | ----------------------------------------------------------------------------- |
| units            | percent (log points)                                                          |
| support          | unbounded                                                                     |
| sessions spanned | **Y1_5D spans 7 sessions**, `t0−1` through `t0+5`; Y1_10D spans 12 |
| label-end date   | `date[t0+5]` / `date[t0+10]`                                              |
| missing when     | fewer than 5 (resp. 10) sessions remain in the series after`t0`             |

**Wording caution.** Calling `Y1_5D` a "five-day window" is inaccurate — the return covers
six steps across seven sessions. Either rename it (e.g. "five-session forward return from
the pre-event close") or change the endpoint to `P[+4]`, which genuinely spans five
post-event sessions. Do not leave the label and the formula disagreeing.

`Y1_10D` is a pre-registered sensitivity analysis of Y1, not a fourth research target.

### Observed values

|        | n  | mean   | sd    | min      | p25     | median | p75    | max     |
| ------ | -- | ------ | ----- | -------- | ------- | ------ | ------ | ------- |
| Y1 5D  | 74 | +0.433 | 3.053 | −7.203  | −1.072 | +0.120 | +1.775 | +9.009  |
| Y1 10D | 74 | +0.749 | 4.983 | −11.981 | −2.053 | +0.349 | +3.888 | +16.150 |

45.9% of events are negative at 5 sessions, 48.6% at 10. The distribution is close to
symmetric around zero — disasters do not, on average, move the CSE in a fixed direction.

---

## 3. Y2 — abnormal trading volume

```
V_base = mean( V[-30] … V[-1] )          30 sessions, t0 excluded

Y2 = mean( V[0] … V[+4] ) / V_base - 1
```

Mean market-wide traded volume over the **five post-event sessions**, `t0` counted as the
first, relative to its own trailing 30-session mean, minus one.

| | |
|---|---|
| units | ratio (0.50 means 50% above the baseline) |
| support | `[-1, inf)` — volume cannot be negative |
| response window | `t0` … `t0+4`, exactly 5 sessions |
| baseline window | `t0-30` … `t0-1`, exactly 30 sessions, `t0` excluded |
| label-end date | `date[t0+4]` |
| missing when | the volume series is unavailable, or fewer than 5 sessions remain after `t0` |

**Label-end date.** `date[t0+4]`, the final session of the response window — not `t0`. The
walk-forward purge reads this column to drop training events whose label reaches into a
test period. Dating it at `t0` would under-purge by four sessions and let test-period
information into training. This is a correctness requirement, not a convention.

**Window alignment.** Y2 closes at `t0+4`; Y1_5D closes at `t0+5`. The two "5-day" targets
therefore end on different sessions, because Y1 is a return measured from `P[-1]` while Y2
is a mean over five post-event sessions. If the thesis needs them aligned, either extend
Y2 to `V[0] … V[+5]` or move Y1's endpoint to `P[+4]`.

**Naming.** "Volume crash magnitude" presupposes a negative sign. 37.7% of observed values
are positive. Prefer "abnormal trading volume" or "post-disaster volume deviation".

### Observed values

| n | mean | sd | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|---|
| 61 | +0.032 | 0.600 | -0.769 | — | -0.080 | — | +2.066 |

Observed for 61 of 74 events. The 13 missing are the 2000 archive year and post-2023
events, where the exchange publishes the index level but not market-wide volume. These
must stay missing — never zero-filled, which would assert "volume exactly at baseline."

---

## 4. Y3 — market recovery sessions

Baseline for recovery is the **last pre-event close**, `P[−1]`.

Three steps:

1. **Adverse-response gate.** Examine `P[0] … P[+5]`. Let the trough be the minimum close
   in that window. A qualifying drawdown occurs when `trough < P[−1]`.
2. **No qualifying drawdown** → `Y3 = 0`, `drawdown_occurred = False`. The market never
   fell below its pre-event level, so no recovery duration is defined. This is a recorded
   state, not a fabricated duration of zero days.
3. **Drawdown occurred** → search forward from the trough session for the first session
   where `P[k] ≥ P[−1]`. Then `Y3 = k`, counted from `t0`.

```
Y3 = k  such that  P[k] ≥ P[−1],  k measured from t0,
     searching forward from the trough of P[0..+5]
```

|                |                                                        |
| -------------- | ------------------------------------------------------ |
| units          | CSE trading sessions                                   |
| support        | `[0, 90]`                                            |
| label-end date | `date[t0 + Y3]` — the recovery or censoring session |

**Two distinct origins.** The recovery *search* begins at the trough; the recovery *count*
runs from `t0`. These are different points and must not be conflated. State both.

### Censoring

Recovery that is not observed is **right-censored**, not recorded as a genuine 90-session
recovery:

| `Y3_censor_reason` | meaning                                        | `Y3_censored` | n  |
| -------------------- | ---------------------------------------------- | --------------- | -- |
| `recovered`        | recovery observed, or no drawdown occurred     | False           | 57 |
| `cap_90`           | still below`P[−1]` after 90 sessions        | True            | 8  |
| `next_disaster`    | a further qualifying disaster intervened first | True            | 9  |

The competing-event case matters: when a second disaster arrives before the first has
recovered, the follow-up is truncated at that session rather than attributing the second
event's market movement to the first.

Note a labelling quirk in the current implementation: events with **no drawdown** are given
`censor_reason = "recovered"`, which is misleading — nothing was recovered because nothing
fell. If you revise, `"no_drawdown"` is the clearer value. It does not affect any
computation, only readability.

Any censoring-aware model (survival, hurdle) must read `Y3_censored`, not infer censoring
from `Y3 >= 90`.

### Observed values

| n  | mean   | sd     | min | p25 | median | p75   | max |
| -- | ------ | ------ | --- | --- | ------ | ----- | --- |
| 74 | 16.554 | 27.654 | 0   | 0   | 4.5    | 16.75 | 90  |

- 51 of 74 events (68.9%) show a qualifying drawdown.
- 23 events (31.1%) record `Y3 = 0` — no drawdown.
- 17 events (23.0%) are censored: 8 at the cap, 9 by a competing disaster.
- Among the 34 events with an **observed** recovery: median 5 sessions, p75 12, max 34.

The distribution is right-skewed with a point mass at zero. Any model whose prediction
carries a floor — for example a hurdle whose expectation includes a `(1 − P) · 90` term —
is structurally disadvantaged against it.

---

## 5. Label-end dates

Every target carries the session on which its value becomes fully known. The walk-forward
purge reads these to drop training events whose label reaches into a test period.

| target | column                                   | value             |
| ------ | ---------------------------------------- | ----------------- |
| Y1 5D  | `Y1_horizon_end_date`                  | `date[t0+5]`    |
| Y1 10D | `Y1_EventWindow_0_10_horizon_end_date` | `date[t0+10]`   |
| Y2     | `Y2_label_end_date`                    | `date[t0]`      |
| Y3     | `Y3_label_end_date`                    | `date[t0 + Y3]` |

If any target's window changes, its label-end date must change with it. This is not
cosmetic — a stale label-end date is a leakage defect.

---

## 6. Worked example — 2004 Indian Ocean tsunami

Disaster date 2004-12-26, a Sunday. 2004-12-27 was a market holiday, so `t0` advances to
**2004-12-28**.

```
P[−1]  = 1,572.55   (2004-12-23)    V_base = mean(V[−30..−1]) = 7,810,033 shares
P[0]   = 1,504.41   (2004-12-28)    V[0]   = 4,374,300 shares
P[+5]  = 1,524.75   (2005-01-04)
P[+10] = 1,562.05   (2005-01-11)

Y1_5D  = 100 · ln(1524.75 / 1572.55)   = −3.0868
Y1_10D = 100 · ln(1562.05 / 1572.55)   = −0.6699
Y2     = mean(V[0..+4]) / 7,810,033 - 1 = -0.3608
Y3     = 12 sessions (drawdown occurred, recovery observed)
```

Y1 and Y2 were reproduced by hand from the raw exchange workbook and matched the code to
the digit.

Read the numbers: five sessions after the tsunami the ASPI stood 3.09% below its
pre-event close; ten sessions after, 0.67% below. Turnover over the five post-event sessions ran 36% below its
30-session norm. The index regained its pre-event level twelve sessions after the
disaster.

Note what the event day itself shows: the ASPI fell from 1,572.55 to 1,504.41 on `t0`, a
4.3% single-session drop, and turnover halved. The index then recovered most of that
within two weeks. This is the pattern the targets are built to measure.

---

## 7. Constants

```
pre-event volume baseline      30 trading sessions
Y2 response window              5 sessions (t0 .. t0+4)
Y1 primary horizon              5 sessions (endpoint t0+5)
Y1 sensitivity horizon         10 sessions (endpoint t0+10)
adverse-response gate           6 sessions (t0 .. t0+5)
maximum recovery follow-up     90 trading sessions
```

Note the gate spans six sessions, `t0` through `t0+5`, not five. If the thesis describes a
five-session detection window, either the text or the constant needs to change.

---

## 8. Checklist before the thesis goes out

- [ ] §3.2.2 Y1 no longer says `ln(P_t / P_{t−1})` — that is a one-day return
- [ ] Y1's label matches its span (7 sessions as implemented, or move the endpoint to `t0+4`)
- [ ] Y2's units stated as a ratio, not a percentage, and not "dollar value" (§1.2.4 says dollars; volume is shares)
- [ ] Y3's baseline named explicitly as `P[−1]`
- [ ] Y3's two origins stated: search from trough, count from `t0`
- [ ] Y3's no-drawdown convention stated: `Y3 = 0` with `drawdown_occurred = False`
- [ ] Censoring distinguished from capping
- [ ] "Trading sessions" used throughout, never "days" (§3.2.1 currently says days)
- [ ] Table 4 target rows replaced
- [ ] Figure 8 output labels updated
