"""Construction of the three research targets, one row per disaster event.

Implements docs/TARGET_DEFINITION_PROTOCOL.md, frozen at commit 928a255 on 2026-09-19
before any model performance under these definitions was observed.

Prediction origin. tau is the time the disaster becomes known. The first COMPLETE trading
session after tau is `position` in this module's indexing, so:

    P0 = market[position - 1]          Pk = market[position + k - 1]
    V_{-i} = market[position - i]      Vi = market[position + i - 1]

Everything at or before P0 is knowable at prediction time; everything from P1 onward is
target-only.

    Y1 = 100 * ln(P5 / P0)
    Y2 = ln( mean(V1..V5) / mean(V_{-30}..V_{-1}) )
    Y3 = time-to-event: sessions from the origin until Pk >= P0 after a prior dip below it,
         right-censored at 90 sessions or at the next qualifying disaster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE_SESSIONS = 30
ASPI_FORWARD_SESSIONS = 5
# 1 session is a robustness test (Revision 2, T4); 10/15/20 are the original
# pre-declared sensitivities. None is ever promoted to primary.
ASPI_SENSITIVITY_HORIZONS = (1, 10, 15, 20)
VOLUME_RESPONSE_SESSIONS = 5
# Response-window sensitivities for Y2 (Revision 2, T5), against the same 30-session
# baseline. The 5-session window stays primary; none of these is ever promoted.
VOLUME_SENSITIVITY_WINDOWS = (1, 10, 20)
# A priori winsorisation limits for the Y2 outlier-sensitivity column. Fixed before any
# result was seen, never tuned.
VOLUME_WINSOR_QUANTILES = (0.05, 0.95)
DRAWDOWN_WINDOW = 5
MAX_RECOVERY_SESSIONS = 90

# Kept because callers pass max_recovery_days= explicitly.
DEFAULT_MAX_RECOVERY_DAYS = MAX_RECOVERY_SESSIONS
ADVERSE_RESPONSE_WINDOW = DRAWDOWN_WINDOW


def find_event_session(market: pd.DataFrame, event_date, date_col: str = "date"):
    """Row index of the first complete trading session on or after the event date.

    That session carries P1. Returns None when the event falls outside the series, or on
    its very first session, because P0 would then not exist.
    """
    matching = market.index[market[date_col] >= event_date]
    if len(matching) == 0:
        return None
    position = market.index.get_loc(matching[0])
    return None if position == 0 else position


def _session(market, position, k):
    """Row index of Pk / Vk, or None when the series ends before it.

    k >= 1 indexes forward sessions; k = 0 is P0, the last session before the origin.
    """
    row = position + k - 1 if k >= 1 else position - 1
    return None if row >= len(market) else row


def calculate_aspi_forward_log_return(market, position, p0, sessions,
                                      date_col="date", price_col="aspi_close"):
    """Y1 = 100 * ln(P_sessions / P0). Returns (value, label_end_date).

    Both are missing when the series ends before the horizon closes, so a truncated
    window is never reported as a full one.
    """
    row = _session(market, position, sessions)
    if row is None:
        return np.nan, pd.NaT
    close = float(market.iloc[row][price_col])
    if not np.isfinite(close) or not np.isfinite(p0) or close <= 0 or p0 <= 0:
        return np.nan, pd.NaT
    return float(100.0 * np.log(close / p0)), market.iloc[row][date_col]


def calculate_forward_abnormal_volume(market, position, volume_col="trading_volume",
                                      baseline_sessions=BASELINE_SESSIONS,
                                      response_sessions=VOLUME_RESPONSE_SESSIONS,
                                      date_col="date"):
    """Y2 = ln( mean(V1..V5) / mean(V_{-30}..V_{-1}) ).

    Returns (value, label_end_date, v_base, v_future); the label end date is V5's session,
    which the purge reads. NaN when the frame carries no volume column, when the baseline
    is short or non-positive, or when the response window runs off the end of the series.
    """
    if volume_col not in market.columns:
        return np.nan, pd.NaT, np.nan, np.nan
    if position < baseline_sessions:
        return np.nan, pd.NaT, np.nan, np.nan
    base_window = market.iloc[position - baseline_sessions:position][volume_col]
    if len(base_window) != baseline_sessions or base_window.isna().any():
        return np.nan, pd.NaT, np.nan, np.nan
    v_base = float(base_window.mean())
    if not np.isfinite(v_base) or v_base <= 0:
        return np.nan, pd.NaT, np.nan, np.nan

    end = _session(market, position, response_sessions)
    if end is None:
        return np.nan, pd.NaT, v_base, np.nan
    window = market.iloc[position:end + 1][volume_col]
    if window.isna().any():
        return np.nan, pd.NaT, v_base, np.nan
    v_future = float(window.mean())
    if not np.isfinite(v_future) or v_future <= 0:
        return np.nan, pd.NaT, v_base, np.nan
    return float(np.log(v_future / v_base)), market.iloc[end][date_col], v_base, v_future


def find_competing_event_position(market, event_dates_sorted, row_index, date_col="date"):
    """Reference session of the next qualifying disaster, or None if this is the last.

    Found independently rather than assumed to be the next row, because two qualifying
    disasters can share or straddle a single trading session.
    """
    if row_index + 1 >= len(event_dates_sorted):
        return None
    matching = market.index[market[date_col] >= event_dates_sorted[row_index + 1]]
    if len(matching) == 0:
        return None
    return market.index.get_loc(matching[0])


def calculate_recovery_time(market, position, b, t_next=None,
                            max_sessions=MAX_RECOVERY_SESSIONS,
                            drawdown_window=DRAWDOWN_WINDOW, price_col="aspi_close"):
    """Y3 as a time-to-event outcome, per TARGET_DEFINITION_PROTOCOL.md section 3.

    Returns (duration, event_observed, censor_reason, drawdown_occurred).
    D = 0 is a distinct state, not a zero-length recovery: duration 0 with no event
    observed, so a survival model cannot read it as an instant recovery.
    """
    # The scan starts at k = 1, not at the trough. Anchoring on the trough would skip a
    # genuine recovery that precedes a later, deeper dip.
    if not np.isfinite(b) or b <= 0:
        return np.nan, 0, "no_baseline", False

    gate_end = _session(market, position, drawdown_window)
    if gate_end is None:
        gate_end = len(market) - 1
    gate = market.iloc[position:gate_end + 1][price_col]
    if gate.empty or gate.isna().all():
        return np.nan, 0, "no_baseline", False
    drawdown = bool(gate.min() < b)
    if not drawdown:
        return 0.0, 0, "no_drawdown", False

    horizon = min(position + max_sessions - 1, len(market) - 1)
    closes = market.iloc[position:horizon + 1][price_col].to_numpy(dtype=float)
    t_recovery = None
    dipped = False
    for k, close in enumerate(closes, start=1):
        if dipped and close >= b:
            t_recovery = k
            break
        if close < b:
            dipped = True

    cap = float(max_sessions)
    nxt = float(t_next) if t_next is not None else np.inf
    if t_recovery is None:
        return float(min(cap, nxt)), 0, ("next_disaster" if nxt < cap else "90_day_cap"), True
    if t_recovery < nxt and t_recovery <= max_sessions:
        return float(t_recovery), 1, "recovered", True
    return float(min(nxt, cap)), 0, ("next_disaster" if nxt <= cap else "90_day_cap"), True


def volume_window_col(window: int) -> str:
    """Dataset column holding Y2 over a `window`-session response window."""
    return f"Y2_{window}D_Forward_AbnormalVolume_LogRatio"


def volume_window_end_col(window: int) -> str:
    """Dataset column holding the session on which the `window`-session Y2 closes."""
    if window == VOLUME_RESPONSE_SESSIONS:
        return "Y2_horizon_end_date"
    return f"Y2_{window}D_horizon_end_date"


def abnormal_volume_percent(log_ratio):
    """Y2 as a percentage deviation from baseline turnover: 100 * (exp(Y2) - 1).

    The reader-facing form. A log ratio of 0.4055 is +50 percent.
    """
    return 100.0 * (np.expm1(np.asarray(log_ratio, dtype=float)))


def winsorise(values, quantiles=VOLUME_WINSOR_QUANTILES):
    """Clip to the given quantiles of the OBSERVED values, leaving NaN untouched.

    The quantiles are full-sample, so the result is a descriptive sensitivity column.
    A model would need in-fold quantiles instead.
    """
    arr = np.asarray(values, dtype=float)
    observed = arr[np.isfinite(arr)]
    if len(observed) == 0:
        return arr
    low, high = np.quantile(observed, quantiles)
    return np.where(np.isfinite(arr), np.clip(arr, low, high), arr)


def build_event_targets(market_df, disaster_df, date_col="date", price_col="aspi_close",
                        volume_col="trading_volume", disaster_date_col="event_date",
                        max_recovery_days=MAX_RECOVERY_SESSIONS) -> pd.DataFrame:
    """One row per qualifying event: the three targets, their label-end dates and the
    inputs needed to recompute each by hand.
    """
    market = market_df.copy().sort_values(date_col).reset_index(drop=True)
    market[date_col] = pd.to_datetime(market[date_col])

    events = disaster_df.copy().sort_values(disaster_date_col)
    events[disaster_date_col] = pd.to_datetime(events[disaster_date_col])
    event_dates_sorted = events[disaster_date_col].tolist()

    rows = []
    for row_index, (_, event) in enumerate(events.iterrows()):
        position = find_event_session(market, event[disaster_date_col], date_col)
        if position is None:
            continue

        p0 = float(market.iloc[position - 1][price_col])
        y1, y1_end = calculate_aspi_forward_log_return(
            market, position, p0, ASPI_FORWARD_SESSIONS, date_col, price_col)
        y2, y2_end, v_base, v_future = calculate_forward_abnormal_volume(
            market, position, volume_col, date_col=date_col)

        competing = find_competing_event_position(
            market, event_dates_sorted, row_index, date_col)
        t_next = None if competing is None else max(0, competing - position + 1)

        duration, event_observed, censor_reason, drawdown = calculate_recovery_time(
            market, position, p0, t_next, max_recovery_days, price_col=price_col)

        offset = 0 if not np.isfinite(duration) else int(duration)
        y3_end = market.iloc[min(position + max(offset - 1, 0), len(market) - 1)][date_col]

        row = {
            disaster_date_col: event[disaster_date_col],
            "prediction_origin_session": market.iloc[position][date_col],
            "P0": p0,
            "Y1_ASPI_5D_Forward_LogReturn_Pct": y1,
            "Y1_horizon_end_date": y1_end,
            "Y2_5D_Forward_AbnormalVolume_LogRatio": y2,
            "Y2_horizon_end_date": y2_end,
            "Y2_V_base": v_base,
            "Y2_V_future5": v_future,
            "Y3_ASPI_Recovery_Time": duration,
            "Y3_event_observed": int(event_observed),
            # ponytail: censoring is the complement of event_observed. Emitted so the
            # existing survival/hurdle consumers keep reading one flag instead of every
            # call site learning the inversion.
            "Y3_censored": bool(not event_observed),
            "Y3_censor_reason": censor_reason,
            "Y3_drawdown_occurred": bool(drawdown),
            "Y3_label_end_date": y3_end,
        }
        for h in ASPI_SENSITIVITY_HORIZONS:
            value, end = calculate_aspi_forward_log_return(
                market, position, p0, h, date_col, price_col)
            row[f"Y1_ASPI_{h}D_Forward_LogReturn_Pct"] = value
            row[f"Y1_{h}D_horizon_end_date"] = end
        for w in VOLUME_SENSITIVITY_WINDOWS:
            value, end, _, _ = calculate_forward_abnormal_volume(
                market, position, volume_col, response_sessions=w, date_col=date_col)
            row[volume_window_col(w)] = value
            row[volume_window_end_col(w)] = end
        rows.append(row)

    frame = pd.DataFrame(rows)
    if len(frame):
        # Reader-facing and outlier-robust views of the primary Y2. Both derived, so a
        # missing volume label stays missing rather than becoming a zero.
        primary = frame[volume_window_col(VOLUME_RESPONSE_SESSIONS)]
        frame["Y2_5D_Forward_AbnormalVolume_Pct"] = abnormal_volume_percent(primary)
        frame["Y2_5D_Forward_AbnormalVolume_LogRatio_Winsorised"] = winsorise(primary)
    return frame


if __name__ == "__main__":
    # Every assertion below is a worked example from docs/TARGET_DEFINITION_PROTOCOL.md
    # with its expected value computed by hand.
    sessions = pd.date_range("2020-01-01", periods=200, freq="B")

    def frame(prices, volumes=None):
        n = len(prices)
        out = {"date": sessions[:n], "aspi_close": np.asarray(prices, dtype=float)}
        if volumes is not None:
            out["trading_volume"] = np.asarray(volumes, dtype=float)
        return pd.DataFrame(out)

    # Y1: the protocol's own example. P0 = 10,000; P5 = 9,700 -> -3.0459
    prices = [10_000.0] * 30 + [9_900.0, 9_880.0, 9_860.0, 9_840.0, 9_700.0] + [9_700.0] * 100
    m = frame(prices)
    y1, end = calculate_aspi_forward_log_return(m, 30, 10_000.0, 5)
    assert abs(y1 - 100 * np.log(9700 / 10000)) < 1e-12, y1
    assert abs(y1 - (-3.0459)) < 1e-3, y1
    # P5 is the FIFTH complete session after the origin, i.e. row position+4.
    assert end == sessions[34], end

    # Y2: the protocol's own example. V_base = 20e6, V_future5 = 30e6 -> 0.4055
    vols = [20e6] * 30 + [30e6] * 5 + [20e6] * 100
    m = frame([100.0] * 135, vols)
    y2, end, vb, vf = calculate_forward_abnormal_volume(m, 30)
    assert abs(y2 - np.log(30 / 20)) < 1e-12, y2
    assert abs(y2 - 0.4055) < 1e-4, y2
    assert vb == 20e6 and vf == 30e6
    assert abs(100 * (np.exp(y2) - 1) - 50.0) < 1e-9        # reads as +50%
    assert end == sessions[34], end                          # V5, not the origin

    # Y3: the protocol's own example. B = 10,000, recovery at k = 12
    path = [9_850.0, 9_600.0, 9_500.0, 9_700.0, 9_850.0, 9_900.0,
            9_910.0, 9_920.0, 9_930.0, 9_940.0, 9_950.0, 10_020.0]
    m = frame([10_000.0] * 30 + path + [10_020.0] * 100)
    dur, ev, reason, d = calculate_recovery_time(m, 30, 10_000.0)
    assert (dur, ev, reason, d) == (12.0, 1, "recovered", True), (dur, ev, reason, d)

    # No drawdown inside P1..P5 is its own state, NOT a zero-length recovery.
    m = frame([100.0] * 30 + [101.0, 102.0, 103.0, 104.0, 105.0] + [90.0] * 100)
    dur, ev, reason, d = calculate_recovery_time(m, 30, 100.0)
    assert (dur, ev, reason, d) == (0.0, 0, "no_drawdown", False), (dur, ev, reason, d)

    # Never recovers -> right-censored at the 90-session cap, not a genuine 90.
    m = frame([100.0] * 30 + [90.0] * 150)
    dur, ev, reason, d = calculate_recovery_time(m, 30, 100.0)
    assert (dur, ev, reason, d) == (90.0, 0, "90_day_cap", True), (dur, ev, reason, d)

    # Competing disaster before recovery -> censored at T_next, event NOT observed.
    m = frame([100.0] * 30 + [95.0] * 20 + [101.0] * 100)
    dur, ev, reason, d = calculate_recovery_time(m, 30, 100.0, t_next=10)
    assert (dur, ev, reason, d) == (10.0, 0, "next_disaster", True), (dur, ev, reason, d)
    # ... but a recovery BEFORE the competing event still counts.
    dur, ev, reason, d = calculate_recovery_time(m, 30, 100.0, t_next=40)
    assert (dur, ev, reason, d) == (21.0, 1, "recovered", True), (dur, ev, reason, d)

    # The scan starts at k=1, not at the trough: 95, 101, 90 recovers at k=2, and a
    # trough-anchored scan (trough at k=3) would have missed it.
    m = frame([100.0] * 30 + [95.0, 101.0, 90.0, 90.0, 90.0] + [90.0] * 100)
    assert calculate_recovery_time(m, 30, 100.0)[0] == 2.0

    # end to end
    prices = [100.0] * 30 + [99.0, 98.0, 97.0, 98.0, 99.0, 101.0] + [101.0] * 100
    vols = [1_000.0] * 30 + [2_000.0] * 5 + [1_000.0] * 101
    market = pd.DataFrame({"date": sessions[:len(prices)], "aspi_close": prices,
                           "trading_volume": vols})
    t0 = market["date"].iloc[30]
    row = build_event_targets(market, pd.DataFrame({"event_date": [t0]})).iloc[0]
    assert row["P0"] == 100.0
    assert abs(row["Y1_ASPI_5D_Forward_LogReturn_Pct"] - 100 * np.log(99 / 100)) < 1e-12
    assert abs(row["Y2_5D_Forward_AbnormalVolume_LogRatio"] - np.log(2.0)) < 1e-12
    assert row["Y1_horizon_end_date"] == sessions[34]
    assert row["Y2_horizon_end_date"] == sessions[34]
    assert row["Y3_ASPI_Recovery_Time"] == 6.0 and row["Y3_event_observed"] == 1
    assert row["Y3_drawdown_occurred"]

    # A horizon running off the end of the series is missing, never truncated.
    late = build_event_targets(market, pd.DataFrame({"event_date": [market["date"].iloc[-2]]}))
    assert np.isnan(late["Y1_ASPI_5D_Forward_LogReturn_Pct"].iloc[0])

    # Y2 response-window sensitivities (T5). Volume is 2000 for the 5 sessions after the
    # origin and 1000 before it, so the 1 and 5 session windows both read ln(2).
    assert abs(row[volume_window_col(1)] - np.log(2.0)) < 1e-12
    assert abs(row[volume_window_col(5)] - np.log(2.0)) < 1e-12
    assert row[volume_window_end_col(1)] == sessions[30]
    assert row[volume_window_end_col(5)] == sessions[34]
    # The 10 and 20 session windows span the return to normal volume, so they sit lower.
    assert row[volume_window_col(10)] < row[volume_window_col(5)]

    # The percentage form is the protocol's own worked example: ln(1.5) reads as +50%.
    assert abs(abnormal_volume_percent(np.log(1.5)) - 50.0) < 1e-9
    assert abs(abnormal_volume_percent(0.4055) - 50.0) < 1e-2

    # Winsorising clips the tails and leaves missing values missing.
    raw = np.array([np.nan, -10.0, -0.1, 0.0, 0.1, 10.0])
    clipped = winsorise(raw)
    assert np.isnan(clipped[0])
    assert clipped[1] > -10.0 and clipped[-1] < 10.0
    assert clipped[2] == -0.1 and clipped[4] == 0.1

    # A frame with no volume column at all must carry missing Y2 everywhere, never zero.
    no_volume = pd.DataFrame({"date": sessions[:140], "aspi_close": [100.0] * 140})
    nv = build_event_targets(no_volume, pd.DataFrame({"event_date": [sessions[40]]}))
    for _w in (1, 5, 10, 20):
        assert nv[volume_window_col(_w)].isna().all(), _w
    assert nv["Y2_5D_Forward_AbnormalVolume_Pct"].isna().all()
    assert nv["Y2_5D_Forward_AbnormalVolume_LogRatio_Winsorised"].isna().all()

    print("event_targets.py self-check passed (TARGET_DEFINITION_PROTOCOL worked examples)")
