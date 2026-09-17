"""Construction of the three research targets, one row per disaster event."""

from __future__ import annotations

import numpy as np
import pandas as pd

ADVERSE_RESPONSE_WINDOW = 5
DEFAULT_MAX_RECOVERY_DAYS = 90
ASPI_FORWARD_SESSIONS = 5
EVENT_WINDOW_SESSIONS = (10,)


def find_event_session(market: pd.DataFrame, event_date, date_col: str = "date"):
    """Position of the first trading session on or after the event date.

    Returns None when the event falls outside the series or on its very first
    session, because there would then be no pre event close to baseline against.
    """
    matching = market.index[market[date_col] >= event_date]
    if len(matching) == 0:
        return None
    position = market.index.get_loc(matching[0])
    return None if position == 0 else position


def calculate_aspi_percentage_change(market, position, pre_event_close, sessions,
                                     date_col="date", price_col="aspi_close"):
    """Forward log return in percent from the pre event close over `sessions` sessions.

    Returns (value, settlement_date). Both are missing when the series ends
    before the horizon closes, so a truncated window is never reported as a full one.
    """
    horizon_position = position + sessions
    if horizon_position >= len(market):
        return np.nan, pd.NaT
    horizon_close = market.iloc[horizon_position][price_col]
    value = float(100.0 * np.log(horizon_close / pre_event_close))
    return value, market.iloc[horizon_position][date_col]


def calculate_volume_crash_magnitude(market, position, volume_col="trading_volume",
                                     baseline_sessions=30):
    """Event day volume relative to its own trailing baseline, minus one.

    Returns NaN when the series carries no volume column, which is the case for
    the sector indices where the exchange publishes volume market wide only.
    """
    if volume_col not in market.columns:
        return np.nan
    baseline_start = max(0, position - baseline_sessions)
    baseline_mean = market.iloc[baseline_start:position][volume_col].mean()
    if not baseline_mean or np.isnan(baseline_mean):
        return np.nan
    return float((market.iloc[position][volume_col] / baseline_mean) - 1.0)


def find_competing_event_position(market, event_dates_sorted, row_index, date_col="date"):
    """Reference session of the next qualifying disaster, or None if this is the last.

    Found independently rather than assumed to be the next row, because two
    qualifying disasters can share or straddle a single trading session.
    """
    if row_index + 1 >= len(event_dates_sorted):
        return None
    matching = market.index[market[date_col] >= event_dates_sorted[row_index + 1]]
    if len(matching) == 0:
        return None
    return market.index.get_loc(matching[0])


def calculate_market_recovery_days(market, position, pre_event_baseline, effective_cap,
                                   max_recovery_days, price_col="aspi_close",
                                   gate_sessions=ADVERSE_RESPONSE_WINDOW):
    """Trading days until the index regains its pre event level, with censoring detail."""
    gate_end = min(position + gate_sessions, position + effective_cap, len(market) - 1)
    gate_window = market.iloc[position:gate_end + 1]
    trough_label = gate_window[price_col].idxmin()
    trough_position = market.index.get_loc(trough_label)
    trough_price = float(market.loc[trough_label, price_col])
    drawdown_occurred = bool(trough_price < pre_event_baseline)

    if not drawdown_occurred:
        return 0.0, False, "recovered", False

    recovery_window = market.iloc[trough_position:position + effective_cap + 1]
    recovered = recovery_window[recovery_window[price_col] >= pre_event_baseline]
    if recovered.empty:
        reason = "next_disaster" if effective_cap < max_recovery_days else "cap_90"
        return float(effective_cap), True, reason, True

    recovery_position = market.index.get_loc(recovered.index[0])
    return float(min(recovery_position - position, effective_cap)), False, "recovered", True


def build_event_targets(market_df, disaster_df, date_col="date", price_col="aspi_close",
                        volume_col="trading_volume", disaster_date_col="event_date",
                        max_recovery_days=DEFAULT_MAX_RECOVERY_DAYS) -> pd.DataFrame:
    """One row per qualifying event holding all three targets and their purge dates.

    Also returns the censoring detail for target three and the pre registered ten
    session variant of target one. Events that cannot be aligned are dropped.
    """
    market = market_df.copy().sort_values(date_col)
    market[date_col] = pd.to_datetime(market[date_col])

    events = disaster_df.copy().sort_values(disaster_date_col)
    events[disaster_date_col] = pd.to_datetime(events[disaster_date_col])
    event_dates_sorted = events[disaster_date_col].tolist()

    rows = []
    for row_index, (_, event) in enumerate(events.iterrows()):
        event_date = event[disaster_date_col]
        position = find_event_session(market, event_date, date_col)
        if position is None:
            continue

        pre_event_close = market.iloc[position - 1][price_col]

        aspi_change, aspi_end_date = calculate_aspi_percentage_change(
            market, position, pre_event_close, ASPI_FORWARD_SESSIONS, date_col, price_col)
        volume_crash = calculate_volume_crash_magnitude(market, position, volume_col)
        volume_label_end_date = market.iloc[position][date_col]

        competing_position = find_competing_event_position(
            market, event_dates_sorted, row_index, date_col)
        effective_cap = max_recovery_days
        if competing_position is not None:
            effective_cap = max(0, min(max_recovery_days, competing_position - position))

        recovery_days, censored, censor_reason, drawdown_occurred = (
            calculate_market_recovery_days(market, position, pre_event_close, effective_cap,
                                           max_recovery_days, price_col))
        recovery_end_position = min(position + int(recovery_days), len(market) - 1)

        event_windows, event_window_end_dates = {}, {}
        for sessions in EVENT_WINDOW_SESSIONS:
            value, end_date = calculate_aspi_percentage_change(
                market, position, pre_event_close, sessions, date_col, price_col)
            event_windows[f"Y1_EventWindow_0_{sessions}_LogReturn_Pct"] = value
            event_window_end_dates[f"Y1_EventWindow_0_{sessions}_horizon_end_date"] = end_date

        rows.append({
            disaster_date_col: event_date,
            "Y1_ASPI_5D_Forward_LogReturn_Pct": aspi_change,
            "Y1_horizon_end_date": aspi_end_date,
            "Y2_label_end_date": volume_label_end_date,
            "Y3_label_end_date": market.iloc[recovery_end_position][date_col],
            "Y2_abnormal_volume": volume_crash,
            "Y3_recovery_days": float(recovery_days),
            "Y3_censored": censored,
            "Y3_censor_reason": censor_reason,
            "Y3_drawdown_occurred": drawdown_occurred,
            **event_windows,
            **event_window_end_dates,
        })

    return pd.DataFrame(rows)
