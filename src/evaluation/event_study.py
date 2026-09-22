"""Event-study inference on the realised market response (Revision 2, T11).

This module answers a different question from `src/evaluation/verification.py`. That one
asks whether a model forecasts the response out of sample. This one asks whether there was
a response at all. A measurable realised reaction does not imply forecastability, and the
distinction between the two is the study's central contribution, so the two live apart and
neither imports the other's decision rules.

Event time. `k` indexes trading sessions relative to the prediction origin, using the same
convention as `docs/TARGET_DEFINITION_PROTOCOL.md`: `k = 1` is the first complete session
after the disaster becomes known and `k = 0` is the session carrying `P0`, the last close
observed before it. A row offset is therefore `position + k - 1` for every `k`.

Benchmark models. Returns use the single-factor market model from
`src/targets/abnormal_returns.py`. Volume uses the mean-adjusted model, the standard choice
when no factor exists for the series (Brown and Warner 1985).

Test statistics, all implemented as published:

* cross-sectional t-test on the cumulative abnormal returns;
* Boehmer, Musumeci and Poulsen (1991), the standardised-residual test, which is robust to
  the variance increase an event itself induces;
* Corrado (1989), the rank test, which assumes no particular distribution;
* Kolari and Pynnonen (2010), which corrects the standardised test for cross-sectional
  correlation. It matters here because Sri Lankan disasters cluster seasonally, so event
  windows overlap and the abnormal returns are not independent draws.

References
    Boehmer, E., Musumeci, J. and Poulsen, A. (1991) Journal of Financial Economics 30(2).
    Corrado, C. (1989) Journal of Financial Economics 23(2).
    Kolari, J. and Pynnonen, S. (2010) Review of Financial Studies 23(11).
    Brown, S. and Warner, J. (1985) Journal of Financial Economics 14(1).
    MacKinlay, A. C. (1997) Journal of Economic Literature 35(1).
    Kothari, S. P. and Warner, J. (2007) Handbook of Corporate Finance, vol. 1.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from src.targets.abnormal_returns import ESTIMATION_WINDOW, MIN_ESTIMATION_ROWS

# Sessions either side of the prediction origin that the event window spans. Five before
# and twenty after: wide enough to show the run-up and the recovery, and matched to the
# longest declared return horizon.
EVENT_WINDOW = (-5, 20)


class AbnormalPanel(NamedTuple):
    """Abnormal observations in event time, with what the tests need to weigh them."""

    abnormal: np.ndarray          # events x event-time sessions
    event_time: np.ndarray        # the k values, one per column
    resid_std: np.ndarray         # estimation-window residual sd, one per event
    n_estimation: np.ndarray      # usable estimation rows, one per event
    factor_deviation: np.ndarray  # events x sessions, market factor minus its window mean
    factor_ss: np.ndarray         # sum of squared factor deviations in the estimation window
    estimation_residuals: list    # per event, residual series indexed by calendar date
    event_ids: np.ndarray
    event_window_start: np.ndarray   # first calendar date of each event's window
    event_window_end: np.ndarray     # last calendar date of each event's window


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of y on x."""
    beta, alpha = np.polyfit(x, y, 1)
    return float(alpha), float(beta)


def build_abnormal_panel(daily: pd.DataFrame, event_dates, model: str = "market",
                         event_window: tuple[int, int] = EVENT_WINDOW,
                         estimation_window: int = ESTIMATION_WINDOW,
                         min_estimation: int = MIN_ESTIMATION_ROWS,
                         date_col: str = "date", value_col: str = "log_return",
                         factor_col: str = "sp500_log_return") -> AbnormalPanel:
    """Abnormal observations for every event across the event window.

    `model` is "market" (single-factor, for returns) or "mean" (mean-adjusted, for volume
    and any series with no factor). Events whose estimation window is too thin are dropped
    rather than estimated from a handful of rows.
    """
    if model not in {"market", "mean"}:
        raise ValueError(f"model must be 'market' or 'mean', got {model!r}")

    frame = daily.sort_values(date_col).reset_index(drop=True)
    dates = pd.to_datetime(frame[date_col])
    date_values = dates.to_numpy()
    values = frame[value_col].to_numpy(dtype=float)
    factor = (frame[factor_col].to_numpy(dtype=float) if model == "market"
              else np.zeros(len(frame)))

    k_values = np.arange(event_window[0], event_window[1] + 1)
    rows_ar, resid_std, n_est, dev, ss, residuals, kept = [], [], [], [], [], [], []
    window_start, window_end = [], []

    for event_id, event_date in enumerate(pd.to_datetime(pd.Series(list(event_dates)))):
        candidates = np.flatnonzero(date_values >= np.datetime64(event_date))
        if len(candidates) == 0 or candidates[0] == 0:
            continue
        position = int(candidates[0])

        last = position - 1
        first = max(0, last - estimation_window + 1)
        est_rows = np.arange(first, last + 1)
        usable = est_rows[np.isfinite(values[est_rows]) & np.isfinite(factor[est_rows])]
        if len(usable) < min_estimation:
            continue

        if model == "market":
            if np.var(factor[usable]) == 0:
                continue
            alpha, beta = _ols(factor[usable], values[usable])
        else:
            alpha, beta = float(np.mean(values[usable])), 0.0

        fitted_est = alpha + beta * factor[usable]
        resid = values[usable] - fitted_est
        ddof = max(len(usable) - (2 if model == "market" else 1), 1)
        sigma = float(np.sqrt(np.sum(resid ** 2) / ddof))

        factor_mean = float(np.mean(factor[usable]))
        factor_ss = float(np.sum((factor[usable] - factor_mean) ** 2))

        event_rows = position + k_values - 1
        inside = (event_rows >= 0) & (event_rows < len(values))
        ar = np.full(len(k_values), np.nan)
        fd = np.full(len(k_values), np.nan)
        safe = event_rows[inside]
        ar[inside] = values[safe] - (alpha + beta * factor[safe])
        fd[inside] = factor[safe] - factor_mean

        rows_ar.append(ar)
        dev.append(fd)
        resid_std.append(sigma)
        n_est.append(len(usable))
        ss.append(factor_ss)
        residuals.append(pd.Series(resid, index=dates.iloc[usable].to_numpy()))
        window_start.append(date_values[safe[0]])
        window_end.append(date_values[safe[-1]])
        kept.append(event_id)

    if not rows_ar:
        raise ValueError("no event had a usable estimation window")
    return AbnormalPanel(np.vstack(rows_ar), k_values, np.array(resid_std),
                         np.array(n_est), np.vstack(dev), np.array(ss),
                         residuals, np.array(kept),
                         np.array(window_start), np.array(window_end))


def average_cross_correlation(panel: AbnormalPanel) -> float:
    """Average pairwise correlation of the abnormal returns across the cross-section.

    The quantity the Kolari and Pynnonen formula needs. Their estimator is the average
    correlation of the estimation-window residuals ACROSS SECURITIES, and it does not
    transfer to this design unchanged: this study has one security, the ASPI, observed at
    74 separate events, so any two events' residual series are the same series on their
    shared calendar dates and correlate at essentially 1.0 by construction. Using that
    number would drive every corrected statistic to zero and say nothing.

    What actually makes two events dependent here is an overlap of their event windows, in
    which case they are built from literally the same sessions and their abnormal returns
    are perfectly dependent; events whose windows do not overlap sit on different sessions
    of a near-uncorrelated series and are independent. The average pairwise correlation is
    therefore the share of pairs whose event windows overlap. The Kolari and Pynnonen
    formula itself is used exactly as published; only this estimator of r is adapted to a
    single-security design, and the adaptation is recorded rather than hidden.
    """
    starts = panel.event_window_start
    ends = panel.event_window_end
    n = len(starts)
    if n < 2:
        return 0.0
    overlapping = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if starts[i] <= ends[j] and starts[j] <= ends[i]:
                overlapping += 1
    return float(overlapping / total) if total else 0.0


def _two_sided_p(statistic: float, dof: Optional[int] = None) -> float:
    """Two-sided p-value, from the t distribution when the degrees of freedom are known."""
    from scipy import stats

    if not np.isfinite(statistic):
        return np.nan
    if dof is not None and dof > 0:
        return float(2 * stats.t.sf(abs(statistic), dof))
    return float(2 * stats.norm.sf(abs(statistic)))


def cumulative_abnormal(panel: AbnormalPanel, start: Optional[int] = None) -> np.ndarray:
    """Cumulative abnormal value per event, accumulated from `start` across event time."""
    begin = panel.event_time[0] if start is None else start
    mask = panel.event_time >= begin
    contribution = np.where(np.isnan(panel.abnormal[:, mask]), 0.0, panel.abnormal[:, mask])
    return np.cumsum(contribution, axis=1)


def cross_sectional_t(car: np.ndarray) -> tuple[float, float]:
    """Plain cross-sectional t-test on the cumulative abnormal values."""
    values = car[np.isfinite(car)]
    n = len(values)
    if n < 2 or np.std(values, ddof=1) == 0:
        return np.nan, np.nan
    t = float(np.mean(values) / (np.std(values, ddof=1) / np.sqrt(n)))
    return t, _two_sided_p(t, dof=n - 1)


def bmp_t(panel: AbnormalPanel, car: np.ndarray, n_days: int,
          cumulative_factor_deviation: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Boehmer, Musumeci and Poulsen (1991) standardised cross-sectional test.

    Each event's cumulative abnormal value is divided by its own forecast-error standard
    error, then a cross-sectional t is taken over the standardised values. Standardising
    first, and taking the cross-sectional spread second, is what makes the test robust to
    the variance an event induces.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        variance_inflation = (n_days
                              + (n_days ** 2) / np.where(panel.n_estimation > 0,
                                                         panel.n_estimation, np.nan)
                              + (cumulative_factor_deviation ** 2)
                              / np.where(panel.factor_ss > 0, panel.factor_ss, np.inf))
        standard_error = panel.resid_std * np.sqrt(variance_inflation)
        scar = np.where(standard_error > 0, car / standard_error, np.nan)

    usable = scar[np.isfinite(scar)]
    n = len(usable)
    if n < 2 or np.std(usable, ddof=1) == 0:
        return np.nan, np.nan, scar
    t = float(np.sqrt(n) * np.mean(usable) / np.std(usable, ddof=1))
    return t, _two_sided_p(t, dof=n - 1), scar


def kolari_pynnonen_t(bmp_statistic: float, n_events: int, mean_correlation: float
                      ) -> tuple[float, float]:
    """Kolari and Pynnonen (2010) correction of the standardised test.

    Scales the standardised statistic by sqrt((1 - r) / (1 + (N - 1) r)), where r is the
    average cross-sectional correlation of the estimation-window residuals. Positive
    correlation inflates the uncorrected statistic, so the corrected one is smaller.
    """
    if not np.isfinite(bmp_statistic) or n_events < 2:
        return np.nan, np.nan
    denominator = 1.0 + (n_events - 1) * mean_correlation
    if denominator <= 0 or (1.0 - mean_correlation) < 0:
        return np.nan, np.nan
    t = float(bmp_statistic * np.sqrt((1.0 - mean_correlation) / denominator))
    return t, _two_sided_p(t, dof=n_events - 1)


def corrado_rank_t(panel: AbnormalPanel, upto_k: int, start: Optional[int] = None
                   ) -> tuple[float, float]:
    """Corrado (1989) rank test, cumulated over the event-time sessions up to `upto_k`.

    Each event's abnormal values are ranked inside its own combined estimation and event
    window, so the test needs no distributional assumption and is not driven by one
    extreme observation.
    """
    begin = panel.event_time[0] if start is None else start
    window = (panel.event_time >= begin) & (panel.event_time <= upto_k)
    if not window.any():
        return np.nan, np.nan

    n_events, n_sessions = panel.abnormal.shape
    centred = np.full((n_events, n_sessions), np.nan)
    for i in range(n_events):
        observations = np.concatenate([panel.estimation_residuals[i].to_numpy(),
                                       panel.abnormal[i]])
        finite = np.isfinite(observations)
        ranks = np.full(len(observations), np.nan)
        ranks[finite] = pd.Series(observations[finite]).rank().to_numpy()
        expected = (finite.sum() + 1) / 2.0
        centred[i] = (ranks[-n_sessions:] - expected)

    # The null standard deviation is estimated across the whole window, per Corrado.
    daily_mean = np.nanmean(centred, axis=0)
    daily_mean = daily_mean[np.isfinite(daily_mean)]
    if len(daily_mean) < 2:
        return np.nan, np.nan
    sigma = float(np.sqrt(np.mean(daily_mean ** 2)))
    if sigma == 0:
        return np.nan, np.nan

    selected = np.nanmean(centred[:, window], axis=0)
    selected = selected[np.isfinite(selected)]
    if len(selected) == 0:
        return np.nan, np.nan
    days = len(selected)
    t = float(np.sum(selected) / (np.sqrt(days) * sigma))
    return t, _two_sided_p(t)


def event_study_table(panel: AbnormalPanel, start: Optional[int] = None,
                      label: str = "") -> pd.DataFrame:
    """Cumulative average abnormal value by event-time session, with every test statistic.

    One row per session in the event window: the CAAR, its cross-sectional confidence
    interval, and the four tests. This is the artifact the thesis quotes.
    """
    begin = panel.event_time[0] if start is None else start
    selected = panel.event_time[panel.event_time >= begin]
    car_matrix = cumulative_abnormal(panel, start=begin)
    cumulative_deviation = np.cumsum(
        np.where(np.isnan(panel.factor_deviation[:, panel.event_time >= begin]), 0.0,
                 panel.factor_deviation[:, panel.event_time >= begin]), axis=1)
    mean_correlation = average_cross_correlation(panel)

    rows = []
    for column, k in enumerate(selected):
        car = car_matrix[:, column]
        observed = car[np.isfinite(car)]
        n = len(observed)
        caar = float(np.mean(observed)) if n else np.nan
        standard_error = (float(np.std(observed, ddof=1) / np.sqrt(n))
                          if n > 1 else np.nan)
        t_cross, p_cross = cross_sectional_t(car)
        n_days = int(k - begin + 1)
        t_bmp, p_bmp, _ = bmp_t(panel, car, n_days, cumulative_deviation[:, column])
        t_kp, p_kp = kolari_pynnonen_t(t_bmp, n, mean_correlation)
        t_rank, p_rank = corrado_rank_t(panel, upto_k=int(k), start=begin)
        rows.append({
            "series": label, "event_time": int(k), "n_events": n,
            "caar": caar,
            "ci_low": caar - 1.96 * standard_error if np.isfinite(standard_error) else np.nan,
            "ci_high": caar + 1.96 * standard_error if np.isfinite(standard_error) else np.nan,
            "t_cross_sectional": t_cross, "p_cross_sectional": p_cross,
            "t_bmp": t_bmp, "p_bmp": p_bmp,
            "t_corrado": t_rank, "p_corrado": p_rank,
            "t_kolari_pynnonen": t_kp, "p_kolari_pynnonen": p_kp,
            "mean_cross_correlation": mean_correlation,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n_sessions, n_events = 900, 40
    sessions = pd.date_range("2015-01-01", periods=n_sessions, freq="B")
    factor = rng.normal(0.0, 0.01, n_sessions)
    asset = 1.2 * factor + rng.normal(0.0, 0.004, n_sessions)
    positions = np.sort(rng.choice(np.arange(400, n_sessions - 60), n_events, replace=False))

    # A null series: the tests must not fire.
    null_daily = pd.DataFrame({"date": sessions, "log_return": asset,
                               "sp500_log_return": factor})
    null_panel = build_abnormal_panel(null_daily, sessions[positions])
    null_table = event_study_table(null_panel, start=1, label="null")
    peak = null_table.loc[null_table.event_time == 5].iloc[0]
    assert abs(peak.t_bmp) < 2.5, peak.t_bmp
    assert abs(peak.t_cross_sectional) < 2.5, peak.t_cross_sectional
    assert abs(peak.t_corrado) < 2.5, peak.t_corrado

    # A planted effect: a clear drop on the five sessions after each event.
    shocked = asset.copy()
    for p in positions:
        shocked[p:p + 5] -= 0.006
    effect_daily = pd.DataFrame({"date": sessions, "log_return": shocked,
                                 "sp500_log_return": factor})
    panel = build_abnormal_panel(effect_daily, sessions[positions])
    table = event_study_table(panel, start=1, label="planted")
    at5 = table.loc[table.event_time == 5].iloc[0]
    assert at5.n_events == n_events, at5.n_events
    # Five sessions at -0.6 percent each, so the CAAR is about -3 percentage points.
    assert abs(at5.caar - (-0.03)) < 5e-3, at5.caar
    assert at5.ci_high < 0, at5.ci_high
    assert at5.t_bmp < -3, at5.t_bmp
    assert at5.t_cross_sectional < -3, at5.t_cross_sectional
    assert at5.t_corrado < -3, at5.t_corrado
    assert at5.p_bmp < 0.01 and at5.p_corrado < 0.01

    # Kolari and Pynnonen can only shrink a statistic when events are positively
    # correlated, and must never make one more significant than the uncorrected test.
    assert abs(at5.t_kolari_pynnonen) <= abs(at5.t_bmp) + 1e-9, at5

    # The mean-adjusted model carries the same machinery for a series with no factor.
    volume = pd.DataFrame({"date": sessions,
                           "log_volume": rng.normal(10.0, 0.3, n_sessions)})
    volume.loc[volume.index.isin(np.concatenate([np.arange(p, p + 5) for p in positions])),
               "log_volume"] += 0.8
    vol_panel = build_abnormal_panel(volume, sessions[positions], model="mean",
                                     value_col="log_volume")
    vol_table = event_study_table(vol_panel, start=1, label="volume")
    vol5 = vol_table.loc[vol_table.event_time == 5].iloc[0]
    # Five sessions at +0.8 log points would be +4.0 against a clean baseline. These 40
    # synthetic events sit close enough together that each estimation window already
    # contains other events' elevated sessions, which lifts the baseline and shrinks the
    # measured response. The effect must still be detected clearly.
    assert 2.5 < vol5.caar < 4.0, vol5.caar
    assert vol5.t_bmp > 3, vol5.t_bmp

    print("event_study.py self-check passed (null does not fire, planted effect detected)")
