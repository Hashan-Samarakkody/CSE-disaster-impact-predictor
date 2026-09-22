"""Market-adjusted abnormal returns for the ASPI event windows (Revision 2, T6).

A raw index move can register as a disaster effect when the cause was global: the sample
spans March 2020 and the 2022 Sri Lankan economic crisis. The abnormal return removes the
part of the move the world market already explains.

Single-factor market model, the standard event-study specification (MacKinlay 1997):

    r_aspi,t = alpha + beta * r_market,t + e_t

estimated by ordinary least squares on daily sessions ending strictly before the
prediction origin. The expected forward return over h sessions is the sum of the fitted
values across the h post-event sessions, and the abnormal return is what the index
actually did minus that. The realised market factor inside the event window is an
observation, not a forecast, so it enters the expected return; only the coefficients are
restricted to pre-origin data.

The raw forward return stays the primary target. These columns are reported alongside it.

Two measured facts a reader needs, both diagnostics rather than choices:

1. The ASPI is close to uncorrelated with the S&P 500 at daily frequency on the CSE
   calendar, correlation 0.014 over 6,264 sessions, full-sample beta 0.013. Market
   adjustment therefore moves this target very little, which is itself the finding.
2. The one-session-lagged S&P correlates far better, 0.106, because the US market closes
   after the CSE and its move reaches Colombo the following session. This module
   implements the **contemporaneous** specification, which is the textbook market model
   and the literal reading of the revision specification. A lagged factor is the standard
   correction for non-overlapping trading hours and would be a different specification,
   so it is recorded here as an open question rather than substituted silently.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

# Daily sessions of estimation history, roughly one trading year, the conventional
# event-study window length. Fixed a priori, never tuned on a result.
ESTIMATION_WINDOW = 250
# Below this the ordinary least squares fit is not worth trusting, so the event's
# abnormal return stays missing rather than being reported from a thin fit.
MIN_ESTIMATION_ROWS = 60

MARKET_FACTOR_COL = "sp500_log_return"


class MarketModel(NamedTuple):
    """Fitted single-factor market model and the window it came from."""

    alpha: float
    beta: float
    resid_std: float
    n: int
    first_row: int
    last_row: int


def abnormal_return_col(horizon: int) -> str:
    """Dataset column holding the market-adjusted abnormal return over `horizon`."""
    return f"Y1_ASPI_{horizon}D_Forward_AbnormalReturn_Pct"


def abnormal_return_end_col(horizon: int) -> str:
    """Dataset column holding the session on which the abnormal return closes."""
    return f"Y1_{horizon}D_abnormal_horizon_end_date"


def align_market_factor(session_dates, factor_dates, factor_returns) -> np.ndarray:
    """Market factor re-expressed on the local trading calendar.

    The S&P 500 and the CSE keep different calendars, so 3.7 percent of CSE sessions have
    no same-day US observation. Joining on the date alone would leave a hole that voids
    the whole event window. Instead the factor's log returns are accumulated into an index,
    carried forward onto the local sessions, and differenced, so each local session carries
    the market return realised over the interval since the previous local session.

    Contemporaneous by construction, which is standard for an event-study market model and
    appropriate here because the abnormal return is an ex post attribution, not a forecast.
    The first local session has no preceding interval and is left missing.
    """
    factor = (pd.DataFrame({"date": pd.to_datetime(pd.Series(list(factor_dates))),
                            "r": np.asarray(factor_returns, dtype=float)})
              .dropna().sort_values("date"))
    if factor.empty:
        return np.full(len(list(session_dates)), np.nan)
    cumulative = pd.Series(factor["r"].cumsum().to_numpy(), index=factor["date"].to_numpy())
    local = pd.to_datetime(pd.Series(list(session_dates))).sort_values()
    carried = cumulative.reindex(cumulative.index.union(local.to_numpy())).ffill()
    return carried.reindex(local.to_numpy()).diff().to_numpy()


def estimate_market_model(asset_returns, market_returns, origin_row: int,
                          window: int = ESTIMATION_WINDOW,
                          min_rows: int = MIN_ESTIMATION_ROWS) -> Optional[MarketModel]:
    """Fit the market model on the `window` sessions ending strictly before `origin_row`.

    `origin_row` is the first post-event session, so the window closes at `origin_row - 1`,
    the last close observed before the prediction origin. Returns None when too few usable
    rows remain.
    """
    asset = np.asarray(asset_returns, dtype=float)
    market = np.asarray(market_returns, dtype=float)
    last_row = origin_row - 1
    first_row = max(0, last_row - window + 1)
    if last_row < first_row:
        return None

    rows = np.arange(first_row, last_row + 1)
    usable = rows[np.isfinite(asset[rows]) & np.isfinite(market[rows])]
    if len(usable) < min_rows:
        return None

    x, y = market[usable], asset[usable]
    if not np.isfinite(np.var(x)) or np.var(x) == 0:
        return None
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (alpha + beta * x)
    ddof = max(len(usable) - 2, 1)
    return MarketModel(alpha=float(alpha), beta=float(beta),
                       resid_std=float(np.sqrt(np.sum(resid ** 2) / ddof)),
                       n=int(len(usable)), first_row=int(usable[0]), last_row=int(usable[-1]))


def expected_forward_return(model: MarketModel, market_returns, origin_row: int,
                            horizon: int) -> float:
    """Fitted cumulative return over the `horizon` sessions from `origin_row`, in percent.

    NaN when the market factor is missing anywhere in the window, so an expected return is
    never formed from a partly observed window.
    """
    market = np.asarray(market_returns, dtype=float)
    end_row = origin_row + horizon - 1
    if end_row >= len(market):
        return np.nan
    window = market[origin_row:end_row + 1]
    if not np.isfinite(window).all():
        return np.nan
    return float(100.0 * np.sum(model.alpha + model.beta * window))


def build_abnormal_return_targets(daily: pd.DataFrame, event_dates, horizons,
                                  date_col: str = "date",
                                  asset_return_col: str = "log_return",
                                  market_factor_col: str = MARKET_FACTOR_COL,
                                  price_col: str = "aspi_close") -> pd.DataFrame:
    """One row per event date, one column pair per horizon, in the order given.

    `daily` must carry the date, the ASPI close, the ASPI daily log return and the market
    factor's daily log return. Rows whose market model cannot be fitted, or whose window
    runs off the end of the series, carry missing values rather than a partial estimate.
    """
    frame = daily.sort_values(date_col).reset_index(drop=True)
    dates = pd.to_datetime(frame[date_col])
    date_values = dates.to_numpy()
    prices = frame[price_col].to_numpy(dtype=float)
    asset = frame[asset_return_col].to_numpy(dtype=float)
    market = frame[market_factor_col].to_numpy(dtype=float)

    rows = []
    for event_date in pd.to_datetime(pd.Series(list(event_dates))):
        row: dict[str, object] = {}
        candidates = np.flatnonzero(date_values >= np.datetime64(event_date))
        position = int(candidates[0]) if len(candidates) and candidates[0] > 0 else None
        model = None if position is None else estimate_market_model(asset, market, position)
        for h in horizons:
            value, end = np.nan, pd.NaT
            if model is not None:
                end_row = position + h - 1
                p0 = prices[position - 1]
                if end_row < len(prices) and np.isfinite(prices[end_row]) and p0 > 0:
                    realised = 100.0 * np.log(prices[end_row] / p0)
                    expected = expected_forward_return(model, market, position, h)
                    if np.isfinite(expected):
                        value, end = realised - expected, dates.iloc[end_row]
            row[abnormal_return_col(h)] = value
            row[abnormal_return_end_col(h)] = end
        row["market_model_alpha"] = np.nan if model is None else model.alpha
        row["market_model_beta"] = np.nan if model is None else model.beta
        row["market_model_n"] = 0 if model is None else model.n
        rows.append(row)
    return pd.DataFrame(rows)


def stage_a_expected_returns(market_feats, event_positions, horizons):
    """Normal-market expected h-session return for every event, estimated ONLY from
    daily rows whose own label was fully settled strictly before that event's reference
    session (protocol 1.2 Stage A).

    Relocated unchanged from scripts/run_aspi_return_grid.py (Revision 2, T6) so that
    every expected-return construction lives beside the abnormal-return target.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    mf = market_feats.sort_values("date").reset_index(drop=True)
    price = mf["aspi_close"].to_numpy(float)
    feat_cols = [c for c in mf.columns
                 if c not in {"date", "aspi_close", "trading_volume", "price_source",
                              "volume_source"}
                 and not c.startswith(("sma_", "ema_"))]  # raw price levels: non-stationary
    F = mf[feat_cols].shift(1)                            # features known at p-1
    out = {}
    for h in horizons:
        # Same alignment as the target: Ph = price[p + h - 1], P0 = price[p - 1].
        fwd = np.full(len(mf), np.nan)
        valid = np.arange(1, len(mf) - h + 1)
        fwd[valid] = 100.0 * np.log(price[valid + h - 1] / price[valid - 1])
        preds = []
        for pos in event_positions:
            if pos is None:
                preds.append(np.nan)
                continue
            # A training row at p settles at session p + h - 1; require that before pos.
            usable = np.arange(1, max(1, pos - h + 1))
            rows = usable[np.isfinite(fwd[usable]) & F.iloc[usable].notna().all(axis=1).to_numpy()]
            if len(rows) < 250:
                preds.append(np.nan)
                continue
            model = Pipeline([("scale", StandardScaler()),
                              ("model", Ridge(alpha=10.0))]).fit(F.iloc[rows], fwd[rows])
            x = F.iloc[[pos]]
            preds.append(float(model.predict(x.fillna(F.iloc[rows].median()))[0]))
        out[h] = np.asarray(preds, dtype=float)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 600
    sessions = pd.date_range("2018-01-01", periods=n, freq="B")
    market_ret = rng.normal(0.0, 0.01, n)

    # A synthetic index that follows the market exactly, beta 1.5 and alpha 0, with a
    # known shock planted on the five sessions after one chosen event.
    true_beta = 1.5
    asset_ret = true_beta * market_ret
    event_position = 400
    shock_per_session = -0.004
    asset_ret[event_position:event_position + 5] += shock_per_session

    prices = 100.0 * np.exp(np.cumsum(asset_ret))
    daily = pd.DataFrame({"date": sessions, "aspi_close": prices,
                          "log_return": asset_ret, "sp500_log_return": market_ret})

    model = estimate_market_model(asset_ret, market_ret, event_position)
    assert model is not None
    assert abs(model.beta - true_beta) < 1e-6, model
    assert abs(model.alpha) < 1e-6, model
    # The estimation window must close on the last session before the origin.
    assert model.last_row == event_position - 1, model
    assert model.n == ESTIMATION_WINDOW, model

    out = build_abnormal_return_targets(daily, [sessions[event_position]], (1, 5, 10))
    # Five sessions of the planted shock, in percent, and nothing else.
    assert abs(out[abnormal_return_col(5)].iloc[0] - 100 * 5 * shock_per_session) < 1e-6
    assert abs(out[abnormal_return_col(1)].iloc[0] - 100 * shock_per_session) < 1e-6
    # By session 10 the shock is over, so the cumulative abnormal return stops growing.
    assert abs(out[abnormal_return_col(10)].iloc[0]
               - out[abnormal_return_col(5)].iloc[0]) < 1e-6
    assert out[abnormal_return_end_col(5)].iloc[0] == sessions[event_position + 4]

    # A quiet event with no shock must produce an abnormal return of zero.
    quiet = build_abnormal_return_targets(daily, [sessions[300]], (5,))
    assert abs(quiet[abnormal_return_col(5)].iloc[0]) < 1e-6, quiet

    # Too little history to fit -> missing, never a thin fit.
    assert estimate_market_model(asset_ret, market_ret, 10) is None
    early = build_abnormal_return_targets(daily, [sessions[10]], (5,))
    assert np.isnan(early[abnormal_return_col(5)].iloc[0])

    # A window running off the end of the series -> missing, never truncated.
    late = build_abnormal_return_targets(daily, [sessions[-2]], (5,))
    assert np.isnan(late[abnormal_return_col(5)].iloc[0])

    # A missing market factor inside the event window -> missing expected return.
    gapped = daily.copy()
    gapped.loc[event_position + 2, "sp500_log_return"] = np.nan
    holed = build_abnormal_return_targets(gapped, [sessions[event_position]], (5,))
    assert np.isnan(holed[abnormal_return_col(5)].iloc[0])

    # Calendar alignment: a local session with no same-day factor observation must carry
    # the factor return accumulated over the interval, not a hole.
    local = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    factor_days = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    factor_r = np.array([0.01, 0.02, 0.03, 0.04])
    aligned = align_market_factor(local, factor_days, factor_r)
    assert np.isnan(aligned[0])                       # no preceding interval
    assert abs(aligned[1] - 0.02) < 1e-12
    # Drop the 2020-01-06 factor session: the 01-07 local session must pick up both days.
    gapped = align_market_factor(local, factor_days[[0, 1, 3]], factor_r[[0, 1, 3]])
    assert abs(gapped[3] - 0.04) < 1e-12
    assert np.isnan(gapped[2]) or abs(gapped[2]) < 1e-12

    print("abnormal_returns.py self-check passed")
