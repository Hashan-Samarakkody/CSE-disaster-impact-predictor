"""Feature engineering for disaster-aware multi-target market prediction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


EXCLUDED_DISASTER_TYPES = {"epidemic", "biological", "biological disaster", "pandemic"}

# Minimum trailing history a calendar year needs before GARCH is allowed to speak for it
# at all. See `_garch_conditional_volatility` for why annual (not per-row) refitting is
# the causal-safe choice here.
GARCH_MIN_HISTORY = 250


def _garch_conditional_volatility(dates: pd.Series, log_returns: pd.Series) -> pd.Series:
    """Causal GARCH(1,1) conditional-volatility forecast, aggregate ASPI level.

    Rationale: `rolling_std_*` (above) is a *realized*, backward-looking dispersion
    measure. A GARCH(1,1) conditional volatility is a one-step-ahead *forecast* that
    weights recent shocks more than old ones and mean-reverts to a long-run level -- a
    genuinely different, complementary signal, not a relabelling of the rolling-std
    columns (see docs/METHODOLOGY_AUDIT.md, "Improvement attempts", for the two papers
    behind this addition). It is added as a *candidate* feature: the existing per-fold
    `select_top_features` RF-importance step decides whether it earns a place, same as
    every other column.

    Causality: refitting a GARCH model on every row's own trailing window (as an
    analyst well past this event date would) is what a live forecaster would do, but
    literally refitting per-row is not what happened here -- for speed, the fit is
    refreshed once per calendar year, using ONLY returns strictly before that year, and
    that single year's fixed (omega, alpha, beta) parameters are then used to filter the
    conditional-variance recursion forward through the year using only its own (already
    causal, same-direction) past returns. No row's feature value at date `t` is ever
    computed from a fit that included any return on or after `t`. The first
    `GARCH_MIN_HISTORY` trading days have no prior year with enough history and are left
    NaN -- consistent with how the rolling-window features above are NaN at the start of
    the sample.
    """
    from arch import arch_model
    from arch.utility.exceptions import ConvergenceWarning

    returns_pct = (log_returns * 100.0).to_numpy()  # arch's default scale is % returns
    years = pd.to_datetime(dates).dt.year.to_numpy()
    out = np.full(len(returns_pct), np.nan)

    for year in sorted(set(years)):
        train_mask = years < year
        if train_mask.sum() < GARCH_MIN_HISTORY:
            continue
        train_returns = returns_pct[train_mask]
        train_returns = train_returns[~np.isnan(train_returns)]
        if len(train_returns) < GARCH_MIN_HISTORY:
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                fitted = arch_model(train_returns, vol="Garch", p=1, q=1, rescale=False).fit(
                    disp="off", show_warning=False)
        except (ValueError, np.linalg.LinAlgError):
            continue  # a year whose training history fails to converge is left NaN, not guessed

        this_year_mask = years == year
        this_year_returns = np.nan_to_num(returns_pct[this_year_mask], nan=0.0)

        # `.fix(params)` re-runs the SAME (already-fitted, strictly-prior-year) GARCH
        # recursion through a longer series without re-estimating anything -- it is a
        # pure filter pass, not a refit, so appending this year's returns and reading off
        # the tail is causal: the recursion at position i uses only returns[0..i-1].
        combined = np.concatenate([train_returns, this_year_returns])
        try:
            filtered = arch_model(combined, vol="Garch", p=1, q=1,
                                   rescale=False).fix(fitted.params)
        except (ValueError, np.linalg.LinAlgError):
            continue
        cond_vol = filtered.conditional_volatility[-this_year_mask.sum():]
        out[this_year_mask] = cond_vol / 100.0  # back to log-return scale

    return pd.Series(out, index=log_returns.index)


@dataclass
class FeatureEngineeringConfig:
    date_col: str = "date"
    price_col: str = "aspi_close"
    volume_col: str = "trading_volume"
    disaster_date_col: str = "event_date"
    disaster_type_col: str = "disaster_type"
    damage_col: str = "financial_damage"
    affected_col: str = "population_affected"
    # Inclusion threshold on total affected, FROZEN at the thesis's original
    # pre-registration (2026-09-16 methodology-audit review). Briefly lowered to 700
    # (2026-09-12) then 500 (2026-09-15) while exploring whether more events were
    # available; verified both admit zero additional matched events over 1000 (raw
    # EM-DAT gains 2 records between 500 and 1000, neither of which lands on a
    # tradable CSE session), so reverting costs nothing and removes the drift.
    # Do not change this value based on downstream model performance -- see
    # docs/METHODOLOGY_AUDIT.md "Inclusion threshold: frozen" for the sensitivity
    # analysis this decision is based on.
    min_affected: int = 1000
    max_recovery_days: int = 90


class FeatureEngineer:
    """Builds exogenous, endogenous and target variables under time-safe constraints."""

    def __init__(self, config: Optional[FeatureEngineeringConfig] = None) -> None:
        self.config = config or FeatureEngineeringConfig()

    def engineer_market_features(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged returns, moving averages, and panic features from market time series."""
        c = self.config
        df = market_df.copy().sort_values(c.date_col)
        df[c.date_col] = pd.to_datetime(df[c.date_col])

        df["log_return"] = np.log(df[c.price_col] / df[c.price_col].shift(1))

        for lag in (1, 2, 3, 5):
            df[f"lag_return_t-{lag}"] = df["log_return"].shift(lag)

        # Shift the price series before rolling, so sma_w/ema_w at row t reflect prices only
        # through t-1. Rolling directly on the unshifted series would let the window's last
        # point be the shock day itself -- a look-ahead bug found during review.
        shifted_price = df[c.price_col].shift(1)
        for window in (5, 10, 20):
            df[f"sma_{window}"] = shifted_price.rolling(window=window, min_periods=window).mean()
            df[f"ema_{window}"] = shifted_price.ewm(span=window, adjust=False, min_periods=window).mean()
            # Stationary counterparts. The raw sma_/ema_ LEVELS run 574 -> 10,000+ over the sample,
            # so a later chronological fold sits outside any training range. The price-relative
            # ratio carries the same momentum information with no trend in the level.
            df[f"price_to_sma_{window}"] = shifted_price / df[f"sma_{window}"] - 1.0
            df[f"price_to_ema_{window}"] = shifted_price / df[f"ema_{window}"] - 1.0

        # Shift by one day to enforce pre-shock boundary (up to t-1) and avoid look-ahead.
        shifted_returns = df["log_return"].shift(1)
        for window in (5, 10, 20):
            df[f"rolling_std_{window}"] = shifted_returns.rolling(window=window, min_periods=window).std()

        # Dedicated 30-day panic-proxy volatility, distinct from the momentum windows above:
        # this is the window that mirrors the 30-day volume baseline used for Y2.
        df["rolling_std_30"] = shifted_returns.rolling(window=30, min_periods=30).std()

        df["squared_return"] = df["log_return"] ** 2

        # One-step-ahead GARCH(1,1) conditional volatility forecast, causal by
        # construction (see `_garch_conditional_volatility`) -- a genuinely different
        # signal from the realized `rolling_std_*` above, added as a candidate feature
        # for the existing per-fold RF-importance selection to keep or drop.
        try:
            df["garch_cond_vol"] = _garch_conditional_volatility(df[c.date_col], df["log_return"])
        except ImportError:
            # `arch` is an added, optional dependency (requirements.txt); a pipeline run
            # without it degrades to not having this one candidate feature, not to a
            # crash.
            pass

        # Pre-event volume block. Y2 is DEFINED as V_t / mean(V_{t-30..t-1}) - 1, so the same
        # functional one trading day earlier is its natural autoregressive predictor. Every
        # term is a ratio or log difference, so no volume LEVEL enters the model.
        if c.volume_col in df.columns:
            shifted_vol = df[c.volume_col].shift(1)
            vol_mean_30 = shifted_vol.rolling(window=30, min_periods=30).mean()
            for window in (1, 5, 10):
                numer = (shifted_vol if window == 1
                         else shifted_vol.rolling(window=window, min_periods=window).mean())
                df[f"vol_ratio_{window}_30"] = numer / vol_mean_30 - 1.0
            # Dispersion of recent volume: a market already trading erratically responds
            # differently from a quiet one, and this is scale-free.
            vol_std_30 = shifted_vol.rolling(window=30, min_periods=30).std()
            df["vol_cv_30"] = vol_std_30 / vol_mean_30
            df["log_vol_change_1"] = np.log(shifted_vol / shifted_vol.shift(1))
            # No vol_trend_10_30: a 10-day mean over the 30-day mean is exactly vol_ratio_10_30.
            # The first version shipped both, and the collinearity check caught them at rho 1.000.

            # Volume is zero on some illiquid sessions and missing for all of 2000, which makes
            # the ratios inf. Leave them NaN so the missingness handling sees them.
            vol_cols = [col for col in df.columns
                        if col.startswith(("vol_ratio_", "vol_cv_", "log_vol_change_"))]
            df[vol_cols] = df[vol_cols].replace([np.inf, -np.inf], np.nan)

        return df

    def engineer_disaster_features(self, disaster_df: pd.DataFrame) -> pd.DataFrame:
        """Create transformed exogenous disaster covariates and one-hot disaster types."""
        c = self.config
        df = disaster_df.copy()
        df[c.disaster_date_col] = pd.to_datetime(df[c.disaster_date_col])

        clean_type = df[c.disaster_type_col].astype(str).str.strip().str.lower()
        keep_mask = ~clean_type.isin(EXCLUDED_DISASTER_TYPES)
        df = df[keep_mask].copy()

        # Read from the config so the value is declared in one place. Widening a
        # pre-registered filter is a deviation and is reported as one -- see
        # FeatureEngineeringConfig.min_affected for the history of changes and why.
        df = df[df[c.affected_col] >= c.min_affected].copy()
        df["log_financial_damage"] = np.log1p(df[c.damage_col].clip(lower=0))
        df["log_population_affected"] = np.log1p(df[c.affected_col].clip(lower=0))

        # Pool types with too few events for their own indicator: a one-hot with a single
        # positive case is a memorisation key, and pooling also gives those events a
        # same-type SMOGN partner. Threshold fixed a priori at 3, not tuned on any result.
        type_counts = df[c.disaster_type_col].value_counts()
        rare_types = type_counts[type_counts < 3].index
        if len(rare_types) > 0:
            df[c.disaster_type_col] = df[c.disaster_type_col].where(
                ~df[c.disaster_type_col].isin(rare_types), "Other"
            )

        one_hot = pd.get_dummies(df[c.disaster_type_col], prefix="disaster", dtype=float)
        return pd.concat([df, one_hot], axis=1)

    def build_targets(self, market_df: pd.DataFrame, disaster_df: pd.DataFrame) -> pd.DataFrame:
        """Build Y1 (% dev. of post-week mean vs pre-30d mean), Y2 (abnormal volume), and
        Y3 (recovery days) per event."""
        c = self.config
        market = market_df.copy().sort_values(c.date_col)
        market[c.date_col] = pd.to_datetime(market[c.date_col])

        events = disaster_df.copy().sort_values(c.disaster_date_col)
        events[c.disaster_date_col] = pd.to_datetime(events[c.disaster_date_col])

        rows = []
        for _, event in events.iterrows():
            event_date = event[c.disaster_date_col]
            event_idx = market.index[market[c.date_col] >= event_date]
            if len(event_idx) == 0:
                continue

            idx = event_idx[0]
            pos = market.index.get_loc(idx)
            if pos == 0:
                continue

            price_t = market.iloc[pos][c.price_col]
            price_tm1 = market.iloc[pos - 1][c.price_col]

            # Y1 = ASPI_5D_Log_Return_Pct -- five-trading-day forward cumulative log
            # return, in percent: 100 * ln(ASPI_(t+5) / ASPI_t). `pos` is the reference
            # trading day (first trading session on/after the event date -- the existing
            # alignment rule, reused as-is). `market` is already indexed one row per
            # TRADING session (non-trading days never appear in it), so pos+5 is exactly
            # the fifth subsequent CSE trading session, not five calendar days.
            # NaN (and the row excluded downstream) if fewer than 5 trading sessions
            # remain after the event -- never fabricate a truncated horizon.
            y1_horizon_pos = pos + 5
            if y1_horizon_pos < len(market):
                price_t5 = market.iloc[y1_horizon_pos][c.price_col]
                y1 = float(100.0 * np.log(price_t5 / price_t))
                y1_horizon_end_date = market.iloc[y1_horizon_pos][c.date_col]
            else:
                y1 = np.nan
                y1_horizon_end_date = pd.NaT

            # Y2 needs a volume series. A sector index has none -- the CSE publishes
            # volume market-wide, not per sector -- so the sector panel calls this with
            # no volume column and gets NaN rather than a fabricated ratio.
            if c.volume_col in market.columns:
                baseline_start = max(0, pos - 30)
                baseline_mean = market.iloc[baseline_start:pos][c.volume_col].mean()
                volume_t = market.iloc[pos][c.volume_col]
                y2 = (float((volume_t / baseline_mean) - 1.0)
                      if baseline_mean and not np.isnan(baseline_mean) else np.nan)
            else:
                y2 = np.nan

            pre_disaster_baseline = price_tm1
            recovery_window = market.iloc[pos : pos + c.max_recovery_days + 1]
            recovered = recovery_window[recovery_window[c.price_col] >= pre_disaster_baseline]
            if recovered.empty:
                y3 = c.max_recovery_days
            else:
                # TRADING days, per thesis Sec. 3.2.2. The previous version returned calendar days
                # while searching a window of 91 trading ROWS, so the 90 cap actually bit at about
                # 62 trading days and slow recoveries were indistinguishable from none at all.
                recovery_pos = market.index.get_loc(recovered.index[0])
                y3 = min(recovery_pos - pos, c.max_recovery_days)

            # Cumulative event-window returns, pre-declared in
            # docs/EXTERNAL_DATA_PRE_DECLARATION.md Sec. 7.2 before anything scored them. NaN rather
            # than a truncated window, so a partial accumulation is never reported as a full one.
            # Named EventWindow (not CAR) and expressed in percent (matching Y1's _Pct units)
            # per the 2026-09-16 methodology-audit freeze: this is a raw cumulative log return
            # from the pre-event close, not an abnormal return against an expected-return model,
            # so calling it "CAR" was inaccurate.
            cars = {}
            for k in (5, 10):
                cars[f"Y1_EventWindow_0_{k}_LogReturn_Pct"] = (
                    float(100.0 * np.log(market.iloc[pos + k][c.price_col] / price_tm1))
                    if pos + k < len(market) else np.nan)

            rows.append({
                c.disaster_date_col: event_date,
                "Y1_ASPI_5D_Forward_LogReturn_Pct": y1,
                # Date of the trading session Y1's numerator is read from -- used only to
                # purge train/test fold-boundary overlap in `walk_forward.purge_horizon_overlap`,
                # never as a model feature.
                "Y1_horizon_end_date": y1_horizon_end_date,
                "Y2_abnormal_volume": y2,
                "Y3_recovery_days": float(y3),
                **cars,
            })

        return pd.DataFrame(rows)

    def build_feature_table(
        self,
        market_df: pd.DataFrame,
        disaster_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        global_df: Optional[pd.DataFrame] = None,
        control_merge_cols: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Combine engineered endogenous/exogenous features and optional control variables."""
        c = self.config
        market_feats = self.engineer_market_features(market_df)
        disaster_feats = self.engineer_disaster_features(disaster_df)

        merged = pd.merge_asof(
            market_feats.sort_values(c.date_col),
            disaster_feats.sort_values(c.disaster_date_col),
            left_on=c.date_col,
            right_on=c.disaster_date_col,
            direction="backward",
        )

        if macro_df is not None:
            macro = macro_df.copy()
            macro[c.date_col] = pd.to_datetime(macro[c.date_col])
            merged = pd.merge_asof(merged.sort_values(c.date_col), macro.sort_values(c.date_col), on=c.date_col, direction="backward")

        if global_df is not None:
            global_market = global_df.copy()
            global_market[c.date_col] = pd.to_datetime(global_market[c.date_col])
            merged = pd.merge_asof(
                merged.sort_values(c.date_col),
                global_market.sort_values(c.date_col),
                on=c.date_col,
                direction="backward",
                suffixes=("", "_global"),
            )

        if control_merge_cols:
            keep_cols = list(dict.fromkeys([c.date_col, *control_merge_cols]))
            keep_cols = [col for col in keep_cols if col in merged.columns]
            merged = merged[keep_cols]

        return merged
