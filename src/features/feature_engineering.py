"""Feature engineering for disaster-aware multi-target market prediction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.config.settings import ASPI_PERCENTAGE_CHANGE, MARKET_RECOVERY_DAYS
from src.targets.event_targets import build_event_targets


EXCLUDED_DISASTER_TYPES = {"epidemic", "biological", "biological disaster", "pandemic"}

# Minimum trailing history a calendar year needs before GARCH is allowed to speak for it
# at all. See `_garch_conditional_volatility` for why annual (not per-row) refitting is
# the causal-safe choice here.
GARCH_MIN_HISTORY = 250


def _garch_conditional_volatility(dates: pd.Series, log_returns: pd.Series) -> pd.Series:
    """Causal GARCH(1,1) conditional-volatility forecast, aggregate ASPI level."""
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
        combined = np.concatenate([train_returns, this_year_returns])
        try:
            filtered = arch_model(combined, vol="Garch", p=1, q=1,
                                   rescale=False).fix(fitted.params)
        except (ValueError, np.linalg.LinAlgError):
            continue
        cond_vol = filtered.conditional_volatility[-this_year_mask.sum():]
        out[this_year_mask] = cond_vol / 100.0  # back to log-return scale

    return pd.Series(out, index=log_returns.index)


def build_sample_flow(raw_disasters: pd.DataFrame, market_df: pd.DataFrame,
                      config: "FeatureEngineeringConfig" = None) -> pd.DataFrame:
    """One row per exclusion criterion, from the raw EM-DAT export to the modelled events.

    Applies the same predicates and the same constants the pipeline uses, in the pipeline's
    order, so the accounting cannot drift from what actually ran. Columns: stage, criterion,
    n_removed, n_remaining.
    """
    c = config or FeatureEngineeringConfig()
    rows, frame = [], raw_disasters.copy()
    frame[c.disaster_date_col] = pd.to_datetime(frame[c.disaster_date_col])

    def record(stage, criterion, kept):
        nonlocal frame
        rows.append({"stage": stage, "criterion": criterion,
                     "n_removed": int(len(frame) - len(kept)),
                     "n_remaining": int(len(kept))})
        frame = kept

    rows.append({"stage": "raw EM-DAT records", "criterion": "none, as exported",
                 "n_removed": 0, "n_remaining": int(len(frame))})

    clean_type = frame[c.disaster_type_col].astype(str).str.strip().str.lower()
    record("disaster type excluded",
           f"disaster_type in {sorted(EXCLUDED_DISASTER_TYPES)}",
           frame[~clean_type.isin(EXCLUDED_DISASTER_TYPES)])

    record("below the affected threshold",
           f"population_affected < {c.min_affected}",
           frame[frame[c.affected_col] >= c.min_affected])

    market = market_df.copy().sort_values(c.date_col).reset_index(drop=True)
    market[c.date_col] = pd.to_datetime(market[c.date_col])
    session_dates = market[c.date_col].to_numpy()
    has_session, has_baseline = [], []
    for event_date in frame[c.disaster_date_col]:
        matching = np.flatnonzero(session_dates >= np.datetime64(event_date))
        has_session.append(len(matching) > 0)
        has_baseline.append(len(matching) > 0 and int(matching[0]) > 0)

    record("prediction origin unalignable",
           "no trading session on or after the event date",
           frame[pd.Series(has_session, index=frame.index)])
    record("no pre-event close",
           "the aligned session is the first in the market series, so P0 does not exist",
           frame[pd.Series(has_baseline, index=frame.index).reindex(frame.index).fillna(False)])

    targets = build_event_targets(
        market, frame, date_col=c.date_col, price_col=c.price_col,
        volume_col=c.volume_col, disaster_date_col=c.disaster_date_col,
        max_recovery_days=c.max_recovery_days)
    complete = targets[[ASPI_PERCENTAGE_CHANGE, MARKET_RECOVERY_DAYS]].notna().any(axis=1)
    rows.append({"stage": "incomplete target vector",
                 "criterion": "neither the return nor the recovery target could be built",
                 "n_removed": int((~complete).sum()),
                 "n_remaining": int(complete.sum())})
    return pd.DataFrame(rows)


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
        # point be the shock day itself, a look-ahead bug found during review.
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
        # this is the window that mirrors the 30-session pre-event volume baseline in Y2.
        df["rolling_std_30"] = shifted_returns.rolling(window=30, min_periods=30).std()

        df["squared_return"] = df["log_return"] ** 2

        # One-step-ahead GARCH(1,1) conditional volatility forecast, causal by
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
        # pre-registered filter is a deviation and is reported as one, see
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
        """Build the three research targets for each qualifying event.

        Delegates to src/targets/event_targets.py so that outcome construction
        lives next to the other target code rather than inside feature engineering.
        """
        c = self.config
        return build_event_targets(
            market_df, disaster_df, date_col=c.date_col, price_col=c.price_col,
            volume_col=c.volume_col, disaster_date_col=c.disaster_date_col,
            max_recovery_days=c.max_recovery_days)


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
