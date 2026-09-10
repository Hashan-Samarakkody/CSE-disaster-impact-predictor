"""Feature engineering for disaster-aware multi-target market prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


EXCLUDED_DISASTER_TYPES = {"epidemic", "biological", "biological disaster", "pandemic"}


@dataclass
class FeatureEngineeringConfig:
    date_col: str = "date"
    price_col: str = "aspi_close"
    volume_col: str = "trading_volume"
    disaster_date_col: str = "event_date"
    disaster_type_col: str = "disaster_type"
    damage_col: str = "financial_damage"
    affected_col: str = "population_affected"
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

        # Shift the price series itself by one day before rolling, so sma_w/ema_w
        # at row t only ever reflect prices through t-1 (thesis Sec. 3.5.1 "strict
        # pre-shock boundary"). Rolling directly on df[price_col] without this
        # shift would let the window's *last* point be the shock day itself --
        # a real look-ahead bug found during this review, independent of any
        # blueprint deviation.
        shifted_price = df[c.price_col].shift(1)
        for window in (5, 10, 20):
            df[f"sma_{window}"] = shifted_price.rolling(window=window, min_periods=window).mean()
            df[f"ema_{window}"] = shifted_price.ewm(span=window, adjust=False, min_periods=window).mean()
            # Stationary counterparts. The sma_/ema_ columns above are raw ASPI price
            # LEVELS: the index runs from ~574 in 2000 to over 10,000 by 2022, so under a
            # chronological split a later test fold occupies a feature range disjoint from
            # anything the model saw in training -- tree models pin at the training
            # boundary and linear models extrapolate off the end. Thesis Sec. 3.5.1
            # requires stationary inputs, and these columns were never tested for it.
            # The price-relative ratio carries the same momentum information (where is
            # price now, relative to its own recent trend) with no trend in the level.
            df[f"price_to_sma_{window}"] = shifted_price / df[f"sma_{window}"] - 1.0
            df[f"price_to_ema_{window}"] = shifted_price / df[f"ema_{window}"] - 1.0

        # Shift by one day to enforce pre-shock boundary (up to t-1) and avoid look-ahead.
        shifted_returns = df["log_return"].shift(1)
        for window in (5, 10, 20):
            df[f"rolling_std_{window}"] = shifted_returns.rolling(window=window, min_periods=window).std()

        # Dedicated 30-day panic-proxy volatility (thesis Sec. 3.5.2/3.2.2) --
        # distinct from the 5/10/20-day momentum windows above; this is the
        # window that mirrors the ATV 30-day baseline used for Y2. Previously
        # missing entirely.
        df["rolling_std_30"] = shifted_returns.rolling(window=30, min_periods=30).std()

        df["squared_return"] = df["log_return"] ** 2
        return df

    def engineer_disaster_features(self, disaster_df: pd.DataFrame) -> pd.DataFrame:
        """Create transformed exogenous disaster covariates and one-hot disaster types."""
        c = self.config
        df = disaster_df.copy()
        df[c.disaster_date_col] = pd.to_datetime(df[c.disaster_date_col])

        clean_type = df[c.disaster_type_col].astype(str).str.strip().str.lower()
        keep_mask = ~clean_type.isin(EXCLUDED_DISASTER_TYPES)
        df = df[keep_mask].copy()

        # Inclusive threshold: the thesis (Sec. 3.3.2) and this notebook both state
        # ">= 1000 affected". The previous strict ">" silently disagreed with the
        # filter described everywhere else in the project.
        df = df[df[c.affected_col] >= 1000].copy()
        df["log_financial_damage"] = np.log1p(df[c.damage_col].clip(lower=0))
        df["log_population_affected"] = np.log1p(df[c.affected_col].clip(lower=0))

        # Pool types with too few events to support their own indicator. A one-hot
        # column with a single positive case is a memorisation key: the model can
        # isolate that one event perfectly in-sample, while out-of-sample the column
        # is all zeros and contributes nothing. Pooling also gives those events a
        # same-type partner for SMOGN, which otherwise skips them entirely.
        # Threshold fixed a priori at 3, not tuned against any result.
        type_counts = df[c.disaster_type_col].value_counts()
        rare_types = type_counts[type_counts < 3].index
        if len(rare_types) > 0:
            df[c.disaster_type_col] = df[c.disaster_type_col].where(
                ~df[c.disaster_type_col].isin(rare_types), "Other"
            )

        one_hot = pd.get_dummies(df[c.disaster_type_col], prefix="disaster", dtype=float)
        return pd.concat([df, one_hot], axis=1)

    def build_targets(self, market_df: pd.DataFrame, disaster_df: pd.DataFrame) -> pd.DataFrame:
        """Build Y1 (log return), Y2 (abnormal volume), and Y3 (recovery days) per event."""
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
            y1 = float(np.log(price_t / price_tm1))

            baseline_start = max(0, pos - 30)
            baseline_mean = market.iloc[baseline_start:pos][c.volume_col].mean()
            volume_t = market.iloc[pos][c.volume_col]
            y2 = float((volume_t / baseline_mean) - 1.0) if baseline_mean and not np.isnan(baseline_mean) else np.nan

            pre_disaster_baseline = price_tm1
            recovery_window = market.iloc[pos : pos + c.max_recovery_days + 1]
            recovered = recovery_window[recovery_window[c.price_col] >= pre_disaster_baseline]
            if recovered.empty:
                y3 = c.max_recovery_days
            else:
                # TRADING days, per thesis Sec. 3.2.2 -- the positional distance within
                # the market calendar. The previous version returned the calendar-day
                # difference (recovery_date - event_date).days while searching a window
                # of 91 trading ROWS, so the 90 cap actually bit at roughly 62 trading
                # days and any event recovering after that was indistinguishable from
                # one that never recovered at all.
                recovery_pos = market.index.get_loc(recovered.index[0])
                y3 = min(recovery_pos - pos, c.max_recovery_days)

            rows.append({
                c.disaster_date_col: event_date,
                "Y1_aspi_log_return": y1,
                "Y2_abnormal_volume": y2,
                "Y3_recovery_days": float(y3),
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
