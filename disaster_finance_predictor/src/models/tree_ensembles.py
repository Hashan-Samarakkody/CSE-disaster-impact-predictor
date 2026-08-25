"""Tree ensemble multi-output regressors."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def build_random_forest_model(
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_state: int = 42,
) -> MultiOutputRegressor:
    return MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
    )


def build_xgboost_model(
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    random_state: int = 42,
) -> MultiOutputRegressor:
    return MultiOutputRegressor(
        XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    )
