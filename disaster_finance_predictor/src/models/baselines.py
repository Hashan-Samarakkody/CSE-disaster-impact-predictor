"""Baseline multi-output regression models."""

from __future__ import annotations

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


def build_ols_model() -> MultiOutputRegressor:
    return MultiOutputRegressor(LinearRegression())


def build_ridge_model(alpha: float = 1.0) -> MultiOutputRegressor:
    return MultiOutputRegressor(Ridge(alpha=alpha))


def build_svm_rbf_model(C: float = 10.0, gamma: str | float = "scale", epsilon: float = 0.1) -> MultiOutputRegressor:
    return MultiOutputRegressor(SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon))
