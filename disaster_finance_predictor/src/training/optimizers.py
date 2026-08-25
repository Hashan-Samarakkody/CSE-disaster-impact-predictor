"""Hyperparameter optimization utilities."""

from __future__ import annotations

from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV


def run_grid_search(model, params, cv, scoring: str = "neg_root_mean_squared_error"):
    search = GridSearchCV(model, params, cv=cv, scoring=scoring, n_jobs=-1)
    return search


def run_bayesian_search(model, params, cv, n_iter: int = 30, scoring: str = "neg_root_mean_squared_error"):
    search = BayesSearchCV(model, params, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1)
    return search
