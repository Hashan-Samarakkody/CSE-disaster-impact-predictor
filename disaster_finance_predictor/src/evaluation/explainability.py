"""SHAP explainability helpers for tree ensembles and neural models."""

from __future__ import annotations

import shap


def build_tree_explainer(model):
    return shap.Explainer(model)


def build_kernel_explainer(model_predict_fn, background_data):
    return shap.KernelExplainer(model_predict_fn, background_data)


def global_importance_plot(shap_values, features):
    shap.summary_plot(shap_values, features, show=False)


def local_waterfall_plot(shap_values, index: int):
    shap.plots.waterfall(shap_values[index], show=False)
