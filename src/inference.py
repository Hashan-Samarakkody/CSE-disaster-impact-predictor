"""Prediction serving for the demo app: real historical events, replayed with
user-editable severity inputs, through the models the pipeline already fitted.

Scope, same as everywhere else in this project: the market/macro/hazard columns of a
feature row are the real conditions on that event's date and are NOT recomputed from
a hypothetical severity -- they are knowable only in hindsight. Editing severity and
re-scoring answers "what would this model have said about a more/less severe version
of this real event", not "what will a future disaster do". That is the ex-post
attribution scope the whole study operates under, not a limitation specific to this
demo.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

DISASTER_TYPES = ["Drought", "Flood", "Other", "Storm"]

TARGET_LABELS = {
    "Y1_ASPI_5D_Forward_LogReturn_Pct": "ASPI % deviation, event day vs 30d pre-disaster mean",
    "Y2_abnormal_volume": "Abnormal trading volume (V / 30d avg - 1)",
    "Y3_recovery_days": "Recovery time (trading days, capped at 90)",
    "Y1_EventWindow_0_5_LogReturn_Pct": "5-day cumulative return",
    "Y1_EventWindow_0_10_LogReturn_Pct": "10-day cumulative return",
}

LABEL_DESCRIPTIONS = {
    "C1_negative_return": "Day-0 return is negative",
    "C1b_adverse_move": "Day-0 return below this fold's bottom tercile",
    "C2_volume_spike": "Volume exceeds its own 30-day baseline",
    "C3_recovers_in_90": "Price recovers within 90 trading days",
    "C3b_slow_recovery": "Recovery slower than this fold's training median",
    "C4_car5_negative": "5-day cumulative return is negative",
}


class ModelBundle:
    """Loads every cached artifact once; the app keeps one instance alive."""

    def __init__(self) -> None:
        self.dataset = pd.read_parquet(ARTIFACTS / "dataset.parquet")
        self.spec = json.loads((ARTIFACTS / "feature_spec.json").read_text(encoding="utf-8"))
        self.feature_cols = self.spec["FEATURE_COLS"]
        self.target_cols = self.spec["TARGET_COLS"]

        with open(ARTIFACTS / "selected_features.pkl", "rb") as fh:
            self.selected_features = pickle.load(fh)
        with open(ARTIFACTS / "final_rf_models.pkl", "rb") as fh:
            self.regressors = pickle.load(fh)
        with open(ARTIFACTS / "final_classifiers.pkl", "rb") as fh:
            self.classifiers = pickle.load(fh)
        with open(ARTIFACTS / "final_hurdle_model.pkl", "rb") as fh:
            self.hurdle = pickle.load(fh)

        self.X = self.dataset[self.feature_cols].fillna(0.0)
        self.y = self.dataset[self.target_cols]

        self.target_bounds = {
            "Y1_ASPI_5D_Forward_LogReturn_Pct": (None, None),
            "Y2_abnormal_volume": (-1.0, None),
            "Y3_recovery_days": (0.0, 90.0),
            "Y1_EventWindow_0_5_LogReturn_Pct": (None, None),
            "Y1_EventWindow_0_10_LogReturn_Pct": (None, None),
        }

    # ------------------------------------------------------------ events

    def list_events(self) -> pd.DataFrame:
        cols = ["event_date", "disaster_type", "population_affected", "financial_damage",
                "total_deaths", "no_homeless", "mag_area_km2", "mag_wind_kph"]
        cols += [c for c in self.target_cols]
        out = self.dataset[cols].copy()
        out.insert(0, "event_id", self.dataset.index)
        out["event_date"] = pd.to_datetime(out["event_date"]).dt.date.astype(str)
        return out.sort_values("event_date")

    def event_row(self, event_id: int) -> pd.Series:
        return self.dataset.loc[event_id]

    # ------------------------------------------------------------ feature reconstruction

    def build_feature_row(self, event_id: int, overrides: dict) -> pd.Series:
        """Real event's feature row, with severity overrides applied and every
        derived column that depends on them recomputed. Everything else -- market,
        macro, hazard, DesInventar, FX, election features -- stays fixed at the
        real value observed on that event's date."""
        row = self.event_row(event_id).copy()

        if "disaster_type" in overrides:
            chosen = overrides["disaster_type"]
            for t in DISASTER_TYPES:
                row[f"disaster_{t}"] = 1.0 if t == chosen else 0.0

        if "financial_damage" in overrides:
            row["financial_damage"] = float(overrides["financial_damage"])
            row["log_financial_damage"] = float(np.log1p(max(0.0, row["financial_damage"])))
            gdp = row.get("gdp_current_usd")
            row["damage_to_gdp"] = (row["financial_damage"] / gdp
                                    if gdp and gdp > 0 else np.nan)

        if "population_affected" in overrides:
            row["population_affected"] = float(overrides["population_affected"])
            row["log_population_affected"] = float(np.log1p(max(0.0, row["population_affected"])))

        if "total_deaths" in overrides:
            row["total_deaths"] = float(overrides["total_deaths"])
            row["deaths_available"] = 1.0

        if "no_homeless" in overrides:
            row["no_homeless"] = float(overrides["no_homeless"])
            row["homeless_available"] = 1.0

        if "mag_area_km2" in overrides:
            row["mag_area_km2"] = float(overrides["mag_area_km2"])
            row["mag_area_available"] = 1.0

        if "mag_wind_kph" in overrides:
            row["mag_wind_kph"] = float(overrides["mag_wind_kph"])
            row["mag_wind_available"] = 1.0

        row["log_damage_x_flood"] = float(row["log_financial_damage"] * row["disaster_Flood"])
        numeric = pd.to_numeric(row.reindex(self.feature_cols), errors="coerce")
        return numeric.fillna(0.0)

    # ------------------------------------------------------------ prediction

    def clip_to_bounds(self, target: str, value: float) -> float:
        lo, hi = self.target_bounds[target]
        return float(np.clip(value, lo if lo is not None else -np.inf,
                             hi if hi is not None else np.inf))

    def predict_regression(self, feature_row: pd.Series) -> dict:
        out = {}
        for target, model in self.regressors.items():
            feats = self.selected_features[target]
            x = feature_row[feats].to_frame().T
            pred = self.clip_to_bounds(target, float(model.predict(x)[0]))
            out[target] = pred
        return out

    def predict_classification(self, feature_row: pd.Series) -> dict:
        out = {}
        for name, bundle in self.classifiers.items():
            feats = bundle["features"]
            x = feature_row[feats].to_frame().T
            model = bundle["model"]
            proba = float(model.predict_proba(x)[0, 1])
            out[name] = {
                "probability": proba,
                "predicted_positive": proba >= 0.5,
                "family": bundle["family"],
                "historical_auc": bundle["historical_auc"],
                "historical_balanced_accuracy": bundle["historical_balanced_accuracy"],
                "historical_prevalence": bundle["historical_prevalence"],
                "beats_baseline": bundle["beats_baseline"],
            }
        return out

    def predict_hurdle(self, feature_row: pd.Series) -> float:
        import warnings
        feats = self.hurdle["features"]
        x = feature_row[feats].to_frame().T
        # HurdleRecoveryModel.predict() does `np.asarray(X)` internally, which drops
        # column names before the pipeline sees them -- cosmetic only, not a bug.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return float(self.hurdle["model"].predict(x)[0])

    def naive_baselines(self, target: str) -> dict:
        """The two baselines every model in this study is measured against."""
        return {"constant_zero": 0.0, "training_mean": float(self.y[target].mean())}


_bundle: ModelBundle | None = None


def get_bundle() -> ModelBundle:
    global _bundle
    if _bundle is None:
        _bundle = ModelBundle()
    return _bundle
