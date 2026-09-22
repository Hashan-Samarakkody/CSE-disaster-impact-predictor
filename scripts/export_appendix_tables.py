"""Export the appendix tables the review asks for: the event list and the search grids.

Both are exports of what the final run already produced or already declares in code. No
model is fitted here and no number is recomputed, so running this cannot change a result.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.artifact_store import artifact_file, save_frame

EVENT_COLS = ["event_date", "disaster_type", "date_is_exact", "population_affected",
              "total_deaths", "financial_damage"]

# The declared search grids, as the code sets them. Kept here so the appendix can state
# what was searched without a reader having to read three modules.
SEARCH_GRIDS = [
    ("ridge", "notebook and grid", "alpha", "logspace(-3, 3, 13)"),
    ("elastic_net", "grid", "alpha, l1_ratio", "alpha logspace(-3, 1, 9); l1_ratio 0.1, 0.5, 0.9"),
    ("random_forest", "notebook and grid", "n_estimators, max_depth, min_samples_leaf",
     "300; None, 3, 5; 1, 3"),
    ("xgboost", "notebook and grid", "n_estimators, max_depth, learning_rate",
     "200, 400; 2, 3; 0.03, 0.1"),
    ("gp", "notebook (T8)", "kernel", "Matern nu 0.5, 1.5, 2.5, each times a constant kernel"),
    ("svr", "notebook (T8)", "C", "0.1, 1.0, 10.0"),
    ("quantile", "notebook (T8)", "alpha", "0.001, 0.01, 0.1"),
    ("mlp", "grid (T8)", "hidden_layer_sizes, alpha", "(8,), (16,); 1.0, 10.0"),
]


def main() -> None:
    data = pd.read_parquet(artifact_file("dataset.parquet"))
    events = data[[c for c in EVENT_COLS if c in data.columns]].copy()
    events.insert(0, "event_id", range(1, len(events) + 1))
    events = events.sort_values("event_date").reset_index(drop=True)
    save_frame(events, "event_list",
               "The modelled events, one row each, for the appendix event list.")
    print(f"wrote event_list.parquet ({len(events)} events, "
          f"{events['event_date'].min()} to {events['event_date'].max()})")

    grids = pd.DataFrame(SEARCH_GRIDS,
                         columns=["model", "where", "hyperparameter", "values searched"])
    save_frame(grids, "search_grids",
               "The declared hyperparameter grids, every one selected on purged inner splits.")
    print(f"wrote search_grids.parquet ({len(grids)} models)")

    selected = pickle.loads((artifact_file("selected_features.pkl")).read_bytes())
    rows = [{"target": target, "n_selected": len(features),
             "features": ", ".join(map(str, features))}
            for target, features in selected.items()]
    save_frame(pd.DataFrame(rows), "selected_features_final",
               "Features the final per-target models selected, for the appendix.")
    print(f"wrote selected_features_final.parquet ({len(rows)} targets)")


if __name__ == "__main__":
    main()
