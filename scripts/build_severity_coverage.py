"""Build severity_coverage.parquet — missingness of severity/damage variables."""

from __future__ import annotations

import json
import pandas as pd
import numpy as np

from src.utils.artifact_store import artifact_file, save_frame


SEVERITY_VARS = [
    "financial_damage",
    "population_affected",
    "total_deaths",
    "no_homeless",
    "mag_area_km2",
    "mag_wind_kph",
    "hz_precip_max3d",
    "hz_precip_mean3d",
    "hz_wind_max3d",
    "di_districts_hit",
]

HAZARD_TYPES = ["Flood", "Storm", "Drought"]


def main():
    dataset = pd.read_parquet(artifact_file("dataset.parquet"))
    n_total = len(dataset)

    event_dates = pd.to_datetime(dataset["event_date"])
    disaster_type = dataset["disaster_type"].fillna("Other")

    # Period splits by year
    year = event_dates.dt.year
    pre2009 = year < 2009
    p2009_2019 = (year >= 2009) & (year <= 2019)
    post2019 = year > 2019

    rows = []
    for var in SEVERITY_VARS:
        if var not in dataset.columns:
            continue

        col = dataset[var]
        n_observed = int(col.notna().sum())
        pct_observed = float(n_observed / n_total * 100)

        # by_hazard: pct observed for each disaster type
        by_hazard = {}
        for dtype in HAZARD_TYPES:
            mask = disaster_type == dtype
            sub = col[mask]
            if len(sub) == 0:
                by_hazard[dtype.lower()] = float("nan")
            else:
                by_hazard[dtype.lower()] = float(sub.notna().sum() / len(sub) * 100)
        # Other
        other_mask = ~disaster_type.isin(HAZARD_TYPES)
        sub_other = col[other_mask]
        by_hazard["other"] = float(
            sub_other.notna().sum() / len(sub_other) * 100
        ) if len(sub_other) > 0 else float("nan")

        # by_period
        by_period = {}
        for period_name, period_mask in [
            ("pre2009", pre2009),
            ("2009to2019", p2009_2019),
            ("post2019", post2019),
        ]:
            sub = col[period_mask]
            if len(sub) == 0:
                by_period[period_name] = float("nan")
            else:
                by_period[period_name] = float(sub.notna().sum() / len(sub) * 100)

        rows.append({
            "variable": var,
            "n_observed": n_observed,
            "n_total": n_total,
            "pct_observed": pct_observed,
            "by_hazard": json.dumps({k: round(v, 1) if not (v != v) else None
                                     for k, v in by_hazard.items()}),
            "by_period": json.dumps({k: round(v, 1) if not (v != v) else None
                                      for k, v in by_period.items()}),
        })

    df = pd.DataFrame(rows)
    save_frame(df, "severity_coverage",
               note="Missingness coverage of severity/damage variables. "
                    "by_hazard and by_period are JSON strings of pct_observed per group.")


if __name__ == "__main__":
    main()
