"""Loader for the EM-DAT export in `data/`.

Maps EM-DAT's 47-column public-table schema onto the columns `FeatureEngineer` expects,
CPI-adjusts damage to a common base year, and records how precisely each event is dated.
Absent damage figures are zero-filled with a `damage_source` flag, never silently."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_emdat(path: Path, sheet_name: str = "EM-DAT Data") -> pd.DataFrame:
    """Load the raw EM-DAT export and map it onto the repo's expected disaster
    schema. Does NOT apply the thesis's inclusion filters (>=1000 affected,
    exclude Biological) -- that's `FeatureEngineer.engineer_disaster_features`'s
    job; this loader only handles the real-file-schema -> repo-schema mapping
    so both stages stay independently testable."""
    raw = pd.read_excel(path, sheet_name=sheet_name)

    # EM-DAT leaves Start Day blank for slow-onset events. The fill of 1 is kept so the row
    # survives, but `date_precision` records the imprecision: 5 modelled events are droughts
    # with no start day, and a drought has no meaningful day-0 at all.
    date_precision = np.select(
        [raw["Start Month"].isna(), raw["Start Day"].isna()],
        ["year_only", "month_only"],
        default="exact_day",
    )
    event_date = pd.to_datetime(
        {
            "year": raw["Start Year"],
            "month": raw["Start Month"].fillna(1),
            "day": raw["Start Day"].fillna(1),
        },
        errors="coerce",
    )

    adjusted = raw["Total Damage, Adjusted ('000 US$)"]
    raw_damage = raw["Total Damage ('000 US$)"]
    damage_source = np.select(
        [adjusted.notna(), raw_damage.notna()],
        ["emdat_cpi_adjusted", "emdat_unadjusted_fallback"],
        default="missing_zero_filled",
    )
    financial_damage = adjusted.fillna(raw_damage).fillna(0.0) * 1000.0  # EM-DAT reports in '000 US$

    # Magnitude's unit varies by hazard (Km2 inundated for floods, Kph wind for storms), so
    # it is split by scale with a flag each. It is populated more often than Total Damage
    # and, unlike damage, is a measurement rather than a post-hoc financial assessment.
    magnitude = pd.to_numeric(raw.get("Magnitude"), errors="coerce")
    scale = raw.get("Magnitude Scale", pd.Series(index=raw.index, dtype=object)).astype(str)

    def _by_scale(unit):
        return magnitude.where(scale.str.strip().str.lower() == unit)

    area_km2 = _by_scale("km2")
    wind_kph = _by_scale("kph")

    out = pd.DataFrame(
        {
            "event_date": event_date,
            "event_date_precision": date_precision,
            "date_is_exact": (date_precision == "exact_day").astype(float),
            "disaster_type": raw["Disaster Type"],
            "disaster_group": raw["Disaster Group"],
            "disaster_subgroup": raw["Disaster Subgroup"],
            "financial_damage": financial_damage,
            "damage_source": damage_source,
            "population_affected": raw["Total Affected"],
            # Each gets an availability flag. The feature table fills NaN with 0.0, and without
            # the flag a zero reads as "nobody died" rather than "EM-DAT recorded no figure" --
            # the defect the audit found in financial_damage's zero fills.
            "total_deaths": pd.to_numeric(raw.get("Total Deaths"), errors="coerce"),
            "deaths_available": pd.to_numeric(raw.get("Total Deaths"),
                                              errors="coerce").notna().astype(float),
            "no_homeless": pd.to_numeric(raw.get("No. Homeless"), errors="coerce"),
            "homeless_available": pd.to_numeric(raw.get("No. Homeless"),
                                                errors="coerce").notna().astype(float),
            "mag_area_km2": area_km2,
            "mag_wind_kph": wind_kph,
            "mag_area_available": area_km2.notna().astype(float),
            "mag_wind_available": wind_kph.notna().astype(float),
            "dis_no": raw["DisNo."],
            "event_name": raw["Event Name"],
        }
    )
    return out.dropna(subset=["event_date"]).sort_values("event_date").reset_index(drop=True)
