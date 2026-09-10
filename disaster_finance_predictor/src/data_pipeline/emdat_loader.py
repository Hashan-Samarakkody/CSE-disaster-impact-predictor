"""Loader for the real EM-DAT export supplied for this project
(`public_emdat_custom_request_2026-02-09_...xlsx`), mapping its native 47-column
public-table schema onto the columns `FeatureEngineer` (see `feature_eng.py`)
expects: `event_date`, `disaster_type`, `financial_damage`, `population_affected`.

Verified live against doc.emdat.be: "Total Damage, Adjusted ('000 US$)"
applies the OECD Consumer Price Index to inflate the raw "Total Damage" figure
relative to the disaster's Start Year -- i.e. it already satisfies the
thesis's Sec. 3.5.3 PPP/inflation-adjustment requirement natively. It is used
as `financial_damage` when present. EM-DAT's own damage-estimate coverage is
sparse, however (many smaller Sri Lankan events have no damage figure at all
-- a known EM-DAT limitation, thesis Sec. 3.9.1, citing Caldera & Wirasinghe,
2022), so unadjusted "Total Damage" is used as a fallback when the adjusted
figure is missing, and 0 only as a last resort. Which source was used for
each row is preserved in a `damage_source` column rather than hidden, so the
notebook can show exactly how much of the disaster set has real-vs-imputed
damage figures.
"""

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

    out = pd.DataFrame(
        {
            "event_date": event_date,
            "disaster_type": raw["Disaster Type"],
            "disaster_group": raw["Disaster Group"],
            "disaster_subgroup": raw["Disaster Subgroup"],
            "financial_damage": financial_damage,
            "damage_source": damage_source,
            "population_affected": raw["Total Affected"],
            "dis_no": raw["DisNo."],
            "event_name": raw["Event Name"],
        }
    )
    return out.dropna(subset=["event_date"]).sort_values("event_date").reset_index(drop=True)
