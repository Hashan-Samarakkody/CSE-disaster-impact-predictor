"""External data sources that fill the gaps found by the missing-data audit.

Four feature blocks: NASA POWER hazard intensity, DesInventar district severity, FRED
LKR/USD, and Wikidata election dates; plus the countryeconomy.com ASPI extension past the
archive cutoff. Every feature was written down in `docs/EXTERNAL_DATA_PRE_DECLARATION.md`
before it was scored."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UA = {"User-Agent": "cse-disaster-thesis/1.0 (academic research)"}

# Seven district points, one per major climatic/administrative region. Fixed in the
# pre-declaration before any correlation with a target was computed.
POWER_POINTS = {
    "Colombo": (6.93, 79.86),
    "Jaffna": (9.66, 80.02),
    "Batticaloa": (7.72, 81.70),
    "NuwaraEliya": (6.97, 80.77),
    "Galle": (6.05, 80.22),
    "Anuradhapura": (8.31, 80.40),
    "Ratnapura": (6.68, 80.40),
}

# Sri Lanka Department of Meteorology heavy-rain advisory level, mm per 3 days.
# A published operational threshold, not one swept for fit.
HEAVY_RAIN_MM_3D = 50.0

# EM-DAT dates a multi-day event by onset; DesInventar records by district report date,
# which lags. Fixed a priori, never varied after scoring.
DI_WINDOW = (-7, 14)

DESINVENTAR_URL = "https://www.desinventar.net/DesInventar/download/DI_export_lka.zip"
FRED_FX_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXSLUS"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
COUNTRYECONOMY_URL = "https://countryeconomy.com/stock-exchange/sri-lanka"
WIKIDATA_URL = "https://query.wikidata.org/sparql"

# DesInventar event codes that are natural hazards. Epidemic, animal attack, fire,
# accident and the other administrative categories in the file are excluded -- the
# study's inclusion criteria cover natural disasters only.
NATURAL_EVENTS = {
    "FLOOD", "STRONG WIND", "DROUGHT", "HEAVY RAINS", "LANDSLIDE", "CYCLONE", "GALE",
    "TSUNAMI", "STORM", "HIGH WIND", "ROCK FALL", "LAND SUBSIDENCE", "TORNADO",
    "SEA SURGE", "FOREST FIRE", "HAIL STORM", "COASTAL EROSION",
}


def _get(url: str, timeout: int = 120) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


# --------------------------------------------------------------- NASA POWER


def fetch_nasa_power(cache_dir: Path, start: str = "19990101", end: str = "20260901",
                     refresh: bool = False) -> pd.DataFrame:
    """Daily precipitation and max wind for seven district points.

    Returns long format: date, district, precip_mm, wind_max_ms.

    This is the only severity source with no missingness and no reporting bias -- it is
    measured by instrument rather than assessed by a reporter afterwards, which is also
    why it is admissible in the ex-ante Model A specification that `financial_damage`
    is not.
    """
    cache = Path(cache_dir) / "nasa_power_districts.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    frames = []
    for name, (lat, lon) in POWER_POINTS.items():
        url = (f"{POWER_URL}?parameters=PRECTOTCORR,WS10M_MAX&community=AG"
               f"&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON")
        payload = json.loads(_get(url))["properties"]["parameter"]
        frame = pd.DataFrame({
            "date": pd.to_datetime(list(payload["PRECTOTCORR"]), format="%Y%m%d"),
            "district": name,
            "precip_mm": list(payload["PRECTOTCORR"].values()),
            "wind_max_ms": list(payload["WS10M_MAX"].values()),
        })
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    # POWER marks gaps with -999, not NaN. Left as NaN they would silently become zeros
    # in a sum, which for a rainfall feature reads as "dry" rather than "unknown".
    out[["precip_mm", "wind_max_ms"]] = out[["precip_mm", "wind_max_ms"]].where(
        out[["precip_mm", "wind_max_ms"]] > -900)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def build_hazard_features(event_dates, power: pd.DataFrame) -> pd.DataFrame:
    """Block A of the pre-declaration: six hazard-intensity features per event."""
    wide = power.pivot_table(index="date", columns="district", values="precip_mm")
    wind = power.pivot_table(index="date", columns="district", values="wind_max_ms")

    rows = []
    for raw in pd.to_datetime(pd.Series(list(event_dates))):
        # 3-day accumulation ending on the event day: the standard flood-generating window.
        window = wide.loc[(wide.index >= raw - pd.Timedelta(days=2)) & (wide.index <= raw)]
        gust = wind.loc[(wind.index >= raw - pd.Timedelta(days=2)) & (wind.index <= raw)]
        # Baseline ends at t-3 so it never overlaps the event window it is compared against.
        base = wide.loc[(wide.index >= raw - pd.Timedelta(days=33))
                        & (wide.index <= raw - pd.Timedelta(days=3))]

        per_district = window.sum(min_count=1)
        baseline_3d = 3.0 * base.mean().mean() if len(base) else np.nan
        rows.append({
            "event_date": raw,
            "hz_precip_max3d": float(per_district.max()) if len(window) else np.nan,
            "hz_precip_mean3d": float(per_district.mean()) if len(window) else np.nan,
            "hz_precip_spread3d": float(per_district.std()) if len(window) else np.nan,
            "hz_districts_wet": float((per_district > HEAVY_RAIN_MM_3D).sum()) if len(window) else np.nan,
            "hz_wind_max3d": float(gust.max().max()) if len(gust) else np.nan,
            "hz_precip_anom": (float(per_district.mean() / baseline_3d - 1.0)
                               if len(window) and baseline_3d and baseline_3d > 0 else np.nan),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- DesInventar


DI_FIELDS = ["serial", "level0", "level1", "name0", "name1", "evento",
             "fechano", "fechames", "fechadia", "muertos", "heridos", "desaparece",
             "afectados", "damnificados", "evacuados", "vivdest", "vivafec",
             "valorloc", "valorus", "nhectareas", "kmvias", "nescuelas", "nhospitales",
             "duracion", "causa"]


def fetch_desinventar(cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """The UNDRR/UNDP national disaster loss database for Sri Lanka.

    The download is an 18.5 MB zip holding a 1.21 GB XML, so it is streamed with
    iterparse and cached as a ~130k-row parquet. Only the `fichas` (record) section is
    read; the file's geography and event-code lookup tables are not needed.
    """
    cache = Path(cache_dir) / "desinventar_lka.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    cache.parent.mkdir(parents=True, exist_ok=True)
    raw_zip = Path(cache_dir) / "DI_export_lka.zip"
    if not raw_zip.exists() or refresh:
        raw_zip.write_bytes(_get(DESINVENTAR_URL, timeout=300))
    with zipfile.ZipFile(raw_zip) as archive:
        xml_name = next(n for n in archive.namelist() if n.endswith(".xml"))
        archive.extract(xml_name, cache_dir)
    xml_path = Path(cache_dir) / xml_name

    rows, inside = [], False
    for event, element in ET.iterparse(xml_path, events=("start", "end")):
        if event == "start" and element.tag == "fichas":
            inside = True
        elif event == "end":
            if element.tag == "fichas":
                element.clear()
                break
            if inside and element.tag == "TR":
                rows.append({k: (element.findtext(k) or "").strip() for k in DI_FIELDS})
                element.clear()

    frame = pd.DataFrame(rows)
    for col in ("fechano", "fechames", "fechadia", "muertos", "heridos", "desaparece",
                "afectados", "damnificados", "evacuados", "vivdest", "vivafec",
                "valorloc", "valorus", "nhectareas", "kmvias", "nescuelas",
                "nhospitales", "duracion"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["date"] = pd.to_datetime(
        dict(year=frame.fechano, month=frame.fechames, day=frame.fechadia), errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame.to_parquet(cache, index=False)
    xml_path.unlink(missing_ok=True)  # 1.2 GB, reconstructible from the cached zip
    return frame


def build_desinventar_features(event_dates, desinventar: pd.DataFrame) -> pd.DataFrame:
    """Block B: physical severity aggregated over [t-7, t+14] around each event.

    `di_available` is not optional. DesInventar ends 2020-12-20, so the 2021-2022 events
    match nothing -- and without the flag a zero reads as "no damage" rather than
    "outside the database's coverage", which is precisely the defect the audit found in
    `financial_damage`'s 47 zero fills.
    """
    natural = desinventar[desinventar.evento.isin(NATURAL_EVENTS)]
    lo, hi = DI_WINDOW

    rows = []
    for raw in pd.to_datetime(pd.Series(list(event_dates))):
        window = natural[(natural.date >= raw + pd.Timedelta(days=lo))
                         & (natural.date <= raw + pd.Timedelta(days=hi))]
        rows.append({
            "event_date": raw,
            "di_districts_hit": float(window.name1.nunique()),
            "di_affected_log": float(np.log1p(window.afectados.sum())),
            "di_houses_destroyed_log": float(np.log1p(window.vivdest.sum())),
            "di_houses_damaged_log": float(np.log1p(window.vivafec.sum())),
            "di_deaths_log": float(np.log1p(window.muertos.sum())),
            "di_records": float(len(window)),
            "di_available": float(len(window) > 0),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- FRED daily FX


def fetch_fred_fx(cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Daily LKR/USD (FRED `DEXSLUS`), 1973 to present.

    Replaces an annual macro series matched against day-0 events -- the single most
    indefensible frequency mismatch the audit found.
    """
    cache = Path(cache_dir) / "fred_dexslus.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    import io
    frame = pd.read_csv(io.BytesIO(_get(FRED_FX_URL)))
    frame.columns = ["date", "lkr_usd"]
    frame["date"] = pd.to_datetime(frame["date"])
    frame["lkr_usd"] = pd.to_numeric(frame["lkr_usd"], errors="coerce")
    frame = frame.dropna(subset=["lkr_usd"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame


def build_fx_features(event_dates, fx: pd.DataFrame) -> pd.DataFrame:
    """Block C: three FX features, all computed strictly from data at or before t-1.

    The exchange rate is a market price, so it gets the same pre-shock boundary as the
    ASPI features. Using the event day's own rate would leak.
    """
    fx = fx.sort_values("date").reset_index(drop=True)
    fx["fx_logret"] = np.log(fx.lkr_usd / fx.lkr_usd.shift(1))

    rows = []
    for raw in pd.to_datetime(pd.Series(list(event_dates))):
        prior = fx[fx.date < raw]  # strictly before the event day
        if len(prior) < 32:
            rows.append({"event_date": raw, "fx_logret_1": np.nan,
                         "fx_logret_5": np.nan, "fx_vol_30": np.nan})
            continue
        level = prior.lkr_usd.to_numpy()
        rows.append({
            "event_date": raw,
            "fx_logret_1": float(np.log(level[-1] / level[-2])),
            "fx_logret_5": float(np.log(level[-1] / level[-6])),
            "fx_vol_30": float(prior.fx_logret.iloc[-30:].std()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- ASPI extension


def fetch_countryeconomy_aspi(cache_dir: Path, months: list[str],
                              refresh: bool = False) -> pd.DataFrame:
    """Daily ASPI closes beyond the local archive's 2023-06-28 cutoff.

    `months` are 'YYYY-MM' strings; the page returns roughly 21 trading days each and
    the ranges overlap, so duplicates are dropped on date.
    """
    cache = Path(cache_dir) / "aspi_extension.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    rows = []
    for month in months:
        html = _get(f"{COUNTRYECONOMY_URL}?dr={month}", timeout=60).decode("utf-8", "replace")
        for chunk in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
            cells = [re.sub(r"<[^>]+>", "", c).strip()
                     for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", chunk, re.S)]
            if len(cells) >= 2 and re.match(r"\d{2}/\d{2}/\d{4}$", cells[0]):
                try:
                    rows.append({"date": pd.to_datetime(cells[0], format="%m/%d/%Y"),
                                 "aspi_close": float(cells[1].replace(",", ""))})
                except ValueError:
                    continue

    frame = (pd.DataFrame(rows).drop_duplicates(subset="date")
             .sort_values("date").reset_index(drop=True))
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame


def extend_market_series(market: pd.DataFrame, extension: pd.DataFrame,
                         date_col: str = "date", price_col: str = "aspi_close") -> pd.DataFrame:
    """Append post-cutoff ASPI rows to the local archive.

    Only rows strictly after the archive's last date are taken, so the archive stays
    authoritative for the period it covers. `trading_volume` is left NaN on appended rows
    (the source publishes the index level only); the per-target NaN mask drops those
    events from Y2 exactly as it already does for the 2000 volume gap."""
    market = market.copy()
    market[date_col] = pd.to_datetime(market[date_col])
    cutoff = market[date_col].max()

    tail = extension[extension.date > cutoff].rename(
        columns={"date": date_col, "aspi_close": price_col})
    if tail.empty:
        return market

    for col in market.columns:
        if col not in tail.columns:
            tail[col] = np.nan
    return (pd.concat([market, tail[market.columns]], ignore_index=True)
            .sort_values(date_col).reset_index(drop=True))


# --------------------------------------------------------------- elections


def fetch_elections(cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """National election dates from Wikidata.

    Motivated by a specific contamination the audit named: the 2005-11-17 presidential
    election falls four days before the 2005-11-21 event that carries the largest
    observed Y1 drop.
    """
    cache = Path(cache_dir) / "sl_elections.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    query = """SELECT ?e ?eLabel ?d WHERE {
      ?e wdt:P31/wdt:P279* wd:Q40231 ; wdt:P17 wd:Q854 ; wdt:P585 ?d .
      FILTER(YEAR(?d) >= 1999 && YEAR(?d) <= 2027)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    } ORDER BY ?d"""
    payload = json.loads(_get(f"{WIKIDATA_URL}?format=json&query={urllib.parse.quote(query)}",
                              timeout=90))
    frame = pd.DataFrame([{"date": pd.to_datetime(b["d"]["value"][:10]),
                           "name": b["eLabel"]["value"]}
                          for b in payload["results"]["bindings"]])
    frame = frame.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame


def build_election_features(event_dates, elections: pd.DataFrame) -> pd.DataFrame:
    """Block D: signed distance to the nearest national election, and a +/-5 day flag."""
    poll = np.sort(pd.to_datetime(elections.date).to_numpy())
    rows = []
    for raw in pd.to_datetime(pd.Series(list(event_dates))):
        deltas = (poll - np.datetime64(raw)) / np.timedelta64(1, "D")
        nearest = float(deltas[np.argmin(np.abs(deltas))]) if len(deltas) else np.nan
        rows.append({"event_date": raw,
                     "days_to_election": nearest,
                     "election_within_5d": float(abs(nearest) <= 5) if len(deltas) else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- assembly


def build_all_external_features(event_dates, cache_dir: Path,
                                refresh: bool = False) -> pd.DataFrame:
    """Blocks A-D joined on event_date, one row per event."""
    cache_dir = Path(cache_dir)
    dates = pd.to_datetime(pd.Series(list(event_dates)))
    out = pd.DataFrame({"event_date": dates})
    for builder, fetcher in (
        (build_hazard_features, fetch_nasa_power),
        (build_desinventar_features, fetch_desinventar),
        (build_fx_features, fetch_fred_fx),
        (build_election_features, fetch_elections),
    ):
        out = out.merge(builder(dates, fetcher(cache_dir, refresh=refresh)),
                        on="event_date", how="left")
    return out


EXTERNAL_FEATURE_BLOCKS = {
    "hazard": ["hz_precip_max3d", "hz_precip_mean3d", "hz_precip_spread3d",
               "hz_districts_wet", "hz_wind_max3d", "hz_precip_anom"],
    "desinventar": ["di_districts_hit", "di_affected_log", "di_houses_destroyed_log",
                    "di_houses_damaged_log", "di_deaths_log", "di_records", "di_available"],
    "fx": ["fx_logret_1", "fx_logret_5", "fx_vol_30"],
    "election": ["days_to_election", "election_within_5d"],
}


if __name__ == "__main__":
    # Self-check on synthetic frames: no network, no cache, just the window arithmetic.
    # Every one of these would have caught a real off-by-one during development.
    days = pd.date_range("2019-01-01", "2019-12-31")
    power = pd.concat([
        pd.DataFrame({"date": days, "district": name,
                      "precip_mm": np.where(days == pd.Timestamp("2019-06-10"), 100.0, 1.0),
                      "wind_max_ms": 5.0})
        for name in POWER_POINTS])

    haz = build_hazard_features([pd.Timestamp("2019-06-10")], power).iloc[0]
    # Event day 100mm + two prior days at 1mm each = 102 in every district.
    assert abs(haz.hz_precip_max3d - 102.0) < 1e-6, haz.hz_precip_max3d
    assert haz.hz_districts_wet == 7.0, haz.hz_districts_wet
    # All districts identical, so the spatial spread must be exactly zero.
    assert abs(haz.hz_precip_spread3d) < 1e-9, haz.hz_precip_spread3d
    # Baseline is 1mm/day -> 3mm over 3 days; 102/3 - 1 = 33.
    assert abs(haz.hz_precip_anom - 33.0) < 1e-6, haz.hz_precip_anom

    # A quiet day three weeks earlier must NOT pick up the storm.
    calm = build_hazard_features([pd.Timestamp("2019-05-20")], power).iloc[0]
    assert abs(calm.hz_precip_max3d - 3.0) < 1e-6, calm.hz_precip_max3d
    assert calm.hz_districts_wet == 0.0

    di = pd.DataFrame({
        "date": [pd.Timestamp("2019-06-12"), pd.Timestamp("2019-06-30")],
        "evento": ["FLOOD", "FLOOD"], "name1": ["Colombo", "Galle"],
        "afectados": [1000.0, 5.0], "vivdest": [10.0, 1.0], "vivafec": [20.0, 2.0],
        "muertos": [3.0, 0.0]})
    b = build_desinventar_features([pd.Timestamp("2019-06-10")], di).iloc[0]
    # +2 days is inside [-7, +14]; +20 days is outside. One district, not two.
    assert b.di_districts_hit == 1.0, b.di_districts_hit
    assert abs(b.di_affected_log - np.log1p(1000.0)) < 1e-9
    assert b.di_available == 1.0
    # An event with nothing in range must flag unavailable, not report zero damage.
    empty = build_desinventar_features([pd.Timestamp("2019-01-01")], di).iloc[0]
    assert empty.di_available == 0.0 and empty.di_records == 0.0

    fx = pd.DataFrame({"date": pd.date_range("2019-01-01", periods=200),
                       "lkr_usd": np.linspace(180, 200, 200)})
    f = build_fx_features([pd.Timestamp("2019-06-10")], fx).iloc[0]
    assert np.isfinite(f.fx_logret_1) and np.isfinite(f.fx_vol_30)
    # Strictly-before-t boundary: the event day's own rate must never be used.
    same_day = fx[fx.date == pd.Timestamp("2019-06-10")].lkr_usd.iloc[0]
    prior = fx[fx.date < pd.Timestamp("2019-06-10")].lkr_usd.to_numpy()
    assert abs(f.fx_logret_1 - np.log(prior[-1] / prior[-2])) < 1e-12
    assert prior[-1] != same_day

    polls = pd.DataFrame({"date": [pd.Timestamp("2019-06-15"), pd.Timestamp("2020-08-05")],
                          "name": ["a", "b"]})
    e = build_election_features([pd.Timestamp("2019-06-10")], polls).iloc[0]
    assert e.days_to_election == 5.0 and e.election_within_5d == 1.0
    # Sign convention: an election already held reads negative.
    past = build_election_features([pd.Timestamp("2019-06-20")], polls).iloc[0]
    assert past.days_to_election == -5.0, past.days_to_election

    archive = pd.DataFrame({"date": pd.date_range("2023-06-01", "2023-06-28"),
                            "aspi_close": 9000.0, "trading_volume": 1e6})
    ext = pd.DataFrame({"date": pd.date_range("2023-06-20", "2023-07-31"),
                        "aspi_close": 9500.0})
    merged = extend_market_series(archive, ext)
    # Overlapping days must keep the archive's value, not the external one.
    assert merged[merged.date == pd.Timestamp("2023-06-28")].aspi_close.iloc[0] == 9000.0
    assert merged.date.max() == pd.Timestamp("2023-07-31")
    assert merged[merged.date > pd.Timestamp("2023-06-28")].trading_volume.isna().all()

    print("external_sources.py self-check passed")
