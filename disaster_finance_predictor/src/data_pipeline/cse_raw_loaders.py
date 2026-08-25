"""Loaders for the real, messy Colombo Stock Exchange archival files supplied for this
project (see `Data/` at the repo root) and for the live gap-fill period the local
archive does not cover.

Data provenance (verified by direct inspection, see the project plan for the full
inventory table):

- ``07Market Indices - Daily.xls`` (sheet "Index"): daily ASPI closing level,
  2-Jan-1985 -> 28-Jun-2023. This is the primary Y1 price source.
- The 23 yearly "``<year>`` [Dd]ata.xls[x]" files: per-security daily closing
  price/volume, 2001 -> Q1-2023. These have **no single consistent schema** --
  three distinct report layouts were produced across the years (a flat
  Date/SecurityID/Price/Volume table in 2001; a block-per-company layout with a
  repeated header roughly 2002-2015; a flat COMPANY-ID-per-row layout with
  drifting column counts/order 2016-2023). Rather than hard-coding a
  year -> layout lookup (fragile, silently wrong if a single file doesn't match
  the assumed pattern for its year), every file is parsed by locating its
  header text and/or by typing individual cells, so the correct layout is
  detected per file. Only Date and (aggregated) Volume are extracted -- the
  per-security identity is irrelevant to a market-wide volume proxy, so no
  attempt is made to reconcile the company-code conventions across eras.
- Live gap-fill for the period after the local archive (Jul-2023 -> 2025,
  which includes the 2025 "Ditwah" event) was attempted via ``yfinance``
  ticker ``^CSE`` -- one of the three sources the thesis itself names for
  this variable (CSE, Yahoo Finance, Bloomberg -- thesis Sec. 3.3.1) -- and
  verified DEAD during development: the ticker's underlying data feed on
  Yahoo's backend stopped updating around 2019 (confirmed via yfinance's
  ``.download()``/``.history()`` and Yahoo's raw chart API directly). No
  working free/scriptable source for that window was found. That period is
  therefore left out of the market series entirely rather than silently
  degraded with a stale/synthetic substitute; disasters falling in it are
  documented as an explicit out-of-scope limitation, not modeled. See
  ``fetch_cse_gap_fill`` for the (unused-by-default, kept for documentation)
  attempt.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Header-text normalization helpers
# ---------------------------------------------------------------------------

_DATE_EXACT_LABELS = {"date", "tradingdate"}
_VOLUME_SUBSTRINGS = ("sharevolume", "sharesno", "noofshares")


def _normalize(text: object) -> str:
    """Lowercase and strip everything but letters/digits, so header labels that
    differ only in spacing/punctuation/parenthetical units compare equal
    (e.g. "Share Volume" vs "SHARE VOLUME (No.)" both -> "sharevolume...")."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _find_flat_header_row(raw: pd.DataFrame, search_rows: int = 15) -> Optional[int]:
    """Locate the header row of a flat table by finding a row containing an
    exact "date"/"trading date" cell. Returns None if no such row exists in
    the first `search_rows` rows (signals this file is not flat-table shaped
    and should fall back to the block-table parser)."""
    for r in range(min(search_rows, len(raw))):
        row_norm = raw.iloc[r].map(_normalize)
        if row_norm.isin(_DATE_EXACT_LABELS).any():
            return r
    return None


def _find_col_by_substring(header_row: pd.Series, substrings: tuple[str, ...]) -> Optional[int]:
    norm = header_row.map(_normalize)
    for i, cell in enumerate(norm):
        if any(s in cell for s in substrings):
            return i
    return None


def _find_col_exact(header_row: pd.Series, labels: set[str]) -> Optional[int]:
    norm = header_row.map(_normalize)
    for i, cell in enumerate(norm):
        if cell in labels:
            return i
    return None


# ---------------------------------------------------------------------------
# Per-security yearly file parsing (aggregated to market-wide daily volume)
# ---------------------------------------------------------------------------


def _parse_flat_sheet(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Parse a flat per-security table (2001 layout; 2016-2023 layout) into a
    (date, volume) frame, one row per security per date. Returns None if this
    sheet is not flat-table shaped."""
    hdr_idx = _find_flat_header_row(raw)
    if hdr_idx is None:
        return None
    header = raw.iloc[hdr_idx]
    date_col = _find_col_exact(header, _DATE_EXACT_LABELS)
    vol_col = _find_col_by_substring(header, _VOLUME_SUBSTRINGS)
    if date_col is None or vol_col is None:
        return None

    data = raw.iloc[hdr_idx + 1 :, [date_col, vol_col]].copy()
    data.columns = ["date", "volume"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
    return data.dropna(subset=["date", "volume"])


def _parse_block_sheet(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Parse a block-per-company table (~2002-2015 layout): the trading date
    lives in column 0 as a native Timestamp on data rows, and everything else
    (title lines, "Company Id :" / "Short Name :" lines, repeated column
    headers, blank spacer rows) is text or NaN in column 0. The volume column
    index is located once via a header-text scan and assumed constant for the
    rest of the sheet (true in every file inspected)."""
    vol_col = None
    for r in range(min(15, len(raw))):
        found = _find_col_by_substring(raw.iloc[r], _VOLUME_SUBSTRINGS)
        if found is not None:
            vol_col = found
            break
    if vol_col is None or vol_col >= raw.shape[1]:
        return None

    date_col0 = raw.iloc[:, 0]
    # xlrd (.xls engine) returns native datetime.datetime, not pd.Timestamp;
    # pd.Timestamp is itself a datetime.datetime subclass, so this one check
    # covers both engines.
    is_data_row = date_col0.map(lambda v: isinstance(v, datetime.datetime))
    if not is_data_row.any():
        return None

    data = pd.DataFrame(
        {
            "date": date_col0[is_data_row],
            "volume": pd.to_numeric(raw.iloc[:, vol_col][is_data_row], errors="coerce"),
        }
    )
    return data.dropna(subset=["date", "volume"])


def load_yearly_security_file(path: Path) -> pd.DataFrame:
    """Load one yearly per-security file (any of the three real layouts,
    single- or multi-sheet) and return the market-wide aggregated daily
    trading volume for that file: columns ``date``, ``trading_volume``."""
    xls = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    for sheet_name in xls.sheet_names:
        raw = xls.parse(sheet_name, header=None)
        if len(raw) < 20:
            continue  # empty/placeholder sheets (observed e.g. in the 2015 file)
        parsed = _parse_flat_sheet(raw)
        if parsed is None:
            parsed = _parse_block_sheet(raw)
        if parsed is not None and not parsed.empty:
            frames.append(parsed)

    if not frames:
        raise ValueError(f"could not parse any recognizable layout from {path}")

    combined = pd.concat(frames, ignore_index=True)
    daily = combined.groupby("date", as_index=False)["volume"].sum()
    daily = daily.rename(columns={"volume": "trading_volume"})
    return daily.sort_values("date").reset_index(drop=True)


def load_all_yearly_security_files(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate every yearly per-security file found in
    ``data_dir`` into one continuous market-wide daily volume series. Files
    that fail to parse are skipped with a printed warning rather than
    aborting the whole load -- report exactly which years are missing rather
    than silently producing a partial series."""
    # \s* is allowed both before AND after the optional "(n)" suffix and right
    # before the extension -- one real file in this archive is literally
    # named "2015 Data .xlsx" (stray space before the dot); a tighter pattern
    # would silently exclude it from the candidate list with no warning at
    # all, which is worse than a reported parse failure.
    pattern = re.compile(r"^\d{4}\s*[Dd]ata\s*(\(\d+\))?\s*\.xlsx?$")
    candidates = sorted(p for p in data_dir.iterdir() if pattern.match(p.name))

    all_frames = []
    failed = []
    for path in candidates:
        try:
            all_frames.append(load_yearly_security_file(path))
        except Exception as exc:  # noqa: BLE001 - report and continue, don't abort the whole load
            failed.append((path.name, str(exc)))

    if failed:
        print(f"[cse_raw_loaders] WARNING: {len(failed)} yearly file(s) failed to parse:")
        for name, err in failed:
            print(f"  - {name}: {err}")

    if not all_frames:
        raise ValueError(f"no yearly security files could be parsed from {data_dir}")

    combined = pd.concat(all_frames, ignore_index=True)
    return combined.groupby("date", as_index=False)["trading_volume"].sum().sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Master ASPI index (07Market Indices - Daily.xls)
# ---------------------------------------------------------------------------


def load_aspi_index(path: Path, min_date: str = "2000-01-01") -> pd.DataFrame:
    """Load the daily ASPI series from the "Index" sheet of the master
    market-indices workbook. Header spans two merged rows (Excel rows 4-5);
    ASPI is column B. Returns columns ``date``, ``aspi_close``, filtered to
    ``min_date`` onward and de-duplicated on date (one exact duplicate row
    observed on 2010-06-30 in the source file)."""
    raw = pd.read_excel(path, sheet_name="Index", header=None, skiprows=5)
    df = raw.iloc[:, [0, 1]].copy()
    df.columns = ["date", "aspi_close"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["aspi_close"] = pd.to_numeric(df["aspi_close"], errors="coerce")
    df = df.dropna(subset=["date", "aspi_close"])
    df = df[df["date"] >= pd.Timestamp(min_date)]
    df = df.drop_duplicates(subset="date", keep="first")
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Live gap-fill: yfinance ^CSE for the period the local archive doesn't cover
# ---------------------------------------------------------------------------


@dataclass
class GapFillResult:
    prices: pd.DataFrame  # columns: date, aspi_close  (real)
    volume_is_reliable: bool  # False if Yahoo's Volume field came back all-zero/empty
    volume: pd.DataFrame  # columns: date, trading_volume (real if reliable, else empty)


def fetch_cse_gap_fill(start: str, end: str) -> GapFillResult:
    """Attempt to pull ^CSE from Yahoo Finance for the [start, end) window the
    local archive does not cover.

    VERIFIED DEAD, NOT USED BY DEFAULT: checked three ways during development
    (yfinance ``.download()``, ``.history()``, and Yahoo's raw chart API
    directly, bypassing yfinance's own delisting heuristic entirely) --
    ``^CSE``'s underlying data feed on Yahoo's backend stopped updating
    around 2019 (``regularMarketTime`` timestamp resolves to Jun-2019,
    ``regularMarketPrice: 0.0``, zero rows returned for any 2020+ range). The
    ticker existing as a browsable quote page is not the same as it having a
    live, queryable feed -- it does not. Kept here (unused by
    ``build_market_dataframe``, which now covers local-archive dates only) as
    a documented, working-as-designed function should Yahoo's feed for this
    ticker ever resume, or as a template for wiring in a different live
    source later. Do not silently re-enable without re-verifying first."""
    import yfinance as yf

    hist = yf.download("^CSE", start=start, end=end, progress=False, auto_adjust=False)
    if hist.empty:
        return GapFillResult(
            prices=pd.DataFrame(columns=["date", "aspi_close"]),
            volume_is_reliable=False,
            volume=pd.DataFrame(columns=["date", "trading_volume"]),
        )

    hist = hist.reset_index()
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [c[0] for c in hist.columns]

    prices = hist[["Date", "Close"]].rename(columns={"Date": "date", "Close": "aspi_close"})
    prices["date"] = pd.to_datetime(prices["date"])

    vol_series = pd.to_numeric(hist["Volume"], errors="coerce").fillna(0)
    volume_is_reliable = bool((vol_series > 0).mean() > 0.5)  # majority of days must have real volume

    volume = pd.DataFrame(columns=["date", "trading_volume"])
    if volume_is_reliable:
        volume = pd.DataFrame({"date": prices["date"], "trading_volume": vol_series})

    return GapFillResult(prices=prices, volume_is_reliable=volume_is_reliable, volume=volume)


# ---------------------------------------------------------------------------
# Master orchestrator: stitches local archive + live gap-fill into the one
# (date, aspi_close, trading_volume) frame the rest of the pipeline expects.
# ---------------------------------------------------------------------------


def build_market_dataframe(
    data_dir: Path,
    market_indices_filename: str = "07Market Indices - Daily.xls",
    min_date: str = "2000-01-01",
) -> pd.DataFrame:
    """Build the full ``date, aspi_close, trading_volume`` market series used
    throughout the pipeline, from the real local archive only.

    Live gap-fill for the period the archive doesn't cover (roughly
    Jul-2023 onward, which includes the 2025 "Ditwah" event) was attempted
    and verified dead (see ``fetch_cse_gap_fill`` docstring) -- there is no
    working free/scriptable source for it. Rather than silently degrade
    quality with a stale/synthetic substitute, that period is left out of
    this series entirely; any disaster event falling after the archive's
    real coverage ends is documented as an explicit out-of-scope limitation
    in the notebook, not modeled. Every row's ``price_source``/
    ``volume_source`` columns make this provenance explicit rather than
    implicit."""
    aspi_local = load_aspi_index(data_dir / market_indices_filename, min_date=min_date)
    volume_local = load_all_yearly_security_files(data_dir)

    price_df = aspi_local.assign(price_source="local_archive")
    volume_df = volume_local.assign(volume_source="local_archive")

    market = pd.merge(price_df, volume_df, on="date", how="left")
    market["volume_source"] = market["volume_source"].fillna("missing")
    market["trading_volume"] = market["trading_volume"].where(market["volume_source"] != "missing")

    return market.sort_values("date").reset_index(drop=True)
