"""Loaders for the archival Colombo Stock Exchange workbooks in `data/`.

The yearly per-security files use three incompatible sheet layouts across 2000-2023, so
headers are located by content rather than by position. Also loads the daily ASPI and
sector index series, and the Yahoo gap-fill for the window the archive does not cover.
Layout details: `architecture/data_acquisition.md`."""

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
    # \s* is allowed before and after the optional "(n)" suffix and before the extension:
    # one real file is named "2015 Data .xlsx" (stray space), and a tighter pattern would
    # drop it from the candidate list with no warning at all.
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
    """Pull ^CSE from Yahoo for the window the local archive does not cover.

    VERIFIED DEAD AND NOT USED BY DEFAULT: Yahoo's feed for this ticker stopped updating
    around Jun-2019 and returns zero rows for any 2020+ range (checked via `download()`,
    `history()` and the raw chart API). Kept as a template for a future live source.
    Do not re-enable without re-verifying the feed first."""
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


# Master orchestrator: stitches the local archive into the one
# (date, aspi_close, trading_volume) frame the pipeline expects.


def build_market_dataframe(
    data_dir: Path,
    market_indices_filename: str = "07Market Indices - Daily.xls",
    min_date: str = "2000-01-01",
) -> pd.DataFrame:
    """Build the `date, aspi_close, trading_volume` series from the local archive only.

    The post-archive period has no working free gap-fill source (see `fetch_cse_gap_fill`),
    so it is left out rather than filled with a stale substitute; events falling after
    coverage ends are documented out of scope. `price_source` / `volume_source` carry the
    provenance of every row."""
    aspi_local = load_aspi_index(data_dir / market_indices_filename, min_date=min_date)
    volume_local = load_all_yearly_security_files(data_dir)

    price_df = aspi_local.assign(price_source="local_archive")
    volume_df = volume_local.assign(volume_source="local_archive")

    market = pd.merge(price_df, volume_df, on="date", how="left")
    market["volume_source"] = market["volume_source"].fillna("missing")
    market["trading_volume"] = market["trading_volume"].where(market["volume_source"] != "missing")

    return market.sort_values("date").reset_index(drop=True)


# Stored NORMALISED, because the membership test below applies `_normalize` (strips
# everything outside [a-z0-9]). S&P Sri Lanka 20 is excluded: it starts only in 2012 and
# is ~51% populated over the study window, against 87-100% for the sector columns.
EXCLUDED_INDEX_COLUMNS = {"spsrilanka20"}


def load_sector_indices(path: Path, min_date: str = "2000-01-01",
                        min_coverage: float = 0.80) -> pd.DataFrame:
    """Load every daily sector index from the "Index" sheet, in long form.

    Returns columns ``date``, ``sector``, ``close``. Sectors whose non-null coverage over
    the retained window falls below ``min_coverage`` are dropped, which removes columns
    that only begin part-way through the sample and would otherwise make the panel look
    balanced when it is not.
    """
    header = pd.read_excel(path, sheet_name="Index", header=None, skiprows=3, nrows=1)
    names = {i: str(v).strip() for i, v in header.iloc[0].items()
             if isinstance(v, str) and str(v).strip()}

    raw = pd.read_excel(path, sheet_name="Index", header=None, skiprows=5)
    dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")

    frames = []
    for col, name in names.items():
        if _normalize(name) in EXCLUDED_INDEX_COLUMNS or col >= raw.shape[1]:
            continue
        values = pd.to_numeric(raw.iloc[:, col], errors="coerce")
        block = pd.DataFrame({"date": dates, "sector": name, "close": values})
        block = block.dropna(subset=["date"])
        block = block[block["date"] >= pd.Timestamp(min_date)]
        if block.empty:
            continue
        if block["close"].notna().mean() < min_coverage:
            continue
        frames.append(block.dropna(subset=["close"]))

    if not frames:
        return pd.DataFrame(columns=["date", "sector", "close"])
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["date", "sector"], keep="first")
    return out.sort_values(["sector", "date"]).reset_index(drop=True)
