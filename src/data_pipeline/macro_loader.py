"""Macroeconomic and global-market control variables.

Annual World Bank series for Sri Lanka (GDP growth, CPI inflation, GDP level) and the
daily S&P 500 log return. Neither exists in the local archive, so both are pulled live
and unpinned -- which is why callers record a provenance sidecar beside the cached copy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_worldbank_macro(country: str = "LKA", start_year: int = 2000, end_year: int = 2025) -> pd.DataFrame:
    """Annual Sri Lanka GDP growth, CPI inflation and GDP level from the World Bank.

    One row per year: year, gdp_growth_pct, inflation_cpi_pct, gdp_current_usd. Annual
    granularity only -- coarser than the monthly CCPI the thesis would prefer.
    `gdp_current_usd` is the damage/GDP denominator, not a growth rate."""
    import wbgapi as wb

    series = {
        "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
        "FP.CPI.TOTL.ZG": "inflation_cpi_pct",
        "NY.GDP.MKTP.CD": "gdp_current_usd",
    }
    df = wb.data.DataFrame(list(series.keys()), economy=country, time=range(start_year, end_year + 1))
    df = df.rename(index=series).T
    df.index = df.index.str.replace("YR", "").astype(int)
    df.index.name = "year"
    return df.reset_index()


def load_sp500_global_control(start: str = "2000-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Daily S&P 500 (^GSPC) log return, the thesis's named global control
    (Table 4). Columns: date, sp500_log_return."""
    import yfinance as yf

    hist = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=False)
    if hist.empty:
        raise RuntimeError("^GSPC returned no data -- verify connectivity before trusting downstream results")

    hist = hist.reset_index()
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [c[0] for c in hist.columns]

    out = hist[["Date", "Close"]].rename(columns={"Date": "date"})
    out["date"] = pd.to_datetime(out["date"])
    out["sp500_log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    return out[["date", "sp500_log_return"]].dropna()
