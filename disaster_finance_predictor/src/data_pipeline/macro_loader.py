"""Macroeconomic and global-market control variables (thesis Table 4:
"Macroeconomic Stability" and "Global Market Conditions").

Neither series exists in the local `Data/` archive at all -- both are pulled
live. Sources match exactly what the thesis itself names in Table 4, not a
substituted source:

- Macroeconomic Stability -> "Central Bank of Sri Lanka; World Bank" (thesis
  Table 4). CBSL's higher-frequency series (monthly CCPI, daily LKR/USD,
  policy rate) have no confirmed programmatic API and are not pulled here --
  documented as a limitation. World Bank's annual GDP growth and CPI
  inflation for Sri Lanka are pulled via `wbgapi` (confirmed working, real
  PyPI package by World Bank staff), used as the best-effort macro-control
  proxy.
- Global Market Conditions -> "Yahoo Finance / Bloomberg" (thesis Table 4).
  S&P 500 (^GSPC) daily return via `yfinance` -- confirmed working (unlike
  ^CSE, this is an actively-traded, actively-updated ticker; verified live
  during development, not assumed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_worldbank_macro(country: str = "LKA", start_year: int = 2000, end_year: int = 2025) -> pd.DataFrame:
    """Annual Sri Lanka GDP growth (%), CPI inflation (%), and GDP level
    (current US$) from the World Bank via `wbgapi`. Returns one row per year,
    columns: year, gdp_growth_pct, inflation_cpi_pct, gdp_current_usd. Annual
    granularity only -- coarser than the thesis's ideal (monthly CCPI),
    documented as a limitation rather than silently presented as
    higher-resolution than it is.

    `gdp_current_usd` (NY.GDP.MKTP.CD) is a separate series from the growth
    rate above -- it's the economy's absolute size that year, used downstream
    to normalize disaster damage figures (damage / GDP), which raw log-damage
    alone can't express."""
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
