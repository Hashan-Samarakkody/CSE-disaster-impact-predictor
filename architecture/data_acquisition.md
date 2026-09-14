# Data acquisition (stage 01)

Stage `01_data_acquisition.ipynb` turns a folder of raw files and a handful of web
sources into four cached tables. Nothing downstream ever touches a raw file again.

| Output artifact | What it holds |
|---|---|
| `market.parquet` | one row per trading day: `date`, `aspi_close`, `trading_volume` |
| `disasters.parquet` | one row per EM-DAT record for Sri Lanka |
| `macro.parquet` | one row per year: GDP growth, CPI inflation, GDP level (US$) |
| `sp500.parquet` | one row per trading day: S&P 500 log return |

## 1. The market series

`src/data_pipeline/cse_raw_loaders.py` reads the Colombo Stock Exchange archive in
`data/`. Two things make this harder than `pd.read_excel`.

**The layouts change.** The yearly per-security workbooks (`2000 data.xls` …
`2023 Data (2).xls`) use three incompatible sheet shapes across the sample:

- a *flat* table, one row per security per day (the 2001 and 2016–2023 layouts);
- a *block* table, where the trading date sits in a header row and the securities
  traded on that date are listed beneath it (roughly 2002–2015);
- minor variations in column labels within each of those.

The loader therefore never addresses a cell by position. `_find_flat_header_row` scans
for a row that *contains* the expected labels, and `_normalize` strips every character
outside `[a-z0-9]` before comparing, so `"Turnover (Rs.)"`, `"turnover rs"` and
`"TURNOVER  (RS)"` all match one key. Adding a year in a fourth layout means teaching
`_parse_*_sheet` one more shape, not rewriting the loader.

**The archive stops before the sample does.** The workbooks end 2023-06-28. Two
attempts to fill the gap are recorded in the code:

- `fetch_cse_gap_fill` pulls `^CSE` from Yahoo Finance. It is **verified dead** — the
  feed stopped updating around June 2019 and returns zero rows for any later window.
  The function is kept, unused, as a template and a warning; re-enabling it without
  re-checking the feed would silently produce an empty series.
- `external_sources.fetch_countryeconomy_aspi` scrapes daily ASPI closes published by
  countryeconomy.com, and `extend_market_series` appends **only** rows strictly after
  the archive's last date. The archive stays authoritative for the period it covers.
  The join is checked on the overlap: countryeconomy's 2023-06-28 close matches the
  archive's final row exactly.

Appended rows carry `trading_volume = NaN`, because that source publishes the index
level only. This is deliberate and it propagates: the per-target missing-value mask
drops those events from Y2 rather than inventing a volume for them.

Every row carries `price_source` and `volume_source`, so the provenance of any single
observation is visible in the table itself rather than implied by the code path.

## 2. The disaster records

`src/data_pipeline/emdat_loader.py` maps EM-DAT's 47-column public-table export onto
the handful of columns the feature builder expects. Three decisions matter.

**Dates are not all equally precise.** EM-DAT leaves `Start Day`, and sometimes
`Start Month`, blank for slow-onset events. Filling those with `1` invents a
January-1st event date — for a day-0 event study that means measuring the market's
response on a day the disaster did not begin. The fill is kept so the row survives, but
a `date_precision` column records what was actually known.

**Magnitude means different things for different hazards.** EM-DAT's `Magnitude`
column holds inundated area in km² for a flood and sustained wind in kph for a storm.
Averaging those into one number would be meaningless, so the loader splits them by
`Magnitude Scale` into `mag_area_km2` and `mag_wind_kph`, each with its own
availability flag.

**Missing damage is flagged, not hidden.** Absent damage figures become `0.0` so the
row survives, and `damage_source` records that the zero means *unknown*, not *none*.
This single column is the reason the exploratory analysis in stage 03 draws
zero-filled cells differently from observed ones — see
[`evaluation.md`](evaluation.md#exploratory-data-analysis-stage-03).

## 3. Macro and global controls

`src/data_pipeline/macro_loader.py` pulls annual Sri Lanka GDP growth, CPI inflation
and GDP level from the World Bank (`wbgapi`), and the daily S&P 500 log return from
Yahoo (`yfinance`). Neither exists in the local archive. Annual granularity is coarser
than the monthly CCPI the study would prefer; that is a stated limitation, not a
silent approximation. `gdp_current_usd` is not a growth rate — it is the denominator
of the `damage_to_gdp` feature built in stage 02.

## 4. External sources

`src/data_pipeline/external_sources.py` adds four feature blocks, all fetched with
stdlib `urllib`:

| Block | Source | What it contributes |
|---|---|---|
| A — hazard | NASA POWER, 7 district points | daily precipitation and wind around each event |
| B — physical severity | DesInventar Sendai (UNDRR/UNDP) | district-level losses over `[t−7, t+14]` |
| C — exchange rate | FRED `DEXSLUS` | daily LKR/USD, computed strictly from data at or before `t−1` |
| D — political calendar | Wikidata | signed distance to the nearest national election |

Every one of these features was written down in
[`../docs/EXTERNAL_DATA_PRE_DECLARATION.md`](../docs/EXTERNAL_DATA_PRE_DECLARATION.md)
**before** any of them was scored, together with what each was expected to do. One of
those written expectations turned out wrong and is recorded as wrong rather than
quietly revised.

## 5. Why every artifact has a provenance sidecar

World Bank and Yahoo data are pulled live from unpinned libraries, so the identical
code can return different numbers months apart. `save_frame` and `save_object` in
`notebooks/_shared.py` therefore write `<name>.provenance.json` beside every cached
file, recording the UTC retrieval time, the Python version, and the version of every
library that could have affected the result. Re-running stage 01 overwrites both.

That sidecar is what makes "re-run and compare" a meaningful check rather than a
coin flip.
