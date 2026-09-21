# Notebook 01, Data Acquisition

Documents `notebooks/01_data_acquisition.ipynb`.

## Purpose

Turn the raw archive and the live external sources into four cached tables that every later
stage reads. This is the only stage that touches the network and the only one that parses
Excel.

## Inputs

1. `data/raw/cse_market_indices_daily.xls`, the exchange's daily index publication.
2. `data/raw/cse_securities_YYYY.xls`, twenty four yearly per security workbooks.
3. `data/raw/emdat_sri_lanka_disasters.xlsx`, the EM-DAT export.
4. Live sources: World Bank via `wbgapi`, S&P 500 via `yfinance`, NASA POWER, DesInventar,
   FRED and Wikidata.

## Main processing stages

Section 1.1 sets up the environment. Section 1.2 validates the sources and prints an
inventory, so the executed notebook records what produced its numbers. Section 1.3 acquires
the four series: the market series, the disaster records, the macroeconomic controls and
the global market control. Section 1.4 cleans the market series. Section 1.5 extends the
ASPI beyond the local archive. Section 1.6 caches everything. Section 1.7 states what was
produced.

## Important functions called

| Function | Module | What it does |
|---|---|---|
| `build_market_dataframe` | `src/data/cse_market_data.py` | combines the index series and the summed volume series into one daily frame |
| `load_all_yearly_security_files` | `src/data/cse_market_data.py` | finds and parses every yearly workbook by pattern, reporting any that fail rather than silently producing a partial series |
| `load_aspi_index` | `src/data/cse_market_data.py` | reads the ASPI close from the two row merged header of the index sheet |
| `load_emdat` | `src/data/emdat_disasters.py` | maps the EM-DAT schema, records date precision, and preserves damage provenance |
| `load_worldbank_macro`, `load_sp500_global_control` | `src/data/macro_indicators.py` | the two control series |
| `forward_fill_missing_values` | `src/data/preprocessing.py` | fills market gaps in a chronology safe direction |
| `extend_market_series` | `src/data/external_sources.py` | extends the index past the end of the local archive |

## Important outputs

Cached to `artifacts/tables/`: `market.parquet`, `disasters.parquet`, `macro.parquet` and
`sp500.parquet`. Cached to `artifacts/external/`: the NASA POWER, DesInventar, FRED,
election and ASPI extension downloads. Each carries a provenance sidecar.

## Relationship with other notebooks

Nothing precedes this stage. Everything follows it. Stage 02 reads all four tables.

## Execution status

This notebook ships without stored cell outputs. Its six external sources are live and
unpinned, so a re run can return revised figures, which would move every downstream number
including the frozen targets. That is a data change rather than a reproduction, so the
cached tables under `artifacts/tables/` and `artifacts/external/`, each with a provenance
sidecar, are the record of what this stage produced. Re run it only when a deliberate data
refresh is intended, and expect every later stage to need re running with it.

## Methodological decisions made here

1. **Damage is missing, not zero.** When EM-DAT records no damage figure the value stays
   missing and a `damage_source` column records why. Zero filling at the loader would
   assert that a disaster caused no damage, which is a different claim from not knowing.
   Only eighteen of seventy four events carry a real damage figure, and that sparsity is
   the reason the hazard and DesInventar blocks exist.
2. **Date precision is recorded, not guessed.** EM-DAT leaves the start day blank for slow
   onset events. Filling it with the first of the month would invent an event date, which
   for an event study changes which session is day zero. The precision is carried through
   as a column and a sensitivity analysis in stage 06 re scores the headline result on
   exact day events only.
3. **Pre 2000 records are loaded but cannot be modelled.** The index series starts in
   January 2000. The count that stage 02 drops is printed here so the loss is visible.
4. **Volume is summed across securities, not taken from an index.** The exchange publishes
   volume per security, so the market wide series is constructed rather than read.

## Justification

Parsing the archive rather than using a convenience API is what makes the volume target
possible at all, since no public index series carries market wide traded volume for this
exchange over this period. Recording provenance for every live source follows standard
reproducibility practice for unpinned remote data: the retrieval date and library versions
are the only way a later reader can tell whether a number changed because the method
changed or because the source was revised.

## Execution requirements

Network access, and the raw workbooks present in `data/raw/`. This is the slowest stage
after the modelling stage. Re run it only when the raw data genuinely changes, because the
live sources can return revised figures that would move every downstream number, including
the frozen volume target.
