# Data Sources Reconciliation

Audit of `docs/data_sources_retrieval.csv` against artifact coverage and git history.

## Background

The original `data_sources_retrieval.csv` was created with
`true_download_date = 2026-08-13T07:13:46` for all nine sources. This date was
researcher-provided as the initial data-collection session. However, several artifact
files contain data with coverage end dates beyond Aug 13, 2026, which is physically
impossible for a file downloaded on that date. The git history shows all external data
files were committed at f076bd9 on 2026-09-21 and the pipeline ran on 2026-09-22.

## Coverage Audit

| artifact | coverage_end | true_download_date | verdict |
|----------|-------------|-------------------|---------|
| market (main CSE ASPI xlsx) | 2026-04-16 | 2026-08-13 | ✓ consistent |
| disasters (EM-DAT export) | 2025-11-27 | 2026-08-13 | ✓ consistent |
| macro (World Bank WDI) | annual, latest ~2024 | 2026-08-13 | ✓ consistent |
| sp500 (Yahoo Finance) | 2025-12-30 | 2026-08-13 | ✓ consistent |
| desinventar_lka | 2020-12-20 | 2026-08-13 | ✓ consistent |
| fred_dexslus (USD/LKR) | **2026-09-04** | 2026-08-13 | ✗ **conflict** |
| nasa_power_districts | **2026-09-01** | 2026-08-13 | ✗ **conflict** |
| sl_elections (Wikidata) | **2026-08-31** | 2026-08-13 | ✗ **conflict** |
| aspi_extension (CSE) | **2026-09-11** | 2026-08-13 | ✗ **conflict** |

## Conflict Resolution

For the four conflicting sources:

- `fred_dexslus`: FRED data through 2026-09-04 cannot have been downloaded on Aug 13.
  FRED is a live API (`fredapi`). The researcher must have re-run the data-pull after
  Aug 13, downloading updated rates. Bounded: [2026-09-04, 2026-09-21] (git commit).
  Corrected `true_download_date`: **2026-09-20** (assigned as day before commit, most
  conservative defensible date; actual date unknown, but certainly ≥ 2026-09-04).

- `nasa_power_districts`: NASA POWER API through 2026-09-01. Same logic.
  Corrected `true_download_date`: **2026-09-20**.

- `sl_elections`: Wikidata SPARQL through 2026-08-31. Election data for the Aug 2026
  general election must have been retrieved after Aug 13. Corrected: **2026-09-20**.

- `aspi_extension`: CSE data through 2026-09-11, downloaded manually from CSE website.
  Must have been downloaded between 2026-09-11 and 2026-09-21. Corrected: **2026-09-20**.

## Git Evidence

```
f076bd9  2026-09-21 11:29:21 +0530  update git ignore
   -- all artifacts/external/*.parquet files present at this commit
Pipeline run: 2026-09-22 (from provenance JSONs: pipeline_run_utc = 2026-09-22T05:48:*)
```

## Conclusion

Five sources (market main, disasters, macro, sp500, desinventar) retain Aug 13 as the
true download date, which is consistent with their data coverage. Four sources (FRED,
NASA, elections, aspi_extension) have been corrected to 2026-09-20 — the last full day
before the git commit that added the external data files. The corrected CSV is at
`docs/data_sources_retrieval.csv`.
