# Data sources

Every input the pipeline reads, where it comes from, what it is used for, and what its
terms of use are. Nothing in this project is synthetic except the oversampled training rows
described in section 5, and those never reach a validation or test fold.

## 1. Local raw inputs

These live in `data/raw/` and are read by stage 01. They are the files that must be present
before anything runs.

| File | Source | Used for |
|---|---|---|
| `cse_market_indices_daily.xls` | Colombo Stock Exchange daily indices publication | the ASPI close series and every sector index |
| `cse_securities_2000.xls` through `cse_securities_2023.xls` | CSE yearly per security trading archive | per security traded volume, summed to a market wide daily series |
| `emdat_sri_lanka_disasters.xlsx` | EM-DAT, the international disaster database maintained by CRED | the disaster event list, dates, affected population, deaths, damage |
| `emdat_sri_lanka_disasters_superseded_2026_02_09.xlsx` | the earlier EM-DAT export | kept for provenance only, not read by any code |

The yearly workbooks are discovered by pattern, not by an enumerated list, so adding a
future year means dropping the file in and re running stage 01. The pattern is defined in
`src/data/cse_market_data.py`.

The superseded EM-DAT export is retained deliberately. The upgrade from the February export
to the September one changed the event count, and the audit cites both. Deleting it would
break that record.

## 2. Local reference inputs

These live in `data/external/` and are not read by the modelling pipeline. They support the
thesis text.

| File | Source | Used for |
|---|---|---|
| `cse_market_capitalization.csv` | CSE market capitalisation publication | context for the market size discussion |
| `adb_asia_sme_monitor_sri_lanka.xlsx` | Asian Development Bank Asia SME Monitor | context for the economic background chapter |

## 3. Live external sources

These are fetched over the network by stage 01 and cached in `artifacts/external/`. Each
one was pre declared before it was scored; the declaration and the reasoning are in
`docs/audit.md` Part 3.

| Source | What is retrieved | Why |
|---|---|---|
| World Bank, via `wbgapi` | annual GDP growth and consumer price inflation | macroeconomic controls, lagged so that only figures published before the event are used |
| Yahoo Finance, via `yfinance` | S&P 500 daily closes | a global market control, so a worldwide move is not attributed to a local disaster |
| NASA POWER | daily precipitation and maximum wind at seven district points | instrument measured hazard intensity, which unlike damage is available for every event |
| DesInventar | Sri Lankan district level loss records | an independent local severity measure, matched to fifty two of the seventy four events |
| FRED | the Sri Lankan rupee to dollar exchange rate | exchange rate level and volatility as a market state control |
| Wikidata | Sri Lankan national election dates | a confounder control, since an election near an event moves the market for its own reasons |

Because these are unpinned and can be revised, every cached download is written with a
`.provenance.json` sidecar recording the retrieval time and the library versions. Re running
stage 01 can therefore change downstream numbers. Do it only when the raw data genuinely
changes, and record the change in `docs/audit.md`.

## 4. Inclusion criteria

Not every disaster in the EM-DAT export enters the study.

1. The event must have at least one thousand people recorded as affected. This threshold
   was pre registered and is frozen. It was briefly explored at seven hundred and at five
   hundred during development, both of which admitted zero additional matched events, and
   it was reverted. It is never changed on the basis of model performance.
2. Epidemics and biological disasters are excluded. The study is about physical hazards.
3. The event must align to a tradable session, and the index series must extend far enough
   past it to measure the targets.
4. Disaster types with fewer than three events are pooled into an other category, because a
   one hot indicator with a single positive case is a memorisation key rather than a
   feature.

Seventy six raw records satisfy the threshold, and seventy four survive alignment to a
tradable session.

## 5. Synthetic rows

Time aware oversampling is applied inside training folds only, capped at a pre registered
twenty five per cent synthetic share, and drawn only from events within a declared time
window of each other. `src/training/time_aware_smogn.py` implements it and
`tests/test_time_aware_smogn.py` asserts each of those constraints.

No synthetic row ever appears in a validation or test fold, and none is used in the
recovery survival models at all, because interpolation can produce a duration but not a
valid event indicator.

## 6. Missing data

Sixteen feature columns carry real missingness, chiefly damage assessments that EM-DAT
never recorded. They are median imputed from training rows only, one fold at a time, and
each carries its own availability indicator so a model can learn that the value was absent.
Damage is a genuine missing value, never a zero fill, because zero damage and unrecorded
damage are different statements.

Real damage figures exist for eighteen of the seventy four events. That sparsity is the
reason the hazard and DesInventar blocks were added, and the block ablation in
`docs/results.md` reports how much they actually bought.

## 7. Terms of use

Each third party source carries its own licence and citation requirements. EM-DAT requires
citation of CRED and the Université catholique de Louvain. NASA POWER, the World Bank,
FRED, DesInventar and Wikidata each publish their own terms. The exchange archive is the
property of the Colombo Stock Exchange.

The `LICENSE` file in this repository covers only the original work of the author. It
grants no rights over any third party data and does not override any source's own terms.
Anyone wishing to use this repository must request permission first, as that file explains.
