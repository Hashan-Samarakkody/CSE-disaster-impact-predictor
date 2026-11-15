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

### SHA-256 checksums

Verify the files have not changed since the study was run:

| File | SHA-256 |
|---|---|
| `cse_market_indices_daily.xls` | `1bda6b39812f22d7b4091b7a59fdde72ad7ffc16349bb7fe2cbd12aa75e5772f` |
| `cse_securities_2000.xls` | `2b8b1e629c09443586dbf878378876154e67795756f17fddd78fb95f4524ba80` |
| `cse_securities_2001.xls` | `990a298750b00376e7187a8821a0a6c390312e89b0453976fd7bc5fc66fba7b6` |
| `cse_securities_2002.xls` | `d6d21ec083db929308a2522a0e4a65e100b5cf76cc193da45676712ce459708a` |
| `cse_securities_2003.xls` | `d29d201d21737e09a993add644b5c60bf71469c3b1b34e18df9b5ba3fdbce8bd` |
| `cse_securities_2004.xls` | `a3d60fdfce5a18ff6f932398f50407f53db85d8cfcb8935f261a2401ef7fd19d` |
| `cse_securities_2005.xls` | `8154def55bcf06372b8d541b0b5aa619ffecdac8e8054164eb7603ebe3b7822e` |
| `cse_securities_2006.xls` | `e8beae81d2c11cf6ee4351222543a8408d11888b6a5b64c749703e1baaeee5a9` |
| `cse_securities_2007.xls` | `5d7a40df55198917a8be26ec6795fa327e99ba547b61545e5f21a5065fbecbd2` |
| `cse_securities_2008.xls` | `f2d1a73294775959dbcfcd8b5e0a49f42900cb9fab1fb400e976a7b000540841` |
| `cse_securities_2009.xls` | `e026373c7549854e9bdb5c0acf3071d2c8de74e8019f00e326253535b96af89d` |
| `cse_securities_2010.xls` | `a77958d9db58b3bd83297f17235ea760ea5089082a326024f9931e86844ff136` |
| `cse_securities_2011.xls` | `22316b6e1808c279a17553cad66e85cbe19d56823d923f336f195b6cf3f4abfe` |
| `cse_securities_2012.xls` | `853b6d03a6eafa1fbd5a558fd62cfbf1db36735e9d4fce1fbe9d935a26f392f3` |
| `cse_securities_2013.xls` | `03807192f846c4fb166df8e063b02190a27547c1f2220f388a0c761ce3434019` |
| `cse_securities_2014.xls` | `472f3bf6edbfecc41a288490874cd0cfee6b9bdcb4b4890af79b21af8a4a4d4c` |
| `cse_securities_2015.xlsx` | `8244436de45be5b65890af97cbccae5990fc640705b8d03902871c7d79954a7c` |
| `cse_securities_2016.xls` | `f2024fbd6f95ebfaf995b9b5ae1dde578ad2ea45bc033c5127bbdfcde49a064d` |
| `cse_securities_2017.xls` | `7c27d189a9db76a1b8f2ccfd104f184131de91faff03c86a516bd58272b96139` |
| `cse_securities_2018.xls` | `fc26d74d616e0c9a8f9c63557028f9718cf567fa6e90ea08061a24d6a108beec` |
| `cse_securities_2019.xls` | `f7fa80f177ed6d1399f0fb42f18c67c25c154a56a39576f8fafdbebe0ed7e90b` |
| `cse_securities_2020.xls` | `4e5be148cc087c9ad0c644a632c44cfad3dad6bb28440de4b6612847261881be` |
| `cse_securities_2021.xls` | `27b4fe74bc9ab910057bcbbd9bc1f2a5103c73f6dcccea4b39d70180379ce4a0` |
| `cse_securities_2022.xls` | `6ee2c45ee8f97c37895fee0ca185c6d000f854b310fedd2445b9255d99c6944b` |
| `cse_securities_2023.xls` | `511d82ea61543a3953024573a59c923d235cb56b3929ea72fe60f55f7389087e` |
| `emdat_sri_lanka_disasters.xlsx` | `5dd49acfb9b3565b703b872c41fe5adf85c2eb7f334f5a39ca36de8f357e1d61` |
| `emdat_sri_lanka_disasters_superseded_2026_02_09.xlsx` | `5e7024e9d210555d01635e1d94a316a8ae272574d6d0714e538cc0db87ff72a2` |

Verify with: `python -c "import hashlib; print(hashlib.sha256(open('data/raw/FILENAME','rb').read()).hexdigest())"`

### Which EM-DAT export the study reads

`emdat_sri_lanka_disasters.xlsx` is the live file and the only one any code opens. It is
the CRED export covering 1986 to 2025, downloaded on 2026-09-12, and it carries **110
records** on its `EM-DAT Data` sheet. Every count in the sample flow accounting starts
from those 110.

`emdat_sri_lanka_disasters_superseded_2026_02_09.xlsx` is the earlier export, downloaded
on 2026-02-09, covering 2000 to 2025 with 86 records. It is read by nothing. It is
retained deliberately because the upgrade to the September export changed the event count,
`docs/audit.md` cites numbers produced under both, and deleting the earlier file would
leave those citations unverifiable. The 24 extra records in the live export are all
pre 2000 and none of them can be modelled: the ASPI series begins on 2000-01-03, so they
have no pre-event close to baseline against. The sample flow table records that exclusion
explicitly rather than leaving it implicit.

The full accounting from 110 raw records to 74 modelled events is in
`artifacts/tables/sample_flow.parquet` and `docs/figures/eda_11_sample_flow.png`.

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

The `LICENSE` file covers only the original work of the author, the code, the notebooks
and the documentation, and since 2026-09-22 it does so under the MIT licence, so the
method and the workflow can be verified by anyone.

It grants no rights over any third party data and overrides no source's own terms. The two
directories that hold third party material are `data/`, which carries the raw EM-DAT export
and the Colombo Stock Exchange archive, and `artifacts/external/`, which carries the cached
downloads. Anyone redistributing or reusing anything in either must satisfy the originating
source directly, whatever the code licence permits.

## 8. Known limitations: historical data vintages and retrospective event selection

The macroeconomic variables used in this study (GDP growth, CPI inflation, lending rate)
were retrieved from the FRED and World Bank databases in 2026. These databases report revised
national accounts, which may differ from the values that were publicly available at the time
each disaster occurred. Although annual observations are post-dated to July 1 of the
following year to approximate publication lag (see Section 3.3), this convention does not
guarantee that the retrieved values match historical first releases. Historical vintage data
for Sri Lanka were not available to the investigator.

The event universe was constructed using finalized EM-DAT affected-population totals, which
are retrospectively revised as reports are consolidated (Centre for Research on the
Epidemiology of Disasters, 2024; see also
https://doc.emdat.be/docs/known-issues-and-limitations/specific-biases/). This creates a
look-ahead bias in event selection: an investor at the time of the disaster would not have
known whether the event would ultimately meet the 1,000-affected threshold used in this
study. The event universe is therefore retrospective and appropriate for research purposes,
but does not automatically describe an operational alert system.

These two limitations qualify the real-time interpretation of the macroeconomic features and
the generalisability of the event-selection criterion. They do not affect the ex-post
attribution analysis or the within-sample event-study findings.
