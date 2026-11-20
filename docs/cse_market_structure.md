# CSE Market Structure Changes 2000–2025

**Task:** A5 — R2-revision6  
**Date compiled:** 2026-11-20  
**Purpose:** Document dated structural changes to the Colombo Stock Exchange (CSE) relevant to the study period, to inform interpretation of return/volume time-series discontinuities.

---

## 1. Trading Technology

| Date | Change |
|------|--------|
| 1997 (pre-study) | Automated Trading System (ATS) launched, replacing open-outcry floor trading |
| December 2014 | ATS upgraded alongside Central Depository System (CDS) replacement; new platform improved order-matching capacity and connectivity |
| June 2020 | SEC Sri Lanka mandated digitisation of share certificates; paperless settlement reinforced |

---

## 2. Trading Hours

| Date | Change |
|------|--------|
| May 2020 | Trading hours reduced to 09:30–12:30 (COVID-19 operational measure) |
| September 2022 | Trading hours extended to 09:30–13:30 |
| November 2022 | Further extension to 09:30–14:30 |
| April 2023 | Full-day session restored: 09:30–14:30 maintained; additional pre-open session added |

---

## 3. Settlement Cycle

| Date | Change |
|------|--------|
| c. 2007 | Settlement cycle standardised to T+3 |
| 10 June 2024 | Settlement shortened to T+2 (aligns with regional peers) |

---

## 4. Circuit Breakers and Market Halts

The CSE operates a three-tier circuit breaker on the ASPI:

| Trigger | Action |
|---------|--------|
| ≥5% intraday move | 30-minute trading halt |
| Additional ≥2.5% after resumption | Full trading halt for the day |
| ≥10% intraday move (single trigger) | Market closed for the day |

**Notable activations:**
- **11 May 2020:** Circuit breaker triggered on market reopening after COVID-19 closure; ASPI fell sharply on first post-lockdown session.
- **February 2022:** Multiple intraday halts during early phase of economic crisis (foreign exchange shortage, sovereign debt stress).
- **18–25 April 2022 (5 trading days):** Exchange suspended trading for five consecutive days during peak political and economic crisis (fuel shortages, protests, government default). Largest closure since the COVID period.

---

## 5. COVID-19 Closure

| Period | Detail |
|--------|--------|
| March 2020 – 10 May 2020 | Exchange closed entirely (~51 trading days); SEC ordered closure following national lockdown |
| 11 May 2020 | Trading resumed with reduced hours (09:30–12:30) and circuit breaker activated on reopening |

---

## 6. Index Changes

| Date | Change |
|------|--------|
| 26 June 2012 | S&P Sri Lanka 20 (S&P SL 20) launched as blue-chip large-cap index; joint venture between S&P Dow Jones Indices and CSE |
| 30 December 2012 | Milanka Price Index (MPI) discontinued; S&P SL 20 became primary large-cap benchmark |

**Study relevance:** The thesis uses ASPI (All Share Price Index) as the market index throughout. The MPI/S&P SL 20 transition does not affect the ASPI series.

---

## 7. Implications for the Study Period (2000–2022)

| Structural break | Study period impact |
|-----------------|---------------------|
| ATS upgrade (Dec 2014) | Potential mild microstructure change in intraday liquidity; unlikely to affect daily close-to-close returns at event-study scale |
| COVID closure (Mar–May 2020) | Introduces ~51 missing trading days; handled by gap in ASPI series. Post-closure circuit breaker on 11 May 2020 is itself a data point in the event window (COVID event). |
| Economic crisis halts (Apr 2022) | 5-day suspension affects any event window spanning April 2022; market model estimation windows may contain these gaps |
| Trading hours changes (2020–2023) | Affect daily volume denominator if comparing to pre-2020 norms; not adjusted in this study (volume is treated as a relative abnormal measure) |
| T+3 → T+2 (June 2024) | Post-study period; no impact |

---

## 8. Sources

- Colombo Stock Exchange official announcements (cse.lk)
- Securities and Exchange Commission of Sri Lanka circulars (sec.gov.lk)
- S&P Dow Jones Indices press release, 26 June 2012
- CSE Annual Reports 2012–2023
- Media reports: Daily FT (Lanka), Colombo Gazette (contemporaneous coverage of 2020 and 2022 closures)
