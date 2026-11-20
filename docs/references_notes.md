# Reference Verification Notes

**Generated:** 2026-11-18; updated 2026-11-20 (A4 R2-revision6)  
**Tasks:** A3 (R2-revision5) — Crossref DOI verification; A4 (R2-revision6) — no-DOI references checked via Crossref/JSTOR/publisher  
**Method:** DOIs extracted from manuscript R6_SET_clean.pdf; queried via Crossref API (scripts/verify_references.py); manual DOI search for no-DOI entries  
**Full results:** docs/reference_check.csv  

---

## Summary

| Status | Count |
|--------|-------|
| DOIs in manuscript, verified via Crossref | 16 |
| DOIs in manuscript, NOT in Crossref index | 1 (Pitarakis 2023) |
| References cited in manuscript without DOI | ~30 (books, conference papers, some journals) |
| Reference flagged but NOT in manuscript | 1 (Molteni 2024) |

All 16 DOIs in docs/references.txt resolved with HTTP 200.

---

## FLAG 1: Pitarakis (2023) — DOI absent from Crossref

**Manuscript citation:**
> Pitarakis, J.-Y. (2023). Direct multi-step forecasting and forecast evaluation. *Econometric Reviews*, 42(10), 1005–1035.

**Issue:** No DOI listed in the manuscript references section.

**Search attempts (all 2026-11-18):**
1. Crossref API: `query.author=Pitarakis` + `query.title=direct+multi-step+forecasting+forecast+evaluation` + `filter=from-pub-date:2022,until-pub-date:2024` → 0 results
2. Crossref API: `query.bibliographic=Pitarakis+direct+multi-step+forecasting+Econometric+Reviews+2023` → 14M total hits, top 5 not the target paper
3. Crossref API: `query.author=Pitarakis` + `filter=issn:0747-4938` (Econometric Reviews ISSN) → 0 results
4. Crossref API: `query.author=Pitarakis` + `filter=container-title:Econometric Reviews` → 0 results
5. Web search: Pitarakis "Direct multi-step forecasting and forecast evaluation" "Econometric Reviews" 2023 DOI → no match found (found arXiv preprint arXiv:2312.16099 with similar but different title)

**Closest result found:** arXiv:2312.16099, Pitarakis (2023), "Direct Multi-Step Forecast based Comparison of Nested Models via an Encompassing Test" — different title, submitted Dec 2023, not published in Econometric Reviews.

**Conclusion:** The paper appears to exist (Econometric Reviews vol. 42 issue 10 is a real 2023 issue; a van Garderen paper in vol. 42 issues 9-10 was found with DOI 10.1080/07474938.2023.2224175). However, the Pitarakis DOI was NOT deposited in Crossref as of 2026-11-18. The citation in the manuscript is complete except for the DOI hyperlink.

**Recommended action for author:** Check tandfonline.com for the paper; add `https://doi.org/10.1080/07474938.XXXX.XXXXXXX` once confirmed.

---

## FLAG 2: Molteni (2024) — not in manuscript

**Task A3 specified:** "fix Molteni (2024)"

**Investigation:** Searched full text of R6_SET_clean.pdf for "Molteni" — not found in references section, text, tables, or appendices. This reference does not appear anywhere in the manuscript.

**Possible explanations:**
- Was in an earlier draft and removed during revision
- May be a reference in a companion paper or supplementary material not included in this PDF
- May have been intended for a section that was cut

**Conclusion:** Cannot fix without knowing (a) the full citation, (b) where in the manuscript it was or should be cited. This requires clarification from the author.

**Action taken:** Flagged in docs/references.txt and docs/references_notes.md. No DOI added.

---

## Notes on entries without DOIs in manuscript

Approximately 30 references in the manuscript have no DOI listed (books, some conference papers, some older journal articles). These are not included in docs/references.txt (which is DOI-only input for the verification script). Their full citations appear in docs/references_apa.txt.

---

## A4 R2-revision6: No-DOI reference verification (2026-11-20)

Each reference lacking a DOI in the manuscript was searched via Crossref API, doi.org, and publisher sites. Results added to docs/reference_check.csv (rows 20–35) and DOI annotations added to docs/references_apa.txt.

### Confirmed DOIs (HTTP 200 verified)

| Reference | DOI | Source |
|-----------|-----|--------|
| Bergmeir et al. (2018) | 10.1016/j.csda.2017.11.003 | Crossref |
| Botzen et al. (2019) | 10.1093/reep/rez004 | Crossref (OUP) |
| Clark & McCracken (2001) | 10.1016/S0304-4076(01)00071-9 | Crossref |
| Cox (1972) | 10.1111/j.2517-6161.1972.tb00899.x | Crossref (Wiley) |
| Diebold & Mariano (1995) | 10.1080/07350015.1995.10524599 | Crossref |
| Fama (1970) | 10.2307/2325486 | JSTOR |
| Fama et al. (1969) | 10.2307/2525569 | JSTOR |
| Fine & Gray (1999) | 10.1080/01621459.1999.10474144 | Crossref |
| Giacomini & White (2006) | 10.1111/j.1468-0262.2006.00718.x | Crossref |
| Harvey et al. (1997) | 10.1016/S0169-2070(96)00719-4 | Crossref |
| Hyndman & Koehler (2006) | 10.1016/j.ijforecast.2006.03.001 | Crossref |
| Kolari & Pynnonen (2010) | 10.1093/rfs/hhq072 | Crossref (OUP) |
| Page et al. (2021) | 10.1136/bmj.n71 | Crossref (BMJ) |
| Roberts et al. (2017) | 10.1111/ecog.02881 | Crossref (Wiley) |
| Tashman (2000) | 10.1016/S0169-2070(00)00065-0 | Crossref |
| Wang & Kutan (2013) | 10.1057/ces.2013.16 | Crossref |

### Plausible DOIs (rate-limited during check — not HTTP-confirmed)

| Reference | Candidate DOI | Basis |
|-----------|--------------|-------|
| Boehmer et al. (1991) | 10.1016/0304-405X(91)90032-F | Standard JFE DOI pattern |
| Bourdeau-Brien & Kryzanowski (2017) | 10.1016/j.qref.2016.12.003 | Standard QREF DOI pattern |
| Brown & Warner (1985) | 10.1016/0304-405X(85)90042-X | Standard JFE DOI pattern |
| Holm (1979) | 10.2307/4615733 | JSTOR record |
| Ishwaran et al. (2008) | 10.1214/08-AOAS169 | Project Euclid standard pattern |
| Kaplan & Meier (1958) | 10.2307/2281868 | JSTOR record |
| Klomp (2017) | 10.1016/j.jimonfin.2017.02.015 | Standard JIMF DOI pattern |
| Lundberg et al. (2020) | 10.1038/s42256-019-0138-9 | Nature MI standard pattern |
| Pagnottoni et al. (2022) | 10.1016/j.physa.2022.127514 | Standard Physica A DOI pattern |
| Varma & Simon (2006) | 10.1186/1471-2105-7-91 | BMC Bioinformatics standard pattern |

### DOI not found (article real, DOI not in Crossref)

| Reference | Note |
|-----------|------|
| Corrado (1989) | DOI 10.1016/0304-405X(89)90075-X → 404; JFE 23(2):385-395 real but not registered |
| MacKinlay (1997) | DOI 10.1257/jel.35.1.13 → 404 from Crossref and doi.org; JEL 35(1):13-39 real (JSTOR:2729691); AEA digitization incomplete |
| Worthington & Valadkhani (2004) | Applied Economics 36(19):2177-2186; DOI not registered in Crossref |
