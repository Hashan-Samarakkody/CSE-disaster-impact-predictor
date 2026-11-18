# Reference Verification Notes

**Generated:** 2026-11-18  
**Task:** A3 — R2 examiner revision  
**Method:** DOIs extracted from manuscript R6_SET_clean.pdf; queried via Crossref API (scripts/verify_references.py)  
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

Approximately 30 references in the manuscript have no DOI listed (books, some conference papers, some older journal articles). These are not included in docs/references.txt (which is DOI-only input for the verification script). Their full citations appear in docs/references_apa.txt with no DOI hyperlink where none was found.

For completeness, known DOIs for some of these (not listed in manuscript) that could be added by the author:
- Kolari & Pynnonen (2010): 10.1093/rfs/hhq072 (Review of Financial Studies)
- Brown & Warner (1985): 10.1016/0304-405X(85)90042-X (Journal of Financial Economics)
- Corrado (1989): 10.1016/0304-405X(89)90075-X (Journal of Financial Economics)
- Boehmer, Musumeci & Poulsen (1991): 10.1016/0304-405X(91)90032-F (Journal of Financial Economics)
- Fama et al. (1969): 10.2307/2525569 (International Economic Review)
- Holm (1979): 10.2307/4615733 (Scandinavian Journal of Statistics)

These are from memory and NOT verified via Crossref in this run. Do not add them without independent verification.
