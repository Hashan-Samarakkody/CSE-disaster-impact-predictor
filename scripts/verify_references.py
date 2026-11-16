"""Query Crossref for every DOI in the reference list; write docs/reference_check.csv.

Usage:
    python scripts/verify_references.py                        # reads docs/references.txt
    python scripts/verify_references.py path/to/refs.txt       # explicit file
    python scripts/verify_references.py --doi 10.1234/abc      # single DOI

Input format (docs/references.txt or supplied file):
    One DOI per line; lines not containing a DOI are skipped.
    DOIs may appear bare (10.xxxx/...) or as URLs (https://doi.org/10.xxxx/...).

Output: docs/reference_check.csv with columns:
    doi, resolved, title, authors, journal, year, publisher, http_status, note

Requires: requests (already in requirements.txt)
"""
from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests not installed: pip install requests")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "references.txt"
OUTPUT = ROOT / "docs" / "reference_check.csv"

DOI_RE = re.compile(r"10\.\d{4,9}/\S+")
CROSSREF_URL = "https://api.crossref.org/works/{doi}"
HEADERS = {"User-Agent": "CSE-disaster-impact-predictor/1.0 (academic thesis verification)"}
RATE_DELAY = 0.5  # seconds between requests (Crossref polite pool)


def extract_dois(text: str) -> list[str]:
    return DOI_RE.findall(text)


def clean_doi(doi: str) -> str:
    doi = doi.rstrip(".,;)")
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    if doi.startswith("http://doi.org/"):
        doi = doi[len("http://doi.org/"):]
    return doi


def query_crossref(doi: str) -> dict:
    url = CROSSREF_URL.format(doi=doi)
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            msg = r.json().get("message", {})
            title = " ".join(msg.get("title", []))
            authors = "; ".join(
                f"{a.get('family', '')}, {a.get('given', '')}"
                for a in msg.get("author", [])
            )
            journal_list = msg.get("container-title", [])
            journal = journal_list[0] if journal_list else ""
            year = ""
            dp = msg.get("published", {}).get("date-parts", [[]])
            if dp and dp[0]:
                year = str(dp[0][0])
            publisher = msg.get("publisher", "")
            return {
                "doi": doi,
                "resolved": "yes",
                "title": title,
                "authors": authors,
                "journal": journal,
                "year": year,
                "publisher": publisher,
                "http_status": r.status_code,
                "note": "",
            }
        else:
            return {
                "doi": doi, "resolved": "no", "title": "", "authors": "",
                "journal": "", "year": "", "publisher": "",
                "http_status": r.status_code,
                "note": f"HTTP {r.status_code}",
            }
    except requests.exceptions.ConnectionError as e:
        return {
            "doi": doi, "resolved": "error", "title": "", "authors": "",
            "journal": "", "year": "", "publisher": "", "http_status": "",
            "note": f"ConnectionError: {e}",
        }
    except Exception as e:
        return {
            "doi": doi, "resolved": "error", "title": "", "authors": "",
            "journal": "", "year": "", "publisher": "", "http_status": "",
            "note": str(e),
        }


def run(dois: list[str]) -> None:
    unique = list(dict.fromkeys(clean_doi(d) for d in dois))
    print(f"Checking {len(unique)} unique DOIs against Crossref …")
    rows = []
    for i, doi in enumerate(unique, 1):
        print(f"  [{i}/{len(unique)}] {doi}", end=" ", flush=True)
        result = query_crossref(doi)
        status = "OK" if result["resolved"] == "yes" else result.get("note", "?")
        print(status)
        rows.append(result)
        time.sleep(RATE_DELAY)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["doi", "resolved", "title", "authors", "journal", "year",
              "publisher", "http_status", "note"]
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(1 for r in rows if r["resolved"] == "yes")
    n_fail = len(rows) - n_ok
    print(f"\nDone. {n_ok} resolved, {n_fail} failed/unresolved.")
    print(f"Written: {OUTPUT}")


def main() -> None:
    args = sys.argv[1:]

    if "--doi" in args:
        idx = args.index("--doi")
        doi = args[idx + 1]
        run([doi])
        return

    input_file = Path(args[0]) if args else DEFAULT_INPUT
    if not input_file.exists():
        sys.exit(
            f"Input file not found: {input_file}\n"
            "Create docs/references.txt with one DOI per line, or pass a path as argument."
        )

    text = input_file.read_text(encoding="utf-8")
    dois = extract_dois(text)
    if not dois:
        sys.exit(f"No DOIs found in {input_file}  (pattern: 10.XXXX/...)")
    run(dois)


if __name__ == "__main__":
    main()
