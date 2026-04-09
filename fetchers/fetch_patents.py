"""
fetch_patents.py
----------------
Fetches patent filings for watchlist companies using:
  1. USPTO EFTS full-text search (assignee queries, no key required)
  2. USPTO Assignment Search API (assignee-based, no key required)

For each patent found, passes title + abstract to Claude for:
  - Claim type classification (composition of matter / method / etc.)
  - Novelty summary (what's novel and competitively relevant)

Writes structured JSON to data/patents.json.

Requires: ANTHROPIC_API_KEY environment variable
"""

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote

import anthropic
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_PATH    = Path(__file__).parent.parent / "data" / "patents.json"
WATCHLIST_PATH = Path(__file__).parent.parent / "watchlist.json"

LOOKBACK_YEARS = 5
MODEL          = "claude-haiku-4-5-20251001"

# USPTO EFTS endpoint (powers ppubs.uspto.gov)
EFTS_BASE = "https://efts.uspto.gov/LATEST/search-index"

# USPTO Assignment Search
ASSIGNMENT_BASE = "https://developer.uspto.gov/ds-api/assignments/patent/query"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

PATENT_ANALYSIS_PROMPT = """\
You are a biotech patent analyst specialising in cell and gene therapy.

Given the following patent filing, provide a structured analysis.
Return ONLY a valid JSON object with these exact keys — no preamble, no markdown:

  "claim_type": one of: "Composition of matter" | "Method of treatment" | "Method (process)" | "Composition + Method" | "Other"
  "novelty_summary": 2-3 sentences explaining what is novel about this patent and its competitive significance for in vivo CAR-T
  "relevant": true or false — is this patent relevant to in vivo CAR-T (delivery, construct design, T cell targeting, etc.)?

Title: {title}
Abstract: {abstract}
Assignee: {assignee}
"""

VALID_CLAIM_TYPES = {
    "Composition of matter",
    "Method of treatment",
    "Method (process)",
    "Composition + Method",
    "Other",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_watchlist() -> list[str]:
    try:
        return json.loads(WATCHLIST_PATH.read_text()).get("companies", [])
    except Exception as e:
        logging.warning(f"Could not load watchlist: {e}")
        return []


def date_cutoff() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=365 * LOOKBACK_YEARS)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# USPTO EFTS search
# ---------------------------------------------------------------------------

def search_efts(assignee: str, cutoff: str) -> list[dict]:
    """Search USPTO full-text for patents by assignee."""
    try:
        params = {
            "q":      f'"{assignee}"',
            "dateRangeField": "datePublished",
            "startdt": cutoff,
            "enddt":  datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "hits.hits.total": "true",
            "hits.hits._source": "patentTitle,assignees,applicationNumber,filingDate,patentNumber,abstractText,documentId",
        }
        headers = {"User-Agent": "Mozilla/5.0 (compatible; cart-intelligence-bot/1.0)"}
        resp = requests.get(EFTS_BASE, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        return hits
    except Exception as e:
        logging.warning(f"  EFTS search failed for '{assignee}': {e}")
        return []


def parse_efts_hit(hit: dict, assignee: str) -> dict:
    src = hit.get("_source", {})
    title    = src.get("patentTitle", "")
    abstract = src.get("abstractText", "") or ""
    app_num  = src.get("applicationNumber", "")
    pat_num  = src.get("patentNumber", "") or src.get("documentId", "")
    filing   = src.get("filingDate", "")
    assignees = src.get("assignees", [])
    if isinstance(assignees, list):
        assignee_str = "; ".join(
            a.get("assigneeName", "") if isinstance(a, dict) else str(a)
            for a in assignees[:3]
        )
    else:
        assignee_str = assignee

    url = f"https://patents.google.com/patent/{pat_num}" if pat_num else ""

    return {
        "id":           pat_num or app_num,
        "application_number": app_num,
        "patent_number": pat_num,
        "title":         title,
        "abstract":      abstract[:1000] if abstract else "",
        "assignee":      assignee_str or assignee,
        "filing_date":   filing,
        "source":        "USPTO",
        "url":           url,
        "claim_type":    None,
        "novelty_summary": None,
        "relevant":      None,
        "watchlist_company": assignee,
    }


# ---------------------------------------------------------------------------
# Claude analysis
# ---------------------------------------------------------------------------

def analyse_patent(patent: dict) -> dict:
    """Use Claude to classify claim type and summarise novelty."""
    if not ANTHROPIC_API_KEY:
        return patent
    if not patent.get("title"):
        return patent

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = PATENT_ANALYSIS_PROMPT.format(
            title=patent.get("title", ""),
            abstract=(patent.get("abstract", "") or "")[:800],
            assignee=patent.get("assignee", ""),
        )
        msg = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        patent["claim_type"]       = result.get("claim_type") if result.get("claim_type") in VALID_CLAIM_TYPES else "Other"
        patent["novelty_summary"]  = result.get("novelty_summary")
        patent["relevant"]         = result.get("relevant", True)
    except Exception as e:
        logging.warning(f"  Claude analysis failed for '{patent.get('title', '')[:40]}': {e}")

    return patent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting patent fetch")

    watchlist = load_watchlist()
    cutoff    = date_cutoff()
    logging.info(f"Searching {len(watchlist)} companies, cutoff: {cutoff}")

    all_patents: dict[str, dict] = {}

    # Load existing to preserve Claude analysis
    if OUTPUT_PATH.exists():
        try:
            existing = json.loads(OUTPUT_PATH.read_text())
            for p in existing.get("patents", []):
                pid = p.get("id")
                if pid:
                    all_patents[pid] = p
            logging.info(f"  Loaded {len(all_patents)} existing patents")
        except Exception as e:
            logging.warning(f"Could not load existing patents: {e}")

    new_count = 0
    for company in watchlist:
        logging.info(f"  Searching: {company}")
        hits = search_efts(company, cutoff)
        logging.info(f"    Found {len(hits)} results")

        for hit in hits:
            parsed = parse_efts_hit(hit, company)
            pid    = parsed["id"]
            if not pid:
                continue
            if pid in all_patents:
                # Update watchlist_company if not set
                if not all_patents[pid].get("watchlist_company"):
                    all_patents[pid]["watchlist_company"] = company
                continue

            # Run Claude analysis on new patents
            if ANTHROPIC_API_KEY:
                parsed = analyse_patent(parsed)
                time.sleep(0.3)

            all_patents[pid] = parsed
            new_count += 1

        time.sleep(0.5)

    logging.info(f"  Added {new_count} new patents")

    # Filter to only relevant patents (where Claude has assessed)
    patents_list = sorted(
        [p for p in all_patents.values() if p.get("relevant") is not False],
        key=lambda p: p.get("filing_date") or "",
        reverse=True,
    )

    if not patents_list and not all_patents:
        logging.warning("No patents found — check EFTS connectivity")

    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count":      len(patents_list),
        "patents":    patents_list,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logging.info(f"Wrote {len(patents_list)} patents to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
