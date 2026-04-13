"""
fetch_patents.py
----------------
Fetches patent filings for watchlist companies using the EPO Open Patent
Services (OPS) API v3.2 — free, covers US, EP, PCT and 50+ jurisdictions.

Two search passes per company:
  1. Assignee name search — finds patents filed under the company name
  2. Applicant name search — finds patent applications (pre-grant)

For each new patent, calls Claude to:
  - Classify claim type
  - Summarise novelty and competitive relevance

Writes structured JSON to data/patents.json.

Requires: EPO_OPS_KEY, EPO_OPS_SECRET, ANTHROPIC_API_KEY env vars
API docs: https://developers.epo.org/ops-v3-2/apis
"""

import base64
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

import anthropic
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_PATH    = Path(__file__).parent.parent / "data" / "patents.json"
WATCHLIST_PATH = Path(__file__).parent.parent / "watchlist.json"

LOOKBACK_YEARS = 5
MODEL          = "claude-haiku-4-5-20251001"

EPO_OPS_KEY    = os.environ.get("EPO_OPS_KEY", "")
EPO_OPS_SECRET = os.environ.get("EPO_OPS_SECRET", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

EPO_AUTH_URL   = "https://ops.epo.org/3.2/auth/accesstoken"
EPO_SEARCH_URL = "https://ops.epo.org/3.2/rest-services/published-data/search"

# CQL date filter — 5 years back
YEAR_FROM = (datetime.now(timezone.utc) - timedelta(days=365 * LOOKBACK_YEARS)).year

PATENT_ANALYSIS_PROMPT = """\
You are a biotech patent analyst specialising in cell and gene therapy.

Given the following patent, provide a structured analysis.
Return ONLY a valid JSON object with these exact keys — no preamble, no markdown:

  "claim_type": one of: "Composition of matter" | "Method of treatment" | "Method (process)" | "Composition + Method" | "Other" | "Unknown"
  "novelty_summary": 2-3 sentences on what is novel and its competitive significance for in vivo CAR-T (or null if insufficient information)
  "relevant": true or false — is this relevant to CAR-T, gene therapy, T cell engineering, or related delivery technology?

Title: {title}
Abstract: {abstract}
Assignee: {assignee}
"""

VALID_CLAIM_TYPES = {
    "Composition of matter", "Method of treatment", "Method (process)",
    "Composition + Method", "Other", "Unknown",
}

# EPO OPS XML namespaces
NS = {
    "ops":  "http://ops.epo.org",
    "epo":  "http://www.epo.org/exchange",
    "exc":  "http://www.epo.org/exchange",
}


# ---------------------------------------------------------------------------
# EPO OPS authentication
# ---------------------------------------------------------------------------

_access_token: str = ""
_token_expiry: float = 0.0


def get_access_token() -> str:
    global _access_token, _token_expiry
    if _access_token and time.time() < _token_expiry - 60:
        return _access_token

    credentials = base64.b64encode(f"{EPO_OPS_KEY}:{EPO_OPS_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {credentials}",
        "Content-Type":  "application/x-www-form-urlencoded",
    }
    resp = requests.post(
        EPO_AUTH_URL,
        data="grant_type=client_credentials",
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    _access_token = data["access_token"]
    _token_expiry = time.time() + int(data.get("expires_in", 1200))
    logging.info("  EPO OPS token acquired")
    return _access_token


# ---------------------------------------------------------------------------
# EPO OPS search
# ---------------------------------------------------------------------------

def search_epo(cql_query: str, start: int = 1, count: int = 25) -> dict:
    """Run a CQL query against EPO OPS published-data search."""
    token = get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/xml",
        "X-OPS-Range":   f"{start}-{start + count - 1}",
    }
    params = {"q": cql_query, "Range": f"{start}-{start + count - 1}"}
    resp = requests.get(EPO_SEARCH_URL, params=params, headers=headers, timeout=30)

    # Handle quota / throttle
    if resp.status_code == 429:
        logging.warning("  EPO OPS rate limit hit — sleeping 10s")
        time.sleep(10)
        return search_epo(cql_query, start, count)

    if resp.status_code == 404:
        return {}  # No results

    resp.raise_for_status()
    return {"xml": resp.text, "total": int(resp.headers.get("X-OPS-Range-total", 0))}


def fetch_all_for_company(company: str) -> list[dict]:
    """Fetch all patents for a company via assignee + applicant CQL queries."""
    results = []
    seen_ids = set()

    # EPO CQL uses pa= for applicant name, pd= for publication date
    # Date format must be YYYYMMDD or just YYYY
    company_base = company.split()[0]  # first word for broader matching
    queries = [
        f'pa="{company}" AND pd>={YEAR_FROM}0101',
        f'pa="{company_base}" AND pd>={YEAR_FROM}0101',
    ]

    for cql in queries:
        try:
            start = 1
            while True:
                data = search_epo(cql, start=start, count=25)
                if not data or "xml" not in data:
                    break

                parsed = parse_epo_xml(data["xml"], company)
                for p in parsed:
                    pid = p.get("id")
                    if pid and pid not in seen_ids:
                        seen_ids.add(pid)
                        results.append(p)

                total = data.get("total", 0)
                if start + 25 > total or start > 100:  # cap at 100 per query
                    break
                start += 25
                time.sleep(0.5)

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                pass  # no results for this query
            else:
                logging.warning(f"  EPO search failed for '{company}': {e}")
            break
        except Exception as e:
            logging.warning(f"  EPO search failed for '{company}': {e}")
            break

    return results


def parse_epo_xml(xml_text: str, watchlist_company: str) -> list[dict]:
    """Parse EPO OPS search result XML into structured dicts."""
    results = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logging.warning(f"  XML parse error: {e}")
        return results

    # Find all exchange-documents
    for doc in root.iter("{http://www.epo.org/exchange}exchange-document"):
        try:
            country    = doc.get("country", "")
            doc_number = doc.get("doc-number", "")
            kind       = doc.get("kind", "")
            pub_date   = doc.get("date", "")

            patent_id = f"{country}{doc_number}.{kind}"

            # Title
            title = ""
            for t in doc.iter("{http://www.epo.org/exchange}invention-title"):
                if t.get("{http://www.w3.org/XML/1998/namespace}lang", "").lower() in ("en", ""):
                    title = (t.text or "").strip()
                    if title:
                        break

            # Abstract
            abstract = ""
            for ab in doc.iter("{http://www.epo.org/exchange}abstract"):
                if ab.get("{http://www.w3.org/XML/1998/namespace}lang", "").lower() in ("en", ""):
                    texts = [p.text or "" for p in ab.iter("{http://www.epo.org/exchange}p")]
                    abstract = " ".join(texts).strip()[:1000]
                    if abstract:
                        break

            # Applicants/assignees
            assignees = []
            for party in doc.iter("{http://www.epo.org/exchange}applicant"):
                name_el = party.find(".//{http://www.epo.org/exchange}name")
                if name_el is not None and name_el.text:
                    assignees.append(name_el.text.strip())

            assignee_str = "; ".join(assignees[:3]) or watchlist_company

            # Filing date
            filing_date = ""
            for fd in doc.iter("{http://www.epo.org/exchange}filing-date"):
                filing_date = (fd.text or "").strip()
                if filing_date and len(filing_date) == 8:
                    filing_date = f"{filing_date[:4]}-{filing_date[4:6]}-{filing_date[6:]}"
                break

            if not title:
                continue

            url = ""
            if country and doc_number:
                url = f"https://patents.google.com/patent/{country}{doc_number}{kind}"

            results.append({
                "id":               patent_id,
                "title":            title,
                "abstract":         abstract,
                "assignee":         assignee_str,
                "watchlist_company": watchlist_company,
                "filing_date":      filing_date or pub_date[:4] if pub_date else "",
                "patent_number":    f"{country}{doc_number}{kind}",
                "application_number": doc_number,
                "source":           "EPO OPS",
                "url":              url,
                "data_type":        "patent",
                "claim_type":       None,
                "novelty_summary":  None,
                "relevant":         None,
            })

        except Exception as e:
            logging.warning(f"  Error parsing patent doc: {e}")
            continue

    return results


# ---------------------------------------------------------------------------
# Claude analysis
# ---------------------------------------------------------------------------

def analyse_patent(patent: dict) -> dict:
    if not ANTHROPIC_API_KEY or not patent.get("title"):
        return patent
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = PATENT_ANALYSIS_PROMPT.format(
            title=patent.get("title", ""),
            abstract=(patent.get("abstract", "") or "")[:800],
            assignee=patent.get("assignee", ""),
        )
        msg = client.messages.create(
            model=MODEL, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        patent["claim_type"]      = result.get("claim_type") if result.get("claim_type") in VALID_CLAIM_TYPES else "Unknown"
        patent["novelty_summary"] = result.get("novelty_summary")
        patent["relevant"]        = result.get("relevant", True)
        logging.info(f"    [{patent['claim_type']}] {patent['title'][:60]}")
    except Exception as e:
        logging.warning(f"  Claude analysis failed: {e}")
    return patent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting patent fetch via EPO OPS")

    if not EPO_OPS_KEY or not EPO_OPS_SECRET:
        logging.error("EPO_OPS_KEY and EPO_OPS_SECRET not set — aborting")
        return

    watchlist = load_watchlist()
    logging.info(f"Processing {len(watchlist)} watchlist companies, from {YEAR_FROM}")

    # Load existing to preserve Claude analysis
    all_patents: dict[str, dict] = {}
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
        try:
            patents = fetch_all_for_company(company)
            logging.info(f"    Found {len(patents)} patents")

            for patent in patents:
                pid = patent["id"]
                if pid in all_patents:
                    continue
                if ANTHROPIC_API_KEY:
                    patent = analyse_patent(patent)
                    time.sleep(0.3)
                if patent.get("relevant") is not False:
                    all_patents[pid] = patent
                    new_count += 1

            time.sleep(1.0)  # respect EPO rate limits
        except Exception as e:
            logging.error(f"  Failed for {company}: {e}")

    logging.info(f"  Added {new_count} new patents")

    patents_list = sorted(
        all_patents.values(),
        key=lambda p: p.get("filing_date") or "",
        reverse=True,
    )

    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count":      len(patents_list),
        "patents":    patents_list,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logging.info(f"Wrote {len(patents_list)} patents to {OUTPUT_PATH}")


def load_watchlist() -> list[str]:
    try:
        return json.loads(WATCHLIST_PATH.read_text()).get("companies", [])
    except Exception as e:
        logging.warning(f"Could not load watchlist: {e}")
        return []


if __name__ == "__main__":
    run()
