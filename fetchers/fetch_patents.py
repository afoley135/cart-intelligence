"""
fetch_patents.py
----------------
Fetches patent filings for watchlist companies using the EPO Open Patent
Services (OPS) API v3.2.

Uses the /search/biblio constituent endpoint — returns full bibliographic
data (title, abstract, assignees, dates) in a single search request,
eliminating the two-step search+biblio pattern that was silently failing.

Three search strategies per company:
  1. pa="CompanyName*"                          — applicant wildcard
  2. pa="Full Company Name"                     — exact name fallback
  3. ta="CompanyName" AND (ic=A61 OR ic=C12N)   — title/abstract fallback

Rate limiting:
  EPO OPS free tier allows ~30 requests/minute.
  With 3 queries × 28 companies = ~84 requests, we sleep 2.5s between
  queries within a company and apply a proportional cooldown after each
  company (3s base + 0.5s per patent found, capped at 30s). This prevents
  large result sets (e.g. 39 results from "Addition Therapeutics") from
  saturating the per-minute bucket and causing downstream 403 cascades.

  403 handling is lifted out of the per-request retry loop. If a 403
  occurs, search_biblio raises ThrottleError; fetch_all_for_company catches
  it, sleeps 45s, then moves on rather than hammering the same endpoint.

For each new patent, calls Claude to classify claim type and summarise
novelty. Existing analysis is cached and not re-run.

Writes structured JSON to data/patents.json.

Requires: EPO_OPS_KEY, EPO_OPS_SECRET, ANTHROPIC_API_KEY env vars
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
# Custom exceptions
# ---------------------------------------------------------------------------

class ThrottleError(Exception):
    """Raised when EPO OPS returns an unrecoverable 403 after retry."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_PATH    = Path(__file__).parent.parent / "data" / "patents.json"
WATCHLIST_PATH = Path(__file__).parent.parent / "watchlist.json"

LOOKBACK_YEARS   = 5
INCREMENTAL_DAYS = 14    # daily runs: fetch patents published in last N days
                         # (buffer beyond 1 day to account for EPO indexing lag)
MODEL            = "claude-haiku-4-5-20251001"

EPO_OPS_KEY       = os.environ.get("EPO_OPS_KEY", "")
EPO_OPS_SECRET    = os.environ.get("EPO_OPS_SECRET", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FULL_SCAN         = os.environ.get("FULL_SCAN", "").lower() in ("1", "true", "yes")

EPO_AUTH_URL      = "https://ops.epo.org/3.2/auth/accesstoken"
# /search/biblio returns full bib data in one shot — no second fetch needed
EPO_SEARCH_BIBLIO = "https://ops.epo.org/3.2/rest-services/published-data/search/biblio"

# Date boundary for EPO CQL queries
if FULL_SCAN:
    DATE_FROM_STR = f"{(datetime.now(timezone.utc) - timedelta(days=365 * LOOKBACK_YEARS)).year}0101"
else:
    DATE_FROM_STR = (datetime.now(timezone.utc) - timedelta(days=INCREMENTAL_DAYS)).strftime("%Y%m%d")

# Sleep between individual queries — keeps us under EPO's ~30 req/min limit
QUERY_SLEEP        = 2.5   # seconds between queries within a company
COMPANY_SLEEP_BASE = 3.0   # minimum seconds between companies
COMPANY_SLEEP_PER  = 0.5   # additional seconds per patent found (proportional cooldown)
COMPANY_SLEEP_MAX  = 30.0  # cap so large result sets don't stall the run unnecessarily
THROTTLE_SLEEP     = 45.0  # seconds to sleep when a 403 is unrecoverable

PATENT_ANALYSIS_PROMPT = """\
You are a biotech patent analyst specialising in cell and gene therapy.

Given the following patent, provide a structured analysis.
Return ONLY a valid JSON object with these exact keys — no preamble, no markdown:

  "claim_type": one of: "Composition of matter" | "Method of treatment" | "Method (process)" | "Composition + Method" | "Other" | "Unknown"
  "novelty_summary": 2-3 sentences on what is novel and its competitive significance for in vivo CAR-T (or null if insufficient information)
  "relevant": true or false — is this relevant to CAR-T, gene therapy, T cell engineering, or related delivery technology?
  "title_en": if the Title below is not in English, provide an accurate English translation; if it is already English return null

Title: {title}
Abstract: {abstract}
Assignee: {assignee}
"""

VALID_CLAIM_TYPES = {
    "Composition of matter", "Method of treatment", "Method (process)",
    "Composition + Method", "Other", "Unknown",
}

# Generic words to strip when building search stems
GENERIC_WORDS = {
    "therapeutics", "biosciences", "biotech", "biotherapeutics",
    "medicines", "biopharma", "pharma", "biologics", "labs",
    "laboratory", "laboratories", "inc", "llc", "ltd", "gmbh",
}

# Stems that match large unrelated companies — always scope to biotech IPC classes.
# "Create" matches many unrelated filers (furniture, software, etc.)
# "Addition" matches many unrelated filers despite being in the watchlist
AMBIGUOUS_STEMS = {"Tessera", "Orbital", "Addition", "Aera", "Integra", "Seamless", "Create"}

# EPO XML namespaces
NS_OPS = "http://ops.epo.org"
NS_EPO = "http://www.epo.org/exchange"
NS_XML = "http://www.w3.org/XML/1998/namespace"


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_access_token: str = ""
_token_expiry: float = 0.0


def get_access_token() -> str:
    global _access_token, _token_expiry
    if _access_token and time.time() < _token_expiry - 60:
        return _access_token
    credentials = base64.b64encode(f"{EPO_OPS_KEY}:{EPO_OPS_SECRET}".encode()).decode()
    resp = requests.post(
        EPO_AUTH_URL,
        data="grant_type=client_credentials",
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type":  "application/x-www-form-urlencoded",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    _access_token = data["access_token"]
    _token_expiry = time.time() + int(data.get("expires_in", 1200))
    logging.info("  EPO token acquired")
    return _access_token


def auth_headers() -> dict:
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Accept": "application/xml",
    }


# ---------------------------------------------------------------------------
# Search/biblio — one-step fetch
# ---------------------------------------------------------------------------

def search_biblio(cql: str, start: int = 1, count: int = 25) -> dict:
    """
    Query the /search/biblio endpoint.
    Returns full bibliographic data in a single request — no second fetch needed.
    Returns {"patents": [...], "total": int} or {"patents": [], "total": 0} on 404.
    """
    headers = {**auth_headers(), "X-OPS-Range": f"{start}-{start + count - 1}"}
    try:
        resp = requests.get(
            EPO_SEARCH_BIBLIO,
            params={"q": cql},
            headers=headers,
            timeout=45,
        )
    except requests.RequestException as e:
        logging.warning(f"  Request error for '{cql[:60]}': {e}")
        return {"patents": [], "total": 0}

    if resp.status_code == 429:
        logging.warning("  Rate limit (429) — sleeping 30s")
        time.sleep(30)
        return search_biblio(cql, start, count)

    if resp.status_code == 403:
        # Single retry with a short sleep. If it fails again, raise ThrottleError
        # so fetch_all_for_company can handle it at the company level (one long
        # sleep) rather than retrying per-request and cascading 403s.
        logging.warning("  Forbidden (403) — sleeping 20s then retrying once")
        time.sleep(20)
        try:
            resp = requests.get(
                EPO_SEARCH_BIBLIO,
                params={"q": cql},
                headers={**auth_headers(), "X-OPS-Range": f"{start}-{start + count - 1}"},
                timeout=45,
            )
        except requests.RequestException as e:
            raise ThrottleError(f"Retry request failed: {e}") from e
        if resp.status_code in (403, 429):
            raise ThrottleError(f"Still throttled after retry: HTTP {resp.status_code}")
        if resp.status_code == 404:
            return {"patents": [], "total": 0}
        if not resp.ok:
            logging.warning(f"  Still failing after retry: HTTP {resp.status_code}")
            return {"patents": [], "total": 0}

    if resp.status_code == 404:
        return {"patents": [], "total": 0}

    if not resp.ok:
        logging.warning(f"  Unexpected HTTP {resp.status_code} for '{cql[:60]}'")
        return {"patents": [], "total": 0}

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        logging.warning(f"  XML parse error: {e}")
        return {"patents": [], "total": 0}

    total_el = root.find(f".//{{{NS_OPS}}}biblio-search")
    total = int(total_el.get("total-result-count", 0)) if total_el is not None else 0

    patents = parse_biblio_xml(root)
    return {"patents": patents, "total": total}


def collect_for_query(cql: str, company: str) -> list[dict]:
    """Paginate through all results for a single CQL query."""
    all_patents = []
    seen_ids = set()
    start = 1

    while True:
        result = search_biblio(cql, start=start, count=25)
        for p in result["patents"]:
            pid = p.get("id")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                p["watchlist_company"] = company
                all_patents.append(p)

        total = result["total"]
        fetched_so_far = start + 24
        if fetched_so_far >= total or start >= 100:  # cap at 100 per query
            break
        start += 25
        time.sleep(QUERY_SLEEP)

    return all_patents


# ---------------------------------------------------------------------------
# XML parsing — biblio constituent format
# ---------------------------------------------------------------------------

def parse_biblio_xml(root: ET.Element) -> list[dict]:
    """
    Parse the XML returned by /search/biblio.
    Each result is an exchange-document with full bib data attached.
    """
    results = []

    for doc in root.iter(f"{{{NS_EPO}}}exchange-document"):
        try:
            country    = doc.get("country", "")
            doc_number = doc.get("doc-number", "")
            kind       = doc.get("kind", "")
            pub_date   = doc.get("date", "")
            patent_id  = f"{country}{doc_number}.{kind}"

            # Title — prefer English
            title = ""
            for t in doc.iter(f"{{{NS_EPO}}}invention-title"):
                lang = t.get(f"{{{NS_XML}}}lang", "").lower()
                if lang in ("en", ""):
                    title = (t.text or "").strip()
                    if title:
                        break
            if not title:
                for t in doc.iter(f"{{{NS_EPO}}}invention-title"):
                    title = (t.text or "").strip()
                    if title:
                        break

            # Abstract — prefer English
            abstract = ""
            for ab in doc.iter(f"{{{NS_EPO}}}abstract"):
                lang = ab.get(f"{{{NS_XML}}}lang", "").lower()
                if lang in ("en", ""):
                    parts = [p.text or "" for p in ab.iter(f"{{{NS_EPO}}}p")]
                    abstract = " ".join(parts).strip()[:1000]
                    if abstract:
                        break

            # Applicants / assignees
            assignees = []
            for party in doc.iter(f"{{{NS_EPO}}}applicant"):
                name_el = party.find(f".//{{{NS_EPO}}}name")
                if name_el is not None and name_el.text:
                    assignees.append(name_el.text.strip())
            assignee_str = "; ".join(assignees[:3])

            # Filing date
            filing_date = ""
            for fd in doc.iter(f"{{{NS_EPO}}}filing-date"):
                raw = (fd.text or "").strip()
                filing_date = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}" if len(raw) == 8 else raw
                break

            if not title and not abstract:
                continue

            url = (
                f"https://patents.google.com/patent/{country}{doc_number}{kind}"
                if country and doc_number else ""
            )

            results.append({
                "id":                patent_id,
                "title":             title,
                "abstract":          abstract,
                "assignee":          assignee_str,
                "watchlist_company": "",   # filled in by caller
                "filing_date":       filing_date or (pub_date[:4] if pub_date else ""),
                "patent_number":     f"{country}{doc_number}{kind}",
                "application_number": doc_number,
                "source":            "EPO OPS",
                "url":               url,
                "data_type":         "patent",
                "claim_type":        None,
                "novelty_summary":   None,
                "relevant":          None,
            })

        except Exception as e:
            logging.warning(f"  Error parsing doc: {e}")
            continue

    return results


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def build_queries(company: str, date_from: str) -> list[str]:
    words = company.split()
    stem_words = []
    for w in words:
        if w.lower() not in GENERIC_WORDS:
            stem_words.append(w)
        if len(stem_words) == 2:
            break
    stem = " ".join(stem_words) if stem_words else words[0]
    date_filter = f"pd>={date_from}"
    # CPC subclass filter: pharmaceuticals (A61K), therapeutic uses (A61P),
    # and genetic/microbiological engineering (C12N).
    # Deliberately excludes A61B (surgery), A61C (dentistry), A61G (nursing),
    # A61L (sterilisation) etc. — those generate most of the irrelevant noise.
    ipc_filter  = "(cpc=A61K OR cpc=A61P OR cpc=C12N)"
    is_ambiguous = stem.split()[0] in AMBIGUOUS_STEMS

    queries = []

    # 1. Applicant wildcard — IPC-scoped for ambiguous stems
    if is_ambiguous:
        queries.append(f'pa="{stem}*" AND {ipc_filter} AND {date_filter}')
    else:
        queries.append(f'pa="{stem}*" AND {date_filter}')

    # 2. Exact full name — only when generic words were stripped
    if len(words) > len(stem_words):
        queries.append(f'pa="{company}" AND {date_filter}')

    # 3. Title/abstract fallback — always IPC-scoped
    queries.append(f'ta="{stem}" AND {ipc_filter} AND {date_filter}')

    return queries


# ---------------------------------------------------------------------------
# Per-company fetch
# ---------------------------------------------------------------------------

def fetch_all_for_company(company: str, date_from: str) -> list[dict]:
    all_patents: list[dict] = []
    seen_ids: set[str] = set()

    for cql in build_queries(company, date_from):
        try:
            patents = collect_for_query(cql, company)
        except ThrottleError as e:
            # EPO is throttling at the per-minute level. Sleep once at company
            # level and move on — retrying the same query would just compound
            # the problem for all downstream companies.
            logging.warning(f"  Throttled on '{cql[:60]}': {e} — sleeping {THROTTLE_SLEEP:.0f}s")
            time.sleep(THROTTLE_SLEEP)
            break
        for p in patents:
            pid = p.get("id")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_patents.append(p)
        time.sleep(QUERY_SLEEP)

    return all_patents


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
            model=MODEL, max_tokens=400,
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
        # Apply English translation if title was non-English
        title_en = (result.get("title_en") or "").strip()
        if title_en:
            patent["title_original"] = patent["title"]
            patent["title"] = title_en
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

    # Load existing patents to preserve Claude analysis
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

    # Auto-detect mode: full scan if cache is empty (or FULL_SCAN env var set),
    # incremental otherwise — no manual workflow changes needed for a rebuild
    cache_empty = len(all_patents) == 0
    do_full_scan = FULL_SCAN or cache_empty
    if do_full_scan:
        year_from = (datetime.now(timezone.utc) - timedelta(days=365 * LOOKBACK_YEARS)).year
        date_from = f"{year_from}0101"
        reason    = "empty cache" if cache_empty else "FULL_SCAN env var"
        logging.info(f"Mode: FULL SCAN (5yr lookback) — {reason}")
    else:
        date_from = DATE_FROM_STR
        logging.info(f"Mode: incremental (last {INCREMENTAL_DAYS}d)")

    logging.info(f"Processing {len(watchlist)} watchlist companies from {date_from}")

    new_count = 0

    for company in watchlist:
        logging.info(f"  Searching: {company}")
        try:
            patents = fetch_all_for_company(company, date_from)
            logging.info(f"    {len(patents)} patents found")

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

            # Proportional cooldown: large result sets burn more API quota,
            # so sleep longer before the next company to let the rate-limit
            # bucket refill. Base 3s + 0.5s per patent found, capped at 30s.
            cooldown = min(
                COMPANY_SLEEP_BASE + len(patents) * COMPANY_SLEEP_PER,
                COMPANY_SLEEP_MAX,
            )
            time.sleep(cooldown)
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
