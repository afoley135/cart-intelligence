"""
fetch_patents.py
----------------
Fetches patent filings for watchlist companies using the EPO Open Patent
Services (OPS) API v3.2.

EPO OPS requires a two-step process:
  1. Search endpoint  → returns publication references (country/number/kind)
  2. Biblio endpoint  → fetch titles, abstracts, assignees for those IDs

Three search strategies per company:
  1. pa="CompanyName*"           — applicant wildcard (most reliable)
  2. pa="Full Company Name"      — exact legal name fallback
  3. ta="CompanyName" AND ic=A61 — title/abstract fallback for parent-entity filings

For each new patent, calls Claude to:
  - Classify claim type
  - Summarise novelty and competitive relevance

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
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_PATH    = Path(__file__).parent.parent / "data" / "patents.json"
WATCHLIST_PATH = Path(__file__).parent.parent / "watchlist.json"

LOOKBACK_YEARS = 5
MODEL          = "claude-haiku-4-5-20251001"

EPO_OPS_KEY       = os.environ.get("EPO_OPS_KEY", "")
EPO_OPS_SECRET    = os.environ.get("EPO_OPS_SECRET", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

EPO_AUTH_URL   = "https://ops.epo.org/3.2/auth/accesstoken"
EPO_SEARCH_URL = "https://ops.epo.org/3.2/rest-services/published-data/search"
EPO_BIBLIO_URL = "https://ops.epo.org/3.2/rest-services/published-data/publication/docdb/bulk/biblio"

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

# Generic trailing words to strip when building CQL stems
GENERIC_WORDS = {
    "therapeutics", "biosciences", "biotech", "biotherapeutics",
    "medicines", "biopharma", "pharma", "biologics", "labs",
    "laboratory", "laboratories", "inc", "llc", "ltd", "gmbh",
}

# Stems that match large unrelated companies — scope these to biotech IPC classes
# to avoid burning Claude API calls on irrelevant results.
# Tessera → Tessera Technologies (semiconductors, 982 results)
# Orbital → Orbital Sciences / ATK (aerospace)
# Addition → generic word, matches many unrelated filers
# Aera    → Aera Energy (oil & gas)
# Integra → Integra LifeSciences (wound care / neurosurgery)
# Seamless → multiple unrelated companies
AMBIGUOUS_STEMS = {"Tessera", "Orbital", "Addition", "Aera", "Integra", "Seamless"}

# EPO OPS namespaces
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
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    _access_token = data["access_token"]
    _token_expiry = time.time() + int(data.get("expires_in", 1200))
    logging.info("  EPO token acquired")
    return _access_token


def auth_headers(accept: str = "application/xml") -> dict:
    return {"Authorization": f"Bearer {get_access_token()}", "Accept": accept}


# ---------------------------------------------------------------------------
# Step 1 — Search: get publication references
# ---------------------------------------------------------------------------

def search_epo(cql: str, start: int = 1, count: int = 25) -> dict:
    """Returns {refs: [(country, doc_number, kind), ...], total: int}"""
    resp = requests.get(
        EPO_SEARCH_URL,
        params={"q": cql},
        headers={**auth_headers(), "X-OPS-Range": f"{start}-{start + count - 1}"},
        timeout=30,
    )

    if resp.status_code == 429:
        logging.warning("  Rate limit — sleeping 15s")
        time.sleep(15)
        return search_epo(cql, start, count)

    if resp.status_code == 404:
        return {"refs": [], "total": 0}

    resp.raise_for_status()

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        logging.warning(f"  Search XML parse error: {e}")
        return {"refs": [], "total": 0}

    total_el = root.find(f".//{{{NS_OPS}}}biblio-search")
    total = int(total_el.get("total-result-count", 0)) if total_el is not None else 0

    refs = []
    for pub_ref in root.iter(f"{{{NS_OPS}}}publication-reference"):
        doc_id = pub_ref.find(f".//{{{NS_EPO}}}document-id[@document-id-type='docdb']")
        if doc_id is None:
            doc_id = pub_ref.find(f".//{{{NS_EPO}}}document-id")
        if doc_id is None:
            continue
        country    = (doc_id.findtext(f"{{{NS_EPO}}}country") or "").strip()
        doc_number = (doc_id.findtext(f"{{{NS_EPO}}}doc-number") or "").strip()
        kind       = (doc_id.findtext(f"{{{NS_EPO}}}kind") or "").strip()
        if country and doc_number:
            refs.append((country, doc_number, kind))

    return {"refs": refs, "total": total}


def collect_refs(cql: str) -> list[tuple]:
    """Paginate through all search results for a CQL query, return all refs."""
    all_refs = []
    seen = set()
    start = 1
    try:
        while True:
            result = search_epo(cql, start=start, count=25)
            for ref in result["refs"]:
                key = f"{ref[0]}{ref[1]}.{ref[2]}"
                if key not in seen:
                    seen.add(key)
                    all_refs.append(ref)
            total = result["total"]
            if start + 25 > total or start >= 100:  # cap at 100 per query
                break
            start += 25
            time.sleep(0.5)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        if status != 404:
            logging.warning(f"  Search failed for '{cql[:60]}': HTTP {status}")
    except Exception as e:
        logging.warning(f"  Search failed for '{cql[:60]}': {e}")
    return all_refs


# ---------------------------------------------------------------------------
# Step 2 — Biblio: fetch full data for a batch of refs
# ---------------------------------------------------------------------------

def fetch_biblio_batch(refs: list[tuple]) -> list[dict]:
    """
    Fetch bibliographic data for up to 10 publication references at once.
    Falls back to individual fetches if bulk fails.
    """
    if not refs:
        return []

    doc_ids = ",".join(f"{c}.{n}.{k}" for c, n, k in refs)

    resp = requests.get(
        f"{EPO_BIBLIO_URL}/{doc_ids}",
        headers=auth_headers(),
        timeout=60,
    )

    if resp.status_code == 429:
        logging.warning("  Biblio rate limit — sleeping 15s")
        time.sleep(15)
        return fetch_biblio_batch(refs)

    if not resp.ok:
        logging.warning(f"  Bulk biblio failed (HTTP {resp.status_code}), trying individually")
        return fetch_biblio_individually(refs)

    return parse_biblio_xml(resp.text)


def fetch_biblio_individually(refs: list[tuple]) -> list[dict]:
    """Fallback: fetch biblio one patent at a time."""
    results = []
    base = "https://ops.epo.org/3.2/rest-services/published-data/publication/docdb/{id}/biblio"
    for country, doc_number, kind in refs:
        doc_id = f"{country}.{doc_number}.{kind}"
        try:
            resp = requests.get(base.format(id=doc_id), headers=auth_headers(), timeout=30)
            if resp.status_code == 429:
                time.sleep(15)
                resp = requests.get(base.format(id=doc_id), headers=auth_headers(), timeout=30)
            if resp.ok:
                results.extend(parse_biblio_xml(resp.text))
            time.sleep(0.3)
        except Exception as e:
            logging.warning(f"  Individual biblio failed for {doc_id}: {e}")
    return results


def parse_biblio_xml(xml_text: str) -> list[dict]:
    """Parse EPO biblio XML — works for both bulk and single responses."""
    results = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logging.warning(f"  Biblio XML parse error: {e}")
        return results

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

            # Assignees / applicants
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

            url = f"https://patents.google.com/patent/{country}{doc_number}{kind}" if country and doc_number else ""

            results.append({
                "id":                patent_id,
                "title":             title,
                "abstract":          abstract,
                "assignee":          assignee_str,
                "watchlist_company": "",   # caller fills this in
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
            logging.warning(f"  Error parsing biblio doc: {e}")
            continue

    return results


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def build_queries(company: str) -> list[str]:
    words = company.split()
    stem_words = []
    for w in words:
        if w.lower() not in GENERIC_WORDS:
            stem_words.append(w)
        if len(stem_words) == 2:
            break
    stem = " ".join(stem_words) if stem_words else words[0]
    date_filter = f"pd>={YEAR_FROM}0101"
    ipc_filter  = "(ic=A61 OR ic=C12N)"

    # For ambiguous stems that match large unrelated companies, always scope
    # to biotech IPC classes to avoid thousands of irrelevant results.
    is_ambiguous = stem.split()[0] in AMBIGUOUS_STEMS

    if is_ambiguous:
        queries = [f'pa="{stem}*" AND {ipc_filter} AND {date_filter}']
    else:
        queries = [f'pa="{stem}*" AND {date_filter}']

    # Exact full name — only needed when generic words were stripped
    if len(words) > len(stem_words):
        queries.append(f'pa="{company}" AND {date_filter}')

    # Title/abstract fallback — always IPC-scoped to reduce noise
    queries.append(f'ta="{stem}" AND {ipc_filter} AND {date_filter}')

    return queries


# ---------------------------------------------------------------------------
# Per-company fetch
# ---------------------------------------------------------------------------

def fetch_all_for_company(company: str) -> list[dict]:
    all_refs: list[tuple] = []
    seen_ids: set[str] = set()

    for cql in build_queries(company):
        refs = collect_refs(cql)
        for ref in refs:
            key = f"{ref[0]}{ref[1]}.{ref[2]}"
            if key not in seen_ids:
                seen_ids.add(key)
                all_refs.append(ref)
        time.sleep(0.5)

    if not all_refs:
        return []

    logging.info(f"    {len(all_refs)} unique refs — fetching biblio in batches")

    # Fetch biblio in batches of 10
    patents = []
    for i in range(0, len(all_refs), 10):
        batch = all_refs[i:i + 10]
        results = fetch_biblio_batch(batch)
        for p in results:
            p["watchlist_company"] = company
        patents.extend(results)
        time.sleep(1.0)

    return patents


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

    new_count = 0

    for company in watchlist:
        logging.info(f"  Searching: {company}")
        try:
            patents = fetch_all_for_company(company)
            logging.info(f"    {len(patents)} patents with biblio data")

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

            time.sleep(1.5)
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
