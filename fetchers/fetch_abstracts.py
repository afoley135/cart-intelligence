"""
fetch_abstracts.py
------------------
Fetches conference abstracts from three sources:

  1. PubMed — AACR (Cancer Research supplement) and ASH (Blood supplement)
     using a 180-day lookback window and conference-specific queries
  2. ASGCT website scraper — runs after April 27 when abstracts go live
     fetches from annualmeeting.asgct.org and filters for in vivo CAR-T

Writes structured JSON to data/abstracts.json.
Idempotent — preserves existing sowhat values.

Requires: no API key for ASGCT scraper
          NCBI_API_KEY optional (higher PubMed rate limits)
"""

import json
import logging
import os
import time
import re
import xml.etree.ElementTree as ET
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_PATH    = Path(__file__).parent.parent / "data" / "abstracts.json"
WATCHLIST_PATH = Path(__file__).parent.parent / "watchlist.json"

NCBI_BASE    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")

# Separate lookback for conference abstracts
ABSTRACT_LOOKBACK_DAYS = 180

# Conference supplement journal identifiers in PubMed
CONFERENCE_JOURNALS = {
    "Cancer Res":          {"conference": "AACR",  "year": 2026},
    "Cancer Research":     {"conference": "AACR",  "year": 2026},
    "Blood":               {"conference": "ASH",   "year": 2026},
    "Mol Ther":            {"conference": "ASGCT", "year": 2026},
    "Molecular Therapy":   {"conference": "ASGCT", "year": 2026},
}

# PubMed queries for conference abstracts
# These target the supplement issues specifically
CONFERENCE_PUBMED_QUERIES = [
    # AACR — Cancer Research supplement
    '("in vivo CAR-T"[tiab] OR "in vivo CAR T"[tiab] OR "lipid nanoparticle CAR"[tiab] OR "non-viral CAR T"[tiab]) AND "Cancer Res"[journal] AND "2026"[dp]',
    # ASH — Blood supplement
    '("in vivo CAR-T"[tiab] OR "in vivo CAR T"[tiab] OR "lipid nanoparticle CAR"[tiab]) AND "Blood"[journal] AND "2026"[dp]',
    # ASGCT — Molecular Therapy supplement (published after meeting)
    '("in vivo CAR-T"[tiab] OR "in vivo CAR T"[tiab] OR "lipid nanoparticle CAR"[tiab]) AND "Mol Ther"[journal] AND "2026"[dp]',
]

# General in vivo CAR-T abstract query with long lookback
GENERAL_ABSTRACT_QUERY = (
    '("in vivo CAR-T"[tiab] OR "in vivo CAR T"[tiab] OR '
    '"in vivo chimeric antigen receptor"[tiab] OR '
    '"lipid nanoparticle CAR"[tiab] OR "non-viral CAR T"[tiab]) '
    'AND ("last {days} days"[dp])'
)

# ASGCT scraper config
ASGCT_ABSTRACT_URL = "https://annualmeeting.asgct.org/abstracts/abstract-search"
ASGCT_EMBARGO_DATE = "2026-04-27"  # abstracts go live this date

IN_VIVO_KEYWORDS = [
    "in vivo car", "in vivo chimeric antigen", "lipid nanoparticle car",
    "lentiviral car", "non-viral car", "direct t cell", "systemic car",
    "in vivo gene therapy", "in vivo t cell engineering",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_watchlist() -> list[str]:
    try:
        return json.loads(WATCHLIST_PATH.read_text()).get("companies", [])
    except Exception:
        return []


def is_past_embargo(embargo_date: str) -> bool:
    embargo = datetime.strptime(embargo_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= embargo


def detect_conference(journal: str, title: str) -> str | None:
    """Detect which conference an abstract is from based on journal."""
    j = (journal or "").strip()
    for key, info in CONFERENCE_JOURNALS.items():
        if key.lower() in j.lower():
            return info["conference"]
    # Also check title for explicit conference mentions
    t = (title or "").lower()
    if "asgct" in t or "american society of gene" in t:
        return "ASGCT"
    if "ash" in t or "american society of hematology" in t:
        return "ASH"
    if "aacr" in t or "american association for cancer" in t:
        return "AACR"
    return None


# ---------------------------------------------------------------------------
# PubMed helpers
# ---------------------------------------------------------------------------

def pubmed_search(query: str, max_results: int = 100) -> list[str]:
    params = {
        "db": "pubmed", "term": query,
        "retmax": max_results, "retmode": "json", "sort": "date",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def pubmed_fetch(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    resp = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    results = []
    for article in root.findall(".//PubmedArticle"):
        try:
            results.append(parse_pubmed_abstract(article))
        except Exception as e:
            logging.warning(f"Failed to parse abstract: {e}")
    return results


def parse_pubmed_abstract(article: ET.Element) -> dict:
    def txt(path, default=""):
        el = article.find(path)
        return "".join(el.itertext()).strip() if el is not None else default

    pmid     = txt(".//PMID")
    title    = txt(".//ArticleTitle")
    journal  = txt(".//Journal/Title") or txt(".//Journal/ISOAbbreviation")
    abstract = txt(".//AbstractText")
    year     = txt(".//PubDate/Year") or txt(".//PubDate/MedlineDate")[:4]
    month    = txt(".//PubDate/Month") or ""
    date_str = f"{year}-{month}" if month else year

    authors = []
    for author in article.findall(".//Author"):
        ln  = author.find("LastName")
        ini = author.find("Initials")
        last     = "".join(ln.itertext()).strip() if ln is not None else ""
        initials = "".join(ini.itertext()).strip() if ini is not None else ""
        if last:
            authors.append(f"{last} {initials}".strip())
    author_str = ", ".join(authors[:5])
    if len(authors) > 5:
        author_str += " et al."

    doi = ""
    for id_el in article.findall(".//ArticleId"):
        if id_el.get("IdType") == "doi":
            doi = id_el.text or ""

    conference = detect_conference(journal, title)

    return {
        "source":      "pubmed",
        "pmid":        pmid,
        "title":       title,
        "journal":     journal,
        "authors":     author_str,
        "abstract":    abstract,
        "date":        date_str,
        "doi":         doi,
        "conference":  conference,
        "preprint":    False,
        "url":         f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "sowhat":      None,
    }


# ---------------------------------------------------------------------------
# ASGCT scraper
# ---------------------------------------------------------------------------

def scrape_asgct_abstracts(watchlist: list[str]) -> list[dict]:
    """
    Scrape ASGCT 2026 abstracts after embargo lifts on April 27.
    Uses the ASGCT abstract search page with keyword queries.
    Falls back gracefully if the page structure has changed.
    """
    if not is_past_embargo(ASGCT_EMBARGO_DATE):
        logging.info(f"  ASGCT abstracts not yet public (embargo: {ASGCT_EMBARGO_DATE})")
        return []

    results = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; cart-intelligence-bot/1.0)"}

    # Search terms to try
    search_terms = [
        "in vivo CAR-T",
        "in vivo CAR T",
        "lipid nanoparticle CAR",
        "lentiviral CAR",
        "non-viral CAR",
    ] + [company for company in watchlist[:15]]  # top watchlist companies

    seen_titles = set()

    for term in search_terms:
        try:
            # ASGCT uses a search parameter in the URL
            url = f"{ASGCT_ABSTRACT_URL}?q={requests.utils.quote(term)}"
            resp = requests.get(url, headers=headers, timeout=30)

            if resp.status_code != 200:
                logging.warning(f"  ASGCT search returned {resp.status_code} for '{term}'")
                continue

            # Parse abstracts from HTML — look for common abstract card patterns
            html = resp.text
            abstracts = parse_asgct_html(html, term)

            for a in abstracts:
                title_key = a["title"][:60].lower()
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    results.append(a)

            logging.info(f"  ASGCT '{term}': {len(abstracts)} abstracts")
            time.sleep(1.0)  # be polite

        except Exception as e:
            logging.error(f"  ASGCT scrape failed for '{term}': {e}")

    return results


def parse_asgct_html(html: str, search_term: str) -> list[dict]:
    """
    Parse ASGCT abstract search results HTML.
    ASGCT's abstract system typically uses a standard format —
    this parser handles common patterns and degrades gracefully.
    """
    results = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Look for abstract title + number patterns
    # ASGCT abstracts typically have format: "123. Title of Abstract"
    title_pattern = re.compile(
        r'<[^>]+class="[^"]*(?:abstract-title|session-title|title)[^"]*"[^>]*>(.*?)</[^>]+>',
        re.IGNORECASE | re.DOTALL
    )

    body_pattern = re.compile(
        r'<[^>]+class="[^"]*(?:abstract-body|abstract-text|body)[^"]*"[^>]*>(.*?)</[^>]+>',
        re.IGNORECASE | re.DOTALL
    )

    def clean_html(text: str) -> str:
        return re.sub(r'<[^>]+>', ' ', text).strip()

    titles = [clean_html(m.group(1)) for m in title_pattern.finditer(html)]
    bodies = [clean_html(m.group(1)) for m in body_pattern.finditer(html)]

    # If we couldn't parse structured data, fall back to full-text search
    if not titles:
        text_lower = html.lower()
        kw_lower   = search_term.lower()
        if kw_lower in text_lower:
            # Abstract found but couldn't parse structure
            # Return a placeholder that signals manual review
            results.append({
                "source":     "ASGCT 2026",
                "pmid":       None,
                "title":      f"[ASGCT 2026 abstract — search: {search_term}]",
                "journal":    "ASGCT 2026 Annual Meeting",
                "authors":    "",
                "abstract":   "",
                "date":       today,
                "doi":        "",
                "conference": "ASGCT",
                "preprint":   False,
                "url":        ASGCT_ABSTRACT_URL,
                "sowhat":     None,
            })
        return results

    for i, title in enumerate(titles):
        if not title or len(title) < 10:
            continue

        # Check relevance
        text_to_check = (title + " " + (bodies[i] if i < len(bodies) else "")).lower()
        if not any(kw in text_to_check for kw in IN_VIVO_KEYWORDS):
            # Also check if it mentions a watchlist company
            continue

        results.append({
            "source":     "ASGCT 2026",
            "pmid":       None,
            "title":      title,
            "journal":    "ASGCT 2026 Annual Meeting",
            "authors":    "",
            "abstract":   bodies[i] if i < len(bodies) else "",
            "date":       today,
            "doi":        "",
            "conference": "ASGCT",
            "preprint":   False,
            "url":        ASGCT_ABSTRACT_URL,
            "sowhat":     None,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting conference abstract fetch")

    watchlist   = load_watchlist()
    all_abstracts: dict[str, dict] = {}

    # ── PubMed conference queries ─────────────────────────────────────────────
    logging.info("PubMed conference supplement queries")
    for query in CONFERENCE_PUBMED_QUERIES:
        try:
            pmids = pubmed_search(query, 100)
            logging.info(f"  Found {len(pmids)} PMIDs for: {query[:60]}...")
            if pmids:
                time.sleep(0.4)
                for a in pubmed_fetch(pmids):
                    key = a["doi"] or f"pmid:{a['pmid']}"
                    if key not in all_abstracts:
                        all_abstracts[key] = a
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"PubMed query failed: {e}")

    # ── General abstract query with long lookback ─────────────────────────────
    logging.info(f"PubMed general abstract query ({ABSTRACT_LOOKBACK_DAYS} day lookback)")
    try:
        query = GENERAL_ABSTRACT_QUERY.format(days=ABSTRACT_LOOKBACK_DAYS)
        pmids = pubmed_search(query, 200)
        logging.info(f"  Found {len(pmids)} PMIDs")
        if pmids:
            time.sleep(0.4)
            for a in pubmed_fetch(pmids):
                key = a["doi"] or f"pmid:{a['pmid']}"
                if key not in all_abstracts:
                    all_abstracts[key] = a
    except Exception as e:
        logging.error(f"General abstract query failed: {e}")

    # ── Watchlist company PubMed queries ──────────────────────────────────────
    logging.info(f"Watchlist company PubMed queries ({len(watchlist)} companies)")
    for company in watchlist:
        try:
            query = f'"{company}"[tiab] AND ("last {ABSTRACT_LOOKBACK_DAYS} days"[dp])'
            pmids = pubmed_search(query, 20)
            if pmids:
                time.sleep(0.4)
                for a in pubmed_fetch(pmids):
                    key = a["doi"] or f"pmid:{a['pmid']}"
                    if key not in all_abstracts:
                        all_abstracts[key] = a
                logging.info(f"  {company}: {len(pmids)} results")
            time.sleep(0.3)
        except Exception as e:
            logging.error(f"Watchlist query failed for '{company}': {e}")

    # ── ASGCT scraper ─────────────────────────────────────────────────────────
    logging.info("ASGCT 2026 abstract scraper")
    asgct_abstracts = scrape_asgct_abstracts(watchlist)
    for a in asgct_abstracts:
        key = a.get("doi") or a["title"][:60]
        if key not in all_abstracts:
            all_abstracts[key] = a
    logging.info(f"  Added {len(asgct_abstracts)} ASGCT abstracts")

    # ── Preserve existing sowhat values ───────────────────────────────────────
    if OUTPUT_PATH.exists():
        try:
            existing = json.loads(OUTPUT_PATH.read_text())
            for a in existing.get("abstracts", []):
                key = a.get("doi") or f"pmid:{a.get('pmid')}"
                if key in all_abstracts and a.get("sowhat"):
                    all_abstracts[key]["sowhat"] = a["sowhat"]
        except Exception as e:
            logging.warning(f"Could not preserve existing data: {e}")

    abstracts_list = sorted(
        all_abstracts.values(),
        key=lambda a: a["date"] or "",
        reverse=True,
    )

    # Safety guard
    if not abstracts_list:
        logging.warning("No abstracts found — writing empty result")

    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count":      len(abstracts_list),
        "abstracts":  abstracts_list,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logging.info(f"Wrote {len(abstracts_list)} abstracts to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
