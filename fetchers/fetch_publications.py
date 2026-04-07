"""
fetch_publications.py
---------------------
Fetches recent in vivo CAR-T publications from:
  - PubMed via NCBI E-utilities API (peer-reviewed)
  - bioRxiv via REST API (preprints)

Two fetch passes:
  1. Keyword queries (broad in vivo CAR-T terms)
  2. Watchlist company queries (one per company in watchlist.json)

Writes structured JSON to data/publications.json.
"""

import json
import os
import time
import logging
import xml.etree.ElementTree as ET
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOOKBACK_DAYS  = 30
PUBMED_MAX     = 100
BIORXIV_MAX    = 50

PUBMED_QUERY = (
    '("in vivo CAR-T"[tiab] OR "in vivo CAR T"[tiab] OR '
    '"in vivo chimeric antigen receptor"[tiab] OR '
    '"lentiviral CAR T in vivo"[tiab] OR '
    '"lipid nanoparticle CAR"[tiab] OR '
    '"non-viral CAR T"[tiab]) '
    'AND ("last {days} days"[dp])'
)

BIORXIV_SEARCH_TERMS = [
    "in vivo CAR-T",
    "in vivo CAR T cell",
    "lipid nanoparticle CAR",
]

NCBI_BASE    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BIORXIV_BASE = "https://api.biorxiv.org"

OUTPUT_PATH    = Path(__file__).parent.parent / "data" / "publications.json"
WATCHLIST_PATH = Path(__file__).parent.parent / "watchlist.json"

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")


# ---------------------------------------------------------------------------
# Watchlist helpers
# ---------------------------------------------------------------------------

def load_watchlist() -> list[str]:
    try:
        data = json.loads(WATCHLIST_PATH.read_text())
        return data.get("companies", [])
    except Exception as e:
        logging.warning(f"Could not load watchlist: {e}")
        return []


# ---------------------------------------------------------------------------
# PubMed helpers
# ---------------------------------------------------------------------------

def pubmed_search(query: str, max_results: int) -> list[str]:
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
            results.append(parse_pubmed_article(article))
        except Exception as e:
            logging.warning(f"Failed to parse PubMed article: {e}")
    return results


def parse_pubmed_article(article: ET.Element) -> dict:
    def txt(path, default=""):
        el = article.find(path)
        return "".join(el.itertext()).strip() if el is not None else default

    pmid     = txt(".//PMID")
    title    = txt(".//ArticleTitle")
    journal  = txt(".//Journal/Title")
    abstract = txt(".//AbstractText")

    authors = []
    for author in article.findall(".//Author"):
        ln = author.find("LastName")
        last = "".join(ln.itertext()).strip() if ln is not None else ""
        ini  = author.find("Initials")
        initials = "".join(ini.itertext()).strip() if ini is not None else ""
        if last:
            authors.append(f"{last} {initials}".strip())
    author_str = ", ".join(authors[:5])
    if len(authors) > 5:
        author_str += " et al."

    year  = txt(".//PubDate/Year") or txt(".//PubDate/MedlineDate")[:4]
    month = txt(".//PubDate/Month") or ""
    date_str = f"{year}-{month}" if month else year

    doi = ""
    for id_el in article.findall(".//ArticleId"):
        if id_el.get("IdType") == "doi":
            doi = id_el.text or ""

    return {
        "source":    "pubmed",
        "pmid":      pmid,
        "title":     title,
        "journal":   journal,
        "authors":   author_str,
        "abstract":  abstract,
        "date":      date_str,
        "doi":       doi,
        "mesh_terms": [],
        "preprint":  False,
        "url":       f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "sowhat":    None,
        "category":  None,
    }


# ---------------------------------------------------------------------------
# bioRxiv helpers
# ---------------------------------------------------------------------------

def biorxiv_fetch(term: str, lookback_days: int, max_results: int) -> list[dict]:
    end_date   = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=lookback_days)
    url = f"{BIORXIV_BASE}/details/biorxiv/{start_date.isoformat()}/{end_date.isoformat()}/0/json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    collection = resp.json().get("collection", [])
    term_lower = term.lower()
    results = []
    for item in collection:
        text = " ".join([item.get("title",""), item.get("abstract",""), item.get("category","")]).lower()
        if term_lower in text:
            results.append(parse_biorxiv_item(item))
        if len(results) >= max_results:
            break
    return results


def parse_biorxiv_item(item: dict) -> dict:
    doi = item.get("doi", "")
    authors_raw = item.get("authors", "")
    author_list = [a.strip() for a in authors_raw.split(";") if a.strip()]
    author_str  = ", ".join(author_list[:5])
    if len(author_list) > 5:
        author_str += " et al."
    return {
        "source":    "biorxiv",
        "pmid":      None,
        "title":     item.get("title", ""),
        "journal":   "bioRxiv (preprint)",
        "authors":   author_str,
        "abstract":  item.get("abstract", ""),
        "date":      item.get("date", ""),
        "doi":       doi,
        "mesh_terms": [],
        "preprint":  True,
        "url":       f"https://doi.org/{doi}" if doi else "",
        "sowhat":    None,
        "category":  None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting publications fetch")

    all_pubs: dict[str, dict] = {}

    # Pass 1 — keyword queries
    logging.info("Pass 1: PubMed keyword query")
    try:
        query = PUBMED_QUERY.format(days=LOOKBACK_DAYS)
        pmids = pubmed_search(query, PUBMED_MAX)
        logging.info(f"  Found {len(pmids)} PMIDs")
        if pmids:
            time.sleep(0.4)
            for a in pubmed_fetch(pmids):
                key = a["doi"] or f"pmid:{a['pmid']}"
                if key not in all_pubs:
                    all_pubs[key] = a
    except requests.RequestException as e:
        logging.error(f"PubMed keyword fetch failed: {e}")

    logging.info("Pass 1: bioRxiv keyword queries")
    for term in BIORXIV_SEARCH_TERMS:
        try:
            preprints = biorxiv_fetch(term, LOOKBACK_DAYS, BIORXIV_MAX)
            for p in preprints:
                key = p["doi"] or p["title"][:60]
                if key not in all_pubs:
                    all_pubs[key] = p
            logging.info(f"  '{term}': {len(preprints)} preprints")
            time.sleep(0.5)
        except requests.RequestException as e:
            logging.error(f"bioRxiv fetch failed for '{term}': {e}")

    logging.info(f"  After keyword pass: {len(all_pubs)} unique publications")

    # Pass 2 — watchlist company queries
    watchlist = load_watchlist()
    logging.info(f"Pass 2: watchlist company queries ({len(watchlist)} companies)")
    new_from_watchlist = 0

    for company in watchlist:
        try:
            # Search PubMed for company name in title/abstract
            company_query = f'"{company}"[tiab] AND ("last {LOOKBACK_DAYS} days"[dp])'
            pmids = pubmed_search(company_query, 20)
            if pmids:
                time.sleep(0.4)
                for a in pubmed_fetch(pmids):
                    key = a["doi"] or f"pmid:{a['pmid']}"
                    if key not in all_pubs:
                        all_pubs[key] = a
                        new_from_watchlist += 1
                logging.info(f"  {company}: {len(pmids)} PubMed results")
            time.sleep(0.3)
        except requests.RequestException as e:
            logging.error(f"PubMed watchlist query '{company}' failed: {e}")

    logging.info(f"  {new_from_watchlist} new publications added from watchlist pass")

    # Preserve existing sowhat and category
    if OUTPUT_PATH.exists():
        try:
            existing = json.loads(OUTPUT_PATH.read_text())
            for p in existing.get("publications", []):
                key = p.get("doi") or f"pmid:{p.get('pmid')}"
                if key in all_pubs:
                    if p.get("sowhat"):
                        all_pubs[key]["sowhat"] = p["sowhat"]
                    if p.get("category"):
                        all_pubs[key]["category"] = p["category"]
        except Exception as e:
            logging.warning(f"Could not preserve existing pub data: {e}")

    pubs_list = sorted(all_pubs.values(), key=lambda p: p["date"] or "", reverse=True)

    output = {
        "fetched_at":   datetime.now(timezone.utc).isoformat(),
        "count":        len(pubs_list),
        "publications": pubs_list,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logging.info(f"Wrote {len(pubs_list)} publications to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
