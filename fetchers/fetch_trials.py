"""
fetch_publications.py
---------------------
Fetches recent in vivo CAR-T publications from:
  - PubMed via NCBI E-utilities API (peer-reviewed)
  - bioRxiv via REST API (preprints)

Writes structured JSON to data/publications.json.

API docs:
  PubMed:  https://www.ncbi.nlm.nih.gov/books/NBK25501/
  bioRxiv: https://api.biorxiv.org/

No API key required, but optionally set NCBI_API_KEY env var for
higher rate limits (10 req/s vs 3 req/s).
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

# How many days back to look for new publications
LOOKBACK_DAYS = 30

# Max results per source
PUBMED_MAX = 100
BIORXIV_MAX = 50

# PubMed search query — uses standard Entrez query syntax
# Adjust MeSH terms / free text as needed
PUBMED_QUERY = (
    '("in vivo CAR-T"[tiab] OR "in vivo CAR T"[tiab] OR '
    '"in vivo chimeric antigen receptor"[tiab] OR '
    '"lentiviral CAR T in vivo"[tiab] OR '
    '"lipid nanoparticle CAR"[tiab] OR '
    '"non-viral CAR T"[tiab]) '
    'AND ("last {days} days"[dp])'
)

# bioRxiv category to search within
BIORXIV_CATEGORY = "immunology"
BIORXIV_SEARCH_TERMS = [
    "in vivo CAR-T",
    "in vivo CAR T cell",
    "lipid nanoparticle CAR",
]

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BIORXIV_BASE = "https://api.biorxiv.org"

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "publications.json"

# Optional NCBI API key — get one free at https://www.ncbi.nlm.nih.gov/account/
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")


# ---------------------------------------------------------------------------
# PubMed helpers
# ---------------------------------------------------------------------------

def pubmed_search(query: str, max_results: int) -> list[str]:
    """Return list of PMIDs matching query."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "date",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def pubmed_fetch(pmids: list[str]) -> list[dict]:
    """Fetch full records for a list of PMIDs."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
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
    def txt(path: str, default="") -> str:
        el = article.find(path)
        return "".join(el.itertext()).strip() if el is not None else default

    pmid = txt(".//PMID")
    title = txt(".//ArticleTitle")
    journal = txt(".//Journal/Title")
    abstract = txt(".//AbstractText")

    # Authors
    authors = []
    for author in article.findall(".//Author"):
        last = txt("LastName", "") if (ln := author.find("LastName")) is not None else ""
        last = "".join(ln.itertext()).strip() if ln is not None else ""
        initials = "".join((author.find("Initials") or ET.Element("x")).itertext()).strip()
        if last:
            authors.append(f"{last} {initials}".strip())
    author_str = ", ".join(authors[:5])
    if len(authors) > 5:
        author_str += " et al."

    # Publication date
    pub_date = article.find(".//PubDate")
    year = txt(".//PubDate/Year") or txt(".//PubDate/MedlineDate")[:4]
    month = txt(".//PubDate/Month") or ""
    date_str = f"{year}-{month}" if month else year

    # DOI
    doi = ""
    for id_el in article.findall(".//ArticleId"):
        if id_el.get("IdType") == "doi":
            doi = id_el.text or ""

    # MeSH terms
    mesh = [
        "".join(m.itertext()).strip()
        for m in article.findall(".//MeshHeading/DescriptorName")
    ]

    return {
        "source": "pubmed",
        "pmid": pmid,
        "title": title,
        "journal": journal,
        "authors": author_str,
        "abstract": abstract,
        "date": date_str,
        "doi": doi,
        "mesh_terms": mesh,
        "preprint": False,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "sowhat": None,  # filled in by AI summarizer step
    }


# ---------------------------------------------------------------------------
# bioRxiv helpers
# ---------------------------------------------------------------------------

def biorxiv_fetch(term: str, lookback_days: int, max_results: int) -> list[dict]:
    """Fetch preprints matching term from bioRxiv."""
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=lookback_days)

    url = (
        f"{BIORXIV_BASE}/details/biorxiv/"
        f"{start_date.isoformat()}/{end_date.isoformat()}/0/json"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    collection = data.get("collection", [])
    term_lower = term.lower()
    results = []

    for item in collection:
        text = " ".join([
            item.get("title", ""),
            item.get("abstract", ""),
            item.get("category", ""),
        ]).lower()
        if term_lower in text:
            results.append(parse_biorxiv_item(item))
        if len(results) >= max_results:
            break

    return results


def parse_biorxiv_item(item: dict) -> dict:
    doi = item.get("doi", "")
    authors_raw = item.get("authors", "")
    # bioRxiv returns "Last1 F1; Last2 F2; ..." format
    author_list = [a.strip() for a in authors_raw.split(";") if a.strip()]
    author_str = ", ".join(author_list[:5])
    if len(author_list) > 5:
        author_str += " et al."

    return {
        "source": "biorxiv",
        "pmid": None,
        "title": item.get("title", ""),
        "journal": "bioRxiv (preprint)",
        "authors": author_str,
        "abstract": item.get("abstract", ""),
        "date": item.get("date", ""),
        "doi": doi,
        "mesh_terms": [],
        "preprint": True,
        "url": f"https://doi.org/{doi}" if doi else "",
        "sowhat": None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting publications fetch")

    all_pubs: dict[str, dict] = {}  # keyed by DOI or PMID for deduplication

    # --- PubMed ---
    logging.info("Fetching from PubMed...")
    try:
        query = PUBMED_QUERY.format(days=LOOKBACK_DAYS)
        pmids = pubmed_search(query, PUBMED_MAX)
        logging.info(f"  Found {len(pmids)} PMIDs")
        if pmids:
            time.sleep(0.4)  # respect rate limit
            articles = pubmed_fetch(pmids)
            for a in articles:
                key = a["doi"] or f"pmid:{a['pmid']}"
                if key not in all_pubs:
                    all_pubs[key] = a
            logging.info(f"  Parsed {len(articles)} PubMed articles")
    except requests.RequestException as e:
        logging.error(f"PubMed fetch failed: {e}")

    # --- bioRxiv ---
    logging.info("Fetching from bioRxiv...")
    for term in BIORXIV_SEARCH_TERMS:
        try:
            logging.info(f"  Searching bioRxiv for: '{term}'")
            preprints = biorxiv_fetch(term, LOOKBACK_DAYS, BIORXIV_MAX)
            for p in preprints:
                key = p["doi"] or p["title"][:60]
                if key not in all_pubs:
                    all_pubs[key] = p
            logging.info(f"  Found {len(preprints)} preprints")
            time.sleep(0.5)
        except requests.RequestException as e:
            logging.error(f"bioRxiv fetch failed for '{term}': {e}")

    pubs_list = sorted(
        all_pubs.values(),
        key=lambda p: p["date"] or "",
        reverse=True,
    )

    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(pubs_list),
        "publications": pubs_list,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logging.info(f"Wrote {len(pubs_list)} publications to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
