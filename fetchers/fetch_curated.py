"""
fetch_curated.py
----------------
Manages manually curated entries: re-fetches live data for trials and
publications, then merges all curated entries into the relevant data/*.json
files so they appear in the dashboard alongside automated results.

Run order in pipeline: AFTER all automated fetchers, BEFORE summarize.py
  (summarize.py will then generate sowhat/category for any new curated items
  that don't yet have them.)

Entry types supported:
  trial       — re-fetches from ClinicalTrials.gov v2 by NCT ID (weekly)
  publication — re-fetches from PubMed by PMID (weekly); DOI-only entries
                are injected as static snapshots (no live re-fetch)
  news        — injected as-is; news articles don't update in place

Merge behaviour:
  Curated entries ALWAYS win over automated entries for the same key
  (NCT ID for trials, DOI/PMID for publications, URL for news). This
  means a curated trial will always display the latest data from CT.gov,
  tagged with a 'curated' flag so the dashboard can badge it.

Reads:  data/curated.json  (your manual entries — never auto-overwritten)
Writes: data/curated.json  (updates last_fetched / data / fetch_error)
        data/trials.json   (injects curated trials)
        data/publications.json (injects curated publications)
        data/news.json     (injects curated news/funding)

Requires: no API keys for CT.gov; NCBI_API_KEY optional (higher rate limit)
"""

import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR       = Path(__file__).parent.parent / "data"
CURATED_PATH   = DATA_DIR / "curated.json"
TRIALS_PATH    = DATA_DIR / "trials.json"
PUBS_PATH      = DATA_DIR / "publications.json"
NEWS_PATH      = DATA_DIR / "news.json"

CT_BASE   = "https://clinicaltrials.gov/api/v2/studies"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")

CT_FIELDS = [
    "NCTId", "BriefTitle", "OfficialTitle", "DetailedDescription",
    "LeadSponsorName", "OverallStatus", "Phase", "Condition",
    "InterventionName", "StartDate", "LastUpdatePostDate",
    "LocationCountry", "EnrollmentCount", "BriefSummary", "StudyType",
]

MONTH_MAP = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------

def load_json(path: Path, default) -> dict | list:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logging.warning(f"Could not load {path.name}: {e}")
        return default


def save_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_curated() -> dict:
    data = load_json(CURATED_PATH, {"entries": []})
    if isinstance(data, list):
        data = {"entries": data}
    return data


def load_trials() -> dict:
    raw = load_json(TRIALS_PATH, {"trials": []})
    # Support both dict-of-nct_id and list formats
    if isinstance(raw, dict) and "trials" in raw:
        return raw
    if isinstance(raw, list):
        return {"fetched_at": "", "count": len(raw), "trials": raw}
    return {"fetched_at": "", "count": 0, "trials": []}


def load_pubs() -> dict:
    raw = load_json(PUBS_PATH, {"publications": []})
    if isinstance(raw, dict) and "publications" in raw:
        return raw
    if isinstance(raw, list):
        return {"fetched_at": "", "count": len(raw), "publications": raw}
    return {"fetched_at": "", "count": 0, "publications": []}


def load_news() -> dict:
    raw = load_json(NEWS_PATH, {"items": []})
    if isinstance(raw, dict):
        return raw
    return {"fetched_at": "", "count": 0, "items": raw if isinstance(raw, list) else []}


# ---------------------------------------------------------------------------
# ClinicalTrials.gov re-fetch
# ---------------------------------------------------------------------------

def fetch_trial(nct_id: str) -> dict | None:
    """Fetch a single trial by NCT ID from the CT.gov v2 API."""
    try:
        resp = requests.get(
            f"{CT_BASE}/{nct_id}",
            params={"fields": ",".join(CT_FIELDS), "format": "json"},
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()
    except requests.RequestException as e:
        logging.warning(f"  CT.gov fetch failed for {nct_id}: {e}")
        return None

    proto        = raw.get("protocolSection", {})
    id_mod       = proto.get("identificationModule", {})
    status_mod   = proto.get("statusModule", {})
    sponsor_mod  = proto.get("sponsorCollaboratorsModule", {})
    desc_mod     = proto.get("descriptionModule", {})
    design_mod   = proto.get("designModule", {})
    cond_mod     = proto.get("conditionsModule", {})
    interv_mod   = proto.get("armsInterventionsModule", {})
    contacts_mod = proto.get("contactsLocationsModule", {})

    nct         = id_mod.get("nctId", nct_id)
    title       = id_mod.get("briefTitle") or id_mod.get("officialTitle", "")
    sponsor     = sponsor_mod.get("leadSponsor", {}).get("name", "")
    status      = status_mod.get("overallStatus", "")
    phases      = design_mod.get("phases", [])
    conditions  = cond_mod.get("conditions", [])
    summary     = desc_mod.get("briefSummary") or desc_mod.get("detailedDescription", "")
    enrollment  = design_mod.get("enrollmentInfo", {}).get("count", "")
    start_date  = status_mod.get("startDateStruct", {}).get("date", "")
    last_updated = status_mod.get("lastUpdatePostDateStruct", {}).get("date", "")
    interventions = [i.get("name", "") for i in interv_mod.get("interventions", [])]
    countries   = list({
        loc.get("country", "") for loc in contacts_mod.get("locations", [])
        if loc.get("country")
    })

    return {
        "nct_id":           nct,
        "title":            title,
        "sponsor":          sponsor,
        "modality":         None,       # summarize.py will classify
        "ai_modality":      None,       # summarize.py will classify
        "conditions":       conditions,
        "phase":            phases,
        "status":           status,
        "interventions":    interventions,
        "primary_outcomes": [],
        "enrollment":       enrollment,
        "start_date":       start_date,
        "last_updated":     last_updated,
        "countries":        countries,
        "summary":          summary,
        "url":              f"https://clinicaltrials.gov/study/{nct}",
        "asset_name":       None,
        "sowhat":           None,
        "curated":          True,
    }


# ---------------------------------------------------------------------------
# PubMed re-fetch
# ---------------------------------------------------------------------------

def fetch_pubmed_by_pmid(pmid: str) -> dict | None:
    """Fetch a single PubMed article by PMID."""
    params = {
        "db": "pubmed", "id": pmid,
        "retmode": "xml", "rettype": "abstract",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    try:
        resp = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logging.warning(f"  PubMed fetch failed for PMID {pmid}: {e}")
        return None

    root = ET.fromstring(resp.text)
    article = root.find(".//PubmedArticle")
    if article is None:
        logging.warning(f"  No article found for PMID {pmid}")
        return None

    def txt(path, default=""):
        el = article.find(path)
        return "".join(el.itertext()).strip() if el is not None else default

    title    = txt(".//ArticleTitle")
    journal  = txt(".//Journal/Title")
    abstract = txt(".//AbstractText")

    authors = []
    for author in article.findall(".//Author"):
        ln  = author.find("LastName")
        ini = author.find("Initials")
        last = "".join(ln.itertext()).strip() if ln is not None else ""
        initials = "".join(ini.itertext()).strip() if ini is not None else ""
        if last:
            authors.append(f"{last} {initials}".strip())
    author_str = ", ".join(authors[:5])
    if len(authors) > 5:
        author_str += " et al."

    year  = txt(".//PubDate/Year") or (txt(".//PubDate/MedlineDate") or "")[:4]
    month = txt(".//PubDate/Month") or ""
    month_num = MONTH_MAP.get(month, month)
    date_str  = f"{year}-{month_num}" if month_num else year

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
        "curated":   True,
    }


# ---------------------------------------------------------------------------
# Active entry filter
# ---------------------------------------------------------------------------

def active_entries(curated: dict) -> list[dict]:
    """Return entries that are not disabled."""
    return [e for e in curated.get("entries", []) if not e.get("_disabled")]


# ---------------------------------------------------------------------------
# Per-type re-fetch
# ---------------------------------------------------------------------------

def refresh_trial_entry(entry: dict) -> dict:
    nct_id = entry.get("nct_id", "").strip()
    if not nct_id:
        entry["fetch_error"] = "Missing nct_id"
        return entry

    logging.info(f"  Re-fetching trial {nct_id} ({entry.get('company', '?')})")
    data = fetch_trial(nct_id)
    now  = datetime.now(timezone.utc).isoformat()

    if data:
        # Preserve curated metadata: override sponsor with curated company name
        # only if the API returned nothing (CT.gov is authoritative for sponsor)
        data["curated"]      = True
        data["curated_notes"] = entry.get("notes")
        data["curated_id"]   = entry.get("id")
        entry["data"]        = data
        entry["last_fetched"] = now
        entry["fetch_error"] = None
    else:
        entry["fetch_error"] = f"CT.gov returned no data at {now}"

    return entry


def refresh_publication_entry(entry: dict) -> dict:
    pmid = str(entry.get("pmid", "")).strip()
    doi  = str(entry.get("doi", "")).strip()

    if not pmid:
        # DOI-only: inject as static if we have a data snapshot, else warn
        if not entry.get("data"):
            entry["fetch_error"] = "No PMID and no existing data snapshot — cannot re-fetch"
        return entry

    logging.info(f"  Re-fetching publication PMID {pmid} ({entry.get('company', '?')})")
    data = fetch_pubmed_by_pmid(pmid)
    now  = datetime.now(timezone.utc).isoformat()

    if data:
        # Fill in doi from curated entry if PubMed didn't return one
        if not data.get("doi") and doi:
            data["doi"] = doi
        data["curated"]       = True
        data["curated_notes"] = entry.get("notes")
        data["curated_id"]    = entry.get("id")
        entry["data"]         = data
        entry["last_fetched"] = now
        entry["fetch_error"]  = None
    else:
        entry["fetch_error"] = f"PubMed returned no data at {now}"

    return entry


def refresh_news_entry(entry: dict) -> dict:
    """News entries are static snapshots — no live re-fetch. Ensure data block exists."""
    if not entry.get("data"):
        # Build a minimal news record from the entry fields
        entry["data"] = {
            "source":      entry.get("source", "Manual"),
            "title":       entry.get("title", ""),
            "summary":     entry.get("notes", ""),
            "url":         entry.get("url", ""),
            "date":        (entry.get("added_at", "") or "")[:10],
            "tags":        [],
            "sowhat":      None,
            "item_type":   entry.get("item_type", "news"),
            "fetch_source": "curated",
            "curated":     True,
            "curated_notes": entry.get("notes"),
            "curated_id":  entry.get("id"),
        }
    return entry


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def merge_trials(trials_store: dict, curated_trials: list[dict]) -> dict:
    """
    Inject curated trial data records into trials_store.
    Curated entries win over automated entries for the same NCT ID.
    """
    # Build index: nct_id -> position in list
    trial_list = trials_store.get("trials", [])
    index = {t.get("nct_id"): i for i, t in enumerate(trial_list)}

    injected = 0
    for entry in curated_trials:
        data = entry.get("data")
        if not data or not data.get("nct_id"):
            continue
        nct = data["nct_id"]
        if nct in index:
            # Preserve existing sowhat/ai_modality if curated record doesn't have them
            existing = trial_list[index[nct]]
            data.setdefault("sowhat",      existing.get("sowhat"))
            data.setdefault("ai_modality", existing.get("ai_modality"))
            data.setdefault("asset_name",  existing.get("asset_name"))
            trial_list[index[nct]] = data
            logging.info(f"  Updated existing trial {nct} with curated data")
        else:
            trial_list.append(data)
            index[nct] = len(trial_list) - 1
            injected += 1
            logging.info(f"  Injected new curated trial {nct}")

    trials_store["trials"] = trial_list
    trials_store["count"]  = len(trial_list)
    return trials_store


def merge_publications(pubs_store: dict, curated_pubs: list[dict]) -> dict:
    """
    Inject curated publication data records into pubs_store.
    Key: doi if present, else pmid.
    """
    pub_list = pubs_store.get("publications", [])
    # Build index by doi and pmid
    doi_index  = {p.get("doi"):  i for i, p in enumerate(pub_list) if p.get("doi")}
    pmid_index = {str(p.get("pmid")): i for i, p in enumerate(pub_list) if p.get("pmid")}

    for entry in curated_pubs:
        data = entry.get("data")
        if not data:
            continue
        doi  = data.get("doi", "")
        pmid = str(data.get("pmid", ""))

        existing_idx = doi_index.get(doi) if doi else None
        if existing_idx is None:
            existing_idx = pmid_index.get(pmid) if pmid else None

        if existing_idx is not None:
            existing = pub_list[existing_idx]
            data.setdefault("sowhat",   existing.get("sowhat"))
            data.setdefault("category", existing.get("category"))
            pub_list[existing_idx] = data
            logging.info(f"  Updated existing pub (doi={doi} pmid={pmid})")
        else:
            pub_list.append(data)
            if doi:
                doi_index[doi] = len(pub_list) - 1
            if pmid:
                pmid_index[pmid] = len(pub_list) - 1
            logging.info(f"  Injected new curated pub (doi={doi} pmid={pmid})")

    pubs_store["publications"] = pub_list
    pubs_store["count"]        = len(pub_list)
    return pubs_store


def merge_news(news_store: dict, curated_news: list[dict]) -> dict:
    """
    Inject curated news/funding data records into news_store.
    Key: url.
    """
    # news.json uses either a top-level "items" or "news" list
    item_key = "items" if "items" in news_store else "news"
    news_list = news_store.get(item_key, [])
    url_index = {n.get("url"): i for i, n in enumerate(news_list) if n.get("url")}

    for entry in curated_news:
        data = entry.get("data")
        if not data or not data.get("url"):
            continue
        url = data["url"]
        if url in url_index:
            # News articles don't update — preserve existing sowhat
            existing = news_list[url_index[url]]
            data.setdefault("sowhat", existing.get("sowhat"))
            news_list[url_index[url]] = data
        else:
            news_list.append(data)
            url_index[url] = len(news_list) - 1
            logging.info(f"  Injected new curated news: {data.get('title', url)[:60]}")

    news_store[item_key] = news_list
    news_store["count"]  = len(news_list)
    return news_store


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting curated entry refresh")

    curated = load_curated()
    entries = active_entries(curated)

    if not entries:
        logging.info("No active curated entries — nothing to do")
        return

    logging.info(f"Found {len(entries)} active entries")

    # Separate by type
    trial_entries = [e for e in entries if e.get("type") == "trial"]
    pub_entries   = [e for e in entries if e.get("type") == "publication"]
    news_entries  = [e for e in entries if e.get("type") == "news"]

    # ── Re-fetch trials ───────────────────────────────────────────────────────
    logging.info(f"Re-fetching {len(trial_entries)} trial(s)")
    for i, entry in enumerate(trial_entries):
        entry = refresh_trial_entry(entry)
        # Update in the original list (entries are dicts, so mutated in-place,
        # but be explicit for clarity)
        for j, e in enumerate(curated["entries"]):
            if e.get("id") == entry.get("id"):
                curated["entries"][j] = entry
        if i < len(trial_entries) - 1:
            time.sleep(0.5)  # be polite to CT.gov

    # ── Re-fetch publications ─────────────────────────────────────────────────
    logging.info(f"Re-fetching {len(pub_entries)} publication(s)")
    for i, entry in enumerate(pub_entries):
        entry = refresh_publication_entry(entry)
        for j, e in enumerate(curated["entries"]):
            if e.get("id") == entry.get("id"):
                curated["entries"][j] = entry
        if i < len(pub_entries) - 1:
            time.sleep(0.4)

    # ── Prepare news entries (static — no network call) ───────────────────────
    logging.info(f"Preparing {len(news_entries)} news/funding entry(ies)")
    for entry in news_entries:
        entry = refresh_news_entry(entry)
        for j, e in enumerate(curated["entries"]):
            if e.get("id") == entry.get("id"):
                curated["entries"][j] = entry

    # ── Save updated curated.json ─────────────────────────────────────────────
    save_json(CURATED_PATH, curated)
    logging.info(f"Saved updated curated.json")

    # ── Merge into data stores ────────────────────────────────────────────────
    if trial_entries:
        logging.info("Merging curated trials into trials.json")
        trials_store = load_trials()
        trials_store = merge_trials(trials_store, trial_entries)
        save_json(TRIALS_PATH, trials_store)

    if pub_entries:
        logging.info("Merging curated publications into publications.json")
        pubs_store = load_pubs()
        pubs_store = merge_publications(pubs_store, pub_entries)
        save_json(PUBS_PATH, pubs_store)

    if news_entries:
        logging.info("Merging curated news into news.json")
        news_store = load_news()
        news_store = merge_news(news_store, news_entries)
        save_json(NEWS_PATH, news_store)

    # ── Summary ───────────────────────────────────────────────────────────────
    errors = [
        f"  [{e.get('id')}] {e.get('fetch_error')}"
        for e in curated.get("entries", [])
        if e.get("fetch_error") and not e.get("_disabled")
    ]
    if errors:
        logging.warning(f"{len(errors)} entries had fetch errors:")
        for msg in errors:
            logging.warning(msg)

    logging.info("Curated entry refresh complete")


if __name__ == "__main__":
    run()
