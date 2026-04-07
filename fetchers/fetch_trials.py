"""
fetch_trials.py
---------------
Fetches in vivo CAR-T (and configurable modality) clinical trials from the
ClinicalTrials.gov v2 API and writes structured JSON to data/trials.json.

Two fetch passes:
  1. Keyword queries (broad in vivo CAR-T terms)
  2. Watchlist sponsor queries (one per company in watchlist.json)

Results are deduplicated by NCT ID.

API docs: https://clinicaltrials.gov/data-api/api
No API key required.
"""

import json
import time
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QUERIES = [
    "in vivo CAR-T",
    "in vivo CAR T cell",
    "lentiviral CAR T in vivo",
    "lipid nanoparticle CAR T",
    "LNP CAR-T",
    "non-viral CAR T",
    "in vivo chimeric antigen receptor",
    "in vivo gene therapy CAR",
]

FIELDS = [
    "NCTId",
    "BriefTitle",
    "OfficialTitle",
    "DetailedDescription",
    "LeadSponsorName",
    "OverallStatus",
    "Phase",
    "Condition",
    "InterventionName",
    "InterventionType",
    "PrimaryOutcomeMeasure",
    "StartDate",
    "LastUpdatePostDate",
    "LocationCountry",
    "EnrollmentCount",
    "BriefSummary",
    "StudyType",
]

BASE_URL   = "https://clinicaltrials.gov/api/v2/studies"
PAGE_SIZE  = 100
OUTPUT_PATH    = Path(__file__).parent.parent / "data" / "trials.json"
WATCHLIST_PATH = Path(__file__).parent.parent / "watchlist.json"

IN_VIVO_SIGNALS = [
    "in vivo car",
    "in vivo chimeric antigen receptor",
    "in vivo generated car",
    "in vivo t cell",
    "in vivo gene therapy to generate",
    "in vivo generation of car",
    "in vivo programming",
    "in vivo reprogramming",
    "lentiviral vector car",
    "lipid nanoparticle car",
    "lnp-delivered car",
    "non-viral car",
    "systemic car delivery",
]

EX_VIVO_SIGNALS = [
    "ex vivo",
    "leukapheresis",
    "autologous car",
    "allogeneic car",
    "manufactured car",
    "cell manufacturing",
]

BISPECIFIC_SIGNALS = [
    "bispecific",
    "t cell engager",
    " tce ",
    "bite ",
    "duobody",
]

CAR_NK_SIGNALS = [
    "car-nk",
    "car nk cell",
    "natural killer car",
]


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
# Modality inference
# ---------------------------------------------------------------------------

def infer_modality(title: str, summary: str, interventions: list[str]) -> str:
    text = " ".join([title, summary] + interventions).lower()
    if any(s in text for s in BISPECIFIC_SIGNALS):
        return "Bispecific TCE"
    if any(s in text for s in CAR_NK_SIGNALS):
        return "CAR-NK"
    if any(s in text for s in IN_VIVO_SIGNALS):
        return "In vivo CAR-T"
    if any(s in text for s in EX_VIVO_SIGNALS):
        return "Ex vivo CAR-T"
    return "Not reported"


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_page(params: dict) -> dict:
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_by_query(query: str) -> list[dict]:
    """Fetch by keyword query term."""
    studies, page_token, page = [], None, 1
    while True:
        logging.info(f"  Keyword '{query}' — page {page}")
        params = {
            "query.term": query,
            "fields": ",".join(FIELDS),
            "pageSize": PAGE_SIZE,
            "format": "json",
        }
        if page_token:
            params["pageToken"] = page_token
        data = fetch_page(params)
        studies.extend(data.get("studies", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
        page += 1
        time.sleep(0.5)
    return studies


def fetch_by_sponsor(sponsor: str) -> list[dict]:
    """Fetch all trials for a specific sponsor name."""
    studies, page_token, page = [], None, 1
    while True:
        params = {
            "query.spons": sponsor,
            "fields": ",".join(FIELDS),
            "pageSize": PAGE_SIZE,
            "format": "json",
        }
        if page_token:
            params["pageToken"] = page_token
        data = fetch_page(params)
        studies.extend(data.get("studies", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
        page += 1
        time.sleep(0.3)
    return studies


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_study(raw: dict) -> dict:
    proto       = raw.get("protocolSection", {})
    id_mod      = proto.get("identificationModule", {})
    status_mod  = proto.get("statusModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    desc_mod    = proto.get("descriptionModule", {})
    design_mod  = proto.get("designModule", {})
    cond_mod    = proto.get("conditionsModule", {})
    interv_mod  = proto.get("armsInterventionsModule", {})
    contacts_mod = proto.get("contactsLocationsModule", {})

    brief_title    = id_mod.get("briefTitle", "") or ""
    official_title = id_mod.get("officialTitle", "") or ""
    title          = brief_title or official_title
    sponsor        = sponsor_mod.get("leadSponsor", {}).get("name", "")
    status         = status_mod.get("overallStatus", "")
    phases         = design_mod.get("phases", [])
    conditions     = cond_mod.get("conditions", [])
    brief_summary  = desc_mod.get("briefSummary", "") or ""
    detailed_desc  = desc_mod.get("detailedDescription", "") or ""
    summary        = brief_summary or detailed_desc
    enrollment     = design_mod.get("enrollmentInfo", {}).get("count", "")
    last_updated   = status_mod.get("lastUpdatePostDateStruct", {}).get("date", "")
    start_date     = status_mod.get("startDateStruct", {}).get("date", "")
    nct_id         = id_mod.get("nctId", "")

    interventions = [i.get("name", "") for i in interv_mod.get("interventions", [])]
    primary_outcomes = [o.get("measure", "") for o in proto.get("outcomesModule", {}).get("primaryOutcomes", [])]
    countries = list({
        loc.get("country", "") for loc in contacts_mod.get("locations", [])
        if loc.get("country")
    })

    modality = infer_modality(
        brief_title + " " + official_title + " " + detailed_desc,
        summary,
        interventions,
    )

    return {
        "nct_id":          nct_id,
        "title":           title,
        "sponsor":         sponsor,
        "modality":        modality,
        "conditions":      conditions,
        "phase":           phases,
        "status":          status,
        "interventions":   interventions,
        "primary_outcomes": primary_outcomes,
        "enrollment":      enrollment,
        "start_date":      start_date,
        "last_updated":    last_updated,
        "countries":       countries,
        "summary":         summary,
        "url":             f"https://clinicaltrials.gov/study/{nct_id}",
        "asset_name":      None,
        "sowhat":          None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting ClinicalTrials.gov fetch")

    all_studies: dict[str, dict] = {}

    # Pass 1 — keyword queries
    logging.info("Pass 1: keyword queries")
    for query in QUERIES:
        try:
            for raw in fetch_by_query(query):
                parsed = parse_study(raw)
                if parsed["nct_id"] and parsed["nct_id"] not in all_studies:
                    all_studies[parsed["nct_id"]] = parsed
        except requests.RequestException as e:
            logging.error(f"Keyword query '{query}' failed: {e}")

    logging.info(f"  After keyword pass: {len(all_studies)} unique trials")

    # Pass 2 — watchlist sponsor queries
    watchlist = load_watchlist()
    logging.info(f"Pass 2: watchlist sponsor queries ({len(watchlist)} companies)")
    new_from_watchlist = 0
    for company in watchlist:
        try:
            raw_studies = fetch_by_sponsor(company)
            for raw in raw_studies:
                parsed = parse_study(raw)
                nct = parsed["nct_id"]
                if nct and nct not in all_studies:
                    all_studies[nct] = parsed
                    new_from_watchlist += 1
            if raw_studies:
                logging.info(f"  {company}: {len(raw_studies)} trials found")
            time.sleep(0.3)
        except requests.RequestException as e:
            logging.error(f"Sponsor query '{company}' failed: {e}")

    logging.info(f"  {new_from_watchlist} new trials added from watchlist pass")

    # Preserve existing sowhat and asset_name from previous run
    existing_path = OUTPUT_PATH
    if existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text())
            for s in existing.get("studies", []):
                nct = s.get("nct_id")
                if nct and nct in all_studies:
                    if s.get("sowhat"):
                        all_studies[nct]["sowhat"] = s["sowhat"]
                    if s.get("asset_name"):
                        all_studies[nct]["asset_name"] = s["asset_name"]
        except Exception as e:
            logging.warning(f"Could not preserve existing data: {e}")

    studies_list = sorted(
        all_studies.values(),
        key=lambda s: s["last_updated"] or "",
        reverse=True,
    )

    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count":      len(studies_list),
        "studies":    studies_list,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logging.info(f"Wrote {len(studies_list)} trials to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
