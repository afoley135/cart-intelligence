"""
fetch_trials.py
---------------
Fetches in vivo CAR-T (and configurable modality) clinical trials from the
ClinicalTrials.gov v2 API and writes structured JSON to data/trials.json.

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
# Configuration — edit these to expand or narrow the search
# ---------------------------------------------------------------------------

# Each entry is a separate API query. Results are deduplicated by NCT ID.
QUERIES = [
    "in vivo CAR-T",
    "in vivo CAR T cell",
    "lentiviral CAR T in vivo",
    "lipid nanoparticle CAR T",
    "LNP CAR-T",
]

# Optional: restrict to specific conditions (leave empty to include all)
CONDITIONS = []  # e.g. ["lymphoma", "leukemia", "multiple myeloma"]

# Fields to retrieve from the API
FIELDS = [
    "NCTId",
    "BriefTitle",
    "OfficialTitle",
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

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
PAGE_SIZE = 100
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "trials.json"

# ---------------------------------------------------------------------------
# Modality inference
# Heuristic — Claude can be used to improve this post-fetch if needed
# ---------------------------------------------------------------------------

# Only include signals that are unambiguous explicit mentions
IN_VIVO_SIGNALS = [
    "in vivo car",
    "in vivo chimeric antigen receptor",
    "in vivo generated car",
    "in vivo t cell",
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


def infer_modality(title: str, summary: str, interventions: list[str]) -> str:
    text = " ".join([title, summary] + interventions).lower()

    # Bispecific and CAR-NK are usually explicitly named
    if any(s in text for s in BISPECIFIC_SIGNALS):
        return "Bispecific TCE"
    if any(s in text for s in CAR_NK_SIGNALS):
        return "CAR-NK"

    # In vivo takes priority — if mentioned at all, classify as in vivo
    # (ex vivo trials almost never reference in vivo CAR-T)
    if any(s in text for s in IN_VIVO_SIGNALS):
        return "In vivo CAR-T"

    # Only classify as ex vivo if no in vivo signal present
    if any(s in text for s in EX_VIVO_SIGNALS):
        return "Ex vivo CAR-T"

    return "Not reported"

# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_page(query: str, page_token: str | None = None) -> dict:
    params = {
        "query.term": query,
        "fields": ",".join(FIELDS),
        "pageSize": PAGE_SIZE,
        "format": "json",
    }
    if page_token:
        params["pageToken"] = page_token
    if CONDITIONS:
        params["query.cond"] = " OR ".join(CONDITIONS)

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_all_for_query(query: str) -> list[dict]:
    studies = []
    page_token = None
    page = 1
    while True:
        logging.info(f"  Query '{query}' — page {page}")
        data = fetch_page(query, page_token)
        studies.extend(data.get("studies", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
        page += 1
        time.sleep(0.5)  # be polite to the API
    return studies


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def extract_field(proto: dict, *keys: str, default="") -> str:
    """Walk nested protocol dict safely."""
    node = proto
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, {})
    return node if isinstance(node, str) else default


def parse_study(raw: dict) -> dict:
    proto = raw.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    desc_mod = proto.get("descriptionModule", {})
    design_mod = proto.get("designModule", {})
    conditions_mod = proto.get("conditionsModule", {})
    interventions_mod = proto.get("armsInterventionsModule", {})
    outcomes_mod = proto.get("outcomesModule", {})
    contacts_mod = proto.get("contactsLocationsModule", {})

    nct_id = id_mod.get("nctId", "")
    title = id_mod.get("briefTitle", "") or id_mod.get("officialTitle", "")
    sponsor = sponsor_mod.get("leadSponsor", {}).get("name", "")
    status = status_mod.get("overallStatus", "")
    phases = design_mod.get("phases", [])
    phase = " / ".join(phases) if phases else "N/A"
    conditions = conditions_mod.get("conditions", [])
    summary = desc_mod.get("briefSummary", "")
    enrollment = design_mod.get("enrollmentInfo", {}).get("count", "")
    last_updated = status_mod.get("lastUpdatePostDateStruct", {}).get("date", "")
    start_date = status_mod.get("startDateStruct", {}).get("date", "")

    interventions = [
        i.get("name", "") for i in
        interventions_mod.get("interventions", [])
    ]

    primary_outcomes = [
        o.get("measure", "") for o in
        outcomes_mod.get("primaryOutcomes", [])
    ]

    countries = list({
        loc.get("country", "") for loc in
        contacts_mod.get("locations", [])
        if loc.get("country")
    })

    modality = infer_modality(title, summary, interventions)

    return {
        "nct_id": nct_id,
        "title": title,
        "sponsor": sponsor,
        "modality": modality,
        "conditions": conditions,
        "phase": phase,
        "status": status,
        "interventions": interventions,
        "primary_outcomes": primary_outcomes,
        "enrollment": enrollment,
        "start_date": start_date,
        "last_updated": last_updated,
        "countries": countries,
        "summary": summary,
        "url": f"https://clinicaltrials.gov/study/{nct_id}",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting ClinicalTrials.gov fetch")

    all_studies: dict[str, dict] = {}  # keyed by NCT ID for deduplication

    for query in QUERIES:
        logging.info(f"Querying: '{query}'")
        try:
            raw_studies = fetch_all_for_query(query)
            for raw in raw_studies:
                parsed = parse_study(raw)
                nct = parsed["nct_id"]
                if nct and nct not in all_studies:
                    all_studies[nct] = parsed
        except requests.RequestException as e:
            logging.error(f"Failed query '{query}': {e}")

    studies_list = sorted(
        all_studies.values(),
        key=lambda s: s["last_updated"] or "",
        reverse=True
    )

    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(studies_list),
        "studies": studies_list,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logging.info(f"Wrote {len(studies_list)} trials to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
