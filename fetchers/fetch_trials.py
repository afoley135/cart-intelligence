"""
fetch_trials.py
---------------
Fetches in vivo CAR-T (and configurable modality) clinical trials from the
ClinicalTrials.gov v2 API and writes structured JSON to data/trials.json.

Two fetch passes:
  1. Keyword queries (broad in vivo CAR-T terms)
  2. Watchlist sponsor queries (one per company in watchlist.json)

Results are deduplicated by NCT ID.

Change detection:
  Before overwriting trials.json, diffs the previous state on fields that
  matter: status, phase, enrollment, start_date, title.
  Writes new and changed watchlist-company trials to data/trial_changes.json.
  The email generator reads trial_changes.json; the dashboard Changes tab
  reads website_changes.json (separate).

API docs: https://clinicaltrials.gov/data-api/api
No API key required.
"""

import json
import os
import time
import logging
import requests
import anthropic
from datetime import datetime, timezone, timedelta
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

BASE_URL       = "https://clinicaltrials.gov/api/v2/studies"
PAGE_SIZE      = 100
OUTPUT_PATH    = Path(__file__).parent.parent / "data" / "trials.json"
CHANGES_PATH   = Path(__file__).parent.parent / "data" / "trial_changes.json"
WATCHLIST_PATH = Path(__file__).parent.parent / "watchlist.json"

# Only flag a trial as "new" if first_posted is within this many days.
# Prevents a flood of stale "new" listings after trials.json is reset.
NEW_TRIAL_LOOKBACK_DAYS = 30

# Changes older than this are pruned from trial_changes.json on each run.
CHANGE_RETENTION_DAYS = 7

# Fields we diff between runs to detect meaningful changes
# (last_updated alone isn't enough — CT.gov updates it for minor admin changes)
DIFF_FIELDS = ["status", "phase", "enrollment", "start_date", "title"]

# Claude classification config
CLASSIFICATION_MODEL = "claude-haiku-4-5-20251001"
VALID_MODALITIES = {
    "In vivo CAR-T — LNP",
    "In vivo CAR-T — Viral vector",
    "In vivo CAR-T — Other",
    "Ex vivo CAR-T — Autologous",
    "Ex vivo CAR-T — Allogeneic",
    "Bispecific TCE",
    "CAR-NK",
    "Not reported",
}

CLASSIFICATION_PROMPT = """You are a cell therapy expert. Classify this clinical trial into exactly one category.

Categories:
  In vivo CAR-T — LNP          : CAR-T generated inside the patient using lipid nanoparticle mRNA delivery
  In vivo CAR-T — Viral vector  : CAR-T generated inside the patient using viral vector (lentiviral, AAV, etc.)
  In vivo CAR-T — Other         : CAR-T generated inside the patient using other or unspecified delivery
  Ex vivo CAR-T — Autologous    : CAR-T manufactured outside the body from the patient's own cells
  Ex vivo CAR-T — Allogeneic    : CAR-T manufactured outside the body from donor cells
  Bispecific TCE                : Bispecific antibody or T cell engager (not CAR-T)
  CAR-NK                        : CAR natural killer cell therapy
  Not reported                  : Insufficient information to classify

Rules:
- Return ONLY the category name exactly as written above
- No explanation, no punctuation, nothing else

Title: {title}
Official title: {official_title}
Sponsor: {sponsor}
Interventions: {interventions}
Summary: {summary}
"""

IN_VIVO_SIGNALS = [
    "in vivo car", "in vivo chimeric antigen receptor", "in vivo generated car",
    "in vivo t cell", "in vivo gene therapy to generate", "in vivo generation of car",
    "in vivo programming", "in vivo reprogramming", "lentiviral vector car",
    "lipid nanoparticle car", "lnp-delivered car", "non-viral car",
    "systemic car delivery",
]

EX_VIVO_SIGNALS = [
    "ex vivo", "leukapheresis", "autologous car", "allogeneic car",
    "manufactured car", "cell manufacturing",
]

BISPECIFIC_SIGNALS = ["bispecific", "t cell engager", " tce ", "bite ", "duobody"]
CAR_NK_SIGNALS     = ["car-nk", "car nk cell", "natural killer car"]


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


def is_watchlisted(sponsor: str, watchlist_lower: list[str]) -> bool:
    if not sponsor:
        return False
    s = sponsor.lower()
    return any(w in s for w in watchlist_lower)


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
    proto        = raw.get("protocolSection", {})
    id_mod       = proto.get("identificationModule", {})
    status_mod   = proto.get("statusModule", {})
    sponsor_mod  = proto.get("sponsorCollaboratorsModule", {})
    desc_mod     = proto.get("descriptionModule", {})
    design_mod   = proto.get("designModule", {})
    cond_mod     = proto.get("conditionsModule", {})
    interv_mod   = proto.get("armsInterventionsModule", {})
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
    first_posted   = status_mod.get("firstPostDateStruct", {}).get("date", "")
    nct_id         = id_mod.get("nctId", "")

    interventions    = [i.get("name", "") for i in interv_mod.get("interventions", [])]
    primary_outcomes = [o.get("measure", "") for o in proto.get("outcomesModule", {}).get("primaryOutcomes", [])]
    countries        = list({
        loc.get("country", "") for loc in contacts_mod.get("locations", [])
        if loc.get("country")
    })

    modality = infer_modality(
        brief_title + " " + official_title + " " + detailed_desc,
        summary,
        interventions,
    )

    return {
        "nct_id":           nct_id,
        "title":            title,
        "sponsor":          sponsor,
        "modality":         modality,
        "conditions":       conditions,
        "phase":            phases,
        "status":           status,
        "interventions":    interventions,
        "primary_outcomes": primary_outcomes,
        "enrollment":       enrollment,
        "start_date":       start_date,
        "first_posted":     first_posted,
        "last_updated":     last_updated,
        "countries":        countries,
        "summary":          summary,
        "url":              f"https://clinicaltrials.gov/study/{nct_id}",
        "asset_name":       None,
        "sowhat":           None,
    }


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def snapshot_trial(trial: dict) -> dict:
    """Extract only the fields we care about diffing."""
    return {
        "status":     trial.get("status", ""),
        "phase":      trial.get("phase", []),
        "enrollment": trial.get("enrollment", ""),
        "start_date": trial.get("start_date", ""),
        "title":      trial.get("title", ""),
    }


def diff_trials(
    new_studies: dict[str, dict],
    old_studies: dict[str, dict],
    watchlist_lower: list[str],
) -> list[dict]:
    """
    Compare new vs old trial state for watchlist companies only.
    Returns a list of change records for trial_changes.json.

    Change types:
      - new_trial       : NCT ID not seen before
      - status_change   : OverallStatus changed (highest signal)
      - phase_change    : Phase changed
      - enrollment_change: Enrollment count changed
      - start_date_set  : Start date newly populated
    """
    changes = []
    detected_at = datetime.now(timezone.utc).isoformat()

    for nct_id, trial in new_studies.items():
        if not is_watchlisted(trial.get("sponsor", ""), watchlist_lower):
            continue

        old = old_studies.get(nct_id)

        if old is None:
            # Only flag as new if recently posted — prevents flood after trials.json reset
            first_posted = trial.get("first_posted", "")
            cutoff = (datetime.now(timezone.utc) - timedelta(days=NEW_TRIAL_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
            if first_posted < cutoff:
                continue  # old trial appearing for first time due to reset — skip
            # Skip completed trials — not actionable as new listings
            if trial.get("status", "").upper() == "COMPLETED":
                continue
            changes.append({
                "nct_id":      nct_id,
                "change_type": "new_trial",
                "sponsor":     trial.get("sponsor", ""),
                "title":       trial.get("title", ""),
                "asset_name":  trial.get("asset_name"),
                "status":      trial.get("status", ""),
                "phase":       trial.get("phase", []),
                "conditions":  trial.get("conditions", []),
                "url":         trial.get("url", ""),
                "detected_at": detected_at,
                "reviewed":    False,
            })
            continue

        # Existing trial — diff meaningful fields
        old_snap = snapshot_trial(old)
        new_snap = snapshot_trial(trial)

        if old_snap["status"] != new_snap["status"]:
            changes.append({
                "nct_id":      nct_id,
                "change_type": "status_change",
                "sponsor":     trial.get("sponsor", ""),
                "title":       trial.get("title", ""),
                "asset_name":  trial.get("asset_name") or old.get("asset_name"),
                "status_old":  old_snap["status"],
                "status_new":  new_snap["status"],
                "url":         trial.get("url", ""),
                "detected_at": detected_at,
                "reviewed":    False,
            })

        # Normalise phase lists before comparing — CT.gov sometimes reorders them
        old_phase = sorted(p.upper() for p in (old_snap["phase"] or []))
        new_phase = sorted(p.upper() for p in (new_snap["phase"] or []))
        if old_phase != new_phase and new_phase:
            changes.append({
                "nct_id":      nct_id,
                "change_type": "phase_change",
                "sponsor":     trial.get("sponsor", ""),
                "title":       trial.get("title", ""),
                "asset_name":  trial.get("asset_name") or old.get("asset_name"),
                "phase_old":   old_snap["phase"],
                "phase_new":   new_snap["phase"],
                "url":         trial.get("url", ""),
                "detected_at": detected_at,
                "reviewed":    False,
            })

        if (
            not old_snap["start_date"]
            and new_snap["start_date"]
        ):
            changes.append({
                "nct_id":      nct_id,
                "change_type": "start_date_set",
                "sponsor":     trial.get("sponsor", ""),
                "title":       trial.get("title", ""),
                "asset_name":  trial.get("asset_name") or old.get("asset_name"),
                "start_date":  new_snap["start_date"],
                "status":      new_snap["status"],
                "url":         trial.get("url", ""),
                "detected_at": detected_at,
                "reviewed":    False,
            })

    return changes


def load_existing_changes() -> list[dict]:
    try:
        if CHANGES_PATH.exists():
            return json.loads(CHANGES_PATH.read_text()).get("changes", [])
    except Exception as e:
        logging.warning(f"Could not load existing trial changes: {e}")
    return []


def save_changes(all_changes: list[dict]) -> None:
    # Prune changes older than CHANGE_RETENTION_DAYS — no manual review needed
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CHANGE_RETENTION_DAYS)).isoformat()
    all_changes = [c for c in all_changes if (c.get("detected_at") or "") >= cutoff]

    # Sort newest first for dashboard display
    all_changes = sorted(all_changes, key=lambda c: c.get("detected_at") or "", reverse=True)

    output = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "count":      len(all_changes),
        "changes":    all_changes,
    }
    CHANGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHANGES_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logging.info(f"Wrote {len(all_changes)} trial changes to {CHANGES_PATH}")


# ---------------------------------------------------------------------------
# Claude classification
# ---------------------------------------------------------------------------

def classify_modality_with_claude(trial: dict) -> str:
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return trial.get("modality", "Not reported")
        client = anthropic.Anthropic(api_key=api_key)
        prompt = CLASSIFICATION_PROMPT.format(
            title=trial.get("title", ""),
            official_title=trial.get("title", ""),
            sponsor=trial.get("sponsor", ""),
            interventions=", ".join(trial.get("interventions", [])[:5]),
            summary=(trial.get("summary", "") or "")[:600],
        )
        msg = client.messages.create(
            model=CLASSIFICATION_MODEL,
            max_tokens=30,
            messages=[{"role": "user", "content": prompt}],
        )
        result = msg.content[0].text.strip()
        return result if result in VALID_MODALITIES else "Not reported"
    except Exception as e:
        logging.warning(f"  Claude classification failed: {e}")
        return trial.get("modality", "Not reported")


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
    watchlist_lower = [w.lower() for w in watchlist]
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

    # Load previous state BEFORE overwriting — used for both preservation and diffing
    old_studies: dict[str, dict] = {}
    if OUTPUT_PATH.exists():
        try:
            existing = json.loads(OUTPUT_PATH.read_text())
            for s in existing.get("studies", []):
                nct = s.get("nct_id")
                if nct:
                    old_studies[nct] = s
            logging.info(f"  Loaded {len(old_studies)} existing trials for diffing")
        except Exception as e:
            logging.warning(f"Could not load existing trials: {e}")

    # Preserve sowhat, asset_name from previous run
    for nct, old in old_studies.items():
        if nct in all_studies:
            if old.get("sowhat"):
                all_studies[nct]["sowhat"] = old["sowhat"]
            if old.get("asset_name"):
                all_studies[nct]["asset_name"] = old["asset_name"]

    # Pass 3 — Claude classification for watchlist company trials
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        needs_classification = [
            nct for nct, s in all_studies.items()
            if not s.get("ai_modality")
            and is_watchlisted(s.get("sponsor", ""), watchlist_lower)
        ]
        logging.info(f"Pass 3: Claude classification for {len(needs_classification)} watchlist trials")
        for nct in needs_classification:
            trial = all_studies[nct]
            ai_modality = classify_modality_with_claude(trial)
            all_studies[nct]["ai_modality"] = ai_modality
            logging.info(f"  {nct} [{trial.get('sponsor','')}]: {ai_modality}")
            time.sleep(0.2)
    else:
        logging.warning("Pass 3 skipped: ANTHROPIC_API_KEY not set")

    # Preserve cached ai_modality from previous run
    for nct, old in old_studies.items():
        if nct in all_studies and old.get("ai_modality"):
            if not all_studies[nct].get("ai_modality"):
                all_studies[nct]["ai_modality"] = old["ai_modality"]

    # ── Change detection ──────────────────────────────────────────────────────
    logging.info("Detecting trial changes for watchlist companies...")
    new_changes = diff_trials(all_studies, old_studies, watchlist_lower)
    logging.info(f"  {len(new_changes)} new changes detected")
    for c in new_changes:
        if c["change_type"] == "new_trial":
            logging.info(f"  NEW: [{c['sponsor']}] {c['title'][:60]}")
        elif c["change_type"] == "status_change":
            logging.info(f"  STATUS: [{c['sponsor']}] {c['status_old']} → {c['status_new']}")
        elif c["change_type"] == "phase_change":
            logging.info(f"  PHASE: [{c['sponsor']}] {c['phase_old']} → {c['phase_new']}")
        elif c["change_type"] == "start_date_set":
            logging.info(f"  START DATE: [{c['sponsor']}] {c['start_date']}")

    # Merge with existing unreviewed changes (keep history)
    existing_changes = load_existing_changes()
    # Avoid duplicating: drop old entries for the same nct_id + change_type
    # that were detected today (idempotent re-runs)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_keys = {(c["nct_id"], c["change_type"]) for c in new_changes}
    filtered_existing = [
        c for c in existing_changes
        if not (
            (c["nct_id"], c["change_type"]) in new_keys
            and c.get("detected_at", "")[:10] == today
        )
    ]
    all_changes = filtered_existing + new_changes
    save_changes(all_changes)

    # Exclude trials last updated more than 2 years ago
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=730)).strftime("%Y-%m-%d")
    before_filter = len(all_studies)
    all_studies = {
        nct: s for nct, s in all_studies.items()
        if (s.get("last_updated") or "") >= cutoff_date
    }
    logging.info(f"  Excluded {before_filter - len(all_studies)} trials updated before {cutoff_date}")

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
