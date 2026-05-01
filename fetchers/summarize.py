"""
summarize.py
------------
Reads data/trials.json, data/publications.json, data/news.json, and
data/abstracts.json, calls the Anthropic API to generate:
  - "so what" one-liners for trials, publications, news, and abstracts
  - category labels for publications
  - asset names for watchlist company trials
  - conference attendance extraction from news items

Conference attendance pass:
  Scans each news item for mentions of watchlist company presentations at
  named conferences. Stores a `conference_mention` field directly on the
  news item (null = processed, nothing found; dict = match found).
  Assembles matched items into data/conference_appearances.json, grouped
  by conference and sorted chronologically.

  Also maintains an `unmatched_conferences` list in that file for any
  conference names Claude detected that don't match the known calendar —
  these surface as a review banner in the dashboard.

Idempotent — skips items that already have values set.

Requires: ANTHROPIC_API_KEY environment variable
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR                = Path(__file__).parent.parent / "data"
TRIALS_PATH             = DATA_DIR / "trials.json"
PUBS_PATH               = DATA_DIR / "publications.json"
NEWS_PATH               = DATA_DIR / "news.json"
ABSTRACTS_PATH          = DATA_DIR / "abstracts.json"
CONF_APPEARANCES_PATH   = DATA_DIR / "conference_appearances.json"
WATCHLIST_PATH          = Path(__file__).parent.parent / "watchlist.json"

MODEL            = "claude-haiku-4-5-20251001"
MAX_TOKENS       = 120
RATE_LIMIT_DELAY = 0.3

VALID_CATEGORIES = {
    "Clinical data",
    "Preclinical data",
    "Manufacturing / process",
    "Binder optimization",
    "Review article",
}

# ---------------------------------------------------------------------------
# Known conference calendar
# Add new conferences here when flagged by the unmatched_conferences banner.
# Keys are canonical names Claude is instructed to use; values are metadata
# displayed in the dashboard.
# ---------------------------------------------------------------------------
KNOWN_CONFERENCES = {
    "ASGCT 2026":  {"start": "2026-05-13", "end": "2026-05-16", "location": "New Orleans, LA"},
    "ASCO 2026":   {"start": "2026-05-30", "end": "2026-06-03", "location": "Chicago, IL"},
    "EHA 2026":    {"start": "2026-06-11", "end": "2026-06-14", "location": "Milan, Italy"},
    "AACR 2026":   {"start": "2026-04-25", "end": "2026-04-30", "location": "Chicago, IL"},
    "ASH 2026":    {"start": "2026-12-05", "end": "2026-12-08", "location": "Orlando, FL"},
    "ESMO 2026":   {"start": "2026-09-11", "end": "2026-09-15", "location": "TBC"},
    "SITC 2026":   {"start": "2026-11-04", "end": "2026-11-08", "location": "Houston, TX"},
    "AACR 2025":   {"start": "2025-04-25", "end": "2025-04-30", "location": "Chicago, IL"},
    "ASGCT 2025":  {"start": "2025-05-13", "end": "2025-05-17", "location": "New Orleans, LA"},
    "ASH 2025":    {"start": "2025-12-06", "end": "2025-12-09", "location": "Orlando, FL"},
    "ASCO 2025":   {"start": "2025-05-30", "end": "2025-06-03", "location": "Chicago, IL"},
    "EHA 2025":    {"start": "2025-06-12", "end": "2025-06-15", "location": "Milan, Italy"},
    "SITC 2025":   {"start": "2025-11-05", "end": "2025-11-09", "location": "Houston, TX"},
}

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

TRIAL_PROMPT = """\
You are a biotech competitive intelligence analyst specialising in cell therapy.

Given the following clinical trial, write a single punchy sentence (max 20 words)
that captures the strategic "so what" for someone tracking the in vivo CAR-T space.

Rules:
- Return ONLY the sentence, nothing else
- No preamble, no sign-off, no offers to help

Title: {title}
Sponsor: {sponsor}
Modality: {modality}
Conditions: {conditions}
Phase: {phase}
Summary: {summary}
"""

ASSET_NAME_PROMPT = """\
Extract the drug/asset name from this clinical trial record.
The asset name is typically a alphanumeric code like "KLN-1010", "UB-VV111", "CTX131", "CABA-201" etc.

Rules:
- Return ONLY the asset name, nothing else
- If multiple assets are listed, return the primary one
- If no clear asset name exists, return: Not reported
- Do not return generic terms like "CAR-T cells" or "lentiviral vector"

Title: {title}
Interventions: {interventions}
Summary: {summary}
"""

PUB_SOWHAT_PROMPT = """\
You are a biotech competitive intelligence analyst specialising in cell therapy.

Given the following publication abstract, write a single punchy sentence (max 20 words)
capturing the strategic "so what" for someone tracking the in vivo CAR-T space.

Rules:
- Return ONLY the sentence, nothing else
- No preamble, no sign-off, no offers to help
- If the abstract is insufficient, return exactly: Abstract not available

Title: {title}
Journal: {journal}
Preprint: {preprint}
Abstract: {abstract}
"""

PUB_CATEGORY_PROMPT = """\
Classify the following publication into exactly one of these five categories:

  Clinical data               - Reports human trial results, patient outcomes, safety/efficacy data
  Preclinical data            - Reports in vitro or animal study results (mouse, NHP, organoids etc.)
  Manufacturing / process     - Focuses on production methods, delivery vectors, LNP formulation, scale-up
  Binder optimization         - Focuses on CAR construct design, scFv/nanobody engineering, target binding
  Review article              - Review, perspective, commentary, or meta-analysis with no new primary data

Rules:
- Return ONLY the category name, exactly as written above
- No preamble, no explanation, no punctuation

Title: {title}
Abstract: {abstract}
"""

NEWS_PROMPT = """\
You are a biotech competitive intelligence analyst specialising in cell therapy.

Given the following news headline and summary, return a JSON object with two fields:
  "sowhat": a single punchy sentence (max 20 words) capturing the strategic significance
            for someone tracking in vivo CAR-T. If insufficient info, use: "Summary not available"
  "item_type": classify as "funding" if this is primarily about a financing round,
               investment, or grant. Otherwise classify as "news".

Return ONLY valid JSON, no preamble, no markdown.

Source: {source}
Title: {title}
Summary: {summary}
"""


ABSTRACT_PROMPT = """\
You are a biotech competitive intelligence analyst specialising in cell and gene therapy.

Given the following conference abstract, write a single punchy sentence (max 20 words)
capturing the strategic "so what" for someone tracking the in vivo CAR-T space.
Focus on the key finding, the presenting company, and why it matters.

Rules:
- Return ONLY the sentence, nothing else
- No preamble, no sign-off, no offers to help
- If the abstract is insufficient, return exactly: Abstract not available

Conference: {conference}
Title: {title}
Authors: {authors}
Abstract: {abstract}
"""

CONF_MENTION_PROMPT = """\
You are a biotech competitive intelligence analyst.

Read the following news item and determine whether it announces or confirms that
a specific watchlist company will present (oral, poster, or invited talk) at a
named scientific or medical conference.

Watchlist companies (match case-insensitively, partial match is fine):
{watchlist}

Known conference canonical names — use EXACTLY one of these if it matches:
{known_conferences}

Return ONLY a valid JSON object with these exact fields:
  "is_conference_mention": true or false
  "company": the matched watchlist company name (exact casing from the list), or null
  "conference_canonical": the canonical conference name from the known list above, or null if not in list
  "conference_raw": the conference name exactly as mentioned in the article (always set if is_conference_mention is true)
  "presentation_type": one of "oral" | "poster" | "invited" | "unspecified" — or null
  "abstract_title": the specific presentation/abstract title if mentioned, or null
  "conference_date_mentioned": any date string mentioned for the conference, or null

Rules:
- Set is_conference_mention to false if the article is not specifically about a conference presentation
- A general news article that happens to mention a conference in passing does not count
- The article must be announcing or confirming a presentation, abstract acceptance, or speaking slot
- Return ONLY the JSON object, no preamble, no markdown

Source: {source}
Title: {title}
Summary: {summary}
"""

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def call_api(prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    try:
        msg = client.messages.create(
            model=MODEL, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logging.warning(f"API call failed: {e}")
        return ""


UNHELPFUL_PHRASES = [
    "i'd be happy", "i would be happy", "i'm unable", "i am unable",
    "i cannot access", "i can't access", "not able to access", "please provide",
]

def is_unhelpful(text: str) -> bool:
    return not text or any(p in text.lower() for p in UNHELPFUL_PHRASES)


def load_watchlist() -> list[str]:
    try:
        data = json.loads(WATCHLIST_PATH.read_text())
        return [c.lower() for c in data.get("companies", [])]
    except Exception:
        return []


def is_watchlisted(sponsor: str, watchlist: list[str]) -> bool:
    if not sponsor:
        return False
    s = sponsor.lower()
    return any(w in s for w in watchlist)


# ---------------------------------------------------------------------------
# Per-source summarisation
# ---------------------------------------------------------------------------

def summarise_trials(trials_data: dict, watchlist: list[str]) -> tuple[dict, int]:
    updated = 0
    for trial in trials_data.get("studies", []):
        on_watchlist = is_watchlisted(trial.get("sponsor", ""), watchlist)
        needs_sowhat = not trial.get("sowhat")
        needs_asset  = on_watchlist and not trial.get("asset_name")

        if not needs_sowhat and not needs_asset:
            continue

        if needs_sowhat:
            prompt = TRIAL_PROMPT.format(
                title=trial.get("title", ""),
                sponsor=trial.get("sponsor", ""),
                modality=trial.get("modality", ""),
                conditions=", ".join(trial.get("conditions", [])),
                phase=trial.get("phase", ""),
                summary=(trial.get("summary", "") or "")[:600],
            )
            sowhat = call_api(prompt)
            if not is_unhelpful(sowhat):
                trial["sowhat"] = sowhat
            time.sleep(RATE_LIMIT_DELAY)

        if needs_asset:
            prompt = ASSET_NAME_PROMPT.format(
                title=trial.get("title", ""),
                interventions=", ".join(trial.get("interventions", [])),
                summary=(trial.get("summary", "") or "")[:400],
            )
            asset = call_api(prompt, max_tokens=40)
            trial["asset_name"] = asset if asset and asset != "Not reported" else None
            time.sleep(RATE_LIMIT_DELAY)

        updated += 1
        logging.info(f"  [{trial.get('nct_id')}] {trial.get('sowhat','')} | asset: {trial.get('asset_name','—')}")

    return trials_data, updated


def summarise_publications(pubs_data: dict) -> tuple[dict, int]:
    updated = 0
    for pub in pubs_data.get("publications", []):
        abstract = (pub.get("abstract") or "").strip()
        needs_sowhat   = not pub.get("sowhat")
        needs_category = not pub.get("category")

        if not needs_sowhat and not needs_category:
            continue

        if not abstract or len(abstract) < 50:
            if needs_sowhat:   pub["sowhat"]   = "Abstract not available"
            if needs_category: pub["category"] = None
            updated += 1
            continue

        if needs_sowhat:
            prompt = PUB_SOWHAT_PROMPT.format(
                title=pub.get("title", ""),
                journal=pub.get("journal", ""),
                preprint=pub.get("preprint", False),
                abstract=abstract[:800],
            )
            sowhat = call_api(prompt)
            pub["sowhat"] = "Abstract not available" if is_unhelpful(sowhat) else sowhat
            time.sleep(RATE_LIMIT_DELAY)

        if needs_category:
            prompt = PUB_CATEGORY_PROMPT.format(
                title=pub.get("title", ""),
                abstract=abstract[:800],
            )
            result = call_api(prompt, max_tokens=40)
            pub["category"] = result if result in VALID_CATEGORIES else None
            time.sleep(RATE_LIMIT_DELAY)

        updated += 1
        logging.info(f"  [{pub.get('pmid') or pub.get('doi', '')[:20]}] {pub.get('sowhat','')} [{pub.get('category','')}]")

    return pubs_data, updated


def summarise_news(news_data: dict) -> tuple[dict, int]:
    updated = 0
    for item in news_data.get("news", []):
        if item.get("sowhat"):
            continue
        summary = (item.get("summary") or "").strip()
        if not summary or len(summary) < 30:
            item["sowhat"] = "Summary not available"
            updated += 1
            continue
        prompt = NEWS_PROMPT.format(
            source=item.get("source", ""),
            title=item.get("title", ""),
            summary=summary[:600],
        )
        sowhat = call_api(prompt)
        if is_unhelpful(sowhat):
            item["sowhat"] = "Summary not available"
        else:
            item["sowhat"] = sowhat
            updated += 1
            logging.info(f"  [{item.get('source','')}] {sowhat}")
        time.sleep(RATE_LIMIT_DELAY)
    return news_data, updated


def summarise_abstracts(abstracts_data: dict) -> tuple[dict, int]:
    updated = 0
    for abstract in abstracts_data.get("abstracts", []):
        if abstract.get("sowhat"):
            continue
        text = (abstract.get("abstract") or "").strip()
        if not text or len(text) < 50:
            abstract["sowhat"] = "Abstract not available"
            updated += 1
            continue
        prompt = ABSTRACT_PROMPT.format(
            conference=abstract.get("conference") or abstract.get("journal",""),
            title=abstract.get("title",""),
            authors=abstract.get("authors",""),
            abstract=text[:800],
        )
        sowhat = call_api(prompt)
        if is_unhelpful(sowhat):
            abstract["sowhat"] = "Abstract not available"
        else:
            abstract["sowhat"] = sowhat
            updated += 1
            logging.info(f"  [{abstract.get('conference','')}] {sowhat}")
        time.sleep(RATE_LIMIT_DELAY)
    return abstracts_data, updated


# ---------------------------------------------------------------------------
# Conference attendance extraction
# ---------------------------------------------------------------------------

def load_watchlist_raw() -> list[str]:
    """Return watchlist companies with original casing (for display)."""
    try:
        data = json.loads(WATCHLIST_PATH.read_text())
        return data.get("companies", [])
    except Exception:
        return []


def extract_conference_mentions(news_data: dict, watchlist_raw: list[str]) -> tuple[dict, int]:
    """
    Scan news items for conference attendance announcements.

    Adds `conference_mention` field to each item:
      None  → not yet processed
      False → processed, no mention found
      dict  → match found (company, conference, presentation_type, etc.)

    Returns updated news_data and count of newly processed items.
    """
    updated = 0
    known_conf_keys = "\n".join(f"  {k}" for k in sorted(KNOWN_CONFERENCES.keys()))
    watchlist_str   = "\n".join(f"  {c}" for c in watchlist_raw)

    for item in news_data.get("news", []):
        # Skip if already processed (field present, even if False)
        if "conference_mention" in item:
            continue

        title   = (item.get("title")   or "").strip()
        summary = (item.get("summary") or "").strip()

        # Quick pre-filter: skip items with no conference-like signals at all
        combined = (title + " " + summary).lower()
        conf_signals = [
            "present", "abstract", "poster", "oral", "conference",
            "congress", "symposium", "meeting", "summit", "annual",
            "asgct", "aacr", "ash ", "asco", "eha", "esmo", "sitc",
        ]
        if not any(sig in combined for sig in conf_signals):
            item["conference_mention"] = False
            updated += 1
            continue

        prompt = CONF_MENTION_PROMPT.format(
            watchlist=watchlist_str,
            known_conferences=known_conf_keys,
            source=item.get("source", ""),
            title=title,
            summary=summary[:600],
        )

        try:
            raw = call_api(prompt, max_tokens=200)
            if not raw:
                item["conference_mention"] = False
                updated += 1
                continue

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw.strip())

            if result.get("is_conference_mention"):
                item["conference_mention"] = {
                    "company":                result.get("company"),
                    "conference_canonical":   result.get("conference_canonical"),
                    "conference_raw":         result.get("conference_raw"),
                    "presentation_type":      result.get("presentation_type", "unspecified"),
                    "abstract_title":         result.get("abstract_title"),
                    "conference_date_mentioned": result.get("conference_date_mentioned"),
                    "source_url":             item.get("url"),
                    "source_date":            item.get("date"),
                    "extracted_at":           datetime.now(timezone.utc).isoformat(),
                }
                logging.info(
                    f"  [CONF] {result.get('company')} @ {result.get('conference_canonical') or result.get('conference_raw')}"
                    f" ({result.get('presentation_type','?')})"
                )
            else:
                item["conference_mention"] = False

            updated += 1

        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"  Conference extraction failed for '{title[:50]}': {e}")
            item["conference_mention"] = False
            updated += 1

        time.sleep(RATE_LIMIT_DELAY)

    return news_data, updated


def build_conference_appearances(news_data: dict) -> None:
    """
    Assemble conference_appearances.json from news items that have a
    conference_mention dict. Groups by conference, sorts chronologically.
    Flags unmatched conference names for dashboard review banner.
    """
    appearances: list[dict] = []
    unmatched:   list[dict] = []

    for item in news_data.get("news", []):
        mention = item.get("conference_mention")
        if not mention or mention is False:
            continue

        canonical = mention.get("conference_canonical")
        raw       = mention.get("conference_raw") or ""

        conf_meta = KNOWN_CONFERENCES.get(canonical, {}) if canonical else {}

        appearances.append({
            "company":           mention.get("company"),
            "conference":        canonical or raw,
            "conference_start":  conf_meta.get("start"),
            "conference_end":    conf_meta.get("end"),
            "conference_location": conf_meta.get("location"),
            "presentation_type": mention.get("presentation_type", "unspecified"),
            "abstract_title":    mention.get("abstract_title"),
            "source_url":        mention.get("source_url"),
            "source_date":       mention.get("source_date"),
            "extracted_at":      mention.get("extracted_at"),
            "is_known_conference": bool(canonical),
        })

        if not canonical and raw:
            # Flag for review if not already in list
            existing_raws = [u["raw_name"] for u in unmatched]
            if raw not in existing_raws:
                unmatched.append({
                    "raw_name":   raw,
                    "company":    mention.get("company"),
                    "source_url": mention.get("source_url"),
                    "detected_at": mention.get("extracted_at", ""),
                })

    # Sort appearances: known conferences by start date, unknown ones last
    def sort_key(a):
        return a.get("conference_start") or "9999-99-99"

    appearances.sort(key=sort_key)

    output = {
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "count":               len(appearances),
        "appearances":         appearances,
        "unmatched_conferences": unmatched,
    }

    CONF_APPEARANCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONF_APPEARANCES_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logging.info(
        f"Wrote {len(appearances)} conference appearances "
        f"({len(unmatched)} unmatched) to {CONF_APPEARANCES_PATH}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting AI summarisation pass")

    watchlist = load_watchlist()
    logging.info(f"Loaded {len(watchlist)} watchlist companies")

    if TRIALS_PATH.exists():
        logging.info("Summarising trials...")
        data = json.loads(TRIALS_PATH.read_text())
        data, n = summarise_trials(data, watchlist)
        TRIALS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logging.info(f"  Processed {n} trials")
    else:
        logging.warning(f"Trials file not found: {TRIALS_PATH}")

    if PUBS_PATH.exists():
        logging.info("Summarising publications...")
        data = json.loads(PUBS_PATH.read_text())
        data, n = summarise_publications(data)
        PUBS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logging.info(f"  Processed {n} publications")
    else:
        logging.warning(f"Publications file not found: {PUBS_PATH}")

    if NEWS_PATH.exists():
        logging.info("Summarising news...")
        data = json.loads(NEWS_PATH.read_text())
        data, n = summarise_news(data)
        NEWS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logging.info(f"  Processed {n} news items")

        logging.info("Extracting conference attendance mentions...")
        watchlist_raw = load_watchlist_raw()
        data, n = extract_conference_mentions(data, watchlist_raw)
        NEWS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logging.info(f"  Processed {n} items for conference mentions")

        logging.info("Building conference_appearances.json...")
        build_conference_appearances(data)
    else:
        logging.warning(f"News file not found: {NEWS_PATH}")

    if ABSTRACTS_PATH.exists():
        logging.info("Summarising conference abstracts...")
        data = json.loads(ABSTRACTS_PATH.read_text())
        data, n = summarise_abstracts(data)
        ABSTRACTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logging.info(f"  Processed {n} abstracts")
    else:
        logging.warning(f"Abstracts file not found: {ABSTRACTS_PATH}")

    logging.info("Summarisation complete")


if __name__ == "__main__":
    run()
