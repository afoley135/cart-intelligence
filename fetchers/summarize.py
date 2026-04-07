"""
summarize.py
------------
Reads data/trials.json, data/publications.json, and data/news.json,
calls the Anthropic API to generate:
  - "so what" one-liners for trials, publications, and news
  - category labels for publications
  - asset names for watchlist company trials

Idempotent — skips items that already have values set.

Requires: ANTHROPIC_API_KEY environment variable
"""

import json
import logging
import os
import time
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR      = Path(__file__).parent.parent / "data"
TRIALS_PATH   = DATA_DIR / "trials.json"
PUBS_PATH     = DATA_DIR / "publications.json"
NEWS_PATH     = DATA_DIR / "news.json"
WATCHLIST_PATH = Path(__file__).parent.parent / "watchlist.json"

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

Given the following news headline and summary, write a single punchy sentence (max 20 words)
capturing the strategic "so what" for someone tracking the in vivo CAR-T space.

Rules:
- Return ONLY the sentence, nothing else
- No preamble, no sign-off, no offers to help
- If the summary is insufficient, return exactly: Summary not available

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
    else:
        logging.warning(f"News file not found: {NEWS_PATH}")

    logging.info("Summarisation complete")


if __name__ == "__main__":
    run()
