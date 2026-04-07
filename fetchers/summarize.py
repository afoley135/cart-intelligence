"""
summarize.py
------------
Reads data/trials.json, data/publications.json, and data/news.json,
calls the Anthropic API to generate a "so what" one-liner for each new
item, and writes the result back to the same files.

Designed to be run AFTER the fetch scripts. Skips items that already have
a sowhat value, so re-runs are idempotent and API calls are minimised.

Requires: ANTHROPIC_API_KEY environment variable
Install:  pip install anthropic
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

DATA_DIR = Path(__file__).parent.parent / "data"
TRIALS_PATH = DATA_DIR / "trials.json"
PUBS_PATH   = DATA_DIR / "publications.json"
NEWS_PATH   = DATA_DIR / "news.json"

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 120
RATE_LIMIT_DELAY = 0.3

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

TRIAL_PROMPT = """\
You are a biotech competitive intelligence analyst specialising in cell therapy.

Given the following clinical trial, write a single punchy sentence (max 20 words)
that captures the strategic "so what" for someone tracking the in vivo CAR-T space.
Focus on what makes this trial notable: novel delivery, first-in-class target,
non-oncology indication, key sponsor, or unusual design.

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

PUB_PROMPT = """\
You are a biotech competitive intelligence analyst specialising in cell therapy.

Given the following publication abstract, write a single punchy sentence (max 20 words)
capturing the strategic "so what" for someone tracking the in vivo CAR-T space.
Focus on the key finding and why it matters competitively or clinically.

Rules:
- Return ONLY the sentence, nothing else
- No preamble, no sign-off, no offers to help
- If the abstract is insufficient to draw a conclusion, return exactly: Abstract not available

Title: {title}
Journal: {journal}
Preprint: {preprint}
Abstract: {abstract}
"""

NEWS_PROMPT = """\
You are a biotech competitive intelligence analyst specialising in cell therapy.

Given the following news headline and summary, write a single punchy sentence (max 20 words)
capturing the strategic "so what" for someone tracking the in vivo CAR-T space.
Focus on competitive implications: funding, clinical data, partnerships, regulatory events,
or key hires.

Rules:
- Return ONLY the sentence, nothing else
- No preamble, no sign-off, no offers to help
- If the summary is insufficient, return exactly: Summary not available

Source: {source}
Title: {title}
Summary: {summary}
"""


# ---------------------------------------------------------------------------
# Core summarisation helper
# ---------------------------------------------------------------------------

def generate_sowhat(prompt: str) -> str:
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
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


# ---------------------------------------------------------------------------
# Per-source summarisation
# ---------------------------------------------------------------------------

def summarise_trials(trials_data: dict) -> tuple[dict, int]:
    updated = 0
    for trial in trials_data.get("studies", []):
        if trial.get("sowhat"):
            continue
        prompt = TRIAL_PROMPT.format(
            title=trial.get("title", ""),
            sponsor=trial.get("sponsor", ""),
            modality=trial.get("modality", ""),
            conditions=", ".join(trial.get("conditions", [])),
            phase=trial.get("phase", ""),
            summary=(trial.get("summary", "") or "")[:600],
        )
        sowhat = generate_sowhat(prompt)
        if not is_unhelpful(sowhat):
            trial["sowhat"] = sowhat
            updated += 1
            logging.info(f"  [{trial.get('nct_id')}] {sowhat}")
        time.sleep(RATE_LIMIT_DELAY)
    return trials_data, updated


def summarise_publications(pubs_data: dict) -> tuple[dict, int]:
    updated = 0
    for pub in pubs_data.get("publications", []):
        if pub.get("sowhat"):
            continue
        abstract = (pub.get("abstract") or "").strip()
        if not abstract or len(abstract) < 50:
            pub["sowhat"] = "Abstract not available"
            updated += 1
            continue
        prompt = PUB_PROMPT.format(
            title=pub.get("title", ""),
            journal=pub.get("journal", ""),
            preprint=pub.get("preprint", False),
            abstract=abstract[:800],
        )
        sowhat = generate_sowhat(prompt)
        if is_unhelpful(sowhat):
            pub["sowhat"] = "Abstract not available"
        else:
            pub["sowhat"] = sowhat
            updated += 1
            logging.info(f"  [{pub.get('pmid') or pub.get('doi', '')[:20]}] {sowhat}")
        time.sleep(RATE_LIMIT_DELAY)
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
        sowhat = generate_sowhat(prompt)
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

    if TRIALS_PATH.exists():
        logging.info("Summarising trials...")
        data = json.loads(TRIALS_PATH.read_text())
        data, n = summarise_trials(data)
        TRIALS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logging.info(f"  Added {n} new trial summaries")
    else:
        logging.warning(f"Trials file not found: {TRIALS_PATH}")

    if PUBS_PATH.exists():
        logging.info("Summarising publications...")
        data = json.loads(PUBS_PATH.read_text())
        data, n = summarise_publications(data)
        PUBS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logging.info(f"  Added {n} new publication summaries")
    else:
        logging.warning(f"Publications file not found: {PUBS_PATH}")

    if NEWS_PATH.exists():
        logging.info("Summarising news...")
        data = json.loads(NEWS_PATH.read_text())
        data, n = summarise_news(data)
        NEWS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logging.info(f"  Added {n} new news summaries")
    else:
        logging.warning(f"News file not found: {NEWS_PATH}")

    logging.info("Summarisation complete")


if __name__ == "__main__":
    run()
