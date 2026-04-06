"""
summarize.py
------------
Reads data/trials.json and data/publications.json, calls the Anthropic API
to generate a "so what" one-liner for each new item, and writes the result
back to the same files.

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
PUBS_PATH = DATA_DIR / "publications.json"

MODEL = "claude-haiku-4-5-20251001"   # fast + cheap for batch summarisation
MAX_TOKENS = 120
RATE_LIMIT_DELAY = 0.3               # seconds between API calls

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

Return ONLY the sentence. No preamble, no punctuation at start, no quotes.

Title: {title}
Sponsor: {sponsor}
Modality: {modality}
Conditions: {conditions}
Phase: {phase}
Summary: {summary}
"""

PUB_PROMPT = """\
You are a biotech competitive intelligence analyst specialising in cell therapy.

Given the following publication, write a single punchy sentence (max 20 words)
that captures the strategic "so what" for someone tracking the in vivo CAR-T space.
Focus on the key finding and why it matters competitively or clinically.

Return ONLY the sentence. No preamble, no punctuation at start, no quotes.

Title: {title}
Journal: {journal}
Preprint: {preprint}
Abstract: {abstract}
"""


# ---------------------------------------------------------------------------
# Summarisation helpers
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


def summarise_trials(trials_data: dict) -> tuple[dict, int]:
    updated = 0
    for trial in trials_data.get("studies", []):
        if trial.get("sowhat"):
            continue  # already done
        prompt = TRIAL_PROMPT.format(
            title=trial.get("title", ""),
            sponsor=trial.get("sponsor", ""),
            modality=trial.get("modality", ""),
            conditions=", ".join(trial.get("conditions", [])),
            phase=trial.get("phase", ""),
            summary=(trial.get("summary", "") or "")[:600],
        )
        sowhat = generate_sowhat(prompt)
        if sowhat:
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
        prompt = PUB_PROMPT.format(
            title=pub.get("title", ""),
            journal=pub.get("journal", ""),
            preprint=pub.get("preprint", False),
            abstract=(pub.get("abstract", "") or "")[:800],
        )
        sowhat = generate_sowhat(prompt)
        if sowhat:
            pub["sowhat"] = sowhat
            updated += 1
            logging.info(f"  [{pub.get('pmid') or pub.get('doi', '')[:20]}] {sowhat}")
        time.sleep(RATE_LIMIT_DELAY)
    return pubs_data, updated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting AI summarisation pass")

    if TRIALS_PATH.exists():
        logging.info("Summarising trials...")
        trials_data = json.loads(TRIALS_PATH.read_text())
        trials_data, n = summarise_trials(trials_data)
        TRIALS_PATH.write_text(json.dumps(trials_data, indent=2, ensure_ascii=False))
        logging.info(f"  Added {n} new trial summaries")
    else:
        logging.warning(f"Trials file not found: {TRIALS_PATH}")

    if PUBS_PATH.exists():
        logging.info("Summarising publications...")
        pubs_data = json.loads(PUBS_PATH.read_text())
        pubs_data, n = summarise_publications(pubs_data)
        PUBS_PATH.write_text(json.dumps(pubs_data, indent=2, ensure_ascii=False))
        logging.info(f"  Added {n} new publication summaries")
    else:
        logging.warning(f"Publications file not found: {PUBS_PATH}")

    logging.info("Summarisation complete")


if __name__ == "__main__":
    run()