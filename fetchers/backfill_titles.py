"""
backfill_titles.py
------------------
ONE-TIME SCRIPT — delete after running.

Translates non-English patent titles already in data/patents.json.
Makes no EPO API calls — only reads the existing cache and calls Claude
(Haiku) once per untranslated patent.

For each patent where title_original is absent:
  - Asks Claude whether the title is English and, if not, to translate it
  - Writes title_original = original title, title = English translation
  - Skips patents that are already English or already have title_original set

Requires: ANTHROPIC_API_KEY env var

Run once, then delete this file.
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

OUTPUT_PATH       = Path(__file__).parent.parent / "data" / "patents.json"
MODEL             = "claude-haiku-4-5-20251001"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

TRANSLATION_PROMPT = """\
You are a patent translator. Examine the title below.

If it is already in English, return exactly: {{"english": true, "title_en": null}}
If it is not in English, return exactly: {{"english": false, "title_en": "<accurate English translation>"}}

Return ONLY the JSON object — no preamble, no markdown.

Title: {title}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def needs_translation(patent: dict) -> bool:
    """Skip if already translated or if title is obviously ASCII/English."""
    if patent.get("title_original"):
        return False
    title = patent.get("title", "")
    if not title:
        return False
    # Heuristic: if all chars are ASCII it's almost certainly English already
    try:
        title.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def translate(client: anthropic.Anthropic, patent: dict) -> dict:
    title = patent["title"]
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": TRANSLATION_PROMPT.format(title=title)}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())

        if not result.get("english") and result.get("title_en"):
            patent["title_original"] = title
            patent["title"]          = result["title_en"].strip()
            logging.info(f"  Translated: {title[:50]} → {patent['title'][:50]}")
        else:
            logging.info(f"  Already English: {title[:60]}")

    except Exception as e:
        logging.warning(f"  Translation failed for '{title[:50]}': {e}")

    return patent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting one-time title translation backfill")

    if not ANTHROPIC_API_KEY:
        logging.error("ANTHROPIC_API_KEY not set — aborting")
        return

    if not OUTPUT_PATH.exists():
        logging.error(f"{OUTPUT_PATH} not found — aborting")
        return

    data = json.loads(OUTPUT_PATH.read_text())
    patents = data.get("patents", [])
    logging.info(f"Loaded {len(patents)} patents")

    client      = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    to_translate = [p for p in patents if needs_translation(p)]
    already_done = len(patents) - len(to_translate)

    logging.info(f"  {already_done} already English or translated — skipping")
    logging.info(f"  {len(to_translate)} patents need translation")

    if not to_translate:
        logging.info("Nothing to do.")
        return

    updated = 0
    for i, patent in enumerate(to_translate, 1):
        logging.info(f"[{i}/{len(to_translate)}] {patent.get('watchlist_company', '')} — {patent['title'][:60]}")
        translate(client, patent)
        updated += 1
        time.sleep(0.3)

    data["patents"] = patents  # mutated in place
    OUTPUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logging.info(f"Done — translated {updated} patents. Delete this script.")


if __name__ == "__main__":
    run()
