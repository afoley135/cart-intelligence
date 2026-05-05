#!/usr/bin/env python3
"""
run_pipeline.py
---------------
Orchestrates the full data pipeline in the correct order:

  1. fetch_trials.py         — ClinicalTrials.gov
  2. fetch_publications.py   — PubMed + bioRxiv
  3. fetch_abstracts.py      — Conference abstracts
  4. fetch_news.py           — NewsAPI + RSS feeds
  5. fetch_patents.py        — EPO OPS patents
  6. fetch_curated.py        — Re-fetch + merge manually curated entries
  7. summarize.py            — AI "so what" generation

fetch_curated.py runs AFTER all automated fetchers so it can:
  - Re-fetch live data for curated trials and publications
  - Merge curated entries into the data files (winning over automated entries
    for the same key, so curated data is always shown)

summarize.py runs LAST so it can generate sowhat summaries for any curated
items that are new or don't yet have one.

Run manually:
  python scripts/run_pipeline.py
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FETCHERS_DIR = Path(__file__).parent.parent / "fetchers"

STEPS = [
    # Website monitoring disabled — virtually all watchlist sites are JS-rendered
    # and return empty content to plain requests. Re-enable once upgraded to
    # Playwright (headless browser). See roadmap notes in website_monitor.py.
    # ("Website monitoring",    FETCHERS_DIR / "website_monitor.py"),
    ("ClinicalTrials.gov",      FETCHERS_DIR / "fetch_trials.py"),
    ("PubMed + bioRxiv",        FETCHERS_DIR / "fetch_publications.py"),
    ("Conference abstracts",    FETCHERS_DIR / "fetch_abstracts.py"),
    ("News + Funding",          FETCHERS_DIR / "fetch_news.py"),
    ("Patents",                 FETCHERS_DIR / "fetch_patents.py"),
    # Curated entries: re-fetch live data and merge into data files.
    # Must run after all automated fetchers and before summarize.py.
    ("Curated entries",         FETCHERS_DIR / "fetch_curated.py"),
    ("AI summarisation",        FETCHERS_DIR / "summarize.py"),
]


def run_step(name: str, script: Path) -> bool:
    logging.info(f"── Starting: {name}")
    result = subprocess.run([sys.executable, str(script)], capture_output=False)
    if result.returncode == 0:
        logging.info(f"── Done: {name}")
        return True
    else:
        logging.error(f"── FAILED: {name} (exit code {result.returncode})")
        return False


def main():
    failures = []
    for name, script in STEPS:
        if not script.exists():
            logging.error(f"Script not found: {script}")
            failures.append(name)
            continue
        if not run_step(name, script):
            failures.append(name)

    if failures:
        logging.error(f"Pipeline completed with failures: {', '.join(failures)}")
        sys.exit(1)
    else:
        logging.info("Pipeline completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
