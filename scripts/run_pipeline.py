#!/usr/bin/env python3
"""
run_pipeline.py
---------------
Orchestrates the full data pipeline in the correct order:

  1. fetch_trials.py       — ClinicalTrials.gov
  2. fetch_publications.py — PubMed + bioRxiv
  3. fetch_news.py         — NewsAPI + RSS feeds
  4. summarize.py          — AI "so what" generation (requires ANTHROPIC_API_KEY)

Run manually:
  python scripts/run_pipeline.py

Or let GitHub Actions call this on a schedule (see .github/workflows/update.yml).

Exit codes:
  0 — all steps succeeded
  1 — one or more steps failed (check logs)
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FETCHERS_DIR = Path(__file__).parent.parent / "fetchers"

STEPS = [
    ("ClinicalTrials.gov",  FETCHERS_DIR / "fetch_trials.py"),
    ("PubMed + bioRxiv",    FETCHERS_DIR / "fetch_publications.py"),
    ("News",                FETCHERS_DIR / "fetch_news.py"),
    ("AI summarisation",    FETCHERS_DIR / "summarize.py"),
]


def run_step(name: str, script: Path) -> bool:
    logging.info(f"── Starting: {name}")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=False,
    )
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
