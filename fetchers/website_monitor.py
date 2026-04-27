"""
website_monitor.py
------------------
Monitors watchlist company websites for changes to key pages
(homepage, pipeline, news/press releases) and flags updates
for analyst review.

For each company and each monitored page, stores:
  - SHA-256 hash of cleaned page text content
  - Word count
  - Snapshot of key text blocks (pipeline items, news headlines)
  - Timestamp of last check and last change

On each run:
  1. Fetches each monitored page
  2. Extracts meaningful text (strips boilerplate nav/footer)
  3. Compares hash to stored snapshot
  4. If changed: records diff metadata, flags for review
  5. Writes updated snapshots + change log to data/

Writes to:
  data/website_snapshots.json  — current state + hashes
  data/website_changes.json    — log of detected changes (last 90 days)

Note: Many biotech sites use JS rendering. This fetcher uses plain
requests, which works for server-rendered content. JS-heavy sites
will show as blank/minimal — those pages are noted in company_profiles.json
and can be upgraded to Playwright if needed.

Requires: no API keys
"""

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR          = Path(__file__).parent.parent / "data"
PROFILES_PATH     = Path(__file__).parent.parent / "company_profiles.json"
SNAPSHOTS_PATH    = DATA_DIR / "website_snapshots.json"
CHANGES_PATH      = DATA_DIR / "website_changes.json"

CHANGE_RETENTION_DAYS = 90
REQUEST_TIMEOUT       = 20
SLEEP_BETWEEN_PAGES   = 2.0   # seconds — be polite
MIN_CONTENT_LENGTH    = 100   # ignore pages that return <100 chars (likely blocked)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Page types to monitor per company (resolved from company_profiles.json)
PAGE_TYPES = ["website", "pipeline_page", "news_page"]

# Tags to strip before hashing — navigation, footers, cookie banners etc.
STRIP_TAGS = [
    "nav", "footer", "header", "script", "style", "noscript",
    "iframe", "svg", "form", "button",
]

# Patterns that indicate a page is a generic "coming soon" / empty state
EMPTY_PATTERNS = [
    "coming soon", "under construction", "page not found",
    "404", "403 forbidden", "access denied",
]

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def fetch_page(url: str) -> tuple[str, int]:
    """
    Fetch a URL and return (cleaned_text, status_code).
    Returns ("", status_code) on failure.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        status = resp.status_code
        if status != 200:
            return "", status
        return resp.text, status
    except requests.exceptions.SSLError:
        logging.warning(f"  SSL error fetching {url}")
        return "", 0
    except requests.exceptions.ConnectionError:
        logging.warning(f"  Connection error fetching {url}")
        return "", 0
    except requests.exceptions.Timeout:
        logging.warning(f"  Timeout fetching {url}")
        return "", 0
    except Exception as e:
        logging.warning(f"  Fetch error for {url}: {e}")
        return "", 0


def extract_text(html: str) -> str:
    """
    Parse HTML and return cleaned, normalised text content.
    Strips nav, footer, scripts; collapses whitespace.
    """
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")

        # Remove boilerplate elements
        for tag in STRIP_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        # Remove elements with common boilerplate class/id names
        boilerplate_patterns = [
            "cookie", "consent", "gdpr", "banner", "popup",
            "modal", "overlay", "nav", "menu", "footer", "header",
            "sidebar", "advertisement", "ad-", "social",
        ]
        for el in soup.find_all(True):
            el_id    = (el.get("id") or "").lower()
            el_class = " ".join(el.get("class") or []).lower()
            combined = el_id + " " + el_class
            if any(p in combined for p in boilerplate_patterns):
                el.decompose()

        text = soup.get_text(separator=" ")
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        logging.warning(f"  HTML parse error: {e}")
        return ""


def content_hash(text: str) -> str:
    """SHA-256 of normalised text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_pipeline_signals(text: str) -> list[str]:
    """
    Heuristically pull out pipeline-related sentences.
    These are stored in snapshots so diffs are human-readable.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    signals = []
    keywords = [
        "phase", "trial", "ind", "clinical", "program", "asset",
        "candidate", "preclinical", "approved", "nda", "bla",
        "fda", "ema", "cleared", "initiated", "enrollment",
        "data", "results", "cohort", "dose", "patient",
    ]
    for s in sentences:
        sl = s.lower()
        if any(k in sl for k in keywords) and 20 < len(s) < 300:
            signals.append(s.strip())
    return signals[:20]  # cap at 20 to keep snapshots lean


def is_empty_page(text: str) -> bool:
    """Return True if the page appears to be a placeholder / error."""
    tl = text.lower()
    return len(text) < MIN_CONTENT_LENGTH or any(p in tl for p in EMPTY_PATTERNS)


# ---------------------------------------------------------------------------
# Snapshot management
# ---------------------------------------------------------------------------

def load_snapshots() -> dict:
    try:
        return json.loads(SNAPSHOTS_PATH.read_text()) if SNAPSHOTS_PATH.exists() else {}
    except Exception:
        return {}


def load_changes() -> list:
    try:
        if CHANGES_PATH.exists():
            data = json.loads(CHANGES_PATH.read_text())
            return data.get("changes", [])
        return []
    except Exception:
        return []


def save_snapshots(snapshots: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_PATH.write_text(json.dumps(snapshots, indent=2, ensure_ascii=False))


def save_changes(changes: list):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Prune old changes
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CHANGE_RETENTION_DAYS)).isoformat()
    changes = [c for c in changes if (c.get("detected_at") or "") >= cutoff]
    output = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "count":      len(changes),
        "changes":    changes,
    }
    CHANGES_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def describe_change(old_snapshot: dict, new_text: str, new_hash: str) -> dict:
    """
    Generate a human-readable change description by comparing
    old and new pipeline signals.
    """
    old_signals = set(old_snapshot.get("pipeline_signals", []))
    new_signals = set(extract_pipeline_signals(new_text))

    added   = list(new_signals - old_signals)
    removed = list(old_signals - new_signals)

    old_wc = old_snapshot.get("word_count", 0)
    new_wc = len(new_text.split())
    wc_delta = new_wc - old_wc

    return {
        "old_hash":          old_snapshot.get("hash", ""),
        "new_hash":          new_hash,
        "word_count_old":    old_wc,
        "word_count_new":    new_wc,
        "word_count_delta":  wc_delta,
        "signals_added":     added[:10],
        "signals_removed":   removed[:10],
        "change_magnitude":  _magnitude(wc_delta, len(added), len(removed)),
    }


def _magnitude(wc_delta: int, added: int, removed: int) -> str:
    """Classify change size."""
    score = abs(wc_delta) / 50 + added * 2 + removed * 2
    if score > 20:  return "major"
    if score > 5:   return "moderate"
    return "minor"


# ---------------------------------------------------------------------------
# Main monitoring loop
# ---------------------------------------------------------------------------

def monitor_company(
    company_name: str,
    profile: dict,
    snapshots: dict,
    new_changes: list,
) -> dict:
    """
    Monitor all pages for one company.
    Updates snapshots in-place, appends to new_changes.
    Returns updated company snapshot dict.
    """
    company_key = company_name.lower().replace(" ", "_")
    company_snap = snapshots.get(company_key, {})
    now = datetime.now(timezone.utc).isoformat()

    for page_type in PAGE_TYPES:
        url = profile.get(page_type)
        if not url or not url.startswith("http"):
            continue

        page_key = page_type  # e.g. "pipeline_page"
        old_snap = company_snap.get(page_key, {})

        logging.info(f"    [{page_type}] {url}")

        html, status = fetch_page(url)
        time.sleep(SLEEP_BETWEEN_PAGES)

        if not html or status != 200:
            logging.warning(f"      Fetch failed (status {status})")
            # Record the failure but don't overwrite a good snapshot
            company_snap.setdefault(page_key, {})
            company_snap[page_key]["last_checked"] = now
            company_snap[page_key]["last_status"]  = status
            continue

        text = extract_text(html)

        if is_empty_page(text):
            logging.info(f"      Empty/JS-rendered page — skipping hash")
            company_snap.setdefault(page_key, {})
            company_snap[page_key]["last_checked"]  = now
            company_snap[page_key]["last_status"]   = status
            company_snap[page_key]["js_rendered"]   = True
            continue

        new_hash     = content_hash(text)
        new_wc       = len(text.split())
        new_signals  = extract_pipeline_signals(text)

        # First time seeing this page
        if not old_snap.get("hash"):
            logging.info(f"      First snapshot — {new_wc} words")
            company_snap[page_key] = {
                "url":              url,
                "hash":             new_hash,
                "word_count":       new_wc,
                "pipeline_signals": new_signals,
                "first_seen":       now,
                "last_checked":     now,
                "last_changed":     now,
                "last_status":      status,
                "js_rendered":      False,
                "change_count":     0,
            }
            continue

        # Check for change
        if new_hash != old_snap.get("hash"):
            change = describe_change(old_snap, text, new_hash)
            logging.info(
                f"      CHANGED [{change['change_magnitude']}] "
                f"Δ{change['word_count_delta']:+d} words, "
                f"+{len(change['signals_added'])} / -{len(change['signals_removed'])} signals"
            )

            change_record = {
                "company":          company_name,
                "page_type":        page_type,
                "url":              url,
                "detected_at":      now,
                "reviewed":         False,
                **change,
            }
            new_changes.append(change_record)

            # Update snapshot
            company_snap[page_key].update({
                "hash":             new_hash,
                "word_count":       new_wc,
                "pipeline_signals": new_signals,
                "last_checked":     now,
                "last_changed":     now,
                "last_status":      status,
                "change_count":     old_snap.get("change_count", 0) + 1,
            })
        else:
            logging.info(f"      No change — {new_wc} words")
            company_snap[page_key]["last_checked"] = now
            company_snap[page_key]["last_status"]  = status

    snapshots[company_key] = company_snap
    return snapshots


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def print_summary(new_changes: list):
    if not new_changes:
        logging.info("No changes detected across all monitored pages.")
        return

    logging.info(f"\n{'='*60}")
    logging.info(f"CHANGES DETECTED: {len(new_changes)}")
    logging.info(f"{'='*60}")
    for c in new_changes:
        logging.info(
            f"  [{c['change_magnitude'].upper()}] {c['company']} — {c['page_type']}\n"
            f"    URL: {c['url']}\n"
            f"    Words: {c['word_count_old']} → {c['word_count_new']} "
            f"({c['word_count_delta']:+d})"
        )
        if c.get("signals_added"):
            for s in c["signals_added"][:3]:
                logging.info(f"    + {s[:120]}")
        if c.get("signals_removed"):
            for s in c["signals_removed"][:3]:
                logging.info(f"    - {s[:120]}")
    logging.info(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting website monitor")

    # Load profiles
    if not PROFILES_PATH.exists():
        logging.error(f"company_profiles.json not found at {PROFILES_PATH}")
        return

    try:
        profiles_data = json.loads(PROFILES_PATH.read_text())
        profiles = profiles_data.get("companies", {})
    except Exception as e:
        logging.error(f"Failed to load company profiles: {e}")
        return

    logging.info(f"Loaded {len(profiles)} company profiles")

    # Load existing snapshots and change log
    snapshots   = load_snapshots()
    old_changes = load_changes()
    new_changes = []

    for company_name, profile in profiles.items():
        logging.info(f"Monitoring: {company_name}")
        try:
            snapshots = monitor_company(company_name, profile, snapshots, new_changes)
        except Exception as e:
            logging.error(f"  Failed for {company_name}: {e}")

    # Save outputs
    save_snapshots(snapshots)

    all_changes = old_changes + new_changes
    save_changes(all_changes)

    print_summary(new_changes)
    logging.info(
        f"Website monitor complete. "
        f"{len(new_changes)} new changes detected. "
        f"Snapshots written to {SNAPSHOTS_PATH}"
    )


if __name__ == "__main__":
    run()
