"""
fetch_patents.py
----------------
Monitors patent activity for watchlist companies using:
  1. Google News RSS — searches for patent filings, grants, and IP news
     per watchlist company (no network restrictions, no API key needed)
  2. Google Patents RSS — searches Google Patents directly for new filings
     by assignee name

For each patent item found, passes title + snippet to Claude for:
  - Relevance assessment (is this a CAR-T / gene therapy patent?)
  - Claim type classification where determinable
  - Novelty summary

Writes structured JSON to data/patents.json.

Requires: ANTHROPIC_API_KEY environment variable
"""

import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import anthropic
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_PATH    = Path(__file__).parent.parent / "data" / "patents.json"
WATCHLIST_PATH = Path(__file__).parent.parent / "watchlist.json"

LOOKBACK_DAYS     = 365 * 5   # 5 years for Google Patents RSS
NEWS_LOOKBACK_DAYS = 90        # 90 days for news-based patent monitoring

MODEL = "claude-haiku-4-5-20251001"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

PATENT_ANALYSIS_PROMPT = """\
You are a biotech patent analyst specialising in cell and gene therapy.

Given the following patent-related item, provide a structured analysis.
Return ONLY a valid JSON object with these exact keys — no preamble, no markdown:

  "claim_type": one of: "Composition of matter" | "Method of treatment" | "Method (process)" | "Composition + Method" | "Other" | "Unknown"
  "novelty_summary": 1-2 sentences on what appears novel and its relevance to in vivo CAR-T (or null if insufficient information)
  "relevant": true or false — is this relevant to CAR-T, gene therapy, T cell engineering, or related delivery technology?

Title: {title}
Summary: {summary}
Assignee/Company: {assignee}
"""

VALID_CLAIM_TYPES = {
    "Composition of matter",
    "Method of treatment",
    "Method (process)",
    "Composition + Method",
    "Other",
    "Unknown",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_watchlist() -> list[str]:
    try:
        return json.loads(WATCHLIST_PATH.read_text()).get("companies", [])
    except Exception as e:
        logging.warning(f"Could not load watchlist: {e}")
        return []


def google_news_rss_url(query: str) -> str:
    encoded = quote_plus(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"


def google_patents_rss_url(assignee: str) -> str:
    """Google Patents search RSS for a specific assignee."""
    query = quote_plus(f'assignee:"{assignee}"')
    return f"https://patents.google.com/xhr/query?url=assignee%3D%22{quote_plus(assignee)}%22&exp=&download=true"


# ---------------------------------------------------------------------------
# Google News RSS patent monitoring
# ---------------------------------------------------------------------------

def fetch_patent_news(company: str, lookback_days: int) -> list[dict]:
    """Fetch patent-related news for a company via Google News RSS."""
    cutoff  = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    results = []

    # Search queries targeting patent activity
    queries = [
        f'"{company}" patent',
        f'"{company}" patent filing',
        f'"{company}" granted patent',
        f'"{company}" intellectual property',
    ]

    seen = set()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; cart-intelligence-bot/1.0)"}

    for query in queries[:2]:  # limit to 2 queries per company to stay within rate limits
        try:
            url  = google_news_rss_url(query)
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            for item in root.findall(".//item"):
                title       = (item.findtext("title") or "").strip()
                link        = (item.findtext("link") or "").strip()
                description = (item.findtext("description") or "").strip()
                pub_date_str = (item.findtext("pubDate") or "").strip()
                source_el   = item.find("{https://news.google.com/rss}source")
                source_name = source_el.text if source_el is not None else "Google News"

                if link in seen:
                    continue
                seen.add(link)

                date_str = ""
                if pub_date_str:
                    try:
                        from email.utils import parsedate_to_datetime
                        pub_dt = parsedate_to_datetime(pub_date_str)
                        if pub_dt.tzinfo is None:
                            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                        if pub_dt < cutoff:
                            continue
                        date_str = pub_dt.strftime("%Y-%m-%d")
                    except Exception:
                        pass

                # Basic relevance filter — must mention patent
                combined = (title + " " + description).lower()
                if "patent" not in combined and "ip " not in combined and "intellectual property" not in combined:
                    continue

                results.append({
                    "id":               link,
                    "title":            title,
                    "abstract":         re.sub(r'<[^>]+>', '', description)[:500],
                    "assignee":         company,
                    "watchlist_company": company,
                    "filing_date":      date_str,
                    "patent_number":    None,
                    "application_number": None,
                    "source":           source_name,
                    "url":              link,
                    "data_type":        "news",
                    "claim_type":       None,
                    "novelty_summary":  None,
                    "relevant":         None,
                })

            time.sleep(0.4)

        except Exception as e:
            logging.warning(f"  Google News patent fetch failed for '{company}' query '{query}': {e}")

    return results


# ---------------------------------------------------------------------------
# Google Patents RSS
# ---------------------------------------------------------------------------

def fetch_google_patents(company: str) -> list[dict]:
    """
    Fetch recent patent filings from Google Patents for a company.
    Uses the Google Patents search page with assignee filter.
    """
    results = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; cart-intelligence-bot/1.0)"}

    try:
        # Google Patents supports a search URL that returns an atom/RSS feed
        encoded_assignee = quote_plus(f'assignee="{company}"')
        url = f"https://patents.google.com/xhr/query?url={encoded_assignee}&exp=&download=true"
        resp = requests.get(url, headers=headers, timeout=30)

        if resp.status_code != 200:
            logging.warning(f"  Google Patents returned {resp.status_code} for '{company}'")
            return []

        # Parse the response — Google Patents returns JSON
        data = resp.json()
        hits = data.get("results", {}).get("cluster", [])

        for cluster in hits[:20]:
            for result in cluster.get("result", [])[:1]:
                patent = result.get("patent", {})
                title   = patent.get("title", "")
                pat_num = patent.get("publication_number", "")
                filing  = patent.get("filing_date", "")
                abstract= patent.get("abstract", "") or ""
                assignees = [a.get("name","") for a in patent.get("assignee", [])]
                assignee_str = "; ".join(assignees[:2]) or company

                if not title:
                    continue

                results.append({
                    "id":               pat_num or title[:60],
                    "title":            title,
                    "abstract":         abstract[:500],
                    "assignee":         assignee_str,
                    "watchlist_company": company,
                    "filing_date":      filing[:10] if filing else "",
                    "patent_number":    pat_num,
                    "application_number": patent.get("application_number",""),
                    "source":           "Google Patents",
                    "url":              f"https://patents.google.com/patent/{pat_num}" if pat_num else "",
                    "data_type":        "patent",
                    "claim_type":       None,
                    "novelty_summary":  None,
                    "relevant":         None,
                })

    except Exception as e:
        logging.warning(f"  Google Patents fetch failed for '{company}': {e}")

    return results


# ---------------------------------------------------------------------------
# Claude analysis
# ---------------------------------------------------------------------------

def analyse_patent(patent: dict) -> dict:
    if not ANTHROPIC_API_KEY or not patent.get("title"):
        return patent
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = PATENT_ANALYSIS_PROMPT.format(
            title=patent.get("title",""),
            summary=(patent.get("abstract","") or "")[:600],
            assignee=patent.get("assignee",""),
        )
        msg = client.messages.create(
            model=MODEL, max_tokens=250,
            messages=[{"role":"user","content":prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        patent["claim_type"]      = result.get("claim_type") if result.get("claim_type") in VALID_CLAIM_TYPES else "Unknown"
        patent["novelty_summary"] = result.get("novelty_summary")
        patent["relevant"]        = result.get("relevant", True)
    except Exception as e:
        logging.warning(f"  Claude analysis failed: {e}")
    return patent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting patent fetch")

    watchlist = load_watchlist()
    logging.info(f"Processing {len(watchlist)} watchlist companies")

    # Load existing to preserve Claude analysis
    all_patents: dict[str, dict] = {}
    if OUTPUT_PATH.exists():
        try:
            existing = json.loads(OUTPUT_PATH.read_text())
            for p in existing.get("patents", []):
                pid = p.get("id")
                if pid:
                    all_patents[pid] = p
            logging.info(f"  Loaded {len(all_patents)} existing entries")
        except Exception as e:
            logging.warning(f"Could not load existing patents: {e}")

    new_count = 0

    for company in watchlist:
        logging.info(f"  {company}")

        # Source 1 — Google News patent monitoring
        news_items = fetch_patent_news(company, NEWS_LOOKBACK_DAYS)
        logging.info(f"    News: {len(news_items)} patent-related items")

        # Source 2 — Google Patents direct search
        patent_items = fetch_google_patents(company)
        logging.info(f"    Patents: {len(patent_items)} filings")

        for item in news_items + patent_items:
            pid = item["id"]
            if pid in all_patents:
                continue
            if ANTHROPIC_API_KEY:
                item = analyse_patent(item)
                time.sleep(0.25)
            if item.get("relevant") is not False:
                all_patents[pid] = item
                new_count += 1

        time.sleep(0.5)

    logging.info(f"  Added {new_count} new entries")

    patents_list = sorted(
        all_patents.values(),
        key=lambda p: p.get("filing_date") or "",
        reverse=True,
    )

    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count":      len(patents_list),
        "patents":    patents_list,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logging.info(f"Wrote {len(patents_list)} patent entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
