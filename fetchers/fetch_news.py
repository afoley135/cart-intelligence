"""
fetch_news.py
-------------
Fetches in vivo CAR-T news from:
  - NewsAPI (structured search across thousands of outlets)
  - RSS feeds from key biotech trade publications

Writes structured JSON to data/news.json.

Requires: NEWS_API_KEY environment variable
API docs: https://newsapi.org/docs
"""

import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOOKBACK_DAYS = 30
MAX_NEWSAPI_RESULTS = 100
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "news.json"

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
NEWSAPI_BASE = "https://newsapi.org/v2/everything"

# NewsAPI search queries — each runs as a separate request, deduplicated by URL
NEWSAPI_QUERIES = [
    '"in vivo CAR-T"',
    '"in vivo CAR T"',
    '"lipid nanoparticle CAR"',
    '"lentiviral CAR T"',
    '"non-viral CAR T"',
]

# RSS feeds from key biotech trade publications
RSS_FEEDS = [
    {
        "name": "BioPharma Dive",
        "url": "https://www.biopharmadive.com/feeds/news/",
    },
    {
        "name": "Fierce Biotech",
        "url": "https://www.fiercebiotech.com/rss/xml",
    },
    {
        "name": "STAT News",
        "url": "https://www.statnews.com/feed/",
    },
    {
        "name": "Endpoints News",
        "url": "https://endpts.com/feed/",
    },
    {
        "name": "BioSpace",
        "url": "https://www.biospace.com/articles/rss/",
    },
]

# Keywords to filter RSS items — must match at least one
RSS_KEYWORDS = [
    "in vivo car-t",
    "in vivo car t",
    "lipid nanoparticle car",
    "lentiviral car",
    "non-viral car",
    "car-t delivery",
    "in vivo t cell",
    "umoja",
    "sana biotechnology",
    "capstan therapeutics",
    "precision biosciences",
    "ensoma",
    "interius",
    "kelonia",
    "orna therapeutics",
    "sail biomedicines",
]


# ---------------------------------------------------------------------------
# NewsAPI helpers
# ---------------------------------------------------------------------------

def fetch_newsapi(query: str, from_date: str) -> list[dict]:
    if not NEWS_API_KEY:
        logging.warning("NEWS_API_KEY not set — skipping NewsAPI")
        return []

    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "pageSize": MAX_NEWSAPI_RESULTS,
        "language": "en",
        "apiKey": NEWS_API_KEY,
    }

    resp = requests.get(NEWSAPI_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        logging.warning(f"NewsAPI error for '{query}': {data.get('message')}")
        return []

    return data.get("articles", [])


def parse_newsapi_article(article: dict) -> dict:
    return {
        "source": article.get("source", {}).get("name", "Unknown"),
        "title": article.get("title", ""),
        "summary": article.get("description", ""),
        "url": article.get("url", ""),
        "date": (article.get("publishedAt", "") or "")[:10],
        "tags": [],
        "sowhat": None,
        "fetch_source": "newsapi",
    }


# ---------------------------------------------------------------------------
# RSS helpers
# ---------------------------------------------------------------------------

def fetch_rss(feed: dict, lookback_days: int) -> list[dict]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; cart-intelligence-bot/1.0)"}
    resp = requests.get(feed["url"], headers=headers, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    results = []

    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        description = (item.findtext("description") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date_str = (item.findtext("pubDate") or "").strip()

        # Parse date
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

        # Keyword filter
        text = (title + " " + description).lower()
        if not any(kw in text for kw in RSS_KEYWORDS):
            continue

        results.append({
            "source": feed["name"],
            "title": title,
            "summary": description[:500] if description else "",
            "url": link,
            "date": date_str,
            "tags": [],
            "sowhat": None,
            "fetch_source": "rss",
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting news fetch")

    from_date = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    all_news: dict[str, dict] = {}  # keyed by URL for deduplication

    # --- NewsAPI ---
    if NEWS_API_KEY:
        for query in NEWSAPI_QUERIES:
            logging.info(f"NewsAPI query: {query}")
            try:
                articles = fetch_newsapi(query, from_date)
                for a in articles:
                    parsed = parse_newsapi_article(a)
                    url = parsed["url"]
                    if url and url not in all_news:
                        all_news[url] = parsed
                logging.info(f"  Found {len(articles)} articles")
                time.sleep(0.5)
            except requests.RequestException as e:
                logging.error(f"NewsAPI failed for '{query}': {e}")
    else:
        logging.warning("NEWS_API_KEY not set — skipping NewsAPI")

    # --- RSS feeds ---
    for feed in RSS_FEEDS:
        logging.info(f"Fetching RSS: {feed['name']}")
        try:
            items = fetch_rss(feed, LOOKBACK_DAYS)
            for item in items:
                url = item["url"]
                if url and url not in all_news:
                    all_news[url] = item
            logging.info(f"  Found {len(items)} relevant items")
            time.sleep(0.3)
        except Exception as e:
            logging.error(f"RSS fetch failed for {feed['name']}: {e}")

    news_list = sorted(
        all_news.values(),
        key=lambda n: n["date"] or "",
        reverse=True,
    )

    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(news_list),
        "news": news_list,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logging.info(f"Wrote {len(news_list)} news items to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
