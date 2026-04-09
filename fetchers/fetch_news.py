"""
fetch_news.py
-------------
Fetches in vivo CAR-T news from multiple sources across three passes:

  Pass 1a — NewsAPI keyword queries (exact phrases)
  Pass 1b — Google News RSS keyword queries (broad outlet coverage)
  Pass 1c — RSS feeds from BioPharma Dive, STAT News, Endpoints (keyword filtered)
  Pass 2a — NewsAPI watchlist company queries (relevance filtered)
  Pass 2b — STAT News + Endpoints RSS: all articles mentioning any watchlist company

Writes structured JSON to data/news.json.

Requires: NEWS_API_KEY environment variable
"""

import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOOKBACK_DAYS       = 30
MAX_NEWSAPI_RESULTS = 100
OUTPUT_PATH         = Path(__file__).parent.parent / "data" / "news.json"
WATCHLIST_PATH      = Path(__file__).parent.parent / "watchlist.json"

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
NEWSAPI_BASE = "https://newsapi.org/v2/everything"

# NewsAPI keyword queries — exact phrase searches
NEWSAPI_QUERIES = [
    '"in vivo CAR-T"',
    '"in vivo CAR T"',
    '"lipid nanoparticle CAR"',
    '"lentiviral CAR T"',
    '"non-viral CAR T"',
]

# Google News RSS — one feed per search term
# These catch articles across all outlets including Fierce Biotech, GEN, BioSpace etc.
GOOGLE_NEWS_QUERIES = [
    "in vivo CAR-T",
    "in vivo CAR T cell therapy",
    "lipid nanoparticle CAR-T",
    "lentiviral CAR T in vivo",
    "non-viral CAR T",
    "in vivo chimeric antigen receptor",
]

# Standard RSS feeds — keyword filtered
RSS_FEEDS = [
    {"name": "BioPharma Dive", "url": "https://www.biopharmadive.com/feeds/news/"},
]

# Premium sources — watchlist company filtered only
# These are fetched separately in Pass 2b with company name matching
PREMIUM_RSS_FEEDS = [
    {"name": "STAT News",      "url": "https://www.statnews.com/feed/"},
    {"name": "Endpoints News", "url": "https://endpts.com/feed/"},
]

# RSS keyword filter for Pass 1c
RSS_KEYWORDS = [
    "in vivo car-t", "in vivo car t", "lipid nanoparticle car",
    "lentiviral car", "non-viral car", "car-t delivery", "in vivo t cell",
    "in vivo chimeric antigen receptor",
]

# Relevance signals for NewsAPI watchlist pass filtering
RELEVANCE_SIGNALS = [
    "car-t", "car t cell", "chimeric antigen receptor",
    "cell therapy", "gene therapy", "t cell",
    "lentiviral", "lipid nanoparticle", "in vivo",
    "immunotherapy", "oncology", "hematol",
    "clinical trial", "phase 1", "phase 2",
]


# ---------------------------------------------------------------------------
# Watchlist helpers
# ---------------------------------------------------------------------------

def load_watchlist() -> list[str]:
    try:
        data = json.loads(WATCHLIST_PATH.read_text())
        return data.get("companies", [])
    except Exception as e:
        logging.warning(f"Could not load watchlist: {e}")
        return []


def is_relevant(article_title: str, article_desc: str) -> bool:
    text = (article_title + " " + (article_desc or "")).lower()
    return any(s in text for s in RELEVANCE_SIGNALS)


def mentions_watchlist_company(text: str, watchlist: list[str]) -> bool:
    text_lower = text.lower()
    return any(company.lower() in text_lower for company in watchlist)


# ---------------------------------------------------------------------------
# NewsAPI helpers
# ---------------------------------------------------------------------------

def fetch_newsapi(query: str, from_date: str) -> list[dict]:
    if not NEWS_API_KEY:
        return []
    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   "publishedAt",
        "pageSize": MAX_NEWSAPI_RESULTS,
        "language": "en",
        "apiKey":   NEWS_API_KEY,
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
        "source":       article.get("source", {}).get("name", "Unknown"),
        "title":        article.get("title", ""),
        "summary":      article.get("description", ""),
        "url":          article.get("url", ""),
        "date":         (article.get("publishedAt", "") or "")[:10],
        "tags":         [],
        "sowhat":       None,
        "fetch_source": "newsapi",
    }


# ---------------------------------------------------------------------------
# Google News RSS helpers
# ---------------------------------------------------------------------------

def google_news_rss_url(query: str) -> str:
    encoded = quote_plus(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"


def fetch_google_news_rss(query: str, lookback_days: int) -> list[dict]:
    url     = google_news_rss_url(query)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; cart-intelligence-bot/1.0)"}
    resp    = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    root   = ET.fromstring(resp.content)
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    results = []

    for item in root.findall(".//item"):
        title        = (item.findtext("title") or "").strip()
        link         = (item.findtext("link") or "").strip()
        pub_date_str = (item.findtext("pubDate") or "").strip()
        source_el    = item.find("{https://news.google.com/rss}source")
        source_name  = source_el.text if source_el is not None else "Google News"

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

        results.append({
            "source":       source_name,
            "title":        title,
            "summary":      "",
            "url":          link,
            "date":         date_str,
            "tags":         [],
            "sowhat":       None,
            "fetch_source": "google_news_rss",
        })

    return results


# ---------------------------------------------------------------------------
# Standard RSS helpers
# ---------------------------------------------------------------------------

def fetch_rss(feed: dict, lookback_days: int, keywords: list[str] = None) -> list[dict]:
    kw      = keywords or RSS_KEYWORDS
    headers = {"User-Agent": "Mozilla/5.0 (compatible; cart-intelligence-bot/1.0)"}
    resp    = requests.get(feed["url"], headers=headers, timeout=30)
    resp.raise_for_status()

    root    = ET.fromstring(resp.content)
    cutoff  = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    results = []

    for item in root.findall(".//item"):
        title        = (item.findtext("title") or "").strip()
        description  = (item.findtext("description") or "").strip()
        link         = (item.findtext("link") or "").strip()
        pub_date_str = (item.findtext("pubDate") or "").strip()

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

        text = (title + " " + description).lower()
        if not any(k in text for k in kw):
            continue

        results.append({
            "source":       feed["name"],
            "title":        title,
            "summary":      description[:500] if description else "",
            "url":          link,
            "date":         date_str,
            "tags":         [],
            "sowhat":       None,
            "fetch_source": "rss",
        })

    return results


def fetch_rss_watchlist(feed: dict, lookback_days: int, watchlist: list[str]) -> list[dict]:
    """Fetch all recent items from a premium RSS feed, filter by watchlist company mention."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; cart-intelligence-bot/1.0)"}
    resp    = requests.get(feed["url"], headers=headers, timeout=30)
    resp.raise_for_status()

    root    = ET.fromstring(resp.content)
    cutoff  = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    results = []

    for item in root.findall(".//item"):
        title        = (item.findtext("title") or "").strip()
        description  = (item.findtext("description") or "").strip()
        link         = (item.findtext("link") or "").strip()
        pub_date_str = (item.findtext("pubDate") or "").strip()

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

        # Include if any watchlist company is mentioned
        combined = title + " " + description
        if not mentions_watchlist_company(combined, watchlist):
            continue

        results.append({
            "source":       feed["name"],
            "title":        title,
            "summary":      description[:500] if description else "",
            "url":          link,
            "date":         date_str,
            "tags":         [],
            "sowhat":       None,
            "fetch_source": "rss_watchlist",
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting news fetch")

    from_date = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    all_news: dict[str, dict] = {}

    # ── Pass 1a — NewsAPI keyword queries ────────────────────────────────────
    if NEWS_API_KEY:
        logging.info("Pass 1a: NewsAPI keyword queries")
        for query in NEWSAPI_QUERIES:
            try:
                articles = fetch_newsapi(query, from_date)
                for a in articles:
                    parsed = parse_newsapi_article(a)
                    url = parsed["url"]
                    if url and url not in all_news:
                        all_news[url] = parsed
                logging.info(f"  '{query}': {len(articles)} articles")
                time.sleep(0.5)
            except requests.RequestException as e:
                logging.error(f"NewsAPI keyword query failed for '{query}': {e}")
    else:
        logging.warning("NEWS_API_KEY not set — skipping NewsAPI")

    # ── Pass 1b — Google News RSS keyword queries ─────────────────────────────
    logging.info("Pass 1b: Google News RSS keyword queries")
    for query in GOOGLE_NEWS_QUERIES:
        try:
            items = fetch_google_news_rss(query, LOOKBACK_DAYS)
            added = 0
            for item in items:
                url = item["url"]
                if url and url not in all_news:
                    all_news[url] = item
                    added += 1
            logging.info(f"  '{query}': {len(items)} items ({added} new)")
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"Google News RSS failed for '{query}': {e}")

    # ── Pass 1c — Standard RSS feeds (keyword filtered) ───────────────────────
    logging.info("Pass 1c: Standard RSS feeds")
    for feed in RSS_FEEDS:
        try:
            items = fetch_rss(feed, LOOKBACK_DAYS)
            for item in items:
                url = item["url"]
                if url and url not in all_news:
                    all_news[url] = item
            logging.info(f"  {feed['name']}: {len(items)} relevant items")
            time.sleep(0.3)
        except Exception as e:
            logging.error(f"RSS fetch failed for {feed['name']}: {e}")

    logging.info(f"After pass 1: {len(all_news)} unique news items")

    # ── Pass 2a — NewsAPI watchlist company queries (relevance filtered) ──────
    watchlist = load_watchlist()
    if NEWS_API_KEY and watchlist:
        logging.info(f"Pass 2a: NewsAPI watchlist queries ({len(watchlist)} companies)")
        new_from_watchlist = 0
        for company in watchlist:
            try:
                articles = fetch_newsapi(f'"{company}"', from_date)
                relevant = [
                    a for a in articles
                    if is_relevant(a.get("title",""), a.get("description",""))
                ]
                for a in relevant:
                    parsed = parse_newsapi_article(a)
                    url = parsed["url"]
                    if url and url not in all_news:
                        all_news[url] = parsed
                        new_from_watchlist += 1
                if relevant:
                    logging.info(f"  {company}: {len(relevant)}/{len(articles)} relevant")
                time.sleep(0.5)
            except requests.RequestException as e:
                logging.error(f"NewsAPI watchlist query '{company}' failed: {e}")
        logging.info(f"  {new_from_watchlist} new items from watchlist pass")

    # ── Pass 2b — STAT News + Endpoints: all articles mentioning watchlist cos ─
    if watchlist:
        logging.info("Pass 2b: Premium RSS feeds (watchlist company filtered)")
        for feed in PREMIUM_RSS_FEEDS:
            try:
                items = fetch_rss_watchlist(feed, LOOKBACK_DAYS, watchlist)
                added = 0
                for item in items:
                    url = item["url"]
                    if url and url not in all_news:
                        all_news[url] = item
                        added += 1
                logging.info(f"  {feed['name']}: {len(items)} watchlist items ({added} new)")
                time.sleep(0.3)
            except Exception as e:
                logging.error(f"Premium RSS fetch failed for {feed['name']}: {e}")

    # ── Preserve existing sowhat / irrelevant flags ───────────────────────────
    if OUTPUT_PATH.exists():
        try:
            existing = json.loads(OUTPUT_PATH.read_text())
            for item in existing.get("news", []):
                url = item.get("url")
                if url and url in all_news:
                    if item.get("sowhat"):
                        all_news[url]["sowhat"] = item["sowhat"]
                    if item.get("irrelevant"):
                        all_news[url]["irrelevant"] = item["irrelevant"]
        except Exception as e:
            logging.warning(f"Could not preserve existing news data: {e}")

    news_list = sorted(all_news.values(), key=lambda n: n["date"] or "", reverse=True)

    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count":      len(news_list),
        "news":       news_list,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logging.info(f"Wrote {len(news_list)} news items to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
