"""
fetch_news.py
-------------
Fetches in vivo CAR-T news and funding announcements from multiple sources:

  Pass 1a — NewsAPI keyword queries (exact phrases, in vivo CAR-T terms)
  Pass 1b — Google News RSS keyword queries (broad outlet coverage)
  Pass 1c — RSS feeds from BioPharma Dive (keyword filtered)
  Pass 2  — Single NewsAPI query per watchlist company (covers both news
             AND funding — replaces separate watchlist passes in both
             fetch_news.py and fetch_funding.py)
  Pass 3  — STAT News + Endpoints RSS: all articles mentioning watchlist cos

Each item gets a 'item_type' field ('news' or 'funding') assigned by Claude
during summarisation based on article content.

Articles are preserved until they age past LOOKBACK_DAYS, regardless of
whether NewsAPI returns them on a given run.

Writes structured JSON to data/news.json.

Requires: NEWS_API_KEY environment variable
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

# Pass 1a — keyword queries
NEWSAPI_QUERIES = [
    '"in vivo CAR-T"',
    '"in vivo CAR T"',
    '"lipid nanoparticle CAR"',
    '"lentiviral CAR T"',
    '"non-viral CAR T"',
]

# Pass 1b — Google News RSS
GOOGLE_NEWS_QUERIES = [
    "in vivo CAR-T",
    "in vivo CAR T cell therapy",
    "lipid nanoparticle CAR-T",
    "lentiviral CAR T in vivo",
    "non-viral CAR T",
    "in vivo chimeric antigen receptor",
]

# Pass 1c — standard RSS feeds (keyword filtered)
RSS_FEEDS = [
    {"name": "BioPharma Dive", "url": "https://www.biopharmadive.com/feeds/news/"},
]

# Pass 3 — premium RSS feeds (watchlist company filtered only)
PREMIUM_RSS_FEEDS = [
    {"name": "STAT News",      "url": "https://www.statnews.com/feed/"},
    {"name": "Endpoints News", "url": "https://endpts.com/feed/"},
]

RSS_KEYWORDS = [
    "in vivo car-t", "in vivo car t", "lipid nanoparticle car",
    "lentiviral car", "non-viral car", "car-t delivery", "in vivo t cell",
    "in vivo chimeric antigen receptor",
]

# Relevance filter for Pass 2 results
RELEVANCE_SIGNALS = [
    "car-t", "car t cell", "chimeric antigen receptor",
    "cell therapy", "gene therapy", "t cell",
    "lentiviral", "lipid nanoparticle", "in vivo",
    "immunotherapy", "oncology", "hematol",
    "clinical trial", "phase 1", "phase 2",
    "raises", "funding", "series", "investment", "million",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_watchlist() -> list[str]:
    try:
        return json.loads(WATCHLIST_PATH.read_text()).get("companies", [])
    except Exception as e:
        logging.warning(f"Could not load watchlist: {e}")
        return []


def is_relevant(title: str, desc: str) -> bool:
    text = (title + " " + (desc or "")).lower()
    return any(s in text for s in RELEVANCE_SIGNALS)


def mentions_watchlist(text: str, watchlist: list[str]) -> bool:
    tl = text.lower()
    return any(c.lower() in tl for c in watchlist)


def google_news_rss_url(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"


# ---------------------------------------------------------------------------
# NewsAPI
# ---------------------------------------------------------------------------

def fetch_newsapi(query: str, from_date: str) -> list[dict]:
    if not NEWS_API_KEY:
        return []
    params = {
        "q": query, "from": from_date, "sortBy": "publishedAt",
        "pageSize": MAX_NEWSAPI_RESULTS, "language": "en", "apiKey": NEWS_API_KEY,
    }
    resp = requests.get(NEWSAPI_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "ok":
        logging.warning(f"NewsAPI error for '{query}': {data.get('message')}")
        return []
    return data.get("articles", [])


def parse_newsapi(article: dict, item_type: str = "news") -> dict:
    return {
        "source":       article.get("source", {}).get("name", "Unknown"),
        "title":        article.get("title", ""),
        "summary":      article.get("description", ""),
        "url":          article.get("url", ""),
        "date":         (article.get("publishedAt", "") or "")[:10],
        "tags":         [],
        "sowhat":       None,
        "item_type":    item_type,
        "fetch_source": "newsapi",
    }


# ---------------------------------------------------------------------------
# Google News RSS
# ---------------------------------------------------------------------------

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
            "source": source_name, "title": title, "summary": "",
            "url": link, "date": date_str, "tags": [],
            "sowhat": None, "item_type": "news", "fetch_source": "google_news_rss",
        })

    return results


# ---------------------------------------------------------------------------
# RSS feeds
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

        if not any(k in (title + " " + description).lower() for k in kw):
            continue

        results.append({
            "source": feed["name"], "title": title,
            "summary": description[:500] if description else "",
            "url": link, "date": date_str, "tags": [],
            "sowhat": None, "item_type": "news", "fetch_source": "rss",
        })

    return results


def fetch_rss_watchlist(feed: dict, lookback_days: int, watchlist: list[str]) -> list[dict]:
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

        if not mentions_watchlist(title + " " + description, watchlist):
            continue

        results.append({
            "source": feed["name"], "title": title,
            "summary": description[:500] if description else "",
            "url": link, "date": date_str, "tags": [],
            "sowhat": None, "item_type": "news", "fetch_source": "rss_watchlist",
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting news + funding fetch")

    from_date = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    cutoff    = from_date  # for date-based expiry
    all_news: dict[str, dict] = {}

    # ── Pass 1a — NewsAPI keyword queries ─────────────────────────────────────
    if NEWS_API_KEY:
        logging.info("Pass 1a: NewsAPI keyword queries")
        for query in NEWSAPI_QUERIES:
            try:
                articles = fetch_newsapi(query, from_date)
                for a in articles:
                    p = parse_newsapi(a, "news")
                    if p["url"] and p["url"] not in all_news:
                        all_news[p["url"]] = p
                logging.info(f"  '{query}': {len(articles)} articles")
                time.sleep(1.5)
            except requests.RequestException as e:
                logging.error(f"NewsAPI keyword failed for '{query}': {e}")

    # ── Pass 1b — Google News RSS ─────────────────────────────────────────────
    logging.info("Pass 1b: Google News RSS")
    for query in GOOGLE_NEWS_QUERIES:
        try:
            items = fetch_google_news_rss(query, LOOKBACK_DAYS)
            added = sum(1 for i in items if i["url"] and i["url"] not in all_news)
            for i in items:
                if i["url"] and i["url"] not in all_news:
                    all_news[i["url"]] = i
            logging.info(f"  '{query}': {len(items)} items ({added} new)")
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"Google News RSS failed for '{query}': {e}")

    # ── Pass 1c — Standard RSS feeds ─────────────────────────────────────────
    logging.info("Pass 1c: Standard RSS feeds")
    for feed in RSS_FEEDS:
        try:
            items = fetch_rss(feed, LOOKBACK_DAYS)
            for i in items:
                if i["url"] and i["url"] not in all_news:
                    all_news[i["url"]] = i
            logging.info(f"  {feed['name']}: {len(items)} items")
            time.sleep(0.3)
        except Exception as e:
            logging.error(f"RSS failed for {feed['name']}: {e}")

    logging.info(f"After pass 1: {len(all_news)} unique items")

    # ── Pass 2 — Single NewsAPI query per watchlist company ───────────────────
    # One bare company name query covers both news AND funding announcements
    watchlist = load_watchlist()
    if NEWS_API_KEY and watchlist:
        logging.info(f"Pass 2: NewsAPI watchlist queries ({len(watchlist)} companies, 1 query each)")
        new_from_watchlist = 0
        for company in watchlist:
            try:
                articles = fetch_newsapi(f'"{company}"', from_date)
                relevant = [a for a in articles if is_relevant(a.get("title",""), a.get("description",""))]
                for a in relevant:
                    p = parse_newsapi(a, "news")  # item_type assigned by Claude in summarise
                    if p["url"] and p["url"] not in all_news:
                        all_news[p["url"]] = p
                        new_from_watchlist += 1
                if relevant:
                    logging.info(f"  {company}: {len(relevant)}/{len(articles)} relevant")
                time.sleep(1.5)
            except requests.RequestException as e:
                logging.error(f"NewsAPI watchlist query '{company}' failed: {e}")
        logging.info(f"  {new_from_watchlist} new items from watchlist pass")

    # ── Pass 3 — Premium RSS watchlist pass ───────────────────────────────────
    if watchlist:
        logging.info("Pass 3: Premium RSS watchlist pass")
        for feed in PREMIUM_RSS_FEEDS:
            try:
                items = fetch_rss_watchlist(feed, LOOKBACK_DAYS, watchlist)
                added = sum(1 for i in items if i["url"] and i["url"] not in all_news)
                for i in items:
                    if i["url"] and i["url"] not in all_news:
                        all_news[i["url"]] = i
                logging.info(f"  {feed['name']}: {len(items)} items ({added} new)")
                time.sleep(0.3)
            except Exception as e:
                logging.error(f"Premium RSS failed for {feed['name']}: {e}")

    # ── Preserve existing items within lookback window ────────────────────────
    if OUTPUT_PATH.exists():
        try:
            existing = json.loads(OUTPUT_PATH.read_text())
            preserved = 0
            for item in existing.get("news", []):
                url = item.get("url")
                if not url:
                    continue
                if (item.get("date") or "") < cutoff:
                    continue  # aged out
                if url in all_news:
                    # Keep existing sowhat, item_type, irrelevant flags
                    if item.get("sowhat"):
                        all_news[url]["sowhat"] = item["sowhat"]
                    if item.get("irrelevant"):
                        all_news[url]["irrelevant"] = item["irrelevant"]
                    if item.get("item_type") and item["item_type"] != "news":
                        all_news[url]["item_type"] = item["item_type"]
                else:
                    # Carry forward — not returned this run but still within window
                    all_news[url] = item
                    preserved += 1
            logging.info(f"  Preserved {preserved} items not returned this run")
        except Exception as e:
            logging.warning(f"Could not preserve existing data: {e}")

    news_list = sorted(all_news.values(), key=lambda n: n["date"] or "", reverse=True)

    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count":      len(news_list),
        "news":       news_list,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logging.info(f"Wrote {len(news_list)} items to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
