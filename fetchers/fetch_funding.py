"""
fetch_funding.py
----------------
Fetches funding information for watchlist companies by:
  1. Searching NewsAPI for funding announcements per company
  2. Passing results to Claude to extract structured funding data
     (round type, amount, date, investors)

Writes structured JSON to data/funding.json.

Requires: NEWS_API_KEY, ANTHROPIC_API_KEY environment variables
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_PATH    = Path(__file__).parent.parent / "data" / "funding.json"
WATCHLIST_PATH = Path(__file__).parent.parent / "watchlist.json"

NEWS_API_KEY  = os.environ.get("NEWS_API_KEY", "")
NEWSAPI_BASE  = "https://newsapi.org/v2/everything"

# Look back further for funding since rounds are infrequent
FUNDING_LOOKBACK_DAYS = 730  # 2 years

MODEL = "claude-haiku-4-5-20251001"

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

FUNDING_EXTRACT_PROMPT = """\
You are a biotech financial analyst. Given the following news articles about {company},
extract all funding rounds mentioned.

For each round return a JSON array where each element has these fields:
  - date: ISO date string (YYYY-MM-DD or YYYY-MM or YYYY)
  - round_type: e.g. "Series A", "Series B", "Seed", "Grant", "IPO", "Undisclosed"
  - amount_usd: numeric value in millions USD, or null if not disclosed
  - investors: array of investor name strings (empty array if none mentioned)
  - source: name of the publication that reported it
  - url: URL of the article

Rules:
- Return ONLY a valid JSON array, no preamble, no markdown backticks
- If no funding rounds are mentioned, return an empty array []
- Do not duplicate rounds — if the same round is mentioned in multiple articles,
  return it once using the most detailed mention
- Sort by date descending (most recent first)

Articles:
{articles}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_watchlist() -> list[str]:
    try:
        data = json.loads(WATCHLIST_PATH.read_text())
        return data.get("companies", [])
    except Exception as e:
        logging.warning(f"Could not load watchlist: {e}")
        return []


def search_funding_news(company: str, lookback_days: int) -> list[dict]:
    """Search NewsAPI for funding-related articles about a company."""
    if not NEWS_API_KEY:
        return []

    from datetime import timedelta
    from_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # Two queries — direct funding language and general company coverage
    queries = [
        f'"{company}" raises',
        f'"{company}" funding',
        f'"{company}" Series',
        f'"{company}" investment',
    ]

    all_articles = {}
    for query in queries:
        try:
            params = {
                "q":        query,
                "from":     from_date,
                "sortBy":   "relevancy",
                "pageSize": 10,
                "language": "en",
                "apiKey":   NEWS_API_KEY,
            }
            resp = requests.get(NEWSAPI_BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "ok":
                for a in data.get("articles", []):
                    url = a.get("url", "")
                    if url and url not in all_articles:
                        all_articles[url] = a
            time.sleep(0.3)
        except Exception as e:
            logging.warning(f"  NewsAPI query '{query}' failed: {e}")

    return list(all_articles.values())


def extract_funding_with_claude(company: str, articles: list[dict]) -> list[dict]:
    """Pass articles to Claude to extract structured funding data."""
    if not articles:
        return []

    # Format articles for the prompt
    article_texts = []
    for a in articles[:10]:  # cap at 10 articles
        title   = a.get("title", "")
        desc    = a.get("description", "") or ""
        source  = a.get("source", {}).get("name", "")
        url     = a.get("url", "")
        date    = (a.get("publishedAt", "") or "")[:10]
        article_texts.append(f"Source: {source} | Date: {date} | URL: {url}\nTitle: {title}\n{desc}")

    prompt = FUNDING_EXTRACT_PROMPT.format(
        company=company,
        articles="\n\n---\n\n".join(article_texts),
    )

    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        rounds = json.loads(raw.strip())
        return rounds if isinstance(rounds, list) else []
    except Exception as e:
        logging.warning(f"  Claude extraction failed for {company}: {e}")
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting funding fetch")

    watchlist = load_watchlist()
    logging.info(f"Processing {len(watchlist)} watchlist companies")

    # Load existing data to preserve manually verified entries
    existing: dict[str, dict] = {}
    if OUTPUT_PATH.exists():
        try:
            existing = json.loads(OUTPUT_PATH.read_text())
        except Exception:
            pass

    all_funding: dict[str, dict] = {}

    for company in watchlist:
        logging.info(f"  Fetching funding for: {company}")

        articles = search_funding_news(company, FUNDING_LOOKBACK_DAYS)
        logging.info(f"    Found {len(articles)} relevant articles")

        rounds = extract_funding_with_claude(company, articles)
        logging.info(f"    Extracted {len(rounds)} funding rounds")

        all_funding[company] = {
            "company":      company,
            "rounds":       rounds,
            "last_fetched": datetime.now(timezone.utc).isoformat(),
        }

        time.sleep(0.5)

    # Safety guard
    if not all_funding:
        logging.error("No funding data extracted — aborting write")
        return

    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "companies":  all_funding,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logging.info(f"Wrote funding data for {len(all_funding)} companies to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
