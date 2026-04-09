"""
fetch_dossiers.py
-----------------
Generates AI-synthesised company dossiers for all watchlist companies
by reading all available data (trials, publications, news, funding)
and calling Claude to write a structured summary.

Runs weekly (see .github/workflows/update_dossiers.yml).
Writes structured JSON to data/dossiers.json.

Requires: ANTHROPIC_API_KEY environment variable
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR       = Path(__file__).parent.parent / "data"
TRIALS_PATH    = DATA_DIR / "trials.json"
PUBS_PATH      = DATA_DIR / "publications.json"
NEWS_PATH      = DATA_DIR / "news.json"
FUNDING_PATH   = DATA_DIR / "funding.json"
OUTPUT_PATH    = DATA_DIR / "dossiers.json"
WATCHLIST_PATH = Path(__file__).parent.parent / "watchlist.json"

# Use Sonnet for better synthesis quality
MODEL            = "claude-sonnet-4-20250514"
MAX_TOKENS       = 1200
RATE_LIMIT_DELAY = 1.0

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

DOSSIER_PROMPT = """\
You are a biotech competitive intelligence analyst specialising in cell and gene therapy,
with deep expertise in in vivo CAR-T approaches.

Based on the data below about {company}, write a concise company dossier with exactly
these six sections. Each section should be 2-3 sentences maximum. Be specific and
factual — only state what the data supports. Do not speculate beyond the data.

Return ONLY a valid JSON object with these exact keys:
  "stage_and_modality"     - Current development stage and in vivo delivery approach
  "best_disclosed_data"    - Most advanced or compelling data disclosed (clinical or preclinical)
  "indications_pursued"    - Disease areas and indications the company is targeting
  "delivery_approach"      - Specific delivery mechanism or platform technology
  "key_differentiators"    - What makes this company's approach distinctive vs competitors
  "last_significant_event" - Most recent notable milestone (data, funding, partnership, trial start)

No preamble, no markdown, no extra keys. If a section cannot be populated from the data,
use null for that field.

--- DATA FOR {company} ---

CLINICAL TRIALS ({trial_count} trials):
{trials_text}

PUBLICATIONS ({pub_count} publications):
{pubs_text}

RECENT NEWS ({news_count} items):
{news_text}

FUNDING:
{funding_text}
"""


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_watchlist() -> list[str]:
    try:
        return json.loads(WATCHLIST_PATH.read_text()).get("companies", [])
    except Exception as e:
        logging.warning(f"Could not load watchlist: {e}")
        return []


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def get_company_trials(company: str, all_trials: list) -> list:
    key = company.lower().split()[0]
    return [t for t in all_trials if key in (t.get("sponsor") or "").lower()]


def get_company_pubs(company: str, all_pubs: list) -> list:
    key = company.lower().split()[0]
    return [p for p in all_pubs if key in " ".join([
        p.get("title",""), p.get("abstract",""), p.get("authors","")
    ]).lower()]


def get_company_news(company: str, all_news: list) -> list:
    key = company.lower().split()[0]
    return [n for n in all_news if key in " ".join([
        n.get("title",""), n.get("summary","")
    ]).lower()]


def format_trials(trials: list) -> str:
    if not trials:
        return "No trials found."
    lines = []
    for t in trials[:5]:
        phase  = (t.get("phase") or [])
        phase  = "/".join(phase) if isinstance(phase, list) else str(phase)
        lines.append(
            f"- {t.get('nct_id','')}: {', '.join((t.get('conditions') or [])[:2])} | "
            f"{phase} | {t.get('status','')} | {t.get('modality','')} | "
            f"Asset: {t.get('asset_name') or 'unknown'}\n"
            f"  Summary: {(t.get('summary') or '')[:300]}"
        )
    return "\n".join(lines)


def format_pubs(pubs: list) -> str:
    if not pubs:
        return "No publications found."
    lines = []
    for p in pubs[:5]:
        sowhat = p.get("sowhat","") or ""
        lines.append(
            f"- {p.get('journal','')} ({p.get('date','')}): {p.get('title','')}\n"
            f"  {sowhat or (p.get('abstract') or '')[:200]}"
        )
    return "\n".join(lines)


def format_news(news: list) -> str:
    if not news:
        return "No recent news found."
    lines = []
    for n in news[:8]:
        sowhat = n.get("sowhat","") or ""
        lines.append(
            f"- {n.get('source','')} ({n.get('date','')}): {n.get('title','')}\n"
            f"  {sowhat or (n.get('summary') or '')[:200]}"
        )
    return "\n".join(lines)


def format_funding(company: str, funding_data: dict) -> str:
    company_funding = funding_data.get(company)
    if not company_funding:
        # Try partial match
        key = company.lower().split()[0]
        for k, v in funding_data.items():
            if key in k.lower():
                company_funding = v
                break
    if not company_funding or not company_funding.get("rounds"):
        return "No funding data found."
    lines = []
    total = 0
    for r in company_funding["rounds"]:
        amount = r.get("amount_usd")
        if amount:
            total += amount
        investors = ", ".join((r.get("investors") or [])[:5])
        lines.append(
            f"- {r.get('date','')}: {r.get('round_type','')} | "
            f"{'$'+str(amount)+'M' if amount else 'Undisclosed'} | "
            f"Investors: {investors or 'not disclosed'}"
        )
    if total:
        lines.append(f"Total disclosed: ${total}M")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def generate_dossier(company: str, trials: list, pubs: list,
                     news: list, funding_data: dict) -> dict | None:
    prompt = DOSSIER_PROMPT.format(
        company=company,
        trial_count=len(trials),
        trials_text=format_trials(trials),
        pub_count=len(pubs),
        pubs_text=format_pubs(pubs),
        news_count=len(news),
        news_text=format_news(news),
        funding_text=format_funding(company, funding_data),
    )

    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        dossier = json.loads(raw.strip())
        return dossier
    except Exception as e:
        logging.warning(f"  Dossier generation failed for {company}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Starting dossier generation")

    watchlist    = load_watchlist()
    trials_data  = load_json(TRIALS_PATH)
    pubs_data    = load_json(PUBS_PATH)
    news_data    = load_json(NEWS_PATH)
    funding_data = load_json(FUNDING_PATH).get("companies", {})

    all_trials = trials_data.get("studies", [])
    all_pubs   = pubs_data.get("publications", [])
    all_news   = news_data.get("news", [])

    logging.info(f"Loaded: {len(all_trials)} trials, {len(all_pubs)} pubs, "
                 f"{len(all_news)} news, {len(funding_data)} funded companies")

    all_dossiers = {}
    success_count = 0

    for company in watchlist:
        logging.info(f"Generating dossier: {company}")

        trials  = get_company_trials(company, all_trials)
        pubs    = get_company_pubs(company, all_pubs)
        news    = get_company_news(company, all_news)

        logging.info(f"  Data: {len(trials)} trials, {len(pubs)} pubs, {len(news)} news items")

        # Skip if no data at all
        if not trials and not pubs and not news:
            logging.info(f"  No data found — skipping")
            all_dossiers[company] = {
                "company":      company,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "has_data":     False,
                "dossier":      None,
            }
            continue

        dossier = generate_dossier(company, trials, pubs, news, funding_data)

        if dossier:
            all_dossiers[company] = {
                "company":      company,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "has_data":     True,
                "dossier":      dossier,
            }
            success_count += 1
            logging.info(f"  Done")
        else:
            all_dossiers[company] = {
                "company":      company,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "has_data":     True,
                "dossier":      None,
            }

        time.sleep(RATE_LIMIT_DELAY)

    if not all_dossiers:
        logging.error("No dossiers generated — aborting write")
        return

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count":        len(all_dossiers),
        "dossiers":     all_dossiers,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logging.info(f"Wrote {success_count}/{len(watchlist)} dossiers to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
