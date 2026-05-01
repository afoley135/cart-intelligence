"""
send_digest.py
--------------
Generates and sends the daily in vivo CAR-T intelligence email digest.

Reads from:
  data/trial_changes.json   — new trials and status changes (watchlist only)
  data/publications.json    — recent publications
  data/news.json            — news and funding items
  data/patents.json         — new patent assignments
  data/abstracts.json       — conference abstracts
  data/conferences.json     — conference calendar
  data/website_changes.json — competitor website changes

Calls Claude (Haiku) once to generate the overall summary paragraph,
then assembles a plain-HTML email and sends via SendGrid.

Requires env vars:
  ANTHROPIC_API_KEY
  SENDGRID_API_KEY
  DIGEST_TO_EMAIL        — recipient address (comma-separated for multiple)
  DIGEST_FROM_EMAIL      — verified sender address in SendGrid

Run manually or triggered by GitHub Actions after the daily pipeline.
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import anthropic
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR       = Path(__file__).parent.parent / "data"
TRIALS_CHANGES = DATA_DIR / "trial_changes.json"
PUBS_PATH      = DATA_DIR / "publications.json"
NEWS_PATH      = DATA_DIR / "news.json"
PATENTS_PATH   = DATA_DIR / "patents.json"
ABSTRACTS_PATH        = DATA_DIR / "abstracts.json"
CONF_APPEARANCES_PATH = DATA_DIR / "conference_appearances.json"
WEBSITE_CHANGES       = DATA_DIR / "website_changes.json"

MODEL = "claude-haiku-4-5-20251001"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SENDGRID_API_KEY  = os.environ.get("SENDGRID_API_KEY", "")
TO_EMAIL          = os.environ.get("DIGEST_TO_EMAIL", "")
FROM_EMAIL        = os.environ.get("DIGEST_FROM_EMAIL", "")

DASHBOARD_URL = "https://afoley135.github.io/cart-intelligence/"

# News source priority order for the digest
# Sources matching these strings (case-insensitive) get priority slots
PRIORITY_SOURCES = ["endpoints", "fierce", "stat"]

# How far back to look for "new" items (patents, pubs use filing/pub date)
LOOKBACK_DAYS = 1  # daily digest — only today's run items


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load(path: Path, key: str) -> list:
    try:
        if path.exists():
            return json.loads(path.read_text()).get(key, [])
    except Exception as e:
        logging.warning(f"Could not load {path}: {e}")
    return []


def load_conference_appearances() -> list:
    try:
        if CONF_APPEARANCES_PATH.exists():
            return json.loads(CONF_APPEARANCES_PATH.read_text()).get("appearances", [])
    except Exception as e:
        logging.warning(f"Could not load conference appearances: {e}")
    return []


def is_recent(date_str: str, days: int = LOOKBACK_DAYS) -> bool:
    if not date_str:
        return False
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        # Handle both date-only and datetime strings
        d = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d >= cutoff
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def build_trial_changes_section(changes: list) -> tuple[str, list]:
    """
    Returns (html_section, plain_items_for_summary).
    Only surfaces new_trial and status_change, unreviewed, detected today.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    relevant = [
        c for c in changes
        if c.get("change_type") in ("new_trial", "status_change")
        and not c.get("reviewed", False)
        and c.get("detected_at", "")[:10] == today
    ]

    if not relevant:
        return "", []

    plain_items = []
    rows = []
    for c in relevant:
        sponsor  = c.get("sponsor", "")
        title    = c.get("title", "")
        asset    = c.get("asset_name") or ""
        url      = c.get("url", "")
        nct_id   = c.get("nct_id", "")
        asset_str = f" ({asset})" if asset else ""

        if c["change_type"] == "new_trial":
            status   = c.get("status", "")
            phase    = "/".join(c.get("phase", [])).replace("PHASE", "Ph ") or "N/A"
            conditions = ", ".join(c.get("conditions", [])[:2]) or "—"
            label    = "New trial"
            detail   = f"{phase} · {status} · {conditions}"
            plain_items.append(f"New trial: {sponsor}{asset_str} — {title[:60]}")
        else:
            old_s = c.get("status_old", "")
            new_s = c.get("status_new", "")
            label = "Status change"
            detail = f"{old_s} → {new_s}"
            plain_items.append(f"Status change: {sponsor}{asset_str} — {old_s} → {new_s}")

        link = f'<a href="{url}" style="color:#1D9E75;text-decoration:none;">{nct_id}</a>' if url else nct_id
        rows.append(f"""
        <tr>
          <td style="padding:8px 12px;border-bottom:1px solid #f0efe9;font-size:13px;color:#5f5e5a;white-space:nowrap;">{label}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #f0efe9;font-size:13px;font-weight:500;color:#1a1a18;">{sponsor}{asset_str}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #f0efe9;font-size:13px;color:#1a1a18;">{title[:70]}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #f0efe9;font-size:13px;color:#5f5e5a;">{detail}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #f0efe9;font-size:13px;">{link}</td>
        </tr>""")

    html = f"""
    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;border:1px solid #e8e7e2;border-radius:8px;overflow:hidden;">
      <thead>
        <tr style="background:#f7f6f3;">
          <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:500;color:#9b9a96;text-transform:uppercase;letter-spacing:0.05em;">Type</th>
          <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:500;color:#9b9a96;text-transform:uppercase;letter-spacing:0.05em;">Sponsor</th>
          <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:500;color:#9b9a96;text-transform:uppercase;letter-spacing:0.05em;">Trial</th>
          <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:500;color:#9b9a96;text-transform:uppercase;letter-spacing:0.05em;">Change</th>
          <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:500;color:#9b9a96;text-transform:uppercase;letter-spacing:0.05em;">Link</th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""

    return html, plain_items


def build_publications_section(pubs: list) -> tuple[str, list]:
    """Up to 5 pubs: PubMed first, then bioRxiv to fill."""
    recent = [p for p in pubs if is_recent(p.get("date", ""), days=2)]

    # Priority: peer-reviewed first, then preprints
    peer_reviewed = [p for p in recent if not p.get("preprint")]
    preprints     = [p for p in recent if p.get("preprint")]
    selected      = (peer_reviewed + preprints)[:5]

    if not selected:
        return "", []

    plain_items = []
    cards = []
    for p in selected:
        title   = p.get("title", "")
        journal = p.get("journal", "")
        sowhat  = p.get("sowhat", "") or ""
        url     = p.get("url", "")
        date    = (p.get("date", "") or "")[:7]
        is_pre  = p.get("preprint", False)
        badge   = "Preprint" if is_pre else "Peer-reviewed"
        badge_color = "#FAEEDA" if is_pre else "#E1F5EE"
        badge_text  = "#854F0B" if is_pre else "#0F6E56"

        title_html = f'<a href="{url}" style="color:#1a1a18;text-decoration:none;font-weight:500;">{title}</a>' if url else f'<span style="font-weight:500;">{title}</span>'

        sowhat_html = ""
        if sowhat and sowhat != "Abstract not available":
            sowhat_html = f'<div style="margin-top:6px;font-size:12px;color:#185FA5;background:#E6F1FB;border-radius:4px;padding:4px 10px;display:inline-block;">→ {sowhat}</div>'
            plain_items.append(f"{title[:70]} — {sowhat}")
        else:
            plain_items.append(title[:70])

        cards.append(f"""
        <div style="padding:14px 0;border-bottom:1px solid #f0efe9;">
          <div style="margin-bottom:6px;">
            <span style="font-size:11px;font-weight:600;color:#5f5e5a;text-transform:uppercase;letter-spacing:0.05em;">{journal}</span>
            <span style="font-size:11px;color:#9b9a96;margin-left:8px;">{date}</span>
            <span style="font-size:11px;font-weight:500;padding:2px 8px;border-radius:20px;background:{badge_color};color:{badge_text};margin-left:8px;">{badge}</span>
          </div>
          <div style="font-size:14px;line-height:1.4;">{title_html}</div>
          {sowhat_html}
        </div>""")

    html = f'<div>{"".join(cards)}</div>'
    return html, plain_items


def build_news_section(news: list) -> tuple[str, str]:
    """
    Priority sources get up to 5 slots total.
    Watchlist newswire items follow separately (all of them).
    Returns (html, plain_text_for_summary).
    """
    recent = [n for n in news if is_recent(n.get("date", ""), days=2)]

    # Priority outlet items
    priority = []
    for source_kw in PRIORITY_SOURCES:
        for item in recent:
            if source_kw.lower() in (item.get("source") or "").lower():
                if item not in priority:
                    priority.append(item)
        if len(priority) >= 5:
            break
    priority = priority[:5]

    # Watchlist newswire items — all of them, deduplicated from priority
    priority_urls = {n.get("url") for n in priority}
    newswire_sources = {"globenewswire", "businesswire", "prnewswire", "pr newswire",
                        "globe newswire", "business wire", "accesswire"}
    watchlist_wire = [
        n for n in recent
        if any(s in (n.get("source") or "").lower() for s in newswire_sources)
        and n.get("url") not in priority_urls
    ]

    def render_card(item: dict, compact: bool = False) -> str:
        title   = item.get("title", "")
        source  = item.get("source", "")
        date    = item.get("date", "")
        url     = item.get("url", "")
        sowhat  = item.get("sowhat", "") or ""
        itype   = item.get("item_type", "news")

        title_html = f'<a href="{url}" style="color:#1a1a18;text-decoration:none;font-weight:500;font-size:{"13px" if compact else "14px"};">{title}</a>' if url else title
        sowhat_html = ""
        if sowhat and sowhat != "Summary not available" and not compact:
            sowhat_html = f'<div style="margin-top:5px;font-size:12px;color:#185FA5;background:#E6F1FB;border-radius:4px;padding:4px 10px;display:inline-block;">→ {sowhat}</div>'

        type_badge = ""
        if itype == "funding":
            type_badge = '<span style="font-size:11px;font-weight:500;padding:2px 8px;border-radius:20px;background:#FAEEDA;color:#854F0B;margin-left:8px;">Funding</span>'

        return f"""
        <div style="padding:{"10px" if compact else "14px"} 0;border-bottom:1px solid #f0efe9;">
          <div style="margin-bottom:5px;">
            <span style="font-size:11px;font-weight:600;color:#5f5e5a;text-transform:uppercase;letter-spacing:0.05em;">{source}</span>
            <span style="font-size:11px;color:#9b9a96;margin-left:8px;">{date}</span>
            {type_badge}
          </div>
          <div style="line-height:1.4;">{title_html}</div>
          {sowhat_html}
        </div>"""

    priority_html = "".join(render_card(n) for n in priority)
    wire_html = ""
    if watchlist_wire:
        wire_items = "".join(render_card(n, compact=True) for n in watchlist_wire)
        wire_html = f"""
        <div style="margin-top:16px;">
          <div style="font-size:11px;font-weight:600;color:#9b9a96;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;">Watchlist company newswire</div>
          {wire_items}
        </div>"""

    html = f'<div>{priority_html}{wire_html}</div>'

    plain = "; ".join(
        (n.get("sowhat") or n.get("title", ""))[:60]
        for n in (priority + watchlist_wire)[:6]
    )
    return html, plain


def build_conferences_section(appearances: list) -> str:
    """
    Show upcoming conferences (next 90 days) with watchlist companies presenting.
    Grouped by conference, sorted chronologically.
    """
    today = datetime.now(timezone.utc).date()
    cutoff = today + timedelta(days=90)

    # Filter to upcoming appearances with a known conference start date
    upcoming = []
    for a in appearances:
        start_str = a.get("conference_start")
        if not start_str:
            continue
        try:
            start = datetime.strptime(start_str, "%Y-%m-%d").date()
        except Exception:
            continue
        if today <= start <= cutoff:
            upcoming.append((start, a))

    if not upcoming:
        return ""

    # Group by conference name
    groups: dict = {}
    for start, a in sorted(upcoming, key=lambda x: x[0]):
        key = a.get("conference") or "Unknown"
        if key not in groups:
            groups[key] = {"start": start, "meta": a, "items": []}
        groups[key]["items"].append(a)

    pt_label = {
        "oral":        ("Oral",        "#E1F5EE", "#0F6E56"),
        "poster":      ("Poster",      "#E6F1FB", "#185FA5"),
        "invited":     ("Invited",     "#EDE9FE", "#4C1D95"),
        "unspecified": ("",            "",        ""),
    }

    cards = []
    for conf_name, group in groups.items():
        start     = group["start"]
        meta      = group["meta"]
        end_str   = meta.get("conference_end")
        location  = meta.get("conference_location") or ""
        days_away = (start - today).days

        if days_away == 0:
            timing = "Today"
        elif days_away == 1:
            timing = "Tomorrow"
        else:
            timing = f"In {days_away} days"

        try:
            end = datetime.strptime(end_str, "%Y-%m-%d").date() if end_str else start
            date_range = (
                f"{start.strftime('%b %-d')}–{end.strftime('%-d, %Y')}"
                if end != start else start.strftime("%b %-d, %Y")
            )
        except Exception:
            date_range = str(start)

        # Presenter rows
        presenter_rows = []
        for a in group["items"]:
            company = a.get("company") or "Unknown"
            pt = a.get("presentation_type") or "unspecified"
            label, bg, fg = pt_label.get(pt, ("", "", ""))
            badge = (
                f'<span style="font-size:11px;font-weight:500;padding:2px 8px;'
                f'border-radius:20px;background:{bg};color:{fg};margin-left:8px;">'
                f'{label}</span>'
            ) if label else ""
            abstract = a.get("abstract_title") or ""
            src_url  = a.get("source_url") or ""
            link = (
                f' <a href="{src_url}" style="color:#185FA5;font-size:11px;'
                f'text-decoration:none;margin-left:8px;">Source →</a>'
            ) if src_url else ""
            abstract_html = (
                f'<div style="font-size:12px;color:#5f5e5a;font-style:italic;margin-top:2px;">'
                f'{abstract}</div>'
            ) if abstract else ""
            presenter_rows.append(f"""
            <div style="padding:8px 0;border-bottom:1px solid #f7f6f3;">
              <div style="font-size:13px;font-weight:500;color:#1a1a18;">
                {company}{badge}{link}
              </div>
              {abstract_html}
            </div>""")

        cards.append(f"""
        <div style="margin-bottom:20px;border:1px solid #e8e7e2;border-radius:8px;overflow:hidden;">
          <div style="background:#f7f6f3;padding:12px 16px;display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;">
            <div>
              <span style="font-size:15px;font-weight:600;color:#1a1a18;">{conf_name}</span>
              <span style="font-size:12px;color:#5f5e5a;margin-left:10px;">{date_range}{' · ' + location if location else ''}</span>
            </div>
            <span style="font-size:12px;font-weight:500;color:#185FA5;">{timing}</span>
          </div>
          <div style="padding:0 16px;">
            {"".join(presenter_rows)}
          </div>
          <div style="padding:8px 16px;font-size:12px;color:#9b9a96;">
            {len(group["items"])} watchlist presenter{"s" if len(group["items"]) != 1 else ""}
          </div>
        </div>""")

    return f'<div>{"".join(cards)}</div>'


def build_patents_section(patents: list) -> tuple[str, list]:
    """New patents detected in the last 2 days, watchlist companies only."""
    recent = [
        p for p in patents
        if p.get("watchlist_company")
        and is_recent(p.get("filing_date", ""), days=7)
        and p.get("relevant") is not False
    ]

    if not recent:
        return "", []

    plain_items = []
    cards = []
    for p in recent:
        title   = p.get("title", "")
        company = p.get("watchlist_company", "")
        assignee = p.get("assignee", "")
        patent_no = p.get("patent_number", "")
        filing   = p.get("filing_date", "")
        claim    = p.get("claim_type", "")
        novelty  = p.get("novelty_summary", "") or ""
        url      = p.get("url", "")

        title_html = f'<a href="{url}" style="color:#1a1a18;text-decoration:none;font-weight:500;">{title}</a>' if url else f'<span style="font-weight:500;">{title}</span>'
        claim_html = f'<span style="font-size:11px;font-weight:500;padding:2px 8px;border-radius:20px;background:#E1F5EE;color:#0F6E56;">{claim}</span>' if claim else ""
        novelty_html = f'<div style="margin-top:5px;font-size:12px;color:#185FA5;background:#E6F1FB;border-radius:4px;padding:4px 10px;">→ {novelty}</div>' if novelty else ""

        plain_items.append(f"{company}: {title[:60]}")
        cards.append(f"""
        <div style="padding:14px 0;border-bottom:1px solid #f0efe9;">
          <div style="margin-bottom:6px;">
            <span style="font-size:11px;font-weight:600;color:#5f5e5a;text-transform:uppercase;letter-spacing:0.05em;">{company}</span>
            <span style="font-size:11px;color:#9b9a96;margin-left:8px;">{patent_no} · Filed {filing}</span>
            <span style="margin-left:8px;">{claim_html}</span>
          </div>
          <div style="font-size:14px;line-height:1.4;">{title_html}</div>
          {novelty_html}
        </div>""")

    html = f'<div>{"".join(cards)}</div>'
    return html, plain_items


def build_website_changes_section(changes: list) -> str:
    """Unreviewed website changes from watchlist companies."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    recent = [
        c for c in changes
        if not c.get("reviewed", False)
        and c.get("detected_at", "")[:10] == today
    ]

    if not recent:
        return ""

    rows = []
    for c in recent:
        company   = c.get("company", "")
        page_type = c.get("page_type", "")
        magnitude = c.get("change_magnitude", "")
        url       = c.get("url", "")
        delta     = c.get("word_count_delta", 0)
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        mag_color = {"major": "#993556", "moderate": "#854F0B", "minor": "#5f5e5a"}.get(magnitude, "#5f5e5a")
        link = f'<a href="{url}" style="color:#185FA5;font-size:12px;text-decoration:none;">Review →</a>' if url else ""

        rows.append(f"""
        <tr>
          <td style="padding:8px 12px;border-bottom:1px solid #f0efe9;font-size:13px;font-weight:500;color:#1a1a18;">{company}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #f0efe9;font-size:13px;color:#5f5e5a;text-transform:capitalize;">{page_type}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #f0efe9;font-size:13px;font-weight:500;color:{mag_color};text-transform:capitalize;">{magnitude}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #f0efe9;font-size:13px;color:#5f5e5a;">{delta_str} words</td>
          <td style="padding:8px 12px;border-bottom:1px solid #f0efe9;">{link}</td>
        </tr>""")

    return f"""
    <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;border:1px solid #e8e7e2;border-radius:8px;overflow:hidden;">
      <thead>
        <tr style="background:#f7f6f3;">
          <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:500;color:#9b9a96;text-transform:uppercase;letter-spacing:0.05em;">Company</th>
          <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:500;color:#9b9a96;text-transform:uppercase;letter-spacing:0.05em;">Page</th>
          <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:500;color:#9b9a96;text-transform:uppercase;letter-spacing:0.05em;">Magnitude</th>
          <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:500;color:#9b9a96;text-transform:uppercase;letter-spacing:0.05em;">Change</th>
          <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:500;color:#9b9a96;text-transform:uppercase;letter-spacing:0.05em;"></th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


# ---------------------------------------------------------------------------
# AI summary
# ---------------------------------------------------------------------------

def generate_summary(
    trial_items: list,
    pub_items: list,
    news_plain: str,
    patent_items: list,
) -> str:
    """Call Claude to write the 2-3 sentence opening summary."""
    if not ANTHROPIC_API_KEY:
        return "In vivo CAR-T intelligence digest."

    bullets = []
    if trial_items:
        bullets.append("Trial updates: " + "; ".join(trial_items[:3]))
    if pub_items:
        bullets.append("Publications: " + "; ".join(pub_items[:3]))
    if news_plain:
        bullets.append("News: " + news_plain[:200])
    if patent_items:
        bullets.append("Patents: " + "; ".join(patent_items[:2]))

    if not bullets:
        return "No significant updates today."

    prompt = f"""You are writing the opening summary for a daily in vivo CAR-T competitive intelligence email.

Based on today's updates below, write 2-3 sentences that are pithy, direct, and tell the reader where to focus their attention. No preamble. No sign-off. Just the sentences.

Today's updates:
{chr(10).join(bullets)}"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logging.warning(f"Summary generation failed: {e}")
        return "In vivo CAR-T intelligence digest — see sections below."


# ---------------------------------------------------------------------------
# Email assembly
# ---------------------------------------------------------------------------

def section_wrapper(title: str, content: str, accent: str = "#1D9E75") -> str:
    """Wrap a content block with a labelled section header."""
    return f"""
    <div style="margin-bottom:28px;">
      <div style="font-size:11px;font-weight:600;color:{accent};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px;padding-bottom:8px;border-bottom:2px solid {accent};">
        {title}
      </div>
      {content}
    </div>"""


def build_email(
    summary: str,
    trial_html: str,
    pubs_html: str,
    conferences_html: str,
    news_html: str,
    patents_html: str,
    website_html: str,
) -> str:
    today_str = datetime.now(timezone.utc).strftime("%B %-d, %Y")

    sections = []
    if trial_html:
        sections.append(section_wrapper("Clinical trial updates", trial_html))
    if pubs_html:
        sections.append(section_wrapper("Publications", pubs_html))
    if conferences_html:
        sections.append(section_wrapper("Conferences", conferences_html, accent="#185FA5"))
    if news_html:
        sections.append(section_wrapper("News", news_html))
    if patents_html:
        sections.append(section_wrapper("Patents", patents_html, accent="#534AB7"))
    if website_html:
        sections.append(section_wrapper("Competitor website changes", website_html, accent="#993556"))

    if not sections:
        sections.append('<p style="font-size:14px;color:#5f5e5a;">No significant updates today.</p>')

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f0efe9;font-family:-apple-system,'Segoe UI',sans-serif;">
  <div style="max-width:680px;margin:0 auto;padding:24px 16px;">

    <!-- Header -->
    <div style="background:#1a1a18;border-radius:10px 10px 0 0;padding:20px 24px;margin-bottom:0;">
      <div style="font-size:12px;font-weight:600;color:#1D9E75;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">In vivo CAR-T Intelligence</div>
      <div style="font-size:22px;font-weight:600;color:#ffffff;line-height:1.2;">{today_str}</div>
    </div>

    <!-- Summary -->
    <div style="background:#ffffff;padding:20px 24px;border-left:1px solid #e8e7e2;border-right:1px solid #e8e7e2;">
      <p style="margin:0;font-size:15px;line-height:1.6;color:#1a1a18;">{summary}</p>
    </div>

    <!-- Main content -->
    <div style="background:#ffffff;border:1px solid #e8e7e2;border-top:none;border-radius:0 0 10px 10px;padding:20px 24px;">
      {"".join(sections)}

      <!-- Footer -->
      <div style="margin-top:28px;padding-top:16px;border-top:1px solid #f0efe9;font-size:12px;color:#9b9a96;text-align:center;">
        <a href="{DASHBOARD_URL}" style="color:#1D9E75;text-decoration:none;font-weight:500;">View full dashboard →</a>
        <span style="margin:0 12px;">·</span>
        Pipeline updated daily at 06:00 UTC
      </div>
    </div>

  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# SendGrid delivery
# ---------------------------------------------------------------------------

def send_email(subject: str, html_body: str) -> bool:
    if not SENDGRID_API_KEY:
        logging.error("SENDGRID_API_KEY not set — cannot send email")
        return False
    if not TO_EMAIL or not FROM_EMAIL:
        logging.error("DIGEST_TO_EMAIL or DIGEST_FROM_EMAIL not set")
        return False

    recipients = [{"email": addr.strip()} for addr in TO_EMAIL.split(",") if addr.strip()]

    payload = {
        "personalizations": [{"to": recipients}],
        "from": {"email": FROM_EMAIL, "name": "CAR-T Intelligence"},
        "subject": subject,
        "content": [{"type": "text/html", "value": html_body}],
    }

    resp = requests.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )

    if resp.status_code == 202:
        logging.info(f"Email sent to {TO_EMAIL}")
        return True
    else:
        logging.error(f"SendGrid error {resp.status_code}: {resp.text[:200]}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("Building daily digest")

    # Load all data
    changes    = load(TRIALS_CHANGES, "changes")
    pubs       = load(PUBS_PATH, "publications")
    news       = load(NEWS_PATH, "news")
    patents    = load(PATENTS_PATH, "patents")
    abstracts  = load(ABSTRACTS_PATH, "abstracts")
    web_chgs   = load(WEBSITE_CHANGES, "changes")
    appearances = load_conference_appearances()

    # Build sections
    trial_html,   trial_items  = build_trial_changes_section(changes)
    pubs_html,    pub_items    = build_publications_section(pubs)
    news_html,    news_plain   = build_news_section(news)
    patents_html, patent_items = build_patents_section(patents)
    conferences_html           = build_conferences_section(appearances)
    website_html               = build_website_changes_section(web_chgs)

    has_content = any([trial_html, pubs_html, news_html, patents_html, website_html])

    # Generate AI summary
    summary = generate_summary(trial_items, pub_items, news_plain, patent_items)
    logging.info(f"Summary: {summary}")

    # Assemble and send
    today_str = datetime.now(timezone.utc).strftime("%b %-d")
    has_trials  = bool(trial_html)
    has_patents = bool(patents_html)
    subject_parts = []
    if has_trials:
        subject_parts.append(f"{len(trial_items)} trial update{'s' if len(trial_items)!=1 else ''}")
    if has_patents:
        subject_parts.append("new patents")
    subject_suffix = " · ".join(subject_parts) if subject_parts else "no updates today"
    subject = f"CAR-T Intel {today_str} — {subject_suffix}"

    html = build_email(
        summary=summary,
        trial_html=trial_html,
        pubs_html=pubs_html,
        conferences_html=conferences_html,
        news_html=news_html,
        patents_html=patents_html,
        website_html=website_html,
    )

    send_email(subject, html)


if __name__ == "__main__":
    run()
