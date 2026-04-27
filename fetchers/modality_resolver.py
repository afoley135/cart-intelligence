"""
modality_resolver.py
--------------------
Single source of truth for modality lookups.

Priority order:
  1. company_profiles.json confirmed_modality  (human-verified, wins always)
  2. ai_modality field on the trial            (Claude classification)
  3. modality field on the trial               (keyword heuristic)
  4. "Not reported"

Usage:
  from modality_resolver import get_confirmed_modality, apply_profile_modalities

Both fetch_trials.py and summarize.py import this so the same logic
is applied consistently whether running the full pipeline or a single step.
"""

import json
import logging
from pathlib import Path

PROFILES_PATH = Path(__file__).parent.parent / "company_profiles.json"

_profiles_cache: dict | None = None


def load_profiles() -> dict:
    global _profiles_cache
    if _profiles_cache is not None:
        return _profiles_cache
    try:
        data = json.loads(PROFILES_PATH.read_text())
        _profiles_cache = data.get("companies", {})
        logging.info(f"Loaded {len(_profiles_cache)} company profiles from company_profiles.json")
    except FileNotFoundError:
        logging.warning(f"company_profiles.json not found at {PROFILES_PATH} — modality overrides disabled")
        _profiles_cache = {}
    except Exception as e:
        logging.warning(f"Failed to load company_profiles.json: {e}")
        _profiles_cache = {}
    return _profiles_cache


def get_confirmed_modality(sponsor: str) -> str | None:
    """
    Return confirmed_modality for a sponsor, or None if not in profiles.
    Matches on company name or any alias, case-insensitive.
    """
    if not sponsor:
        return None
    profiles = load_profiles()
    sl = sponsor.lower()
    for company_name, profile in profiles.items():
        if company_name.lower() in sl or sl in company_name.lower():
            modality = profile.get("confirmed_modality")
            if modality:
                return modality
        # Check aliases
        for alias in profile.get("aliases", []):
            if alias.lower() in sl or sl in alias.lower():
                modality = profile.get("confirmed_modality")
                if modality:
                    return modality
    return None


def apply_profile_modalities(studies: list[dict]) -> tuple[list[dict], int]:
    """
    For each study, apply confirmed_modality from company_profiles.json
    if the sponsor matches a watchlist company. Sets profile_modality field.

    Returns (updated_studies, override_count).
    """
    profiles = load_profiles()
    if not profiles:
        return studies, 0

    override_count = 0
    for study in studies:
        sponsor = study.get("sponsor", "")
        confirmed = get_confirmed_modality(sponsor)
        if confirmed and confirmed != study.get("profile_modality"):
            study["profile_modality"] = confirmed
            override_count += 1

    return studies, override_count


def resolve_display_modality(study: dict) -> str:
    """
    Return the best available modality for display, in priority order:
      1. profile_modality  (from company_profiles.json — human verified)
      2. ai_modality       (Claude classification)
      3. modality          (keyword heuristic)
      4. "Not reported"
    """
    return (
        study.get("profile_modality")
        or study.get("ai_modality")
        or study.get("modality")
        or "Not reported"
    )
