"""
debug_epo.py
------------
Diagnostic script for EPO OPS patent search.
Run locally to see exactly what the API is returning.

Usage:
  EPO_OPS_KEY=xxx EPO_OPS_SECRET=xxx python debug_epo.py

Prints raw HTTP status, headers, and truncated XML for each test query.
"""

import base64
import os
import time
import requests

EPO_OPS_KEY    = os.environ.get("EPO_OPS_KEY", "")
EPO_OPS_SECRET = os.environ.get("EPO_OPS_SECRET", "")
EPO_AUTH_URL   = "https://ops.epo.org/3.2/auth/accesstoken"
EPO_SEARCH_URL = "https://ops.epo.org/3.2/rest-services/published-data/search"

# Test queries — ordered from most to least permissive
# Edit these to test different companies or strategies
TEST_QUERIES = [
    # 1. Wildcard on distinctive name — should be the most likely to hit
    'pa="Tessera*"',
    'pa="Umoja*"',
    'pa="Capstan*"',
    'pa="Sana*" AND ic=C12N',

    # 2. Title/abstract keyword search — catches filings under parent entities
    'ta="Tessera Therapeutics"',
    'ta="Umoja Biopharma"',
    'ta="Capstan Therapeutics"',

    # 3. Broad CAR-T landscape check — confirms the API is working at all
    'ta="in vivo CAR-T" AND pd>=20230101',
    'ta="chimeric antigen receptor" AND pa="Novartis*" AND pd>=20230101',  # known large filer

    # 4. Date-scoped applicant wildcard
    'pa="Tessera*" AND pd>=20200101',
    'pa="Umoja*" AND pd>=20200101',
]


def get_token() -> str:
    credentials = base64.b64encode(f"{EPO_OPS_KEY}:{EPO_OPS_SECRET}".encode()).decode()
    resp = requests.post(
        EPO_AUTH_URL,
        data="grant_type=client_credentials",
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=30,
    )
    print(f"Auth status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"Auth response: {resp.text[:500]}")
        raise SystemExit("Authentication failed — check EPO_OPS_KEY and EPO_OPS_SECRET")
    data = resp.json()
    print(f"Token acquired, expires in {data.get('expires_in')}s\n")
    return data["access_token"]


def run_query(token: str, cql: str) -> None:
    print(f"{'─'*70}")
    print(f"Query: {cql}")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/xml",
        "X-OPS-Range": "1-5",  # just first 5 for diagnosis
    }
    try:
        resp = requests.get(
            EPO_SEARCH_URL,
            params={"q": cql},
            headers=headers,
            timeout=30,
        )
        print(f"Status: {resp.status_code}")
        print(f"X-OPS-Range-total: {resp.headers.get('X-OPS-Range-total', 'not present')}")
        print(f"X-Rejection-Reason: {resp.headers.get('X-Rejection-Reason', 'none')}")

        if resp.status_code == 200:
            xml = resp.text
            # Print first 800 chars of XML to see structure
            print(f"XML preview:\n{xml[:800]}")
        elif resp.status_code == 404:
            print("404 — No results found for this query")
            # 404 body sometimes contains useful info
            print(f"Body: {resp.text[:300]}")
        elif resp.status_code == 400:
            print(f"400 Bad Request — likely CQL syntax error")
            print(f"Body: {resp.text[:500]}")
        elif resp.status_code == 429:
            print("429 — Rate limit hit")
        else:
            print(f"Unexpected status. Body: {resp.text[:500]}")

    except Exception as e:
        print(f"Exception: {e}")

    print()
    time.sleep(1.0)  # be polite


def main():
    if not EPO_OPS_KEY or not EPO_OPS_SECRET:
        print("ERROR: Set EPO_OPS_KEY and EPO_OPS_SECRET environment variables")
        raise SystemExit(1)

    print("EPO OPS Diagnostic Script")
    print(f"Key: {EPO_OPS_KEY[:8]}...\n")

    token = get_token()

    for cql in TEST_QUERIES:
        run_query(token, cql)


if __name__ == "__main__":
    main()
