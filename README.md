# In vivo CAR-T intelligence dashboard — data pipeline

This repo contains the automated data pipeline and dashboard UI for tracking
competitive developments in the in vivo CAR-T space.

## Project structure

```
cart-intelligence/
├── fetchers/
│   ├── fetch_trials.py        # ClinicalTrials.gov v2 API
│   ├── fetch_publications.py  # PubMed + bioRxiv
│   └── summarize.py           # AI "so what" generation via Claude API
├── scripts/
│   └── run_pipeline.py        # Orchestrates all fetchers in sequence
├── data/                      # Auto-generated JSON files (committed by CI)
│   ├── trials.json
│   └── publications.json
├── .github/
│   └── workflows/
│       └── update.yml         # GitHub Actions: runs pipeline daily at 06:00 UTC
├── index.html                 # Dashboard UI (served via GitHub Pages)
└── requirements.txt
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/cart-intelligence.git
cd cart-intelligence
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
# Required for AI summarisation step
export ANTHROPIC_API_KEY=sk-ant-...

# Optional: raises PubMed rate limit from 3 to 10 req/s
# Get a free key at https://www.ncbi.nlm.nih.gov/account/
export NCBI_API_KEY=your_ncbi_key
```

### 3. Run the pipeline locally

```bash
python scripts/run_pipeline.py
```

This will:
- Fetch trials from ClinicalTrials.gov and write `data/trials.json`
- Fetch publications from PubMed + bioRxiv and write `data/publications.json`
- Call Claude to generate "so what" summaries for each new item

---

## GitHub deployment

### Enable GitHub Pages

1. Go to **Settings → Pages** in your repo
2. Set source to **Deploy from branch → main → / (root)**
3. Your dashboard will be live at `https://YOUR_USERNAME.github.io/cart-intelligence/`

### Add secrets for GitHub Actions

Go to **Settings → Secrets and variables → Actions** and add:

| Secret | Value |
|--------|-------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key |
| `NCBI_API_KEY` | *(optional)* Your NCBI API key |

The workflow in `.github/workflows/update.yml` will then run automatically
every day at 06:00 UTC, update the JSON data files, and commit them back to
the repo. GitHub Pages will serve the updated dashboard immediately.

### Manual trigger

You can also trigger a run manually from the **Actions** tab in GitHub.

---

## Customising search terms

All search queries are defined at the top of each fetcher script:

- `fetchers/fetch_trials.py` → `QUERIES` list and `LOOKBACK_DAYS`
- `fetchers/fetch_publications.py` → `PUBMED_QUERY`, `BIORXIV_SEARCH_TERMS`, `LOOKBACK_DAYS`

To add new modalities (e.g. bispecific T cell engagers), add terms to the
`QUERIES` list in `fetch_trials.py` and update the `BISPECIFIC_SIGNALS` list
in the modality inference logic.

---

## Adding more data sources

Future phases:
- `fetchers/fetch_news.py` — Google News RSS / NewsAPI
- `fetchers/fetch_patents.py` — USPTO PatentsView + Lens.org full text
- `fetchers/fetch_regulatory.py` — FDA.gov open data

Each new fetcher follows the same pattern: fetch → parse → write JSON.
Add it to `scripts/run_pipeline.py` and the GitHub Actions workflow.
