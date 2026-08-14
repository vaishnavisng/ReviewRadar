# ReviewRadar — App Review Intelligence

A **config-driven, app-agnostic** pipeline that scrapes Google Play reviews for
*any* app and turns 20,000+ reviews into sentiment trends, auto-discovered
issues, version-level problems, and a prioritized, evidence-backed fix list —
served in an interactive Streamlit dashboard.

> The engine hardcodes **no app and no issues**. Switch apps by editing two lines
> in `config.py`. Topics are discovered from the review text, never hardcoded.

## Pipeline (run in order)

```
config.py                         # the ONLY place an app is named
   │
1  src/scraper.py          → data/raw/reviews.csv              (Google Play scrape)
2  src/cleaner.py          → data/processed/reviews_clean.csv  (clean + derive columns)
3  src/sentiment.py        → data/processed/reviews_with_sentiment.csv  (Transformers)
4  src/topics.py           → data/processed/reviews_with_topics.csv     (TF-IDF + K-Means)
                             outputs/topic_summary.csv
5  src/version_analysis.py → outputs/version_impact.csv, version_issue_analysis.csv
                             data/processed/reviewradar.db     (SQLite)
6  src/prioritization.py   → outputs/product_priorities.csv, executive_summary.txt
   │
   ▼
dashboard/app.py           # Streamlit — reads the files above, runs no ML
```

Notebook `notebooks/01_eda.ipynb` holds the exploratory analysis.

## Setup

```bash
pip install -r requirements-pipeline.txt   # full deps to run the pipeline locally
```

`requirements.txt` is intentionally slim (streamlit + pandas + matplotlib + seaborn) —
it is what the **deployed dashboard** installs, since the dashboard runs no ML.

## Run the full pipeline

```bash
python src/scraper.py
python src/cleaner.py
python src/sentiment.py          # first run downloads a ~1GB multilingual model
python src/topics.py
python src/version_analysis.py
python src/prioritization.py
```

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

### Dashboard sections
- **Overview** — KPI cards + a generated executive summary + live SQL metrics.
- **Rating & Sentiment** — rating distribution, rating/volume/negative-% over time, sentiment breakdowns.
- **Issues** — issues discovered by the TF-IDF + K-Means layer, with representative real reviews.
- **Version Intelligence** — rating / negative-% / volume / dominant issue per version, plus a version×issue heatmap.
- **Product Priorities** — P0–P3 ranked issues with evidence and recommended actions.
- **Review Explorer** — filterable table of the raw reviews.

Sidebar filters (date, rating, sentiment, version, issue) update the review-level
views live — no model ever re-runs.

### Which files the dashboard reads
`reviews_with_topics.csv`, `topic_summary.csv`, `version_impact.csv`,
`version_issue_analysis.csv`, `product_priorities.csv`, and `reviewradar.db`
(for the SQL metrics). It reads results only.

### Why heavy NLP/ML runs *before* the dashboard
Sentiment (Transformers), TF-IDF, and K-Means take minutes and load large models.
Running them inside Streamlit would make every click slow. So all heavy analysis
is done once in the pipeline and saved to CSV/SQLite; the dashboard just reads
those results and stays instant.

## Switching to another app

Edit **only** `config.py`:

```python
APP_NAME     = "Zepto"
PACKAGE_NAME = "com.zeptoconsumerapp"
```

| App   | PACKAGE_NAME            |
|-------|------------------------|
| Groww | com.nextbillion.groww  |
| Zepto | com.zeptoconsumerapp   |
| Paytm | net.one97.paytm        |

Then re-run the pipeline and the dashboard. Outputs overwrite in place, and the
dashboard title becomes **"ReviewRadar — Zepto"** automatically. The scraper reads
`PACKAGE_NAME` from `config.py`, so there is one source of truth.

## Tech stack
Python · Pandas · google-play-scraper · Transformers (multilingual sentiment) ·
scikit-learn (TF-IDF + K-Means) · SQL / SQLite · Matplotlib / Seaborn · Streamlit.

## Honest limitations
- Play Store reviews are **self-selected** — not all users.
- Complaint frequency ≠ real incidence of a problem.
- Sentiment models and topic clusters can be wrong or noisy (2 clusters are flagged as noise/language).
- Version association is **correlation, not causation**; `reviewCreatedVersion` is not an official release date.
- Product recommendations are **hypotheses to validate**, not confirmed root causes.
