"""
ReviewRadar — Layer 2: Data Cleaning & Preprocessing
====================================================

Input : data/raw/reviews.csv   (from src/scraper.py)
Output: data/processed/reviews_clean.csv

Philosophy: clean GENTLY. This data feeds sentiment analysis and topic modeling
later, so we deliberately keep things most "text cleaners" throw away:
  - emojis  (🙂😡) carry strong sentiment signal
  - Hinglish / Indian-language script ("bakwas app", "बहुत अच्छा") is real content
  - stopwords stay (topic models and sentiment models need natural sentences)

We remove only true noise: duplicates, blank reviews, URLs, and messy whitespace.

Run it:
    python src/cleaner.py
"""

import logging
import os
import re

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
INPUT_PATH = "data/raw/reviews.csv"
OUTPUT_PATH = "data/processed/reviews_clean.csv"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cleaner")

# Matches http/https links and bare "www." links. We strip these because URLs
# are noise for sentiment/topics and can leak into word clouds.
URL_RE = re.compile(r"http\S+|www\.\S+", flags=re.IGNORECASE)
# Collapses any run of whitespace (spaces, tabs, newlines) into a single space.
WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# STEP 1 — Text cleaning (produces content_clean, keeps content_raw intact)
# ---------------------------------------------------------------------------
def clean_text(text):
    """
    Lightly normalise one review string.

    We intentionally DO NOT:
      - strip emojis or non-English characters (needed for sentiment/Hinglish)
      - remove stopwords (topic models need whole sentences)
    We DO:
      - drop URLs (noise)
      - collapse repeated whitespace
      - lowercase (helps group 'Crash'/'crash' in topic modeling)
    """
    text = str(text)
    text = URL_RE.sub(" ", text)        # remove links
    text = WS_RE.sub(" ", text)         # normalise whitespace
    return text.strip().lower()


# ---------------------------------------------------------------------------
# STEP 2 — Row-level cleaning (dedupe + drop empty reviews)
# ---------------------------------------------------------------------------
def clean_rows(df):
    """Remove duplicate reviewIds and reviews with no usable text."""
    before = len(df)

    # Preserve the untouched original text before we transform anything.
    df["content_raw"] = df["content"]

    # Drop duplicate reviews (same reviewId = same review fetched twice).
    df = df.drop_duplicates(subset=["reviewId"])
    dupes_removed = before - len(df)

    # Handle missing text: blank/NaN content can't be analysed, so drop it.
    df["content_raw"] = df["content_raw"].fillna("").astype(str)
    df["content_clean"] = df["content_raw"].apply(clean_text)
    missing = int((df["content_clean"].str.len() == 0).sum())
    df = df[df["content_clean"].str.len() > 0]

    return df.reset_index(drop=True), dupes_removed, missing


# ---------------------------------------------------------------------------
# STEP 3 — Dates (parse timestamps, drop malformed ones, derive date parts)
# ---------------------------------------------------------------------------
def add_date_columns(df):
    """Convert 'at' to real datetimes and derive date/month/year helpers."""
    # errors='coerce' turns any malformed date into NaT instead of crashing.
    df["review_datetime"] = pd.to_datetime(df["at"], errors="coerce")
    df = df.dropna(subset=["review_datetime"])          # drop unparseable dates

    df["review_date"] = df["review_datetime"].dt.date
    df["review_month"] = df["review_datetime"].dt.to_period("M").astype(str)  # e.g. 2026-08
    df["review_year"] = df["review_datetime"].dt.year
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP 4 — Derived analytical columns
# ---------------------------------------------------------------------------
def add_derived_columns(df):
    """Add review_length and a human-readable rating_category."""
    # Word count of the cleaned text — a quick proxy for review effort/detail.
    df["review_length"] = df["content_clean"].str.split().str.len()

    # Bucket star ratings into sentiment-style categories.
    def bucket(score):
        if score <= 2:
            return "Negative"
        if score == 3:
            return "Neutral"
        return "Positive"

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    df["rating_category"] = df["score"].astype(int).apply(bucket)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP 5 — Cleaning report
# ---------------------------------------------------------------------------
def print_report(rows_before, df, dupes_removed, missing):
    """Log a before/after summary so the cleaning is transparent."""
    log.info("=" * 55)
    log.info("CLEANING REPORT")
    log.info("Rows before cleaning : %d", rows_before)
    log.info("Rows after cleaning  : %d", len(df))
    log.info("Duplicates removed   : %d", dupes_removed)
    log.info("Missing/blank text   : %d", missing)
    log.info("Date range           : %s -> %s",
             df["review_date"].min(), df["review_date"].max())
    log.info("Rating distribution:")
    for cat, count in df["rating_category"].value_counts().items():
        log.info("   %-9s : %d", cat, count)
    log.info("=" * 55)


# ---------------------------------------------------------------------------
# STEP 6 — Save
# ---------------------------------------------------------------------------
def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Saved cleaned data -> %s", path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    log.info("Loading %s", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)
    rows_before = len(df)

    df, dupes_removed, missing = clean_rows(df)
    df = add_date_columns(df)
    df = add_derived_columns(df)

    print_report(rows_before, df, dupes_removed, missing)
    save_csv(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
