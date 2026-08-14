"""
ReviewRadar — Layer 1: Data Collection
======================================

Scrapes Google Play Store reviews for a configurable app and saves the raw
data to CSV. This layer does NOTHING else — no sentiment, no topics, no charts.

How it works (in one line):
    google-play-scraper gives us reviews in pages; we keep asking for the next
    page (using a "continuation token") until we have enough, then dedupe + save.

Run it:
    python src/scraper.py
"""

import logging
import os

import os
import sys

import pandas as pd
from google_play_scraper import Sort, reviews
from google_play_scraper.exceptions import NotFoundError

# Read the app from the central config so there's ONE place to switch apps.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ---------------------------------------------------------------------------
# CONFIGURATION  — the app comes from config.py; tune the scrape knobs here
# ---------------------------------------------------------------------------
PACKAGE_NAME = config.PACKAGE_NAME        # switch apps in config.py, not here
TARGET_REVIEWS = 20000                    # how many reviews we want (at least)
LANG = "en"                               # review language
COUNTRY = "in"                            # store country
BATCH_SIZE = 200                          # reviews per API call (200 is the max)
OUTPUT_PATH = "data/raw/reviews.csv"      # where the raw CSV is written

# The exact columns we keep from each review.
# NOTE: the library calls the reply timestamp "repliedAt"; we rename it to
# "replyAt" below so the CSV matches our intended schema.
FIELDS = [
    "reviewId", "userName", "content", "score", "thumbsUpCount",
    "reviewCreatedVersion", "at", "replyContent", "repliedAt",
]

# ---------------------------------------------------------------------------
# LOGGING  — so we can see what the scraper is doing while it runs
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scraper")


# ---------------------------------------------------------------------------
# STEP 1 — Fetch reviews page by page
# ---------------------------------------------------------------------------
def fetch_reviews(package_name, target, lang, country, batch_size):
    """
    Ask the Play Store for `target` reviews, one page at a time.

    google-play-scraper returns two things on each call:
      1. a list of reviews (this page)
      2. a `continuation_token` — pass it back to get the NEXT page
    We loop until we have enough reviews or there are no more pages.
    """
    collected = []          # every raw review dict we've received
    token = None            # None on the first call = start from the newest
    seen_ids = set()        # reviewIds we've already stored (for deduping)

    while len(seen_ids) < target:
        try:
            # Request one page of reviews. Sort.NEWEST keeps pagination stable.
            page, token = reviews(
                package_name,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=batch_size,
                continuation_token=token,
            )
        except NotFoundError:
            # The package id does not exist on this store.
            log.error("Package '%s' not found on Play Store (%s).", package_name, country)
            break
        except Exception as err:
            # Network hiccup, rate limit, format change, etc. Fail gracefully.
            log.error("Stopped early due to error: %s", err)
            break

        # An empty page means there are no more reviews to fetch.
        if not page:
            log.info("No more reviews available from the store.")
            break

        # Keep only NEW reviews (skip any reviewId we've already seen).
        for r in page:
            if r["reviewId"] not in seen_ids:
                seen_ids.add(r["reviewId"])
                collected.append(r)

        log.info("Fetched %d so far (unique: %d)...", len(collected), len(seen_ids))

        # If the store gives us no token, we've reached the end.
        if token is None:
            log.info("Reached the last page of reviews.")
            break

    return collected


# ---------------------------------------------------------------------------
# STEP 2 — Turn the raw reviews into a clean table
# ---------------------------------------------------------------------------
def to_dataframe(raw_reviews, fields):
    """Build a pandas DataFrame with only the columns we care about, deduped."""
    df = pd.DataFrame(raw_reviews)
    # Keep only the fields we asked for (in case the library returns extras).
    df = df[[c for c in fields if c in df.columns]]
    # Rename the library's "repliedAt" to our schema's "replyAt".
    df = df.rename(columns={"repliedAt": "replyAt"})
    # Safety net: drop any duplicate reviewIds that slipped through.
    df = df.drop_duplicates(subset=["reviewId"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# STEP 3 — Print a summary so we can sanity-check the data
# ---------------------------------------------------------------------------
def log_summary(df):
    """Log counts, date range, and breakdowns by rating and app version."""
    log.info("=" * 55)
    log.info("Total reviews fetched : %d", len(df))
    log.info("Unique reviews        : %d", df["reviewId"].nunique())

    # Date range of the reviews.
    dates = pd.to_datetime(df["at"], errors="coerce")
    log.info("Date range            : %s  ->  %s",
             dates.min(), dates.max())

    # How many reviews at each star rating (5,4,3,2,1).
    log.info("Reviews by rating:")
    for score, count in df["score"].value_counts().sort_index(ascending=False).items():
        log.info("   %d star : %d", int(score), count)

    # How many reviews per app version (top 10 versions).
    log.info("Reviews by app version (top 10):")
    versions = df["reviewCreatedVersion"].fillna("unknown").value_counts().head(10)
    for version, count in versions.items():
        log.info("   %-12s : %d", version, count)
    log.info("=" * 55)


# ---------------------------------------------------------------------------
# STEP 4 — Save to CSV
# ---------------------------------------------------------------------------
def save_csv(df, path):
    """Write the DataFrame to `path`, creating the folder if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Saved %d reviews -> %s", len(df), path)


# ---------------------------------------------------------------------------
# MAIN — glue the steps together
# ---------------------------------------------------------------------------
def main():
    log.info("Scraping reviews for: %s", PACKAGE_NAME)

    raw = fetch_reviews(PACKAGE_NAME, TARGET_REVIEWS, LANG, COUNTRY, BATCH_SIZE)
    if not raw:
        log.error("No reviews collected. Exiting.")
        return

    df = to_dataframe(raw, FIELDS)
    log_summary(df)
    save_csv(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
