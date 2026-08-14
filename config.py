"""ReviewRadar central config — the ONE place an app is named.

Switch the whole pipeline to another app by editing these two lines and
re-running the scraper + layers. Nothing else hardcodes an app.

    APP_NAME = "Zepto"; PACKAGE_NAME = "com.zeptoconsumerapp"
    APP_NAME = "Paytm"; PACKAGE_NAME = "net.one97.paytm"
"""

APP_NAME = "Groww"
PACKAGE_NAME = "com.nextbillion.groww"

# Shared data paths (kept flat & simple; every layer reads/writes these).
RAW_REVIEWS = "data/raw/reviews.csv"
CLEAN_REVIEWS = "data/processed/reviews_clean.csv"
REVIEWS_WITH_SENTIMENT = "data/processed/reviews_with_sentiment.csv"
REVIEWS_WITH_TOPICS = "data/processed/reviews_with_topics.csv"
TOPIC_SUMMARY = "outputs/topic_summary.csv"
