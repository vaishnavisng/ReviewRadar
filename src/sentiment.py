"""
ReviewRadar — Layer 4: Sentiment Analysis (multilingual)
========================================================

Input : data/processed/reviews_clean.csv
Output: data/processed/reviews_with_sentiment.csv
Charts: reports/figures/*.png

GOAL
----
Find the sentiment actually EXPRESSED in the review text, instead of trusting the
star rating alone. Stars and words don't always agree (a 5-star "worst app ever",
or a 1-star with calm feedback), and those mismatches are the interesting cases.

MODEL: cardiffnlp/twitter-xlm-roberta-base-sentiment
Why this model for THIS data:
  * Multilingual (XLM-RoBERTa) -> handles English, Hindi, and code-mixed Hinglish,
    which an English-only model (e.g. DistilBERT/SST-2) would mangle.
  * Trained on ~198M tweets -> short, slangy, emoji-heavy text, exactly like
    Play Store reviews (our median review is ~2 words).
  * 3 classes: negative / neutral / positive -> a real 'neutral', matching how
    star ratings behave.

MODEL LIMITATIONS (be honest about these):
  * Sarcasm / irony ("wow, great, crashed again") is often misread.
  * Hinglish & regional slang are only partially covered; rare dialect words may
    be scored wrong.
  * Very short reviews ("ok", "nice") carry little signal -> low-confidence guesses.
  * Mixed-language reviews (half Hindi, half English) can confuse tokenisation.
  * The model reports CORRELATION with text tone only. We do NOT claim sentiment
    CAUSES the star rating — they are two separate signals we compare.

Run it:
    python src/sentiment.py
"""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import pipeline

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
INPUT_PATH = "data/processed/reviews_clean.csv"
OUTPUT_PATH = "data/processed/reviews_with_sentiment.csv"
FIG_DIR = "reports/figures"
REPORT_PATH = "reports/sentiment_report.txt"
MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BATCH_SIZE = 64          # reviews per forward pass; raise if you have a GPU
MAX_LEN = 256            # truncate long reviews (most are short anyway)
MIN_CHARS = 2            # shorter than this = treat as empty, skip the model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sentiment")
# Quiet the noisy HTTP/download loggers so our report stays readable.
for noisy in ("httpx", "urllib3", "huggingface_hub", "filelock"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

sns.set_theme(style="whitegrid")
# Fixed colours so every chart reads the same way.
PALETTE = {"negative": "#C44E52", "neutral": "#B0B0B0", "positive": "#55A868"}


# ---------------------------------------------------------------------------
# STEP 1 — Build the model
# ---------------------------------------------------------------------------
def build_classifier():
    """Load the multilingual pipeline once (downloads the model on first run)."""
    log.info("Loading model: %s (first run downloads ~1GB)", MODEL)
    return pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL,
                    truncation=True, max_length=MAX_LEN, batch_size=BATCH_SIZE)


def _norm_label(raw):
    """Normalise the model's label to negative/neutral/positive.

    Some model revisions return 'LABEL_0/1/2' instead of words, so map both.
    """
    raw = str(raw).lower()
    return {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}.get(raw, raw)


# ---------------------------------------------------------------------------
# STEP 2 — Score reviews (batched, with empty-review handling)
# ---------------------------------------------------------------------------
def add_sentiment(df, clf):
    """
    Attach:
      sentiment_label      -> negative / neutral / positive
      sentiment_score      -> model confidence for that label (0-1)
      sentiment_signed     -> directional score in [-1,1] for trend charts
                              (negative<0, neutral=0, positive>0)

    Empty / extremely short reviews are labelled 'neutral' with score 0 and are
    NOT sent to the model (no text = no signal, and it avoids wasted compute).
    """
    df = df.copy()
    text = df["content_clean"].fillna("").astype(str).str.strip()
    is_empty = text.str.len() < MIN_CHARS

    # Defaults cover the empty rows; the model overwrites the rest.
    df["sentiment_label"] = "neutral"
    df["sentiment_score"] = 0.0

    to_score = text[~is_empty].tolist()
    log.info("Scoring %d reviews (%d skipped as empty/too short)...",
             len(to_score), int(is_empty.sum()))
    if to_score:
        results = clf(to_score)
        labels = [_norm_label(r["label"]) for r in results]
        scores = [round(r["score"], 4) for r in results]
        df.loc[~is_empty, "sentiment_label"] = labels
        df.loc[~is_empty, "sentiment_score"] = scores

    sign = {"negative": -1, "neutral": 0, "positive": 1}
    df["sentiment_signed"] = df["sentiment_label"].map(sign) * df["sentiment_score"]
    return df


# ---------------------------------------------------------------------------
# STEP 3 — Compare stars vs text, flag mismatches
# ---------------------------------------------------------------------------
def add_mismatches(df):
    """
    Flag reviews where the star rating and the text sentiment disagree.
    These are the reviews worth a human read.
    """
    df = df.copy()
    high = df["score"] >= 4
    low = df["score"] <= 2
    neutral_rating = df["score"] == 3
    neg = df["sentiment_label"] == "negative"
    pos = df["sentiment_label"] == "positive"

    def kind(row_high, row_low, row_neu, row_neg, row_pos):
        if row_high and row_neg:
            return "high_rating_negative_text"
        if row_low and row_pos:
            return "low_rating_positive_text"
        if row_neu and row_neg:
            return "neutral_rating_negative_text"
        return "aligned"

    df["mismatch_type"] = [
        kind(h, l, nr, ng, ps)
        for h, l, nr, ng, ps in zip(high, low, neutral_rating, neg, pos)
    ]
    return df


# ---------------------------------------------------------------------------
# STEP 4 — Charts
# ---------------------------------------------------------------------------
def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("chart -> %s", path)


def make_charts(df):
    order = ["negative", "neutral", "positive"]

    # 4a. Sentiment distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="sentiment_label", order=order,
                  hue="sentiment_label", palette=PALETTE, legend=False, ax=ax)
    ax.set(title="Sentiment Distribution", xlabel="", ylabel="Reviews")
    _save(fig, "sentiment_distribution.png")

    # 4b. Sentiment over time (monthly share of each label)
    share = (df.groupby(["review_month", "sentiment_label"]).size()
               .groupby(level=0).apply(lambda s: 100 * s / s.sum())
               .unstack(fill_value=0).reindex(columns=order, fill_value=0))
    fig, ax = plt.subplots(figsize=(10, 5))
    share.plot(kind="area", stacked=True, ax=ax,
               color=[PALETTE[c] for c in order], alpha=0.9)
    ax.set(title="Sentiment Over Time (% of monthly reviews)",
           xlabel="Month", ylabel="% of reviews", ylim=(0, 100))
    _save(fig, "sentiment_over_time.png")

    # 4c. Sentiment by star rating
    ct = (pd.crosstab(df["score"], df["sentiment_label"], normalize="index") * 100)
    ct = ct.reindex(columns=order, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=[PALETTE[c] for c in order])
    ax.set(title="Text Sentiment by Star Rating",
           xlabel="Star rating", ylabel="% of reviews")
    _save(fig, "sentiment_by_rating.png")

    # 4d. Sentiment by app version (versions with enough reviews)
    vc = df["reviewCreatedVersion"].fillna("unknown").value_counts()
    keep = vc[(vc.index != "unknown") & (vc >= 30)].index[:12]
    vdf = df[df["reviewCreatedVersion"].isin(keep)]
    if len(vdf):
        vshare = (vdf.groupby(["reviewCreatedVersion", "sentiment_label"]).size()
                    .groupby(level=0).apply(lambda s: 100 * s / s.sum())
                    .unstack(fill_value=0).reindex(columns=order, fill_value=0)
                    .sort_values("negative", ascending=False))
        fig, ax = plt.subplots(figsize=(10, 5))
        vshare.plot(kind="bar", stacked=True, ax=ax,
                    color=[PALETTE[c] for c in order])
        ax.set(title="Sentiment by App Version (>=30 reviews)",
               xlabel="Version", ylabel="% of reviews")
        _save(fig, "sentiment_by_version.png")


# ---------------------------------------------------------------------------
# STEP 5 — Report
# ---------------------------------------------------------------------------
def print_report(df):
    """Build the report, log it, AND save it to a text file so it isn't lost
    in the console scroll."""
    lines = ["=" * 55, "SENTIMENT REPORT", "-" * 55]
    for label, count in df["sentiment_label"].value_counts().items():
        lines.append(f"   {label:<9} : {count} ({100*count/len(df):.1f}%)")

    # Agreement between stars and text (exclude 3-star, model has neutral).
    star_pos = df["score"] >= 4
    star_neg = df["score"] <= 2
    checkable = star_pos | star_neg
    agree = ((star_pos & (df["sentiment_label"] == "positive")) |
             (star_neg & (df["sentiment_label"] == "negative")))
    if checkable.any():
        lines.append(f"Star-vs-text agreement (excl. 3-star): {100*agree[checkable].mean():.1f}%")

    lines.append("Mismatch breakdown:")
    for kind, count in df["mismatch_type"].value_counts().items():
        if kind != "aligned":
            lines.append(f"   {kind:<30} : {count}")
    lines.append("NOTE: mismatches show where tone and stars differ, not causation.")
    lines.append("=" * 55)

    for ln in lines:
        log.info(ln)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("Report saved -> %s", REPORT_PATH)


def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("Saved -> %s", path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    log.info("Loading %s", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    clf = build_classifier()
    df = add_sentiment(df, clf)
    df = add_mismatches(df)

    make_charts(df)
    save_csv(df, OUTPUT_PATH)
    print_report(df)          # last, so it sits at the bottom of the console too


if __name__ == "__main__":
    main()
