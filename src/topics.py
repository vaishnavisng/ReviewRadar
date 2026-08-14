"""
ReviewRadar — Layer 6: Topic / Issue Mining (TF-IDF + K-Means)
==============================================================

Input : data/processed/reviews_with_sentiment.csv
Output: data/processed/reviews_with_topics.csv   (full dataset + issue columns)
        outputs/topic_summary.csv                (one row per discovered issue)
        outputs/topic_validation.txt             (human validation notes)
        reports/figures/*.png                    (7 charts)

IDEA (interview version):
  TF-IDF turns each review into numbers that highlight its RARE, meaningful words
  ("otp", "refund", "crash") over common filler ("app", "the", "good"). K-Means
  then groups reviews that share such words -> each group is a recurring issue.
  We read the top TF-IDF terms of each group to name it. No BERTopic, no
  embeddings, no black boxes -- everything is explainable from the words.

We deliberately AVOID: BERTopic, sentence-transformers, embeddings, vector DBs.

Run it:
    python src/topics.py
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
INPUT_PATH = "data/processed/reviews_with_sentiment.csv"
OUTPUT_REVIEWS = "data/processed/reviews_with_topics.csv"
OUTPUT_SUMMARY = "outputs/topic_summary.csv"
VALIDATION_PATH = "outputs/topic_validation.txt"
FIG_DIR = "reports/figures"

K_RANGE = range(3, 13)          # candidate cluster counts to evaluate
MIN_WORDS = 3                   # ignore ultra-short reviews when clustering (noise)
N_REP_REVIEWS = 5               # representative reviews to pull per cluster
TOP_TERMS = 8                   # top TF-IDF terms to show per cluster

# Business band for K: we won't accept a mathematically-best K outside this
# range because too many clusters fragment into meaningless topics.
K_MIN_BUSINESS, K_MAX_BUSINESS = 4, 8

# OPTIONAL manual renaming AFTER you read the clusters. Leave empty on first run;
# fill in like {0: "Login / OTP Issues", 3: "Refund Delays"} once you've seen
# the keywords. Names are NOT hardcoded before analysis (requirement 8).
# Filled in AFTER reading outputs/topic_validation.txt (K=8 run). Cluster ids are
# stable for a fixed random_state; re-check keywords if you change K or the data.
MANUAL_TOPIC_NAMES = {
    0: "High Brokerage / Charges",
    1: "App Update Complaints",
    2: "General / Mixed (noisy, other-language)",   # junk-drawer cluster
    3: "Hinglish Reviews (language, not an issue)",
    4: "Generic Negative Experience",
    5: "Severe Dissatisfaction ('worst app')",
    6: "Customer Support",
    7: "Charts / App Not Working",
}

# Clusters that are language/noise artifacts, not real product issues. They stay
# in the ranking (for honesty about the method) but are flagged is_noise=True.
NOISE_CLUSTERS = {2, 3}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("topics")
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# STEP 1 — Load full data, carve out the negative subset (full df stays intact)
# ---------------------------------------------------------------------------
def load_and_split(path):
    df = pd.read_csv(path)
    df["content_clean"] = df["content_clean"].fillna("").astype(str)

    # "Negative" = low stars OR the NLP model flagged negative sentiment.
    is_negative = (df["score"] <= 2) | (df["sentiment_label"] == "negative")
    # Only cluster reviews with enough words to carry a topic (noise control).
    long_enough = df["content_clean"].str.split().str.len() >= MIN_WORDS

    neg = df[is_negative & long_enough].copy()
    log.info("Full dataset: %d | negative & >=%d words (clustered): %d",
             len(df), MIN_WORDS, len(neg))
    return df, neg


# ---------------------------------------------------------------------------
# STEP 2 — TF-IDF features
# ---------------------------------------------------------------------------
def build_tfidf(texts):
    """
    Parameter choices (and WHY):
      max_features=1000  -> keep the 1000 most informative terms; caps noise/size.
      min_df=5           -> a term must appear in >=5 reviews; drops typos/one-offs.
      max_df=0.6         -> drop terms in >60% of reviews; they're too generic to
                            separate issues (e.g. 'app').
      ngram_range=(1,2)  -> unigrams AND bigrams, so 'log in', 'not working',
                            'money debited' are captured as single features.
      stop_words='english' -> remove English filler ('the','is','and'). NOTE this
                            keeps product terms (otp, login, refund, kyc, crash) --
                            they are not stopwords, so nothing important is lost.
    """
    vec = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.6,
                          ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(texts)
    log.info("TF-IDF matrix: %d reviews x %d terms", X.shape[0], X.shape[1])
    return X, vec


# ---------------------------------------------------------------------------
# STEP 3 — Choose K via elbow (inertia) + silhouette
# ---------------------------------------------------------------------------
def evaluate_k(X, k_range):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        # sample_size keeps silhouette fast on a big matrix (it's O(n^2)).
        sil = silhouette_score(X, labels, sample_size=min(2000, X.shape[0]),
                               random_state=42)
        rows.append({"k": k, "inertia": round(km.inertia_, 2),
                     "silhouette": round(sil, 4)})
        log.info("   k=%2d | inertia=%.1f | silhouette=%.4f", k, km.inertia_, sil)
    return pd.DataFrame(rows)


def choose_k(metrics):
    """Pick the highest-silhouette K *inside the business-interpretable band*,
    rather than blindly taking the global optimum (which may over-fragment)."""
    band = metrics[(metrics["k"] >= K_MIN_BUSINESS) & (metrics["k"] <= K_MAX_BUSINESS)]
    best = band.loc[band["silhouette"].idxmax()]
    k = int(best["k"])
    log.info("Chosen K = %d (best silhouette in business band %d-%d)",
             k, K_MIN_BUSINESS, K_MAX_BUSINESS)
    return k


# ---------------------------------------------------------------------------
# STEP 4 — Final clustering + interpretation
# ---------------------------------------------------------------------------
def cluster_and_name(neg, X, vec, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    neg = neg.copy()
    neg["issue_cluster"] = km.fit_predict(X)

    terms = np.array(vec.get_feature_names_out())
    dists = km.transform(X)                 # distance of each review to each centroid

    top_terms, candidate_names, rep_reviews = {}, {}, {}
    for c in range(k):
        # Top TF-IDF terms = the words closest to this cluster's centre.
        center = km.cluster_centers_[c]
        top_idx = center.argsort()[::-1][:TOP_TERMS]
        kws = list(terms[top_idx])
        top_terms[c] = kws
        # Candidate name = first few keywords joined (data-driven, editable later).
        candidate_names[c] = MANUAL_TOPIC_NAMES.get(c, " / ".join(kws[:3]))

        # Representative reviews = actual reviews nearest this centroid.
        members = np.where(neg["issue_cluster"].values == c)[0]
        nearest = members[np.argsort(dists[members, c])[:N_REP_REVIEWS]]
        text_col = "content_raw" if "content_raw" in neg.columns else "content_clean"
        rep_reviews[c] = neg.iloc[nearest][text_col].tolist()

    neg["issue_topic"] = neg["issue_cluster"].map(candidate_names)
    return neg, km, top_terms, candidate_names, rep_reviews


# ---------------------------------------------------------------------------
# STEP 5 — Per-cluster summary table
# ---------------------------------------------------------------------------
def summarize(neg, top_terms, names, rep_reviews):
    total_neg = len(neg)
    has_sent = "sentiment_score" in neg.columns
    rows = []
    for c, g in neg.groupby("issue_cluster"):
        rows.append({
            "cluster_id": c,
            "topic_name": names[c],
            "is_noise": c in NOISE_CLUSTERS,
            "review_count": len(g),
            "negative_percentage": round(100 * len(g) / total_neg, 1),
            "one_star_percentage": round(100 * (g["score"] == 1).mean(), 1),
            "two_star_percentage": round(100 * (g["score"] == 2).mean(), 1),
            "average_rating": round(g["score"].mean(), 2),
            "average_sentiment": round(g["sentiment_score"].mean(), 3) if has_sent else None,
            "dominant_sentiment": g["sentiment_label"].mode().iat[0]
                                  if "sentiment_label" in g else None,
            "top_keywords": ", ".join(top_terms[c]),
            "representative_review": (rep_reviews[c][0] if rep_reviews[c] else "")[:300],
        })
    return pd.DataFrame(rows).sort_values("review_count", ascending=False)


# ---------------------------------------------------------------------------
# STEP 6 — Rank top issues by SEVERITY, not frequency alone
# ---------------------------------------------------------------------------
def rank_issues(summary):
    """
    Priority blends four signals so a smaller-but-angrier issue can outrank a
    large-but-mild one:
      volume      (more affected users)          weight 0.30
      1-star %    (intensity of anger)           weight 0.30
      low rating  (5 - avg rating, dissatisfaction) weight 0.20
      negativity  (- avg sentiment, tone)        weight 0.20
    Each signal is min-max normalised to [0,1] so they're comparable.
    """
    s = summary.copy()

    def norm(col):
        lo, hi = col.min(), col.max()
        return (col - lo) / (hi - lo) if hi > lo else col * 0

    s["_vol"] = norm(s["review_count"])
    s["_1star"] = norm(s["one_star_percentage"])
    s["_lowrating"] = norm(5 - s["average_rating"])
    neg_tone = -s["average_sentiment"] if s["average_sentiment"].notna().all() else 0
    s["_negtone"] = norm(neg_tone) if np.ndim(neg_tone) else 0

    s["priority_score"] = (0.30 * s["_vol"] + 0.30 * s["_1star"]
                           + 0.20 * s["_lowrating"] + 0.20 * s["_negtone"]).round(4)
    s = s.sort_values("priority_score", ascending=False)
    return s.drop(columns=["_vol", "_1star", "_lowrating", "_negtone"])


# ---------------------------------------------------------------------------
# STEP 7 — Charts
# ---------------------------------------------------------------------------
def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("chart -> %s/%s", FIG_DIR, name)


def make_charts(metrics, summary):
    # Elbow (inertia vs K)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics["k"], metrics["inertia"], marker="o")
    ax.set(title="Elbow Method (Inertia vs K)", xlabel="K", ylabel="Inertia")
    _save(fig, "topics_elbow.png")

    # Silhouette vs K
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics["k"], metrics["silhouette"], marker="o", color="#55A868")
    ax.set(title="Silhouette Score vs K", xlabel="K", ylabel="Silhouette")
    _save(fig, "topics_silhouette.png")

    s = summary.sort_values("review_count", ascending=True)
    label = s["topic_name"].str.slice(0, 30)

    # Top issues by review count
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(label, s["review_count"], color="#4C72B0")
    ax.set(title="Issues by Review Count", xlabel="Reviews")
    _save(fig, "topics_by_count.png")

    # Issue frequency among negative reviews (%)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(label, s["negative_percentage"], color="#8172B3")
    ax.set(title="Issue Share of Negative Reviews (%)", xlabel="% of negative reviews")
    _save(fig, "topics_negative_share.png")

    # Issue vs average rating
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(label, s["average_rating"], color="#DD8452")
    ax.set(title="Average Rating by Issue", xlabel="Avg rating", xlim=(1, 5))
    _save(fig, "topics_avg_rating.png")

    # Issue vs sentiment
    if summary["average_sentiment"].notna().any():
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(label, s["average_sentiment"], color="#C44E52")
        ax.set(title="Average Sentiment by Issue", xlabel="Avg sentiment score")
        _save(fig, "topics_avg_sentiment.png")

    # 1-star % by issue
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(label, s["one_star_percentage"], color="#C44E52")
    ax.set(title="1-Star % by Issue", xlabel="% 1-star reviews")
    _save(fig, "topics_one_star_pct.png")


# ---------------------------------------------------------------------------
# STEP 8 — Topic validation notes (human-readable)
# ---------------------------------------------------------------------------
def write_validation(summary, top_terms, rep_reviews, k):
    os.makedirs(os.path.dirname(VALIDATION_PATH), exist_ok=True)
    lines = ["TOPIC VALIDATION", "=" * 60,
             f"K = {k} clusters. For each: keywords, size, sample reviews,",
             "and a quick coherence note (needs human judgement).", ""]
    for _, row in summary.iterrows():
        c = row["cluster_id"]
        lines.append(f"[Cluster {c}] {row['topic_name']}  (n={row['review_count']}, "
                     f"1-star={row['one_star_percentage']}%, avg={row['average_rating']})")
        lines.append(f"  keywords: {', '.join(top_terms[c])}")
        lines.append("  representative reviews:")
        for r in rep_reviews[c][:N_REP_REVIEWS]:
            lines.append(f"    - {str(r)[:200]}")
        # Simple coherence heuristic: do the top keywords look related? We can't
        # judge automatically, so we cue the human with the sign to look for.
        lines.append("  coherence: review keywords above -- coherent if they point "
                     "to ONE theme; broad/noisy if they mix unrelated complaints.")
        lines.append("")

    lines += [
        "", "=" * 60,
        "LIMITATIONS OF TF-IDF + K-MEANS",
        "-" * 60,
        "- TF-IDF relies on word OVERLAP, not meaning: two reviews about the same",
        "  problem in different words can land in different clusters.",
        "- Synonyms/paraphrases ('cant login' vs 'unable to sign in') may split.",
        "- Hinglish / multilingual reviews cluster partly by LANGUAGE, not issue",
        "  (see any 'hai/bahut/nahi' style cluster) -- reduces quality.",
        "- Sarcasm and context are invisible to a bag-of-words model.",
        "- K-Means assumes round, similar-sized clusters; real issues aren't.",
        "- Low silhouette scores here mean clusters overlap: treat topics as",
        "  DIRECTIONAL, and always human-validate before acting.",
        "", "=" * 60,
        "WHY TOPIC FREQUENCY ALONE SHOULD NOT SET PRIORITY",
        "-" * 60,
        "A 3,000-review topic can be LESS urgent than an 800-review topic if the",
        "smaller one has much higher 1-star concentration, a much lower average",
        "rating, stronger negative sentiment, or is growing fast. Frequency = how",
        "many; severity = how badly. Our priority_score blends volume (0.30),",
        "1-star % (0.30), low rating (0.20) and negative tone (0.20) for this reason.",
        "", "=" * 60,
        "NO CAUSAL CLAIMS",
        "-" * 60,
        "We describe correlation only. Say 'authentication complaints rose around",
        "the period of version X', NOT 'version X caused authentication problems'.",
    ]

    with open(VALIDATION_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log.info("validation -> %s", VALIDATION_PATH)


# ---------------------------------------------------------------------------
# STEP 9 — Merge issue labels back onto the FULL dataset & save
# ---------------------------------------------------------------------------
def save_outputs(full_df, neg, summary, ranked):
    # Non-clustered rows (positive / too-short) get -1 / 'not_analyzed'.
    full_df = full_df.copy()
    full_df["issue_cluster"] = -1
    full_df["issue_topic"] = "not_analyzed"
    full_df.loc[neg.index, "issue_cluster"] = neg["issue_cluster"].values
    full_df.loc[neg.index, "issue_topic"] = neg["issue_topic"].values

    os.makedirs(os.path.dirname(OUTPUT_REVIEWS), exist_ok=True)
    full_df.to_csv(OUTPUT_REVIEWS, index=False, encoding="utf-8")
    log.info("reviews+topics -> %s", OUTPUT_REVIEWS)

    os.makedirs(os.path.dirname(OUTPUT_SUMMARY), exist_ok=True)
    ranked.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")
    log.info("topic summary -> %s", OUTPUT_SUMMARY)

    # Console: the top-5 prioritised issues.
    log.info("=" * 60)
    log.info("TOP 5 ISSUES BY SEVERITY (not frequency alone)")
    cols = ["topic_name", "review_count", "one_star_percentage",
            "average_rating", "priority_score"]
    for _, r in ranked.head(5).iterrows():
        flag = " [NOISE]" if r["is_noise"] else ""
        log.info("  %.4f | %-32s | n=%d | 1star=%.0f%% | avg=%.2f%s",
                 r["priority_score"], str(r["topic_name"])[:32],
                 r["review_count"], r["one_star_percentage"], r["average_rating"], flag)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    full_df, neg = load_and_split(INPUT_PATH)
    X, vec = build_tfidf(neg["content_clean"])

    metrics = evaluate_k(X, K_RANGE)
    print("\nK comparison table:\n", metrics.to_string(index=False), "\n")
    k = choose_k(metrics)

    neg, km, top_terms, names, rep_reviews = cluster_and_name(neg, X, vec, k)
    summary = summarize(neg, top_terms, names, rep_reviews)
    ranked = rank_issues(summary)

    make_charts(metrics, summary)
    write_validation(summary, top_terms, rep_reviews, k)
    save_outputs(full_df, neg, summary, ranked)


if __name__ == "__main__":
    main()
