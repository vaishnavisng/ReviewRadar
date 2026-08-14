"""
ReviewRadar — Layer 8: Product Prioritization & Business Recommendations
=======================================================================

Turns the earlier analysis into a TRANSPARENT, explainable priority ranking for
a Product Manager. No new ML -- just Pandas + SQL + Matplotlib/Seaborn.

Inputs : data/processed/reviews_with_topics.csv   (sentiment + issue clusters)
         outputs/version_issue_analysis.csv        (issue share per version)
         outputs/topic_summary.csv                 (topic names, keywords, is_noise)
         data/processed/reviewradar.db             (SQLite; created in Layer 7)
SQL    : sql/product_prioritization.sql
Outputs: outputs/product_priorities.csv
         outputs/executive_summary.txt
         outputs/charts/*.png

PRIORITY FORMULA (fully documented, not a black box):
    score = 100 * (0.40*Reach + 0.30*Severity + 0.20*Trend + 0.10*VersionAssoc)
  where each dimension is min-max normalised to [0,1] across the analysed issues:
    Reach     = review_count
    Severity  = mean of normalised(1-star %), normalised(5 - avg_rating),
                normalised(negative-sentiment %)
    Trend     = max(recent_share - earlier_share, 0)   (only rising issues gain)
    VersionAssoc = max(version's issue-share - overall issue-share, 0)

PRIORITY BANDS (documented thresholds on the 0-100 score):
    P0 Critical >= 70 | P1 High 50-69 | P2 Medium 30-49 | P3 Low < 30

NOISE clusters (flagged is_noise in topic_summary) are EXCLUDED: they are
language/junk artifacts, and their large volume would distort Reach normalisation.

We keep strict wording: association, not causation. Recommendations are
hypotheses tied to observed keywords, never invented technical root causes.

Run it:
    python src/prioritization.py
"""

import os
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REVIEWS_PATH = "data/processed/reviews_with_topics.csv"
VERSION_ISSUE_PATH = "outputs/version_issue_analysis.csv"
TOPIC_SUMMARY_PATH = "outputs/topic_summary.csv"
DB_PATH = "data/processed/reviewradar.db"
SQL_PATH = "sql/product_prioritization.sql"
OUT_CSV = "outputs/product_priorities.csv"
SUMMARY_PATH = "outputs/executive_summary.txt"
CHART_DIR = "outputs/charts"

WEIGHTS = {"reach": 0.40, "severity": 0.30, "trend": 0.20, "version": 0.10}
BANDS = [(70, "P0 - Critical"), (50, "P1 - High"), (30, "P2 - Medium"), (0, "P3 - Low")]
TREND_REL = 0.15          # >=15% relative change = Increasing/Decreasing, else Stable
MIN_PERIOD_REVIEWS = 10   # per-period reviews needed to trust a trend
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_inputs():
    df = pd.read_csv(REVIEWS_PATH)
    df["review_datetime"] = pd.to_datetime(df["review_datetime"], errors="coerce")
    topic = pd.read_csv(TOPIC_SUMMARY_PATH)
    ver_issue = pd.read_csv(VERSION_ISSUE_PATH) if os.path.exists(VERSION_ISSUE_PATH) \
        else pd.DataFrame(columns=["version", "issue", "review_count", "pct_of_version_issues"])
    return df, topic, ver_issue


def load_sql_queries(path):
    queries, name, buf = {}, None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip().lower().startswith("-- name:"):
                if name:
                    queries[name] = "".join(buf).strip()
                name, buf = line.split(":", 1)[1].strip(), []
            elif name:
                buf.append(line)
    if name:
        queries[name] = "".join(buf).strip()
    return queries


def norm(s):
    """Min-max to [0,1]; if all equal, return 0.5 so it neither helps nor hurts."""
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=s.index)


# ---------------------------------------------------------------------------
# 1. Issue-level metrics
# ---------------------------------------------------------------------------
def issue_metrics(df, topic):
    """One row per (real) issue cluster with all raw metrics."""
    total_negative = int((df["sentiment_label"] == "negative").sum())
    real = topic[~topic["is_noise"]]                       # drop noise/language clusters
    clustered = df[df["issue_cluster"].isin(real["cluster_id"])]

    rows = []
    for cid, g in clustered.groupby("issue_cluster"):
        trow = topic[topic["cluster_id"] == cid].iloc[0]
        rows.append({
            "issue_cluster": cid,
            "issue_topic": trow["topic_name"],
            "top_keywords": trow["top_keywords"],
            "review_count": len(g),
            "negative_percentage": round(100 * (g["sentiment_label"] == "negative").sum()
                                         / total_negative, 1) if total_negative else 0,
            "one_star_percentage": round(100 * (g["score"] == 1).mean(), 1),
            "two_star_percentage": round(100 * (g["score"] == 2).mean(), 1),
            "average_rating": round(g["score"].mean(), 2),
            "average_sentiment": round(g["sentiment_score"].mean(), 3)
                if "sentiment_score" in g else None,
            "negative_sentiment_percentage": round(100 * (g["sentiment_label"] == "negative").mean(), 1),
        })
    return pd.DataFrame(rows), clustered


# ---------------------------------------------------------------------------
# 2. Recent trend (explainable earlier-vs-recent comparison)
# ---------------------------------------------------------------------------
def add_trend(metrics, clustered):
    """Split clustered reviews at the midpoint date; compare each issue's SHARE
    of clustered reviews in the recent half vs the earlier half."""
    dmin, dmax = clustered["review_datetime"].min(), clustered["review_datetime"].max()
    mid = dmin + (dmax - dmin) / 2
    earlier = clustered[clustered["review_datetime"] < mid]
    recent = clustered[clustered["review_datetime"] >= mid]

    def share(period, cid):
        return 100 * (period["issue_cluster"] == cid).mean() if len(period) else None

    trends, deltas = [], []
    for cid in metrics["issue_cluster"]:
        e_n = int((earlier["issue_cluster"] == cid).sum())
        r_n = int((recent["issue_cluster"] == cid).sum())
        e_share, r_share = share(earlier, cid), share(recent, cid)
        if e_n < MIN_PERIOD_REVIEWS or r_n < MIN_PERIOD_REVIEWS or not e_share:
            trends.append("Insufficient data"); deltas.append(0.0); continue
        rel = (r_share - e_share) / e_share
        label = ("Increasing" if rel >= TREND_REL else
                 "Decreasing" if rel <= -TREND_REL else "Stable")
        trends.append(label)
        deltas.append(round(r_share - e_share, 2))
    metrics = metrics.copy()
    metrics["recent_trend"] = trends
    metrics["_trend_delta"] = deltas
    return metrics


# ---------------------------------------------------------------------------
# 3. Version association (association wording, never causal)
# ---------------------------------------------------------------------------
def add_version_assoc(metrics, ver_issue, clustered):
    total_clustered = len(clustered)
    assoc_text, assoc_raw = [], []
    for _, m in metrics.iterrows():
        overall = 100 * m["review_count"] / total_clustered if total_clustered else 0
        sub = ver_issue[(ver_issue["issue"] == m["issue_topic"]) &
                        (ver_issue["review_count"] >= MIN_PERIOD_REVIEWS)]
        if sub.empty:
            assoc_text.append("No strong version concentration"); assoc_raw.append(0.0); continue
        top = sub.loc[sub["pct_of_version_issues"].idxmax()]
        delta = top["pct_of_version_issues"] - overall
        if delta >= 10:
            assoc_text.append(f"Concentrated in v{top['version']} "
                              f"({top['pct_of_version_issues']}% vs ~{overall:.0f}% overall)")
        else:
            assoc_text.append("No strong version concentration")
        assoc_raw.append(max(delta, 0.0))
    metrics = metrics.copy()
    metrics["version_association"] = assoc_text
    metrics["_version_raw"] = assoc_raw
    return metrics


# ---------------------------------------------------------------------------
# 4+5. Priority score (transparent, normalised, weighted)
# ---------------------------------------------------------------------------
def add_priority(metrics):
    m = metrics.copy()
    reach = norm(m["review_count"])
    severity = (norm(m["one_star_percentage"]) +
                norm(5 - m["average_rating"]) +
                norm(m["negative_sentiment_percentage"])) / 3
    trend = norm(m["_trend_delta"].clip(lower=0))
    version = norm(pd.Series(m["_version_raw"].values, index=m.index).clip(lower=0))

    m["priority_score"] = (100 * (WEIGHTS["reach"] * reach +
                                  WEIGHTS["severity"] * severity +
                                  WEIGHTS["trend"] * trend +
                                  WEIGHTS["version"] * version)).round(1)

    def band(score):
        for cutoff, label in BANDS:
            if score >= cutoff:
                return label
        return "P3 - Low"
    m["priority_level"] = m["priority_score"].apply(band)
    return m.sort_values("priority_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Recommendations (Evidence -> Inference -> Recommendation, no invented causes)
# ---------------------------------------------------------------------------
def add_recommendations(m):
    m = m.copy()
    # Potential product area = the business-named topic (not a guessed root cause).
    m["potential_product_area"] = m["issue_topic"]
    actions = []
    for _, r in m.iterrows():
        kw = ", ".join(str(r["top_keywords"]).split(", ")[:4])
        actions.append(
            f"Investigate the '{r['issue_topic']}' user journey (reviews cluster on: "
            f"{kw}). Treat as a hypothesis to validate, not a confirmed root cause.")
    m["recommended_action"] = actions
    return m


# ---------------------------------------------------------------------------
# 7. SQL analysis + consistency check
# ---------------------------------------------------------------------------
def run_sql_checks(metrics):
    if not os.path.exists(DB_PATH):
        print("[sql] DB missing (run Layer 7 first) - skipping SQL checks")
        return None, (None, None)
    con = sqlite3.connect(DB_PATH)
    queries = load_sql_queries(SQL_PATH)
    results = {name: pd.read_sql_query(q, con) for name, q in queries.items()}
    for name, r in results.items():
        print(f"[sql] {name}: {len(r)} rows")
    # Compare SQL avg-rating-by-issue with our Pandas metric.
    sql_avg = results["avg_rating_by_issue"].set_index("issue")["avg_rating"]
    pd_avg = metrics.set_index("issue_topic")["average_rating"]
    aligned = pd_avg.align(sql_avg, join="inner")
    max_diff = (aligned[0] - aligned[1]).abs().max() if len(aligned[0]) else 0.0
    ok = max_diff <= 0.01
    print(f"[check] SQL vs Pandas avg-rating-by-issue max diff = {max_diff:.4f} -> "
          f"{'CONSISTENT' if ok else 'MISMATCH'}")
    con.close()
    return results, (ok, float(max_diff))


# ---------------------------------------------------------------------------
# 14. Charts
# ---------------------------------------------------------------------------
def _save(fig, name):
    os.makedirs(CHART_DIR, exist_ok=True)
    fig.savefig(os.path.join(CHART_DIR, name), dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_charts(m):
    label = m["issue_topic"].str.slice(0, 26)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(label[::-1], m["priority_score"][::-1], color="#4C72B0")
    ax.set(title="Priority Score by Issue (0-100)", xlabel="Priority score")
    _save(fig, "prio_score_by_issue.png")

    # severity proxy for scatter = 1-star %
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(m["review_count"], m["one_star_percentage"], s=80, color="#C44E52")
    for _, r in m.iterrows():
        ax.annotate(str(r["issue_topic"])[:18], (r["review_count"], r["one_star_percentage"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set(title="Issue Frequency vs Severity (1-star %)",
           xlabel="Review count (reach)", ylabel="1-star %")
    _save(fig, "prio_freq_vs_severity.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(m["review_count"], m["average_rating"], s=80, color="#DD8452")
    for _, r in m.iterrows():
        ax.annotate(str(r["issue_topic"])[:18], (r["review_count"], r["average_rating"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set(title="Issue Frequency vs Average Rating", xlabel="Review count", ylabel="Avg rating")
    _save(fig, "prio_freq_vs_rating.png")

    # Reach vs Severity priority matrix (quadrants at medians)
    fig, ax = plt.subplots(figsize=(7, 6))
    reach, sev = m["review_count"], m["one_star_percentage"]
    ax.scatter(reach, sev, s=m["priority_score"] * 3, color="#8172B3", alpha=0.7)
    ax.axvline(reach.median(), ls="--", color="grey"); ax.axhline(sev.median(), ls="--", color="grey")
    for _, r in m.iterrows():
        ax.annotate(str(r["issue_topic"])[:18], (r["review_count"], r["one_star_percentage"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set(title="Priority Matrix: Reach vs Severity (bubble = score)",
           xlabel="Reach (review count)", ylabel="Severity (1-star %)")
    _save(fig, "prio_matrix_reach_severity.png")

    top5 = m.head(5)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"P0 - Critical": "#C44E52", "P1 - High": "#DD8452",
              "P2 - Medium": "#55A868", "P3 - Low": "#B0B0B0"}
    ax.barh(top5["issue_topic"][::-1], top5["priority_score"][::-1],
            color=[colors.get(p, "#4C72B0") for p in top5["priority_level"][::-1]])
    ax.set(title="Top 5 Prioritised Issues", xlabel="Priority score")
    _save(fig, "prio_top5.png")


# ---------------------------------------------------------------------------
# 15. Validation
# ---------------------------------------------------------------------------
def validate(df, topic, m):
    top5 = m.head(5)
    checks = {
        "issues_analyzed": len(m),
        "noise_clusters_excluded": int(topic["is_noise"].sum()),
        "reviews_in_analyzed_issues": int(m["review_count"].sum()),
        "missing_issue_labels": int((df["issue_cluster"] == -1).sum()),
        "missing_sentiment": int(df["sentiment_label"].isna().sum()),
        "missing_version": int(df["reviewCreatedVersion"].isna().sum()),
        "score_in_0_100": bool(m["priority_score"].between(0, 100).all()),
        "every_issue_has_level": bool(m["priority_level"].notna().all()),
        "top5_are_highest_scores":
            list(top5["issue_cluster"]) == list(m.nlargest(5, "priority_score")["issue_cluster"]),
    }
    return checks


# ---------------------------------------------------------------------------
# 13. Executive summary (generated from real numbers)
# ---------------------------------------------------------------------------
def executive_summary(df, m, checks, consistency):
    overall_rating = round(df["score"].mean(), 2)
    overall_neg = round(100 * (df["sentiment_label"] == "negative").mean(), 1)
    top5 = m.head(5)
    rising = m[m["recent_trend"] == "Increasing"]["issue_topic"].tolist()
    concentrated = m[~m["version_association"].str.startswith("No strong")]

    L = ["EXECUTIVE SUMMARY - Product Prioritization", "=" * 60,
         f"App overall: avg rating {overall_rating}/5, {overall_neg}% negative (text).",
         f"Analyzed {checks['issues_analyzed']} real issue clusters "
         f"({checks['noise_clusters_excluded']} noise/language clusters excluded).", ""]

    L += ["BIGGEST PROBLEMS (by reach):"]
    for _, r in m.nlargest(3, "review_count").iterrows():
        L.append(f"  - {r['issue_topic']}: {r['review_count']} reviews, "
                 f"avg {r['average_rating']}, 1-star {r['one_star_percentage']}%.")

    L += ["", "ADDRESS FIRST (by transparent priority score):"]
    for _, r in top5.iterrows():
        L.append(f"  [{r['priority_level']}] {r['issue_topic']} "
                 f"(score {r['priority_score']}) - {r['review_count']} reviews, "
                 f"avg {r['average_rating']}, 1-star {r['one_star_percentage']}%, "
                 f"trend: {r['recent_trend']}.")

    L += ["", "GETTING WORSE RECENTLY:"]
    L.append("  " + (", ".join(rising) if rising else
             "No issue shows a clear recent increase (or insufficient data)."))

    L += ["", "VERSION-CONCENTRATED ISSUES (association, not causation):"]
    if concentrated.empty:
        L.append("  None showed strong single-version concentration.")
    else:
        for _, r in concentrated.iterrows():
            L.append(f"  - {r['issue_topic']}: {r['version_association']}")

    L += ["", "INVESTIGATE FIRST:"]
    top = top5.iloc[0]
    L.append(f"  {top['issue_topic']} - {top['recommended_action']}")

    ok, diff = consistency
    L += ["", f"(SQL vs Pandas consistency: "
          f"{'CONSISTENT' if ok else 'n/a' if ok is None else 'MISMATCH'}"
          f"{'' if diff is None else f', max diff {diff:.4f}'})"]

    L += ["", "LIMITATIONS", "-" * 60,
          "- Play Store reviews are self-selected feedback, not all users.",
          "- Complaint frequency != actual incidence rate of the problem.",
          "- Sentiment model and topic clusters can be wrong or noisy.",
          "- Version association does not prove causation.",
          "- Recommendations are hypotheses requiring product validation.",
          "- Review text cannot reveal technical root causes directly."]

    text = "\n".join(L)
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    df, topic, ver_issue = load_inputs()

    metrics, clustered = issue_metrics(df, topic)
    metrics = add_trend(metrics, clustered)
    metrics = add_version_assoc(metrics, ver_issue, clustered)
    metrics = add_priority(metrics)
    metrics = add_recommendations(metrics)

    _, consistency = run_sql_checks(metrics)
    make_charts(metrics)

    cols = ["issue_cluster", "issue_topic", "priority_score", "priority_level",
            "review_count", "negative_percentage", "one_star_percentage",
            "average_rating", "negative_sentiment_percentage", "recent_trend",
            "version_association", "top_keywords", "potential_product_area",
            "recommended_action"]
    out = metrics[cols]
    os.makedirs("outputs", exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[out] {OUT_CSV}  ({len(out)} issues)")

    checks = validate(df, topic, metrics)
    print("\nVALIDATION CHECKS")
    for k, v in checks.items():
        print(f"  {k:<28}: {v}")

    print()
    print(executive_summary(df, metrics, checks, consistency))
    print(f"\n[summary] -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
