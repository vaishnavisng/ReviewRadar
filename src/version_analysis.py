"""
ReviewRadar — Layer 7: Version & Time-Series Analysis
=====================================================

Input : data/processed/reviews_with_topics.csv  (sentiment + topics already added)
DB    : data/processed/reviewradar.db            (SQLite; loaded if missing)
SQL   : sql/version_analysis.sql                 (queries executed from here)
Output: outputs/version_impact.csv
        outputs/version_issue_analysis.csv
        outputs/version_report.txt               (validation + insights + limits)
        outputs/charts/*.png

Goal: see how feedback moves over TIME and across app VERSIONS, and flag versions
associated with unusually high negative feedback -- using only Pandas + SQL/SQLite
+ Matplotlib/Seaborn. No new/advanced tech.

IMPORTANT framing:
  * reviewCreatedVersion = the app version the reviewer was on. It is NOT the
    official release date. We never invent release dates.
  * We describe CORRELATION only ("complaints rose around version X"), never
    causation ("version X caused ...").

Run it:
    python src/version_analysis.py
"""

import os
import re
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INPUT_PATH = "data/processed/reviews_with_topics.csv"
DB_PATH = "data/processed/reviewradar.db"
SQL_PATH = "sql/version_analysis.sql"
CHART_DIR = "outputs/charts"
IMPACT_CSV = "outputs/version_impact.csv"
ISSUE_CSV = "outputs/version_issue_analysis.csv"
REPORT_PATH = "outputs/version_report.txt"

MIN_VERSION_REVIEWS = 30      # below this, per-version percentages are unstable
sns.set_theme(style="whitegrid")


# ===========================================================================
# 1. DATA LOADING (+ load into SQLite if not already there)
# ===========================================================================
def load_data():
    df = pd.read_csv(INPUT_PATH)
    df["reviewCreatedVersion"] = df["reviewCreatedVersion"].fillna("unknown").astype(str)
    df["review_datetime"] = pd.to_datetime(df["review_datetime"], errors="coerce")
    return df


def load_into_sqlite(df):
    """Create the reviews table only if it isn't already present."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    exists = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='reviews'"
    ).fetchone()
    if not exists:
        df.to_sql("reviews", con, index=False)
        print(f"[sqlite] loaded {len(df)} rows into {DB_PATH}")
    else:
        print(f"[sqlite] reviews table already present in {DB_PATH}")
    return con


def load_sql_queries(path):
    """Parse the .sql file into {name: query} using '-- name:' markers."""
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


# ===========================================================================
# Version ordering — natural sort so 8.10 > 8.9 (not alphabetical)
# ===========================================================================
def version_key(v):
    """Turn '18.10.1' into (18,10,1) for correct numeric ordering. Non-numeric
    versions fall back to () so they sort first and we flag the limitation."""
    nums = re.findall(r"\d+", str(v))
    return tuple(int(n) for n in nums) if nums else ()


def sort_versions(versions):
    return sorted(versions, key=version_key)


# ===========================================================================
# 2. TIME AGGREGATION
# ===========================================================================
def choose_granularity(df):
    """Pick weekly vs monthly from the data span, not arbitrarily.
    Rule: if fewer than ~12 months of data, monthly gives too few points, so
    use weekly; otherwise monthly."""
    span_days = (df["review_datetime"].max() - df["review_datetime"].min()).days
    months = span_days / 30
    freq, label = ("W", "weekly") if months < 12 else ("ME", "monthly")
    print(f"[time] span ~{span_days} days (~{months:.1f} months) -> {label} granularity")
    return freq, label


def time_series(df, freq):
    ts = df.set_index("review_datetime").sort_index()
    out = pd.DataFrame({
        "review_volume": ts["score"].resample(freq).size(),
        "avg_rating": ts["score"].resample(freq).mean(),
        "negative_pct": ts["sentiment_label"].resample(freq).apply(
            lambda s: 100 * (s == "negative").mean()),
        "avg_sentiment": ts["sentiment_signed"].resample(freq).mean()
            if "sentiment_signed" in ts else None,
    })
    return out.dropna(how="all")


# ===========================================================================
# 3+2. VERSION AGGREGATION -> version-impact table
# ===========================================================================
def dominant_issue(group):
    """Most common discovered issue among a version's clustered reviews, and its
    share. Uses the topic labels from the mining layer (never hardcoded)."""
    clustered = group[group["issue_cluster"] != -1]
    if clustered.empty:
        return "n/a", 0.0
    counts = clustered["issue_topic"].value_counts()
    return counts.index[0], round(100 * counts.iloc[0] / len(clustered), 1)


def version_table(df):
    rows = []
    for version, g in df[df["reviewCreatedVersion"] != "unknown"].groupby("reviewCreatedVersion"):
        issue, issue_pct = dominant_issue(g)
        rows.append({
            "version": version,
            "review_count": len(g),
            "average_rating": round(g["score"].mean(), 2),
            "negative_percentage": round(100 * (g["sentiment_label"] == "negative").mean(), 1),
            "one_star_percentage": round(100 * (g["score"] == 1).mean(), 1),
            "two_star_percentage": round(100 * (g["score"] == 2).mean(), 1),
            "average_sentiment": round(g["sentiment_signed"].mean(), 3)
                if "sentiment_signed" in g else None,
            "dominant_issue": issue,
            "issue_percentage": issue_pct,
        })
    tbl = pd.DataFrame(rows)
    tbl["_key"] = tbl["version"].map(version_key)
    tbl = tbl.sort_values("_key").drop(columns="_key").reset_index(drop=True)
    return tbl


# ===========================================================================
# 4. IDENTIFY UNUSUAL VERSIONS (data-driven, no arbitrary threshold)
# ===========================================================================
def flag_unusual(tbl):
    """Compare each stable version's negative % against the cross-version mean
    using a z-score. |z|>1 AND negative-% above mean = 'unusually negative'.
    This is relative to the app's own versions, not a magic constant."""
    stable = tbl[tbl["review_count"] >= MIN_VERSION_REVIEWS].copy()
    mean, std = stable["negative_percentage"].mean(), stable["negative_percentage"].std()
    if std and std > 0:
        stable["neg_zscore"] = ((stable["negative_percentage"] - mean) / std).round(2)
    else:
        stable["neg_zscore"] = 0.0
    stable["unusual_negative"] = (stable["neg_zscore"] > 1)
    return stable, round(mean, 1), round(std, 1) if std else 0.0


# ===========================================================================
# 5. ISSUE-BY-VERSION analysis (+ matrix for the heatmap)
# ===========================================================================
def issue_by_version(df):
    """Long table: for each version, each issue's share of that version's
    clustered reviews. Also returns a wide matrix (versions x issues, % values)."""
    clustered = df[(df["issue_cluster"] != -1) &
                   (df["reviewCreatedVersion"] != "unknown")]
    counts = (clustered.groupby(["reviewCreatedVersion", "issue_topic"]).size()
              .rename("review_count").reset_index())
    totals = counts.groupby("reviewCreatedVersion")["review_count"].transform("sum")
    counts["pct_of_version_issues"] = (100 * counts["review_count"] / totals).round(1)
    counts = counts.rename(columns={"reviewCreatedVersion": "version",
                                     "issue_topic": "issue"})

    matrix = counts.pivot(index="version", columns="issue",
                          values="pct_of_version_issues").fillna(0)
    matrix = matrix.reindex(sort_versions(matrix.index))
    return counts.sort_values(["version", "pct_of_version_issues"], ascending=[True, False]), matrix


# ===========================================================================
# 6. SQL ANALYSIS (+ consistency check vs Pandas)
# ===========================================================================
def run_sql(con, queries):
    results = {name: pd.read_sql_query(q, con) for name, q in queries.items()}
    for name, r in results.items():
        print(f"[sql] {name}: {len(r)} rows")
    return results


def check_consistency(df, sql_results):
    """Compare SQL avg-rating-by-version with the Pandas computation."""
    pd_avg = (df[df["reviewCreatedVersion"] != "unknown"]
              .groupby("reviewCreatedVersion")["score"].mean().round(2)
              .sort_index())
    sql_avg = (sql_results["avg_rating_by_version"]
               .set_index("version")["avg_rating"].sort_index())
    aligned = pd_avg.align(sql_avg, join="inner")
    max_diff = (aligned[0] - aligned[1]).abs().max()
    ok = max_diff <= 0.01
    print(f"[check] SQL vs Pandas avg-rating max diff = {max_diff:.4f} -> "
          f"{'CONSISTENT' if ok else 'MISMATCH'}")
    return ok, float(max_diff)


# ===========================================================================
# 9. BEFORE / AFTER a version first appears (rough proxy — no release dates!)
# ===========================================================================
def before_after(df, version):
    """Split ALL reviews by the date this version FIRST appears in the data and
    compare. LIMITATION: first-seen date is a proxy, not the release date, and
    the split is confounded by the app's overall time trend."""
    vdf = df[df["reviewCreatedVersion"] == version]
    if vdf.empty:
        return None
    pivot = vdf["review_datetime"].min()
    before = df[df["review_datetime"] < pivot]
    after = df[df["review_datetime"] >= pivot]

    def snap(g):
        return (round(g["score"].mean(), 2),
                round(100 * (g["sentiment_label"] == "negative").mean(), 1))
    b_rating, b_neg = snap(before) if len(before) else (None, None)
    a_rating, a_neg = snap(after) if len(after) else (None, None)
    return {"version": version, "first_seen": pivot.date(),
            "before_n": len(before), "before_avg_rating": b_rating, "before_neg_pct": b_neg,
            "after_n": len(after), "after_avg_rating": a_rating, "after_neg_pct": a_neg}


# ===========================================================================
# 11+12. VISUALIZATIONS
# ===========================================================================
def _save(fig, name):
    os.makedirs(CHART_DIR, exist_ok=True)
    fig.savefig(os.path.join(CHART_DIR, name), dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_charts(ts, tbl, matrix, freq_label):
    stable = tbl[tbl["review_count"] >= MIN_VERSION_REVIEWS]
    vlabel = stable["version"]

    # --- Time-series (3) ---
    for col, title, color, fname in [
        ("avg_rating", f"Average Rating Over Time ({freq_label})", "#55A868", "ts_avg_rating.png"),
        ("negative_pct", f"Negative Review % Over Time ({freq_label})", "#C44E52", "ts_negative_pct.png"),
        ("review_volume", f"Review Volume Over Time ({freq_label})", "#4C72B0", "ts_review_volume.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ts.index, ts[col], marker="o", color=color, ms=3)
        ax.set(title=title, xlabel="Time", ylabel=col)
        if col == "avg_rating":
            ax.set_ylim(1, 5)
        _save(fig, fname)

    # --- Version bars (4) ---
    for col, title, color, fname in [
        ("average_rating", "Average Rating by Version", "#DD8452", "ver_avg_rating.png"),
        ("negative_percentage", "Negative Review % by Version", "#C44E52", "ver_negative_pct.png"),
        ("one_star_percentage", "1-Star % by Version", "#C44E52", "ver_one_star_pct.png"),
        ("average_sentiment", "Average Sentiment by Version", "#8172B3", "ver_sentiment.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(vlabel, stable[col], color=color)
        ax.set(title=f"{title} (>={MIN_VERSION_REVIEWS} reviews)", xlabel="Version", ylabel=col)
        plt.xticks(rotation=90)
        if col == "average_rating":
            ax.set_ylim(1, 5)
        _save(fig, fname)

    # --- Top issue % by version (bar, labelled with the issue name) ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(vlabel, stable["issue_percentage"], color="#4C72B0")
    for x, (_, row) in zip(range(len(stable)), stable.iterrows()):
        ax.text(x, row["issue_percentage"] + 1, str(row["dominant_issue"])[:14],
                rotation=90, ha="center", va="bottom", fontsize=7)
    ax.set(title="Dominant Issue Share by Version", xlabel="Version",
           ylabel="% of version's issues")
    plt.xticks(rotation=90)
    _save(fig, "ver_top_issue.png")

    # --- Version x Issue heatmap ---
    heat = matrix.loc[matrix.index.intersection(stable["version"])]
    if not heat.empty:
        fig, ax = plt.subplots(figsize=(max(8, len(heat.columns)), max(5, 0.4 * len(heat))))
        sns.heatmap(heat, annot=True, fmt=".0f", cmap="Reds", ax=ax, cbar_kws={"label": "%"})
        ax.set(title="Issue Concentration by Version (% of version's issues)",
               xlabel="Issue", ylabel="Version")
        _save(fig, "version_issue_heatmap.png")


# ===========================================================================
# 16. VALIDATION  +  13. INSIGHTS  +  17. LIMITATIONS
# ===========================================================================
def validate(df, tbl):
    checks = {
        "source_reviews": len(df),
        "reviews_with_version": int((df["reviewCreatedVersion"] != "unknown").sum()),
        "missing_version": int((df["reviewCreatedVersion"] == "unknown").sum()),
        "missing_dates": int(df["review_datetime"].isna().sum()),
        "unique_versions": df.loc[df["reviewCreatedVersion"] != "unknown",
                                  "reviewCreatedVersion"].nunique(),
        "versions_in_impact_table": len(tbl),
        "stable_versions(>=%d)" % MIN_VERSION_REVIEWS:
            int((tbl["review_count"] >= MIN_VERSION_REVIEWS).sum()),
    }
    return checks


def build_insights(df, tbl, unusual, mean_neg):
    overall_rating = round(df["score"].mean(), 2)
    overall_neg = round(100 * (df["sentiment_label"] == "negative").mean(), 1)
    stable = tbl[tbl["review_count"] >= MIN_VERSION_REVIEWS]
    ins = []
    if stable.empty:
        return ["Not enough per-version data (no version has >= "
                f"{MIN_VERSION_REVIEWS} reviews) to draw stable version insights."]

    worst = stable.loc[stable["negative_percentage"].idxmax()]
    ins.append(
        f"Version {worst['version']} had the highest negative share at "
        f"{worst['negative_percentage']}% (avg rating {worst['average_rating']}, "
        f"1-star {worst['one_star_percentage']}%) vs the overall {overall_neg}% "
        f"negative and {overall_rating} avg rating.")

    best = stable.loc[stable["average_rating"].idxmax()]
    ins.append(
        f"Best-received stable version was {best['version']} at {best['average_rating']} "
        f"avg rating ({best['negative_percentage']}% negative).")

    if not unusual.empty and unusual["unusual_negative"].any():
        u = unusual[unusual["unusual_negative"]].sort_values("neg_zscore", ascending=False)
        names = ", ".join(f"{r.version} (z={r.neg_zscore})" for r in u.itertuples())
        ins.append(f"Versions with unusually high negative feedback (z>1 vs the "
                   f"cross-version mean of {mean_neg}%): {names}.")
    else:
        ins.append("No version stood out as statistically unusual (all within ~1 SD "
                   "of the cross-version mean) — differences are mild.")

    # Dominant issue callout (correlation language only).
    top_issue_row = stable.loc[stable["issue_percentage"].idxmax()]
    ins.append(
        f"'{top_issue_row['dominant_issue']}' was the most concentrated issue in a "
        f"single version: {top_issue_row['issue_percentage']}% of version "
        f"{top_issue_row['version']}'s clustered reviews (association, not causation).")
    return ins


LIMITATIONS = [
    "Not every review carries an app version (missing values excluded from version stats).",
    "reviewCreatedVersion is the reviewer's version, NOT the official release date.",
    "Review timing may not match the update date; we use first-seen-in-data as a proxy only.",
    "Reviews are self-selected - unhappy users often over-represented.",
    "Version/complaint correlation does NOT prove the version caused the complaint.",
    "Versions with few reviews give unstable percentages (we require >= "
    f"{MIN_VERSION_REVIEWS} for stable comparisons).",
    "Before/after split is confounded by the app's overall time trend.",
]


def write_report(checks, insights, ba, consistency):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    lines = ["VERSION & TIME-SERIES REPORT", "=" * 60, "", "VALIDATION CHECKS", "-" * 60]
    for k, v in checks.items():
        lines.append(f"  {k:<32}: {v}")
    ok, diff = consistency
    lines.append(f"  {'sql_vs_pandas_avg_rating':<32}: "
                 f"{'CONSISTENT' if ok else 'MISMATCH'} (max diff {diff:.4f})")

    lines += ["", "INSIGHTS (data-backed)", "-" * 60]
    lines += [f"  {i+1}. {t}" for i, t in enumerate(insights)]

    if ba:
        lines += ["", "BEFORE / AFTER highest-volume version appears (PROXY, confounded)",
                  "-" * 60,
                  f"  version {ba['version']} first seen {ba['first_seen']}",
                  f"  before: n={ba['before_n']}, avg={ba['before_avg_rating']}, neg={ba['before_neg_pct']}%",
                  f"  after : n={ba['after_n']}, avg={ba['after_avg_rating']}, neg={ba['after_neg_pct']}%",
                  "  NOTE: split reflects overall time trend too, not the version alone."]

    lines += ["", "LIMITATIONS", "-" * 60]
    lines += [f"  - {t}" for t in LIMITATIONS]
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[report] -> {REPORT_PATH}")


# ===========================================================================
# MAIN — wires the modular steps together
# ===========================================================================
def main():
    df = load_data()
    con = load_into_sqlite(df)
    queries = load_sql_queries(SQL_PATH)

    freq, freq_label = choose_granularity(df)
    ts = time_series(df, freq)

    tbl = version_table(df)
    unusual, mean_neg, std_neg = flag_unusual(tbl)
    issue_long, matrix = issue_by_version(df)

    sql_results = run_sql(con, queries)
    consistency = check_consistency(df, sql_results)

    # before/after on the highest-volume version (most data = least noisy proxy)
    stable = tbl[tbl["review_count"] >= MIN_VERSION_REVIEWS]
    ba = before_after(df, stable.loc[stable["review_count"].idxmax(), "version"]) \
        if not stable.empty else None

    make_charts(ts, tbl, matrix, freq_label)

    # Outputs
    os.makedirs("outputs", exist_ok=True)
    # merge the unusual-version z-scores back onto the impact table for the CSV
    tbl_out = tbl.merge(unusual[["version", "neg_zscore", "unusual_negative"]],
                        on="version", how="left")
    tbl_out.to_csv(IMPACT_CSV, index=False, encoding="utf-8")
    issue_long.to_csv(ISSUE_CSV, index=False, encoding="utf-8")
    print(f"[out] {IMPACT_CSV}  |  {ISSUE_CSV}  |  {CHART_DIR}/*.png")

    checks = validate(df, tbl)
    insights = build_insights(df, tbl, unusual, mean_neg)
    write_report(checks, insights, ba, consistency)
    con.close()


if __name__ == "__main__":
    main()
