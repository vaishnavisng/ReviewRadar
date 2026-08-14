"""
ReviewRadar — Streamlit Dashboard
=================================

Reads ONLY the pre-computed outputs from the earlier layers and presents them as
a product-analytics dashboard. No NLP/ML runs here — all heavy work (sentiment,
TF-IDF, K-Means, version + priority analysis) happened before, in the pipeline.

Run:
    streamlit run dashboard/app.py

App-agnostic: the title and every number come from config + data, so switching
apps is just editing config.py.
"""

import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Central config = the one place the app is named.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

sns.set_theme(style="whitegrid")
SENT_ORDER = ["negative", "neutral", "positive"]
SENT_COLORS = {"negative": "#C44E52", "neutral": "#B0B0B0", "positive": "#55A868"}

# File paths (outputs of the pipeline).
P_REVIEWS = config.REVIEWS_WITH_TOPICS
P_TOPICS = config.TOPIC_SUMMARY
P_VERSION = "outputs/version_impact.csv"
P_VER_ISSUE = "outputs/version_issue_analysis.csv"
P_PRIORITIES = "outputs/product_priorities.csv"
P_DB = "data/processed/reviewradar.db"

st.set_page_config(page_title=f"ReviewRadar — {config.APP_NAME}", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders (so files are read once, not on every interaction)
# ---------------------------------------------------------------------------
@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_reviews(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["review_datetime"] = pd.to_datetime(df["review_datetime"], errors="coerce")
    df["reviewCreatedVersion"] = df["reviewCreatedVersion"].fillna("unknown").astype(str)
    return df.dropna(subset=["review_datetime"])


@st.cache_data
def run_sql(query):
    """Run a read-only SQL query against the SQLite DB (cached by query text)."""
    if not os.path.exists(P_DB):
        return None
    with sqlite3.connect(P_DB) as con:
        return pd.read_sql_query(query, con)


# ---------------------------------------------------------------------------
# Load primary data (hard requirement) — fail gracefully if missing
# ---------------------------------------------------------------------------
reviews = load_reviews(P_REVIEWS)
if reviews is None:
    st.title(f"ReviewRadar — {config.APP_NAME}")
    st.error(f"Missing `{P_REVIEWS}`. Run the pipeline first:\n\n"
             "`python src/scraper.py` → `cleaner.py` → `sentiment.py` → "
             "`topics.py` → `version_analysis.py` → `prioritization.py`")
    st.stop()

topics = load_csv(P_TOPICS)
version_impact = load_csv(P_VERSION)
ver_issue = load_csv(P_VER_ISSUE)
priorities = load_csv(P_PRIORITIES)


def missing(name, step):
    st.info(f"`{name}` not found — run `{step}` to generate it.")


# ---------------------------------------------------------------------------
# Header (gradient banner) + a bit of CSS polish
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800&display=swap');
  .block-container { padding-top: 1.2rem; }
  .rr-banner {
      background: linear-gradient(90deg, #0A1F44 0%, #123C7A 100%);
      padding: 22px 30px; border-radius: 16px; margin-bottom: 18px;
      box-shadow: 0 6px 20px rgba(10,31,68,.35);
  }
  .rr-banner h1 {
      color: #fff; margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing:.5px;
      font-family: 'Montserrat', sans-serif;
  }
  .rr-kpi { padding: 16px 18px; border-radius: 14px; box-shadow: 0 4px 14px rgba(0,0,0,.08); }
  .rr-kpi .lbl { font-size:.72rem; font-weight:700; text-transform:uppercase;
                 letter-spacing:.6px; color: rgba(255,255,255,.9); }
  .rr-kpi .val { font-size:1.55rem; font-weight:800; color:#fff; margin-top:6px; line-height:1.2; }
  .stTabs [data-baseweb="tab"] { font-weight:600; }
  .stTabs [aria-selected="true"] { color:#6C5CE7 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    f'<div class="rr-banner"><h1>ReviewRadar — {config.APP_NAME}</h1></div>',
    unsafe_allow_html=True)


def kpi_card(col, label, value, gradient):
    col.markdown(
        f'<div class="rr-kpi" style="background:{gradient};">'
        f'<div class="lbl">{label}</div><div class="val">{value}</div></div>',
        unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SECTION 8 — Sidebar filters (empty multiselect = no filter / all)
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
dmin, dmax = reviews["review_datetime"].min().date(), reviews["review_datetime"].max().date()
start, end = st.sidebar.slider("Date range", min_value=dmin, max_value=dmax, value=(dmin, dmax))
sel_rating = st.sidebar.multiselect("Star rating", [1, 2, 3, 4, 5], default=[])
sel_sent = st.sidebar.multiselect("Sentiment", SENT_ORDER, default=[])
issue_opts = sorted(reviews.loc[reviews["issue_topic"] != "not_analyzed", "issue_topic"].unique())
sel_issue = st.sidebar.multiselect("Issue / topic", issue_opts, default=[])
ver_opts = sorted(reviews.loc[reviews["reviewCreatedVersion"] != "unknown",
                              "reviewCreatedVersion"].unique())
sel_version = st.sidebar.multiselect("App version", ver_opts, default=[])
st.sidebar.caption("Leave a filter empty to include everything.")


def apply_filters(df):
    m = (df["review_datetime"].dt.date >= start) & (df["review_datetime"].dt.date <= end)
    if sel_rating:
        m &= df["score"].isin(sel_rating)
    if sel_sent:
        m &= df["sentiment_label"].isin(sel_sent)
    if sel_issue:
        m &= df["issue_topic"].isin(sel_issue)
    if sel_version:
        m &= df["reviewCreatedVersion"].isin(sel_version)
    return df[m]


fdf = apply_filters(reviews)
st.sidebar.metric("Reviews in view", f"{len(fdf):,}")

if fdf.empty:
    st.warning("No reviews match the current filters. Widen them in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# SECTION 1 — Executive Overview KPIs (from filtered data)
# ---------------------------------------------------------------------------
neg_pct = 100 * (fdf["sentiment_label"] == "negative").mean()
n_issues = int((~topics["is_noise"]).sum()) if topics is not None else 0
critical = "n/a"
if priorities is not None and len(priorities):
    critical = priorities.sort_values("priority_score", ascending=False).iloc[0]["issue_topic"]

k1, k2, k3, k4, k5 = st.columns(5)
kpi_card(k1, "Total reviews", f"{len(fdf):,}", "linear-gradient(135deg,#4C6FFF,#6C8DFF)")
kpi_card(k2, "Avg rating", f"{fdf['score'].mean():.2f} ★", "linear-gradient(135deg,#00B894,#55EFC4)")
kpi_card(k3, "Negative %", f"{neg_pct:.1f}%", "linear-gradient(135deg,#C0392B,#E74C3C)")
kpi_card(k4, "Major issues", n_issues, "linear-gradient(135deg,#F39C12,#F1C40F)")
kpi_card(k5, "Most critical", critical, "linear-gradient(135deg,#E84393,#FDA7DF)")
st.write("")

tabs = st.tabs(["Overview", "Rating & Sentiment", "Issues",
                "Version Intelligence", "Product Priorities", "Review Explorer"])

# ===========================================================================
# TAB 1 — OVERVIEW (executive summary + SQL demo)
# ===========================================================================
with tabs[0]:
    # SECTION 2 — Executive summary, generated from real numbers
    st.subheader("Executive summary")
    half = fdf.sort_values("review_datetime")
    first, second = half.iloc[:len(half)//2], half.iloc[len(half)//2:]
    direction = "improving" if second["score"].mean() > first["score"].mean() else "declining"
    noise_names = set(topics[topics["is_noise"]]["topic_name"]) if topics is not None else set()
    real_issues = fdf[(fdf["issue_topic"] != "not_analyzed") &
                      (~fdf["issue_topic"].isin(noise_names))]
    top_issue_counts = real_issues["issue_topic"].value_counts().head(3)

    st.markdown(f"""
**Observations (data-backed):**
- Overall sentiment: **{fdf['score'].mean():.2f}/5** average rating, **{neg_pct:.1f}%** negative reviews (in current view).
- Rating trend within this view appears **{direction}** (first-half vs second-half average).
- Biggest problem areas by volume: {", ".join(f"**{k}** ({v})" for k, v in top_issue_counts.items()) or "n/a"}.
""")
    if priorities is not None and len(priorities):
        p0 = priorities[priorities["priority_level"].str.startswith("P0")]
        top = priorities.sort_values("priority_score", ascending=False).iloc[0]
        st.markdown(f"""
**Recommendation (from priority framework):**
- Investigate first: **{top['issue_topic']}** (priority {top['priority_score']}, {top['priority_level']}).
- {len(p0)} issue(s) are P0-Critical. {top['recommended_action']}
""")
    if version_impact is not None:
        unusual = version_impact[version_impact.get("unusual_negative", False) == True]
        if len(unusual):
            vs = ", ".join(unusual["version"].astype(str))
            st.caption(f"Version note (association, not causation): versions {vs} showed "
                       "unusually high negative feedback vs the app's own cross-version average.")

    # SECTION 10 — SQL integration (full dataset), query shown for transparency
    st.subheader("SQL metrics (full dataset)")
    st.caption("Demonstrates SQL on the SQLite DB. These are full-dataset figures, "
               "independent of the sidebar filters above.")
    q_overview = """-- headline metrics straight from SQLite
SELECT COUNT(*)                                          AS total_reviews,
       ROUND(AVG(score), 2)                              AS avg_rating,
       ROUND(100.0 * SUM(CASE WHEN sentiment_label='negative'
             THEN 1 ELSE 0 END) / COUNT(*), 1)           AS negative_pct
FROM reviews;"""
    res = run_sql(q_overview)
    if res is not None:
        c = res.iloc[0]
        a, b, d = st.columns(3)
        a.metric("Total reviews (SQL)", f"{int(c['total_reviews']):,}")
        b.metric("Avg rating (SQL)", f"{c['avg_rating']}")
        d.metric("Negative % (SQL)", f"{c['negative_pct']}%")
        with st.expander("Show SQL"):
            st.code(q_overview, language="sql")
    else:
        st.info(f"SQLite DB `{P_DB}` not found — run `python src/version_analysis.py`.")

# ===========================================================================
# TAB 2 — RATING & SENTIMENT
# ===========================================================================
with tabs[1]:
    # SECTION 3 — Rating analysis
    st.subheader("Rating analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Rating distribution** — the shape of user satisfaction.")
        st.bar_chart(fdf["score"].value_counts().sort_index())
    with c2:
        st.markdown("**Review volume over time (weekly)** — when users were most vocal.")
        st.line_chart(fdf.set_index("review_datetime")["score"].resample("W").size()
                      .rename("reviews"))
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Average rating over time (monthly)** — satisfaction trend.")
        st.line_chart(fdf.set_index("review_datetime")["score"].resample("ME").mean()
                      .rename("avg_rating"))
    with c4:
        st.markdown("**Negative % over time (monthly)** — when frustration rose.")
        neg_ts = (fdf.set_index("review_datetime")["sentiment_label"]
                  .resample("ME").apply(lambda s: 100 * (s == "negative").mean())
                  .rename("negative_pct"))
        st.line_chart(neg_ts)

    # SECTION 4 — Sentiment intelligence
    st.subheader("Sentiment intelligence")
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("**Sentiment distribution** — overall tone of the reviews.")
        st.bar_chart(fdf["sentiment_label"].value_counts().reindex(SENT_ORDER).fillna(0))
    with c6:
        st.markdown("**Sentiment by star rating** — do words match the stars?")
        ct = pd.crosstab(fdf["score"], fdf["sentiment_label"], normalize="index") * 100
        ct = ct.reindex(columns=SENT_ORDER, fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ct.plot(kind="bar", stacked=True, ax=ax, color=[SENT_COLORS[c] for c in SENT_ORDER])
        ax.set(xlabel="Star rating", ylabel="% of reviews", title="Sentiment by rating")
        ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 1))
        st.pyplot(fig)

    st.markdown("**Sentiment mix over time (monthly)** — is the tone shifting?")
    share = (fdf.groupby(["review_month", "sentiment_label"]).size()
             .groupby(level=0).apply(lambda s: 100 * s / s.sum())
             .unstack(fill_value=0).reindex(columns=SENT_ORDER, fill_value=0))
    fig, ax = plt.subplots(figsize=(11, 4))
    share.plot(kind="area", stacked=True, ax=ax, color=[SENT_COLORS[c] for c in SENT_ORDER], alpha=0.9)
    ax.set(xlabel="Month", ylabel="% of reviews", ylim=(0, 100), title="Sentiment mix over time")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 1))
    st.pyplot(fig)

    st.markdown("**Sentiment by app version** — which builds felt worst? "
                "(association with a version, not proof it caused anything).")
    vdf = fdf[fdf["reviewCreatedVersion"] != "unknown"]
    vc = vdf["reviewCreatedVersion"].value_counts()
    keep = vc[vc >= 30].index
    if len(keep):
        sub = vdf[vdf["reviewCreatedVersion"].isin(keep)]
        vs = (pd.crosstab(sub["reviewCreatedVersion"], sub["sentiment_label"],
                          normalize="index") * 100)
        vs = vs.reindex(columns=SENT_ORDER, fill_value=0).sort_values("negative", ascending=False)
        st.bar_chart(vs)
    else:
        st.info("No version has ≥30 reviews in the current view.")

# ===========================================================================
# TAB 3 — ISSUES (recomputed counts from filtered reviews; topics NOT re-mined)
# ===========================================================================
with tabs[2]:
    st.subheader("Issue intelligence")
    st.caption("Issues were discovered once by the TF-IDF + K-Means layer. Here we "
               "only re-aggregate their pre-assigned labels for the current filter.")
    # Exclude noise/language clusters so this tab matches the KPI and priorities.
    noise_names = set(topics[topics["is_noise"]]["topic_name"]) if topics is not None else set()
    clustered = fdf[(fdf["issue_topic"] != "not_analyzed") &
                    (~fdf["issue_topic"].isin(noise_names))]
    if topics is None:
        missing(P_TOPICS, "python src/topics.py")
    elif clustered.empty:
        st.info("No clustered (issue-labelled) reviews in the current view.")
    else:
        agg = (clustered.groupby("issue_topic")
               .agg(reviews=("score", "size"),
                    avg_rating=("score", "mean"),
                    one_star_pct=("score", lambda s: 100 * (s == 1).mean()),
                    neg_pct=("sentiment_label", lambda s: 100 * (s == "negative").mean()),
                    avg_sentiment=("sentiment_score", "mean"))
               .round(2).sort_values("reviews", ascending=False))
        c1, c2 = st.columns(2)
        c1.markdown("**Top issues by review count**"); c1.bar_chart(agg["reviews"])
        c2.markdown("**1-star % by issue**"); c2.bar_chart(agg["one_star_pct"])
        c3, c4 = st.columns(2)
        c3.markdown("**Negative % by issue**"); c3.bar_chart(agg["neg_pct"])
        c4.markdown("**Avg rating by issue**"); c4.bar_chart(agg["avg_rating"])
        st.dataframe(agg, use_container_width=True)

        # Inspect representative real reviews per issue (from the mining layer).
        st.markdown("**Inspect representative reviews**")
        pick = st.selectbox("Choose an issue", agg.index.tolist())
        samples = clustered[clustered["issue_topic"] == pick]
        text_col = "content_raw" if "content_raw" in samples else "content_clean"
        st.dataframe(samples[["review_date", "score", "sentiment_label", text_col]]
                     .head(15).rename(columns={text_col: "review"}), use_container_width=True)

# ===========================================================================
# TAB 4 — VERSION INTELLIGENCE (from precomputed version outputs)
# ===========================================================================
with tabs[3]:
    st.subheader("Version intelligence")
    st.caption("From the version-analysis layer. Language is associative — a low-rated "
               "version is *associated with* complaints, not proven to have caused them. "
               "reviewCreatedVersion is the reviewer's version, not an official release date.")
    if version_impact is None:
        missing(P_VERSION, "python src/version_analysis.py")
    else:
        vi = version_impact.copy()
        if sel_version:
            vi = vi[vi["version"].astype(str).isin(sel_version)]
        vi = vi[vi["review_count"] >= 30].sort_values("negative_percentage", ascending=False)
        if vi.empty:
            st.info("No versions with ≥30 reviews match the current selection.")
        else:
            c1, c2 = st.columns(2)
            c1.markdown("**Avg rating by version**")
            c1.bar_chart(vi.set_index("version")["average_rating"])
            c2.markdown("**Negative % by version**")
            c2.bar_chart(vi.set_index("version")["negative_percentage"])
            c3, c4 = st.columns(2)
            c3.markdown("**Review volume by version**")
            c3.bar_chart(vi.set_index("version")["review_count"])
            c4.markdown("**Dominant issue by version**")
            c4.dataframe(vi[["version", "dominant_issue", "issue_percentage"]],
                         use_container_width=True, hide_index=True)

        # Version × issue heatmap
        if ver_issue is not None and len(ver_issue):
            st.markdown("**Issue concentration by version (heatmap, % of a version's issues)**")
            mat = ver_issue.pivot_table(index="version", columns="issue",
                                        values="pct_of_version_issues", fill_value=0)
            stable_versions = version_impact[version_impact["review_count"] >= 30]["version"].astype(str)
            mat = mat.loc[mat.index.astype(str).isin(stable_versions)]
            if sel_version:
                mat = mat.loc[mat.index.astype(str).isin(sel_version)]
            if not mat.empty:
                fig, ax = plt.subplots(figsize=(max(8, len(mat.columns)), max(4, 0.4 * len(mat))))
                sns.heatmap(mat, annot=True, fmt=".0f", cmap="Reds", ax=ax, cbar_kws={"label": "%"})
                ax.set(xlabel="Issue", ylabel="Version")
                st.pyplot(fig)

# ===========================================================================
# TAB 5 — PRODUCT PRIORITIES
# ===========================================================================
with tabs[4]:
    st.subheader("Product priorities")
    st.caption("Transparent score = 40% reach + 30% severity + 20% trend + "
               "10% version-association (each min-max normalised). Higher = act sooner.")
    if priorities is None:
        missing(P_PRIORITIES, "python src/prioritization.py")
    else:
        pr = priorities.sort_values("priority_score", ascending=False)
        top = pr.iloc[0]
        st.success(f"**Investigate first:** {top['issue_topic']} "
                   f"(score {top['priority_score']}, {top['priority_level']})")
        level_color = {"P0 - Critical": "#E74C3C", "P1 - High": "#E67E22",
                       "P2 - Medium": "#27AE60", "P3 - Low": "#95A5A6"}
        for level in ["P0 - Critical", "P1 - High", "P2 - Medium", "P3 - Low"]:
            block = pr[pr["priority_level"] == level]
            if block.empty:
                continue
            st.markdown(
                f'<div style="margin:14px 0 6px;"><span style="background:{level_color[level]};'
                f'color:#fff;padding:5px 12px;border-radius:8px;font-weight:700;font-size:.9rem;">'
                f'{level}</span> <span style="color:#666;">· {len(block)} issue(s)</span></div>',
                unsafe_allow_html=True)
            for _, r in block.iterrows():
                with st.expander(f"{r['issue_topic']}  —  score {r['priority_score']}"):
                    a, b, c, d = st.columns(4)
                    a.metric("Reviews", int(r["review_count"]))
                    b.metric("Negative %", f"{r['negative_percentage']}%")
                    c.metric("Avg rating", r["average_rating"])
                    d.metric("Trend", r["recent_trend"])
                    st.write(f"**Version association:** {r['version_association']}")
                    st.write(f"**Evidence (keywords):** {r['top_keywords']}")
                    st.write(f"**Recommended action:** {r['recommended_action']}")

# ===========================================================================
# TAB 6 — REVIEW EXPLORER
# ===========================================================================
with tabs[5]:
    st.subheader("Review explorer")
    st.caption(f"{len(fdf):,} reviews match the current filters.")
    text_col = "content_raw" if "content_raw" in fdf else "content_clean"
    show = (fdf.sort_values("review_datetime", ascending=False)
            [["review_date", "score", "sentiment_label", "issue_topic",
              "reviewCreatedVersion", text_col]]
            .rename(columns={"reviewCreatedVersion": "version", text_col: "review"}))
    st.dataframe(show.head(500), use_container_width=True, hide_index=True)
    st.download_button("Download filtered reviews (CSV)",
                       show.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_reviews.csv", mime="text/csv")
