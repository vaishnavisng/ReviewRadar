-- ReviewRadar — Version & Time-Series SQL
-- Executed from Python (parsed by the "-- name:" markers). Table: reviews
-- (loaded from data/processed/reviews_with_topics.csv into SQLite).
-- 'unknown' = review had no app version recorded.

-- name: avg_rating_by_version
SELECT reviewCreatedVersion AS version,
       COUNT(*)             AS review_count,
       ROUND(AVG(score), 2) AS avg_rating
FROM reviews
WHERE reviewCreatedVersion != 'unknown'
GROUP BY version;

-- name: review_count_by_version
SELECT reviewCreatedVersion AS version,
       COUNT(*)             AS review_count
FROM reviews
WHERE reviewCreatedVersion != 'unknown'
GROUP BY version
ORDER BY review_count DESC;

-- name: negative_pct_by_version
SELECT reviewCreatedVersion AS version,
       COUNT(*) AS review_count,
       ROUND(100.0 * SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END)
             / COUNT(*), 1) AS negative_pct
FROM reviews
WHERE reviewCreatedVersion != 'unknown'
GROUP BY version;

-- name: review_volume_by_month
SELECT review_month AS month,
       COUNT(*)     AS review_count
FROM reviews
GROUP BY month
ORDER BY month;

-- name: avg_rating_by_month
SELECT review_month        AS month,
       ROUND(AVG(score), 2) AS avg_rating
FROM reviews
GROUP BY month
ORDER BY month;

-- name: issue_freq_by_version
SELECT reviewCreatedVersion AS version,
       issue_topic          AS issue,
       COUNT(*)             AS review_count
FROM reviews
WHERE issue_cluster != -1
  AND reviewCreatedVersion != 'unknown'
GROUP BY version, issue;
