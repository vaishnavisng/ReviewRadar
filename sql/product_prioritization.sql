-- ReviewRadar — Product Prioritization SQL
-- Executed from Python (parsed by "-- name:" markers). Table: reviews
-- (loaded from reviews_with_topics.csv). issue_cluster=-1 => not clustered.

-- name: issue_frequency
SELECT issue_topic AS issue,
       COUNT(*)    AS review_count
FROM reviews
WHERE issue_cluster != -1
GROUP BY issue
ORDER BY review_count DESC;

-- name: negative_pct_by_issue
SELECT issue_topic AS issue,
       ROUND(100.0 * SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END)
             / COUNT(*), 1) AS negative_sentiment_pct
FROM reviews
WHERE issue_cluster != -1
GROUP BY issue;

-- name: avg_rating_by_issue
SELECT issue_topic         AS issue,
       ROUND(AVG(score), 2) AS avg_rating
FROM reviews
WHERE issue_cluster != -1
GROUP BY issue;

-- name: one_star_pct_by_issue
SELECT issue_topic AS issue,
       ROUND(100.0 * SUM(CASE WHEN score = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS one_star_pct
FROM reviews
WHERE issue_cluster != -1
GROUP BY issue;

-- name: issue_frequency_by_month
SELECT review_month AS month,
       issue_topic  AS issue,
       COUNT(*)     AS review_count
FROM reviews
WHERE issue_cluster != -1
GROUP BY month, issue
ORDER BY month;

-- name: issue_frequency_by_version
SELECT reviewCreatedVersion AS version,
       issue_topic          AS issue,
       COUNT(*)             AS review_count
FROM reviews
WHERE issue_cluster != -1
  AND reviewCreatedVersion != 'unknown'
GROUP BY version, issue;
