-- Campaign performance analysis queries.
-- Run after sql/load_db.py has built data/processed/marketing_analytics.db:
--   sqlite3 -header -column data/processed/marketing_analytics.db < sql/campaign_analysis.sql

-- 1. Overall campaign conversion rate
SELECT
    COUNT(*)                                   AS total_contacts,
    SUM(y)                                      AS total_subscriptions,
    ROUND(100.0 * SUM(y) / COUNT(*), 2)         AS conversion_rate_pct
FROM campaign_contacts;

-- 2. Conversion rate by job (min 200 contacts to avoid noisy small segments)
SELECT
    job,
    COUNT(*)                                   AS contacts,
    SUM(y)                                      AS subscriptions,
    ROUND(100.0 * SUM(y) / COUNT(*), 2)         AS conversion_rate_pct
FROM campaign_contacts
GROUP BY job
HAVING COUNT(*) >= 200
ORDER BY conversion_rate_pct DESC;

-- 3. Conversion rate by age group
SELECT
    age_group,
    COUNT(*)                                   AS contacts,
    ROUND(100.0 * SUM(y) / COUNT(*), 2)         AS conversion_rate_pct
FROM campaign_contacts
GROUP BY age_group
ORDER BY age_group;

-- 4. Seasonality: conversion rate by month (chronological, not alphabetical)
SELECT
    month,
    quarter,
    COUNT(*)                                   AS contacts,
    ROUND(100.0 * SUM(y) / COUNT(*), 2)         AS conversion_rate_pct
FROM campaign_contacts
GROUP BY month, quarter
ORDER BY CASE month
    WHEN 'jan' THEN 1 WHEN 'feb' THEN 2 WHEN 'mar' THEN 3 WHEN 'apr' THEN 4
    WHEN 'may' THEN 5 WHEN 'jun' THEN 6 WHEN 'jul' THEN 7 WHEN 'aug' THEN 8
    WHEN 'sep' THEN 9 WHEN 'oct' THEN 10 WHEN 'nov' THEN 11 WHEN 'dec' THEN 12
END;

-- 5. Diminishing returns: conversion rate vs number of contacts this campaign
SELECT
    campaign_bucket,
    COUNT(*)                                   AS contacts,
    ROUND(100.0 * SUM(y) / COUNT(*), 2)         AS conversion_rate_pct
FROM campaign_contacts
GROUP BY campaign_bucket
ORDER BY CASE campaign_bucket
    WHEN '1' THEN 1 WHEN '2-3' THEN 2 WHEN '4-6' THEN 3 WHEN '7+' THEN 4 END;

-- 6. Contact channel effectiveness
SELECT
    contact AS contact_channel,
    COUNT(*)                                   AS contacts,
    ROUND(100.0 * SUM(y) / COUNT(*), 2)         AS conversion_rate_pct
FROM campaign_contacts
GROUP BY contact
ORDER BY conversion_rate_pct DESC;

-- 7. The single strongest signal: prior campaign outcome
SELECT
    poutcome,
    COUNT(*)                                   AS contacts,
    ROUND(100.0 * SUM(y) / COUNT(*), 2)         AS conversion_rate_pct
FROM campaign_contacts
GROUP BY poutcome
ORDER BY conversion_rate_pct DESC;

-- 8. Recency since last contact (pdays_bucket) vs conversion
SELECT
    pdays_bucket,
    COUNT(*)                                   AS contacts,
    ROUND(100.0 * SUM(y) / COUNT(*), 2)         AS conversion_rate_pct
FROM campaign_contacts
GROUP BY pdays_bucket
ORDER BY conversion_rate_pct DESC;

-- 9. Best-converting segments (job x education), minimum sample size 100,
--    for prioritizing the next campaign's target list
SELECT
    job,
    education,
    COUNT(*)                                   AS contacts,
    SUM(y)                                      AS subscriptions,
    ROUND(100.0 * SUM(y) / COUNT(*), 2)         AS conversion_rate_pct
FROM campaign_contacts
GROUP BY job, education
HAVING COUNT(*) >= 100
ORDER BY conversion_rate_pct DESC
LIMIT 15;

-- 10. Macro-economic context: average indicators for converters vs non-converters
SELECT
    CASE y WHEN 1 THEN 'subscribed' ELSE 'not_subscribed' END AS outcome,
    ROUND(AVG(emp_var_rate), 3)     AS avg_emp_var_rate,
    ROUND(AVG(cons_price_idx), 3)   AS avg_cons_price_idx,
    ROUND(AVG(cons_conf_idx), 3)    AS avg_cons_conf_idx,
    ROUND(AVG(euribor3m), 3)        AS avg_euribor3m,
    ROUND(AVG(nr_employed), 1)      AS avg_nr_employed
FROM campaign_contacts
GROUP BY y;

-- 11. Untapped-opportunity segments: high conversion rate but low contact
--     volume relative to their share of the client base (candidates to
--     contact MORE in the next campaign)
SELECT
    job,
    COUNT(*)                                             AS contacts,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM campaign_contacts), 2) AS pct_of_all_contacts,
    ROUND(100.0 * SUM(y) / COUNT(*), 2)                   AS conversion_rate_pct
FROM campaign_contacts
GROUP BY job
HAVING COUNT(*) >= 200
ORDER BY conversion_rate_pct DESC
LIMIT 5;
