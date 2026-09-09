-- Schema for the Bank Marketing analytics database.
-- Loaded from data/processed/bank_marketing_features.csv (see sql/load_db.py).
-- A single denormalized fact table mirrors the source dataset; the star
-- schema used by Power BI (powerbi/) is derived from this table.

DROP TABLE IF EXISTS campaign_contacts;

CREATE TABLE campaign_contacts (
    contact_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    age              INTEGER NOT NULL,
    age_group        TEXT NOT NULL,
    job              TEXT NOT NULL,
    marital          TEXT NOT NULL,
    education        TEXT NOT NULL,
    "default"        TEXT NOT NULL,
    housing          TEXT NOT NULL,
    loan             TEXT NOT NULL,
    contact          TEXT NOT NULL,       -- contact channel: cellular / telephone
    month            TEXT NOT NULL,
    quarter          TEXT NOT NULL,
    day_of_week      TEXT NOT NULL,
    duration         INTEGER NOT NULL,    -- seconds; benchmark-only, see README
    campaign         INTEGER NOT NULL,    -- # contacts this campaign
    campaign_bucket  TEXT NOT NULL,
    pdays            INTEGER NOT NULL,    -- 999 = never contacted before
    pdays_bucket     TEXT NOT NULL,
    previous         INTEGER NOT NULL,
    contacted_before INTEGER NOT NULL,    -- 0/1
    poutcome         TEXT NOT NULL,
    prior_success    INTEGER NOT NULL,    -- 0/1
    emp_var_rate     REAL NOT NULL,
    cons_price_idx   REAL NOT NULL,
    cons_conf_idx    REAL NOT NULL,
    euribor3m        REAL NOT NULL,
    nr_employed      REAL NOT NULL,
    y                INTEGER NOT NULL     -- target: subscribed (1) / not (0)
);

CREATE INDEX idx_contacts_job ON campaign_contacts(job);
CREATE INDEX idx_contacts_month ON campaign_contacts(month);
CREATE INDEX idx_contacts_poutcome ON campaign_contacts(poutcome);
