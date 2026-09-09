"""
Loads data/processed/bank_marketing_features.csv into a local SQLite
database (data/processed/marketing_analytics.db) using the schema in
schema.sql, so sql/campaign_analysis.sql can be run with any SQLite client:

    python sql/load_db.py
    sqlite3 data/processed/marketing_analytics.db < sql/campaign_analysis.sql
"""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "marketing_analytics.db"
CSV_PATH = ROOT / "data" / "processed" / "bank_marketing_features.csv"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

COLUMNS = [
    "age", "age_group", "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "quarter", "day_of_week", "duration",
    "campaign", "campaign_bucket", "pdays", "pdays_bucket", "previous",
    "contacted_before", "poutcome", "prior_success", "emp_var_rate",
    "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed", "y",
]


def main():
    df = pd.read_csv(CSV_PATH)[COLUMNS]

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    df.to_sql("campaign_contacts", conn, if_exists="append", index=False)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM campaign_contacts").fetchone()[0]
    print(f"Loaded {count:,} rows into {DB_PATH}")
    conn.close()


if __name__ == "__main__":
    main()
