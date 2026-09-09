"""
Data cleaning utilities for the Bank Marketing campaign dataset.

Source: UCI Machine Learning Repository - Bank Marketing (with social/economic
context), Moro, S., Cortez, P., & Rita, P. (2014). Also mirrored on Kaggle as
"Bank Marketing Dataset". See README.md for full citation.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "bank_marketing_raw.csv"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

CATEGORICAL_UNKNOWNS = [
    "job", "marital", "education", "default", "housing", "loan", "poutcome",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw semicolon-delimited UCI bank-marketing CSV."""
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip().lower().replace(".", "_") for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply core cleaning: dedupe, standardize dtypes, normalize placeholders.

    The source dataset encodes missing categorical values as the literal
    string "unknown" rather than NaN. We keep "unknown" as its own category
    (it is informative - e.g. undisclosed job/education is not random) but
    expose a missing-value count so this decision is auditable.
    """
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)

    df["y"] = df["y"].map({"yes": 1, "no": 0}).astype(int)

    # pdays == 999 is a sentinel for "not previously contacted", not a real
    # numeric distance. Keep the raw column but also expose a boolean flag
    # (see feature_engineering.py) so models don't treat 999 as a magnitude.
    df["pdays"] = df["pdays"].astype(int)

    for col in CATEGORICAL_UNKNOWNS:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df["duration"] = df["duration"].astype(int)
    df["campaign"] = df["campaign"].astype(int)
    df["previous"] = df["previous"].astype(int)
    df["age"] = df["age"].astype(int)

    numeric_cols = ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    quality_report = {
        "rows_in": before,
        "duplicate_rows_dropped": dropped,
        "rows_out": len(df),
        "unknown_value_counts": {
            col: int((df[col] == "unknown").sum()) for col in CATEGORICAL_UNKNOWNS
        },
        "target_positive_rate": float(df["y"].mean()),
    }
    return df, quality_report


def run(save: bool = True):
    df = load_raw()
    clean_df, report = clean(df)
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(PROCESSED_DIR / "bank_marketing_clean.csv", index=False)
    return clean_df, report


if __name__ == "__main__":
    import json

    clean_df, report = run()
    print(json.dumps(report, indent=2))
