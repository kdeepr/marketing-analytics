"""
Feature engineering for the Bank Marketing campaign dataset.

Builds analysis-ready features used by both the EDA notebooks and the
modeling pipeline. Two feature sets are exposed on purpose:

- "benchmark" includes `duration` (last call length in seconds). The data
  dictionary explicitly warns this leaks the outcome (a call with
  duration=0 is always "no", and duration is only known *after* the call
  ends), so it is kept only to reproduce the original paper's benchmark
  numbers.
- "realistic" drops `duration` and any other post-outcome information,
  which is the feature set a marketing team could actually act on
  *before* placing a call.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_cleaning import PROCESSED_DIR, run as run_cleaning

MONTH_TO_QUARTER = {
    "jan": "Q1", "feb": "Q1", "mar": "Q1",
    "apr": "Q2", "may": "Q2", "jun": "Q2",
    "jul": "Q3", "aug": "Q3", "sep": "Q3",
    "oct": "Q4", "nov": "Q4", "dec": "Q4",
}

AGE_BINS = [17, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]

CAMPAIGN_BINS = [0, 1, 3, 6, 100]
CAMPAIGN_LABELS = ["1", "2-3", "4-6", "7+"]

CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan", "contact",
    "month", "day_of_week", "poutcome", "age_group", "campaign_bucket",
    "pdays_bucket", "quarter",
]
NUMERIC_REALISTIC = [
    "age", "campaign", "previous", "contacted_before", "prior_success",
    "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed",
]
NUMERIC_BENCHMARK = NUMERIC_REALISTIC + ["duration"]

TARGET = "y"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True)
    df["campaign_bucket"] = pd.cut(df["campaign"], bins=CAMPAIGN_BINS, labels=CAMPAIGN_LABELS)
    df["quarter"] = df["month"].map(MONTH_TO_QUARTER)

    df["contacted_before"] = (df["previous"] > 0).astype(int)
    df["prior_success"] = (df["poutcome"] == "success").astype(int)

    def pdays_bucket(v):
        if v == 999:
            return "never_contacted"
        if v <= 7:
            return "0-7_days"
        if v <= 14:
            return "8-14_days"
        if v <= 30:
            return "15-30_days"
        return "30+_days"

    df["pdays_bucket"] = df["pdays"].apply(pdays_bucket)

    return df


def get_feature_matrix(df: pd.DataFrame, benchmark: bool = False):
    numeric_cols = NUMERIC_BENCHMARK if benchmark else NUMERIC_REALISTIC
    X = df[CATEGORICAL_FEATURES + numeric_cols].copy()
    y = df[TARGET].copy()
    return X, y, CATEGORICAL_FEATURES, numeric_cols


def run(save: bool = True) -> pd.DataFrame:
    clean_df, _ = run_cleaning(save=False)
    feat_df = add_features(clean_df)
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        feat_df.to_csv(PROCESSED_DIR / "bank_marketing_features.csv", index=False)
    return feat_df


if __name__ == "__main__":
    feat_df = run()
    print(f"Feature-engineered dataset: {feat_df.shape[0]} rows x {feat_df.shape[1]} cols")
    print(feat_df[["age_group", "campaign_bucket", "pdays_bucket", "quarter"]].head())
