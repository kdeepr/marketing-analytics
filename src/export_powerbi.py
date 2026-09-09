"""
Builds a small star schema (fact + date dimension) for Power BI, and scores
every contact with the saved "realistic" model to add a predicted-propensity
field for a "next best targets" visual.

IMPORTANT - synthetic dates: the source dataset gives `month` and
`day_of_week` for each contact but not the day or year, only that rows are
ordered chronologically across the campaign window May 2008 - Nov 2010
(per the UCI data dictionary). `contact_date` below is RECONSTRUCTED to be
consistent with that ordering, month, and weekday, purely so Power BI has a
real Date dimension to demonstrate time-intelligence DAX (WTD/MTD/QTD,
trailing trends, etc.). Treat it as illustrative, not the literal calendar
date of each historical call. See powerbi/README.md for details.

Outputs (under powerbi/data/):
- dim_date.csv
- fact_campaign_contacts.csv
- segment_summary.csv   (pre-aggregated, for fast-loading Power BI visuals)
"""
import datetime as dt
from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering import get_feature_matrix, run as run_features

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "powerbi" / "data"
MODEL_PATH = ROOT / "models" / "best_model_realistic.joblib"

MONTH_ABBR = {m.lower(): i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1
)}
DOW_ABBR = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def build_synthetic_dates(df: pd.DataFrame) -> pd.Series:
    current = dt.date(2008, 5, 1)
    dates = []
    for month, dow in zip(df["month"], df["day_of_week"]):
        target_month, target_dow = MONTH_ABBR[month], DOW_ABBR[dow]
        while not (current.month == target_month and current.weekday() == target_dow):
            current += dt.timedelta(days=1)
        dates.append(current)
    return pd.Series(dates, index=df.index)


def build_dim_date(dates: pd.Series) -> pd.DataFrame:
    unique_dates = sorted(dates.unique())
    dim = pd.DataFrame({"date": unique_dates})
    dim["date_key"] = dim["date"].apply(lambda d: int(d.strftime("%Y%m%d")))
    dim["year"] = dim["date"].apply(lambda d: d.year)
    dim["month_num"] = dim["date"].apply(lambda d: d.month)
    dim["month_name"] = dim["date"].apply(lambda d: d.strftime("%B"))
    dim["quarter"] = dim["date"].apply(lambda d: f"Q{((d.month - 1) // 3) + 1}")
    dim["day_of_week"] = dim["date"].apply(lambda d: d.strftime("%A"))
    return dim[["date_key", "date", "year", "month_num", "month_name", "quarter", "day_of_week"]]


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat_df = run_features(save=False)

    dates = build_synthetic_dates(feat_df)
    dim_date = build_dim_date(dates)
    dim_date.to_csv(OUT_DIR / "dim_date.csv", index=False)

    model = joblib.load(MODEL_PATH)
    X, y, _, _ = get_feature_matrix(feat_df, benchmark=False)
    propensity = model.predict_proba(X)[:, 1]

    fact = feat_df.copy()
    fact.insert(0, "contact_id", range(1, len(fact) + 1))
    fact["date_key"] = dates.apply(lambda d: int(d.strftime("%Y%m%d"))).values
    fact["predicted_propensity"] = propensity
    fact["predicted_label"] = (propensity >= 0.5).astype(int)
    fact["subscribed"] = fact["y"]

    export_cols = [
        "contact_id", "date_key", "age", "age_group", "job", "marital", "education",
        "default", "housing", "loan", "contact", "month", "quarter", "day_of_week",
        "duration", "campaign", "campaign_bucket", "pdays", "pdays_bucket", "previous",
        "contacted_before", "poutcome", "prior_success", "emp_var_rate", "cons_price_idx",
        "cons_conf_idx", "euribor3m", "nr_employed", "subscribed", "predicted_propensity",
        "predicted_label",
    ]
    fact[export_cols].to_csv(OUT_DIR / "fact_campaign_contacts.csv", index=False)

    segment_summary = (
        fact.groupby(["job", "age_group", "education", "month", "quarter"], observed=True)
        .agg(contacts=("contact_id", "count"), subscriptions=("subscribed", "sum"),
             avg_predicted_propensity=("predicted_propensity", "mean"))
        .reset_index()
    )
    segment_summary["conversion_rate"] = segment_summary["subscriptions"] / segment_summary["contacts"]
    segment_summary.to_csv(OUT_DIR / "segment_summary.csv", index=False)

    print(f"dim_date: {dim_date.shape}, fact_campaign_contacts: {fact.shape}, "
          f"segment_summary: {segment_summary.shape}")


if __name__ == "__main__":
    run()
