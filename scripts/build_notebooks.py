"""
Programmatically builds the project's Jupyter notebooks with nbformat so
the .ipynb JSON is always valid, then executes them in place with
nbconvert so the committed notebooks contain real output (tables, charts).

Run with: python scripts/build_notebooks.py
"""
import subprocess
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def build_eda_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""# 01 - Exploratory Data Analysis
Bank Marketing campaign dataset (UCI / Kaggle). See `README.md` for the full
data source citation and project overview.

This notebook explores data quality, the target class balance, and how
subscription rate (`y`) varies across client attributes and campaign
mechanics - the questions a marketing analytics stakeholder actually asks
before any modeling happens."""),
        code("""import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_cleaning import run as run_cleaning

sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', 50)

df, quality_report = run_cleaning(save=True)
quality_report"""),
        md("## Shape, dtypes, and missingness (`unknown` placeholders)"),
        code("""print(f"Rows: {df.shape[0]:,}  Columns: {df.shape[1]}")
df.info()"""),
        code("""df.describe(include='all').T"""),
        md("""## Target imbalance
Only ~11% of clients subscribed to a term deposit. This imbalance drives
every modeling choice downstream (class weighting, PR-AUC over accuracy,
threshold selection)."""),
        code("""rate = df['y'].mean()
ax = df['y'].value_counts(normalize=True).sort_index().plot(
    kind='bar', color=['#b0b7c3', '#2f6f4f'], figsize=(4,4))
ax.set_xticklabels(['No (0)', 'Yes (1)'], rotation=0)
ax.set_ylabel('Share of clients')
ax.set_title(f'Target balance - overall subscription rate = {rate:.1%}')
plt.tight_layout()
plt.savefig('../reports/figures/target_balance.png', dpi=150)
plt.show()"""),
        md("## Subscription rate by demographic and campaign attributes"),
        code("""def conversion_by(col, top_n=None, figsize=(7,4)):
    g = df.groupby(col)['y'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    if top_n:
        g = g.head(top_n)
    ax = g['mean'].plot(kind='bar', color='#2f6f4f', figsize=figsize)
    ax.set_ylabel('Subscription rate')
    ax.set_title(f'Subscription rate by {col}')
    plt.tight_layout()
    plt.savefig(f'../reports/figures/conversion_by_{col}.png', dpi=150)
    plt.show()
    return g

conversion_by('job')"""),
        code("conversion_by('education')"),
        code("conversion_by('marital')"),
        code("conversion_by('month')"),
        code("conversion_by('poutcome')"),
        code("conversion_by('contact')"),
        md("""## Numeric feature distributions and correlation
`emp_var_rate`, `euribor3m`, and `nr_employed` are macro-economic indicators
published quarterly/monthly by Banco de Portugal - they move together, so we
expect multicollinearity among them (relevant for the linear model)."""),
        code("""numeric_cols = ['age','campaign','previous','emp_var_rate','cons_price_idx',
                'cons_conf_idx','euribor3m','nr_employed','duration']
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(df[numeric_cols + ['y']].corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Correlation matrix - numeric features vs target')
plt.tight_layout()
plt.savefig('../reports/figures/correlation_heatmap.png', dpi=150)
plt.show()"""),
        code("""fig, axes = plt.subplots(1, 2, figsize=(11,4))
sns.boxplot(data=df, x='y', y='age', ax=axes[0])
axes[0].set_xticklabels(['No', 'Yes'])
axes[0].set_title('Age vs subscription')
sns.boxplot(data=df, x='y', y='campaign', ax=axes[1])
axes[1].set_xticklabels(['No', 'Yes'])
axes[1].set_ylim(0, 20)
axes[1].set_title('# contacts this campaign vs subscription')
plt.tight_layout()
plt.savefig('../reports/figures/age_campaign_boxplots.png', dpi=150)
plt.show()"""),
        md("""## Key takeaways
- Severe class imbalance (~11% positive) - accuracy alone is a misleading
  metric; ROC-AUC / PR-AUC and recall on the positive class matter more.
- Students and retirees convert far above average, despite being contacted
  less often - a classic under-targeted segment.
- Clients with a successful previous campaign (`poutcome=success`) convert
  at a dramatically higher rate - prior engagement is the single strongest
  behavioral signal.
- Conversion is highly seasonal by month, tracking the bank's own campaign
  calendar and the macro-economic cycle (`emp_var_rate`, `euribor3m`).
- `duration` correlates strongly with `y` by construction (the call has to
  happen for a "yes"), confirming the data dictionary's leakage warning -
  it must be excluded from any prospective targeting model.

See `notebooks/02_feature_engineering.ipynb` and
`notebooks/03_modeling.ipynb` for how these findings shape the feature set
and model design."""),
    ]
    return nb


def build_feature_engineering_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""# 02 - Feature Engineering
Builds the analysis-ready feature set from the cleaned data: bucketed
numeric fields, a quarter-of-year seasonal flag, and behavioral flags
derived from prior-campaign history. Logic lives in
`src/feature_engineering.py` so the EDA, SQL, R, and modeling assets all
consume an identical, versioned definition."""),
        code("""import sys
sys.path.append('..')
import pandas as pd

from src.feature_engineering import run as run_features, CATEGORICAL_FEATURES, NUMERIC_REALISTIC, NUMERIC_BENCHMARK

feat_df = run_features(save=True)
print(feat_df.shape)
feat_df.head()"""),
        md("## Engineered columns"),
        code("""feat_df[['age', 'age_group', 'campaign', 'campaign_bucket',
         'pdays', 'pdays_bucket', 'month', 'quarter',
         'previous', 'contacted_before', 'poutcome', 'prior_success']].sample(10, random_state=1)"""),
        md("""## Why these features
- **age_group**: raw age is noisy for a marketing segment; 6 life-stage
  bins map directly onto campaign personas.
- **campaign_bucket**: the number of contacts this campaign has strong
  diminishing (and eventually negative) returns - bucketing exposes that
  non-linearity to linear models and to a Power BI slicer.
- **pdays_bucket**: `pdays=999` is a sentinel meaning "never contacted",
  not a numeric distance of 999 days - leaving it untouched would badly
  mislead any model or chart that treats it as a magnitude.
- **quarter**: coarser seasonality than month, useful for a compact
  dashboard filter and for R's ANOVA test in `r/statistical_analysis.R`.
- **contacted_before / prior_success**: collapse `previous` and
  `poutcome` into the two behavioral signals that actually matter for
  targeting - has this client ever responded before?"""),
        md("## Feature sets used downstream"),
        code("""print('Categorical features:', CATEGORICAL_FEATURES)
print()
print('Numeric (realistic - no duration):', NUMERIC_REALISTIC)
print()
print('Numeric (benchmark - includes duration):', NUMERIC_BENCHMARK)"""),
        code("""feat_df.to_csv('../data/processed/bank_marketing_features.csv', index=False)
print('Saved data/processed/bank_marketing_features.csv:', feat_df.shape)"""),
    ]
    return nb


def build_modeling_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""# 03 - Modeling: Predicting Term Deposit Subscription
Trains Logistic Regression, Random Forest, and Gradient Boosting on two
feature sets (`realistic` vs `benchmark`, see `src/modeling.py`) and
compares them on ROC-AUC, PR-AUC, and recall on the positive class - the
metric that matters when the business cost of a missed subscriber is much
higher than the cost of an extra call."""),
        code("""import sys
sys.path.append('..')
import json
import pandas as pd
from IPython.display import Image, display

from src.modeling import run as run_modeling

results = run_modeling()
pd.DataFrame(results).drop(index=[], errors='ignore')"""),
        code("""metrics_df = pd.DataFrame({k: v for k, v in results.items() if isinstance(v, dict)}).T
metrics_df.sort_values('roc_auc', ascending=False)"""),
        md("""## Reading the comparison
- The **benchmark** feature set (includes `duration`) reproduces the
  ~0.94-0.95 ROC-AUC reported in Moro et al. (2014) - useful as a sanity
  check that the pipeline is correct, but not deployable (duration is only
  known *after* the call).
- The **realistic** feature set (~0.80-0.81 ROC-AUC) is what a marketing
  team could actually use to prioritize *before* dialing - this is the
  model saved to `models/best_model_realistic.joblib` and the one whose
  scores back the Power BI "predicted propensity" fields."""),
        md("## Best realistic model: ROC curve, confusion matrix, feature importance"),
        code("""best_name = results['best_realistic_model']
print('Best realistic-feature-set model:', best_name)
display(Image(filename=f'../reports/figures/roc_realistic_{best_name}.png'))
display(Image(filename=f'../reports/figures/confusion_realistic_{best_name}.png'))
display(Image(filename=f'../reports/figures/feature_importance_realistic_{best_name}.png'))"""),
        md("""## Key takeaways
- Prior campaign outcome, macro-economic context (`euribor3m`,
  `emp_var_rate`, `nr_employed`), and contact month dominate feature
  importance - reinforcing the EDA finding that *when* and *who you've
  already reached* matter more than static demographics.
- Recall on the positive class (not accuracy) is the metric to optimize
  given the ~11% base rate and the asymmetric cost of a missed subscriber
  vs. a wasted call.
- The saved model (`models/best_model_realistic.joblib`) can score new,
  uncalled leads to produce a propensity rank for the marketing team -
  the same scores feed the "Predicted Propensity" table exported for
  Power BI in `powerbi/`."""),
    ]
    return nb


def execute(nb_path: Path):
    subprocess.run(
        [
            "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=600", str(nb_path),
        ],
        check=True,
        cwd=ROOT,
    )


def main():
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    builders = {
        "01_eda.ipynb": build_eda_notebook,
        "02_feature_engineering.ipynb": build_feature_engineering_notebook,
        "03_modeling.ipynb": build_modeling_notebook,
    }
    for filename, builder in builders.items():
        path = NOTEBOOKS / filename
        nbf.write(builder(), path)
        print(f"Built {path}, executing...")
        execute(path)
        print(f"Executed {path}")


if __name__ == "__main__":
    main()
