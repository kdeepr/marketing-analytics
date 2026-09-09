"""
Modeling pipeline for predicting term-deposit subscription (target `y`).

Trains and compares Logistic Regression, Random Forest, and Gradient
Boosting classifiers on two feature sets:

- "realistic": features known BEFORE a call is made (actionable for
  targeting future campaigns).
- "benchmark": adds `duration`, reproducing the literature's benchmark
  setup even though it cannot be used for prospective targeting.

Outputs (written under reports/):
- metrics.json               summary metrics for every model x feature-set
- figures/roc_curves_*.png
- figures/confusion_matrix_*.png
- figures/feature_importance_*.png
- models/best_model_realistic.joblib
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import get_feature_matrix, run as run_features

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"

RANDOM_STATE = 42


def build_pipeline(categorical_cols, numeric_cols, estimator):
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])


def get_estimators():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate(pipeline, X_test, y_test, name, feature_set):
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_title(f"ROC Curve - {name} ({feature_set})")
    fig.tight_layout()
    fig.savefig(FIGURES / f"roc_{feature_set}_{name}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {name} ({feature_set})")
    fig.tight_layout()
    fig.savefig(FIGURES / f"confusion_{feature_set}_{name}.png", dpi=150)
    plt.close(fig)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_positive": report["1"]["precision"],
        "recall_positive": report["1"]["recall"],
        "f1_positive": report["1"]["f1-score"],
        "accuracy": report["accuracy"],
    }


def plot_feature_importance(pipeline, categorical_cols, numeric_cols, name, feature_set, top_n=15):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_") and not hasattr(model, "coef_"):
        return

    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(categorical_cols))
    all_names = cat_names + numeric_cols

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([all_names[i] for i in order][::-1], importances[order][::-1], color="#2f6f4f")
    ax.set_title(f"Top {top_n} Feature Importances - {name} ({feature_set})")
    fig.tight_layout()
    fig.savefig(FIGURES / f"feature_importance_{feature_set}_{name}.png", dpi=150)
    plt.close(fig)


def run():
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    feat_df = run_features(save=True)
    results = {}
    best = {"roc_auc": -1}

    for feature_set, benchmark in [("realistic", False), ("benchmark", True)]:
        X, y, cat_cols, num_cols = get_feature_matrix(feat_df, benchmark=benchmark)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )

        for name, estimator in get_estimators().items():
            pipeline = build_pipeline(cat_cols, num_cols, estimator)
            pipeline.fit(X_train, y_train)
            metrics = evaluate(pipeline, X_test, y_test, name, feature_set)
            plot_feature_importance(pipeline, cat_cols, num_cols, name, feature_set)

            key = f"{feature_set}__{name}"
            results[key] = metrics
            print(f"{key}: ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
                  f"Recall(+)={metrics['recall_positive']:.3f}")

            if feature_set == "realistic" and metrics["roc_auc"] > best["roc_auc"]:
                best = {**metrics, "name": name, "pipeline": pipeline}

    joblib.dump(best["pipeline"], MODELS / "best_model_realistic.joblib")
    results["best_realistic_model"] = best["name"]

    with open(REPORTS / "metrics.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != "pipeline"}, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    run()
