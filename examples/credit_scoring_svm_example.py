"""
Same credit workflow as ``credit_scoring_example.py``, but with a scikit-learn SVM classifier.

Uses ``StandardScaler`` + ``SVC(..., probability=True)`` in a ``Pipeline`` so scores come from
``predict_proba`` and line up with the SDK's default decision threshold (0.5).

Run from the ``examples/`` directory:

    cd examples && python credit_scoring_svm_example.py

Requires optional dependencies: ``pip install "darsha[ml]"`` (pandas, scikit-learn, shap).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import darsha


CSV_PATH = Path(__file__).resolve().parent / "credit.csv"

dm_svm = darsha.init(
    project="credit-scoring-svm",
    regulation="eu-ai-act",
    risk_level="high",
)


@dm_svm.track_data
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["annual_income"] = df["annual_income"].replace(0, pd.NA)
    df["credit_limit"] = df["credit_limit"].replace(0, pd.NA)
    df["debt_ratio"] = df["total_debt"] / df["annual_income"]
    df["credit_util"] = df["balance"] / df["credit_limit"]
    return df


@dm_svm.track_training
def train_svm(X_train, y_train) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


@dm_svm.explain(counterfactual=True)
def score_applicant(model: Pipeline, applicant_row: pd.DataFrame):
    return model.predict_proba(applicant_row)


def load_credit_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#")
    required = {"total_debt", "annual_income", "balance", "credit_limit", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"credit.csv is missing columns: {sorted(missing)}")
    return df


REPORT_DIR = Path(__file__).resolve().parent / "reports_svm"


if __name__ == "__main__":
    raw = load_credit_data(CSV_PATH)
    featured = prepare_features(raw)

    if "period" in featured.columns:
        train_df = featured[featured["period"] == "baseline"]
        test_df = featured[featured["period"] == "drift"]
        y_train = train_df["target"]
        y_test = test_df["target"]
        X_train = train_df.drop(columns=["target", "period"])
        X_test = test_df.drop(columns=["target", "period"])
    else:
        X = featured.drop(columns=["target"])
        y = featured["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

    model = train_svm(X_train, y_train)

    dm_svm.register_shap_feature_importance(
        model,
        X_train,
        max_samples=min(200, len(X_train)),
    )

    print("--- Regime summary (covariate + label shift) ---")
    print(f"  Train n={len(y_train)}, label positive rate = {y_train.mean():.2f}")
    print(f"  Test  n={len(y_test)}, label positive rate = {y_test.mean():.2f}")
    print(
        f"  Mean debt_ratio:  train {X_train['debt_ratio'].mean():.3f}  →  test {X_test['debt_ratio'].mean():.3f}"
    )
    print(
        f"  Mean credit_util: train {X_train['credit_util'].mean():.3f}  →  test {X_test['credit_util'].mean():.3f}"
    )
    print()

    threshold = 0.5
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= threshold).astype(int)
    income_band = pd.cut(
        X_test["annual_income"],
        bins=[-np.inf, 55000, 75000, np.inf],
        labels=["low", "mid", "high"],
    )
    baseline_positive_rate = float(y_train.mean())

    monitor_report = dm_svm.monitor(
        y_true=y_test.to_numpy(),
        y_pred=y_pred_test,
        expected_rate=baseline_positive_rate,
        protected_feature=income_band.astype(str).to_numpy(),
    )
    y_t = y_test.to_numpy()
    prec = float(precision_score(y_t, y_pred_test, zero_division=0))
    rec = float(recall_score(y_t, y_pred_test, zero_division=0))
    f1v = float(f1_score(y_t, y_pred_test, zero_division=0))
    try:
        auc = float(roc_auc_score(y_t, y_proba_test))
    except ValueError:
        auc = 0.0
    cm_test = confusion_matrix(y_t, y_pred_test)
    split_desc = (
        "Train on period=baseline, test on period=drift (hold-out)"
        if "period" in featured.columns
        else "70/30 stratified random split (sklearn train_test_split, random_state=42)"
    )
    dm_svm.register_governance(
        intended_use="Demo credit default risk triage; not for automated denial as the sole input.",
        known_limitations="Example dataset; SVM probabilities are Platt-scaled. Legal/fairness review required for production.",
        human_oversight="Underwriter reviews model-assisted decisions; overrides supported.",
        data_provenance="examples/credit.csv (synthetic/illustrative for SDK demo).",
        model_changelog=[{"version": "demo-svm-1.0", "date": "2026-04-23", "notes": "SVM + SHAP + audit template"}],
    )
    dm_svm.register_evaluation(
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        split_description=split_desc,
        test_metrics={
            "precision": prec,
            "recall": rec,
            "f1": f1v,
            "roc_auc": auc,
            "accuracy": float(monitor_report["accuracy"]),
        },
        confusion_matrix=cm_test.tolist(),
    )
    print("--- monitor() ---")
    for key, value in monitor_report.items():
        print(f"  {key}: {value}")

    applicant_row = X_test.iloc[[0]]
    result = score_applicant(model, applicant_row)

    print()
    print(result.decision.to_text())
    print(result.explanation.plain_language)
    print(result.audit_id)
    print()
    print("--- This decision (tabular) ---")
    print(result.as_markdown_table())
    print()
    print("--- Full audit trail (all steps, tabular) ---")
    print(dm_svm.format_audit_log_markdown())

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit_md = dm_svm.write_audit_report(
        REPORT_DIR / "audit_report.md", title="Credit scoring (SVM) — audit trail"
    )
    audit_html = dm_svm.write_audit_report(
        REPORT_DIR / "audit_report.html",
        title="Credit scoring (SVM) — audit trail",
        format="html",
    )
    decision_html = result.write_report(
        REPORT_DIR / "last_decision.html", title="Last decision (SVM)", format="html"
    )
    print()
    print("--- Downloadable reports ---")
    print(f"  {audit_md}")
    print(f"  {audit_html}")
    print(f"  {decision_html}")
