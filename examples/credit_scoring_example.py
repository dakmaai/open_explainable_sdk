"""
End-to-end sample: load `credit.csv`, engineer features, train XGBoost, explain one prediction.

Run from the `examples/` directory (or anywhere if you pass the CSV path):

    cd examples && python credit_scoring_example.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import dakma


# Default CSV next to this script (same folder as credit.csv)
CSV_PATH = Path(__file__).resolve().parent / "credit.csv"

dm_c = dakma.init(
    project="credit-scoring-package",
    regulation="eu-ai-act",
    risk_level="high",
)


@dm_c.track_data
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Avoid division by zero on bad data
    df["annual_income"] = df["annual_income"].replace(0, pd.NA)
    df["credit_limit"] = df["credit_limit"].replace(0, pd.NA)
    df["debt_ratio"] = df["total_debt"] / df["annual_income"]
    df["credit_util"] = df["balance"] / df["credit_limit"]
    return df


@dm_c.track_training
def train(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


@dm_c.explain(counterfactual=True)
def score_applicant(model, applicant_row):
    return model.predict_proba(applicant_row)


def load_credit_data(csv_path: Path) -> pd.DataFrame:
    """Load credit.csv with required columns: total_debt, annual_income, balance, credit_limit, target.

    Optional column ``period`` (e.g. ``baseline`` vs ``drift``) enables a train/eval split that shows
    covariate and label shift in ``monitor()`` (see README).
    Lines starting with ``#`` are treated as comments.
    """
    df = pd.read_csv(csv_path, comment="#")
    required = {"total_debt", "annual_income", "balance", "credit_limit", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"credit.csv is missing columns: {sorted(missing)}")
    return df


REPORT_DIR = Path(__file__).resolve().parent / "reports"


if __name__ == "__main__":
    raw = load_credit_data(CSV_PATH)

    featured = prepare_features(raw)

    # Train on ``baseline`` rows, evaluate on ``drift`` rows so the dataset exhibits clear data / label shift.
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

    model = train(X_train, y_train)

    # Global SHAP-based feature importance (shown on decision / audit reports for this model).
    dm_c.register_shap_feature_importance(model, X_train, max_samples=min(300, len(X_train)))

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

    # --- Monitoring (test set): accuracy, positive rate, drift vs baseline, optional group rates ---
    # Same threshold as @dm_c.explain (default 0.5) for comparable "positive class" rate.
    threshold = 0.5
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= threshold).astype(int)

    # Optional: proxy for a "sensitive" attribute — here, income band from test rows only.
    # Replace with a real column (e.g. region) if present in your data.
    income_band = pd.cut(
        X_test["annual_income"],
        bins=[-np.inf, 55000, 75000, np.inf],
        labels=["low", "mid", "high"],
    )

    baseline_positive_rate = float(y_train.mean())  # training-set label prevalence as a simple baseline

    monitor_report = dm_c.monitor(
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
    dm_c.register_governance(
        intended_use="Demo credit default risk triage; not for automated denial as the sole input.",
        known_limitations="Example dataset; real deployment needs calibration, monitoring, and legal review. Features are correlational.",
        human_oversight="Underwriter reviews model-assisted decisions; overrides supported.",
        data_provenance="examples/credit.csv (synthetic/illustrative for SDK demo).",
        model_changelog=[
            {"version": "demo-1.0", "date": "2026-04-23", "notes": "XGBoost + SHAP + audit report template"},
        ],
    )
    dm_c.register_evaluation(
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

    print(result.decision_output.to_text())
    print(result.explain.plain_language)
    print(result.explain.audit_trail_id)
    print()
    print("--- This decision (tabular) ---")
    print(result.as_markdown_table())
    print()
    print("--- Full audit trail (all steps, tabular) ---")
    print(dm_c.format_audit_log_markdown())

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit_md = dm_c.write_audit_report(REPORT_DIR / "audit_report.md", title="Credit scoring — audit trail")
    audit_html = dm_c.write_audit_report(
        REPORT_DIR / "audit_report.html", title="Credit scoring — audit trail", format="html"
    )
    decision_html = result.write_report(REPORT_DIR / "last_decision.html", title="Last decision", format="html")
    print()
    print("--- Downloadable reports written ---")
    print(f"  {audit_md}")
    print(f"  {audit_html}")
    print(f"  {decision_html}")
