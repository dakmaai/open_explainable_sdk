"""
Multi-class sample: sklearn ``load_iris`` (3 species), RandomForest, ``target_class`` demo.

dakma explains the *predicted* class by default. ``target_class`` asks for the explanation of a
different class ("why not class 1?") without changing the decision. This script takes a row the
model predicts as class 0 and explains it twice: by default (class 0) and with ``target_class=1``.

Run from the `examples/` directory:

    cd examples && python iris_multiclass_example.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import shap
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

import dakma


dm_i = dakma.init(
    project="iris-multiclass",
    regulation="internal-policy",
    risk_level="low",
)


@dm_i.track_training
def train(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
    model.fit(X_train, y_train)
    return model


@dm_i.explain(counterfactual=True)
def classify_default(model, flower_row):
    """Explains the predicted class."""
    return model.predict_proba(flower_row)


@dm_i.explain(counterfactual=True, target_class=1)
def classify_target_class_1(model, flower_row):
    """Same prediction, but the explanation is for class 1."""
    return model.predict_proba(flower_row)


@dm_i.explain(target_class=7)
def classify_bad_target(model, flower_row):
    return model.predict_proba(flower_row)


def reference_shap_top3(model, row, class_index: int):
    """Top-3 SHAP factors for one class, computed directly (independent of dakma)."""
    values = np.asarray(shap.TreeExplainer(model)(row).values)  # (1, n_features, n_classes)
    per_feature = values[0, :, class_index]
    order = np.argsort(-np.abs(per_feature))[:3]
    return [(row.columns[i], float(per_feature[i])) for i in order]


def show(title: str, result, model, row, class_index: int) -> None:
    print(f"=== {title} ===")
    print(f"  Decision                 : {result.decision.to_text()}")
    print(f"  Plain language           : {result.explanation.plain_language}")
    print(f"  Counterfactual           : {result.explanation.counterfactual}")
    print(f"  Explained class (index)  : {result.explanation.metadata.get('explained_class_index')}")
    print("  dakma top factors        :")
    for f in result.explanation.factors:
        print(f"      {f.name:<22} value={f.value:<8.3f} impact={f.impact:+.4f} ({f.direction})")
    print(f"  SHAP top factors (class {class_index}, computed directly):")
    for name, impact in reference_shap_top3(model, row, class_index):
        print(f"      {name:<22} value={float(row.iloc[0][name]):<8.3f} impact={impact:+.4f}")
    print()


REPORT_DIR = Path(__file__).resolve().parent / "reports_iris"


if __name__ == "__main__":
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = train(X_train, y_train)
    dm_i.register_shap_feature_importance(model, X_train)

    # Hold-out evaluation (multi-class: macro-averaged) so reports have no "not provided" gaps.
    y_pred_test = model.predict(X_test)
    y_t = y_test.to_numpy()
    dm_i.register_governance(
        intended_use="Demo multi-class classification of iris species; shows explaining a non-predicted class.",
        known_limitations="Tiny public dataset (150 rows); for SDK testing only.",
        human_oversight="Not used for decisions; explainability test harness.",
        data_provenance="sklearn.datasets.load_iris (Fisher's Iris dataset).",
        model_changelog=[
            {"version": "demo-1.1.0", "date": "2026-11-04", "notes": "RandomForest + target_class demo"},
        ],
    )
    dm_i.register_evaluation(
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        split_description="70/30 stratified random split (sklearn train_test_split, random_state=42)",
        test_metrics={
            "precision_macro": float(precision_score(y_t, y_pred_test, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_t, y_pred_test, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_t, y_pred_test, average="macro", zero_division=0)),
            "roc_auc_ovr_macro": float(
                roc_auc_score(y_t, model.predict_proba(X_test), multi_class="ovr", average="macro")
            ),
            "accuracy": float(accuracy_score(y_t, y_pred_test)),
        },
        confusion_matrix=confusion_matrix(y_t, y_pred_test).tolist(),
    )

    # First test row the model predicts as class 0, so target_class=1 differs from the prediction.
    predicted = model.predict(X_test)
    row = X_test[predicted == 0].iloc[[0]]
    proba = model.predict_proba(row)[0]
    print(f"Row index {row.index[0]}: predict_proba = {np.round(proba, 3).tolist()} -> predicted class {int(np.argmax(proba))}")
    print()

    default_result = classify_default(model, row)
    target_result = classify_target_class_1(model, row)

    show("Default: explains the predicted class (0)", default_result, model, row, 0)
    show("target_class=1: same prediction, explanation is for class 1", target_result, model, row, 1)

    print("=== target_class=7 (out of range) ===")
    try:
        classify_bad_target(model, row)
    except ValueError as exc:
        print(f"  ValueError: {exc}")
    print()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        default_result.write_report(
            REPORT_DIR / "decision_default.html", title="Iris decision — predicted class 0", format="html"
        ),
        target_result.write_report(
            REPORT_DIR / "decision_target_class_1.html",
            title="Iris decision — predicted class 0, explained class 1",
            format="html",
        ),
        dm_i.write_audit_report(REPORT_DIR / "audit_report.html", title="Iris multi-class — audit trail", format="html"),
    ]
    print("--- Downloadable reports written ---")
    for path in paths:
        print(f"  {path}")
