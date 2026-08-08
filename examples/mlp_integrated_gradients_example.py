"""
Train a small PyTorch MLP on sklearn's breast cancer dataset and explain one row with
Integrated Gradients via :meth:`darsha.DarshaClient.explain_integrated_gradients`.

Install: ``pip install "darsha[ml,dl]"`` (pandas, scikit-learn, torch, captum).

Run: ``cd examples && python mlp_integrated_gradients_example.py``

Writes:
- ``examples/reports_ig/audit_report.md``
- ``examples/reports_ig/audit_report.html``
- ``examples/reports_ig/mlp_decision.html``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import darsha

FEATURE_NAMES = list(load_breast_cancer().feature_names)

dm = darsha.init(
    project="breast-cancer-mlp",
    regulation="eu-ai-act",
    risk_level="high",
)


class MLP(nn.Module):
    def __init__(self, n_in: int, n_hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dm.track_training
def train_mlp(X_train: np.ndarray, y_train: np.ndarray) -> MLP:
    n_features = X_train.shape[1]
    model = MLP(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.from_numpy(X_train).float()
    yt = torch.from_numpy(y_train).long()
    model.train()
    for _ in range(400):
        opt.zero_grad()
        logits = model(xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


@dm.explain_integrated_gradients(
    counterfactual=True,
    feature_names=FEATURE_NAMES,
    n_steps=40,
)
def predict_proba(model: MLP, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(x)
        return torch.softmax(logits, dim=-1)


REPORT_DIR = Path(__file__).resolve().parent / "reports_ig"


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    model = train_mlp(X_train, y_train)

    with torch.no_grad():
        xt_test = torch.from_numpy(X_test)
        prob_test = torch.softmax(model(xt_test), dim=-1).numpy()
    y_pred = prob_test.argmax(axis=1)
    proba_pos = prob_test[:, 1]
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1v = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, proba_pos))
    except ValueError:
        auc = 0.0
    cm = confusion_matrix(y_test, y_pred).tolist()
    dm.register_governance(
        intended_use="Illustrative breast-cancer risk demo; not for clinical or diagnostic use.",
        known_limitations="Small MLP on toy data; not externally validated. IG explains local behavior, not causal effects.",
        human_oversight="Qualified clinician is required for any real diagnostic workflow; this SDK sample is for governance reporting only.",
        data_provenance="sklearn `load_breast_cancer` (UCI / Wisconsin-related features; see sklearn docs for citation).",
        model_changelog=[{"version": "ig-demo-1.0", "date": "2026-04-23", "notes": "MLP + Integrated Gradients + audit template"}],
    )
    dm.register_evaluation(
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        split_description="25% hold-out, stratified (train_test_split, random_state=0) on load_breast_cancer",
        test_metrics={"precision": prec, "recall": rec, "f1": f1v, "roc_auc": auc, "accuracy": acc},
        confusion_matrix=cm,
    )

    dm.register_integrated_gradients_feature_importance(
        model,
        torch.from_numpy(X_train),
        max_samples=min(80, len(X_train)),
        feature_names=FEATURE_NAMES,
        n_steps=32,
    )

    x1 = torch.from_numpy(X_test[:1])
    result = predict_proba(model, x1)

    print(result.decision.to_text())
    print(result.explanation.plain_language)
    print(result.audit_id)
    print()
    print(result.as_markdown_table()[:2000])
    print("...")
    print()
    print("--- Full audit trail (all steps, tabular) ---")
    print(dm.format_audit_log_markdown())

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit_md = dm.write_audit_report(REPORT_DIR / "audit_report.md", title="MLP + Integrated Gradients — audit trail")
    audit_html = dm.write_audit_report(
        REPORT_DIR / "audit_report.html",
        title="MLP + Integrated Gradients — audit trail",
        format="html",
    )
    decision_html = result.write_report(
        REPORT_DIR / "mlp_decision.html", title="MLP + Integrated Gradients", format="html"
    )
    print()
    print("--- Downloadable reports written ---")
    print(f"  {audit_md}")
    print(f"  {audit_html}")
    print(f"  {decision_html}")


if __name__ == "__main__":
    main()
