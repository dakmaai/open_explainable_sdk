# darsha

`darsha` is a Python explainability SDK for model governance and decision transparency across:

- data ingestion (`dtype`, null counts)
- feature engineering lineage
- training metadata (params and metrics)
- inference-time explainability: **SHAP** for common tabular / tree models, or **Integrated Gradients** (Captum) for differentiable models such as **PyTorch** `nn.Module` (+ plain language + audit id)
- basic monitoring hooks (bias and drift indicators)

## Install

```bash
pip install darsha
```

For tabular / tree ML (pandas, scikit-learn, XGBoost, SHAP):

```bash
pip install "darsha[ml]"
```

For **deep learning** (PyTorch + Captum, used by Integrated Gradients), install the `dl` extra as well (often with `ml` for sklearn data utilities in the examples):

```bash
pip install "darsha[ml,dl]"
```

## Imports

Use either the short name or the implementation package:

```python
import darsha
# or
import darsha_sdk
```

Both expose `init`, `DarshaClient`, and the result types from `darsha_sdk.models`.

For **tabular audit output** (Markdown tables: data schema, training params, monitoring, inference), use `DarshaClient.format_audit_log_markdown()` or `result.as_markdown_table()` on a single decision.

**Downloadable reports:** write UTF-8 files you can open, share, or print to PDF:

- `dm_c.write_audit_report("audit_report.md")` or `dm_c.write_audit_report("audit_report.html", format="html")` — full audit trail
- `result.write_report("decision.md")` or `result.write_report("decision.html", format="html")` — one decision

**Governance and evaluation (EU AI Act Art. 13 support):** every full audit and single-decision report starts with (1) **Evaluation and dataset** — train/test description, `n_train` / `n_test`, hold-out `test_metrics` (precision, recall, F1, ROC-AUC, etc.), optional `confusion_matrix`, plus automated gap notes when items are missing; and (2) **EU AI Act (Art. 13) — documentation** — intended use, limitations, human oversight, data provenance, model changelog. Populate these with `DarshaClient.register_evaluation(...)` and `DarshaClient.register_governance(...)` before running explained inference so they appear on downloaded reports and are snapshotted on each `DecisionEvent` (and nested `Explanation`).

The tabular XGBoost example writes `examples/reports/audit_report.md`, `audit_report.html`, and `last_decision.html`. The **MLP (PyTorch) Integrated Gradients** example writes the same style of reports under `examples/reports_ig/`. Both scripts register sample governance and test-set metrics for demonstration.

**Deep learning (PyTorch):** decorate inference with `DarshaClient.explain_integrated_gradients` and optionally call `register_integrated_gradients_feature_importance` for a global snapshot in the audit. See the **Example: PyTorch MLP and Integrated Gradients** section below.

## Quickstart

Decision events are the center of the SDK. Provenance (model / dataset), explanation,
governance, monitoring, and audit all contribute to a single :class:`~darsha_sdk.models.DecisionEvent`.

```python
import darsha
from xgboost import XGBClassifier

dk = darsha.init(project="credit-scoring")

dk.dataset(id="credit-training", version="2026-08")
model = dk.model(
    name="credit-risk",
    version="3.2.1",
    artifact=classifier,  # optional: bind for SHAP when predict() only takes x
)

@model.explain()
def predict(x):
    return classifier.predict_proba(x)

result = predict(applicant)
print(result.decision)       # Decision(value=..., score=..., threshold=...)
print(result.explanation)    # Explanation(method="shap", status=..., factors=...)
print(result.audit_id)
```

Core objects: ``Project``, ``Model``, ``Dataset``, ``Decision``, ``Explanation``, ``AuditEvent``
(plus ``DecisionEvent`` as the runtime aggregate returned from explained inference).

Legacy client-level decorators still work:

```python
import darsha
from xgboost import XGBClassifier

dm_c = darsha.init(
    project="credit-scoring-package",
    regulation="eu-ai-act",
    risk_level="high",
)

@dm_c.track_data
def prepare_features(df):
    df["debt_ratio"] = df["total_debt"] / df["annual_income"]
    df["credit_util"] = df["balance"] / df["credit_limit"]
    return df

@dm_c.track_training
def train(X_train, y_train):
    model = XGBClassifier(n_estimators=200, max_depth=5)
    model.fit(X_train, y_train)
    return model

@dm_c.explain(counterfactual=True)
def score_applicant(model, applicant_row):
    return model.predict_proba(applicant_row)

result = score_applicant(model, applicant_row)
print(result.decision.to_text())
print(result.explanation.plain_language)
print(result.audit_id)
print(result.as_markdown_table())
```

## Example: `credit.csv` and XGBoost

The repo includes `examples/credit.csv` with columns `total_debt`, `annual_income`, `balance`, `credit_limit`, `target`, and optional `period` (`baseline` vs `drift`). A few cells are intentionally **empty** (nulls) so `@track_data` schema snapshots and engineered features reflect missing values. The example **trains on baseline** and **evaluates on drift** so feature means (debt_ratio, credit_util) and label prevalence shift—`monitor()` then reports **drift** (`positive_prediction_rate` vs training prevalence) alongside accuracy. Run:

```bash
pip install "darsha[ml]"
cd examples && python credit_scoring_example.py
```

Override the path by editing `CSV_PATH` in `credit_scoring_example.py` or copy `credit.csv` beside your script.

The same script calls `dm_c.monitor(...)` on the hold-out test set: it compares `y_true` vs thresholded `y_pred`, optional drift vs `expected_rate` (here, training-set label prevalence), and optional `group_positive_rates` when you pass `protected_feature` (the example uses income bands as a stand-in).

## Example: PyTorch MLP and Integrated Gradients

`examples/mlp_integrated_gradients_example.py` trains a small MLP on the sklearn breast cancer dataset and explains a decision with Captum’s Integrated Gradients, using the same audit and downloadable-report flow as the XGBoost example. It uses `@dm.explain_integrated_gradients`, `register_integrated_gradients_feature_importance`, and writes `audit_report` / `mlp_decision` Markdown and HTML under `examples/reports_ig/`.

```bash
pip install "darsha[ml,dl]"
cd examples && python mlp_integrated_gradients_example.py
```
