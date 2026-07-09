# dakma-sdk

PyPI package **`dakma-sdk`** — source repo: [dakmaai/open_explainable_sdk](https://github.com/dakmaai/open_explainable_sdk). Import as `dakma` or `dakma_sdk`.

`dakma-sdk` is a Python explainability SDK for model governance and decision transparency across:

- data ingestion (`dtype`, null counts)
- feature engineering lineage
- training metadata (params and metrics)
- inference-time explainability: **SHAP** for common tabular / tree models, or **Integrated Gradients** (Captum) for differentiable models such as **PyTorch** `nn.Module` (+ plain language + audit id)
- basic monitoring hooks (bias and drift indicators)

> **Disclaimer:** dakma-sdk helps you *document* model lifecycle and explainability information. It does **not** provide legal advice or regulatory certification.

## Install

```bash
pip install dakma-sdk
```

For tabular / tree ML (pandas, scikit-learn, XGBoost, SHAP):

```bash
pip install "dakma-sdk[ml]"
```

For **deep learning** (PyTorch + Captum, used by Integrated Gradients), install the `dl` extra as well (often with `ml` for sklearn data utilities in the examples):

```bash
pip install "dakma-sdk[ml,dl]"
```

## Imports

Use either the short name or the implementation package:

```python
import dakma
# or
import dakma_sdk
```

Both expose `init`, `DakmaClient`, and the result types from `dakma_sdk.models`.

For **tabular audit output** (Markdown tables: data schema, training params, monitoring, inference), use `DakmaClient.format_audit_log_markdown()` or `result.as_markdown_table()` on a single decision.

**Downloadable reports:** write UTF-8 files you can open, share, or print to PDF:

- `dm_c.write_audit_report("audit_report.md")` or `dm_c.write_audit_report("audit_report.html", format="html")` — full audit trail
- `result.write_report("decision.md")` or `result.write_report("decision.html", format="html")` — one decision

**Governance and evaluation templates:** every full audit and single-decision report can start with (1) **Evaluation and dataset** — train/test description, `n_train` / `n_test`, hold-out `test_metrics` (precision, recall, F1, ROC-AUC, etc.), optional `confusion_matrix`, plus automated gap notes when items are missing; and (2) **Governance documentation** — intended use, limitations, human oversight, data provenance, model changelog. These sections are *documentation aids only*; populate them with `DakmaClient.register_evaluation(...)` and `DakmaClient.register_governance(...)` before running explained inference so they appear on downloaded reports and are snapshotted on each `EnrichedResult.explain` payload.

The tabular XGBoost example writes `examples/reports/audit_report.md`, `audit_report.html`, and `last_decision.html`. The **MLP (PyTorch) Integrated Gradients** example writes the same style of reports under `examples/reports_ig/`. Both scripts register sample governance and test-set metrics for demonstration.

**Deep learning (PyTorch):** decorate inference with `DakmaClient.explain_integrated_gradients` and optionally call `register_integrated_gradients_feature_importance` for a global snapshot in the audit. See the **Example: PyTorch MLP and Integrated Gradients** section below.

## Quickstart (tabular ML)

Requires the **`[ml]`** extra (XGBoost, SHAP, pandas):

```bash
pip install "dakma-sdk[ml]"
```

```python
import dakma
from xgboost import XGBClassifier

dm_c = dakma.init(
    project="credit-scoring-package",
    regulation="internal-policy",
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
print(result.decision_output.to_text())
print(result.explain.plain_language)
print(result.explain.audit_trail_id)
print(result.as_markdown_table())
```

## Example: `credit.csv` and XGBoost

The repo includes `examples/credit.csv` with columns `total_debt`, `annual_income`, `balance`, `credit_limit`, `target`, and optional `period` (`baseline` vs `drift`). A few cells are intentionally **empty** (nulls) so `@track_data` schema snapshots and engineered features reflect missing values. The example **trains on baseline** and **evaluates on drift** so feature means (debt_ratio, credit_util) and label prevalence shift—`monitor()` then reports **drift** (`positive_prediction_rate` vs training prevalence) alongside accuracy. Run:

```bash
pip install "dakma-sdk[ml]"
cd examples && python credit_scoring_example.py
```

Override the path by editing `CSV_PATH` in `credit_scoring_example.py` or copy `credit.csv` beside your script.

The same script calls `dm_c.monitor(...)` on the hold-out test set: it compares `y_true` vs thresholded `y_pred`, optional drift vs `expected_rate` (here, training-set label prevalence), and optional `group_positive_rates` when you pass `protected_feature` (the example uses income bands as a stand-in).

## Example: PyTorch MLP and Integrated Gradients

`examples/mlp_integrated_gradients_example.py` trains a small MLP on the sklearn breast cancer dataset and explains a decision with Captum’s Integrated Gradients, using the same audit and downloadable-report flow as the XGBoost example. It uses `@dm.explain_integrated_gradients`, `register_integrated_gradients_feature_importance`, and writes `audit_report` / `mlp_decision` Markdown and HTML under `examples/reports_ig/`.

```bash
pip install "dakma-sdk[ml,dl]"
cd examples && python mlp_integrated_gradients_example.py
```

## Testing

From a clone of this repository:

```bash
pip install -e ".[dev]"        # unit tests only (numpy)
pip install -e ".[dev,ml,dl]"  # full suite including example smoke tests
pytest
pytest -m smoke              # example scripts only
```

CI runs `pytest` on Python 3.9–3.12 (see `.github/workflows/ci.yml`).

## Limitations

- **Documentation aid only** — not legal advice or compliance certification.
- **SHAP / IG** require optional deps; failures may return empty explanations unless deps are installed.
- **Feature lineage** is inferred from simple `df["col"] = ...` source patterns only.
- **Decision labels** default to `APPROVED` / `DECLINED` (credit-style); not generic for all ML tasks.
- **`monitor()`** group rates are illustrative; not a full fairness or bias audit framework.
