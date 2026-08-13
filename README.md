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

## The decision event

The **decision event** is the SDK's fundamental object. Every explained inference produces one, and
provenance, explanation, and governance all hang off it:

```
Project → AI Decision Event → AuditEvent
             ├── Provenance   (Model, Dataset)
             ├── Explanation  (method, status, factors, quality)
             └── Governance   (documentation fields, evaluation)
```

Register a model, decorate inference, and read the event:

```python
import dakma

dk = dakma.init(project="credit-scoring")

dk.dataset(id="credit-training", version="2026-08", row_count=1823921)
model = dk.model(
    name="credit-risk",
    version="3.2.1",
    artifact=classifier,   # lets SHAP and the model hash resolve without a model argument
)

@model.explain()
def predict(x):
    return classifier.predict_proba(x)

result = predict(applicant)
print(result.decision)      # Decision(value="DECLINED", score=0.421, threshold=0.5)
print(result.explanation)   # Explanation(method="shap", status="success", factors=[...])
print(result.audit_id)
```

The six core objects are `Project`, `Model`, `Dataset`, `Decision`, `Explanation`, and `AuditEvent`;
`DecisionEvent` is the runtime aggregate returned to you, and an `AuditEvent` of
`type="decision"` is appended to the audit log for each one.

`regulation` and `risk_level` on `init` remain available and are recorded on the project and in
report documentation blocks.

## Quickstart (tabular ML)

Requires the **`[ml]`** extra (XGBoost, SHAP, pandas):

```bash
pip install "dakma-sdk[ml]"
```

Decorating with `DakmaClient.explain` also returns a decision event, and keeps the
`decision_output` / `explain` names used by earlier releases:

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

CI runs unit tests on Python 3.9–3.12 and example smoke tests on 3.11 (see `.github/workflows/ci.yml`).

## Publishing to PyPI

Releases are published by [`.github/workflows/publish.yml`](.github/workflows/publish.yml) when you:

- push a version tag (e.g. `v0.1.0`), or
- publish a GitHub Release, or
- run the workflow manually (**Actions → Publish to PyPI → Run workflow**).

### One-time PyPI setup (trusted publishing)

1. Create a PyPI account and project for **`dakma-sdk`** (if not already created).
2. On PyPI → **Account settings → Publishing** → **Add a new pending publisher**:
   - **PyPI project name:** `dakma-sdk`
   - **Owner:** `dakmaai`
   - **Repository name:** `open_explainable_sdk`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`
3. In GitHub → **Settings → Environments** → create environment **`pypi`** (optional approval rules recommended).
4. Bump `version` in `pyproject.toml`, update `CHANGELOG.md`, commit, then:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The workflow builds with `python -m build`, runs `twine check`, and uploads via [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) (no long-lived API token in the repo).

**Fallback:** set a repository secret `PYPI_API_TOKEN` if you prefer token-based upload instead of trusted publishing.

## Limitations

- **Documentation aid only** — not legal advice or compliance certification.
- **SHAP / IG** require optional deps; failures may return empty explanations unless deps are installed.
- **Feature lineage** is inferred from simple `df["col"] = ...` source patterns only.
- **Decision labels** default to `APPROVED` / `DECLINED` (credit-style); not generic for all ML tasks.
- **`monitor()`** group rates are illustrative; not a full fairness or bias audit framework.
