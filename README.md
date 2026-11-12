# dakma-sdk

PyPI package **`dakma-sdk`** — source repo: [dakmaai/open_explainable_sdk](https://github.com/dakmaai/open_explainable_sdk). Import as `dakma` or `dakma_sdk`.

`dakma-sdk` records a model's lifecycle and explains the decision that model just made. Every explained inference returns a decision event with a plain-language reason, the factors behind it, and an audit id.

> **Disclaimer:** dakma-sdk helps you *document* model lifecycle and explainability information. It does **not** provide legal advice or regulatory certification.

## Capabilities

| Capability | API | Extra | What you get |
| --- | --- | --- | --- |
| Data ingestion | `@track_data` | — | Column dtypes, null counts, and feature-engineering lineage |
| Training record | `@track_training` | — | Training parameters and metrics |
| Tabular binary decision | `@explain` | `[ml]` | `APPROVED` / `DECLINED` from the score and threshold, with SHAP factors |
| Tabular multi-class | `@explain` | `[ml]` | The predicted class. `target_class` explains a different output without changing the decision |
| PyTorch tabular | `@explain_integrated_gradients` | `[dl]` | The same decision, attributed with Integrated Gradients |
| Forecasting | `@explain_forecast` | `[ml]` | The forecast at `horizon`, attributed with SHAP from the feature metrics passed in |
| Text classification | `@explain_text` | `[dl]` | Per-token impacts, highlighted in the text and in reports |
| Image classification | `@explain_image` | `[dl]` | The predicted class, with those regions painted on the input image |
| Governance and evaluation | `register_governance`, `register_evaluation` | — | Intended use, limitations, oversight, provenance, hold-out metrics, and a confusion matrix on each report |
| Monitoring | `monitor` | — | Accuracy, positive-prediction rate, drift against an expected rate, and group positive rates |
| Reports | `write_report`, `write_audit_report` | — | One decision, or the full audit trail, as Markdown or HTML |

SHAP is used for tabular models and forecasts. Integrated Gradients is used for PyTorch tabular inputs, text, and images. A missing optional dependency still returns the decision, with `explanation.status == "degraded"`.

Binary tabular decisions stay `APPROVED` / `DECLINED`. Tabular models with 3 or more classes, and image classifiers, use the predicted class. Forecasts record the predicted value. Image overlays are a translucent tint: red supported the explained class, blue counted against it, and the color deepens where that region mattered more. The input stays visible. `highlighted_image()` returns the PNG. Text uses `highlighted_text()`.

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

Each example under `examples/` registers sample governance and metrics, then writes a report. The scripts are listed in **Examples** below.

**Deep learning (PyTorch):** decorate inference with `DakmaClient.explain_integrated_gradients` and optionally call `register_integrated_gradients_feature_importance` for a global snapshot in the audit. See **Example: PyTorch MLP and Integrated Gradients**.

**Text classification:** decorate inference with `DakmaClient.explain_text`. See **Text classification (highlighted words)**.

**Image classification:** decorate inference with `DakmaClient.explain_image`. See **Image classification**.

**Forecasting:** decorate inference with `DakmaClient.explain_forecast`. See **Forecasting**.

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

## Forecasting

`explain_forecast` is for a regression model that returns a number, or one number per horizon.
Pass the feature metrics for the forecast origin into the wrapped function. The decision is that
forecast, and SHAP attributes it. `horizon` selects the step (default `0`, the first step ahead).

```python
import dakma

dm = dakma.init(project="demand-forecast")

@dm.explain_forecast(horizon=0, counterfactual=True)
def forecast(model, feature_metrics):
    return model.predict(feature_metrics)

result = forecast(model, feature_metrics)
print(result.decision.to_text())          # forecast 12.40 at horizon 0
print(result.explanation.plain_language)
print(result.explanation.metadata["feature_metrics"])
```

`feature_metrics` is a one-row DataFrame or a mapping of feature name to value (lags, calendar
measures, exogenous drivers). A multi-step return such as `[y1, y2, y3]` is explained at
`horizon`. `@model.explain()` stays approve/decline for tabular binary classifiers.

## Image classification

Requires the **`[dl]`** extra. `explain_image` explains a binary or multi-class image model.
The decision is the predicted class, and the regions behind that class are painted on the input
image. Red supported the class. Blue counted against it. Pass `class_names` for the labels, and
`target_class` to inspect a different class without changing the decision.

```python
import dakma

dm = dakma.init(project="image-classification")

@dm.explain_image(class_names=["cat", "dog", "bird"])
def classify(model, image):
    return model(image)

result = classify(model, image)          # image is (1, C, H, W)
print(result.decision.to_text())         # class dog (probability: 0.91)
result.highlighted_image()               # PNG bytes of the overlay
```

A model that returns two scores is binary. Three or more scores are multi-class. One score is the
positive class compared with `threshold`; pass `class_names=["negative", "positive"]` for those labels.

## Text classification (highlighted words)

Requires the **`[dl]`** extra. `explain_text` attributes the decision to individual tokens with
Integrated Gradients, so you can see which words drove it:

```python
import dakma
import torch

dk = dakma.init(project="review-sentiment")
model = dk.model(name="review-sentiment", version="1.0.0", artifact=classifier)

tokens = review.split()

@model.explain_text(tokens=tokens)
def classify(ids):
    return torch.softmax(classifier(ids), dim=1)

event = classify(encode(review))

print(event.highlighted_text())        # the review, each word shaded in the terminal
print(event.explanation.plain_language)
for a in event.explanation.text_attributions:
    print(a.token, a.impact)           # rude -2.2189, terrible -2.0800, ...
```

The wrapped function receives the model and a `(1, seq_len)` tensor of token ids and returns logits
or probabilities. Attributions are taken over **word embeddings** through a forward hook, so any
model containing an `nn.Embedding` works — including Hugging Face encoders — and nothing needs to be
restructured. Pass `embedding_layer=` when a model embeds several inputs.

Display tokens come from `tokens=` (a list, or a callable receiving the id tensor) or `tokenizer=`
(anything exposing `convert_ids_to_tokens`); without either you get positional `token_0`, `token_1`
labels. Impacts explain the score the threshold read — the last model output — so a positive impact
pushed the text towards `APPROVED`; pass `target_class=` to attribute a different class.

Each decision event carries the per-token impacts on `explanation.text_attributions` and renders
them three ways: `highlighted_text()` for terminals and notebooks, `highlighted_text("markdown")`,
and `highlighted_text("html")`. Markdown and HTML reports written with `write_report(...)` include a
**Highlighted text** section automatically, alongside the usual per-token impact bars.

When attribution cannot run — no embedding layer, no id tensor — inference still returns its
decision event, with `explanation.status == "degraded"` and the reason recorded in
`explanation.metadata["attribution_error"]`.

For tabular tensors keep using `explain_integrated_gradients`. Its
`reduce_attributions_to_features` helper now also accepts `reduce="tokens"` if you need per-token
impacts from an attribution matrix you computed yourself.

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

For **text**, run `cd examples && python text_highlight_example.py` after `pip install "dakma-sdk[dl]"`, then open `examples/reports_text/text_decision.html` in a browser.

```bash
pip install "dakma-sdk[ml,dl]"
cd examples && python mlp_integrated_gradients_example.py
```

## Examples

Run these from `examples/` after installing the extra named in the table.

| Script | Capability | Extra |
| --- | --- | --- |
| `credit_scoring_example.py` | Tabular XGBoost: data tracking, training, SHAP, monitoring, audit report | `[ml]` |
| `credit_scoring_svm_example.py` | The same credit workflow with a scikit-learn SVM | `[ml]` |
| `iris_multiclass_example.py` | Three-class decision, plus `target_class` for a class the model did not pick | `[ml]` |
| `wine_multiclass_example.py` | Three-class decision, one explained prediction per class | `[ml]` |
| `forecast_example.py` | Demand forecast from lag and calendar feature metrics | `[ml]` |
| `mlp_integrated_gradients_example.py` | PyTorch MLP explained with Integrated Gradients | `[ml,dl]` |
| `text_highlight_example.py` | Token highlights for a text classifier | `[dl]` |
| `image_classification_example.py` | Image classifier with the decision painted on the input | `[dl]` |

Reports are written next to each script, under `examples/reports/`, `examples/reports_svm/`, `examples/reports_ig/`, `examples/reports_text/`, `examples/reports_forecast/`, `examples/reports_iris/`, `examples/reports_wine/`, and `examples/reports_image/`.

## Testing

From a clone of this repository:

```bash
pip install -e ".[dev]"        # unit tests only (numpy)
pip install -e ".[dev,ml,dl]"  # full suite including example smoke tests
pytest
pytest -m smoke              # example scripts only
```

CI runs unit tests on Python 3.9–3.12 and example smoke tests on 3.11 (see `.github/workflows/ci.yml`).

## Release 3.0.0

**3.0.0** adds image classification explanations. Decorate a PyTorch classifier with `explain_image`. The decision is the predicted class, and Integrated Gradients paints the pixels behind that decision on the test image. Red supported the class and blue counted against it. The tint is translucent and deepens where a region mattered more, so the input stays visible. `highlighted_image()` returns the PNG, and Markdown and HTML reports include the overlay.

Install the release once it is on PyPI:

```bash
pip install "dakma-sdk==3.0.0"
```

Image explanations need the deep-learning extra:

```bash
pip install "dakma-sdk[dl]==3.0.0"
```

| Requirement | Detail |
| --- | --- |
| Python | `>=3.9` |
| Package | `dakma-sdk==3.0.0` (`pyproject.toml` version is already `3.0.0`) |
| Runtime | `numpy>=1.22` |
| Image explanations | extra `[dl]`: `torch>=2.0`, `captum>=0.6` |
| Tabular explanations | extra `[ml]`: `pandas>=1.5`, `scikit-learn>=1.3`, `xgboost>=1.7`, `shap>=0.44` |
| Model | one PyTorch `nn.Module`, one image `(1, C, H, W)` or `(C, H, W)` |

## Publishing to PyPI

Releases are published by [`.github/workflows/publish.yml`](.github/workflows/publish.yml) when you:

- push a version tag (e.g. `v3.0.0`), or
- publish a GitHub Release, or
- run the workflow manually (**Actions → Publish to PyPI → Run workflow**).

The workflow builds with `python -m build`, runs `twine check`, and uploads via [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) (no long-lived API token in the repo). A tag push and the GitHub Release for that same tag both trigger the workflow. The second run skips a version that is already on PyPI.

### One-time PyPI setup (trusted publishing)

1. Create a PyPI account and project for **`dakma-sdk`** (if not already created).
2. On PyPI → **Account settings → Publishing** → **Add a new pending publisher**:
   - **PyPI project name:** `dakma-sdk`
   - **Owner:** `dakmaai`
   - **Repository name:** `open_explainable_sdk`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`
3. In GitHub → **Settings → Environments** → create environment **`pypi`** (optional approval rules recommended).

**Fallback:** token upload is a separate setup. This workflow publishes with trusted publishing.

### Publish 3.0.0

Do this from the commit that contains `version = "3.0.0"` and the 3.0.0 changelog, after that commit is on `main`. Pushing the tag uploads the package.

```bash
git checkout main
git pull
git tag v3.0.0
git push origin v3.0.0
```

Then publish the GitHub Release for `v3.0.0` with the 3.0.0 notes from `CHANGELOG.md`. Confirm [dakma-sdk 3.0.0](https://pypi.org/project/dakma-sdk/3.0.0/) on PyPI.

## Limitations

- **Documentation aid only** — not legal advice or compliance certification.
- **SHAP / IG** require optional deps; failures may return empty explanations unless deps are installed.
- **Feature lineage** is inferred from simple `df["col"] = ...` source patterns only.
- **Tabular binary decisions** are `APPROVED` / `DECLINED`. Tabular models with 3 or more classes, and image classifiers, use the predicted class. Forecasts use `explain_forecast`.
- **`monitor()`** group rates are illustrative; not a full fairness or bias audit framework.
