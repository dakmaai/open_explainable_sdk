## Credit scoring (SVM) — audit trail

## EU AI Act (Art. 13) — documentation

Transparency fields below support Art. 13-style disclosure. Use ``DakmaClient.register_governance(...)`` to fill them. Empty items are flagged in reports.

| Topic | Documentation |
|---|---|
| Intended use | Demo credit default risk triage; not for automated denial as the sole input. |
| Known limitations | Example dataset; SVM probabilities are Platt-scaled. Legal/fairness review required for production. |
| Human oversight mechanism | Underwriter reviews model-assisted decisions; overrides supported. |
| Data source / provenance (training data) | examples/credit.csv (synthetic/illustrative for SDK demo). |
| Model version history / change log | - **demo-svm-1.0** (2026-04-23): SVM + SHAP + audit template |

## Evaluation and dataset

### Dataset and train/test split

| Item | Value |
|---|---|
| Split / methodology description | Train on period=baseline, test on period=drift (hold-out) |
| Training set size (n_train) | 14 |
| Test / hold-out set size (n_test) | 12 |

### Test-set / hold-out metrics
Report metrics on **held-out** data (not training fit / training_score only). Use ``DakmaClient.register_evaluation(test_metrics={...}, ...)``.

| Metric (test/hold-out) | Value |
|---|---|
| accuracy | 0.25 |
| f1 | 0.0 |
| precision | 0.0 |
| recall | 0.0 |
| roc_auc | 0.7037037037037037 |

### Checklist: recommended test metrics
| Item | Status |
|---|---|
| Precision (test) | ✓ |
| Recall (test) | ✓ |
| F1 (test) | ✓ |
| ROC-AUC (test) | ✓ |
| Accuracy (test) | ✓ |

### Confusion matrix (test / hold-out)
|  | Pred 0 | Pred 1 |
|---|---|---|
| Actual 0 | 3 | 0 |
| Actual 1 | 9 | 0 |

### Gaps to address (automated notes)
- *No automated gap flags (still verify precision/recall/F1, lineage, and oversight manually).*

---


*Total entries: 4*

### 1. data_tracking

| Field | Value |
|---|---|
| timestamp | 2026-04-22T19:00:18.531142+00:00 |
| function | prepare_features |

**Input schema**
| Column | dtype | nulls |
|---|---|---|
| annual_income | float64 | 1 |
| balance | float64 | 2 |
| credit_limit | float64 | 1 |
| period | object | 0 |
| target | int64 | 0 |
| total_debt | float64 | 1 |

**Output schema**
| Column | dtype | nulls |
|---|---|---|
| annual_income | float64 | 1 |
| balance | float64 | 2 |
| credit_limit | float64 | 1 |
| credit_util | float64 | 3 |
| debt_ratio | float64 | 2 |
| period | object | 0 |
| target | int64 | 0 |
| total_debt | float64 | 1 |

**Feature lineage (new/updated columns)**
| Feature | Expression / source |
|---|---|
| annual_income | df["annual_income"].replace(0, pd.NA) |
| credit_limit | df["credit_limit"].replace(0, pd.NA) |
| credit_util | df["balance"] / df["credit_limit"] |
| debt_ratio | df["total_debt"] / df["annual_income"] |

### 2. training_tracking

| Field | Value |
|---|---|
| timestamp | 2026-04-22T19:00:19.481632+00:00 |
| function | train_svm |
| model_class | Pipeline |
| started_at | 2026-04-22T19:00:18.532550+00:00 |
| finished_at | 2026-04-22T19:00:19.481632+00:00 |

**Model parameters**
| Parameter | Value |
|---|---|
| imputer | SimpleImputer(strategy='median') |
| imputer__add_indicator | False |
| imputer__copy | True |
| imputer__fill_value | — |
| imputer__keep_empty_features | False |
| imputer__missing_values | nan |
| imputer__strategy | median |
| memory | — |
| scaler | StandardScaler() |
| scaler__copy | True |
| scaler__with_mean | True |
| scaler__with_std | True |
| steps | [('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('svc', SVC(class_weight='balanced', probability=True, random_state=42))] |
| svc | SVC(class_weight='balanced', probability=True, random_state=42) |
| svc__C | 1.0 |
| svc__break_ties | False |
| svc__cache_size | 200 |
| svc__class_weight | balanced |
| svc__coef0 | 0.0 |
| svc__decision_function_shape | ovr |
| svc__degree | 3 |
| svc__gamma | scale |
| svc__kernel | rbf |
| svc__max_iter | -1 |
| svc__probability | True |
| svc__random_state | 42 |
| svc__shrinking | True |
| svc__tol | 0.001 |
| svc__verbose | False |
| transform_input | — |
| verbose | False |

**Metrics**
| Metric | Value |
|---|---|
| training_score | 1.0 |

**Compute usage** (this process)
| Metric | Value |
|---|---|
| Wall time | 949.02 ms |
| CPU time (user + system) | 689.64 ms |
| RAM RSS (start) | 160.09 MB |
| RAM RSS (end) | 279.27 MB |
| RAM delta | +119.172 MB |
| GPU peak memory allocated | — |
| GPU device | — |

*CPU and RAM refer to this Python process (via psutil). GPU peak uses PyTorch CUDA when available.*

### 3. monitoring

| Field | Value |
|---|---|
| timestamp | 2026-04-22T19:00:19.485858+00:00 |

**Monitoring summary**
| Metric | Value |
|---|---|
| accuracy | 0.25 |
| positive_prediction_rate | 0.0 |
| drift_from_expected_rate | 0.5 |

**Group positive prediction rates** (bias / fairness signal)
| Group | Mean predicted positive |
|---|---|
| low | 0.0 |
| mid | 0.0 |

### 4. inference_explain

| Field | Value |
|---|---|
| timestamp | 2026-04-22T19:00:19.490128+00:00 |
| function | score_applicant |

**Decision**
| Field | Value |
|---|---|
| label | DECLINED |
| score | 0.36 |
| threshold | 0.50 |

**Raw model output** (unchanged)
| decision |
|---|
| [[0.6437720492565571, 0.35622795074344293]] |

**Explainability**
| Field | Value |
|---|---|
| audit_trail_id | ax-2026-04-22-c32a10c |
| model_version | credit-scoring-svm / pipeline |
| attribution | shap |
| plain_language | Declined. Main reason: high annual income which moved the score up. |
| counterfactual | Would decline if annual_income worsens further |

**Regulation flags**
| Flag |
|---|
| EU AI Act Art.13 ✓ |
| GDPR Art.22 ✓ |

**Compute usage** (this process)
| Metric | Value |
|---|---|
| Wall time | 0.71 ms |
| CPU time (user + system) | 0.63 ms |
| RAM RSS (start) | 279.73 MB |
| RAM RSS (end) | 279.73 MB |
| RAM delta | +0.000 MB |
| GPU peak memory allocated | — |
| GPU device | — |

*CPU and RAM refer to this Python process (via psutil). GPU peak uses PyTorch CUDA when available.*

**Top factors (SHAP for this row)**

<figure class="dakma-shap-chart"><svg xmlns="http://www.w3.org/2000/svg" width="560" height="106" viewBox="0 0 560 106" role="img"><line x1="302.0" y1="10" x2="302.0" y2="102" stroke="#ccc" stroke-width="1"/><text x="132" y="12" font-size="11" fill="#666">SHAP (this row)</text><text x="0" y="34" font-size="12" fill="#1a1a1a">annual_income</text><rect x="302.0" y="20" width="170.0" height="18" fill="#1d4ed8" rx="2" opacity="0.92"><title>annual_income: +550.000000</title></rect><text x="476.0" y="34" font-size="11" fill="#444">+550.0000</text><text x="0" y="60" font-size="12" fill="#1a1a1a">total_debt</text><rect x="302.0" y="46" width="160.7" height="18" fill="#1d4ed8" rx="2" opacity="0.92"><title>total_debt: +520.000000</title></rect><text x="476.0" y="60" font-size="11" fill="#444">+520.0000</text><text x="0" y="86" font-size="12" fill="#1a1a1a">credit_limit</text><rect x="302.0" y="72" width="55.6" height="18" fill="#1d4ed8" rx="2" opacity="0.92"><title>credit_limit: +180.000000</title></rect><text x="476.0" y="86" font-size="11" fill="#444">+180.0000</text></svg></figure>

**Metadata**
| Key | Value |
|---|---|
| project | credit-scoring-svm |
| regulation | eu-ai-act |
| risk_level | high |

**Feature lineage (snapshot)**
| Feature | Expression |
|---|---|
| annual_income | df["annual_income"].replace(0, pd.NA) |
| credit_limit | df["credit_limit"].replace(0, pd.NA) |
| credit_util | df["balance"] / df["credit_limit"] |
| debt_ratio | df["total_debt"] / df["annual_income"] |
