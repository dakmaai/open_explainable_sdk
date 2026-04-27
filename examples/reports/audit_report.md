## Credit scoring — audit trail

*Total entries: 4*

### 1. data_tracking

| Field | Value |
|---|---|
| timestamp | 2026-04-18T00:46:09.337799+00:00 |
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
| timestamp | 2026-04-18T00:46:09.367317+00:00 |
| function | train |
| model_class | XGBClassifier |
| started_at | 2026-04-18T00:46:09.339155+00:00 |
| finished_at | 2026-04-18T00:46:09.367317+00:00 |

**Model parameters**
| Parameter | Value |
|---|---|
| base_score | — |
| booster | — |
| callbacks | — |
| colsample_bylevel | — |
| colsample_bynode | — |
| colsample_bytree | — |
| device | — |
| early_stopping_rounds | — |
| enable_categorical | False |
| eval_metric | logloss |
| feature_types | — |
| feature_weights | — |
| gamma | — |
| grow_policy | — |
| importance_type | — |
| interaction_constraints | — |
| learning_rate | — |
| max_bin | — |
| max_cat_threshold | — |
| max_cat_to_onehot | — |
| max_delta_step | — |
| max_depth | 5 |
| max_leaves | — |
| min_child_weight | — |
| missing | nan |
| monotone_constraints | — |
| multi_strategy | — |
| n_estimators | 200 |
| n_jobs | — |
| num_parallel_tree | — |
| objective | binary:logistic |
| random_state | 42 |
| reg_alpha | — |
| reg_lambda | — |
| sampling_method | — |
| scale_pos_weight | — |
| subsample | — |
| tree_method | — |
| validate_parameters | — |
| verbosity | — |

**Metrics**
| Metric | Value |
|---|---|
| training_score | 1.0 |

### 3. monitoring

| Field | Value |
|---|---|
| timestamp | 2026-04-18T00:46:10.652702+00:00 |

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
| timestamp | 2026-04-18T00:46:10.670091+00:00 |
| function | score_applicant |

**Decision**
| Field | Value |
|---|---|
| label | DECLINED |
| score | 0.16 |
| threshold | 0.50 |

**Raw model output** (unchanged)
| decision |
|---|
| [[0.8350871205329895, 0.1649128645658493]] |

**Explainability**
| Field | Value |
|---|---|
| audit_trail_id | ax-2026-04-18-88a6997 |
| model_version | credit-scoring-package / xgb-200-d5 |
| plain_language | Declined. Main reason: high total debt which moved the score down. |
| counterfactual | Would approve if total_debt < 36400.00 |

**Regulation flags**
| Flag |
|---|
| EU AI Act Art.13 ✓ |
| GDPR Art.22 ✓ |

**Top factors (SHAP for this row)**

<figure class="dakma-shap-chart"><svg xmlns="http://www.w3.org/2000/svg" width="560" height="106" viewBox="0 0 560 106" role="img"><line x1="302.0" y1="10" x2="302.0" y2="102" stroke="#ccc" stroke-width="1"/><text x="132" y="12" font-size="11" fill="#666">SHAP (this row)</text><text x="0" y="34" font-size="12" fill="#1a1a1a">total_debt</text><rect x="132.0" y="20" width="170.0" height="18" fill="#b91c1c" rx="2" opacity="0.92"><title>total_debt: -1.622119</title></rect><text x="476.0" y="34" font-size="11" fill="#444">-1.6221</text><text x="0" y="60" font-size="12" fill="#1a1a1a">annual_income</text><rect x="302.0" y="46" width="0.5" height="18" fill="#1d4ed8" rx="2" opacity="0.92"><title>annual_income: +0.000000</title></rect><text x="476.0" y="60" font-size="11" fill="#444">+0.0000</text><text x="0" y="86" font-size="12" fill="#1a1a1a">balance</text><rect x="302.0" y="72" width="0.5" height="18" fill="#1d4ed8" rx="2" opacity="0.92"><title>balance: +0.000000</title></rect><text x="476.0" y="86" font-size="11" fill="#444">+0.0000</text></svg></figure>

**Feature importance (global)**
*Mean |SHAP| averaged over the reference sample passed to register_shap_feature_importance.*

<figure class="dakma-shap-chart"><svg xmlns="http://www.w3.org/2000/svg" width="560" height="180" viewBox="0 0 560 180" role="img"><text x="132" y="10" font-size="11" fill="#666">Mean |SHAP|</text><text x="0" y="30" font-size="12" fill="#1a1a1a">total_debt</text><rect x="132" y="16" width="356.0" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>total_debt: 1.622119</title></rect><text x="492.0" y="30" font-size="11" fill="#444">1.6221</text><text x="0" y="56" font-size="12" fill="#1a1a1a">annual_income</text><rect x="132" y="42" width="0.5" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>annual_income: 0.000000</title></rect><text x="136.0" y="56" font-size="11" fill="#444">0.0000</text><text x="0" y="82" font-size="12" fill="#1a1a1a">balance</text><rect x="132" y="68" width="0.5" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>balance: 0.000000</title></rect><text x="136.0" y="82" font-size="11" fill="#444">0.0000</text><text x="0" y="108" font-size="12" fill="#1a1a1a">credit_limit</text><rect x="132" y="94" width="0.5" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>credit_limit: 0.000000</title></rect><text x="136.0" y="108" font-size="11" fill="#444">0.0000</text><text x="0" y="134" font-size="12" fill="#1a1a1a">debt_ratio</text><rect x="132" y="120" width="0.5" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>debt_ratio: 0.000000</title></rect><text x="136.0" y="134" font-size="11" fill="#444">0.0000</text><text x="0" y="160" font-size="12" fill="#1a1a1a">credit_util</text><rect x="132" y="146" width="0.5" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>credit_util: 0.000000</title></rect><text x="136.0" y="160" font-size="11" fill="#444">0.0000</text></svg></figure>

**Metadata**
| Key | Value |
|---|---|
| project | credit-scoring-package |
| regulation | eu-ai-act |
| risk_level | high |

**Feature lineage (snapshot)**
| Feature | Expression |
|---|---|
| annual_income | df["annual_income"].replace(0, pd.NA) |
| credit_limit | df["credit_limit"].replace(0, pd.NA) |
| credit_util | df["balance"] / df["credit_limit"] |
| debt_ratio | df["total_debt"] / df["annual_income"] |
