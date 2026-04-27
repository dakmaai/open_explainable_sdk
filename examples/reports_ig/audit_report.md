## MLP + Integrated Gradients — audit trail

*Total entries: 2*

### 1. training_tracking

| Field | Value |
|---|---|
| timestamp | 2026-04-22T18:13:44.558038+00:00 |
| function | train_mlp |
| model_class | MLP |
| started_at | 2026-04-22T18:13:43.728373+00:00 |
| finished_at | 2026-04-22T18:13:44.558038+00:00 |

**Compute usage** (this process)
| Metric | Value |
|---|---|
| Wall time | 829.56 ms |
| CPU time (user + system) | 1911.34 ms |
| RAM RSS (start) | 265.02 MB |
| RAM RSS (end) | 328.23 MB |
| RAM delta | +63.219 MB |
| GPU peak memory allocated | — |
| GPU device | — |

*CPU and RAM refer to this Python process (via psutil). GPU peak uses PyTorch CUDA when available.*

### 2. inference_explain

| Field | Value |
|---|---|
| timestamp | 2026-04-22T18:13:44.836183+00:00 |
| function | predict_proba |

**Decision**
| Field | Value |
|---|---|
| label | APPROVED |
| score | 1.00 |
| threshold | 0.50 |

**Raw model output** (unchanged)
| decision |
|---|
| tensor([[4.6812e-04, 9.9953e-01]]) |

**Explainability**
| Field | Value |
|---|---|
| audit_trail_id | ax-2026-04-22-cbf1294 |
| model_version | breast-cancer-mlp / torch-mlp |
| attribution | integrated_gradients |
| plain_language | Approved. Main reason: low texture error which moved the score up. |
| counterfactual | Would decline if texture error worsens further |

**Regulation flags**
| Flag |
|---|
| EU AI Act Art.13 ✓ |
| GDPR Art.22 ✓ |

**Compute usage** (this process)
| Metric | Value |
|---|---|
| Wall time | 5.2 ms |
| CPU time (user + system) | 4.50 ms |
| RAM RSS (start) | 328.66 MB |
| RAM RSS (end) | 328.72 MB |
| RAM delta | +0.062 MB |
| GPU peak memory allocated | — |
| GPU device | — |

*CPU and RAM refer to this Python process (via psutil). GPU peak uses PyTorch CUDA when available.*

**Top factors (Integrated Gradients for this row)**

<figure class="dakma-shap-chart"><svg xmlns="http://www.w3.org/2000/svg" width="560" height="106" viewBox="0 0 560 106" role="img"><line x1="302.0" y1="10" x2="302.0" y2="102" stroke="#ccc" stroke-width="1"/><text x="132" y="12" font-size="11" fill="#666">Integrated Gradients (this row)</text><text x="0" y="34" font-size="12" fill="#1a1a1a">texture error</text><rect x="302.0" y="20" width="170.0" height="18" fill="#1d4ed8" rx="2" opacity="0.92"><title>texture error: +1.194235</title></rect><text x="476.0" y="34" font-size="11" fill="#444">+1.1942</text><text x="0" y="60" font-size="12" fill="#1a1a1a">symmetry error</text><rect x="185.9" y="46" width="116.1" height="18" fill="#b91c1c" rx="2" opacity="0.92"><title>symmetry error: -0.815651</title></rect><text x="476.0" y="60" font-size="11" fill="#444">-0.8157</text><text x="0" y="86" font-size="12" fill="#1a1a1a">radius error</text><rect x="302.0" y="72" width="109.9" height="18" fill="#1d4ed8" rx="2" opacity="0.92"><title>radius error: +0.771736</title></rect><text x="476.0" y="86" font-size="11" fill="#444">+0.7717</text></svg></figure>

**Feature importance (global)**
*Mean |Integrated Gradients| averaged over the reference sample passed to register_integrated_gradients_feature_importance.*

<figure class="dakma-shap-chart"><svg xmlns="http://www.w3.org/2000/svg" width="560" height="544" viewBox="0 0 560 544" role="img"><text x="132" y="10" font-size="11" fill="#666">Mean |IG|</text><text x="0" y="30" font-size="12" fill="#1a1a1a">radius error</text><rect x="132" y="16" width="356.0" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>radius error: 0.735601</title></rect><text x="492.0" y="30" font-size="11" fill="#444">0.7356</text><text x="0" y="56" font-size="12" fill="#1a1a1a">worst smoothness</text><rect x="132" y="42" width="326.0" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>worst smoothness: 0.673641</title></rect><text x="462.0137013970318" y="56" font-size="11" fill="#444">0.6736</text><text x="0" y="82" font-size="12" fill="#1a1a1a">worst perimeter</text><rect x="132" y="68" width="315.4" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>worst perimeter: 0.651701</title></rect><text x="451.3956065987473" y="82" font-size="11" fill="#444">0.6517</text><text x="0" y="108" font-size="12" fill="#1a1a1a">mean concavity</text><rect x="132" y="94" width="293.2" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>mean concavity: 0.605809</title></rect><text x="429.1860354488478" y="108" font-size="11" fill="#444">0.6058</text><text x="0" y="134" font-size="12" fill="#1a1a1a">worst texture</text><rect x="132" y="120" width="292.4" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>worst texture: 0.604264</title></rect><text x="428.4381232069387" y="134" font-size="11" fill="#444">0.6043</text><text x="0" y="160" font-size="12" fill="#1a1a1a">worst radius</text><rect x="132" y="146" width="269.5" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>worst radius: 0.556782</title></rect><text x="405.4592493786095" y="160" font-size="11" fill="#444">0.5568</text><text x="0" y="186" font-size="12" fill="#1a1a1a">mean area</text><rect x="132" y="172" width="265.3" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>mean area: 0.548187</title></rect><text x="401.2995303752587" y="186" font-size="11" fill="#444">0.5482</text><text x="0" y="212" font-size="12" fill="#1a1a1a">worst concave points</text><rect x="132" y="198" width="265.2" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>worst concave points: 0.548049</title></rect><text x="401.23257852901355" y="212" font-size="11" fill="#444">0.5480</text><text x="0" y="238" font-size="12" fill="#1a1a1a">mean concave points</text><rect x="132" y="224" width="263.0" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>mean concave points: 0.543385</title></rect><text x="398.9755012871496" y="238" font-size="11" fill="#444">0.5434</text><text x="0" y="264" font-size="12" fill="#1a1a1a">texture error</text><rect x="132" y="250" width="241.9" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>texture error: 0.499867</title></rect><text x="377.9144437705008" y="264" font-size="11" fill="#444">0.4999</text><text x="0" y="290" font-size="12" fill="#1a1a1a">symmetry error</text><rect x="132" y="276" width="222.2" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>symmetry error: 0.459169</title></rect><text x="358.218588191935" y="290" font-size="11" fill="#444">0.4592</text><text x="0" y="316" font-size="12" fill="#1a1a1a">compactness error</text><rect x="132" y="302" width="219.3" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>compactness error: 0.453225</title></rect><text x="355.3418532423364" y="316" font-size="11" fill="#444">0.4532</text><text x="0" y="342" font-size="12" fill="#1a1a1a">worst concavity</text><rect x="132" y="328" width="211.2" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>worst concavity: 0.436395</title></rect><text x="347.1966507359025" y="342" font-size="11" fill="#444">0.4364</text><text x="0" y="368" font-size="12" fill="#1a1a1a">worst fractal dimension</text><rect x="132" y="354" width="192.1" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>worst fractal dimension: 0.396995</title></rect><text x="328.128877192266" y="368" font-size="11" fill="#444">0.3970</text><text x="0" y="394" font-size="12" fill="#1a1a1a">perimeter error</text><rect x="132" y="380" width="188.4" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>perimeter error: 0.389261</title></rect><text x="324.3859436001954" y="394" font-size="11" fill="#444">0.3893</text><text x="0" y="420" font-size="12" fill="#1a1a1a">worst area</text><rect x="132" y="406" width="185.4" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>worst area: 0.383018</title></rect><text x="321.3645753357564" y="420" font-size="11" fill="#444">0.3830</text><text x="0" y="446" font-size="12" fill="#1a1a1a">fractal dimension error</text><rect x="132" y="432" width="175.6" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>fractal dimension error: 0.362860</title></rect><text x="311.6088992689597" y="446" font-size="11" fill="#444">0.3629</text><text x="0" y="472" font-size="12" fill="#1a1a1a">mean perimeter</text><rect x="132" y="458" width="163.1" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>mean perimeter: 0.336965</title></rect><text x="299.0767201597556" y="472" font-size="11" fill="#444">0.3370</text><text x="0" y="498" font-size="12" fill="#1a1a1a">area error</text><rect x="132" y="484" width="162.6" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>area error: 0.335883</title></rect><text x="298.55336177708614" y="498" font-size="11" fill="#444">0.3359</text><text x="0" y="524" font-size="12" fill="#1a1a1a">worst compactness</text><rect x="132" y="510" width="135.5" height="18" fill="#2563eb" rx="2" opacity="0.92"><title>worst compactness: 0.279992</title></rect><text x="271.5044014907006" y="524" font-size="11" fill="#444">0.2800</text></svg></figure><p class="dakma-chart-note"><em>Showing top 20 features.</em></p>

**Metadata**
| Key | Value |
|---|---|
| project | breast-cancer-mlp |
| regulation | eu-ai-act |
| risk_level | high |
