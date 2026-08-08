from __future__ import annotations

import pytest

from darsha_sdk.models import FeatureImportance, TopFactor


def test_top_factor_to_text():
    factor = TopFactor(name="debt_ratio", value=0.42, impact=-0.15, direction="down")
    text = factor.to_text()
    assert "debt_ratio" in text
    assert "-0.15" in text


def test_feature_importance_to_text_shap_and_ig():
    shap_row = FeatureImportance(name="income", mean_abs_shap=0.31, source="shap")
    ig_row = FeatureImportance(name="income", mean_abs_shap=0.31, source="integrated_gradients")
    assert "mean |SHAP|" in shap_row.to_text()
    assert "mean |IG|" in ig_row.to_text()


def test_decision_output_to_text():
    from darsha_sdk.models import Decision, DecisionOutput

    out = Decision(value="APPROVED", score=0.72, threshold=0.5)
    assert out.to_text() == "APPROVED (score: 0.72, threshold: 0.50)"
    assert out.label == "APPROVED"
    assert DecisionOutput is Decision


def test_core_objects_construct():
    from darsha_sdk.models import AuditEvent, Dataset, Decision, Explanation, Model, Project

    project = Project(id="credit-scoring", regulation="eu-ai-act", risk_level="high")
    model = Model(id="credit-risk", version="3.2.1", framework="xgboost", model_hash="sha256:abc")
    dataset = Dataset(id="credit-training", version="2026-08", schema_hash="sha256:def", row_count=100)
    decision = Decision(value="DECLINED", score=0.421, threshold=0.5)
    explanation = Explanation(method="shap", status="success", factors=[], quality={"factor_count": 0})
    audit = AuditEvent(
        id="ax-1",
        type="decision",
        timestamp="2026-08-09T00:00:00+00:00",
        model=model,
        dataset=dataset,
        decision=decision,
        explanation=explanation,
        project=project,
    )
    assert audit.decision.value == "DECLINED"
    assert audit.model.id == "credit-risk"
