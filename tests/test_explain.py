from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from darsha_sdk.models import DecisionEvent, EnrichedResult
from tests.conftest import DummyModel


def test_explain_returns_decision_event(client, dummy_model):
    row = pd.DataFrame({"f1": [1.0], "f2": [2.0]})

    @client.explain(counterfactual=True, threshold=0.5)
    def score(model, applicant_row):
        return model.predict_proba(applicant_row)

    result = score(dummy_model, row)

    assert isinstance(result, DecisionEvent)
    assert EnrichedResult is DecisionEvent
    assert result.decision.value == "APPROVED"
    assert result.decision_output.label == "APPROVED"
    assert result.decision.score == pytest.approx(0.7)
    assert result.audit_id.startswith("ax-")
    assert result.explanation.method == "shap"
    assert result.explain.attribution_backend == "shap"
    assert "test-project" in (result.explanation.model_version or "")
    assert len(client.export_audit_log()) >= 1
    audit_entry = client.export_audit_log()[-1]
    assert audit_entry["audit_event"]["type"] == "decision"
    assert audit_entry["audit_event"]["id"] == result.audit_id


def test_model_handle_explain_api(client, dummy_model):
    client.dataset(id="credit-training", version="2026-08", row_count=100)
    handle = client.model(name="credit-risk", version="3.2.1", artifact=dummy_model)
    row = pd.DataFrame({"f1": [1.0], "f2": [2.0]})

    @handle.explain(threshold=0.5)
    def predict(x):
        return dummy_model.predict_proba(x)

    result = predict(row)
    assert isinstance(result, DecisionEvent)
    assert result.decision.value == "APPROVED"
    assert result.explanation.status in {"success", "degraded"}
    assert result.model is not None
    assert result.model.id == "credit-risk"
    assert result.model.version == "3.2.1"
    assert result.model.framework is not None
    assert result.dataset is not None
    assert result.dataset.id == "credit-training"
    assert result.audit_id.startswith("ax-")
    print_fields = (result.decision, result.explanation, result.audit_id)
    assert all(x is not None for x in print_fields)


def test_explain_without_model_still_returns_result(client):
    @client.explain()
    def predict_only(row):
        return [0.2, 0.8]

    result = predict_only({"x": 1})
    assert isinstance(result, DecisionEvent)
    assert result.decision.score == pytest.approx(0.8)
    assert (result.explanation.model_version or "").endswith("unknown-model")


def test_explain_snapshots_governance_and_evaluation(client, dummy_model):
    client.register_governance(intended_use="Unit test only")
    client.register_evaluation(n_train=10, n_test=5, test_metrics={"accuracy": 0.9})

    row = pd.DataFrame({"f1": [1.0]})

    @client.explain()
    def score(model, applicant_row):
        return model.predict_proba(applicant_row)

    result = score(dummy_model, row)
    assert result.explanation.governance == {"intended_use": "Unit test only"}
    assert result.governance == {"intended_use": "Unit test only"}
    assert result.explanation.evaluation["n_train"] == 10
    assert result.explanation.evaluation["test_metrics"]["accuracy"] == 0.9
