"""Decision events: the object every explained inference produces."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

import dakma
from dakma_sdk.core import DakmaClient
from dakma_sdk.models import (
    AuditEvent,
    Dataset,
    Decision,
    DecisionEvent,
    Explanation,
    Model,
    Project,
)


class DummyModel:
    """Minimal sklearn-like model so these tests need no optional ML dependencies."""

    def __init__(self, proba: float = 0.7) -> None:
        self._proba = proba

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.array([[1.0 - self._proba, self._proba]])

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"n_estimators": 10, "max_depth": 3}


@pytest.fixture
def client() -> DakmaClient:
    return dakma.init(project="test-project", regulation="internal-policy", risk_level="high")


@pytest.fixture
def dummy_model() -> DummyModel:
    return DummyModel(proba=0.7)


class TestCoreObjects:
    def test_decision_exposes_value_and_label(self) -> None:
        decision = Decision(value="DECLINED", score=0.421, threshold=0.5)
        assert decision.value == "DECLINED"
        assert decision.label == "DECLINED"
        assert decision.to_text() == "DECLINED (score: 0.42, threshold: 0.50)"

    def test_explanation_aliases_match_explain_payload(self) -> None:
        explanation = Explanation(method="shap", status="success", quality={"factor_count": 0})
        assert explanation.attribution_backend == "shap"
        assert explanation.top_factors == []

    def test_audit_event_holds_the_contributing_objects(self) -> None:
        audit = AuditEvent(
            id="ax-1",
            type="decision",
            timestamp="2026-08-13T00:00:00+00:00",
            model=Model(id="credit-risk", version="3.2.1", framework="xgboost", model_hash="sha256:abc"),
            dataset=Dataset(id="credit-training", version="2026-08", row_count=1823921),
            decision=Decision(value="DECLINED", score=0.421, threshold=0.5),
            explanation=Explanation(method="shap", status="success"),
            project=Project(id="credit-scoring"),
        )
        assert audit.type == "decision"
        assert audit.model.id == "credit-risk"
        assert audit.dataset.row_count == 1823921
        assert audit.as_dict()["decision"]["value"] == "DECLINED"


class TestModelHandleApi:
    def test_explain_returns_decision_event_with_provenance(self, client: DakmaClient) -> None:
        classifier = DummyModel(proba=0.7)
        client.dataset(id="credit-training", version="2026-08", row_count=100)
        model = client.model(name="credit-risk", version="3.2.1", artifact=classifier)

        @model.explain()
        def predict(x):
            return classifier.predict_proba(x)

        result = predict({"total_debt": 1.0})

        assert isinstance(result, DecisionEvent)
        assert result.decision.value == "APPROVED"
        assert result.decision.score == pytest.approx(0.7)
        assert result.explanation.method == "shap"
        assert result.explanation.status in {"success", "degraded"}
        assert result.audit_id.startswith("ax-")
        assert result.model.id == "credit-risk"
        assert result.model.version == "3.2.1"
        assert result.model.model_hash is not None
        assert result.dataset.id == "credit-training"
        assert result.project.id == "test-project"

    def test_dataset_infers_row_count_and_schema_hash(self, client: DakmaClient) -> None:
        dataset = client.dataset(id="credit", version="1", data={"total_debt": 1.0, "income": 2.0})
        assert dataset.row_count == 2
        assert dataset.schema_hash.startswith("sha256:")

    def test_bind_and_use_dataset_are_chainable(self, client: DakmaClient) -> None:
        dataset = client.dataset(id="reference", version="1", set_default=False)
        handle = client.model(name="m", version="1").bind(DummyModel()).use_dataset(dataset)
        assert handle.spec.model_hash is not None
        assert handle.dataset is dataset

    def test_integrated_gradients_handle_records_method(self, client: DakmaClient) -> None:
        handle = client.model(name="mlp", version="0.1")

        @handle.explain_integrated_gradients()
        def predict(x):
            return [0.1, 0.9]

        result = predict([1.0, 2.0])
        assert result.explanation.method == "integrated_gradients"
        assert result.decision.value == "APPROVED"


class TestClientExplainStillWorks:
    def test_returns_decision_event_with_legacy_aliases(self, client: DakmaClient, dummy_model: DummyModel) -> None:
        @client.explain(counterfactual=True)
        def score(model, row):
            return model.predict_proba(row)

        result = score(dummy_model, {"total_debt": 1.0})

        assert isinstance(result, DecisionEvent)
        assert result.decision_output.label == "APPROVED"
        assert result.explain.plain_language.startswith("Approved")
        assert result.explain.audit_trail_id == result.audit_id

    def test_writes_a_decision_audit_event(self, client: DakmaClient, dummy_model: DummyModel) -> None:
        @client.explain()
        def score(model, row):
            return model.predict_proba(row)

        result = score(dummy_model, {"total_debt": 1.0})
        entry = client.export_audit_log()[-1]

        assert entry["type"] == "inference_explain"
        assert entry["audit_event"]["type"] == "decision"
        assert entry["audit_event"]["id"] == result.audit_id

    def test_snapshots_governance_and_evaluation(self, client: DakmaClient, dummy_model: DummyModel) -> None:
        client.register_governance(intended_use="Unit test only")
        client.register_evaluation(n_train=10, n_test=5, test_metrics={"accuracy": 0.9})

        @client.explain()
        def score(model, row):
            return model.predict_proba(row)

        result = score(dummy_model, {"total_debt": 1.0})
        assert result.governance == {"intended_use": "Unit test only"}
        assert result.evaluation["test_metrics"]["accuracy"] == 0.9

    def test_markdown_table_renders_from_decision_event(self, client: DakmaClient, dummy_model: DummyModel) -> None:
        @client.explain()
        def score(model, row):
            return model.predict_proba(row)

        table = score(dummy_model, {"total_debt": 1.0}).as_markdown_table()
        assert "APPROVED" in table


class TestPublicApi:
    def test_init_accepts_project_only(self) -> None:
        client = dakma.init(project="credit-scoring")
        assert client.regulation == ""
        assert client.project_info.id == "credit-scoring"
        assert "Policy context" not in " ".join(client._regulation_flags())

    def test_core_objects_are_exported(self) -> None:
        for name in ("Project", "Model", "Dataset", "Decision", "Explanation", "AuditEvent", "DecisionEvent"):
            assert hasattr(dakma, name), name
