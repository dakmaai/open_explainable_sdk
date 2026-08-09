from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from dakma_sdk.models import EnrichedResult
from tests.conftest import DummyModel


def test_register_governance_and_evaluation_merge(client):
    client.register_governance(intended_use="Demo", known_limitations="Synthetic data")
    client.register_evaluation(n_train=100, n_test=30, test_metrics={"f1": 0.75})
    client.register_evaluation(test_metrics={"accuracy": 0.9})

    assert client._governance["intended_use"] == "Demo"
    assert client._evaluation["n_train"] == 100
    assert client._evaluation["test_metrics"]["f1"] == 0.75
    assert client._evaluation["test_metrics"]["accuracy"] == 0.9


def test_format_audit_log_markdown_includes_governance(client):
    client.register_governance(intended_use="Demo use")
    md = client.format_audit_log_markdown(title="Test audit")
    assert "EU AI Act" in md
    assert "Demo use" in md


def test_write_audit_and_decision_reports(tmp_path: Path, client, dummy_model):
    client.register_governance(intended_use="Report demo")
    client.register_evaluation(n_train=5, n_test=2, test_metrics={"accuracy": 1.0})

    row = pd.DataFrame({"f1": [1.0]})

    @client.explain()
    def score(model, applicant_row):
        return model.predict_proba(applicant_row)

    result = score(dummy_model, row)

    audit_md = client.write_audit_report(tmp_path / "audit.md")
    audit_html = client.write_audit_report(tmp_path / "audit.html", format="html")
    decision_md = result.write_report(tmp_path / "decision.md")
    decision_html = result.write_report(tmp_path / "decision.html", format="html")

    assert audit_md.exists()
    assert audit_html.exists()
    assert decision_md.exists()
    assert decision_html.exists()
    assert "Report demo" in audit_md.read_text(encoding="utf-8")
    assert "Decision" in decision_md.read_text(encoding="utf-8")


def test_enriched_result_as_dict_and_markdown(client, dummy_model):
    row = pd.DataFrame({"f1": [1.0]})

    @client.explain()
    def score(model, applicant_row):
        return model.predict_proba(applicant_row)

    result = score(dummy_model, row)
    as_dict = result.as_dict()
    assert "audit_trail_id" in as_dict
    assert "plain_language" in as_dict
    md = result.as_markdown_table()
    assert "Decision" in md
