from __future__ import annotations

from dataclasses import asdict

import numpy as np

from dakma_sdk.audit_format import (
    DOCUMENTATION_DISCLAIMER_MARKDOWN,
    format_audit_log_markdown,
    format_compliance_preamble_markdown,
    format_evaluation_block_markdown,
    format_governance_eu13_markdown,
    format_inference_result_html,
    format_inference_result_markdown,
    write_report_file,
)
from dakma_sdk.models import DecisionOutput, EnrichedResult, ExplainPayload, TopFactor


def _sample_result_dict(*, with_governance: bool = True) -> dict:
    governance = (
        {
            "intended_use": "Demo triage only",
            "known_limitations": "Synthetic data",
            "human_oversight": "Human review required",
        }
        if with_governance
        else None
    )
    evaluation = {
        "n_train": 100,
        "n_test": 30,
        "split_description": "70/30 hold-out",
        "test_metrics": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "roc_auc": 0.82},
        "confusion_matrix": [[10, 2], [3, 15]],
    }
    explain = ExplainPayload(
        audit_trail_id="ax-2026-01-01-abc1234",
        model_version="test-project / demo",
        plain_language="Approved. Main reason: high debt_ratio which moved the score up.",
        top_factors=[
            TopFactor(name="debt_ratio", value=0.4, impact=0.12, direction="up"),
        ],
        regulation_flags=[
            "EU AI Act Art. 13 documentation fields: 3/5 populated",
            "Documentation aid only — not legal or compliance certification",
        ],
        governance=governance,
        evaluation=evaluation,
        attribution_backend="shap",
    )
    result = EnrichedResult(
        decision=np.array([[0.2, 0.8]]),
        explain=explain,
        decision_output=DecisionOutput(label="APPROVED", score=0.8, threshold=0.5),
    )
    return asdict(result)


class TestGovernanceAndEvaluationBlocks:
    def test_governance_template_heading_and_missing_fields(self) -> None:
        md = format_governance_eu13_markdown({"intended_use": "Credit demo"})
        assert "documentation template" in md.lower()
        assert "Intended use" in md
        assert "Credit demo" in md
        assert "not documented" in md

    def test_evaluation_block_renders_metrics_and_split(self) -> None:
        md = format_evaluation_block_markdown(
            {
                "n_train": 80,
                "n_test": 20,
                "split_description": "Stratified hold-out",
                "test_metrics": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "roc_auc": 0.91},
                "confusion_matrix": [[5, 1], [2, 12]],
            }
        )
        assert "Evaluation and dataset" in md
        assert "Stratified hold-out" in md
        assert "precision" in md
        assert "roc_auc" in md

    def test_evaluation_gaps_when_empty(self) -> None:
        md = format_evaluation_block_markdown({})
        assert "not provided" in md.lower()
        assert "No **test-set** metrics" in md or "No test-set metrics" in md.replace("*", "")

    def test_compliance_preamble_includes_disclaimer(self) -> None:
        md = format_compliance_preamble_markdown(
            governance={"intended_use": "x"},
            evaluation={"n_train": 1, "n_test": 1},
        )
        assert DOCUMENTATION_DISCLAIMER_MARKDOWN in md
        assert "GDPR" in md  # mentioned only in disclaimer negation
        assert "Art.13 ✓" not in md
        assert "GDPR Art.22" not in md


class TestInferenceReportFormatting:
    def test_markdown_decision_and_documentation_status(self) -> None:
        md = format_inference_result_markdown(_sample_result_dict())
        assert "**Decision**" in md
        assert "APPROVED" in md
        assert "**Documentation status**" in md
        assert "Documentation aid only" in md
        assert "Regulation flags" not in md

    def test_html_escapes_user_governance_text(self, tmp_path) -> None:
        payload = _sample_result_dict()
        payload["explain"]["governance"]["intended_use"] = '<script>alert("x")</script>'
        html = format_inference_result_html(payload)
        assert "<script>" not in html
        assert "alert" in html

    def test_write_report_file_creates_utf8_file(self, tmp_path) -> None:
        path = write_report_file(tmp_path / "nested" / "report.md", "# Title\n\nbody")
        assert path.read_text(encoding="utf-8") == "# Title\n\nbody"


class TestAuditLogFormatting:
    def test_audit_log_markdown_includes_entry_types(self) -> None:
        entries = [
            {
                "type": "monitoring",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "payload": {"accuracy": 0.9, "positive_prediction_rate": 0.4},
            },
            {
                "type": "data_tracking",
                "timestamp": "2026-01-01T00:01:00+00:00",
                "function": "prepare_features",
                "input_schema": {"x": {"dtype": "float64", "nulls": 0}},
                "output_schema": {"x": {"dtype": "float64", "nulls": 0}},
                "lineage_added": {"y": "df['x'] + 1"},
            },
        ]
        md = format_audit_log_markdown(
            entries,
            title="Test audit",
            governance={"intended_use": "demo"},
            evaluation={"n_train": 10, "n_test": 5, "test_metrics": {"f1": 0.7}},
        )
        assert "Test audit" in md
        assert "monitoring" in md
        assert "data_tracking" in md
        assert DOCUMENTATION_DISCLAIMER_MARKDOWN in md
