from __future__ import annotations

import math

import numpy as np
import pytest

from dakma_sdk.core import DakmaClient


@pytest.fixture
def client() -> DakmaClient:
    return DakmaClient(project="test-project", regulation="internal-policy", risk_level="high")


class TestExtractScore:
    def test_scalar(self) -> None:
        assert DakmaClient._extract_score(0.73) == pytest.approx(0.73)

    def test_zero_dimensional_array(self) -> None:
        assert DakmaClient._extract_score(np.array(0.42)) == pytest.approx(0.42)

    def test_one_dimensional_proba_vector(self) -> None:
        assert DakmaClient._extract_score(np.array([0.1, 0.9])) == pytest.approx(0.9)

    def test_two_dimensional_batch(self) -> None:
        assert DakmaClient._extract_score(np.array([[0.2, 0.8]])) == pytest.approx(0.8)

    def test_python_list(self) -> None:
        assert DakmaClient._extract_score([0.05, 0.95]) == pytest.approx(0.95)


class TestMonitor:
    def test_accuracy_and_positive_rate(self, client: DakmaClient) -> None:
        report = client.monitor(y_true=[1, 0, 1, 0], y_pred=[1, 0, 0, 0])
        assert report["accuracy"] == pytest.approx(0.75)
        assert report["positive_prediction_rate"] == pytest.approx(0.25)
        assert report["drift_from_expected_rate"] is None
        assert report["group_positive_rates"] == {}

    def test_drift_from_expected_rate(self, client: DakmaClient) -> None:
        report = client.monitor(y_true=[1, 1, 0, 0], y_pred=[1, 1, 1, 0], expected_rate=0.25)
        assert report["positive_prediction_rate"] == pytest.approx(0.75)
        assert report["drift_from_expected_rate"] == pytest.approx(0.5)

    def test_group_positive_rates(self, client: DakmaClient) -> None:
        report = client.monitor(
            y_true=[1, 0, 1, 0],
            y_pred=[1, 0, 1, 1],
            protected_feature=["A", "A", "B", "B"],
        )
        assert report["group_positive_rates"]["A"] == pytest.approx(0.5)
        assert report["group_positive_rates"]["B"] == pytest.approx(1.0)

    def test_mismatched_protected_feature_length_ignored(self, client: DakmaClient) -> None:
        report = client.monitor(
            y_true=[1, 0],
            y_pred=[1, 0],
            protected_feature=["only-one"],
        )
        assert report["group_positive_rates"] == {}

    def test_empty_predictions(self, client: DakmaClient) -> None:
        report = client.monitor(y_true=[], y_pred=[])
        assert math.isnan(report["accuracy"])
        assert report["positive_prediction_rate"] == 0.0


class TestFeatureLineage:
    def test_extracts_df_column_assignments(self, client: DakmaClient) -> None:
        def prepare_features(df):
            df["debt_ratio"] = df["total_debt"] / df["annual_income"]
            df["credit_util"] = df["balance"] / df["credit_limit"]
            return df

        lineage = client._extract_feature_lineage(prepare_features)
        assert lineage["debt_ratio"] == 'df["total_debt"] / df["annual_income"]'
        assert lineage["credit_util"] == 'df["balance"] / df["credit_limit"]'

    def test_ignores_non_matching_assignments(self, client: DakmaClient) -> None:
        def other_style(df):
            out = df.copy()
            out["x"] = 1
            return out

        assert client._extract_feature_lineage(other_style) == {}


class TestRegulationFlags:
    def test_counts_populated_governance_fields(self, client: DakmaClient) -> None:
        client.register_governance(intended_use="demo", human_oversight="review")
        flags = client._regulation_flags()
        assert flags[0] == "Governance documentation fields: 2/5 populated"
        assert flags[1] == "Policy context: internal-policy"
        assert "Documentation aid only" in flags[2]
        assert "✓" not in " ".join(flags)

    def test_empty_governance_still_reports_policy_context(self) -> None:
        client = DakmaClient(project="p", regulation="custom-policy", risk_level="low")
        flags = client._regulation_flags()
        assert flags[0] == "Governance documentation fields: 0/5 populated"
        assert flags[1] == "Policy context: custom-policy"
        assert "Documentation aid only" in flags[2]
