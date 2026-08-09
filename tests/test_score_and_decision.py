from __future__ import annotations

import numpy as np

from dakma_sdk.core import DakmaClient


def test_extract_score_scalar():
    assert DakmaClient._extract_score(0.82) == 0.82


def test_extract_score_1d_uses_last_element():
    assert DakmaClient._extract_score(np.array([0.2, 0.8])) == 0.8


def test_extract_score_2d_uses_first_row_last_column():
    batch = np.array([[0.1, 0.9], [0.4, 0.6]])
    assert DakmaClient._extract_score(batch) == 0.9


def test_to_decision_output_approved_and_declined():
    approved = DakmaClient._to_decision_output(0.7, threshold=0.5)
    declined = DakmaClient._to_decision_output(0.3, threshold=0.5)
    assert approved.value == "APPROVED"
    assert approved.label == "APPROVED"
    assert approved.score == 0.7
    assert declined.value == "DECLINED"
    assert declined.score == 0.3


def test_model_version_for_unknown_model(client):
    assert client._model_version(None) == "test-project / unknown-model"
