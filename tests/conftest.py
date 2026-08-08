from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

import darsha
from darsha_sdk.core import DarshaClient


@pytest.fixture
def client() -> DarshaClient:
    return darsha.init(project="test-project", regulation="eu-ai-act", risk_level="high")


class DummyModel:
    """Minimal sklearn-like model for decorator tests without optional ML deps."""

    def __init__(self, proba: float = 0.7) -> None:
        self._proba = proba

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.array([[1.0 - self._proba, self._proba]])

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"n_estimators": 10, "max_depth": 3}

    def score(self, X: Any, y: Any) -> float:
        return 0.91


@pytest.fixture
def dummy_model() -> DummyModel:
    return DummyModel(proba=0.7)
