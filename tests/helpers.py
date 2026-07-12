"""Shared paths and optional-dependency skip markers for tests."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _imports_available(*module_names: str) -> bool:
    for name in module_names:
        try:
            __import__(name)
        except Exception:
            return False
    return True


def has_ml_stack() -> bool:
    return _imports_available("pandas", "sklearn", "shap", "xgboost")


def has_dl_stack() -> bool:
    return has_ml_stack() and _imports_available("torch", "captum")


def has_sklearn_ml_stack() -> bool:
    return _imports_available("pandas", "sklearn", "shap")


requires_ml = pytest.mark.skipif(not has_ml_stack(), reason="requires dakma-sdk[ml] dependencies")
requires_sklearn_ml = pytest.mark.skipif(
    not has_sklearn_ml_stack(), reason="requires pandas, scikit-learn, and shap"
)
requires_dl = pytest.mark.skipif(not has_dl_stack(), reason="requires dakma-sdk[ml,dl] dependencies")
