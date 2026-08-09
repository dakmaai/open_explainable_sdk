from __future__ import annotations

import logging

import dakma
import dakma_sdk
from dakma_sdk.models import DecisionEvent, EnrichedResult


def test_import_dakma_and_dakma_sdk():
    assert dakma.init is not None
    assert dakma_sdk.DakmaClient is not None


def test_init_returns_client_with_metadata():
    client = dakma.init(project="p1", regulation="eu-ai-act", risk_level="high")
    assert client.project == "p1"
    assert client.regulation == "eu-ai-act"
    assert client.risk_level == "high"
    assert client.project_info.id == "p1"


def test_init_project_only():
    client = dakma.init(project="credit-scoring")
    assert client.project == "credit-scoring"
    assert client.regulation == ""
    assert client.risk_level == ""


def test_enable_logging_is_idempotent(caplog):
    caplog.set_level(logging.DEBUG, logger="dakma_sdk")
    logger1 = dakma.enable_logging(logging.DEBUG)
    handler_count = len(
        [h for h in logger1.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)]
    )
    logger2 = dakma.enable_logging(logging.DEBUG)
    handler_count_after = len(
        [h for h in logger2.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)]
    )
    assert logger1 is logger2
    assert handler_count_after == handler_count


def test_decision_event_type_exported():
    assert DecisionEvent.__name__ == "DecisionEvent"
    assert EnrichedResult is DecisionEvent
    assert dakma.Model is not None
    assert dakma.Dataset is not None
    assert dakma.AuditEvent is not None
