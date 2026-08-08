from __future__ import annotations

import logging

import darsha
import darsha_sdk
from darsha_sdk.models import DecisionEvent, EnrichedResult


def test_import_darsha_and_darsha_sdk():
    assert darsha.init is not None
    assert darsha_sdk.DarshaClient is not None


def test_init_returns_client_with_metadata():
    client = darsha.init(project="p1", regulation="eu-ai-act", risk_level="high")
    assert client.project == "p1"
    assert client.regulation == "eu-ai-act"
    assert client.risk_level == "high"
    assert client.project_info.id == "p1"


def test_init_project_only():
    client = darsha.init(project="credit-scoring")
    assert client.project == "credit-scoring"
    assert client.regulation == ""
    assert client.risk_level == ""


def test_enable_logging_is_idempotent(caplog):
    caplog.set_level(logging.DEBUG, logger="darsha_sdk")
    logger1 = darsha.enable_logging(logging.DEBUG)
    handler_count = len(
        [h for h in logger1.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)]
    )
    logger2 = darsha.enable_logging(logging.DEBUG)
    handler_count_after = len(
        [h for h in logger2.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)]
    )
    assert logger1 is logger2
    assert handler_count_after == handler_count


def test_decision_event_type_exported():
    assert DecisionEvent.__name__ == "DecisionEvent"
    assert EnrichedResult is DecisionEvent
    assert darsha.Model is not None
    assert darsha.Dataset is not None
    assert darsha.AuditEvent is not None
