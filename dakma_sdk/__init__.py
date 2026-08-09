import logging
from typing import Optional, Union

from .audit_format import (
    format_audit_entry_markdown,
    format_audit_log_markdown,
    format_inference_result_markdown,
    write_audit_report,
    write_inference_report,
    write_report_file,
)
from .compute_usage import ComputeUsageRecorder, ComputeUsageSnapshot
from .core import DakmaClient, ModelHandle
from .models import (
    AuditEvent,
    Dataset,
    Decision,
    DecisionEvent,
    DecisionOutput,
    EnrichedResult,
    Explanation,
    ExplainPayload,
    FeatureImportance,
    Model,
    Project,
    TopFactor,
)

# Library logging best practice: attach a NullHandler so importing the SDK never
# emits "No handlers could be found" warnings and never configures logging for the
# host application. Applications opt in via standard ``logging`` config or
# :func:`enable_logging`.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "init",
    "enable_logging",
    "DakmaClient",
    "ModelHandle",
    "ComputeUsageRecorder",
    "ComputeUsageSnapshot",
    "Project",
    "Model",
    "Dataset",
    "Decision",
    "Explanation",
    "AuditEvent",
    "DecisionEvent",
    "DecisionOutput",
    "EnrichedResult",
    "ExplainPayload",
    "FeatureImportance",
    "TopFactor",
    "format_audit_log_markdown",
    "format_audit_entry_markdown",
    "format_inference_result_markdown",
    "write_audit_report",
    "write_inference_report",
    "write_report_file",
]


def init(
    project: str,
    regulation: str = "",
    risk_level: str = "",
) -> DakmaClient:
    """Create a project-scoped client. DecisionEvents are produced via ``dk.model(...).explain()``."""
    return DakmaClient(project=project, regulation=regulation, risk_level=risk_level)


def enable_logging(
    level: Union[int, str] = logging.INFO,
    *,
    handler: Optional[logging.Handler] = None,
    fmt: str = "%(asctime)s %(levelname)s %(name)s: %(message)s",
) -> logging.Logger:
    """Convenience helper to see Dakma's logs without configuring logging yourself.

    Attaches a single ``StreamHandler`` (stderr) to the ``dakma_sdk`` logger and sets
    its level. Intended for scripts, notebooks, and demos; production apps should
    configure the standard :mod:`logging` framework instead. Returns the package logger.

    Calling this repeatedly will not add duplicate stream handlers.
    """
    pkg_logger = logging.getLogger(__name__)
    pkg_logger.setLevel(level)

    if handler is None:
        # Avoid stacking duplicate stream handlers on repeated calls.
        for existing in pkg_logger.handlers:
            if isinstance(existing, logging.StreamHandler) and not isinstance(existing, logging.NullHandler):
                existing.setLevel(level)
                return pkg_logger
        handler = logging.StreamHandler()

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    pkg_logger.addHandler(handler)
    return pkg_logger
