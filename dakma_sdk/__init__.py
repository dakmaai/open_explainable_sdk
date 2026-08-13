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
    ExplainPayload,
    Explanation,
    FeatureImportance,
    Model,
    Project,
    TopFactor,
)

__all__ = [
    "init",
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


def init(project: str, regulation: str = "", risk_level: str = "") -> DakmaClient:
    """Create a project-scoped client; ``regulation`` and ``risk_level`` are optional."""
    return DakmaClient(project=project, regulation=regulation, risk_level=risk_level)
