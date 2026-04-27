from .audit_format import (
    format_audit_entry_markdown,
    format_audit_log_markdown,
    format_inference_result_markdown,
    write_audit_report,
    write_inference_report,
    write_report_file,
)
from .compute_usage import ComputeUsageRecorder, ComputeUsageSnapshot
from .core import DakmaClient
from .models import DecisionOutput, EnrichedResult, ExplainPayload, FeatureImportance, TopFactor

__all__ = [
    "init",
    "DakmaClient",
    "ComputeUsageRecorder",
    "ComputeUsageSnapshot",
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


def init(project: str, regulation: str, risk_level: str) -> DakmaClient:
    return DakmaClient(project=project, regulation=regulation, risk_level=risk_level)
