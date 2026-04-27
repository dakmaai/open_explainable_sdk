"""Convenience import: ``import dakma`` re-exports the public API from ``dakma_sdk``."""

from dakma_sdk import (
    ComputeUsageRecorder,
    ComputeUsageSnapshot,
    DakmaClient,
    DecisionOutput,
    EnrichedResult,
    ExplainPayload,
    FeatureImportance,
    TopFactor,
    format_audit_entry_markdown,
    format_audit_log_markdown,
    format_inference_result_markdown,
    init,
    write_audit_report,
    write_inference_report,
    write_report_file,
)

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
