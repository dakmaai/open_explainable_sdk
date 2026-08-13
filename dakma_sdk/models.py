from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass
class TopFactor:
    name: str
    value: Any
    impact: float
    direction: str

    def to_text(self) -> str:
        sign = "+" if self.impact >= 0 else ""
        direction_text = "up" if self.impact >= 0 else "down"
        return f"{self.name} = {self.value} (pushed score {direction_text} {sign}{self.impact:.2f})"


@dataclass
class FeatureImportance:
    """Global importance for one feature (e.g. mean |SHAP| or mean |IG| over a reference sample)."""

    name: str
    mean_abs_shap: float
    source: str = "shap"  # "shap" | "integrated_gradients"

    def to_text(self) -> str:
        metric = "mean |SHAP|" if self.source == "shap" else "mean |IG|"
        return f"{self.name}: {metric} = {self.mean_abs_shap:.4f}"


@dataclass
class ExplainPayload:
    audit_trail_id: str
    model_version: str
    plain_language: str
    top_factors: List[TopFactor] = field(default_factory=list)
    counterfactual: Optional[str] = None
    #: Documentation status notes for reports (not legal or compliance certification).
    regulation_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    feature_importance: List[FeatureImportance] = field(default_factory=list)
    #: ``"shap"`` (default) or ``"integrated_gradients"`` when using :meth:`DakmaClient.explain_integrated_gradients`.
    attribution_backend: Optional[str] = None
    #: RAM, CPU, and optional GPU stats for the explained inference (see :mod:`dakma_sdk.compute_usage`).
    compute_usage: Optional[Dict[str, Any]] = None
    #: Governance documentation template fields; set via :meth:`dakma_sdk.core.DakmaClient.register_governance` (snapshotted at inference time).
    governance: Optional[Dict[str, Any]] = None
    #: Test/hold-out metrics, split size, etc.; set via :meth:`dakma_sdk.core.DakmaClient.register_evaluation`.
    evaluation: Optional[Dict[str, Any]] = None


@dataclass
class DecisionOutput:
    label: str
    score: float
    threshold: float

    def to_text(self) -> str:
        return f"{self.label} (score: {self.score:.2f}, threshold: {self.threshold:.2f})"


@dataclass
class EnrichedResult:
    decision: Any
    explain: ExplainPayload
    decision_output: DecisionOutput

    def as_dict(self) -> Dict[str, Any]:
        fi = self.explain.feature_importance
        return {
            "decision": self.decision_output.to_text(),
            "audit_trail_id": self.explain.audit_trail_id,
            "model_version": self.explain.model_version,
            "top_factor[0]": self.explain.top_factors[0].to_text() if len(self.explain.top_factors) > 0 else "N/A",
            "top_factor[1]": self.explain.top_factors[1].to_text() if len(self.explain.top_factors) > 1 else "N/A",
            "top_factor[2]": self.explain.top_factors[2].to_text() if len(self.explain.top_factors) > 2 else "N/A",
            "feature_importance[0]": fi[0].to_text() if len(fi) > 0 else "N/A",
            "feature_importance[1]": fi[1].to_text() if len(fi) > 1 else "N/A",
            "feature_importance[2]": fi[2].to_text() if len(fi) > 2 else "N/A",
            "counterfactual": self.explain.counterfactual or "N/A",
            "plain_language": self.explain.plain_language,
            "regulation_flags": " ".join(self.explain.regulation_flags),
        }

    def as_markdown_table(self) -> str:
        from .audit_format import format_inference_result_markdown

        return format_inference_result_markdown(asdict(self))

    def write_report(
        self,
        path: Union[str, Path],
        *,
        title: str = "Decision report",
        format: Literal["markdown", "html"] = "markdown",
    ) -> Path:
        """Write this decision to a downloadable ``.md`` or ``.html`` file."""
        from .audit_format import write_inference_report

        return write_inference_report(asdict(self), path, title=title, format=format)


@dataclass
class Project:
    """Workspace context shared by a family of models, datasets, and decisions."""

    id: str
    regulation: Optional[str] = None
    risk_level: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Model:
    """Model identity recorded for provenance on every decision event."""

    id: str
    version: str
    framework: Optional[str] = None
    model_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    """Dataset identity recorded for provenance (training / reference / evaluation)."""

    id: str
    version: str
    schema_hash: Optional[str] = None
    row_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """Scored outcome of one AI decision event."""

    value: str
    score: float
    threshold: float

    @property
    def label(self) -> str:
        """Alias for :attr:`value`, matching :class:`DecisionOutput`."""
        return self.value

    def to_text(self) -> str:
        return f"{self.value} (score: {self.score:.2f}, threshold: {self.threshold:.2f})"


@dataclass
class Explanation:
    """Attribution and plain-language explanation contributing to a decision event."""

    method: str
    status: str
    factors: List[TopFactor] = field(default_factory=list)
    quality: Optional[Dict[str, Any]] = None
    plain_language: str = ""
    counterfactual: Optional[str] = None
    #: Documentation status notes for reports (not legal or compliance certification).
    regulation_flags: List[str] = field(default_factory=list)
    feature_importance: List[FeatureImportance] = field(default_factory=list)
    compute_usage: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    audit_trail_id: Optional[str] = None
    model_version: Optional[str] = None
    governance: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None

    @property
    def top_factors(self) -> List[TopFactor]:
        """Alias for :attr:`factors`, matching :class:`ExplainPayload`."""
        return self.factors

    @property
    def attribution_backend(self) -> Optional[str]:
        """Alias for :attr:`method`, matching :class:`ExplainPayload`."""
        return self.method


@dataclass
class AuditEvent:
    """Persisted audit record; decisions are stored as ``type="decision"`` events."""

    id: str
    type: str
    timestamp: str
    model: Optional[Model] = None
    dataset: Optional[Dataset] = None
    decision: Optional[Decision] = None
    explanation: Optional[Explanation] = None
    governance: Optional[Dict[str, Any]] = None
    project: Optional[Project] = None
    evaluation: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionEvent:
    """One AI decision plus the evidence that produced it.

    Returned by ``@model.explain()`` and ``@DakmaClient.explain()``. Provenance
    (:class:`Model`, :class:`Dataset`), :class:`Explanation`, and governance all hang off
    this event, and an :class:`AuditEvent` is written to the audit log for it.
    """

    decision: Decision
    explanation: Explanation
    audit_id: str
    model: Optional[Model] = None
    dataset: Optional[Dataset] = None
    project: Optional[Project] = None
    governance: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    raw_output: Any = None

    @property
    def decision_output(self) -> Decision:
        """Alias for :attr:`decision`, matching :class:`EnrichedResult`."""
        return self.decision

    @property
    def explain(self) -> Explanation:
        """Alias for :attr:`explanation`, matching :class:`EnrichedResult`."""
        return self.explanation

    def to_audit_event(self, *, timestamp: str, event_type: str = "decision") -> AuditEvent:
        return AuditEvent(
            id=self.audit_id,
            type=event_type,
            timestamp=timestamp,
            model=self.model,
            dataset=self.dataset,
            decision=self.decision,
            explanation=self.explanation,
            governance=self.governance,
            project=self.project,
            evaluation=self.evaluation,
            metadata={"raw_output": _reportable(self.raw_output)},
        )

    def as_dict(self) -> Dict[str, Any]:
        fi = self.explanation.feature_importance
        factors = self.explanation.factors
        return {
            "decision": self.decision.to_text(),
            "audit_trail_id": self.audit_id,
            "model_version": self.explanation.model_version or "",
            "top_factor[0]": factors[0].to_text() if len(factors) > 0 else "N/A",
            "top_factor[1]": factors[1].to_text() if len(factors) > 1 else "N/A",
            "top_factor[2]": factors[2].to_text() if len(factors) > 2 else "N/A",
            "feature_importance[0]": fi[0].to_text() if len(fi) > 0 else "N/A",
            "feature_importance[1]": fi[1].to_text() if len(fi) > 1 else "N/A",
            "feature_importance[2]": fi[2].to_text() if len(fi) > 2 else "N/A",
            "counterfactual": self.explanation.counterfactual or "N/A",
            "plain_language": self.explanation.plain_language,
            "regulation_flags": " ".join(self.explanation.regulation_flags),
        }

    def to_report_dict(self) -> Dict[str, Any]:
        """Serialize into the layout :mod:`dakma_sdk.audit_format` expects."""
        explanation = asdict(self.explanation)
        explanation["top_factors"] = explanation.get("factors") or []
        explanation["attribution_backend"] = self.explanation.method
        explanation["audit_trail_id"] = self.audit_id
        return {
            "decision": self.raw_output,
            "decision_output": {
                "label": self.decision.value,
                "value": self.decision.value,
                "score": self.decision.score,
                "threshold": self.decision.threshold,
            },
            "explain": explanation,
            "explanation": explanation,
            "audit_id": self.audit_id,
            "model": asdict(self.model) if self.model else None,
            "dataset": asdict(self.dataset) if self.dataset else None,
            "project": asdict(self.project) if self.project else None,
            "governance": self.governance,
            "evaluation": self.evaluation,
        }

    def as_markdown_table(self) -> str:
        from .audit_format import format_inference_result_markdown

        return format_inference_result_markdown(self.to_report_dict())

    def write_report(
        self,
        path: Union[str, Path],
        *,
        title: str = "Decision report",
        format: Literal["markdown", "html"] = "markdown",
    ) -> Path:
        """Write this decision event to a downloadable ``.md`` or ``.html`` file."""
        from .audit_format import write_inference_report

        return write_inference_report(self.to_report_dict(), path, title=title, format=format)


def _reportable(value: Any) -> Any:
    """Convert raw model output into something safe to store in an audit record."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    try:
        return repr(value)[:500]
    except Exception:
        return "<unrepresentable>"
