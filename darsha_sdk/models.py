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
class Project:
    """Workspace / compliance context for a family of models and decisions."""

    id: str
    regulation: Optional[str] = None
    risk_level: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Model:
    """Immutable model identity used for provenance and audit."""

    id: str
    version: str
    framework: Optional[str] = None
    model_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    """Dataset identity used for provenance (training / reference / evaluation)."""

    id: str
    version: str
    schema_hash: Optional[str] = None
    row_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """The scored outcome of an AI decision event."""

    value: str
    score: float
    threshold: float

    @property
    def label(self) -> str:
        """Backward-compatible alias for :attr:`value`."""
        return self.value

    def to_text(self) -> str:
        return f"{self.value} (score: {self.score:.2f}, threshold: {self.threshold:.2f})"


# Backward-compatible name used by older callers / tests.
DecisionOutput = Decision


@dataclass
class Explanation:
    """Attribution / plain-language explanation attached to a decision event."""

    method: str
    status: str
    factors: List[TopFactor] = field(default_factory=list)
    quality: Optional[Dict[str, Any]] = None
    plain_language: str = ""
    counterfactual: Optional[str] = None
    regulation_flags: List[str] = field(default_factory=list)
    feature_importance: List[FeatureImportance] = field(default_factory=list)
    compute_usage: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: Present for report / legacy ``ExplainPayload`` compatibility.
    audit_trail_id: Optional[str] = None
    model_version: Optional[str] = None
    governance: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None

    @property
    def top_factors(self) -> List[TopFactor]:
        """Alias used by legacy report formatters."""
        return self.factors

    @property
    def attribution_backend(self) -> Optional[str]:
        return self.method


# Backward-compatible name used by older callers / tests.
ExplainPayload = Explanation


@dataclass
class AuditEvent:
    """Persisted audit record. Decision events are the primary ``type=\"decision\"`` form."""

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
    """Fundamental runtime object: one AI decision with contributing evidence.

    Returned by ``@model.explain()`` / ``@client.explain()``. Provenance (model,
    dataset), explanation, and governance all hang off this event; an
    :class:`AuditEvent` is written to the audit log underneath.
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

    # --- backward-compatible surface (pre-DecisionEvent API) ---

    @property
    def decision_output(self) -> Decision:
        return self.decision

    @property
    def explain(self) -> Explanation:
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
            metadata={"raw_output": _safe_jsonish(self.raw_output)},
        )

    def as_dict(self) -> Dict[str, Any]:
        """Human-facing summary (legacy ``EnrichedResult.as_dict`` shape)."""
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
        """Serialize for :mod:`darsha_sdk.audit_format` (legacy EnrichedResult layout)."""
        explanation = asdict(self.explanation)
        # Report formatters historically read ``top_factors`` and ``attribution_backend``.
        explanation["top_factors"] = explanation.get("factors") or []
        explanation["attribution_backend"] = self.explanation.method
        explanation["audit_trail_id"] = self.audit_id
        if self.governance is not None:
            explanation["governance"] = self.governance
        if self.evaluation is not None:
            explanation["evaluation"] = self.evaluation
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


# Backward-compatible name used by older callers / tests.
EnrichedResult = DecisionEvent


def _safe_jsonish(value: Any) -> Any:
    """Best-effort conversion for audit metadata (avoid huge / non-serializable blobs)."""
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
        return "<unreprable>"
