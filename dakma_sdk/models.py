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
    #: Art. 13-style documentation template fields; set via :meth:`dakma_sdk.core.DakmaClient.register_governance` (snapshotted at inference time).
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
