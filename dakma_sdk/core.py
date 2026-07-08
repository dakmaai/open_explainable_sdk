from __future__ import annotations

import functools
import inspect
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from .compute_usage import ComputeUsageRecorder
from .models import DecisionOutput, EnrichedResult, ExplainPayload, FeatureImportance, TopFactor


def _safe_import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception:
        return None


def _safe_import_shap():
    try:
        import shap  # type: ignore

        return shap
    except Exception:
        return None


_EU13_GOVERNANCE_KEYS = (
    "intended_use",
    "known_limitations",
    "human_oversight",
    "data_provenance",
    "model_changelog",
)

_DOCUMENTATION_AID_NOTE = "Documentation aid only — not legal or compliance certification"


class DakmaClient:
    def __init__(self, project: str, regulation: str, risk_level: str):
        self.project = project
        self.regulation = regulation
        self.risk_level = risk_level
        self._audit_log: List[Dict[str, Any]] = []
        self._feature_lineage: Dict[str, str] = {}
        self._data_history: List[Dict[str, Any]] = []
        self._training_history: List[Dict[str, Any]] = []
        self._shap_fi_model_id: Optional[int] = None
        self._shap_fi_rows: List[FeatureImportance] = []
        self._ig_fi_model_id: Optional[int] = None
        self._ig_fi_rows: List[FeatureImportance] = []
        self._governance: Dict[str, Any] = {}
        self._evaluation: Dict[str, Any] = {}

    def track_data(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            input_df = self._first_dataframe(args, kwargs)
            before_schema = self._schema_snapshot(input_df) if input_df is not None else {}

            result = fn(*args, **kwargs)

            output_df = result if self._is_dataframe(result) else None
            after_schema = self._schema_snapshot(output_df) if output_df is not None else {}
            lineage = self._extract_feature_lineage(fn)
            self._feature_lineage.update(lineage)

            entry = {
                "type": "data_tracking",
                "function": fn.__name__,
                "timestamp": self._utc_now(),
                "input_schema": before_schema,
                "output_schema": after_schema,
                "lineage_added": lineage,
            }
            self._data_history.append(entry)
            self._audit_log.append(entry)
            return result

        return wrapper

    def track_training(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = self._utc_now()
            with ComputeUsageRecorder() as rec:
                model = fn(*args, **kwargs)
            finished = self._utc_now()
            compute_usage = rec.usage

            model_params = self._extract_model_params(model)
            training_metrics = self._extract_training_metrics(model, args, kwargs)
            entry = {
                "type": "training_tracking",
                "function": fn.__name__,
                "timestamp": finished,
                "started_at": started,
                "finished_at": finished,
                "model_class": model.__class__.__name__,
                "model_params": model_params,
                "metrics": training_metrics,
                "compute_usage": compute_usage,
            }
            self._training_history.append(entry)
            self._audit_log.append(entry)
            return model

        return wrapper

    def register_governance(
        self,
        *,
        intended_use: Optional[str] = None,
        known_limitations: Optional[str] = None,
        human_oversight: Optional[str] = None,
        data_provenance: Optional[str] = None,
        model_changelog: Any = None,
        **extra: Any,
    ) -> None:
        """Record Art. 13-style transparency text for audit and decision reports.

        This is a documentation template only; it does not certify regulatory compliance.
        Unknown keyword arguments are merged into the governance record. Empty values are
        ignored. Documentation appears in :meth:`write_audit_report` and in decision reports for
        subsequent explained inferences.
        """
        for key, value in (
            ("intended_use", intended_use),
            ("known_limitations", known_limitations),
            ("human_oversight", human_oversight),
            ("data_provenance", data_provenance),
            ("model_changelog", model_changelog),
        ):
            if value is not None and value != "":
                self._governance[key] = value
        for k, v in extra.items():
            if v is not None and v != "":
                self._governance[k] = v

    def register_evaluation(
        self,
        *,
        n_train: Optional[int] = None,
        n_test: Optional[int] = None,
        split_description: Optional[str] = None,
        test_metrics: Optional[Dict[str, Any]] = None,
        confusion_matrix: Any = None,
        **extra: Any,
    ) -> None:
        """Register hold-out / test evaluation and dataset split information for audit reports.

        Use ``test_metrics`` for test-set precision, recall, F1, ROC-AUC, etc. (not training
        fit scores). ``confusion_matrix`` may be a 2×2 list (e.g. ``[[tn, fp], [fn, tp]]``) or
        a mapping. Extra keys are merged into the evaluation record.
        """
        if n_train is not None:
            self._evaluation["n_train"] = n_train
        if n_test is not None:
            self._evaluation["n_test"] = n_test
        if split_description is not None and split_description != "":
            self._evaluation["split_description"] = split_description
        if test_metrics is not None:
            self._evaluation["test_metrics"] = {**self._evaluation.get("test_metrics", {}), **test_metrics}
        if confusion_matrix is not None:
            self._evaluation["confusion_matrix"] = confusion_matrix
        for k, v in extra.items():
            if v is not None and v != "":
                self._evaluation[k] = v

    def explain(
        self,
        *,
        counterfactual: bool = False,
        threshold: float = 0.5,
        top_k: int = 3,
    ) -> Callable[[Callable[..., Any]], Callable[..., EnrichedResult]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., EnrichedResult]:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> EnrichedResult:
                with ComputeUsageRecorder() as rec:
                    raw_decision = fn(*args, **kwargs)
                    model = self._first_model(args, kwargs)
                    row = self._first_dataframe_or_mapping(args, kwargs)

                    score = self._extract_score(raw_decision)
                    decision_output = self._to_decision_output(score, threshold)
                    top_factors = self._top_factors(model, row, top_k=top_k)
                    counterfactual_text = self._counterfactual(top_factors) if counterfactual else None

                    fi_rows: List[FeatureImportance] = []
                    if self._shap_fi_model_id is not None and model is not None and id(model) == self._shap_fi_model_id:
                        fi_rows = list(self._shap_fi_rows)

                compute_usage = rec.usage

                explain_payload = ExplainPayload(
                    audit_trail_id=self._audit_trail_id(),
                    model_version=self._model_version(model),
                    plain_language=self._plain_language(decision_output, top_factors),
                    top_factors=top_factors,
                    counterfactual=counterfactual_text,
                    regulation_flags=self._regulation_flags(),
                    metadata={
                        "project": self.project,
                        "regulation": self.regulation,
                        "risk_level": self.risk_level,
                        "feature_lineage": self._feature_lineage,
                    },
                    feature_importance=fi_rows,
                    attribution_backend="shap",
                    compute_usage=compute_usage,
                    governance=(dict(self._governance) if self._governance else None),
                    evaluation=(dict(self._evaluation) if self._evaluation else None),
                )
                result = EnrichedResult(
                    decision=raw_decision,
                    explain=explain_payload,
                    decision_output=decision_output,
                )

                self._audit_log.append(
                    {
                        "type": "inference_explain",
                        "timestamp": self._utc_now(),
                        "function": fn.__name__,
                        "result": asdict(result),
                    }
                )
                return result

            return wrapper

        return decorator

    def register_shap_feature_importance(
        self,
        model: Any,
        X_reference: Any,
        *,
        max_samples: int = 500,
        top_k: Optional[int] = None,
    ) -> None:
        """Compute mean |SHAP| per feature over ``X_reference`` and attach it to later inference reports.

        Call after training with a background matrix (e.g. training or validation rows). Values are
        included on :class:`~dakma_sdk.models.EnrichedResult` only when the same ``model`` object is
        passed into the explained function. Requires optional dependencies ``shap`` and ``pandas``.
        """
        rows = self._global_shap_feature_importance(model, X_reference, max_samples=max_samples, top_k=top_k)
        self._shap_fi_rows = rows
        self._shap_fi_model_id = id(model) if model is not None and rows else None

    def explain_integrated_gradients(
        self,
        *,
        counterfactual: bool = False,
        threshold: float = 0.5,
        top_k: int = 3,
        n_steps: int = 32,
        feature_names: Optional[Sequence[str]] = None,
        baseline: Any = None,
        target_class: Optional[int] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., EnrichedResult]]:
        """Explain inference using Integrated Gradients (PyTorch ``nn.Module`` + tensor inputs).

        The wrapped function should return model outputs (logits or probabilities). Pass the trained
        ``model`` and input batch ``x`` (``torch.Tensor``) as arguments — convention:
        ``fn(model, x_tensor, ...)``. Requires ``torch``; optional ``captum`` for a fast IG
        implementation (``pip install dakma-sdk[dl]``).

        Global feature rankings can be precomputed with
        :meth:`register_integrated_gradients_feature_importance` using the same ``model`` object.
        """
        def decorator(fn: Callable[..., Any]) -> Callable[..., EnrichedResult]:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> EnrichedResult:
                with ComputeUsageRecorder() as rec:
                    raw_decision = fn(*args, **kwargs)
                    model = self._first_pytorch_module(args, kwargs)
                    x_tensor = self._first_torch_tensor(args, kwargs)

                    score = self._extract_score(raw_decision)
                    decision_output = self._to_decision_output(score, threshold)
                    top_factors = self._top_factors_integrated_gradients(
                        model,
                        x_tensor,
                        feature_names=feature_names,
                        top_k=top_k,
                        n_steps=n_steps,
                        baseline=baseline,
                        target_class=target_class,
                    )
                    counterfactual_text = self._counterfactual(top_factors) if counterfactual else None

                    fi_rows: List[FeatureImportance] = []
                    if self._ig_fi_model_id is not None and model is not None and id(model) == self._ig_fi_model_id:
                        fi_rows = list(self._ig_fi_rows)

                compute_usage = rec.usage

                explain_payload = ExplainPayload(
                    audit_trail_id=self._audit_trail_id(),
                    model_version=self._model_version(model),
                    plain_language=self._plain_language(decision_output, top_factors),
                    top_factors=top_factors,
                    counterfactual=counterfactual_text,
                    regulation_flags=self._regulation_flags(),
                    metadata={
                        "project": self.project,
                        "regulation": self.regulation,
                        "risk_level": self.risk_level,
                        "feature_lineage": self._feature_lineage,
                    },
                    feature_importance=fi_rows,
                    attribution_backend="integrated_gradients",
                    compute_usage=compute_usage,
                    governance=(dict(self._governance) if self._governance else None),
                    evaluation=(dict(self._evaluation) if self._evaluation else None),
                )
                result = EnrichedResult(
                    decision=raw_decision,
                    explain=explain_payload,
                    decision_output=decision_output,
                )

                self._audit_log.append(
                    {
                        "type": "inference_explain",
                        "timestamp": self._utc_now(),
                        "function": fn.__name__,
                        "result": asdict(result),
                    }
                )
                return result

            return wrapper

        return decorator

    def register_integrated_gradients_feature_importance(
        self,
        model: Any,
        X_reference: Any,
        *,
        max_samples: int = 200,
        top_k: Optional[int] = None,
        n_steps: int = 32,
        baseline: Any = None,
        target_class: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Mean |Integrated Gradients| per feature over a reference tensor (tabular / channel-wise).

        Same pattern as :meth:`register_shap_feature_importance`: values appear on reports when the
        same ``model`` instance is used with :meth:`explain_integrated_gradients`.
        """
        rows = self._global_integrated_gradients_importance(
            model,
            X_reference,
            max_samples=max_samples,
            top_k=top_k,
            n_steps=n_steps,
            baseline=baseline,
            target_class=target_class,
            feature_names=feature_names,
        )
        self._ig_fi_rows = rows
        self._ig_fi_model_id = id(model) if model is not None and rows else None

    def monitor(
        self,
        *,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        protected_feature: Optional[Sequence[Any]] = None,
        expected_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        accuracy = float((y_true_arr == y_pred_arr).mean())
        pred_positive_rate = float(y_pred_arr.mean()) if len(y_pred_arr) else 0.0
        drift = None
        if expected_rate is not None:
            drift = abs(pred_positive_rate - expected_rate)

        bias_metrics = {}
        if protected_feature is not None and len(protected_feature) == len(y_pred_arr):
            groups = {}
            for group, pred in zip(protected_feature, y_pred_arr):
                groups.setdefault(str(group), []).append(int(pred))
            bias_metrics = {g: float(np.mean(v)) for g, v in groups.items()}

        payload = {
            "accuracy": accuracy,
            "positive_prediction_rate": pred_positive_rate,
            "drift_from_expected_rate": drift,
            "group_positive_rates": bias_metrics,
        }
        self._audit_log.append({"type": "monitoring", "timestamp": self._utc_now(), "payload": payload})
        return payload

    def export_audit_log(self) -> List[Dict[str, Any]]:
        return list(self._audit_log)

    def format_audit_log_markdown(self, *, title: str = "Audit trail") -> str:
        """Return the full audit log as readable Markdown tables (per entry type)."""
        from .audit_format import format_audit_log_markdown

        return format_audit_log_markdown(
            self._audit_log,
            title=title,
            governance=self._governance,
            evaluation=self._evaluation,
        )

    def write_audit_report(
        self,
        path: Union[str, Path],
        *,
        title: str = "Audit trail",
        format: Literal["markdown", "html"] = "markdown",
    ) -> Path:
        """Write the full audit log to a downloadable ``.md`` or ``.html`` file (UTF-8)."""
        from .audit_format import write_audit_report as _write_audit_report

        return _write_audit_report(
            self._audit_log,
            path,
            title=title,
            format=format,
            governance=self._governance,
            evaluation=self._evaluation,
        )

    def last_data_report(self) -> Optional[Dict[str, Any]]:
        return self._data_history[-1] if self._data_history else None

    def last_training_report(self) -> Optional[Dict[str, Any]]:
        return self._training_history[-1] if self._training_history else None

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _is_dataframe(value: Any) -> bool:
        pd = _safe_import_pandas()
        return pd is not None and isinstance(value, pd.DataFrame)

    def _first_dataframe(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        pd = _safe_import_pandas()
        if pd is None:
            return None
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                return arg
        for val in kwargs.values():
            if isinstance(val, pd.DataFrame):
                return val
        return None

    def _schema_snapshot(self, df: Any) -> Dict[str, Dict[str, Any]]:
        if df is None:
            return {}
        return {
            col: {
                "dtype": str(df[col].dtype),
                "nulls": int(df[col].isna().sum()),
            }
            for col in df.columns
        }

    def _extract_feature_lineage(self, fn: Callable[..., Any]) -> Dict[str, str]:
        source = inspect.getsource(fn)
        pattern = r"""df\["([^"]+)"\]\s*=\s*(.+)"""
        found = re.findall(pattern, source)
        lineage = {}
        for feature, expression in found:
            lineage[feature] = expression.strip()
        return lineage

    @staticmethod
    def _extract_model_params(model: Any) -> Dict[str, Any]:
        if hasattr(model, "get_params"):
            try:
                return model.get_params()
            except Exception:
                return {}
        return {}

    @staticmethod
    def _extract_training_metrics(model: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, float]:
        metrics = {}
        if hasattr(model, "score"):
            X_train = kwargs.get("X_train")
            if X_train is None and len(args) > 0:
                X_train = args[0]
            y_train = kwargs.get("y_train")
            if y_train is None and len(args) > 1:
                y_train = args[1]
            if X_train is not None and y_train is not None:
                try:
                    metrics["training_score"] = float(model.score(X_train, y_train))
                except Exception:
                    pass
        return metrics

    @staticmethod
    def _first_model(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        for arg in args:
            if hasattr(arg, "predict") or hasattr(arg, "predict_proba") or hasattr(arg, "get_params"):
                return arg
        for value in kwargs.values():
            if hasattr(value, "predict") or hasattr(value, "predict_proba") or hasattr(value, "get_params"):
                return value
        return None

    @staticmethod
    def _first_pytorch_module(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        try:
            import torch.nn as nn  # type: ignore
        except Exception:
            return None
        for arg in args:
            if isinstance(arg, nn.Module):
                return arg
        for value in kwargs.values():
            if isinstance(value, nn.Module):
                return value
        return None

    @staticmethod
    def _first_torch_tensor(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        try:
            import torch  # type: ignore
        except Exception:
            return None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                return arg
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                return value
        return None

    def _first_dataframe_or_mapping(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        df = self._first_dataframe(args, kwargs)
        if df is not None:
            return df
        for arg in args:
            if isinstance(arg, dict):
                return arg
        for value in kwargs.values():
            if isinstance(value, dict):
                return value
        return None

    @staticmethod
    def _extract_score(raw_decision: Any) -> float:
        try:
            import torch  # type: ignore

            if isinstance(raw_decision, torch.Tensor):
                raw_decision = raw_decision.detach().cpu().numpy()
        except Exception:
            pass
        arr = np.asarray(raw_decision)
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1:
            return float(arr[-1])
        if arr.ndim >= 2:
            return float(arr[0, -1])
        return 0.0

    @staticmethod
    def _to_decision_output(score: float, threshold: float) -> DecisionOutput:
        if score >= threshold:
            return DecisionOutput(label="APPROVED", score=score, threshold=threshold)
        return DecisionOutput(label="DECLINED", score=score, threshold=threshold)

    def _top_factors(self, model: Any, row: Any, top_k: int = 3) -> List[TopFactor]:
        factors = self._top_factors_shap(model, row)
        if not factors:
            factors = self._top_factors_fallback(model, row)
        return factors[:top_k]

    def _reference_dataframe(self, X_reference: Any) -> Optional[Any]:
        pd = _safe_import_pandas()
        if pd is None:
            return None
        if isinstance(X_reference, pd.DataFrame):
            return X_reference
        if isinstance(X_reference, dict):
            return pd.DataFrame([X_reference])
        if isinstance(X_reference, np.ndarray):
            arr = X_reference
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                return None
            cols = [f"f{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols)
        return None

    def _global_shap_feature_importance(
        self,
        model: Any,
        X_reference: Any,
        *,
        max_samples: int,
        top_k: Optional[int],
    ) -> List[FeatureImportance]:
        shap = _safe_import_shap()
        row_df = self._reference_dataframe(X_reference)
        if shap is None or model is None or row_df is None or len(row_df) == 0:
            return []
        try:
            if len(row_df) > max_samples:
                row_df = row_df.sample(n=max_samples, random_state=0)
            # Pass the same reference rows as the masker so sklearn / non-tree models work with SHAP.
            explanation = shap.Explainer(model, row_df)(row_df)
            vals = np.asarray(explanation.values, dtype=float)
            if vals.ndim == 1:
                vals = vals.reshape(1, -1)
            if vals.ndim == 3:
                mean_abs = np.mean(np.abs(vals), axis=(0, 2))
            elif vals.ndim == 2:
                mean_abs = np.mean(np.abs(vals), axis=0)
            else:
                return []
            features = list(row_df.columns)
            if mean_abs.shape[0] != len(features):
                return []
            pairs = sorted(zip(features, mean_abs.tolist()), key=lambda x: x[1], reverse=True)
            if top_k is not None:
                pairs = pairs[: int(top_k)]
            return [FeatureImportance(name=str(n), mean_abs_shap=float(v), source="shap") for n, v in pairs]
        except Exception:
            return []

    def _global_integrated_gradients_importance(
        self,
        model: Any,
        X_reference: Any,
        *,
        max_samples: int,
        top_k: Optional[int],
        n_steps: int,
        baseline: Any,
        target_class: Optional[int],
        feature_names: Optional[Sequence[str]],
    ) -> List[FeatureImportance]:
        try:
            import torch  # type: ignore

            from .integrated_gradients import (
                compute_integrated_gradients_attributions,
                reduce_attributions_to_features,
            )
        except Exception:
            return []

        if model is None:
            return []
        try:
            if isinstance(X_reference, torch.Tensor):
                x_all = X_reference.float()
            else:
                x_all = torch.as_tensor(np.asarray(X_reference), dtype=torch.float32)
            if x_all.dim() < 2:
                return []
            n = min(int(x_all.shape[0]), int(max_samples))
            if n <= 0:
                return []
            mean_abs: Optional[np.ndarray] = None
            names: Optional[List[str]] = None
            for i in range(n):
                row = x_all[i : i + 1]
                attr = compute_integrated_gradients_attributions(
                    model,
                    row,
                    baseline=baseline,
                    n_steps=n_steps,
                    target_class=target_class,
                )
                nms, _, impacts = reduce_attributions_to_features(attr, feature_names=feature_names)
                contrib = np.abs(np.asarray(impacts, dtype=float))
                if mean_abs is None:
                    mean_abs = contrib
                    names = nms
                elif contrib.shape == mean_abs.shape:
                    mean_abs = mean_abs + contrib
                else:
                    return []
            assert mean_abs is not None and names is not None
            mean_abs = mean_abs / float(n)
            pairs = sorted(zip(names, mean_abs.tolist()), key=lambda x: x[1], reverse=True)
            if top_k is not None:
                pairs = pairs[: int(top_k)]
            return [
                FeatureImportance(name=str(a), mean_abs_shap=float(b), source="integrated_gradients")
                for a, b in pairs
            ]
        except Exception:
            return []

    def _top_factors_integrated_gradients(
        self,
        model: Any,
        x_tensor: Any,
        *,
        feature_names: Optional[Sequence[str]],
        top_k: int,
        n_steps: int,
        baseline: Any,
        target_class: Optional[int],
    ) -> List[TopFactor]:
        if model is None or x_tensor is None:
            return []
        try:
            from .integrated_gradients import (
                compute_integrated_gradients_attributions,
                input_values_for_display,
                reduce_attributions_to_features,
            )

            attr = compute_integrated_gradients_attributions(
                model,
                x_tensor,
                baseline=baseline,
                n_steps=n_steps,
                target_class=target_class,
            )
            names, _, impacts = reduce_attributions_to_features(attr, feature_names=feature_names)
            vals = input_values_for_display(x_tensor, len(names))
            sortable = list(zip(names, vals, impacts))
            sortable.sort(key=lambda x: abs(x[2]), reverse=True)
            out: List[TopFactor] = []
            for name, val, impact in sortable[:top_k]:
                out.append(
                    TopFactor(
                        name=name,
                        value=float(val),
                        impact=float(impact),
                        direction="up" if impact >= 0 else "down",
                    )
                )
            return out
        except Exception:
            return []

    def _top_factors_shap(self, model: Any, row: Any) -> List[TopFactor]:
        shap = _safe_import_shap()
        pd = _safe_import_pandas()
        if shap is None or model is None or row is None or pd is None:
            return []

        try:
            row_df = row if isinstance(row, pd.DataFrame) else pd.DataFrame([row])
            explainer = shap.Explainer(model)
            values = explainer(row_df)
            shap_vals = np.asarray(values.values[0])
            features = list(row_df.columns)
            sortable = list(zip(features, row_df.iloc[0].tolist(), shap_vals))
            sortable.sort(key=lambda x: abs(x[2]), reverse=True)
            result = []
            for name, val, impact in sortable[:3]:
                direction = "up" if impact >= 0 else "down"
                result.append(TopFactor(name=name, value=val, impact=float(impact), direction=direction))
            return result
        except Exception:
            return []

    def _top_factors_fallback(self, model: Any, row: Any) -> List[TopFactor]:
        pd = _safe_import_pandas()
        if row is None:
            return []
        if pd is not None and isinstance(row, pd.DataFrame):
            features = list(row.columns)
            values = row.iloc[0].tolist()
        elif isinstance(row, dict):
            features = list(row.keys())
            values = list(row.values())
        else:
            return []

        impacts = np.zeros(len(features), dtype=float)
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_).reshape(-1)
            if len(coef) == len(features):
                impacts = coef * np.asarray(values, dtype=float)
        else:
            impacts = np.asarray(values, dtype=float) * 0.01

        sortable = list(zip(features, values, impacts))
        sortable.sort(key=lambda x: abs(x[2]), reverse=True)
        top = []
        for name, val, impact in sortable[:3]:
            top.append(TopFactor(name=name, value=val, impact=float(impact), direction="up" if impact >= 0 else "down"))
        return top

    @staticmethod
    def _counterfactual(factors: List[TopFactor]) -> Optional[str]:
        if not factors:
            return None
        dominant = factors[0]
        if dominant.impact < 0:
            try:
                target = float(dominant.value) * 0.7
                return f"Would approve if {dominant.name} < {target:.2f}"
            except Exception:
                return f"Would approve if {dominant.name} is improved"
        return f"Would decline if {dominant.name} worsens further"

    def _model_version(self, model: Any) -> str:
        if model is None:
            return f"{self.project} / unknown-model"
        try:
            import torch.nn as nn  # type: ignore

            if isinstance(model, nn.Module):
                return f"{self.project} / torch-{model.__class__.__name__.lower()}"
        except Exception:
            pass
        params = self._extract_model_params(model)
        n_estimators = params.get("n_estimators")
        max_depth = params.get("max_depth")
        suffix = f"{model.__class__.__name__.lower()}"
        if n_estimators is not None and max_depth is not None:
            suffix = f"xgb-{n_estimators}-d{max_depth}"
        return f"{self.project} / {suffix}"

    def _plain_language(self, decision: DecisionOutput, factors: List[TopFactor]) -> str:
        if not factors:
            return f"{decision.label.title()}. Decision generated by model score and threshold."
        reason = factors[0]
        direction = "high" if isinstance(reason.value, (int, float)) and float(reason.value) > 0 else "low"
        return (
            f"{decision.label.title()}. Main reason: {direction} {reason.name.replace('_', ' ')} "
            f"which moved the score {'down' if reason.impact < 0 else 'up'}."
        )

    def _regulation_flags(self) -> List[str]:
        """Human-readable documentation status notes (not compliance certification)."""
        norm = self.regulation.lower().strip()
        if norm == "eu-ai-act":
            populated = sum(
                1 for key in _EU13_GOVERNANCE_KEYS if self._governance.get(key) not in (None, "")
            )
            total = len(_EU13_GOVERNANCE_KEYS)
            return [
                f"EU AI Act Art. 13 documentation fields: {populated}/{total} populated",
                _DOCUMENTATION_AID_NOTE,
            ]
        return [
            f"Regulation context: {self.regulation}",
            _DOCUMENTATION_AID_NOTE,
        ]

    @staticmethod
    def _audit_trail_id() -> str:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        token = uuid.uuid4().hex[:7]
        return f"ax-{date}-{token}"
