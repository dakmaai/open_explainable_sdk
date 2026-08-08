from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import pickle
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from .compute_usage import ComputeUsageRecorder
from .models import (
    AuditEvent,
    Dataset,
    Decision,
    DecisionEvent,
    DecisionOutput,
    Explanation,
    FeatureImportance,
    Model,
    Project,
    TopFactor,
)

logger = logging.getLogger(__name__)


def _safe_import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception:
        logger.debug("pandas not available; DataFrame features disabled", exc_info=True)
        return None


def _safe_import_shap():
    try:
        import shap  # type: ignore

        return shap
    except Exception:
        logger.debug("shap not available; SHAP explanations disabled", exc_info=True)
        return None


def _sha256_hex(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _hash_model_artifact(artifact: Any) -> Optional[str]:
    if artifact is None:
        return None
    try:
        return _sha256_hex(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        logger.debug("Could not pickle model artifact for hashing", exc_info=True)
        try:
            return _sha256_hex(repr(artifact).encode("utf-8", errors="replace"))
        except Exception:
            return None


def _infer_framework(artifact: Any) -> Optional[str]:
    if artifact is None:
        return None
    try:
        import torch.nn as nn  # type: ignore

        if isinstance(artifact, nn.Module):
            return "pytorch"
    except Exception:
        pass
    module = getattr(type(artifact), "__module__", "") or ""
    name = type(artifact).__name__.lower()
    if "xgboost" in module or name.startswith("xgb"):
        return "xgboost"
    if "sklearn" in module or "scikit" in module:
        return "sklearn"
    if module:
        return module.split(".", 1)[0]
    return None


class ModelHandle:
    """Project-scoped model identity with ``@handle.explain()`` for DecisionEvents."""

    def __init__(
        self,
        client: "DarshaClient",
        *,
        name: str,
        version: str,
        framework: Optional[str] = None,
        model_hash: Optional[str] = None,
        artifact: Any = None,
        dataset: Optional[Dataset] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._client = client
        self._artifact = artifact
        self._dataset = dataset
        resolved_framework = framework or _infer_framework(artifact)
        resolved_hash = model_hash or _hash_model_artifact(artifact)
        self.spec = Model(
            id=name,
            version=version,
            framework=resolved_framework,
            model_hash=resolved_hash,
            metadata=dict(metadata or {}),
        )

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def version(self) -> str:
        return self.spec.version

    @property
    def artifact(self) -> Any:
        return self._artifact

    def bind(self, artifact: Any, *, rehash: bool = True) -> "ModelHandle":
        """Attach the runtime model object used for scoring / attribution."""
        self._artifact = artifact
        if self.spec.framework is None:
            self.spec.framework = _infer_framework(artifact)
        if rehash or not self.spec.model_hash:
            self.spec.model_hash = _hash_model_artifact(artifact)
        return self

    def use_dataset(self, dataset: Dataset) -> "ModelHandle":
        self._dataset = dataset
        return self

    def explain(
        self,
        *,
        counterfactual: bool = False,
        threshold: float = 0.5,
        top_k: int = 3,
    ) -> Callable[[Callable[..., Any]], Callable[..., DecisionEvent]]:
        """Decorate an inference function; returns a :class:`DecisionEvent`."""
        return self._client._explain_with_handle(
            self,
            backend="shap",
            counterfactual=counterfactual,
            threshold=threshold,
            top_k=top_k,
        )

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
    ) -> Callable[[Callable[..., Any]], Callable[..., DecisionEvent]]:
        return self._client._explain_with_handle(
            self,
            backend="integrated_gradients",
            counterfactual=counterfactual,
            threshold=threshold,
            top_k=top_k,
            n_steps=n_steps,
            feature_names=feature_names,
            baseline=baseline,
            target_class=target_class,
        )


class DarshaClient:
    def __init__(
        self,
        project: str,
        regulation: str = "",
        risk_level: str = "",
    ):
        self.project = project
        self.regulation = regulation
        self.risk_level = risk_level
        self.project_info = Project(
            id=project,
            regulation=regulation or None,
            risk_level=risk_level or None,
        )
        self._audit_log: List[Dict[str, Any]] = []
        self._feature_lineage: Dict[str, str] = {}
        self._data_history: List[Dict[str, Any]] = []
        self._training_history: List[Dict[str, Any]] = []
        self._models: Dict[str, ModelHandle] = {}
        self._datasets: Dict[str, Dataset] = {}
        self._default_dataset: Optional[Dataset] = None
        self._shap_fi_model_id: Optional[int] = None
        self._shap_fi_rows: List[FeatureImportance] = []
        self._ig_fi_model_id: Optional[int] = None
        self._ig_fi_rows: List[FeatureImportance] = []
        self._governance: Dict[str, Any] = {}
        self._evaluation: Dict[str, Any] = {}
        logger.debug(
            "DarshaClient initialized (project=%s, regulation=%s, risk_level=%s)",
            project,
            regulation,
            risk_level,
        )

    def model(
        self,
        name: str,
        version: str = "0",
        *,
        framework: Optional[str] = None,
        model_hash: Optional[str] = None,
        artifact: Any = None,
        dataset: Optional[Union[str, Dataset]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelHandle:
        """Register (or fetch) a typed :class:`Model` handle for DecisionEvents."""
        bound_dataset: Optional[Dataset] = None
        if isinstance(dataset, Dataset):
            bound_dataset = dataset
        elif isinstance(dataset, str) and dataset in self._datasets:
            bound_dataset = self._datasets[dataset]
        elif self._default_dataset is not None:
            bound_dataset = self._default_dataset

        key = f"{name}@{version}"
        handle = ModelHandle(
            self,
            name=name,
            version=version,
            framework=framework,
            model_hash=model_hash,
            artifact=artifact,
            dataset=bound_dataset,
            metadata=metadata,
        )
        self._models[key] = handle
        logger.debug("model: registered %s (framework=%s)", key, handle.spec.framework)
        return handle

    def dataset(
        self,
        id: str,
        version: str = "0",
        *,
        schema_hash: Optional[str] = None,
        row_count: Optional[int] = None,
        data: Any = None,
        set_default: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dataset:
        """Register a typed :class:`Dataset` for provenance on DecisionEvents."""
        resolved_rows = row_count
        resolved_hash = schema_hash
        if data is not None:
            if resolved_rows is None:
                try:
                    resolved_rows = int(len(data))
                except Exception:
                    resolved_rows = None
            if resolved_hash is None:
                resolved_hash = self._schema_hash_from_data(data)
        ds = Dataset(
            id=id,
            version=version,
            schema_hash=resolved_hash,
            row_count=resolved_rows,
            metadata=dict(metadata or {}),
        )
        self._datasets[id] = ds
        if set_default:
            self._default_dataset = ds
        logger.debug("dataset: registered %s@version=%s rows=%s", id, version, resolved_rows)
        return ds

    def track_data(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.debug("track_data: running feature function '%s'", fn.__name__)
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
            logger.info(
                "track_data: '%s' tracked (input_cols=%d, output_cols=%d, lineage_added=%d)",
                fn.__name__,
                len(before_schema),
                len(after_schema),
                len(lineage),
            )
            if input_df is not None and not before_schema:
                logger.debug("track_data: input DataFrame had no columns to snapshot")
            return result

        return wrapper

    def track_training(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.info("track_training: starting training function '%s'", fn.__name__)
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
            wall_ms = (compute_usage or {}).get("wall_time_ms")
            logger.info(
                "track_training: '%s' finished (model=%s, params=%d, metrics=%s, wall_time_ms=%s)",
                fn.__name__,
                model.__class__.__name__,
                len(model_params),
                training_metrics or "none",
                wall_ms if wall_ms is not None else "n/a",
            )
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
        """Declare EU AI Act Art. 13 style transparency text for audit and decision reports.

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
        logger.debug("register_governance: %d governance field(s) now set", len(self._governance))

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
        logger.debug(
            "register_evaluation: n_train=%s, n_test=%s, test_metrics=%d",
            self._evaluation.get("n_train"),
            self._evaluation.get("n_test"),
            len(self._evaluation.get("test_metrics", {})),
        )

    def explain(
        self,
        *,
        counterfactual: bool = False,
        threshold: float = 0.5,
        top_k: int = 3,
    ) -> Callable[[Callable[..., Any]], Callable[..., DecisionEvent]]:
        """Decorate inference; returns a :class:`DecisionEvent` (legacy client-level API)."""
        return self._explain_with_handle(
            None,
            backend="shap",
            counterfactual=counterfactual,
            threshold=threshold,
            top_k=top_k,
        )

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
        included on :class:`~darsha_sdk.models.DecisionEvent` only when the same ``model`` object is
        passed into the explained function. Requires optional dependencies ``shap`` and ``pandas``.
        """
        rows = self._global_shap_feature_importance(model, X_reference, max_samples=max_samples, top_k=top_k)
        self._shap_fi_rows = rows
        self._shap_fi_model_id = id(model) if model is not None and rows else None
        if rows:
            logger.info("register_shap_feature_importance: computed global importance for %d feature(s)", len(rows))
        else:
            logger.warning(
                "register_shap_feature_importance: no SHAP importance computed "
                "(check that shap/pandas are installed and the model is supported)"
            )

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
    ) -> Callable[[Callable[..., Any]], Callable[..., DecisionEvent]]:
        """Explain inference using Integrated Gradients (PyTorch ``nn.Module`` + tensor inputs).

        The wrapped function should return model outputs (logits or probabilities). Pass the trained
        ``model`` and input batch ``x`` (``torch.Tensor``) as arguments — convention:
        ``fn(model, x_tensor, ...)``. Requires ``torch``; optional ``captum`` for a fast IG
        implementation (``pip install darsha[dl]``).

        Global feature rankings can be precomputed with
        :meth:`register_integrated_gradients_feature_importance` using the same ``model`` object.
        """
        return self._explain_with_handle(
            None,
            backend="integrated_gradients",
            counterfactual=counterfactual,
            threshold=threshold,
            top_k=top_k,
            n_steps=n_steps,
            feature_names=feature_names,
            baseline=baseline,
            target_class=target_class,
        )

    def _explain_with_handle(
        self,
        handle: Optional[ModelHandle],
        *,
        backend: Literal["shap", "integrated_gradients"],
        counterfactual: bool = False,
        threshold: float = 0.5,
        top_k: int = 3,
        n_steps: int = 32,
        feature_names: Optional[Sequence[str]] = None,
        baseline: Any = None,
        target_class: Optional[int] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., DecisionEvent]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., DecisionEvent]:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> DecisionEvent:
                logger.debug(
                    "explain: running inference function '%s' (backend=%s)",
                    fn.__name__,
                    backend,
                )
                with ComputeUsageRecorder() as rec:
                    raw_decision = fn(*args, **kwargs)
                    if backend == "integrated_gradients":
                        model = (
                            handle.artifact
                            if handle is not None and handle.artifact is not None
                            else self._first_pytorch_module(args, kwargs)
                        )
                        x_tensor = self._first_torch_tensor(args, kwargs)
                        row = None
                        if model is None:
                            logger.warning(
                                "explain_integrated_gradients: no torch.nn.Module detected for '%s'; "
                                "attributions will be empty",
                                fn.__name__,
                            )
                        if x_tensor is None:
                            logger.warning(
                                "explain_integrated_gradients: no torch.Tensor input detected for '%s'; "
                                "attributions will be empty",
                                fn.__name__,
                            )
                        top_factors = self._top_factors_integrated_gradients(
                            model,
                            x_tensor,
                            feature_names=feature_names,
                            top_k=top_k,
                            n_steps=n_steps,
                            baseline=baseline,
                            target_class=target_class,
                        )
                        fi_rows: List[FeatureImportance] = []
                        if (
                            self._ig_fi_model_id is not None
                            and model is not None
                            and id(model) == self._ig_fi_model_id
                        ):
                            fi_rows = list(self._ig_fi_rows)
                    else:
                        model = (
                            handle.artifact
                            if handle is not None and handle.artifact is not None
                            else self._first_model(args, kwargs)
                        )
                        row = self._first_dataframe_or_mapping(args, kwargs)
                        if model is None:
                            logger.warning(
                                "explain: no model argument detected for '%s'; "
                                "model_version and SHAP attributions will be limited",
                                fn.__name__,
                            )
                        if row is None:
                            logger.warning(
                                "explain: no input row (DataFrame/dict) detected for '%s'; "
                                "per-decision factors cannot be computed",
                                fn.__name__,
                            )
                        top_factors = self._top_factors(model, row, top_k=top_k)
                        fi_rows = []
                        if (
                            self._shap_fi_model_id is not None
                            and model is not None
                            and id(model) == self._shap_fi_model_id
                        ):
                            fi_rows = list(self._shap_fi_rows)

                    score = self._extract_score(raw_decision)
                    decision = self._to_decision_output(score, threshold)
                    counterfactual_text = self._counterfactual(top_factors) if counterfactual else None

                compute_usage = rec.usage
                event = self._build_decision_event(
                    handle=handle,
                    model_artifact=model,
                    decision=decision,
                    factors=top_factors,
                    method=backend,
                    counterfactual_text=counterfactual_text,
                    feature_importance=fi_rows,
                    compute_usage=compute_usage,
                    raw_output=raw_decision,
                )
                self._record_decision_event(event, function_name=fn.__name__)
                logger.info(
                    "explain: %s decision (score=%.4f, threshold=%.2f, factors=%d, audit_id=%s)",
                    decision.value,
                    decision.score,
                    decision.threshold,
                    len(top_factors),
                    event.audit_id,
                )
                return event

            return wrapper

        return decorator

    def _build_decision_event(
        self,
        *,
        handle: Optional[ModelHandle],
        model_artifact: Any,
        decision: Decision,
        factors: List[TopFactor],
        method: str,
        counterfactual_text: Optional[str],
        feature_importance: List[FeatureImportance],
        compute_usage: Optional[Dict[str, Any]],
        raw_output: Any,
    ) -> DecisionEvent:
        audit_id = self._audit_trail_id()
        model_spec = handle.spec if handle is not None else self._model_spec_from_artifact(model_artifact)
        dataset = None
        if handle is not None:
            dataset = handle._dataset
        if dataset is None:
            dataset = self._default_dataset

        status = "success" if factors else "degraded"
        quality = {
            "factor_count": len(factors),
            "has_global_importance": bool(feature_importance),
            "method": method,
        }
        model_version = (
            f"{model_spec.id}:{model_spec.version}"
            if handle is not None
            else self._model_version(model_artifact)
        )
        governance = dict(self._governance) if self._governance else None
        evaluation = dict(self._evaluation) if self._evaluation else None
        explanation = Explanation(
            method=method,
            status=status,
            factors=factors,
            quality=quality,
            plain_language=self._plain_language(decision, factors),
            counterfactual=counterfactual_text,
            regulation_flags=self._regulation_flags(),
            feature_importance=feature_importance,
            compute_usage=compute_usage,
            metadata={
                "project": self.project,
                "regulation": self.regulation,
                "risk_level": self.risk_level,
                "feature_lineage": dict(self._feature_lineage),
            },
            audit_trail_id=audit_id,
            model_version=model_version,
            governance=governance,
            evaluation=evaluation,
        )
        return DecisionEvent(
            decision=decision,
            explanation=explanation,
            audit_id=audit_id,
            model=model_spec,
            dataset=dataset,
            project=self.project_info,
            governance=governance,
            evaluation=evaluation,
            raw_output=raw_output,
        )

    def _record_decision_event(self, event: DecisionEvent, *, function_name: str) -> AuditEvent:
        timestamp = self._utc_now()
        audit_event = event.to_audit_event(timestamp=timestamp, event_type="decision")
        self._audit_log.append(
            {
                "type": "inference_explain",
                "timestamp": timestamp,
                "function": function_name,
                "result": event.to_report_dict(),
                "audit_event": audit_event.as_dict(),
            }
        )
        return audit_event

    def _model_spec_from_artifact(self, artifact: Any) -> Model:
        version = self._model_version(artifact)
        return Model(
            id=self.project,
            version=version,
            framework=_infer_framework(artifact),
            model_hash=_hash_model_artifact(artifact) if artifact is not None else None,
        )

    def _schema_hash_from_data(self, data: Any) -> Optional[str]:
        pd = _safe_import_pandas()
        try:
            if pd is not None and isinstance(data, pd.DataFrame):
                payload = "|".join(f"{c}:{data[c].dtype}" for c in data.columns)
                return _sha256_hex(payload.encode("utf-8"))
            if isinstance(data, dict):
                payload = "|".join(f"{k}:{type(v).__name__}" for k, v in sorted(data.items()))
                return _sha256_hex(payload.encode("utf-8"))
            arr = np.asarray(data)
            return _sha256_hex(f"ndarray:{arr.shape}:{arr.dtype}".encode("utf-8"))
        except Exception:
            logger.debug("Could not compute schema hash", exc_info=True)
            return None

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
        if rows:
            logger.info(
                "register_integrated_gradients_feature_importance: computed global importance for %d feature(s)",
                len(rows),
            )
        else:
            logger.warning(
                "register_integrated_gradients_feature_importance: no IG importance computed "
                "(check that torch is installed and the model/reference inputs are valid)"
            )

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
        if len(y_true_arr) != len(y_pred_arr):
            logger.warning(
                "monitor: y_true (n=%d) and y_pred (n=%d) lengths differ; accuracy may be misleading",
                len(y_true_arr),
                len(y_pred_arr),
            )
        if protected_feature is not None and len(protected_feature) != len(y_pred_arr):
            logger.warning(
                "monitor: protected_feature length (%d) does not match y_pred (%d); "
                "group fairness rates were skipped",
                len(protected_feature),
                len(y_pred_arr),
            )
        logger.info(
            "monitor: accuracy=%.4f, positive_rate=%.4f, drift=%s, groups=%d",
            accuracy,
            pred_positive_rate,
            f"{drift:.4f}" if drift is not None else "n/a",
            len(bias_metrics),
        )
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

        written = _write_audit_report(
            self._audit_log,
            path,
            title=title,
            format=format,
            governance=self._governance,
            evaluation=self._evaluation,
        )
        logger.info(
            "write_audit_report: wrote %s report with %d audit entries to %s",
            format,
            len(self._audit_log),
            written,
        )
        return written

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
                logger.warning("Failed to read model params via get_params()", exc_info=True)
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
                    logger.debug("Could not compute training_score via model.score()", exc_info=True)
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
            logger.debug("torch not available; cannot detect nn.Module argument", exc_info=True)
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
            logger.debug("torch not available; cannot detect Tensor argument", exc_info=True)
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
            logger.debug("torch not available while extracting score; treating decision as array-like")
        arr = np.asarray(raw_decision)
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1:
            return float(arr[-1])
        if arr.ndim >= 2:
            return float(arr[0, -1])
        return 0.0

    @staticmethod
    def _to_decision_output(score: float, threshold: float) -> Decision:
        if score >= threshold:
            return Decision(value="APPROVED", score=score, threshold=threshold)
        return Decision(value="DECLINED", score=score, threshold=threshold)

    def _top_factors(self, model: Any, row: Any, top_k: int = 3) -> List[TopFactor]:
        factors = self._top_factors_shap(model, row)
        if not factors:
            logger.warning(
                "Local SHAP attribution unavailable; falling back to a heuristic estimate. "
                "Reported factors are approximate and not true SHAP values."
            )
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
            logger.debug(
                "Global SHAP skipped (shap=%s, model=%s, reference_rows=%s)",
                shap is not None,
                model is not None,
                None if row_df is None else len(row_df),
            )
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
                logger.warning("Global SHAP produced unexpected ndim=%d; skipping importance", vals.ndim)
                return []
            features = list(row_df.columns)
            if mean_abs.shape[0] != len(features):
                logger.warning(
                    "Global SHAP value count (%d) does not match feature count (%d); skipping",
                    mean_abs.shape[0],
                    len(features),
                )
                return []
            pairs = sorted(zip(features, mean_abs.tolist()), key=lambda x: x[1], reverse=True)
            if top_k is not None:
                pairs = pairs[: int(top_k)]
            return [FeatureImportance(name=str(n), mean_abs_shap=float(v), source="shap") for n, v in pairs]
        except Exception:
            logger.warning("Global SHAP feature importance failed", exc_info=True)
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
            logger.debug("torch unavailable; skipping global Integrated Gradients importance", exc_info=True)
            return []

        if model is None:
            logger.debug("Global IG skipped: no model provided")
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
            logger.warning("Global Integrated Gradients importance failed", exc_info=True)
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
            logger.debug(
                "Local IG skipped (model=%s, x_tensor=%s)",
                model is not None,
                x_tensor is not None,
            )
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
            logger.warning("Local Integrated Gradients attribution failed", exc_info=True)
            return []

    def _top_factors_shap(self, model: Any, row: Any) -> List[TopFactor]:
        shap = _safe_import_shap()
        pd = _safe_import_pandas()
        if shap is None or model is None or row is None or pd is None:
            logger.debug(
                "Local SHAP skipped (shap=%s, pandas=%s, model=%s, row=%s)",
                shap is not None,
                pd is not None,
                model is not None,
                row is not None,
            )
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
            logger.debug("Local SHAP attribution failed; caller will use fallback", exc_info=True)
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
                logger.debug("Counterfactual value not numeric for '%s'; using generic text", dominant.name)
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
            logger.debug("torch not available while resolving model version", exc_info=True)
        params = self._extract_model_params(model)
        n_estimators = params.get("n_estimators")
        max_depth = params.get("max_depth")
        suffix = f"{model.__class__.__name__.lower()}"
        if n_estimators is not None and max_depth is not None:
            suffix = f"xgb-{n_estimators}-d{max_depth}"
        return f"{self.project} / {suffix}"

    def _plain_language(self, decision: Decision, factors: List[TopFactor]) -> str:
        if not factors:
            return f"{decision.value.title()}. Decision generated by model score and threshold."
        reason = factors[0]
        direction = "high" if isinstance(reason.value, (int, float)) and float(reason.value) > 0 else "low"
        return (
            f"{decision.value.title()}. Main reason: {direction} {reason.name.replace('_', ' ')} "
            f"which moved the score {'down' if reason.impact < 0 else 'up'}."
        )

    def _regulation_flags(self) -> List[str]:
        norm = (self.regulation or "").lower().strip()
        if not norm:
            return []
        if norm == "eu-ai-act":
            return ["EU AI Act Art.13 ✓", "GDPR Art.22 ✓"]
        return [f"{self.regulation} ✓"]

    @staticmethod
    def _audit_trail_id() -> str:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        token = uuid.uuid4().hex[:7]
        return f"ax-{date}-{token}"
