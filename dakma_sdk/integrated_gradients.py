"""Integrated Gradients attributions for differentiable models (typically PyTorch ``nn.Module``)."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _torch_nn():
    try:
        import torch
        import torch.nn as nn

        return torch, nn
    except Exception:
        logger.debug("torch not available; Integrated Gradients disabled", exc_info=True)
        return None, None


def _safe_import_captum():
    try:
        from captum.attr import IntegratedGradients  # type: ignore

        return IntegratedGradients
    except Exception:
        logger.debug("captum not available; using built-in Riemann IG approximation", exc_info=True)
        return None


def _ensure_torch_tensor(
    x: Any,
    *,
    device: Optional[Any] = None,
    dtype: Optional[Any] = None,
) -> Any:
    torch, _ = _torch_nn()
    if torch is None:
        raise RuntimeError("PyTorch is required for Integrated Gradients. Install torch (see darsha[dl]).")
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(np.asarray(x), dtype=dtype or torch.float32)
    if device is not None:
        t = t.to(device)
    return t.float()


def _forward_logits(model: Any, inp: Any) -> Any:
    out = model(inp)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def compute_integrated_gradients_attributions(
    model: Any,
    x: Any,
    *,
    baseline: Optional[Any] = None,
    n_steps: int = 32,
    target_class: Optional[int] = None,
    internal_batch_size: Optional[int] = None,
) -> Any:
    """Return attribution tensor with the same shape as ``x`` (preserves batch).

    ``model`` must be a ``torch.nn.Module`` in eval mode. If ``target_class`` is ``None``,
    the argmax class of the first row is used for the attribution target.
    """
    torch, nn = _torch_nn()
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for Integrated Gradients.")
    if not isinstance(model, nn.Module):
        raise TypeError("Integrated Gradients expects a torch.nn.Module as model.")

    model.eval()
    x_t = _ensure_torch_tensor(x)
    if x_t.dim() < 2:
        x_t = x_t.unsqueeze(0)
    device = x_t.device
    dtype = x_t.dtype

    if baseline is None:
        baseline_t = torch.zeros_like(x_t)
    else:
        baseline_t = _ensure_torch_tensor(baseline, device=device, dtype=dtype)
        if baseline_t.shape != x_t.shape:
            if baseline_t.shape == x_t.shape[1:] or baseline_t.numel() == x_t[0].numel():
                baseline_t = baseline_t.reshape_as(x_t[0]).unsqueeze(0).expand_as(x_t)
            else:
                raise ValueError("baseline must match x shape (or be broadcastable to x).")

    with torch.no_grad():
        logits = _forward_logits(model, x_t)
        if target_class is None:
            target_class = int(torch.argmax(logits[0]).item())
            logger.debug("Integrated Gradients target_class not given; using argmax class %d", target_class)
        tgt = int(target_class)

    IG = _safe_import_captum()
    if IG is not None:
        logger.debug("Computing Integrated Gradients via captum (n_steps=%d, target=%d)", int(n_steps), tgt)

        def _forward(inp: Any) -> Any:
            return _forward_logits(model, inp)

        ig = IG(_forward)
        return ig.attribute(
            x_t,
            baselines=baseline_t,
            target=tgt,
            n_steps=int(n_steps),
            internal_batch_size=internal_batch_size,
        )

    logger.debug("Computing Integrated Gradients via Riemann approximation (n_steps=%d, target=%d)", int(n_steps), tgt)
    return _integrated_gradients_riemann(model, x_t, baseline_t, n_steps=n_steps, target_class=tgt)


def _integrated_gradients_riemann(
    model: Any,
    x: Any,
    baseline: Any,
    *,
    n_steps: int,
    target_class: int,
) -> Any:
    torch, _ = _torch_nn()
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    device = x.device
    dtype = x.dtype
    alphas = torch.linspace(0.0, 1.0, int(n_steps) + 1, device=device, dtype=dtype)
    total_grad = torch.zeros_like(x)
    for i in range(int(n_steps)):
        a0, a1 = alphas[i], alphas[i + 1]
        alpha = (a0 + a1) * 0.5
        x_interp = baseline + alpha * (x - baseline)
        x_interp = x_interp.detach().requires_grad_(True)
        logits = _forward_logits(model, x_interp)
        if logits.dim() == 1:
            scalar = logits.sum()
        else:
            scalar = logits[:, int(target_class)].sum()
        scalar.backward()
        g = x_interp.grad
        if g is None:
            continue
        total_grad = total_grad + g * (a1 - a0)
    return (x - baseline) * total_grad


def reduce_attributions_to_features(
    attr: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Reduce IG tensor to signed impact per logical feature (first batch row).

    Returns ``(names, display_values, signed_impacts)``. Display values mirror the first
    input row where shapes align (tabular); otherwise zeros.
    """
    torch, _ = _torch_nn()
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    if not isinstance(attr, torch.Tensor):
        raise TypeError("attr must be a torch.Tensor")
    a = attr[0].detach().cpu().numpy()

    if a.ndim == 1:
        n = int(a.shape[0])
        names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n)]
        if len(names) != n:
            names = [f"f{i}" for i in range(n)]
        impacts = a.astype(float)
        return names, np.zeros(n, dtype=float), impacts

    if a.ndim == 2:
        impacts = np.sum(a, axis=0).astype(float)
        n = int(impacts.shape[0])
        names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n)]
        if len(names) != n:
            names = [f"f{i}" for i in range(n)]
        return names, np.zeros(n, dtype=float), impacts

    if a.ndim == 3:
        c = int(a.shape[0])
        impacts = np.sum(a, axis=(1, 2)).astype(float)
        names = (
            list(feature_names)
            if feature_names is not None and len(feature_names) == c
            else [f"channel_{i}" for i in range(c)]
        )
        return names, np.zeros(c, dtype=float), impacts

    flat = a.reshape(-1)
    n = int(flat.shape[0])
    names = list(feature_names) if feature_names is not None and len(feature_names) == n else [f"f{i}" for i in range(n)]
    if len(names) != n:
        names = [f"f{i}" for i in range(n)]
    return names, np.zeros(n, dtype=float), flat.astype(float)


def input_values_for_display(x: Any, n_features: int) -> np.ndarray:
    """First row of ``x`` as floats for tabular display (best effort)."""
    xt = _ensure_torch_tensor(x)
    if xt.dim() >= 2:
        v = xt[0].detach().cpu().numpy().reshape(-1)
        if v.size >= n_features:
            return v[:n_features].astype(float)
    v = xt.detach().cpu().numpy().reshape(-1)
    out = np.zeros(int(n_features), dtype=float)
    out[: min(v.size, n_features)] = v[: min(v.size, n_features)]
    return out
