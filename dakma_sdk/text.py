"""Token-level Integrated Gradients for text classifiers.

Attributions are computed over word embeddings: the embedding lookup is scaled by one
weight per token, so Integrated Gradients returns a single signed impact per token
instead of one per embedding dimension. Works with any model containing an
``nn.Embedding`` (including Hugging Face encoders), since the embedding output is
replaced through a forward hook rather than by rewriting the model.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .integrated_gradients import _torch_nn, compute_integrated_gradients_attributions
from .models import TokenAttribution

TokenSource = Union[Sequence[str], Callable[[Any], Sequence[str]], None]


def find_embedding_layer(model: Any) -> Any:
    """Return the first ``nn.Embedding`` in ``model``.

    For most text classifiers this is the word embedding table; pass the layer
    explicitly when a model embeds several inputs (e.g. word plus segment ids).
    """
    torch, nn = _torch_nn()
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for text attributions. Install torch (see dakma-sdk[dl]).")
    if not isinstance(model, nn.Module):
        raise TypeError("Text attributions expect a torch.nn.Module as model.")
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            return module
    raise LookupError(
        "No nn.Embedding found on the model; pass embedding_layer= explicitly, or use "
        "explain_integrated_gradients for models that take continuous inputs."
    )


def resolve_tokens(
    ids: Any,
    *,
    tokens: TokenSource = None,
    tokenizer: Any = None,
) -> List[str]:
    """Resolve display tokens for ``ids`` from an explicit list, a callable, or a tokenizer."""
    if callable(tokens):
        return [str(t) for t in tokens(ids)]
    if tokens is not None:
        return [str(t) for t in tokens]
    if tokenizer is not None:
        flat = _first_row(ids)
        convert = getattr(tokenizer, "convert_ids_to_tokens", None)
        if callable(convert):
            return [str(t) for t in convert(flat)]
        decode = getattr(tokenizer, "decode", None)
        if callable(decode):
            return [str(decode([i])).strip() for i in flat]
    return [f"token_{i}" for i in range(len(_first_row(ids)))]


def token_attributions(
    model: Any,
    ids: Any,
    *,
    tokens: TokenSource = None,
    tokenizer: Any = None,
    embedding_layer: Any = None,
    n_steps: int = 32,
    target_class: Optional[int] = None,
    baseline: Any = None,
    internal_batch_size: Optional[int] = 1,
    forward_args: Sequence[Any] = (),
    forward_kwargs: Optional[Dict[str, Any]] = None,
) -> List[TokenAttribution]:
    """Signed Integrated Gradients impact per token, in sequence order.

    ``ids`` is a single sequence of token ids (shape ``(1, seq_len)`` or ``(seq_len,)``).
    A positive impact pushed the target class score up. ``baseline`` defaults to zeroed
    embeddings; pass a ``(1, seq_len)`` weight tensor to interpolate from something else.
    ``internal_batch_size`` stays at 1 so models that mix in batch-1 tensors such as
    attention masks keep broadcasting correctly; raise it for speed on simple models.
    """
    torch, _ = _torch_nn()
    if torch is None:
        raise RuntimeError("PyTorch is required for text attributions. Install torch (see dakma-sdk[dl]).")

    ids_t = _as_batched_ids(ids)
    embedding = embedding_layer if embedding_layer is not None else find_embedding_layer(model)
    names = resolve_tokens(ids_t, tokens=tokens, tokenizer=tokenizer)

    scaled, seq_len = _token_scaling_module(
        model,
        embedding,
        ids_t,
        tuple(forward_args),
        dict(forward_kwargs or {}),
    )
    weights = torch.ones(1, seq_len)
    attr = compute_integrated_gradients_attributions(
        scaled,
        weights,
        baseline=baseline,
        n_steps=n_steps,
        target_class=target_class,
        internal_batch_size=internal_batch_size,
    )
    impacts = attr[0].detach().cpu().reshape(-1).tolist()

    if len(names) != seq_len:
        names = [f"token_{i}" for i in range(seq_len)]
    return [
        TokenAttribution(token=names[i], index=i, impact=float(impacts[i]))
        for i in range(seq_len)
    ]


def _as_batched_ids(ids: Any) -> Any:
    torch, _ = _torch_nn()
    t = ids if isinstance(ids, torch.Tensor) else torch.as_tensor(ids)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() != 2 or t.shape[0] != 1:
        raise ValueError("Text attributions expect one sequence at a time, shaped (1, seq_len).")
    return t


def _first_row(ids: Any) -> List[int]:
    torch, _ = _torch_nn()
    if torch is not None and isinstance(ids, torch.Tensor):
        flat = ids[0] if ids.dim() > 1 else ids
        return [int(i) for i in flat.detach().cpu().reshape(-1).tolist()]
    seq = list(ids)
    if seq and isinstance(seq[0], (list, tuple)):
        seq = list(seq[0])
    return [int(i) for i in seq]


def _token_scaling_module(
    model: Any,
    embedding: Any,
    ids: Any,
    forward_args: Tuple[Any, ...],
    forward_kwargs: Dict[str, Any],
) -> Tuple[Any, int]:
    """Wrap ``model`` so its input is one weight per token, scaling the embedding output."""
    torch, nn = _torch_nn()
    base = embedding(ids).detach()
    if base.dim() != 3:
        raise ValueError("Expected embedding output shaped (1, seq_len, dim).")
    seq_len = int(base.shape[1])

    class _TokenScaledModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wrapped = model

        def forward(self, weights: Any) -> Any:
            def hook(_module: Any, _inputs: Any, _output: Any) -> Any:
                return base * weights.unsqueeze(-1)

            handle = embedding.register_forward_hook(hook)
            try:
                return self.wrapped(ids, *forward_args, **forward_kwargs)
            finally:
                handle.remove()

    return _TokenScaledModel().eval(), seq_len
