"""Spatial Integrated Gradients for image classifiers.

The wrapped model receives one channels-first image, ``(1, C, H, W)`` or ``(C, H, W)``.
Attributions for the explained class are summed over channels into a heatmap and painted
onto that input image as a translucent tint: red where the region supported the class,
blue where it counted against it. The color deepens with the attribution, so weaker
regions stay on the same gradient. The input image stays visible under the tint.
"""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .integrated_gradients import _torch_nn, compute_integrated_gradients_attributions
from .models import ImageAttribution, TopFactor


def explain_image_prediction(
    raw_decision: Any,
    *,
    class_names: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
) -> Tuple[int, str, float, int]:
    """Return ``(predicted_index, class_name, probability, n_outputs)``.

    Two or more outputs are a classifier, binary or multi-class: the decision is the
    highest score. One output is a positive-class score compared with ``threshold``;
    pass two ``class_names`` for the negative and positive labels.
    """
    values = _output_vector(raw_decision)
    n_outputs = int(values.shape[0])
    if n_outputs < 1:
        raise ValueError("explain_image expected at least one model output")
    if n_outputs == 1:
        raw = float(values[0])
        positive = raw if 0.0 <= raw <= 1.0 else float(1.0 / (1.0 + np.exp(-raw)))
        index = 1 if positive >= float(threshold) else 0
        probability = positive if index == 1 else 1.0 - positive
        names = list(class_names) if class_names is not None else ["0", "1"]
        if len(names) < 2:
            names = ["0", "1"]
        return index, str(names[index]), probability, n_outputs
    probabilities = _as_probabilities(values)
    index = int(np.argmax(probabilities))
    return index, _lookup_name(class_names, index, n_outputs), float(probabilities[index]), n_outputs


def image_areas_of_interest(
    model: Any,
    image: Any,
    *,
    explained_class: int,
    explained_class_name: str,
    n_steps: int = 16,
    baseline: Any = None,
    channels_last: bool = False,
    top_k: int = 4,
    target_output: Optional[int] = None,
    polarity: float = 1.0,
) -> Tuple[List[TopFactor], ImageAttribution]:
    """Attribute one class score and overlay the heatmap on ``image``.

    ``target_output`` is the model output index Integrated Gradients explains.
    It defaults to ``explained_class``. ``polarity`` flips the heatmap so red
    still marks regions that supported the recorded class when that class is the
    negative side of a single score.
    """
    torch, _ = _torch_nn()
    if torch is None:
        raise RuntimeError("PyTorch is required for image explanations. Install torch (dakma-sdk[dl]).")
    batched = _as_nchw(image, channels_last=channels_last)
    attr = compute_integrated_gradients_attributions(
        model,
        batched,
        baseline=baseline,
        n_steps=n_steps,
        target_class=int(explained_class if target_output is None else target_output),
    )
    signed = np.sum(attr[0].detach().cpu().numpy(), axis=0).astype(float) * float(polarity)
    display = _display_rgb(batched[0].detach().cpu().numpy())
    overlay = _blend_overlay(display, signed)
    factors = _region_factors(signed, top_k=top_k)
    attribution = ImageAttribution(
        width=int(display.shape[1]),
        height=int(display.shape[0]),
        explained_class=int(explained_class),
        explained_class_name=explained_class_name,
        overlay_png=base64.b64encode(_encode_png(overlay)).decode("ascii"),
    )
    return factors, attribution


def _output_vector(raw_decision: Any) -> np.ndarray:
    torch, _ = _torch_nn()
    if torch is not None and isinstance(raw_decision, torch.Tensor):
        raw_decision = raw_decision.detach().cpu().numpy()
    try:
        arr = np.asarray(raw_decision, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("explain_image expected a numeric class score") from exc
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("explain_image expected a finite numeric class score")
    return np.asarray(arr.reshape(-1, arr.shape[-1])[0], dtype=float)


def _as_probabilities(values: np.ndarray) -> np.ndarray:
    if np.all(values >= 0) and abs(float(values.sum()) - 1.0) <= 0.05:
        return values
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    total = float(exp.sum())
    if total == 0.0:
        return np.full(values.shape, 1.0 / len(values))
    return exp / total


def _lookup_name(class_names: Optional[Sequence[str]], index: int, n_outputs: int) -> str:
    if class_names is not None and len(class_names) == n_outputs:
        return str(class_names[index])
    return str(index)


def _as_nchw(image: Any, *, channels_last: bool) -> Any:
    torch, _ = _torch_nn()
    tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(np.asarray(image))
    tensor = tensor.float()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 4 or int(tensor.shape[0]) != 1:
        raise ValueError("explain_image expects one image shaped (1, C, H, W) or (C, H, W).")
    if channels_last:
        tensor = tensor.permute(0, 3, 1, 2)
    if int(tensor.shape[1]) < 1 or int(tensor.shape[2]) < 2 or int(tensor.shape[3]) < 2:
        raise ValueError("explain_image expects a channels-first image with height and width of at least 2.")
    return tensor


def _display_rgb(chw: np.ndarray) -> np.ndarray:
    channels = np.transpose(chw, (1, 2, 0))
    if channels.shape[-1] == 1:
        channels = np.repeat(channels, 3, axis=2)
    elif channels.shape[-1] > 3:
        channels = channels[:, :, :3]
    finite = channels[np.isfinite(channels)]
    low = float(np.min(finite)) if finite.size else 0.0
    high = float(np.max(finite)) if finite.size else 1.0
    if low >= 0.0 and high <= 1.0:
        scaled = channels
    elif low >= 0.0 and high <= 255.0:
        scaled = channels / 255.0
    else:
        span = high - low
        scaled = np.zeros_like(channels) if span == 0 else (channels - low) / span
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


# Peak tint opacity. The rest of each pixel is the original image, so a bright input
# stays bright instead of being replaced by a flat red or blue block.
_MAX_OVERLAY_ALPHA = 0.4
# Below 1 the scale is compressed, so a region with a fraction of the peak attribution
# still receives a visible tint instead of disappearing next to the strongest pixels.
_OVERLAY_GAMMA = 0.45


def _blend_overlay(image: np.ndarray, signed: np.ndarray) -> np.ndarray:
    """Tint ``image`` on a gradient. Deeper red supports the class; deeper blue counts against it."""
    magnitude = np.abs(signed)
    peak = float(magnitude.max())
    if peak == 0.0:
        strength = np.zeros_like(magnitude)
    else:
        strength = np.power(magnitude / peak, _OVERLAY_GAMMA)
    alpha = (strength * _MAX_OVERLAY_ALPHA)[..., None]
    red = np.zeros_like(image)
    red[..., 0] = 255
    blue = np.zeros_like(image)
    blue[..., 2] = 255
    color = np.where(signed[..., None] >= 0, red, blue)
    blended = image.astype(float) * (1.0 - alpha) + color.astype(float) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _region_factors(signed: np.ndarray, *, top_k: int) -> List[TopFactor]:
    height, width = signed.shape
    row_cuts = [0, height // 2, height]
    col_cuts = [0, width // 2, width]
    names = (("upper left", "upper right"), ("lower left", "lower right"))
    factors: List[TopFactor] = []
    for row in range(2):
        for col in range(2):
            block = signed[row_cuts[row] : row_cuts[row + 1], col_cuts[col] : col_cuts[col + 1]]
            impact = float(np.sum(block))
            factors.append(
                TopFactor(
                    name=names[row][col],
                    value=float(np.mean(np.abs(block))),
                    impact=impact,
                    direction="up" if impact >= 0 else "down",
                )
            )
    factors.sort(key=lambda factor: abs(factor.impact), reverse=True)
    return factors[: max(1, int(top_k))]


def _encode_png(rgb: np.ndarray) -> bytes:
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("PNG encoder expected an RGB image.")
    raw = b"".join(b"\x00" + rgb[row].astype(np.uint8).tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
