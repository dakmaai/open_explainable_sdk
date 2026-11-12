"""Image classification: binary and multi-class decisions, painted on the input."""

from __future__ import annotations

import pytest

import numpy as np

import dakma
from dakma_sdk.image import _blend_overlay, explain_image_prediction


def _side(factors, side: str) -> float:
    return sum(factor.impact for factor in factors if side in factor.name)


def test_overlay_tints_the_input_without_covering_it() -> None:
    image = np.full((4, 8, 3), 255, dtype=np.uint8)
    image[:, :4] = 0
    signed = np.zeros((4, 8))
    signed[:, 4:] = 1.0
    overlay = _blend_overlay(image, signed)

    assert overlay[:, :4].max() == 0
    right = overlay[:, 4:]
    assert right[..., 0].min() == 255
    assert right[..., 1].min() > 140
    assert right[..., 2].min() > 140


def test_overlay_shows_weaker_regions_on_the_same_gradient() -> None:
    image = np.full((1, 4, 3), 200, dtype=np.uint8)
    signed = np.array([[0.0, 0.1, 0.4, 1.0]], dtype=float)
    red = _blend_overlay(image, signed)[0, :, 0].astype(int)

    assert red[0] == 200
    assert red[3] > red[2] > red[1] > red[0]


def test_two_outputs_pick_the_higher_class() -> None:
    index, name, probability, n_outputs = explain_image_prediction(
        [0.2, 0.8], class_names=["cat", "dog"]
    )
    assert (index, name, n_outputs) == (1, "dog", 2)
    assert probability == pytest.approx(0.8)


def test_three_outputs_pick_the_predicted_class() -> None:
    index, name, _, n_outputs = explain_image_prediction(
        [0.1, 0.2, 0.7], class_names=["cat", "dog", "bird"]
    )
    assert (index, name, n_outputs) == (2, "bird", 3)


def test_one_score_uses_the_threshold() -> None:
    index, name, _, n_outputs = explain_image_prediction(
        [2.0], class_names=["negative", "positive"]
    )
    assert (index, name, n_outputs) == (1, "positive", 1)
    index, name, _, _ = explain_image_prediction([-2.0], class_names=["negative", "positive"])
    assert (index, name) == (0, "negative")


def test_overlay_localizes_binary_and_multiclass_regions(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    class StripClassifier(nn.Module):
        """Class i is the mean of vertical strip i, so its regions sit in that strip."""

        def __init__(self, n_classes: int) -> None:
            super().__init__()
            self.n_classes = n_classes

        def forward(self, image):
            width = image.shape[-1]
            scores = []
            for index in range(self.n_classes):
                start = int(round(index * width / self.n_classes))
                end = int(round((index + 1) * width / self.n_classes))
                scores.append(image[:, :, :, start:end].mean(dim=(1, 2, 3)))
            return torch.stack(scores, dim=-1)

    def bright_on_the_right(width: int) -> "torch.Tensor":
        image = torch.zeros(1, 1, 8, width)
        image[:, :, :, width // 2 :] = 1.0
        return image

    dm = dakma.init(project="image-classification")

    @dm.explain_image(class_names=["cat", "dog"], n_steps=2, top_k=4)
    def classify_binary(model, image):
        return model(image)

    binary = StripClassifier(2).eval()
    event = classify_binary(binary, bright_on_the_right(8))
    assert event.decision.predicted_class == "dog"
    assert event.decision.value not in {"APPROVED", "DECLINED"}
    assert _side(event.explanation.factors, "right") > _side(event.explanation.factors, "left")
    png = event.highlighted_image()
    assert png.startswith(b"\x89PNG")
    assert event.explanation.image_attribution is not None
    assert event.explanation.image_attribution.explained_class_name == "dog"

    @dm.explain_image(class_names=["cat", "dog", "bird"], n_steps=2, top_k=4)
    def classify_multi(model, image):
        return model(image)

    multi = StripClassifier(3).eval()
    event = classify_multi(multi, bright_on_the_right(12))
    assert event.decision.predicted_class == "bird"
    assert _side(event.explanation.factors, "right") > _side(event.explanation.factors, "left")

    # A dim left strip still loses to the bright right strip, but it is no longer
    # identical to the zero baseline, so its class has a visible region.
    dim_left = bright_on_the_right(12)
    dim_left[:, :, :, :4] = 0.25

    @dm.explain_image(class_names=["cat", "dog", "bird"], target_class=0, n_steps=2, top_k=4)
    def classify_other(model, image):
        return model(image)

    other = classify_other(multi, dim_left)
    assert other.decision.predicted_class == "bird"
    assert other.explanation.metadata["explained_class"] == "cat"
    assert "Explaining class cat" in other.explanation.plain_language
    assert _side(other.explanation.factors, "left") > _side(other.explanation.factors, "right")

    with pytest.raises(ValueError, match="out of range"):
        classify_other_bad = dm.explain_image(class_names=["cat", "dog", "bird"], target_class=5)(
            lambda model, image: model(image)
        )
        classify_other_bad(multi, bright_on_the_right(12))

    report = other.write_report(tmp_path / "decision.html", format="html")
    html = report.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    assert "Areas of interest" in html
    assert "cat" in html


def test_single_score_heatmap_supports_the_predicted_class() -> None:
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    class SideScore(nn.Module):
        def forward(self, image):
            mid = image.shape[-1] // 2
            right = image[:, :, :, mid:].mean(dim=(1, 2, 3))
            left = image[:, :, :, :mid].mean(dim=(1, 2, 3))
            return (right - left).unsqueeze(-1)

    dm = dakma.init(project="image-score")
    model = SideScore().eval()

    @dm.explain_image(class_names=["negative", "positive"], n_steps=2, top_k=4)
    def classify(model, image):
        return model(image)

    left = torch.zeros(1, 1, 8, 8)
    left[:, :, :, :4] = 1.0
    negative = classify(model, left)
    assert negative.decision.predicted_class == "negative"
    assert _side(negative.explanation.factors, "left") > _side(negative.explanation.factors, "right")

    positive = classify(model, left.flip(-1))
    assert positive.decision.predicted_class == "positive"
    assert _side(positive.explanation.factors, "right") > _side(positive.explanation.factors, "left")


def test_channels_last_layout_is_accepted() -> None:
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    class StripClassifier(nn.Module):
        def forward(self, image):
            mid = image.shape[-1] // 2
            left = image[:, :, :, :mid].mean(dim=(1, 2, 3))
            right = image[:, :, :, mid:].mean(dim=(1, 2, 3))
            return torch.stack([left, right], dim=-1)

    image = torch.zeros(1, 8, 8, 1)
    image[:, :, 4:, :] = 1.0
    dm = dakma.init(project="image-nhwc")

    @dm.explain_image(class_names=["cat", "dog"], channels_last=True, n_steps=2)
    def classify(model, image):
        return model(image.permute(0, 3, 1, 2))

    event = classify(StripClassifier().eval(), image)
    assert event.decision.predicted_class == "dog"
    assert event.highlighted_image().startswith(b"\x89PNG")
