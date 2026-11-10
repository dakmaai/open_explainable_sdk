"""
Explain a small image classifier and paint the decision's regions on the input.

Install: ``pip install "dakma-sdk[dl]"`` (torch; captum is optional).

Run: ``cd examples && python image_classification_example.py``

Writes:
- ``examples/reports_image/image_decision.html`` — open in a browser to see the overlay
- ``examples/reports_image/image_decision.md``
- ``examples/reports_image/decision_overlay.png``
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import dakma


class StripClassifier(nn.Module):
    """Three classes, one per vertical strip: left cat, middle dog, right bird."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        width = image.shape[-1]
        scores = []
        for index in range(3):
            start = int(round(index * width / 3))
            end = int(round((index + 1) * width / 3))
            scores.append(image[:, :, :, start:end].mean(dim=(1, 2, 3)))
        return torch.stack(scores, dim=-1)


def bird_image() -> torch.Tensor:
    """A bright right strip, which this model reads as bird."""
    image = torch.zeros(1, 3, 32, 48)
    image[:, :, :, 32:] = 1.0
    return image


REPORT_DIR = Path(__file__).resolve().parent / "reports_image"
CLASS_NAMES = ["cat", "dog", "bird"]


def main() -> None:
    classifier = StripClassifier().eval()
    image = bird_image()

    dm = dakma.init(
        project="image-classification",
        regulation="internal-policy",
        risk_level="limited",
    )
    dm.register_governance(
        intended_use="Illustrative image explainability demo; not a production vision model.",
        known_limitations="The class score is the mean of a vertical strip. Regions follow that rule.",
        human_oversight="Human review required before any automated image decision.",
        data_provenance="One synthetic image defined in this example script.",
        model_changelog=[
            {
                "version": "image-classification-1.0",
                "date": "2026-11-07",
                "notes": "Strip classifier + explain_image overlay",
            }
        ],
    )
    dm.register_evaluation(
        n_train=0,
        n_test=1,
        split_description="One synthetic image; not a real evaluation.",
        test_metrics={"accuracy": 1.0},
    )

    @dm.explain_image(class_names=CLASS_NAMES, n_steps=8, top_k=4)
    def classify(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
        return model(image)

    event = classify(classifier, image)

    print(event.decision.to_text())
    print(event.explanation.plain_language)
    print(event.explanation.audit_trail_id)
    print()
    print("--- Regions ---")
    for factor in event.explanation.factors:
        print(f"  {factor.name}: {factor.impact:+.4f} ({factor.direction})")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_path = REPORT_DIR / "decision_overlay.png"
    overlay_path.write_bytes(event.highlighted_image())
    decision_md = event.write_report(
        REPORT_DIR / "image_decision.md",
        title="Image classification — areas of interest",
    )
    decision_html = event.write_report(
        REPORT_DIR / "image_decision.html",
        title="Image classification — areas of interest",
        format="html",
    )

    print()
    print("--- Downloadable reports written ---")
    print(f"  {overlay_path}")
    print(f"  {decision_md}")
    print(f"  {decision_html}")
    print()
    print("Open the HTML decision report in a browser to see the regions on the input image.")


if __name__ == "__main__":
    main()
