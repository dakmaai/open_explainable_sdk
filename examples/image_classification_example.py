"""
Train a small CNN on MNIST, then explain why it reads one test image as a "7".

Install: ``pip install torchvision``

The first run downloads MNIST (about 11 MB) into ``examples/data/`` and trains for two
epochs on CPU, which takes about a minute.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets

import dakma

EXAMPLE_DIR = Path(__file__).resolve().parent
REPORT_DIR = EXAMPLE_DIR / "reports_image"
CLASS_NAMES = [str(digit) for digit in range(10)]


def build_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 10),
    )


def load_mnist(train: bool):
    data = datasets.MNIST(EXAMPLE_DIR / "data", train=train, download=True)
    return data.data.float().div(255.0).unsqueeze(1), data.targets


def train_cnn(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, epochs: int = 2) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(images))
        for start in range(0, len(images), 128):
            batch = order[start : start + 128]
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images[batch]), labels[batch])
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch + 1}/{epochs} done (last batch loss {loss.item():.3f})")
    model.eval()


def main() -> None:
    torch.manual_seed(0)
    train_x, train_y = load_mnist(train=True)
    test_x, test_y = load_mnist(train=False)

    model = build_cnn()
    train_cnn(model, train_x, train_y)
    with torch.no_grad():
        accuracy = float((model(test_x).argmax(dim=1) == test_y).float().mean())
    print(f"Test accuracy: {accuracy:.3f}")

    dm = dakma.init(
        project="mnist-digit-classifier",
        regulation="internal-policy",
        risk_level="limited",
    )
    dm.register_governance(
        intended_use="Illustrative image explainability demo; not a production vision model.",
        known_limitations="Two training epochs on MNIST. Explanations use a black baseline image.",
        human_oversight="Human review required before any automated image decision.",
        data_provenance="MNIST handwritten digits (60,000 train / 10,000 test) via torchvision.",
        model_changelog=[
            {
                "version": "mnist-cnn-1.0",
                "date": "2026-11-20",
                "notes": "Small CNN + explain_image overlay",
            }
        ],
    )
    dm.register_evaluation(
        n_train=len(train_x),
        n_test=len(test_x),
        split_description="Standard MNIST train/test split.",
        test_metrics={"accuracy": round(accuracy, 4)},
    )

    @dm.explain_image(class_names=CLASS_NAMES, top_k=4)
    def classify(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
        return model(image)

    seven = test_x[int((test_y == 7).nonzero()[0])].unsqueeze(0)
    event = classify(model, seven)

    print(event.decision.to_text())
    print(event.explanation.plain_language)
    print(event.explanation.audit_trail_id)
    print()
    print("--- Regions (4x4 grid) ---")
    for factor in event.explanation.factors:
        print(f"  {factor.name}: {factor.impact:+.4f} ({factor.direction})")
    check = event.explanation.image_attribution
    print()
    print(
        f"IG: {check.n_steps} steps, {check.baseline} baseline; attributions sum to "
        f"{sum(map(sum, check.attribution)):+.4f} vs score change {check.score_change:+.4f} "
        f"(gap {check.convergence_delta:+.4f})"
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_path = REPORT_DIR / "decision_overlay.png"
    overlay_path.write_bytes(event.highlighted_image())
    decision_md = event.write_report(
        REPORT_DIR / "image_decision.md",
        title="MNIST digit — areas of interest",
    )
    decision_html = event.write_report(
        REPORT_DIR / "image_decision.html",
        title="MNIST digit — areas of interest",
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
