"""
Explain a short review with per-token Integrated Gradients and highlighted text reports.

Install: ``pip install "dakma-sdk[dl]"`` (torch, captum).

Run: ``cd examples && python text_highlight_example.py``

Writes:
- ``examples/reports_text/text_decision.html`` — open in a browser to see shaded tokens
- ``examples/reports_text/text_decision.md``
- ``examples/reports_text/audit_report.html``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

import dakma

REVIEW = "the service was terrible and rude"
POSITIVE_REVIEW = "the staff was great and kind"

VOCAB: Dict[str, int] = {
    w: i + 1
    for i, w in enumerate(sorted({w for w in (REVIEW + " " + POSITIVE_REVIEW).split()}))
}


def encode(text: str) -> torch.Tensor:
    return torch.tensor([[VOCAB[w] for w in text.split()]], dtype=torch.long)


class ReviewSentimentClassifier(nn.Module):
    """Embedding -> mean pool -> linear; ``terrible`` / ``rude`` drive negative sentiment."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB) + 1, 8, padding_idx=0)
        self.head = nn.Linear(8, 2)
        with torch.no_grad():
            self.emb.weight.zero_()
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.emb.weight[VOCAB["terrible"], 0] = 1.0
            self.emb.weight[VOCAB["rude"], 0] = 1.0
            self.head.weight[0, 0] = 3.0
            self.head.weight[1, 0] = -3.0

    def forward(self, ids: Any) -> Any:
        return self.head(self.emb(ids).mean(dim=1))


REPORT_DIR = Path(__file__).resolve().parent / "reports_text"


def main() -> None:
    classifier = ReviewSentimentClassifier().eval()
    tokens: List[str] = REVIEW.split()

    dm = dakma.init(
        project="review-sentiment",
        regulation="internal-policy",
        risk_level="limited",
    )
    dm.register_governance(
        intended_use="Illustrative text explainability demo; not a production sentiment model.",
        known_limitations="Hand-wired toy weights for teaching token attributions only.",
        human_oversight="Human review required before any automated moderation decision.",
        data_provenance="Synthetic review strings defined in this example script.",
        model_changelog=[
            {
                "version": "text-highlight-1.0",
                "date": "2026-10-07",
                "notes": "Embedding classifier + explain_text + highlighted reports",
            }
        ],
    )
    dm.register_evaluation(
        n_train=0,
        n_test=2,
        split_description="Two fixed demo strings (negative and positive); not a real evaluation.",
        test_metrics={"accuracy": 1.0},
    )

    handle = dm.model(name="review-sentiment", version="1.0.0", artifact=classifier)

    @handle.explain_text(tokens=tokens, top_k=3, n_steps=32)
    def classify(ids: torch.Tensor) -> torch.Tensor:
        return torch.softmax(classifier(ids), dim=1)

    event = classify(encode(REVIEW))

    print(event.decision_output.to_text())
    print(event.explanation.plain_language)
    print(event.explanation.audit_trail_id)
    print()
    print("--- Terminal highlight (ANSI) ---")
    print(event.highlighted_text())
    print()
    print("--- Top token impacts ---")
    for factor in event.explanation.top_factors:
        print(f"  {factor.name}: {factor.impact:+.4f} ({factor.direction})")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    decision_md = event.write_report(
        REPORT_DIR / "text_decision.md",
        title="Review sentiment — token highlights",
    )
    decision_html = event.write_report(
        REPORT_DIR / "text_decision.html",
        title="Review sentiment — token highlights",
        format="html",
    )
    audit_html = dm.write_audit_report(
        REPORT_DIR / "audit_report.html",
        title="Text highlight example — audit trail",
        format="html",
    )

    print()
    print("--- Downloadable reports written ---")
    print(f"  {decision_md}")
    print(f"  {decision_html}")
    print(f"  {audit_html}")
    print()
    print("Open the HTML decision report in a browser to see shaded tokens.")


if __name__ == "__main__":
    main()
