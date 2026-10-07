"""Token-level Integrated Gradients for text, and the highlighted text it renders."""

from __future__ import annotations

import re
from typing import List

import pytest

import dakma
from dakma_sdk.audit_format import (
    format_inference_result_html,
    format_inference_result_markdown,
    format_text_highlight_ansi,
    format_text_highlight_html,
    format_text_highlight_markdown,
)
from dakma_sdk.core import DakmaClient
from dakma_sdk.models import Decision, DecisionEvent, Explanation, TokenAttribution

REVIEW = "the service was terrible and rude"


@pytest.fixture
def attributions() -> List[TokenAttribution]:
    impacts = [0.45, 0.54, 0.04, 2.03, 0.03, -2.18]
    return [
        TokenAttribution(token=tok, index=i, impact=impact)
        for i, (tok, impact) in enumerate(zip(REVIEW.split(), impacts))
    ]


def _event(attrs: List[TokenAttribution]) -> DecisionEvent:
    return DecisionEvent(
        decision=Decision(value="DECLINED", score=0.02, threshold=0.5),
        explanation=Explanation(method="integrated_gradients", status="success", text_attributions=attrs),
        audit_id="ax-test-1",
    )


class TestTokenAttribution:
    def test_direction_follows_the_sign(self) -> None:
        assert TokenAttribution(token="rude", index=5, impact=-2.18).direction == "down"
        assert TokenAttribution(token="great", index=1, impact=0.4).direction == "up"
        assert "pushed score down -2.1800" in TokenAttribution(token="rude", index=5, impact=-2.18).to_text()

    def test_is_exported(self) -> None:
        assert hasattr(dakma, "TokenAttribution")


class TestHighlightRendering:
    def test_ansi_colours_every_token(self, attributions: List[TokenAttribution]) -> None:
        out = format_text_highlight_ansi(attributions)
        for token in REVIEW.split():
            assert token in out
        assert out.count("\x1b[0m") == len(attributions)

    def test_markdown_is_inline_svg_with_a_legend(self, attributions: List[TokenAttribution]) -> None:
        out = format_text_highlight_markdown(attributions)
        assert "<svg" in out and "terrible" in out
        assert "blue pushed the score up" in out

    def test_html_shades_tokens_by_sign(self, attributions: List[TokenAttribution]) -> None:
        out = format_text_highlight_html(attributions)
        assert "rgba(185,28,28" in out  # "rude" pushed the score down
        assert "rgba(29,78,216" in out  # "terrible" pushed it up
        assert 'title="rude: -2.180000"' in out

    def test_strongest_token_is_the_most_opaque(self, attributions: List[TokenAttribution]) -> None:
        out = format_text_highlight_html(attributions)
        opacities = {
            token: float(alpha)
            for alpha, token in re.findall(r"rgba\(\d+,\d+,\d+,([\d.]+)\)[^>]*>(\w+)</span>", out)
        }
        assert opacities["rude"] > opacities["service"] > opacities["was"]

    def test_renderers_accept_report_dicts(self, attributions: List[TokenAttribution]) -> None:
        as_dicts = [{"token": a.token, "index": a.index, "impact": a.impact} for a in attributions]
        assert "rude" in format_text_highlight_html(as_dicts)
        assert "rude" in format_text_highlight_ansi(as_dicts)

    def test_empty_attributions_render_nothing(self) -> None:
        assert format_text_highlight_markdown([]) == ""
        assert format_text_highlight_html([]) == ""
        assert format_text_highlight_ansi([]) == ""

    def test_long_text_is_truncated_with_a_note(self) -> None:
        many = [TokenAttribution(token=f"w{i}", index=i, impact=0.1) for i in range(350)]
        assert "Showing the first 300 of 350 tokens" in format_text_highlight_markdown(many)


class TestExplanationApi:
    def test_highlighted_text_formats(self, attributions: List[TokenAttribution]) -> None:
        event = _event(attributions)
        assert "\x1b[48;2;" in event.highlighted_text()
        assert "<svg" in event.highlighted_text("markdown")
        assert "dakma-token" in event.highlighted_text("html")

    def test_no_attributions_means_no_highlight(self) -> None:
        assert _event([]).highlighted_text() == ""

    def test_unknown_format_is_rejected(self, attributions: List[TokenAttribution]) -> None:
        with pytest.raises(ValueError, match="ansi"):
            _event(attributions).highlighted_text("latex")  # type: ignore[arg-type]


class TestReportsIncludeHighlight:
    def test_markdown_report_has_a_highlight_section(self, attributions: List[TokenAttribution]) -> None:
        report = format_inference_result_markdown(_event(attributions).to_report_dict())
        assert "**Highlighted text (this decision)**" in report
        assert "terrible" in report

    def test_html_report_has_a_highlight_section(self, attributions: List[TokenAttribution]) -> None:
        report = format_inference_result_html(_event(attributions).to_report_dict())
        assert "Highlighted text (this decision)" in report
        assert "dakma-token" in report

    def test_tabular_decisions_have_no_highlight_section(self) -> None:
        report = format_inference_result_markdown(_event([]).to_report_dict())
        assert "Highlighted text" not in report
