"""Per-token Integrated Gradients on a PyTorch text classifier (needs torch)."""

from __future__ import annotations

from typing import Any, List

import pytest

import dakma
from dakma_sdk.core import DakmaClient
from dakma_sdk.models import DecisionEvent

torch = pytest.importorskip("torch", reason="token attributions need torch (pip install dakma-sdk[dl])")
nn = torch.nn

REVIEW = "the service was terrible and rude"
VOCAB = {w: i + 1 for i, w in enumerate(sorted({w for w in (REVIEW + " great kind").split()}))}


def _encode(text: str) -> Any:
    return torch.tensor([[VOCAB[w] for w in text.split()]], dtype=torch.long)


class _TextClassifier(nn.Module):
    """Embedding -> mean pool -> linear, with "terrible" and "rude" wired to the negative class."""

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
            self.head.weight[0, 0] = 3.0  # class 0 = negative sentiment
            self.head.weight[1, 0] = -3.0

    def forward(self, ids: Any) -> Any:
        return self.head(self.emb(ids).mean(dim=1))


@pytest.fixture
def classifier() -> _TextClassifier:
    return _TextClassifier().eval()


@pytest.fixture
def client() -> DakmaClient:
    return dakma.init(project="review-sentiment")


class TestTokenAttributions:
    def test_sentiment_words_dominate(self, classifier: _TextClassifier) -> None:
        from dakma_sdk.text import token_attributions

        tokens = REVIEW.split()
        attrs = token_attributions(classifier, _encode(REVIEW), tokens=tokens)

        assert [a.token for a in attrs] == tokens
        assert [a.index for a in attrs] == list(range(len(tokens)))
        ranked = sorted(attrs, key=lambda a: abs(a.impact), reverse=True)
        assert {ranked[0].token, ranked[1].token} == {"terrible", "rude"}
        assert all(abs(a.impact) < 1e-6 for a in attrs if a.token not in {"terrible", "rude"})

    def test_accepts_a_tokenizer(self, classifier: _TextClassifier) -> None:
        from dakma_sdk.text import token_attributions

        class Tokenizer:
            def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
                lookup = {v: k for k, v in VOCAB.items()}
                return [lookup[i] for i in ids]

        attrs = token_attributions(classifier, _encode(REVIEW), tokenizer=Tokenizer())
        assert [a.token for a in attrs] == REVIEW.split()

    def test_accepts_a_callable_token_source(self, classifier: _TextClassifier) -> None:
        from dakma_sdk.text import token_attributions

        attrs = token_attributions(classifier, _encode(REVIEW), tokens=lambda ids: REVIEW.split())
        assert [a.token for a in attrs] == REVIEW.split()

    def test_falls_back_to_positional_names(self, classifier: _TextClassifier) -> None:
        from dakma_sdk.text import token_attributions

        attrs = token_attributions(classifier, _encode(REVIEW))
        assert [a.token for a in attrs] == [f"token_{i}" for i in range(6)]

    def test_accepts_an_unbatched_sequence(self, classifier: _TextClassifier) -> None:
        from dakma_sdk.text import token_attributions

        attrs = token_attributions(classifier, _encode(REVIEW)[0], tokens=REVIEW.split())
        assert len(attrs) == 6

    def test_finds_the_embedding_layer(self, classifier: _TextClassifier) -> None:
        from dakma_sdk.text import find_embedding_layer

        assert find_embedding_layer(classifier) is classifier.emb

    def test_requires_an_embedding_layer(self) -> None:
        from dakma_sdk.text import find_embedding_layer

        with pytest.raises(LookupError, match="No nn.Embedding"):
            find_embedding_layer(nn.Linear(2, 2))

    def test_rejects_batched_input(self, classifier: _TextClassifier) -> None:
        from dakma_sdk.text import token_attributions

        batched = torch.cat([_encode(REVIEW), _encode(REVIEW)], dim=0)
        with pytest.raises(ValueError, match="one sequence at a time"):
            token_attributions(classifier, batched)


class TestExplainTextDecorator:
    def test_decision_event_carries_token_impacts(
        self, client: DakmaClient, classifier: _TextClassifier
    ) -> None:
        tokens = REVIEW.split()

        @client.explain_text(tokens=tokens)
        def predict(model: Any, ids: Any) -> Any:
            return torch.softmax(model(ids), dim=1)

        event = predict(classifier, _encode(REVIEW))

        assert isinstance(event, DecisionEvent)
        assert event.explanation.method == "integrated_gradients"
        assert event.explanation.status == "success"
        assert [a.token for a in event.explanation.text_attributions] == tokens
        assert event.explanation.metadata["input_kind"] == "text"
        assert event.explanation.quality["token_count"] == len(tokens)

    def test_top_factors_are_tokens_ranked_by_impact(
        self, client: DakmaClient, classifier: _TextClassifier
    ) -> None:
        @client.explain_text(tokens=REVIEW.split(), top_k=2)
        def predict(model: Any, ids: Any) -> Any:
            return torch.softmax(model(ids), dim=1)

        factors = predict(classifier, _encode(REVIEW)).explanation.top_factors
        assert {f.name for f in factors} == {"terrible", "rude"}
        assert abs(factors[0].impact) >= abs(factors[1].impact)

    def test_impacts_explain_the_score_not_the_argmax(
        self, client: DakmaClient, classifier: _TextClassifier
    ) -> None:
        from dakma_sdk.text import token_attributions

        @client.explain_text(tokens=REVIEW.split())
        def predict(model: Any, ids: Any) -> Any:
            return torch.softmax(model(ids), dim=1)

        event = predict(classifier, _encode(REVIEW))
        decorated = {a.token: a.impact for a in event.explanation.text_attributions}
        direct = {a.token: a.impact for a in token_attributions(classifier, _encode(REVIEW), tokens=REVIEW.split())}

        # The decorated score is P(class 1), which "terrible" and "rude" push down, whereas
        # calling token_attributions directly targets the argmax class (0) and so flips the sign.
        assert decorated["rude"] < 0 and decorated["terrible"] < 0
        assert direct["rude"] > 0 and direct["terrible"] > 0
        assert event.decision.value == "DECLINED"

    def test_plain_language_names_the_token(self, client: DakmaClient, classifier: _TextClassifier) -> None:
        @client.explain_text(tokens=REVIEW.split())
        def predict(model: Any, ids: Any) -> Any:
            return torch.softmax(model(ids), dim=1)

        plain = predict(classifier, _encode(REVIEW)).explanation.plain_language
        assert "Main driver: the token" in plain

    def test_model_handle_records_provenance(self, client: DakmaClient, classifier: _TextClassifier) -> None:
        handle = client.model(name="review-sentiment", version="1.0.0", artifact=classifier)

        @handle.explain_text(tokens=REVIEW.split())
        def predict(ids: Any) -> Any:
            return torch.softmax(classifier(ids), dim=1)

        event = predict(_encode(REVIEW))
        assert event.model.id == "review-sentiment"
        assert event.model.framework == "pytorch"
        assert event.explanation.text_attributions

    def test_report_highlights_the_text(self, client: DakmaClient, classifier: _TextClassifier) -> None:
        @client.explain_text(tokens=REVIEW.split())
        def predict(model: Any, ids: Any) -> Any:
            return torch.softmax(model(ids), dim=1)

        event = predict(classifier, _encode(REVIEW))
        assert "Highlighted text (this decision)" in event.as_markdown_table()
        assert "terrible" in event.highlighted_text("html")

    def test_audit_log_keeps_token_impacts(self, client: DakmaClient, classifier: _TextClassifier) -> None:
        @client.explain_text(tokens=REVIEW.split())
        def predict(model: Any, ids: Any) -> Any:
            return torch.softmax(model(ids), dim=1)

        predict(classifier, _encode(REVIEW))
        entry = client.export_audit_log()[-1]
        recorded = entry["result"]["explain"]["text_attributions"]
        assert [r["token"] for r in recorded] == REVIEW.split()

    def test_failure_degrades_instead_of_raising(self, client: DakmaClient) -> None:
        @client.explain_text(tokens=REVIEW.split())
        def predict(model: Any, ids: Any) -> Any:
            return torch.softmax(model(ids), dim=1)

        event = predict(nn.Linear(6, 2), torch.ones(1, 6))

        assert event.decision.value in {"APPROVED", "DECLINED"}
        assert event.explanation.status == "degraded"
        assert "No nn.Embedding" in event.explanation.metadata["attribution_error"]


class TestReductionAxis:
    def test_tokens_reduce_over_the_embedding_axis(self) -> None:
        from dakma_sdk.integrated_gradients import reduce_attributions_to_features

        attr = torch.tensor([[[1.0, 1.0], [0.0, 0.0], [-2.0, -1.0]]])  # (1, seq_len=3, dim=2)
        names, _, impacts = reduce_attributions_to_features(attr, feature_names=["a", "b", "c"])

        assert names == ["a", "b", "c"]
        assert impacts.tolist() == [2.0, 0.0, -3.0]

    def test_features_reduce_over_the_sequence_axis(self) -> None:
        from dakma_sdk.integrated_gradients import reduce_attributions_to_features

        attr = torch.tensor([[[1.0, 1.0], [0.0, 0.0], [-2.0, -1.0]]])
        names, _, impacts = reduce_attributions_to_features(attr, reduce="features")

        assert names == ["f0", "f1"]
        assert impacts.tolist() == [-1.0, 0.0]

    def test_tokens_without_names_get_positional_labels(self) -> None:
        from dakma_sdk.integrated_gradients import reduce_attributions_to_features

        attr = torch.tensor([[[1.0, 1.0], [0.0, 0.0], [-2.0, -1.0]]])
        names, _, _ = reduce_attributions_to_features(attr, reduce="tokens")

        assert names == ["token_0", "token_1", "token_2"]

    def test_unknown_reduce_is_rejected(self) -> None:
        from dakma_sdk.integrated_gradients import reduce_attributions_to_features

        with pytest.raises(ValueError, match="reduce must be"):
            reduce_attributions_to_features(torch.ones(1, 2, 2), reduce="sideways")
