"""Tests for deterministic, local vector retrieval."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pytest

from ai_knowledge_assistant.models import (
    DocumentChunk,
    DocumentType,
    SourceLocator,
    SourceLocatorKind,
)
from ai_knowledge_assistant.retrieval import (
    EmbeddingProvider,
    LocalVectorIndex,
    RetrievalError,
    build_index,
    retrieve,
)


class FixedEmbeddingProvider:
    """Explicit vectors make ranking expectations independently reviewable."""

    def __init__(
        self,
        document_vectors: Sequence[Sequence[float]],
        query_vector: Sequence[float],
    ) -> None:
        self.document_vectors = tuple(tuple(vector) for vector in document_vectors)
        self.query_vector = tuple(query_vector)
        self.document_texts: tuple[str, ...] = ()
        self.query_text: str | None = None

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.document_texts = tuple(texts)
        return self.document_vectors

    def embed_query(self, text: str) -> Sequence[float]:
        self.query_text = text
        return self.query_vector


def _chunk(index: int, text: str | None = None) -> DocumentChunk:
    source_text = text or f"Exact source text {index}."
    return DocumentChunk(
        chunk_id=f"chunk-{index}",
        document_id=f"document-{index}",
        document_name=f"document-{index}.md",
        document_type=DocumentType.MARKDOWN,
        section_id=f"section-{index}",
        chunk_index=index,
        text=source_text,
        source_locator=SourceLocator(
            kind=SourceLocatorKind.DOCUMENT_SECTION,
            section_label=f"Section {index}",
            heading_level=2,
            line_start=index + 1,
            line_end=index + 2,
        ),
        source_char_start=0,
        source_char_end=len(source_text),
        primary_char_start=0,
        primary_char_end=len(source_text),
        source_section_content_hash="section-hash-" + str(index),
        content_hash=hashlib.sha256(source_text.encode()).hexdigest(),
        chunking_version="v1",
    )


def test_provider_contract_is_structural_and_exact_text_is_embedded() -> None:
    chunks = (_chunk(0, "Exact text one."), _chunk(1, "Exact text two."))
    provider: EmbeddingProvider = FixedEmbeddingProvider(((1, 0), (0, 1)), (1, 0))

    index = build_index(chunks, provider)

    assert provider.document_texts == tuple(chunk.text for chunk in chunks)
    assert [entry.text for entry in index.entries] == list(provider.document_texts)


def test_cosine_ranking_top_k_and_threshold_are_explicit() -> None:
    # Cosines against [1, 0] are 1.0, sqrt(1/2), and 0.0 respectively.
    provider = FixedEmbeddingProvider(((1, 0), (1, 1), (0, 1)), (1, 0))
    index = build_index((_chunk(0), _chunk(1), _chunk(2)), provider)

    result = retrieve(index, " question ", provider, top_k=2, minimum_score=0.7)

    assert result.question == "question"
    assert [source.chunk_id for source in result.sources] == ["chunk-0", "chunk-1"]
    assert result.sources[0].score == pytest.approx(1.0)
    assert result.sources[1].score == pytest.approx(2**-0.5)
    assert [source.rank for source in result.sources] == [1, 2]


def test_equal_scores_use_original_chunk_order_and_are_repeatable() -> None:
    provider = FixedEmbeddingProvider(((1, 0), (1, 0)), (1, 0))
    index = build_index((_chunk(7), _chunk(2)), provider)

    first = retrieve(index, "same", provider, top_k=5, minimum_score=0)
    second = retrieve(index, "same", provider, top_k=5, minimum_score=0)

    assert [source.chunk_id for source in first.sources] == ["chunk-7", "chunk-2"]
    assert first == second


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        (((1, 0),), "unexpected count"),
        (((1, 0), (1, 0, 0)), "dimensions must match"),
        (((0, 0), (1, 0)), "zero vector"),
        (((float("nan"), 0), (1, 0)), "finite values"),
        (((float("inf"), 0), (1, 0)), "finite values"),
    ],
)
def test_build_rejects_invalid_provider_outputs(
    vectors: Sequence[Sequence[float]], message: str
) -> None:
    provider = FixedEmbeddingProvider(vectors, (1, 0))

    with pytest.raises(RetrievalError, match=message):
        build_index((_chunk(0), _chunk(1)), provider)


@pytest.mark.parametrize(
    ("query_vector", "message"),
    [
        ((0, 0), "zero vector"),
        ((float("nan"), 0), "finite values"),
        ((1, 0, 0), "does not match"),
    ],
)
def test_query_rejects_invalid_vectors(
    query_vector: Sequence[float], message: str
) -> None:
    provider = FixedEmbeddingProvider(((1, 0),), query_vector)
    index = build_index((_chunk(0),), provider)

    with pytest.raises(RetrievalError, match=message):
        retrieve(index, "question", provider)


def test_empty_index_returns_no_results_without_embedding_a_query() -> None:
    provider = FixedEmbeddingProvider((), (1, 0))

    result = retrieve(build_index((), provider), " question ", provider)

    assert result.question == "question"
    assert not result.has_results
    assert provider.query_text is None


@pytest.mark.parametrize("question", ("", "  ", 7))
def test_blank_or_non_string_question_is_rejected(question: object) -> None:
    provider = FixedEmbeddingProvider(((1, 0),), (1, 0))
    index = build_index((_chunk(0),), provider)

    with pytest.raises(RetrievalError, match="Question"):
        retrieve(index, question, provider)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", (0, -1, True, 1.2))
def test_invalid_top_k_is_rejected(top_k: object) -> None:
    provider = FixedEmbeddingProvider(((1, 0),), (1, 0))
    index = build_index((_chunk(0),), provider)

    with pytest.raises(RetrievalError, match="top_k"):
        retrieve(index, "question", provider, top_k=top_k)  # type: ignore[arg-type]


def test_no_qualifying_score_is_a_clear_no_results_state() -> None:
    provider = FixedEmbeddingProvider(((1, 0),), (0, 1))
    result = retrieve(build_index((_chunk(0),), provider), "question", provider)

    assert not result.has_results
    assert result.sources == ()


def test_retrieved_source_preserves_complete_lineage_without_filesystem_paths(
    tmp_path: pytest.TempPathFactory,
) -> None:
    chunk = _chunk(3, "Unchanged citation source text.")
    provider = FixedEmbeddingProvider(((1, 0),), (1, 0))

    source = retrieve(build_index((chunk,), provider), "question", provider).sources[0]

    assert source.chunk_id == chunk.chunk_id
    assert source.document_id == chunk.document_id
    assert source.section_id == chunk.section_id
    assert source.source_locator == chunk.source_locator
    assert source.text == chunk.text
    assert source.content_hash == chunk.content_hash
    assert str(tmp_path) not in repr(source)


def test_provider_failure_does_not_echo_client_text() -> None:
    class FailingProvider:
        def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            raise RuntimeError("secret source text")

        def embed_query(self, text: str) -> Sequence[float]:
            raise RuntimeError("secret question")

    with pytest.raises(RetrievalError) as captured:
        build_index((_chunk(0),), FailingProvider())

    assert "secret source text" not in str(captured.value)
    assert captured.value.__cause__ is None


def test_query_provider_failure_does_not_echo_client_text() -> None:
    class QueryFailingProvider:
        def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            return ((1, 0),)

        def embed_query(self, text: str) -> Sequence[float]:
            raise RuntimeError("secret question")

    provider = QueryFailingProvider()
    index = build_index((_chunk(0),), provider)

    with pytest.raises(RetrievalError) as captured:
        retrieve(index, "secret question", provider)

    assert "secret question" not in str(captured.value)
    assert captured.value.__cause__ is None


def test_local_index_constructor_validates_embedded_dimensions() -> None:
    provider = FixedEmbeddingProvider(((1, 0), (0, 1)), (1, 0))
    index = build_index((_chunk(0), _chunk(1)), provider)

    assert isinstance(index, LocalVectorIndex)
    assert index.dimension == 2
