"""Inspectable, in-memory vector retrieval with path-free citation lineage."""

from __future__ import annotations

import math
from numbers import Real
from typing import Protocol, Sequence

import numpy as np

from .models import DocumentChunk, EmbeddedChunk, RetrievalResult, RetrievedSource


class EmbeddingProvider(Protocol):
    """Provider boundary for embeddings; implementations must not mutate text."""

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return exactly one vector for every supplied document text."""

    def embed_query(self, text: str) -> Sequence[float]:
        """Return one vector for a validated query."""


class RetrievalError(ValueError):
    """A safe, stable retrieval-boundary validation failure."""


class LocalVectorIndex:
    """Read-only, process-local vectors for a modest business document set.

    The index deliberately has no persistence or filesystem dependency. Entries
    preserve their build order; equal scores are resolved by that order.
    """

    def __init__(self, entries: Sequence[EmbeddedChunk]) -> None:
        self._entries = tuple(entries)
        if not self._entries:
            self._dimension: int | None = None
            self._matrix = np.empty((0, 0), dtype=np.float64)
            self._norms = np.empty(0, dtype=np.float64)
            return
        vectors = [
            _validated_vector(entry.embedding, "document embedding")
            for entry in entries
        ]
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise RetrievalError("Document embedding dimensions must match.")
        self._dimension = dimension
        self._matrix = np.asarray(vectors, dtype=np.float64)
        self._norms = np.linalg.norm(self._matrix, axis=1)
        if not np.all(np.isfinite(self._norms)) or np.any(self._norms == 0.0):
            raise RetrievalError(
                "Document embedding norms must be finite and non-zero."
            )
        self._matrix.setflags(write=False)
        self._norms.setflags(write=False)

    @property
    def entries(self) -> tuple[EmbeddedChunk, ...]:
        """Ordered immutable records used by this process-local index."""
        return self._entries

    @property
    def dimension(self) -> int | None:
        """Embedding dimension, or ``None`` when no records were indexed."""
        return self._dimension

    def search(
        self, query_vector: Sequence[float], *, top_k: int
    ) -> tuple[tuple[int, float], ...]:
        """Return build-order indexes and cosine scores in deterministic rank order."""
        _validate_top_k(top_k)
        if not self._entries:
            return ()
        query = _validated_vector(query_vector, "query embedding")
        if len(query) != self._dimension:
            raise RetrievalError("Query embedding dimension does not match the index.")
        query_array = np.asarray(query, dtype=np.float64)
        query_norm = float(np.linalg.norm(query_array))
        # Document zero vectors are rejected during build, so only query can be zero.
        if not math.isfinite(query_norm) or query_norm == 0.0:
            raise RetrievalError("Query embedding norm must be finite and non-zero.")
        scores = (self._matrix @ query_array) / (self._norms * query_norm)
        if not np.all(np.isfinite(scores)):
            raise RetrievalError("Cosine similarity could not be computed safely.")
        ordered = sorted(
            enumerate(scores.tolist()), key=lambda item: (-item[1], item[0])
        )
        return tuple((position, float(score)) for position, score in ordered[:top_k])


def build_index(
    chunks: Sequence[DocumentChunk], embedding_provider: EmbeddingProvider
) -> LocalVectorIndex:
    """Embed exact chunk text and build a validated, in-memory local index."""
    ordered_chunks = tuple(chunks)
    texts = tuple(chunk.text for chunk in ordered_chunks)
    if not texts:
        return LocalVectorIndex(())
    try:
        vectors = embedding_provider.embed_documents(texts)
    except Exception:
        raise RetrievalError("Document embedding provider failed.") from None
    try:
        vector_count = len(vectors)
    except TypeError as error:
        raise RetrievalError(
            "Document embedding provider returned an unexpected count."
        ) from error
    if vector_count != len(ordered_chunks):
        raise RetrievalError(
            "Document embedding provider returned an unexpected count."
        )
    entries = tuple(
        _embedded_chunk(chunk, _validated_vector(vector, "document embedding"))
        for chunk, vector in zip(ordered_chunks, vectors, strict=True)
    )
    return LocalVectorIndex(entries)


def retrieve(
    index: LocalVectorIndex,
    question: str,
    embedding_provider: EmbeddingProvider,
    *,
    top_k: int = 5,
    minimum_score: float = 0.2,
) -> RetrievalResult:
    """Retrieve local evidence candidates; this function never generates answers.

    Questions are normalized by trimming leading and trailing whitespace. V1's
    default 0.2 threshold is intentionally conservative for candidate discovery,
    not a statement that an answer is supported or factually correct.
    """
    if not isinstance(question, str) or not (normalized_question := question.strip()):
        raise RetrievalError("Question must contain non-whitespace text.")
    _validate_top_k(top_k)
    _validate_minimum_score(minimum_score)
    if not index.entries:
        return RetrievalResult(question=normalized_question, sources=())
    try:
        query_vector = embedding_provider.embed_query(normalized_question)
    except Exception:
        raise RetrievalError("Query embedding provider failed.") from None
    ranked = index.search(query_vector, top_k=top_k)
    qualifying = tuple(
        _retrieved_source(index.entries[position], rank, score)
        for rank, (position, score) in enumerate(ranked, start=1)
        if score >= minimum_score
    )
    return RetrievalResult(question=normalized_question, sources=qualifying)


def _validated_vector(vector: Sequence[float], label: str) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes)):
        raise RetrievalError(
            f"{label.capitalize()} must be a non-empty numeric vector."
        )
    try:
        values = tuple(vector)
    except TypeError as error:
        raise RetrievalError(
            f"{label.capitalize()} must be a non-empty numeric vector."
        ) from error
    if not values or any(
        not isinstance(value, Real) or isinstance(value, bool) for value in values
    ):
        raise RetrievalError(
            f"{label.capitalize()} must be a non-empty numeric vector."
        )
    normalized = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in normalized):
        raise RetrievalError(f"{label.capitalize()} must contain only finite values.")
    if not any(value != 0.0 for value in normalized):
        raise RetrievalError(f"{label.capitalize()} must not be a zero vector.")
    return normalized


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise RetrievalError("top_k must be a positive integer.")


def _validate_minimum_score(minimum_score: float) -> None:
    if (
        not isinstance(minimum_score, Real)
        or isinstance(minimum_score, bool)
        or not math.isfinite(float(minimum_score))
        or not -1.0 <= float(minimum_score) <= 1.0
    ):
        raise RetrievalError("minimum_score must be a finite value from -1 to 1.")


def _embedded_chunk(
    chunk: DocumentChunk, embedding: tuple[float, ...]
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_name=chunk.document_name,
        document_type=chunk.document_type,
        section_id=chunk.section_id,
        chunk_index=chunk.chunk_index,
        source_locator=chunk.source_locator,
        text=chunk.text,
        content_hash=chunk.content_hash,
        source_section_content_hash=chunk.source_section_content_hash,
        source_char_start=chunk.source_char_start,
        source_char_end=chunk.source_char_end,
        primary_char_start=chunk.primary_char_start,
        primary_char_end=chunk.primary_char_end,
        embedding=embedding,
    )


def _retrieved_source(entry: EmbeddedChunk, rank: int, score: float) -> RetrievedSource:
    return RetrievedSource(
        rank=rank,
        score=score,
        chunk_id=entry.chunk_id,
        document_id=entry.document_id,
        document_name=entry.document_name,
        document_type=entry.document_type,
        section_id=entry.section_id,
        chunk_index=entry.chunk_index,
        source_locator=entry.source_locator,
        text=entry.text,
        content_hash=entry.content_hash,
        source_section_content_hash=entry.source_section_content_hash,
        source_char_start=entry.source_char_start,
        source_char_end=entry.source_char_end,
        primary_char_start=entry.primary_char_start,
        primary_char_end=entry.primary_char_end,
    )
