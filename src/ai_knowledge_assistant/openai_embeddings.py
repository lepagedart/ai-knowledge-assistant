"""OpenAI-backed embeddings behind the local retrieval provider protocol.

This module intentionally owns all OpenAI SDK interaction.  It neither creates
an index nor logs client text, and constructing a provider never makes an API
request.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from enum import StrEnum
from numbers import Real
from typing import Any

DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_BATCH_SIZE = 100


class EmbeddingProviderErrorCode(StrEnum):
    """Stable, sanitized public failure codes for the embedding boundary."""

    MISSING_API_KEY = "MISSING_API_KEY"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    EMBEDDING_REQUEST_FAILED = "EMBEDDING_REQUEST_FAILED"
    INVALID_EMBEDDING_RESPONSE = "INVALID_EMBEDDING_RESPONSE"


class EmbeddingProviderError(ValueError):
    """A safe provider error whose message never includes SDK request details."""

    def __init__(self, code: EmbeddingProviderErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


class OpenAIEmbeddingProvider:
    """Create validated embeddings with the official OpenAI Python SDK.

    A client can be injected for deterministic tests or custom application
    wiring.  Without one, the provider reads ``OPENAI_API_KEY`` and optionally
    ``OPENAI_EMBEDDING_MODEL`` from the environment, then constructs an SDK
    client without making a request.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> None:
        self._model = _validated_model(
            model if model is not None else os.environ.get("OPENAI_EMBEDDING_MODEL")
        )
        self._batch_size = _validated_batch_size(batch_size)
        self._client = client if client is not None else self._build_client(api_key)

    @property
    def model(self) -> str:
        """The configured embedding model identifier."""
        return self._model

    @property
    def batch_size(self) -> int:
        """Maximum document texts sent in one embedding request."""
        return self._batch_size

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Embed exact, non-blank chunk text in order using bounded batches."""
        document_texts = _validated_document_texts(texts)
        vectors: list[tuple[float, ...]] = []
        expected_dimension: int | None = None
        for start in range(0, len(document_texts), self._batch_size):
            batch = document_texts[start : start + self._batch_size]
            batch_vectors = self._embed_batch(batch)
            batch_dimension = len(batch_vectors[0])
            if expected_dimension is not None and batch_dimension != expected_dimension:
                raise EmbeddingProviderError(
                    EmbeddingProviderErrorCode.INVALID_EMBEDDING_RESPONSE
                )
            expected_dimension = batch_dimension
            vectors.extend(batch_vectors)
        return tuple(vectors)

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embed one exact, validated question without semantic rewriting."""
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingProviderError(
                EmbeddingProviderErrorCode.INVALID_CONFIGURATION
            )
        return self._embed_batch((text,))[0]

    def _build_client(self, api_key: str | None) -> Any:
        resolved_key = (
            api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        )
        if not isinstance(resolved_key, str) or not resolved_key.strip():
            raise EmbeddingProviderError(EmbeddingProviderErrorCode.MISSING_API_KEY)
        try:
            from openai import OpenAI
        except ImportError:  # pragma: no cover
            raise EmbeddingProviderError(
                EmbeddingProviderErrorCode.INVALID_CONFIGURATION
            ) from None
        return OpenAI(api_key=resolved_key, max_retries=0)

    def _embed_batch(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        try:
            response = self._client.embeddings.create(
                model=self._model, input=list(texts)
            )
        except Exception:
            raise EmbeddingProviderError(
                EmbeddingProviderErrorCode.EMBEDDING_REQUEST_FAILED
            ) from None
        try:
            return _validated_response(response, expected_count=len(texts))
        except EmbeddingProviderError:
            raise
        except Exception:
            raise EmbeddingProviderError(
                EmbeddingProviderErrorCode.INVALID_EMBEDDING_RESPONSE
            ) from None


def _validated_model(model: str | None) -> str:
    if model is None:
        return DEFAULT_OPENAI_EMBEDDING_MODEL
    if not isinstance(model, str) or not model.strip():
        raise EmbeddingProviderError(EmbeddingProviderErrorCode.INVALID_CONFIGURATION)
    return model


def _validated_batch_size(batch_size: int) -> int:
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise EmbeddingProviderError(EmbeddingProviderErrorCode.INVALID_CONFIGURATION)
    return batch_size


def _validated_document_texts(texts: Sequence[str]) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)):
        raise EmbeddingProviderError(EmbeddingProviderErrorCode.INVALID_CONFIGURATION)
    try:
        values = tuple(texts)
    except TypeError:
        raise EmbeddingProviderError(
            EmbeddingProviderErrorCode.INVALID_CONFIGURATION
        ) from None
    if not values or any(
        not isinstance(text, str) or not text.strip() for text in values
    ):
        raise EmbeddingProviderError(EmbeddingProviderErrorCode.INVALID_CONFIGURATION)
    return values


def _validated_response(
    response: Any, *, expected_count: int
) -> tuple[tuple[float, ...], ...]:
    data = getattr(response, "data", None)
    if isinstance(data, (str, bytes)):
        raise EmbeddingProviderError(
            EmbeddingProviderErrorCode.INVALID_EMBEDDING_RESPONSE
        )
    try:
        records = tuple(data)
    except TypeError:
        raise EmbeddingProviderError(
            EmbeddingProviderErrorCode.INVALID_EMBEDDING_RESPONSE
        ) from None
    if len(records) != expected_count:
        raise EmbeddingProviderError(
            EmbeddingProviderErrorCode.INVALID_EMBEDDING_RESPONSE
        )
    vectors: list[tuple[float, ...]] = []
    dimension: int | None = None
    for expected_index, record in enumerate(records):
        if getattr(record, "index", None) != expected_index:
            raise EmbeddingProviderError(
                EmbeddingProviderErrorCode.INVALID_EMBEDDING_RESPONSE
            )
        vector = _validated_vector(getattr(record, "embedding", None))
        if dimension is not None and len(vector) != dimension:
            raise EmbeddingProviderError(
                EmbeddingProviderErrorCode.INVALID_EMBEDDING_RESPONSE
            )
        dimension = len(vector)
        vectors.append(vector)
    return tuple(vectors)


def _validated_vector(vector: Any) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes)):
        raise EmbeddingProviderError(
            EmbeddingProviderErrorCode.INVALID_EMBEDDING_RESPONSE
        )
    try:
        values = tuple(vector)
    except TypeError:
        raise EmbeddingProviderError(
            EmbeddingProviderErrorCode.INVALID_EMBEDDING_RESPONSE
        ) from None
    if not values or any(
        not isinstance(value, Real) or isinstance(value, bool) for value in values
    ):
        raise EmbeddingProviderError(
            EmbeddingProviderErrorCode.INVALID_EMBEDDING_RESPONSE
        )
    normalized = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in normalized):
        raise EmbeddingProviderError(
            EmbeddingProviderErrorCode.INVALID_EMBEDDING_RESPONSE
        )
    return normalized
