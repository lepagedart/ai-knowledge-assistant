"""Offline tests for the isolated OpenAI embedding provider."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from ai_knowledge_assistant.models import (
    DocumentChunk,
    DocumentType,
    SourceLocator,
    SourceLocatorKind,
)
from ai_knowledge_assistant.openai_embeddings import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    EmbeddingProviderError,
    EmbeddingProviderErrorCode,
    OpenAIEmbeddingProvider,
)
from ai_knowledge_assistant.retrieval import EmbeddingProvider, build_index, retrieve


@dataclass(frozen=True)
class FakeEmbedding:
    index: int
    embedding: Sequence[float]


@dataclass(frozen=True)
class FakeResponse:
    data: Sequence[FakeEmbedding]


class FakeEmbeddingsAPI:
    def __init__(self, responses: Sequence[FakeResponse | Exception]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, *, model: str, input: list[str]) -> FakeResponse:
        self.calls.append({"model": model, "input": input})
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeOpenAIClient:
    def __init__(self, responses: Sequence[FakeResponse | Exception]) -> None:
        self.embeddings = FakeEmbeddingsAPI(responses)


class PoisonedResponse:
    @property
    def data(self) -> Sequence[FakeEmbedding]:
        raise RuntimeError("private response details")


def _response(*vectors: Sequence[float]) -> FakeResponse:
    return FakeResponse(
        tuple(FakeEmbedding(index, vector) for index, vector in enumerate(vectors))
    )


def _chunk(index: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk-{index}",
        document_id="document-1",
        document_name="synthetic.md",
        document_type=DocumentType.MARKDOWN,
        section_id="section-1",
        chunk_index=index,
        text=text,
        source_locator=SourceLocator(
            kind=SourceLocatorKind.DOCUMENT_SECTION, section_label="Synthetic"
        ),
        source_char_start=0,
        source_char_end=len(text),
        primary_char_start=0,
        primary_char_end=len(text),
        source_section_content_hash="section-hash",
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        chunking_version="v1",
    )


def test_provider_satisfies_protocol_and_injected_construction_makes_no_call() -> None:
    client = FakeOpenAIClient(())
    provider: EmbeddingProvider = OpenAIEmbeddingProvider(client=client)

    assert provider.embed_documents  # Structural protocol methods are available.
    assert client.embeddings.calls == []


def test_environment_model_configuration_is_local_and_does_not_make_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "environment-model")
    client = FakeOpenAIClient(())
    provider = OpenAIEmbeddingProvider(client=client)

    assert provider.model == "environment-model"
    assert provider.batch_size == 100
    assert client.embeddings.calls == []


def test_default_model_is_local_and_does_not_make_api_call() -> None:
    client = FakeOpenAIClient(())

    provider = OpenAIEmbeddingProvider(client=client)

    assert provider.model == DEFAULT_OPENAI_EMBEDDING_MODEL
    assert client.embeddings.calls == []


def test_missing_api_key_fails_clearly_without_an_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(EmbeddingProviderError) as captured:
        OpenAIEmbeddingProvider()

    assert captured.value.code is EmbeddingProviderErrorCode.MISSING_API_KEY
    assert str(captured.value) == "MISSING_API_KEY"
    assert captured.value.__cause__ is None


def test_sdk_client_construction_with_a_supplied_key_makes_no_request() -> None:
    provider = OpenAIEmbeddingProvider(api_key="test-key-not-used-for-a-request")

    assert provider.model == DEFAULT_OPENAI_EMBEDDING_MODEL


@pytest.mark.parametrize("model", ("", "   ", 42))
def test_invalid_configuration_is_rejected_without_a_request(model: object) -> None:
    client = FakeOpenAIClient(())

    with pytest.raises(EmbeddingProviderError) as captured:
        OpenAIEmbeddingProvider(client=client, model=model)  # type: ignore[arg-type]

    assert captured.value.code is EmbeddingProviderErrorCode.INVALID_CONFIGURATION
    assert client.embeddings.calls == []


def test_documents_preserve_exact_text_order_and_request_shape() -> None:
    client = FakeOpenAIClient((_response((1, 0), (0, 1)),))
    provider = OpenAIEmbeddingProvider(client=client, model="test-model")
    texts = ("  Exact chunk one.  ", "Exact\nchunk two.")

    vectors = provider.embed_documents(texts)

    assert vectors == ((1.0, 0.0), (0.0, 1.0))
    assert client.embeddings.calls == [{"model": "test-model", "input": list(texts)}]


def test_documents_batch_and_reassemble_in_original_order() -> None:
    client = FakeOpenAIClient(
        (_response((1, 0), (2, 0)), _response((3, 0), (4, 0)), _response((5, 0)))
    )
    provider = OpenAIEmbeddingProvider(client=client, batch_size=2)

    vectors = provider.embed_documents(("one", "two", "three", "four", "five"))

    assert vectors == ((1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (5.0, 0.0))
    assert [call["input"] for call in client.embeddings.calls] == [
        ["one", "two"],
        ["three", "four"],
        ["five"],
    ]


@pytest.mark.parametrize("texts", ((), ("valid", "   "), "not a sequence"))
def test_invalid_document_texts_do_not_call_the_client(texts: object) -> None:
    client = FakeOpenAIClient(())
    provider = OpenAIEmbeddingProvider(client=client)

    with pytest.raises(EmbeddingProviderError) as captured:
        provider.embed_documents(texts)  # type: ignore[arg-type]

    assert captured.value.code is EmbeddingProviderErrorCode.INVALID_CONFIGURATION
    assert client.embeddings.calls == []


def test_query_preserves_exact_text_and_expects_one_vector() -> None:
    client = FakeOpenAIClient((_response((0.25, 0.75)),))
    provider = OpenAIEmbeddingProvider(client=client)
    question = "  Exact question?  "

    assert provider.embed_query(question) == (0.25, 0.75)
    assert client.embeddings.calls[0]["input"] == [question]


@pytest.mark.parametrize("question", ("", "  ", 99))
def test_invalid_queries_do_not_call_the_client(question: object) -> None:
    client = FakeOpenAIClient(())
    provider = OpenAIEmbeddingProvider(client=client)

    with pytest.raises(EmbeddingProviderError) as captured:
        provider.embed_query(question)  # type: ignore[arg-type]

    assert captured.value.code is EmbeddingProviderErrorCode.INVALID_CONFIGURATION
    assert client.embeddings.calls == []


@pytest.mark.parametrize(
    "response",
    (
        FakeResponse(()),
        FakeResponse((FakeEmbedding(1, (1, 0)),)),
        FakeResponse((FakeEmbedding(0, ()),)),
        FakeResponse((FakeEmbedding(0, (float("nan"),)),)),
        FakeResponse((FakeEmbedding(0, (float("inf"),)),)),
        FakeResponse((FakeEmbedding(0, (1, 0)), FakeEmbedding(1, (1, 0, 0)))),
    ),
)
def test_malformed_responses_are_sanitized(response: FakeResponse) -> None:
    client = FakeOpenAIClient((response,))
    provider = OpenAIEmbeddingProvider(client=client)

    with pytest.raises(EmbeddingProviderError) as captured:
        provider.embed_query("secret question")

    assert captured.value.code is EmbeddingProviderErrorCode.INVALID_EMBEDDING_RESPONSE
    assert "secret question" not in str(captured.value)


def test_document_response_with_inconsistent_dimensions_is_rejected() -> None:
    client = FakeOpenAIClient((_response((1, 0), (1, 0, 0)),))
    provider = OpenAIEmbeddingProvider(client=client)

    with pytest.raises(EmbeddingProviderError) as captured:
        provider.embed_documents(("first", "second"))

    assert captured.value.code is EmbeddingProviderErrorCode.INVALID_EMBEDDING_RESPONSE


def test_uninspectable_response_is_sanitized() -> None:
    client = FakeOpenAIClient((PoisonedResponse(),))  # type: ignore[arg-type]
    provider = OpenAIEmbeddingProvider(client=client)

    with pytest.raises(EmbeddingProviderError) as captured:
        provider.embed_query("secret question")

    assert captured.value.code is EmbeddingProviderErrorCode.INVALID_EMBEDDING_RESPONSE
    assert "private response details" not in str(captured.value)
    assert captured.value.__cause__ is None


def test_sdk_failure_is_sanitized_without_retaining_request_details() -> None:
    client = FakeOpenAIClient((RuntimeError("key=secret text=private document"),))
    provider = OpenAIEmbeddingProvider(client=client)

    with pytest.raises(EmbeddingProviderError) as captured:
        provider.embed_documents(("private document",))

    assert captured.value.code is EmbeddingProviderErrorCode.EMBEDDING_REQUEST_FAILED
    assert str(captured.value) == "EMBEDDING_REQUEST_FAILED"
    assert "private document" not in str(captured.value)
    assert captured.value.__cause__ is None


def test_retrieval_works_with_concrete_provider_and_mocked_client() -> None:
    client = FakeOpenAIClient((_response((1, 0), (0, 1)), _response((1, 0))))
    provider = OpenAIEmbeddingProvider(client=client)
    index = build_index((_chunk(0, "first"), _chunk(1, "second")), provider)

    result = retrieve(index, "question", provider, minimum_score=0)

    assert [source.chunk_id for source in result.sources] == ["chunk-0", "chunk-1"]
