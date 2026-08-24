"""Offline tests for the isolated OpenAI Responses API answer adapter."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ai_knowledge_assistant.answer_generation import AnswerProvider
from ai_knowledge_assistant.openai_answers import (
    DEFAULT_OPENAI_ANSWER_MODEL,
    AnswerProviderError,
    AnswerProviderErrorCode,
    OpenAIAnswerProvider,
)


def _source() -> object:
    # Import locally so tests remain independent from package installation layout.
    from ai_knowledge_assistant.models import (
        DocumentType,
        RetrievedSource,
        SourceLocator,
        SourceLocatorKind,
    )

    return RetrievedSource(
        rank=1, score=1.0, chunk_id="chunk-0", document_id="document-0",
        document_name="Document 0.md", document_type=DocumentType.MARKDOWN,
        section_id="section-0", chunk_index=0,
        source_locator=SourceLocator(kind=SourceLocatorKind.DOCUMENT_SECTION),
        text="Evidence.", content_hash="hash", source_section_content_hash="hash",
        source_char_start=0, source_char_end=9, primary_char_start=0,
        primary_char_end=9,
    )


@dataclass(frozen=True)
class FakeResponse:
    output_text: object


class FakeResponsesAPI:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeClient:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.responses = FakeResponsesAPI(response)


def test_construction_is_protocol_compatible_and_makes_no_request() -> None:
    client = FakeClient(FakeResponse("{}"))
    provider: AnswerProvider = OpenAIAnswerProvider(client=client)

    assert provider.generate_answer
    assert DEFAULT_OPENAI_ANSWER_MODEL == "gpt-5.6-luna"
    assert provider.model == DEFAULT_OPENAI_ANSWER_MODEL
    assert client.responses.calls == []


def test_environment_model_override_is_local_and_makes_no_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_ANSWER_MODEL", "configured-answer-model")
    client = FakeClient(FakeResponse("{}"))

    provider = OpenAIAnswerProvider(client=client)

    assert provider.model == "configured-answer-model"
    assert client.responses.calls == []


def test_responses_request_is_structured_and_exactly_once() -> None:
    client = FakeClient(
        FakeResponse('{"status":"supported","answer":"answer","citations":[{"chunk_id":"chunk-0","claim":"claim"}]}')
    )
    provider = OpenAIAnswerProvider(client=client, model="answer-model")

    response = provider.generate_answer("question", (_source(),))  # type: ignore[arg-type]

    assert response.status == "supported"
    assert response.citations[0].chunk_id == "chunk-0"
    assert len(client.responses.calls) == 1
    call = client.responses.calls[0]
    assert call["model"] == "answer-model"
    assert call["text"]["format"]["type"] == "json_schema"  # type: ignore[index]
    assert "SOURCE chunk_id=chunk-0" in call["input"]  # type: ignore[operator]


@pytest.mark.parametrize(
    "response, code",
    [
        (FakeResponse("not json"), AnswerProviderErrorCode.INVALID_ANSWER_RESPONSE),
        (FakeResponse("[]"), AnswerProviderErrorCode.INVALID_ANSWER_RESPONSE),
        (
            RuntimeError("api key=secret question=private"),
            AnswerProviderErrorCode.ANSWER_REQUEST_FAILED,
        ),
    ],
)
def test_response_and_request_failures_are_sanitized(
    response: FakeResponse | Exception, code: AnswerProviderErrorCode
) -> None:
    client = FakeClient(response)
    provider = OpenAIAnswerProvider(client=client)

    with pytest.raises(AnswerProviderError) as captured:
        provider.generate_answer("secret question", (_source(),))  # type: ignore[arg-type]

    assert captured.value.code is code
    assert "secret" not in str(captured.value)
    assert len(client.responses.calls) == 1


def test_missing_key_and_invalid_model_do_not_make_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(AnswerProviderError) as captured:
        OpenAIAnswerProvider()
    assert captured.value.code is AnswerProviderErrorCode.MISSING_API_KEY

    client = FakeClient(FakeResponse("{}"))
    with pytest.raises(AnswerProviderError) as captured:
        OpenAIAnswerProvider(client=client, model=" ")
    assert captured.value.code is AnswerProviderErrorCode.INVALID_CONFIGURATION
    assert client.responses.calls == []
