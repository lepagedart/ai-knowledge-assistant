"""OpenAI Responses API adapter for the grounded answer provider boundary."""

from __future__ import annotations

import json
import os
from enum import StrEnum
from typing import Any

from .answer_generation import build_grounded_prompt
from .models import ProviderAnswer, ProviderCitation, RetrievedSource

DEFAULT_OPENAI_ANSWER_MODEL = "gpt-5.6-luna"

ANSWER_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "answer", "citations"],
    "properties": {
        "status": {"type": "string", "enum": ["supported", "unsupported"]},
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["chunk_id", "claim"],
                "properties": {
                    "chunk_id": {"type": "string"},
                    "claim": {"type": "string"},
                },
            },
        },
    },
}


class AnswerProviderErrorCode(StrEnum):
    MISSING_API_KEY = "MISSING_API_KEY"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    ANSWER_REQUEST_FAILED = "ANSWER_REQUEST_FAILED"
    INVALID_ANSWER_RESPONSE = "INVALID_ANSWER_RESPONSE"


class AnswerProviderError(ValueError):
    """Sanitized provider boundary error; no SDK detail is retained."""

    def __init__(self, code: AnswerProviderErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


class OpenAIAnswerProvider:
    """One structured Responses API request per provider invocation, with no retries."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._model = _validated_model(
            model if model is not None else os.environ.get("OPENAI_ANSWER_MODEL")
        )
        self._client = client if client is not None else self._build_client(api_key)

    @property
    def model(self) -> str:
        return self._model

    def generate_answer(
        self, question: str, sources: tuple[RetrievedSource, ...]
    ) -> ProviderAnswer:
        prompt = build_grounded_prompt(question, sources)
        try:
            response = self._client.responses.create(
                model=self._model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "grounded_answer",
                        "strict": True,
                        "schema": ANSWER_RESPONSE_SCHEMA,
                    }
                },
            )
        except Exception:
            raise AnswerProviderError(
                AnswerProviderErrorCode.ANSWER_REQUEST_FAILED
            ) from None
        try:
            return _parse_response(response)
        except AnswerProviderError:
            raise
        except Exception:
            raise AnswerProviderError(
                AnswerProviderErrorCode.INVALID_ANSWER_RESPONSE
            ) from None

    def _build_client(self, api_key: str | None) -> Any:
        resolved_key = (
            api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        )
        if not isinstance(resolved_key, str) or not resolved_key.strip():
            raise AnswerProviderError(AnswerProviderErrorCode.MISSING_API_KEY)
        try:
            from openai import OpenAI
        except ImportError:  # pragma: no cover
            raise AnswerProviderError(
                AnswerProviderErrorCode.INVALID_CONFIGURATION
            ) from None
        return OpenAI(api_key=resolved_key, max_retries=0)


def _validated_model(model: str | None) -> str:
    if model is None:
        return DEFAULT_OPENAI_ANSWER_MODEL
    if not isinstance(model, str) or not model.strip():
        raise AnswerProviderError(AnswerProviderErrorCode.INVALID_CONFIGURATION)
    return model


def _parse_response(response: Any) -> ProviderAnswer:
    raw = getattr(response, "output_text", None)
    if not isinstance(raw, str):
        raise AnswerProviderError(AnswerProviderErrorCode.INVALID_ANSWER_RESPONSE)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise AnswerProviderError(AnswerProviderErrorCode.INVALID_ANSWER_RESPONSE)
    citations = parsed.get("citations")
    if isinstance(citations, list):
        parsed_citations: object = tuple(
            ProviderCitation(item.get("chunk_id"), item.get("claim"))
            if isinstance(item, dict)
            else item
            for item in citations
        )
    else:
        parsed_citations = citations
    return ProviderAnswer(
        status=parsed.get("status"),
        answer=parsed.get("answer"),
        citations=parsed_citations,
    )
