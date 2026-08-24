"""End-to-end deterministic retrieval over the fictional Harbor & Hearth corpus."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from ai_knowledge_assistant.answer_generation import generate_grounded_answer
from ai_knowledge_assistant.chunking import chunk_document
from ai_knowledge_assistant.extraction import extract_document
from ai_knowledge_assistant.models import (
    GroundedAnswerStatus,
    ProviderAnswer,
    ProviderCitation,
    RetrievedSource,
)
from ai_knowledge_assistant.retrieval import LocalVectorIndex, build_index, retrieve
from ai_knowledge_assistant.uploads import accept_upload
from ai_knowledge_assistant.workspace import UploadWorkspace

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIRECTORY = ROOT / "demo_documents" / "harbor_and_hearth"
_WORD = re.compile(r"[a-z]+")


class KeywordTestEmbeddingProvider:
    """A transparent fixed lexical map for offline integration tests only.

    Dimensions are, in order: call-out/shift, cash/drawer/opening, allergen,
    refund/recovery, a document-only Harbor marker, an opening-cash-drawer phrase,
    and an unsupported-query-only dimension. Binary values make expected cosine
    ranking easy to inspect.
    """

    _groups = (
        frozenset({"call", "callout", "calling", "attendance", "shift"}),
        frozenset({"cash", "drawer", "opening", "opener", "float", "register"}),
        frozenset({"allergen", "allergy", "allergic"}),
        frozenset({"refund", "refunds", "recovery", "replacement", "discount"}),
        frozenset({"harbor"}),
    )

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return tuple(self._document_vector(text) for text in texts)

    def embed_query(self, text: str) -> Sequence[float]:
        words = set(_WORD.findall(text.lower()))
        vector = [float(bool(words & group)) for group in self._groups]
        if not any(vector):
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        cash_drawer = float({"cash", "drawer"} <= words)
        return (*vector, cash_drawer, 0.0)

    def _document_vector(self, text: str) -> tuple[float, ...]:
        words = set(_WORD.findall(text.lower()))
        vector = tuple(float(bool(words & group)) for group in self._groups)
        assert any(vector), "Test lexical mapping must cover every demo chunk."
        opening_cash_drawer = float("opening the cash drawer" in text.lower())
        return (*vector, opening_cash_drawer, 0.0)


class ScenarioAnswerProvider:
    """Deterministic answer provider proving the answer layer wiring offline."""

    def __init__(self, source_ids: tuple[str, ...]) -> None:
        self.source_ids = source_ids
        self.calls = 0

    def generate_answer(
        self, question: str, sources: tuple[RetrievedSource, ...]
    ) -> ProviderAnswer:
        del question, sources
        self.calls += 1
        return ProviderAnswer(
            "supported",
            "Supported by the selected Harbor & Hearth material.",
            tuple(
                ProviderCitation(chunk_id, "Supported operational guidance.")
                for chunk_id in self.source_ids
            ),
        )


def _demo_index(
    tmp_path: Path,
) -> tuple[LocalVectorIndex, KeywordTestEmbeddingProvider]:
    workspace = UploadWorkspace.create(tmp_path)
    chunks = []
    for source_path in sorted(DEMO_DIRECTORY.glob("*.md")):
        accepted = accept_upload(workspace, source_path.name, source_path.read_bytes())
        extracted = extract_document(workspace, accepted)
        chunks.extend(chunk_document(extracted))
    provider = KeywordTestEmbeddingProvider()
    return build_index(tuple(chunks), provider), provider


def test_harbor_and_hearth_callout_question_retrieves_callout_policy(
    tmp_path: Path,
) -> None:
    index, provider = _demo_index(tmp_path)

    result = retrieve(
        index,
        "How far in advance should I call out for a shift?",
        provider,
        minimum_score=0.5,
    )

    assert result.sources[0].document_name == "callout_attendance_policy.md"
    assert "at least two hours" in result.sources[0].text


def test_harbor_and_hearth_cash_question_retrieves_opening_sop(tmp_path: Path) -> None:
    index, provider = _demo_index(tmp_path)

    result = retrieve(
        index,
        "What are the opening cash-drawer steps?",
        provider,
        minimum_score=0.5,
    )

    assert result.sources[0].document_name == "opening_closing_sop.md"
    assert "opening float" in result.sources[0].text.lower()


def test_harbor_and_hearth_allergen_refund_question_surfaces_both_documents(
    tmp_path: Path,
) -> None:
    index, provider = _demo_index(tmp_path)

    result = retrieve(
        index,
        (
            "What should a team member do if a guest asks for a refund because "
            "of an allergen concern?"
        ),
        provider,
        top_k=8,
        minimum_score=0.5,
    )

    document_names = {source.document_name for source in result.sources}
    assert "refund_service_recovery_policy.md" in document_names
    assert "menu_product_reference.md" in document_names


def test_harbor_and_hearth_unsupported_question_has_no_qualifying_source(
    tmp_path: Path,
) -> None:
    index, provider = _demo_index(tmp_path)

    result = retrieve(
        index,
        "What is the CEO's home address?",
        provider,
        minimum_score=0.2,
    )

    assert not result.has_results


def test_harbor_and_hearth_answer_pipeline_supports_callout_policy(
    tmp_path: Path,
) -> None:
    index, embedding_provider = _demo_index(tmp_path)
    question = "How far in advance should I call out for a shift?"
    result = retrieve(index, question, embedding_provider, minimum_score=0.5)
    provider = ScenarioAnswerProvider((result.sources[0].chunk_id,))

    answer = generate_grounded_answer(question, result.sources, provider)

    assert answer.status is GroundedAnswerStatus.SUPPORTED
    assert answer.citations[0].document_name == "callout_attendance_policy.md"
    assert provider.calls == 1


def test_harbor_and_hearth_answer_pipeline_supports_opening_sop(tmp_path: Path) -> None:
    index, embedding_provider = _demo_index(tmp_path)
    question = "What are the opening cash-drawer steps?"
    result = retrieve(index, question, embedding_provider, minimum_score=0.5)

    answer = generate_grounded_answer(
        question, result.sources, ScenarioAnswerProvider((result.sources[0].chunk_id,))
    )

    assert answer.status is GroundedAnswerStatus.SUPPORTED
    assert answer.citations[0].document_name == "opening_closing_sop.md"


def test_harbor_and_hearth_answer_pipeline_supports_refund_allergen_synthesis(
    tmp_path: Path,
) -> None:
    index, embedding_provider = _demo_index(tmp_path)
    question = (
        "What should a team member do if a guest asks for a refund because of an "
        "allergen concern?"
    )
    result = retrieve(index, question, embedding_provider, top_k=8, minimum_score=0.5)
    selected = tuple(
        source.chunk_id
        for source in result.sources
        if source.document_name
        in {"refund_service_recovery_policy.md", "menu_product_reference.md"}
    )

    answer = generate_grounded_answer(
        question, result.sources, ScenarioAnswerProvider(selected)
    )

    assert answer.status is GroundedAnswerStatus.SUPPORTED
    assert {citation.document_name for citation in answer.citations} == {
        "refund_service_recovery_policy.md",
        "menu_product_reference.md",
    }
