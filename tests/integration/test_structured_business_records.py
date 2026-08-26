"""Offline end-to-end coverage for fictional structured business evidence."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from ai_knowledge_assistant.answer_generation import generate_grounded_answer
from ai_knowledge_assistant.chunking import chunk_document
from ai_knowledge_assistant.extraction import extract_document, read_accepted_content
from ai_knowledge_assistant.models import (
    DocumentType,
    GroundedAnswerStatus,
    ProviderAnswer,
    ProviderCitation,
    RetrievedSource,
    StructuredRecordType,
)
from ai_knowledge_assistant.retrieval import build_index, retrieve
from ai_knowledge_assistant.structured_records import (
    parse_structured_document,
    structured_evidence,
)
from ai_knowledge_assistant.uploads import accept_upload
from ai_knowledge_assistant.web import DEMO_DIRECTORY, _locator_text
from ai_knowledge_assistant.workspace import UploadWorkspace

WORD = re.compile(r"[a-z0-9]+")
VOCABULARY = (
    "inv",
    "invoice",
    "1048",
    "1050",
    "gin",
    "london",
    "dry",
    "reserve",
    "tonic",
    "221",
    "quantity",
    "sku",
    "001",
    "granite",
    "state",
    "beverage",
    "vendor",
    "products",
    "product",
    "patio",
    "call",
    "shift",
    "cash",
    "drawer",
    "opening",
    "allergen",
    "refund",
)


class StructuredKeywordEmbeddingProvider:
    """A transparent lexical embedding provider for deterministic offline tests."""

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return tuple(self._vector(text) for text in texts)

    def embed_query(self, text: str) -> Sequence[float]:
        return self._vector(text)

    def _vector(self, text: str) -> tuple[float, ...]:
        words = set(WORD.findall(text.lower()))
        return tuple(float(word in words) for word in VOCABULARY) + (1.0,)


class FixedAnswerProvider:
    def __init__(self, answer: str, source_ids: tuple[str, ...]) -> None:
        self.answer = answer
        self.source_ids = source_ids

    def generate_answer(
        self, question: str, sources: tuple[RetrievedSource, ...]
    ) -> ProviderAnswer:
        del question, sources
        return ProviderAnswer(
            "supported",
            self.answer,
            tuple(
                ProviderCitation(source_id, "Direct structured evidence.")
                for source_id in self.source_ids
            ),
        )


def _index(tmp_path: Path, *, generic: bool = False):
    workspace = UploadWorkspace.create(tmp_path)
    accepted_documents = []
    for path in sorted(DEMO_DIRECTORY.iterdir()):
        if path.is_file():
            accepted_documents.append(
                accept_upload(workspace, path.name, path.read_bytes())
            )
    if generic:
        accepted_documents.append(
            accept_upload(
                workspace,
                "fictional_service_log.csv",
                b"Station,Checked By,Status\nPatio,Avery,Ready\n",
            )
        )
    chunks = []
    structured = []
    for accepted in accepted_documents:
        if accepted.document_type in {DocumentType.CSV, DocumentType.XLSX}:
            parsed = parse_structured_document(
                accepted, read_accepted_content(workspace, accepted)
            )
            structured.append(parsed)
            chunks.extend(chunk_document(structured_evidence(parsed)))
        else:
            chunks.extend(chunk_document(extract_document(workspace, accepted)))
    provider = StructuredKeywordEmbeddingProvider()
    return (
        workspace,
        tuple(accepted_documents),
        tuple(structured),
        tuple(chunks),
        build_index(tuple(chunks), provider),
        provider,
    )


def _source(result, contains: str) -> RetrievedSource:
    return next(
        source for source in result.sources if contains.lower() in source.text.lower()
    )


def _supported(question: str, result, answer_text: str, source_ids: tuple[str, ...]):
    return generate_grounded_answer(
        question,
        result.sources,
        FixedAnswerProvider(answer_text, source_ids),
        max_sources=15,
    )


def test_invoice_question_retrieves_and_cites_exact_invoice_row(tmp_path: Path) -> None:
    _, _, _, _, index, provider = _index(tmp_path)
    question = "What did we pay for London Dry Gin on invoice INV-1048?"
    result = retrieve(index, question, provider, top_k=5, minimum_score=0.1)
    invoice = _source(result, "Invoice Number: INV-1048")
    answer = _supported(
        question, result, "We paid $147.00 for London Dry Gin.", (invoice.chunk_id,)
    )

    assert answer.status is GroundedAnswerStatus.SUPPORTED
    assert "$147.00" in answer.answer
    assert answer.citations[0].document_name == "harbor_hearth_invoices.csv"
    assert answer.citations[0].source_locator.record_label == "INV-1048"
    assert all(
        "INV-1050" not in citation.source_excerpt for citation in answer.citations
    )
    assert "INV-1048" in _locator_text(answer.citations[0])


def test_purchase_order_question_retrieves_exact_tonic_row(tmp_path: Path) -> None:
    _, _, _, _, index, provider = _index(tmp_path)
    question = "What quantity of tonic was ordered on PO-221?"
    result = retrieve(index, question, provider, top_k=5, minimum_score=0.1)
    po = _source(result, "Po Number: PO-221")
    answer = _supported(
        question, result, "24 units of Tonic Water were ordered.", (po.chunk_id,)
    )

    assert answer.status is GroundedAnswerStatus.SUPPORTED
    assert "24" in answer.answer
    assert answer.citations[0].source_locator.record_label == "PO-221"
    assert answer.citations[0].source_locator.row_number == 2


def test_vendor_product_and_invoice_discovery_questions_preserve_cross_file_lineage(
    tmp_path: Path,
) -> None:
    _, _, _, _, index, provider = _index(tmp_path)
    product_question = "Which vendor supplies SKU GIN-001?"
    product_result = retrieve(
        index, product_question, provider, top_k=15, minimum_score=0.1
    )
    product = next(
        source
        for source in product_result.sources
        if source.document_name == "harbor_hearth_products.xlsx"
        and "Sku: GIN-001" in source.text
    )
    product_answer = _supported(
        product_question,
        product_result,
        "Granite State Beverage supplies SKU GIN-001.",
        (product.chunk_id,),
    )
    invoice_question = "Which invoice contains London Dry Gin?"
    invoice_result = retrieve(
        index, invoice_question, provider, top_k=15, minimum_score=0.1
    )
    invoice = _source(invoice_result, "Invoice Number: INV-1048")
    invoice_answer = _supported(
        invoice_question,
        invoice_result,
        "INV-1048 contains London Dry Gin.",
        (invoice.chunk_id,),
    )

    assert product_answer.status is GroundedAnswerStatus.SUPPORTED
    assert product_answer.citations[0].document_name == "harbor_hearth_products.xlsx"
    assert "Granite State Beverage" in product_answer.answer
    assert invoice_answer.status is GroundedAnswerStatus.SUPPORTED
    assert invoice_answer.citations[0].source_locator.record_label == "INV-1048"


def test_product_list_and_mixed_document_answer_use_only_selected_evidence(
    tmp_path: Path,
) -> None:
    _, _, _, _, index, provider = _index(tmp_path)
    product_question = "What products are listed for Granite State Beverage?"
    product_result = retrieve(
        index, product_question, provider, top_k=15, minimum_score=0.1
    )
    product = next(
        source
        for source in product_result.sources
        if source.document_name == "harbor_hearth_products.xlsx"
        and "Sku: GIN-001" in source.text
    )
    product_answer = _supported(
        product_question,
        product_result,
        "London Dry Gin is listed for Granite State Beverage.",
        (product.chunk_id,),
    )
    mixed_question = "What call-out rule applies while reviewing invoice INV-1048?"
    mixed_result = retrieve(index, mixed_question, provider, top_k=8, minimum_score=0.1)
    callout = _source(mixed_result, "at least two hours")
    invoice = _source(mixed_result, "Invoice Number: INV-1048")
    mixed_answer = _supported(
        mixed_question,
        mixed_result,
        "Call out at least two hours ahead; INV-1048 records London Dry Gin.",
        (callout.chunk_id, invoice.chunk_id),
    )

    assert product_answer.status is GroundedAnswerStatus.SUPPORTED
    assert "London Dry Gin" in product_answer.answer
    assert {citation.document_name for citation in mixed_answer.citations} == {
        "callout_attendance_policy.md",
        "harbor_hearth_invoices.csv",
    }
    assert mixed_answer.citations[0].source_locator.kind.value != "structured_row"
    assert mixed_answer.citations[1].source_locator.kind.value == "structured_row"


def test_invalid_structured_citation_fails_closed_and_locators_are_local(
    tmp_path: Path,
) -> None:
    _, _, structured, _, index, provider = _index(tmp_path, generic=True)
    question = "What is the fictional patio station status?"
    result = retrieve(index, question, provider, top_k=5, minimum_score=0.1)
    generic = _source(result, "Station: Patio")
    invalid = _supported(
        question, result, "Patio is ready.", ("not-a-retrieved-record",)
    )
    record_types = {
        record.record_type
        for document in structured
        for sheet in document.sheets
        for record in sheet.records
    }

    assert invalid.status is GroundedAnswerStatus.UNSUPPORTED
    assert StructuredRecordType.GENERIC_TABULAR in record_types
    assert generic.source_locator.record_label is None
    assert (
        _locator_text(
            type(
                "Citation",
                (),
                {
                    "source_locator": generic.source_locator,
                    "document_name": generic.document_name,
                },
            )()
        )
        == "fictional_service_log.csv — Row 2"
    )
    assert str(tmp_path) not in generic.text


def test_structured_ids_order_and_citation_lineage_are_deterministic(
    tmp_path: Path,
) -> None:
    workspace, documents, _, chunks, _, _ = _index(tmp_path)
    first = [
        parse_structured_document(document, read_accepted_content(workspace, document))
        for document in documents
        if document.document_type in {DocumentType.CSV, DocumentType.XLSX}
    ]
    second = [
        parse_structured_document(document, read_accepted_content(workspace, document))
        for document in documents
        if document.document_type in {DocumentType.CSV, DocumentType.XLSX}
    ]

    assert [
        record.record_id
        for doc in first
        for sheet in doc.sheets
        for record in sheet.records
    ] == [
        record.record_id
        for doc in second
        for sheet in doc.sheets
        for record in sheet.records
    ]
    assert [chunk.chunk_id for chunk in chunks] == [chunk.chunk_id for chunk in chunks]
    assert all(
        "/" not in record.document_name
        for doc in first
        for sheet in doc.sheets
        for record in sheet.records
    )


def test_all_structured_locator_types_are_derived_from_record_provenance(
    tmp_path: Path,
) -> None:
    _, _, structured, _, _, _ = _index(tmp_path, generic=True)
    records = {
        record.record_type: record
        for document in structured
        for sheet in document.sheets
        for record in sheet.records
    }
    # The generic file is deliberately not classified as a business record family.
    generic = next(
        record
        for document in structured
        for sheet in document.sheets
        for record in sheet.records
        if record.document_name == "fictional_service_log.csv"
    )
    assert _locator_text(
        type(
            "Citation",
            (),
            {
                "source_locator": structured_evidence(
                    next(
                        doc
                        for doc in structured
                        if doc.document_id
                        == records[StructuredRecordType.INVOICE].document_id
                    )
                )
                .sections[0]
                .source_locator,
                "document_name": "harbor_hearth_invoices.csv",
            },
        )()
    ).startswith("INV-1048 —")
    assert records[
        StructuredRecordType.PURCHASE_ORDER
    ].source_locator.record_label.startswith("PO-")
    assert (
        records[StructuredRecordType.VENDOR].source_locator.record_label
        == "North Coast Coffee"
    )
    assert (
        records[StructuredRecordType.PRODUCT_CATALOG].source_locator.record_label
        == "Espresso Beans"
    )
    assert generic.record_type is StructuredRecordType.GENERIC_TABULAR
    assert generic.source_locator.sheet_name is None
    assert generic.source_locator.row_number == 2
