"""Typed data contracts for the document-ingestion boundary."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class DocumentType(StrEnum):
    """Document types accepted by the V1 upload boundary."""

    PDF = "pdf"
    DOCX = "docx"
    TEXT = "txt"
    MARKDOWN = "md"
    CSV = "csv"
    XLSX = "xlsx"


class UploadErrorCode(StrEnum):
    """Stable error codes intended for tests and a future user interface."""

    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FILE_SIGNATURE = "INVALID_FILE_SIGNATURE"
    INVALID_TEXT_ENCODING = "INVALID_TEXT_ENCODING"
    UNSAFE_FILENAME = "UNSAFE_FILENAME"
    EMPTY_FILE = "EMPTY_FILE"
    MALFORMED_DOCX = "MALFORMED_DOCX"
    MALFORMED_XLSX = "MALFORMED_XLSX"
    STRUCTURED_LIMIT_EXCEEDED = "STRUCTURED_LIMIT_EXCEEDED"
    STORAGE_ERROR = "STORAGE_ERROR"


class UploadValidationError(Exception):
    """A predictable upload rejection with a stable code and safe message."""

    def __init__(self, code: UploadErrorCode, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class AcceptedDocument:
    """Metadata for one accepted file stored in a controlled workspace."""

    document_id: str
    original_display_name: str
    stored_filename: str
    document_type: DocumentType
    size_bytes: int
    content_hash: str
    run_id: str


class SourceLocatorKind(StrEnum):
    """Citation-friendly forms of source location preserved by extraction."""

    PDF_PAGE = "pdf_page"
    DOCUMENT_SECTION = "document_section"
    TEXT_LINE_RANGE = "text_line_range"
    STRUCTURED_ROW = "structured_row"


@dataclass(frozen=True, slots=True)
class SourceLocator:
    """A source location without filesystem details."""

    kind: SourceLocatorKind
    page_number: int | None = None
    section_label: str | None = None
    heading_level: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    sheet_name: str | None = None
    row_number: int | None = None
    record_label: str | None = None


class StructuredRecordType(StrEnum):
    """Conservative deterministic classifications for tabular records."""

    INVOICE = "invoice"
    PURCHASE_ORDER = "purchase_order"
    VENDOR = "vendor"
    PRODUCT_CATALOG = "product_catalog"
    GENERIC_TABULAR = "generic_tabular"


class ReconciliationStatus(StrEnum):
    """Deterministic state assigned to one invoice/PO comparison."""

    MATCHED = "MATCHED"
    VARIANCE = "VARIANCE"
    MISSING_ON_INVOICE = "MISSING_ON_INVOICE"
    MISSING_ON_PO = "MISSING_ON_PO"
    UNMATCHED = "UNMATCHED"


class ReconciliationIssueCode(StrEnum):
    """Stable, machine-readable reconciliation outcomes and safe failures."""

    MISSING_PO_NUMBER = "MISSING_PO_NUMBER"
    PO_NOT_FOUND = "PO_NOT_FOUND"
    ITEM_NOT_FOUND = "ITEM_NOT_FOUND"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
    DUPLICATE_PO_IDENTITY = "DUPLICATE_PO_IDENTITY"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    INVALID_UNIT_PRICE = "INVALID_UNIT_PRICE"
    INVALID_LINE_TOTAL = "INVALID_LINE_TOTAL"
    WRONG_RECORD_TYPE = "WRONG_RECORD_TYPE"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"


@dataclass(frozen=True, slots=True)
class MatchedRecordReference:
    """Path-free provenance for a single invoice or purchase-order row."""

    document_id: str
    document_name: str
    record_id: str
    sheet_name: str | None
    row_number: int
    record_label: str | None


@dataclass(frozen=True, slots=True)
class QuantityVariance:
    invoice_quantity: Decimal
    po_quantity: Decimal
    variance: Decimal
    direction: str


@dataclass(frozen=True, slots=True)
class MoneyVariance:
    invoice_amount: Decimal
    po_amount: Decimal
    variance: Decimal
    direction: str


@dataclass(frozen=True, slots=True)
class ReconciliationLine:
    """One locally calculated comparison and its original-record provenance."""

    reconciliation_id: str
    status: ReconciliationStatus
    issue_codes: tuple[ReconciliationIssueCode, ...]
    invoice: MatchedRecordReference | None
    purchase_order: MatchedRecordReference | None
    invoice_number: str | None
    po_number: str | None
    item_name: str | None
    sku: str | None
    unit: str | None
    invoice_unit: str | None
    po_unit: str | None
    invoice_source_line_total: Decimal | None
    quantity_variance: QuantityVariance | None
    unit_price_variance: MoneyVariance | None
    invoice_line_total_variance: MoneyVariance | None
    extended_variance: MoneyVariance | None


@dataclass(frozen=True, slots=True)
class ReconciliationSummary:
    matched_line_count: int
    variance_line_count: int
    missing_on_po_count: int
    missing_on_invoice_count: int
    total_monetary_variance: Decimal
    monetary_variance_line_count: int


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Immutable output of the offline reconciliation engine."""

    reconciliation_version: str
    lines: tuple[ReconciliationLine, ...]
    summary: ReconciliationSummary


@dataclass(frozen=True, slots=True)
class StructuredField:
    """One source-preserving field from a tabular row."""

    original_header: str
    canonical_name: str
    original_value: str | None
    normalized_value: str | None


@dataclass(frozen=True, slots=True)
class StructuredSourceLocator:
    """Exact path-free lineage for one structured source row."""

    sheet_name: str | None
    row_number: int
    record_label: str | None


@dataclass(frozen=True, slots=True)
class StructuredRecord:
    """A normalized immutable record with original values retained."""

    record_id: str
    document_id: str
    document_name: str
    document_type: DocumentType
    record_type: StructuredRecordType
    record_index: int
    fields: tuple[StructuredField, ...]
    source_locator: StructuredSourceLocator
    content_hash: str


@dataclass(frozen=True, slots=True)
class StructuredSheet:
    """One non-empty CSV table or XLSX worksheet."""

    sheet_name: str | None
    headers: tuple[str, ...]
    records: tuple[StructuredRecord, ...]


@dataclass(frozen=True, slots=True)
class StructuredDocument:
    """Deterministic parsed representation of one structured upload."""

    parsing_version: str
    document_id: str
    document_display_name: str
    document_type: DocumentType
    source_content_hash: str
    sheets: tuple[StructuredSheet, ...]


@dataclass(frozen=True, slots=True)
class ExtractedSection:
    """One deterministically extracted, citation-ready source segment."""

    section_id: str
    document_id: str
    document_display_name: str
    document_type: DocumentType
    source_locator: SourceLocator
    text: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Deterministic extracted representation of an accepted document."""

    extraction_version: str
    document_id: str
    document_display_name: str
    document_type: DocumentType
    source_content_hash: str
    sections: tuple[ExtractedSection, ...]


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """An immutable, citation-preserving segment of an extracted section."""

    chunk_id: str
    document_id: str
    document_name: str
    document_type: DocumentType
    section_id: str
    chunk_index: int
    text: str
    source_locator: SourceLocator
    source_char_start: int
    source_char_end: int
    primary_char_start: int
    primary_char_end: int
    source_section_content_hash: str
    content_hash: str
    chunking_version: str


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A chunk and its validated local embedding, with citation lineage intact."""

    chunk_id: str
    document_id: str
    document_name: str
    document_type: DocumentType
    section_id: str
    chunk_index: int
    source_locator: SourceLocator
    text: str
    content_hash: str
    source_section_content_hash: str
    source_char_start: int
    source_char_end: int
    primary_char_start: int
    primary_char_end: int
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RetrievedSource:
    """One ranked, citation-ready local retrieval candidate."""

    rank: int
    score: float
    chunk_id: str
    document_id: str
    document_name: str
    document_type: DocumentType
    section_id: str
    chunk_index: int
    source_locator: SourceLocator
    text: str
    content_hash: str
    source_section_content_hash: str
    source_char_start: int
    source_char_end: int
    primary_char_start: int
    primary_char_end: int


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """The normalized question and ordered evidence candidates for it."""

    question: str
    sources: tuple[RetrievedSource, ...]

    @property
    def has_results(self) -> bool:
        """Whether one or more candidates met the requested relevance threshold."""
        return bool(self.sources)


class GroundedAnswerStatus(StrEnum):
    """Whether an answer passed V1's local grounding checks."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class UnsupportedReasonCode(StrEnum):
    """Safe reasons exposed when an answer cannot be presented as grounded."""

    NO_QUALIFYING_SOURCES = "NO_QUALIFYING_SOURCES"
    PROVIDER_INVALID = "PROVIDER_INVALID"
    PROVIDER_ERROR = "PROVIDER_ERROR"


@dataclass(frozen=True, slots=True)
class ProviderCitation:
    """Untrusted structured citation proposed by an answer provider."""

    chunk_id: object
    claim: object


@dataclass(frozen=True, slots=True)
class ProviderAnswer:
    """Untrusted structured answer returned by an injected provider."""

    status: object
    answer: object
    citations: object


@dataclass(frozen=True, slots=True)
class GroundedCitation:
    """Validated display citation derived exclusively from a retrieved source."""

    chunk_id: str
    claim: str
    document_name: str
    source_locator: SourceLocator
    source_excerpt: str
    rank: int


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """Safe answer state returned after deterministic local citation validation."""

    status: GroundedAnswerStatus
    answer: str
    citations: tuple[GroundedCitation, ...]
    source_ids_used: tuple[str, ...]
    unsupported_reason_code: UnsupportedReasonCode | None = None


class ExtractionErrorCode(StrEnum):
    """Stable errors produced by the deterministic extraction boundary."""

    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    NO_EXTRACTABLE_TEXT = "NO_EXTRACTABLE_TEXT"
    UNSUPPORTED_DOCUMENT_STRUCTURE = "UNSUPPORTED_DOCUMENT_STRUCTURE"
    PDF_PAGE_EXTRACTION_FAILED = "PDF_PAGE_EXTRACTION_FAILED"
    DOCX_EXTRACTION_FAILED = "DOCX_EXTRACTION_FAILED"


class ExtractionError(Exception):
    """A deterministic extraction failure with a stable code."""

    def __init__(self, code: ExtractionErrorCode, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)
