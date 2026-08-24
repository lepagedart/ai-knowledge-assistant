"""Typed data contracts for the document-ingestion boundary."""

from dataclasses import dataclass
from enum import StrEnum


class DocumentType(StrEnum):
    """Document types accepted by the V1 upload boundary."""

    PDF = "pdf"
    DOCX = "docx"
    TEXT = "txt"
    MARKDOWN = "md"


class UploadErrorCode(StrEnum):
    """Stable error codes intended for tests and a future user interface."""

    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FILE_SIGNATURE = "INVALID_FILE_SIGNATURE"
    INVALID_TEXT_ENCODING = "INVALID_TEXT_ENCODING"
    UNSAFE_FILENAME = "UNSAFE_FILENAME"
    EMPTY_FILE = "EMPTY_FILE"
    MALFORMED_DOCX = "MALFORMED_DOCX"
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
