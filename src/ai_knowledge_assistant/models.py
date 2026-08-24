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
