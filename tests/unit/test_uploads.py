from __future__ import annotations

import hashlib
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from ai_knowledge_assistant.models import (
    DocumentType,
    UploadErrorCode,
    UploadValidationError,
)
from ai_knowledge_assistant.uploads import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    accept_upload,
)
from ai_knowledge_assistant.workspace import UploadWorkspace


def _valid_docx() -> bytes:
    document = BytesIO()
    with zipfile.ZipFile(document, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", "<w:document />")
    return document.getvalue()


@pytest.fixture
def workspace(tmp_path: Path) -> UploadWorkspace:
    created = UploadWorkspace.create(tmp_path)
    yield created
    created.cleanup()


@pytest.mark.parametrize(
    ("filename", "content", "document_type"),
    [
        ("policy.pdf", b"%PDF-1.4\nfictional fixture", DocumentType.PDF),
        ("guide.docx", _valid_docx(), DocumentType.DOCX),
        ("notes.txt", "Caf\u00e9 notes".encode(), DocumentType.TEXT),
        ("policy.md", "# Policy\n\nDetails".encode(), DocumentType.MARKDOWN),
    ],
)
def test_accepts_supported_valid_files(
    workspace: UploadWorkspace,
    filename: str,
    content: bytes,
    document_type: DocumentType,
) -> None:
    accepted = accept_upload(workspace, filename, content)

    stored = workspace.uploads_dir / accepted.stored_filename
    assert accepted.document_type is document_type
    assert accepted.original_display_name == filename
    assert accepted.run_id == workspace.run_id
    assert accepted.size_bytes == len(content)
    assert accepted.content_hash == hashlib.sha256(content).hexdigest()
    assert accepted.document_id in accepted.stored_filename
    assert stored.read_bytes() == content


def test_duplicate_client_names_are_stored_separately(
    workspace: UploadWorkspace,
) -> None:
    first = accept_upload(workspace, "policy.txt", b"first")
    second = accept_upload(workspace, "policy.txt", b"second")

    assert first.document_id != second.document_id
    assert first.stored_filename != second.stored_filename
    assert (workspace.uploads_dir / first.stored_filename).read_bytes() == b"first"
    assert (workspace.uploads_dir / second.stored_filename).read_bytes() == b"second"


def test_accepts_odd_unicode_filename(workspace: UploadWorkspace) -> None:
    accepted = accept_upload(workspace, "r\u00e9sum\u00e9 \U0001f331.md", b"# Demo")

    assert accepted.original_display_name == "r\u00e9sum\u00e9 \U0001f331.md"


@pytest.mark.parametrize(
    "filename",
    [
        "../../policy.pdf",
        "..\\..\\policy.pdf",
        "/etc/passwd",
        "C:\\Windows\\system.ini",
        "policy\x00.pdf",
        "policy\n.pdf",
        "a" * 256 + ".txt",
    ],
)
def test_rejects_unsafe_filenames(workspace: UploadWorkspace, filename: str) -> None:
    with pytest.raises(UploadValidationError) as error:
        accept_upload(workspace, filename, b"safe text")

    assert error.value.code is UploadErrorCode.UNSAFE_FILENAME
    assert list(workspace.uploads_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("filename", "content", "code"),
    [
        ("policy.exe", b"MZ", UploadErrorCode.UNSUPPORTED_FILE_TYPE),
        ("policy.pdf", b"not a PDF", UploadErrorCode.INVALID_FILE_SIGNATURE),
        ("policy.docx", b"not a ZIP", UploadErrorCode.MALFORMED_DOCX),
        ("policy.docx", b"PK\x03\x04", UploadErrorCode.MALFORMED_DOCX),
        ("policy.txt", b"binary\x00payload", UploadErrorCode.INVALID_TEXT_ENCODING),
        ("policy.md", b"\xff\xfe", UploadErrorCode.INVALID_TEXT_ENCODING),
        ("policy.txt", b"", UploadErrorCode.EMPTY_FILE),
    ],
)
def test_rejects_invalid_content(
    workspace: UploadWorkspace,
    filename: str,
    content: bytes,
    code: UploadErrorCode,
) -> None:
    with pytest.raises(UploadValidationError) as error:
        accept_upload(workspace, filename, content)

    assert error.value.code is code


def test_rejects_oversized_file_before_storage(workspace: UploadWorkspace) -> None:
    with pytest.raises(UploadValidationError) as error:
        accept_upload(workspace, "policy.txt", b"12345", max_file_size_bytes=4)

    assert error.value.code is UploadErrorCode.FILE_TOO_LARGE
    assert list(workspace.uploads_dir.iterdir()) == []


def test_default_file_size_limit_is_ten_megabytes() -> None:
    assert DEFAULT_MAX_FILE_SIZE_BYTES == 10 * 1024 * 1024


def test_upload_stays_inside_current_workspace(workspace: UploadWorkspace) -> None:
    accepted = accept_upload(workspace, "policy.txt", b"safe text")
    stored = (workspace.uploads_dir / accepted.stored_filename).resolve()

    assert stored.parent == workspace.uploads_dir.resolve()
    assert workspace.root.resolve() in stored.parents
