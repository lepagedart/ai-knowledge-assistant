"""Validation and safe storage of untrusted V1 business-document uploads."""

from __future__ import annotations

import hashlib
import os
import zipfile
from io import BytesIO
from pathlib import Path, PureWindowsPath
from uuid import uuid4

from .models import (
    AcceptedDocument,
    DocumentType,
    UploadErrorCode,
    UploadValidationError,
)
from .workspace import UploadWorkspace

DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
MAX_DISPLAY_NAME_LENGTH = 255
_TYPE_BY_EXTENSION = {
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.DOCX,
    ".txt": DocumentType.TEXT,
    ".md": DocumentType.MARKDOWN,
}
_DOCX_REQUIRED_MEMBERS = {"[Content_Types].xml", "word/document.xml"}


def accept_upload(
    workspace: UploadWorkspace,
    original_filename: str,
    content: bytes,
    *,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> AcceptedDocument:
    """Validate and store a complete untrusted upload in ``workspace``.

    The caller receives metadata only; the generated filename is not derived
    from the untrusted client filename.
    """
    _validate_workspace(workspace)
    display_name = _validate_filename(original_filename)
    _validate_size_limit(max_file_size_bytes)
    if not isinstance(content, bytes):
        raise TypeError("Upload content must be bytes.")
    if not content:
        raise UploadValidationError(
            UploadErrorCode.EMPTY_FILE, "Uploaded files must not be empty."
        )
    if len(content) > max_file_size_bytes:
        raise UploadValidationError(
            UploadErrorCode.FILE_TOO_LARGE,
            f"Files must be {max_file_size_bytes} bytes or smaller.",
        )

    document_type = _document_type_for(display_name)
    _validate_content(document_type, content)

    document_id = uuid4().hex
    stored_filename = f"{document_id}{Path(display_name).suffix.lower()}"
    _write_new_file(stored_filename, content, workspace)

    return AcceptedDocument(
        document_id=document_id,
        original_display_name=display_name,
        stored_filename=stored_filename,
        document_type=document_type,
        size_bytes=len(content),
        content_hash=hashlib.sha256(content).hexdigest(),
        run_id=workspace.run_id,
    )


def _validate_workspace(workspace: UploadWorkspace) -> None:
    if workspace.uploads_dir.parent != workspace.root:
        raise ValueError("Upload directory is not owned by the workspace.")
    if workspace.root.parent != workspace._temporary_root:
        raise ValueError("Workspace is outside the controlled temporary root.")
    if workspace.root.is_symlink() or workspace.uploads_dir.is_symlink():
        raise ValueError("Workspace directories must not be symbolic links.")
    if not workspace.uploads_dir.is_dir():
        raise ValueError("Workspace upload directory is unavailable.")


def _validate_filename(filename: str) -> str:
    if not isinstance(filename, str) or not filename or not filename.strip():
        raise UploadValidationError(
            UploadErrorCode.UNSAFE_FILENAME, "A file name is required."
        )
    if len(filename) > MAX_DISPLAY_NAME_LENGTH:
        raise UploadValidationError(
            UploadErrorCode.UNSAFE_FILENAME, "The file name is too long."
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in filename):
        raise UploadValidationError(
            UploadErrorCode.UNSAFE_FILENAME,
            "The file name contains unsupported control characters.",
        )
    if "/" in filename or "\\" in filename or PureWindowsPath(filename).drive:
        raise UploadValidationError(
            UploadErrorCode.UNSAFE_FILENAME,
            "File names must not include a path.",
        )
    if filename in {".", ".."}:
        raise UploadValidationError(
            UploadErrorCode.UNSAFE_FILENAME, "The file name is not valid."
        )
    return filename


def _validate_size_limit(max_file_size_bytes: int) -> None:
    if max_file_size_bytes <= 0:
        raise ValueError("max_file_size_bytes must be positive.")


def _document_type_for(filename: str) -> DocumentType:
    document_type = _TYPE_BY_EXTENSION.get(Path(filename).suffix.lower())
    if document_type is None:
        raise UploadValidationError(
            UploadErrorCode.UNSUPPORTED_FILE_TYPE,
            "V1 accepts PDF, DOCX, TXT, and Markdown files only.",
        )
    return document_type


def _validate_content(document_type: DocumentType, content: bytes) -> None:
    if document_type is DocumentType.PDF:
        if not content.startswith(b"%PDF-"):
            raise UploadValidationError(
                UploadErrorCode.INVALID_FILE_SIGNATURE,
                "The PDF file signature is invalid.",
            )
    elif document_type is DocumentType.DOCX:
        _validate_docx(content)
    else:
        _validate_utf8_text(content)


def _validate_docx(content: bytes) -> None:
    if not content.startswith(b"PK"):
        raise UploadValidationError(
            UploadErrorCode.MALFORMED_DOCX,
            "A DOCX file must be a valid Office ZIP container.",
        )
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            members = set(archive.namelist())
    except zipfile.BadZipFile as error:
        raise UploadValidationError(
            UploadErrorCode.MALFORMED_DOCX, "The DOCX archive is malformed."
        ) from error
    if not _DOCX_REQUIRED_MEMBERS.issubset(members):
        raise UploadValidationError(
            UploadErrorCode.MALFORMED_DOCX,
            "The DOCX archive is missing required document parts.",
        )


def _validate_utf8_text(content: bytes) -> None:
    if b"\x00" in content:
        raise UploadValidationError(
            UploadErrorCode.INVALID_TEXT_ENCODING,
            "Text files must be UTF-8 text, not binary data.",
        )
    try:
        content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise UploadValidationError(
            UploadErrorCode.INVALID_TEXT_ENCODING,
            "Text files must use UTF-8 encoding.",
        ) from error


def _write_new_file(
    stored_filename: str, content: bytes, workspace: UploadWorkspace
) -> None:
    """Write through the workspace directory descriptor without following links."""
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        directory_descriptor = os.open(workspace.uploads_dir, directory_flags)
    except FileExistsError as error:
        raise UploadValidationError(
            UploadErrorCode.STORAGE_ERROR, "Could not safely store this upload."
        ) from error
    except OSError as error:
        raise UploadValidationError(
            UploadErrorCode.STORAGE_ERROR, "Could not access upload storage."
        ) from error

    try:
        descriptor = os.open(stored_filename, flags, 0o600, dir_fd=directory_descriptor)
        with os.fdopen(descriptor, "wb") as stored_file:
            stored_file.write(content)
    except FileExistsError as error:
        raise UploadValidationError(
            UploadErrorCode.STORAGE_ERROR, "Could not safely store this upload."
        ) from error
    except OSError as error:
        try:
            os.unlink(stored_filename, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        raise UploadValidationError(
            UploadErrorCode.STORAGE_ERROR, "Could not safely store this upload."
        ) from error
    finally:
        os.close(directory_descriptor)
