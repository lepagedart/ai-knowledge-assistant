from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ai_knowledge_assistant.uploads import accept_upload
from ai_knowledge_assistant.workspace import UploadWorkspace

ROOT = Path(__file__).resolve().parents[2]


def test_upload_does_not_modify_repository_or_demo_documents(tmp_path: Path) -> None:
    tracked_files = [
        ROOT / "README.md",
        ROOT / "demo_documents" / "harbor_and_hearth" / "employee_handbook.md",
    ]
    before = {path: path.read_bytes() for path in tracked_files}
    workspace = UploadWorkspace.create(tmp_path)

    accept_upload(workspace, "client-notes.txt", b"untrusted client content")

    assert {path: path.read_bytes() for path in tracked_files} == before
    assert list(workspace.uploads_dir.iterdir())
    assert not (ROOT / "client-notes.txt").exists()


def test_runtime_upload_directory_is_ignored_by_git() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "uploads/example.txt"],
        cwd=ROOT,
        check=False,
    )

    assert result.returncode == 0


def test_upload_rejects_a_symlinked_upload_directory(tmp_path: Path) -> None:
    workspace = UploadWorkspace.create(tmp_path)
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    symlinked_directory = workspace.root / "symlinked_uploads"
    symlinked_directory.symlink_to(outside_directory, target_is_directory=True)
    workspace.uploads_dir = symlinked_directory

    with pytest.raises(ValueError, match="symbolic"):
        accept_upload(workspace, "policy.txt", b"untrusted client content")

    assert list(outside_directory.iterdir()) == []
