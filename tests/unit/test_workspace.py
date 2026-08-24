from pathlib import Path

import pytest

from ai_knowledge_assistant.workspace import UploadWorkspace


def test_workspaces_have_unique_ids_and_isolated_directories(tmp_path: Path) -> None:
    first = UploadWorkspace.create(tmp_path)
    second = UploadWorkspace.create(tmp_path)

    assert first.run_id != second.run_id
    assert first.root.parent == tmp_path.resolve()
    assert second.root.parent == tmp_path.resolve()
    assert first.uploads_dir == first.root / "uploads"
    assert second.uploads_dir == second.root / "uploads"
    assert first.root.is_dir()
    assert second.root.is_dir()


def test_cleanup_removes_only_the_current_workspace(tmp_path: Path) -> None:
    workspace = UploadWorkspace.create(tmp_path)
    other_workspace = UploadWorkspace.create(tmp_path)

    workspace.cleanup()

    assert not workspace.root.exists()
    assert other_workspace.root.is_dir()


def test_cleanup_rejects_a_workspace_outside_its_controlled_root(
    tmp_path: Path,
) -> None:
    workspace = UploadWorkspace.create(tmp_path)
    workspace.root = tmp_path.parent

    with pytest.raises(ValueError, match="outside"):
        workspace.cleanup()
