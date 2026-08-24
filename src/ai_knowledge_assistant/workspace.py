"""Isolated, temporary workspace management for untrusted uploads."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

_APPLICATION_TEMP_DIRECTORY = "ai_knowledge_assistant"


@dataclass(slots=True)
class UploadWorkspace:
    """Own one random run directory beneath a controlled temporary root."""

    run_id: str
    root: Path
    uploads_dir: Path
    _temporary_root: Path

    @classmethod
    def create(cls, temporary_root: Path | None = None) -> UploadWorkspace:
        """Create an empty workspace under an application-controlled root."""
        configured_root = temporary_root or (
            Path(tempfile.gettempdir()) / _APPLICATION_TEMP_DIRECTORY
        )
        base_root = configured_root.resolve()
        base_root.mkdir(mode=0o700, parents=True, exist_ok=True)

        while True:
            run_id = uuid4().hex
            workspace_root = base_root / run_id
            try:
                workspace_root.mkdir(mode=0o700)
            except FileExistsError:
                continue
            break

        uploads_dir = workspace_root / "uploads"
        uploads_dir.mkdir(mode=0o700)
        return cls(
            run_id=run_id,
            root=workspace_root,
            uploads_dir=uploads_dir,
            _temporary_root=base_root,
        )

    def cleanup(self) -> None:
        """Permanently delete this run, rejecting unexpected workspace paths."""
        expected_root = (self._temporary_root / self.run_id).resolve()
        if self.root != expected_root or self.root.parent != self._temporary_root:
            raise ValueError("Workspace root is outside the controlled temporary root.")
        if self.root.is_symlink():
            raise ValueError("Workspace root must not be a symbolic link.")
        if self.root.exists():
            shutil.rmtree(self.root)
