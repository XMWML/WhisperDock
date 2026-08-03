"""Portable, project-local filesystem layout helpers."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .errors import WhisperDockError


_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


class ProjectPaths:
    """Own all persistent locations used by the application.

    ``root`` is configurable for tests and packaged builds.  The default is
    the directory containing the ``backend`` package's parent (the project
    root), so moving the complete directory keeps all app data together.
    """

    def __init__(self, root: str | Path | None = None):
        configured_root = root or os.environ.get("WHISPERDOCK_HOME")
        self.root = Path(configured_root).expanduser().resolve() if configured_root else Path(__file__).resolve().parents[1]
        self.config = self.root / "config"
        self.models = self.root / "models"
        self.cache = self.root / "cache"
        self.workspace = self.root / "workspace"
        self.outputs = self.root / "outputs"
        self.logs = self.root / "logs"

    def ensure(self) -> None:
        for directory in (
            self.root,
            self.config,
            self.models,
            self.cache,
            self.workspace,
            self.outputs,
            self.logs,
            self.models / "openai-whisper",
            self.models / "custom",
            self.models / "huggingface",
            self.workspace / "uploads",
            self.workspace / "realtime",
            self.outputs / "jobs",
            self.cache / "pip",
            self.cache / "huggingface",
            self.cache / "torch",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def resolve_inside(self, parent: Path, relative_path: str | Path) -> Path:
        """Resolve a user-supplied relative path without allowing traversal."""
        candidate = (parent / relative_path).resolve()
        parent = parent.resolve()
        if not _is_within(candidate, parent):
            raise WhisperDockError("The requested path must stay inside the WhisperDock project.", code="unsafe_path")
        return candidate

    def relative(self, path: str | Path) -> str:
        path = Path(path).resolve()
        if not _is_within(path, self.root):
            raise WhisperDockError("Only files inside the WhisperDock project may be recorded.", code="unsafe_path")
        return path.relative_to(self.root).as_posix()

    def from_relative(self, relative_path: str | Path) -> Path:
        return self.resolve_inside(self.root, relative_path)

    @staticmethod
    def safe_filename(name: str, fallback: str = "file") -> str:
        name = Path(name).name.strip()
        cleaned = _SAFE_FILENAME.sub("-", name).strip(".-")
        return cleaned[:160] or fallback

    def upload_destination(self, original_name: str, *, prefix: str = "audio") -> Path:
        safe_name = self.safe_filename(original_name, "audio")
        return self.workspace / "uploads" / f"{prefix}-{uuid.uuid4().hex[:12]}-{safe_name}"

    def job_directory(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise WhisperDockError("Invalid job identifier.", code="invalid_job_id")
        directory = self.outputs / "jobs" / job_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def read_json(self, path: Path, default: Any) -> Any:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return default

    def write_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
