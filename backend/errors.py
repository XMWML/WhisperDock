"""Small, framework-independent errors used by the service layer."""

from __future__ import annotations


class WhisperDockError(Exception):
    """An expected error that can be shown directly in the Web UI."""

    def __init__(self, message: str, *, code: str = "whisperdock_error", status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self)}


class DependencyUnavailable(WhisperDockError):
    def __init__(self, package: str, detail: str | None = None):
        message = f"{package} is not installed in this WhisperDock environment."
        if detail:
            message = f"{message} {detail}"
        super().__init__(message, code="dependency_unavailable", status_code=503)
