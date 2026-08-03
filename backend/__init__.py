"""WhisperDock local transcription backend.

The package deliberately keeps mutable data outside the source tree only in
the project directories created by :mod:`backend.paths`.
"""

from .paths import ProjectPaths

__all__ = ["ProjectPaths"]
