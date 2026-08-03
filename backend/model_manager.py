"""Model catalog, storage, loading, and inference adapters.

The default engine is OpenAI's ``openai-whisper`` package.  A second optional
Transformers adapter is included because many community Whisper fine-tunes on
Hugging Face (including the desktop Teochew example) are saved in that format.
Both adapters keep downloaded model files under ``models/``.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DependencyUnavailable, WhisperDockError
from .options import normalize_whisper_options
from .paths import ProjectPaths


BUILTIN_MODELS: dict[str, dict[str, Any]] = {
    "tiny": {"label": "Tiny", "parameters": "39M", "vram": "~1 GB", "languages": "Multilingual"},
    "tiny.en": {"label": "Tiny English", "parameters": "39M", "vram": "~1 GB", "languages": "English only"},
    "base": {"label": "Base", "parameters": "74M", "vram": "~1 GB", "languages": "Multilingual"},
    "base.en": {"label": "Base English", "parameters": "74M", "vram": "~1 GB", "languages": "English only"},
    "small": {"label": "Small", "parameters": "244M", "vram": "~2 GB", "languages": "Multilingual"},
    "small.en": {"label": "Small English", "parameters": "244M", "vram": "~2 GB", "languages": "English only"},
    "medium": {"label": "Medium", "parameters": "769M", "vram": "~5 GB", "languages": "Multilingual"},
    "medium.en": {"label": "Medium English", "parameters": "769M", "vram": "~5 GB", "languages": "English only"},
    "large-v1": {"label": "Large v1", "parameters": "1.55B", "vram": "~10 GB", "languages": "Multilingual"},
    "large-v2": {"label": "Large v2", "parameters": "1.55B", "vram": "~10 GB", "languages": "Multilingual"},
    "large-v3": {"label": "Large v3", "parameters": "1.55B", "vram": "~10 GB", "languages": "Multilingual"},
    "turbo": {"label": "Turbo", "parameters": "809M", "vram": "~6 GB", "languages": "Multilingual"},
}

ENGINE_OPENAI = "openai-whisper"
ENGINE_TRANSFORMERS = "transformers"
_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,80}")
_HF_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_CHECKPOINT_SUFFIXES = {".pt", ".bin", ".ckpt"}


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip(".-")
    return result[:64] or "model"


def _normalise_engine(value: Any) -> str:
    aliases = {
        "openai": ENGINE_OPENAI,
        "whisper": ENGINE_OPENAI,
        "openai_whisper": ENGINE_OPENAI,
        ENGINE_OPENAI: ENGINE_OPENAI,
        "huggingface": ENGINE_TRANSFORMERS,
        "hf": ENGINE_TRANSFORMERS,
        ENGINE_TRANSFORMERS: ENGINE_TRANSFORMERS,
    }
    result = aliases.get(str(value or ENGINE_OPENAI).strip().lower())
    if result is None:
        raise WhisperDockError("engine must be 'openai-whisper' or 'transformers'.", code="invalid_engine")
    return result


@dataclass
class LoadedModel:
    model_id: str
    engine: str
    device: str
    model: Any
    processor: Any = None
    pipeline: Any = None
    loaded_at: str = ""


class ModelManager:
    """Own model metadata and the process-local model cache."""

    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self.paths.ensure()
        self.index_path = self.paths.config / "models.json"
        self._lock = threading.RLock()
        self._inference_lock = threading.RLock()
        self._loaded: dict[str, LoadedModel] = {}
        self._index = self._load_index()

    def _load_index(self) -> dict[str, Any]:
        value = self.paths.read_json(self.index_path, {"version": 1, "models": {}})
        if not isinstance(value, dict) or not isinstance(value.get("models"), dict):
            return {"version": 1, "models": {}}
        return value

    def _save_index(self) -> None:
        self.paths.write_json(self.index_path, self._index)

    @staticmethod
    def _builtin_record(model_id: str) -> dict[str, Any]:
        details = BUILTIN_MODELS[model_id]
        return {
            "id": model_id,
            "name": details["label"],
            "engine": ENGINE_OPENAI,
            "source": {"kind": "builtin", "model_name": model_id},
            "path": f"models/openai-whisper/{model_id}.pt",
            "parameters": details["parameters"],
            "estimated_vram": details["vram"],
            "languages": details["languages"],
            "created_at": None,
            "builtin": True,
        }

    def _record_for(self, model_id: str) -> dict[str, Any]:
        if model_id in BUILTIN_MODELS:
            return self._builtin_record(model_id)
        try:
            return dict(self._index["models"][model_id])
        except KeyError as exc:
            raise WhisperDockError(f"Unknown model '{model_id}'.", code="model_not_found", status_code=404) from exc

    def _model_path(self, record: dict[str, Any]) -> Path:
        path = record.get("path")
        if not isinstance(path, str) or not path:
            raise WhisperDockError("This model does not have a project-local file path.", code="model_path_missing")
        return self.paths.from_relative(path)

    def _installed(self, record: dict[str, Any]) -> bool:
        try:
            location = self._model_path(record)
        except WhisperDockError:
            return False
        if record.get("engine") == ENGINE_TRANSFORMERS:
            return self._transformers_files_present(location)
        return location.is_file() and location.stat().st_size > 0

    @staticmethod
    def _transformers_files_present(location: Path) -> bool:
        """Reject half-downloaded HF snapshots that only contain cache files."""
        if not location.is_dir() or not (location / "config.json").is_file():
            return False
        weight_suffixes = {".safetensors", ".bin", ".pt", ".ckpt"}
        for child in location.iterdir():
            if child.is_file() and (child.suffix.lower() in weight_suffixes or child.name.endswith(".index.json")):
                return True
        return False

    def _public_record(self, record: dict[str, Any]) -> dict[str, Any]:
        record = dict(record)
        location = self._model_path(record)
        installed = self._installed(record)
        loaded = record["id"] in self._loaded
        state = "loaded" if loaded else "downloaded" if installed else "not_downloaded"
        record.update(
            installed=installed,
            loaded=loaded,
            state=state,
            size_bytes=_directory_size(location) if installed else 0,
            device=self._loaded[record["id"]].device if loaded else None,
        )
        record.pop("local_source", None)
        return record

    def list_models(self) -> list[dict[str, Any]]:
        with self._lock:
            records = [self._builtin_record(model_id) for model_id in BUILTIN_MODELS]
            records.extend(dict(record) for record in self._index["models"].values())
            return [self._public_record(record) for record in records]

    def catalog(self) -> list[dict[str, Any]]:
        return [self._public_record(self._builtin_record(model_id)) for model_id in BUILTIN_MODELS]

    def get_model(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            return self._public_record(self._record_for(model_id))

    def dependency_status(self) -> dict[str, bool]:
        return {
            "openai_whisper": importlib.util.find_spec("whisper") is not None,
            "torch": importlib.util.find_spec("torch") is not None,
            "transformers": importlib.util.find_spec("transformers") is not None,
            "huggingface_hub": importlib.util.find_spec("huggingface_hub") is not None,
            "ffmpeg": shutil.which("ffmpeg") is not None,
        }

    def create_custom_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Register a remote model or import a local model into ``models/``.

        Remote models are only registered here.  Calling ``download`` performs
        network IO explicitly, making it safe for a UI to show the destination
        and ask for confirmation before a large download.
        """
        if not isinstance(payload, dict):
            raise WhisperDockError("Model details must be a JSON object.", code="invalid_model")
        nested_source = payload.get("source")
        if nested_source is not None and not isinstance(nested_source, dict):
            raise WhisperDockError("source must be an object when provided.", code="invalid_source")
        source_kind = str(payload.get("source_type") or (nested_source or {}).get("kind") or "").strip().lower()
        if source_kind not in {"url", "huggingface", "local"}:
            raise WhisperDockError("source_type must be 'url', 'huggingface', or 'local'.", code="invalid_source")
        engine = _normalise_engine(payload.get("engine"))
        name = str(payload.get("name") or "").strip()
        if not name or len(name) > 120:
            raise WhisperDockError("Model name must be between 1 and 120 characters.", code="invalid_model")
        requested_value = str(payload.get("id") or "").strip()
        # The bundled Web UI uses its source field as a convenience ID.  URLs,
        # Hugging Face repo ids, and local paths are not valid filesystem-safe
        # IDs, so derive one from the display name only when the two values are
        # intentionally the same. A distinct malformed API id remains a 400.
        source_value = str(
            payload.get("url")
            or payload.get("hf_repo")
            or payload.get("repo_id")
            or payload.get("local_path")
            or payload.get("path")
            or ""
        ).strip()
        requested_id = requested_value.lower() if requested_value else _slug(name)
        if requested_value and not _MODEL_ID.fullmatch(requested_id) and requested_value == source_value:
            requested_id = _slug(name)
        if not _MODEL_ID.fullmatch(requested_id) or requested_id in BUILTIN_MODELS:
            raise WhisperDockError("Model id must use lowercase letters, numbers, dots, underscores, or hyphens and cannot shadow a built-in model.", code="invalid_model_id")

        with self._lock:
            if requested_id in self._index["models"]:
                raise WhisperDockError(f"A model named '{requested_id}' already exists.", code="model_exists", status_code=409)

            record: dict[str, Any] = {
                "id": requested_id,
                "name": name,
                "engine": engine,
                "source": {"kind": source_kind},
                "path": "",
                "parameters": str(payload.get("parameters") or payload.get("size") or "Custom"),
                "estimated_vram": str(payload.get("estimated_vram") or "Unknown"),
                "languages": str(payload.get("languages") or "Model-defined"),
                "notes": str(payload.get("notes") or ""),
                "created_at": _utc_timestamp(),
                "builtin": False,
            }

            if source_kind == "url":
                url = str(payload.get("url") or "").strip()
                parsed = urllib.parse.urlparse(url)
                if parsed.scheme != "https" or not parsed.netloc:
                    raise WhisperDockError("Direct model links must be complete HTTPS URLs.", code="invalid_url")
                if engine != ENGINE_OPENAI:
                    raise WhisperDockError("A single direct file is supported only for the openai-whisper engine. Use a Hugging Face repository or local folder for Transformers.", code="invalid_source")
                suffix = Path(parsed.path).suffix.lower()
                if suffix not in _CHECKPOINT_SUFFIXES:
                    raise WhisperDockError("OpenAI Whisper checkpoints should be a .pt, .bin, or .ckpt file. Use a Hugging Face repository for a multi-file Transformers model.", code="invalid_checkpoint")
                filename = self.paths.safe_filename(str(payload.get("filename") or Path(parsed.path).name), "model.pt")
                record["path"] = f"models/custom/{requested_id}/{filename}"
                record["source"].update({"url": url, "filename": filename})

            elif source_kind == "huggingface":
                repo_id = str(payload.get("hf_repo") or payload.get("repo_id") or "").strip()
                if not _HF_REPO.fullmatch(repo_id):
                    raise WhisperDockError("Hugging Face repository must look like 'owner/model-name'.", code="invalid_huggingface_repo")
                revision = str(payload.get("hf_revision") or payload.get("revision") or "main").strip() or "main"
                if engine == ENGINE_OPENAI:
                    filename = str(payload.get("hf_filename") or payload.get("filename") or "").strip()
                    if not filename or Path(filename).name != filename or Path(filename).suffix.lower() not in _CHECKPOINT_SUFFIXES:
                        raise WhisperDockError("For the openai-whisper engine, provide the checkpoint filename (.pt, .bin, or .ckpt) in the Hugging Face repository.", code="invalid_checkpoint")
                    filename = self.paths.safe_filename(filename)
                    record["path"] = f"models/custom/{requested_id}/{filename}"
                    record["source"].update({"repo_id": repo_id, "revision": revision, "filename": filename})
                else:
                    record["path"] = f"models/huggingface/{requested_id}"
                    record["source"].update({"repo_id": repo_id, "revision": revision})

            else:  # local import
                source_value = str(payload.get("local_path") or payload.get("path") or "").strip()
                source_path = Path(source_value).expanduser().resolve()
                if not source_value or not source_path.exists():
                    raise WhisperDockError("The local model path does not exist.", code="local_model_missing")
                if engine == ENGINE_OPENAI and (not source_path.is_file() or source_path.suffix.lower() not in _CHECKPOINT_SUFFIXES):
                    raise WhisperDockError("An openai-whisper local import must be a .pt, .bin, or .ckpt checkpoint file.", code="invalid_checkpoint")
                if engine == ENGINE_TRANSFORMERS and not self._transformers_files_present(source_path):
                    raise WhisperDockError("A Transformers local import must be a model directory containing config.json and weight files.", code="invalid_transformers_model")
                if engine == ENGINE_OPENAI:
                    destination = self.paths.models / "custom" / requested_id / self.paths.safe_filename(source_path.name, "model.pt")
                else:
                    destination = self.paths.models / "huggingface" / requested_id
                self._import_local_model(source_path, destination)
                record["path"] = self.paths.relative(destination)
                record["source"].update({"imported": True})

            self._index["models"][requested_id] = record
            self._save_index()
            return self._public_record(record)

    def _import_local_model(self, source: Path, destination: Path) -> None:
        """Copy instead of linking so an imported model survives project moves."""
        destination_parent = destination.parent.resolve()
        destination_parent.mkdir(parents=True, exist_ok=True)
        if source == destination or (source.is_dir() and destination.is_relative_to(source)):
            raise WhisperDockError("The import destination cannot be inside the source directory.", code="invalid_local_model")
        if destination.exists():
            raise WhisperDockError("The target model directory already exists.", code="model_exists", status_code=409)
        try:
            if source.is_dir():
                shutil.copytree(source, destination, symlinks=False)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        except OSError as exc:
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination, ignore_errors=True)
                else:
                    destination.unlink(missing_ok=True)
            raise WhisperDockError(f"Could not import local model: {exc}", code="local_model_copy_failed") from exc

    def download(self, model_id: str) -> dict[str, Any]:
        """Download a registered model into the project-local models directory."""
        with self._lock:
            record = self._record_for(model_id)
            if self._installed(record):
                return self._public_record(record)
            source = record["source"]
            kind = source["kind"]
            try:
                if kind == "builtin":
                    self._download_builtin(record)
                elif kind == "url":
                    self._download_file(source["url"], self._model_path(record))
                elif kind == "huggingface":
                    self._download_huggingface(record)
                elif kind == "local":
                    raise WhisperDockError("This local model import did not complete. Remove it and import it again.", code="local_model_missing")
                else:
                    raise WhisperDockError("Unknown model source.", code="invalid_source")
            except WhisperDockError:
                raise
            except Exception as exc:
                raise WhisperDockError(f"Model download failed: {exc}", code="download_failed") from exc
            return self._public_record(record)

    def _download_builtin(self, record: dict[str, Any]) -> None:
        if importlib.util.find_spec("whisper") is None:
            raise DependencyUnavailable("openai-whisper", "Run the project-local installer, then retry the download.")
        if importlib.util.find_spec("torch") is None:
            raise DependencyUnavailable("PyTorch", "Install the runtime that matches this computer before downloading a built-in model.")
        import whisper  # type: ignore[import-not-found]

        model_name = record["source"]["model_name"]
        try:
            # ``_download`` is the package's checksum-verified model fetcher.
            # Calling ``load_model`` just to download a large checkpoint would
            # deserialize it into memory first, which defeats a separate
            # download/load UI and can exhaust RAM on portable computers.
            url = whisper._MODELS.get(model_name)
            downloader = getattr(whisper, "_download", None)
            if not url or downloader is None:
                raise RuntimeError("This openai-whisper version does not expose the requested official model URL.")
            downloaded_path = Path(downloader(url, str(self.paths.models / "openai-whisper"), in_memory=False))
            expected_path = self._model_path(record)
            if downloaded_path.resolve() != expected_path.resolve():
                expected_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(downloaded_path), expected_path)
        except Exception as exc:
            raise WhisperDockError(f"Could not download OpenAI Whisper model '{model_name}': {exc}", code="download_failed") from exc

    def _download_file(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.unlink(missing_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "WhisperDock/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
            if temporary.stat().st_size == 0:
                raise WhisperDockError("The model link returned an empty file.", code="download_failed")
            temporary.replace(destination)
        except urllib.error.URLError as exc:
            temporary.unlink(missing_ok=True)
            raise WhisperDockError(f"Could not reach the model link: {exc.reason}", code="download_failed") from exc
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise WhisperDockError(f"Could not save model file: {exc}", code="download_failed") from exc

    def _download_huggingface(self, record: dict[str, Any]) -> None:
        source = record["source"]
        destination = self._model_path(record)
        if record["engine"] == ENGINE_OPENAI:
            url = f"https://huggingface.co/{source['repo_id']}/resolve/{urllib.parse.quote(source['revision'], safe='')}/{urllib.parse.quote(source['filename'])}"
            self._download_file(url, destination)
            return
        if importlib.util.find_spec("huggingface_hub") is None:
            raise DependencyUnavailable("huggingface-hub", "Install the optional Transformers runtime to download a Hugging Face model folder.")
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=source["repo_id"],
                revision=source["revision"],
                local_dir=str(destination),
                cache_dir=str(self.paths.cache / "huggingface"),
            )
        except Exception as exc:
            raise WhisperDockError(f"Could not download Hugging Face model '{source['repo_id']}': {exc}", code="download_failed") from exc

    def _require_installed(self, record: dict[str, Any]) -> Path:
        location = self._model_path(record)
        if not self._installed(record):
            raise WhisperDockError("This model is not downloaded. Download or import it before loading.", code="model_not_downloaded", status_code=409)
        return location

    def _resolve_device(self, requested_device: str) -> str:
        requested_device = str(requested_device or "auto").lower()
        if requested_device not in {"auto", "cpu", "mps", "cuda"}:
            raise WhisperDockError("device must be auto, cpu, mps, or cuda.", code="invalid_device")
        if importlib.util.find_spec("torch") is None:
            raise DependencyUnavailable("PyTorch", "Install it in WhisperDock's local environment before loading a model.")
        import torch  # type: ignore[import-not-found]

        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if requested_device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if mps_available:
                return "mps"
            return "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise WhisperDockError("CUDA was selected but this PyTorch installation cannot see an NVIDIA GPU. Choose auto or cpu.", code="cuda_unavailable")
        if requested_device == "mps" and not mps_available:
            raise WhisperDockError("MPS was selected but this PyTorch installation cannot use Apple Silicon acceleration. Choose auto or cpu.", code="mps_unavailable")
        return requested_device

    def load(self, model_id: str, *, device: str = "auto") -> dict[str, Any]:
        with self._lock:
            record = self._record_for(model_id)
            location = self._require_installed(record)
            actual_device = self._resolve_device(device)
            existing = self._loaded.get(model_id)
            if existing and existing.device == actual_device:
                return self._public_record(record)
            if existing:
                self.unload(model_id)
            try:
                if record["engine"] == ENGINE_OPENAI:
                    loaded = self._load_openai(model_id, location, actual_device)
                else:
                    loaded = self._load_transformers(model_id, location, actual_device)
                self._loaded[model_id] = loaded
            except WhisperDockError:
                raise
            except Exception as exc:
                raise WhisperDockError(f"Could not load model '{record['name']}': {exc}", code="load_failed") from exc
            return self._public_record(record)

    def _load_openai(self, model_id: str, location: Path, device: str) -> LoadedModel:
        if importlib.util.find_spec("whisper") is None:
            raise DependencyUnavailable("openai-whisper", "Install it in WhisperDock's local environment before loading a model.")
        import whisper  # type: ignore[import-not-found]

        model = whisper.load_model(str(location), device=device, download_root=str(self.paths.models / "openai-whisper"))
        return LoadedModel(model_id=model_id, engine=ENGINE_OPENAI, device=device, model=model, loaded_at=_utc_timestamp())

    def _load_transformers(self, model_id: str, location: Path, device: str) -> LoadedModel:
        missing = [name for name in ("transformers", "torch") if importlib.util.find_spec(name) is None]
        if missing:
            raise DependencyUnavailable(", ".join(missing), "Install the optional Transformers runtime before loading this Hugging Face model.")
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline  # type: ignore[import-not-found]

        dtype = torch.float16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(str(location), local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(location),
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=0 if device == "cuda" else ("mps" if device == "mps" else -1),
        )
        return LoadedModel(
            model_id=model_id,
            engine=ENGINE_TRANSFORMERS,
            device=device,
            model=model,
            processor=processor,
            pipeline=transcriber,
            loaded_at=_utc_timestamp(),
        )

    def unload(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._record_for(model_id)
            loaded = self._loaded.pop(model_id, None)
            if loaded is not None:
                del loaded
                self._release_memory()
            return self._public_record(record)

    @staticmethod
    def _release_memory() -> None:
        gc.collect()
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch  # type: ignore[import-not-found]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

    def delete(self, model_id: str) -> None:
        if model_id in BUILTIN_MODELS:
            record = self._builtin_record(model_id)
        else:
            record = self._record_for(model_id)
        with self._lock:
            self.unload(model_id)
            location = self._model_path(record)
            if location.exists():
                if location.is_dir():
                    shutil.rmtree(location)
                else:
                    location.unlink()
                parent = location.parent
                # Remove only the per-model empty directory; never prune a shared root.
                if parent != self.paths.models and parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
            if model_id not in BUILTIN_MODELS:
                self._index["models"].pop(model_id, None)
                self._save_index()

    def transcribe(self, model_id: str, audio_path: Path, raw_options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run one file using an already-loaded model.

        The lock avoids concurrent calls into the same Torch module from batch
        jobs and segmented realtime requests.  The API remains asynchronous at
        the job level, while inference itself stays deterministic and safe.
        """
        with self._inference_lock:
            with self._lock:
                record = self._record_for(model_id)
                loaded = self._loaded.get(model_id)
                if loaded is None:
                    raise WhisperDockError("Load this model into memory before transcription.", code="model_not_loaded", status_code=409)
            if not audio_path.exists() or not audio_path.is_file():
                raise WhisperDockError("Audio input no longer exists in the project workspace.", code="audio_not_found", status_code=404)
            if loaded.engine == ENGINE_OPENAI:
                options = normalize_whisper_options(raw_options, device=loaded.device)
                thread_count = options.pop("threads", None)
                try:
                    old_thread_count = None
                    if thread_count is not None and importlib.util.find_spec("torch") is not None:
                        import torch  # type: ignore[import-not-found]

                        old_thread_count = torch.get_num_threads()
                        if int(thread_count) > 0:
                            torch.set_num_threads(int(thread_count))
                    result = loaded.model.transcribe(str(audio_path), **options)
                except Exception as exc:
                    raise WhisperDockError(f"Whisper could not transcribe '{audio_path.name}': {exc}", code="transcription_failed") from exc
                finally:
                    if old_thread_count is not None:
                        try:
                            torch.set_num_threads(old_thread_count)
                        except Exception:
                            pass
                return self._normalise_openai_result(result, record, options)
            return self._transcribe_transformers(loaded, record, audio_path, raw_options or {})

    @staticmethod
    def _normalise_openai_result(result: Any, record: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        result = result if isinstance(result, dict) else {"text": str(result)}
        return {
            "text": str(result.get("text") or "").strip(),
            "language": result.get("language"),
            "segments": result.get("segments") or [],
            "engine": record["engine"],
            "model_id": record["id"],
            "options": options,
        }

    def _transcribe_transformers(self, loaded: LoadedModel, record: dict[str, Any], audio_path: Path, raw_options: dict[str, Any]) -> dict[str, Any]:
        """Map common controls for community Transformers Whisper checkpoints.

        Community pipelines do not expose every OpenAI decoder option.  We
        preserve the OpenAI option schema for the default engine and apply the
        supported overlapping controls here.
        """
        task = raw_options.get("task", "transcribe")
        language = raw_options.get("language")
        generate_kwargs: dict[str, Any] = {}
        if task in {"transcribe", "translate"}:
            generate_kwargs["task"] = task
        if language not in {None, "", "auto"}:
            generate_kwargs["language"] = str(language)
        if raw_options.get("beam_size") not in {None, ""}:
            try:
                generate_kwargs["num_beams"] = int(raw_options["beam_size"])
            except (TypeError, ValueError) as exc:
                raise WhisperDockError("beam_size must be an integer for Transformers models.", code="invalid_option") from exc
        try:
            result = loaded.pipeline(
                str(audio_path),
                return_timestamps="word" if raw_options.get("word_timestamps") else False,
                generate_kwargs=generate_kwargs or None,
            )
        except Exception as exc:
            raise WhisperDockError(f"Transformers could not transcribe '{audio_path.name}': {exc}", code="transcription_failed") from exc
        if not isinstance(result, dict):
            result = {"text": str(result)}
        segments: list[dict[str, Any]] = []
        for index, chunk in enumerate(result.get("chunks") or []):
            timestamp = chunk.get("timestamp") if isinstance(chunk, dict) else None
            start = timestamp[0] if isinstance(timestamp, (list, tuple)) and timestamp else None
            end = timestamp[1] if isinstance(timestamp, (list, tuple)) and len(timestamp) > 1 else None
            segments.append({"id": index, "start": start, "end": end, "text": str(chunk.get("text") or "")})
        return {
            "text": str(result.get("text") or "").strip(),
            "language": language or None,
            "segments": segments,
            "engine": record["engine"],
            "model_id": record["id"],
            "options": raw_options,
        }

    def model_guide(self) -> dict[str, Any]:
        """Structured instructions for the UI's custom-model help panel."""
        return {
            "openai_whisper": {
                "engine": ENGINE_OPENAI,
                "description": "OpenAI Whisper's official built-ins are the easiest option. Custom checkpoints must be a single compatible PyTorch .pt, .bin, or .ckpt file.",
                "acceptable_sources": [
                    "A direct HTTPS link to one compatible checkpoint file.",
                    "A Hugging Face repo plus the exact checkpoint filename, for example owner/repo and model.pt.",
                    "A local compatible checkpoint file; WhisperDock copies it into models/.",
                ],
                "where_to_look": [
                    "https://huggingface.co/models?pipeline_tag=automatic-speech-recognition",
                    "OpenAI Whisper compatible fine-tunes that explicitly provide a PyTorch checkpoint.",
                ],
                "warning": "Most Hugging Face Whisper fine-tunes are Transformers model folders, not a single OpenAI Whisper checkpoint. Select the Transformers engine for those.",
            },
            "transformers": {
                "engine": ENGINE_TRANSFORMERS,
                "description": "Use for Hugging Face Whisper model repositories containing config.json, processor/tokenizer files, and weights such as model.safetensors or pytorch_model.bin.",
                "acceptable_sources": [
                    "Hugging Face repo id in owner/model-name form. WhisperDock downloads a complete snapshot into models/huggingface/.",
                    "A local exported Hugging Face model folder; WhisperDock copies the whole folder into models/huggingface/.",
                ],
                "where_to_look": [
                    "https://huggingface.co/models?pipeline_tag=automatic-speech-recognition",
                    "The desktop Chaozhou/Teochew example: panlr/whisper-finetune-teochew (a Transformers-format fine-tune).",
                ],
                "warning": "This adapter supports common task, language, beam-size, and word-timestamp controls. The complete OpenAI Whisper decoder controls apply to the default openai-whisper engine.",
            },
        }
