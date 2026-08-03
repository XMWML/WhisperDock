"""FastAPI application for the local WhisperDock Web UI.

The API intentionally has no cloud account or external database dependency.
Uploads, models, settings, and transcription outputs are all project-relative.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from .errors import WhisperDockError
from .jobs import JobManager, normalize_output_formats
from .model_manager import ENGINE_OPENAI, ENGINE_TRANSFORMERS, ModelManager
from .options import WHISPER_OPTION_METADATA, option_defaults
from .paths import ProjectPaths


DEFAULT_SETTINGS: dict[str, Any] = {
    "default_model": "base",
    "default_device": "auto",
    "default_output_formats": ["txt", "json", "srt", "vtt", "tsv"],
    "keep_uploads": True,
    "default_language": None,
    "default_prompt": "",
}


def _load_settings(paths: ProjectPaths) -> dict[str, Any]:
    saved = paths.read_json(paths.config / "settings.json", {})
    settings = dict(DEFAULT_SETTINGS)
    if isinstance(saved, dict):
        settings.update({key: value for key, value in saved.items() if key in DEFAULT_SETTINGS})
    return settings


def _write_settings(paths: ProjectPaths, candidate: dict[str, Any], models: ModelManager) -> dict[str, Any]:
    settings = _load_settings(paths)
    allowed = set(DEFAULT_SETTINGS)
    unknown = set(candidate) - allowed
    if unknown:
        raise WhisperDockError(f"Unsupported setting(s): {', '.join(sorted(unknown))}.", code="invalid_setting")
    settings.update(candidate)
    if not isinstance(settings["default_model"], str):
        raise WhisperDockError("default_model must be a model id.", code="invalid_setting")
    models.get_model(settings["default_model"])
    if settings["default_device"] not in {"auto", "cpu", "mps", "cuda"}:
        raise WhisperDockError("default_device must be auto, cpu, mps, or cuda.", code="invalid_setting")
    if not isinstance(settings["default_output_formats"], list) or not all(isinstance(item, str) for item in settings["default_output_formats"]):
        raise WhisperDockError("default_output_formats must be a list of format names.", code="invalid_setting")
    try:
        settings["default_output_formats"] = normalize_output_formats(settings["default_output_formats"])
    except WhisperDockError as exc:
        raise WhisperDockError(str(exc), code="invalid_setting") from exc
    if not isinstance(settings["keep_uploads"], bool):
        raise WhisperDockError("keep_uploads must be true or false.", code="invalid_setting")
    if settings["default_language"] is not None and not isinstance(settings["default_language"], str):
        raise WhisperDockError("default_language must be a language code or null.", code="invalid_setting")
    if not isinstance(settings["default_prompt"], str):
        raise WhisperDockError("default_prompt must be text.", code="invalid_setting")
    paths.write_json(paths.config / "settings.json", settings)
    return settings


def _parse_json_object(value: str | None, field: str) -> dict[str, Any]:
    if value is None or not value.strip():
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise WhisperDockError(f"{field} must be valid JSON.", code="invalid_json") from exc
    if not isinstance(decoded, dict):
        raise WhisperDockError(f"{field} must be a JSON object.", code="invalid_json")
    return decoded


def _prepare_transcription_options(raw_options: dict[str, Any]) -> tuple[dict[str, Any], Any | None]:
    """Translate the bundled UI's convenience fields to Whisper controls.

    ``vad_filter`` and ``keep_temp`` are UI/workspace preferences, not OpenAI
    Whisper parameters.  They are intentionally not forwarded to the model.
    ``hotwords`` is folded into ``initial_prompt`` because that is Whisper's
    native vocabulary-bias mechanism.
    """
    options = dict(raw_options)
    output_formats = options.pop("output_formats", options.pop("output_format", None))
    if "log_prob_threshold" in options and "logprob_threshold" not in options:
        options["logprob_threshold"] = options.pop("log_prob_threshold")
    hotwords = str(options.pop("hotwords", "") or "").strip()
    if hotwords and not options.get("initial_prompt"):
        options["initial_prompt"] = hotwords
    options.pop("vad_filter", None)
    options.pop("keep_temp", None)
    return options, output_formats


def _apply_transcription_defaults(options: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Apply project-persisted defaults only when the request omits a field."""
    effective = dict(options)
    default_language = settings.get("default_language")
    if "language" not in effective and isinstance(default_language, str) and default_language.strip() and default_language != "auto":
        effective["language"] = default_language
    default_prompt = settings.get("default_prompt")
    if "initial_prompt" not in effective and isinstance(default_prompt, str) and default_prompt.strip():
        effective["initial_prompt"] = default_prompt
    return effective


def _directory_size(directory: Path) -> int:
    total = 0
    if directory.exists():
        for child in directory.rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    continue
    return total


def _safe_audio_suffix(filename: str | None, content_type: str | None, *, fallback: str = ".wav") -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix and len(suffix) <= 12 and suffix[1:].replace("-", "").isalnum():
        return suffix
    content_type = (content_type or "").lower()
    for media_type, extension in {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
    }.items():
        if media_type in content_type:
            return extension
    return fallback


def _workspace_audio_path(paths: ProjectPaths, value: str) -> Path:
    """Accept an upload-relative path, but never a path outside workspace/."""
    raw_path = Path(value)
    if raw_path.is_absolute():
        raise WhisperDockError("Workspace audio paths must be relative.", code="unsafe_audio_path")
    try:
        candidate = (
            paths.from_relative(raw_path)
            if raw_path.parts and raw_path.parts[0] == "workspace"
            else paths.resolve_inside(paths.workspace, raw_path)
        )
    except WhisperDockError as exc:
        # Keep the public workspace endpoint's error vocabulary independent of
        # the lower-level project path helper used by other APIs.
        raise WhisperDockError("Audio must remain inside workspace/.", code="unsafe_audio_path") from exc
    try:
        candidate.relative_to(paths.workspace.resolve())
    except ValueError as exc:
        raise WhisperDockError("Audio must remain inside workspace/.", code="unsafe_audio_path") from exc
    return candidate


def create_app(root: str | Path | None = None):
    """Create the web service.

    FastAPI itself is a required web dependency; Whisper/Torch are deliberately
    *not* required for startup.  Their absence is reported through health and
    model endpoints, allowing the UI to show an install action instead of a
    server crash.
    """
    try:
        from fastapi import Body, FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FastAPI is not installed. Create WhisperDock's project-local virtual environment and install requirements.txt before starting the web server."
        ) from exc

    # ``from __future__ import annotations`` keeps import-time dependencies
    # optional, but FastAPI resolves endpoint annotations from module globals.
    # Publish these delayed imports before defining file/WebSocket routes.
    globals().update({"Request": Request, "UploadFile": UploadFile, "WebSocket": WebSocket, "WebSocketDisconnect": WebSocketDisconnect})

    paths = ProjectPaths(root)
    paths.ensure()
    # imageio-ffmpeg provides a bundled platform binary inside the project-local
    # environment, so Whisper can decode common media without a system install.
    try:
        import imageio_ffmpeg

        ffmpeg_parent = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
        if ffmpeg_parent not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = ffmpeg_parent + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
    models = ModelManager(paths)
    jobs = JobManager(paths, models)

    @asynccontextmanager
    async def lifespan(_: Any):
        try:
            yield
        finally:
            jobs.shutdown()

    app = FastAPI(
        title="WhisperDock API",
        version="1.0.0",
        description="Local, portable transcription API powered by OpenAI Whisper and optional Hugging Face Transformers models.",
        lifespan=lifespan,
    )
    app.state.paths = paths
    app.state.models = models
    app.state.jobs = jobs
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:8000", "http://localhost:8000"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(WhisperDockError)
    async def whisperdock_error_handler(_: Request, error: WhisperDockError):
        return JSONResponse(status_code=error.status_code, content={"detail": str(error), "error": error.as_dict()})

    @app.get("/")
    def root_status() -> Any:
        index = paths.root / "frontend" / "index.html"
        if index.is_file():
            return FileResponse(index)
        return {"name": "WhisperDock", "api": "/api", "docs": "/docs"}

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "WhisperDock",
            "version": "1.0.0",
            "project_root": str(paths.root),
            "workspace_path": str(paths.root),
            "storage": {
                "used_bytes": _directory_size(paths.models) + _directory_size(paths.outputs) + _directory_size(paths.workspace),
                "models_bytes": _directory_size(paths.models),
            },
            "dependencies": models.dependency_status(),
        }

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        dependencies = models.dependency_status()
        whisper_ready = dependencies["openai_whisper"] and dependencies["torch"]
        return {
            "engines": {
                ENGINE_OPENAI: {"available": whisper_ready, "default": True},
                ENGINE_TRANSFORMERS: {
                    "available": dependencies["transformers"] and dependencies["torch"],
                    "default": False,
                },
            },
            "audio": {"ffmpeg_available": dependencies["ffmpeg"], "note": "Whisper uses ffmpeg for compressed browser and media formats."},
            "realtime": {
                "available": whisper_ready,
                "mode": "segmented",
                "websocket": "/api/realtime",
                "http_chunk_endpoint": "/api/realtime/chunks",
                "note": "OpenAI Whisper is not a native streaming engine. WhisperDock transcribes self-contained audio chunks and returns one partial result per chunk.",
            },
            "storage": {
                "models": "models/",
                "cache": "cache/",
                "uploads": "workspace/",
                "outputs": "outputs/",
                "config": "config/",
            },
        }

    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return _load_settings(paths)

    @app.put("/api/settings")
    def update_settings(candidate: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return _write_settings(paths, candidate, models)

    @app.get("/api/whisper/options")
    def whisper_options() -> dict[str, Any]:
        return {
            "engine": ENGINE_OPENAI,
            "fields": WHISPER_OPTION_METADATA,
            "defaults": option_defaults(),
            "notes": [
                "All fields map to OpenAI Whisper transcribe/decode options.",
                "When beam_size is set, best_of is omitted because Whisper does not allow sampling and beam search together.",
                "fp16 is disabled automatically for CPU execution.",
            ],
        }

    @app.get("/api/model-guide")
    def model_guide() -> dict[str, Any]:
        return models.model_guide()

    @app.get("/api/models")
    def list_models() -> list[dict[str, Any]]:
        return models.list_models()

    @app.get("/api/models/catalog")
    def model_catalog() -> list[dict[str, Any]]:
        return models.catalog()

    @app.get("/api/models/{model_id}")
    def get_model(model_id: str) -> dict[str, Any]:
        return models.get_model(model_id)

    @app.post("/api/models", status_code=201)
    def create_model(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return models.create_custom_model(payload)

    @app.post("/api/models/import-file", status_code=201)
    async def import_checkpoint(
        file: UploadFile = File(...),
        name: str = Form(...),
        engine: str = Form(ENGINE_OPENAI),
        model_id: str | None = Form(None),
        parameters: str | None = Form(None),
        estimated_vram: str | None = Form(None),
        languages: str | None = Form(None),
    ) -> dict[str, Any]:
        """Import one OpenAI checkpoint supplied by the browser.

        Transformers checkpoints are multi-file directories, so their UI path
        is Hugging Face repository download or a local-folder import request.
        """
        normalized_engine = engine.strip().lower()
        if normalized_engine not in {ENGINE_OPENAI, "openai", "openai_whisper", "whisper"}:
            raise WhisperDockError("Browser file import accepts a single openai-whisper checkpoint. Use a Hugging Face repository or local folder for a Transformers model.", code="invalid_source")
        suffix = _safe_audio_suffix(file.filename, file.content_type, fallback="")
        if suffix not in {".pt", ".bin", ".ckpt"}:
            raise WhisperDockError("Upload a compatible .pt, .bin, or .ckpt OpenAI Whisper checkpoint.", code="invalid_checkpoint")
        temporary = paths.upload_destination(file.filename or "model.pt", prefix="model-import")
        await _save_upload(file, temporary)
        try:
            return models.create_custom_model(
                {
                    "name": name,
                    "id": model_id,
                    "engine": ENGINE_OPENAI,
                    "source_type": "local",
                    "local_path": str(temporary),
                    "parameters": parameters,
                    "estimated_vram": estimated_vram,
                    "languages": languages,
                }
            )
        finally:
            temporary.unlink(missing_ok=True)

    @app.post("/api/models/{model_id}/download")
    async def download_model(model_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(models.download, model_id)

    @app.post("/api/models/{model_id}/load")
    async def load_model(model_id: str, payload: dict[str, Any] | None = Body(None)) -> dict[str, Any]:
        requested_device = (payload or {}).get("device", _load_settings(paths)["default_device"])
        return await asyncio.to_thread(models.load, model_id, device=requested_device)

    @app.post("/api/models/{model_id}/unload")
    async def unload_model(model_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(models.unload, model_id)

    @app.delete("/api/models/{model_id}", status_code=204)
    async def delete_model(model_id: str) -> None:
        await asyncio.to_thread(models.delete, model_id)

    async def _save_upload(file: UploadFile, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("wb") as handle:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        except OSError as exc:
            destination.unlink(missing_ok=True)
            raise WhisperDockError(f"Could not save uploaded file: {exc}", code="upload_failed") from exc
        finally:
            await file.close()
        if not destination.exists() or destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise WhisperDockError("The uploaded audio file is empty.", code="empty_upload")
        return destination

    @app.post("/api/uploads", status_code=201)
    async def upload_audio(file: UploadFile = File(...)) -> dict[str, Any]:
        destination = paths.upload_destination(file.filename or "audio", prefix="audio")
        await _save_upload(file, destination)
        return {
            "name": file.filename or destination.name,
            "path": paths.relative(destination),
            "size_bytes": destination.stat().st_size,
        }

    @app.post("/api/transcriptions", status_code=202)
    async def transcribe_single(
        file: UploadFile = File(...),
        model_id: str | None = Form(None),
        options: str | None = Form(None),
        output_formats: str | None = Form(None),
    ) -> dict[str, Any]:
        destination = paths.upload_destination(file.filename or "audio", prefix="audio")
        await _save_upload(file, destination)
        parsed_options, ui_formats = _prepare_transcription_options(_parse_json_object(options, "options"))
        settings = _load_settings(paths)
        return jobs.submit(
            [destination],
            model_id=model_id or settings["default_model"],
            options=_apply_transcription_defaults(parsed_options, settings),
            output_formats=output_formats or ui_formats or settings["default_output_formats"],
            kind="single",
            keep_uploads=settings["keep_uploads"],
        )

    @app.post("/api/transcriptions/batch", status_code=202)
    @app.post("/api/batches", status_code=202)
    async def transcribe_batch(
        files: list[UploadFile] = File(...),
        model_id: str | None = Form(None),
        options: str | None = Form(None),
        output_formats: str | None = Form(None),
    ) -> dict[str, Any]:
        destinations: list[Path] = []
        for file in files:
            destination = paths.upload_destination(file.filename or "audio", prefix="audio")
            await _save_upload(file, destination)
            destinations.append(destination)
        parsed_options, ui_formats = _prepare_transcription_options(_parse_json_object(options, "options"))
        settings = _load_settings(paths)
        return jobs.submit(
            destinations,
            model_id=model_id or settings["default_model"],
            options=_apply_transcription_defaults(parsed_options, settings),
            output_formats=output_formats or ui_formats or settings["default_output_formats"],
            kind="batch",
            keep_uploads=settings["keep_uploads"],
        )

    @app.post("/api/transcriptions/workspace", status_code=202)
    def transcribe_workspace(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        model_id = payload.get("model_id") or _load_settings(paths)["default_model"]
        file_values = payload.get("files")
        if not isinstance(model_id, str) or not isinstance(file_values, list) or not all(isinstance(item, str) for item in file_values):
            raise WhisperDockError("model_id and a list of workspace file paths are required.", code="invalid_transcription")
        files = [_workspace_audio_path(paths, value) for value in file_values]
        options = payload.get("options") or {}
        if not isinstance(options, dict):
            raise WhisperDockError("options must be a JSON object.", code="invalid_options")
        options, ui_formats = _prepare_transcription_options(options)
        settings = _load_settings(paths)
        return jobs.submit(
            files,
            model_id=model_id,
            options=_apply_transcription_defaults(options, settings),
            output_formats=payload.get("output_formats") or ui_formats or settings["default_output_formats"],
            kind="batch" if len(files) > 1 else "single",
            # Workspace callers may have intentionally kept input material in
            # the project, so only upload endpoints participate in cleanup.
            keep_uploads=True,
        )

    @app.get("/api/jobs")
    def list_jobs(limit: int = 100) -> list[dict[str, Any]]:
        return jobs.list_jobs(limit=limit)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return jobs.get_job(job_id)

    @app.get("/api/jobs/{job_id}/download/{output_format}")
    def download_job_output(job_id: str, output_format: str, result_index: int | None = None):
        path = jobs.get_download(job_id, output_format, result_index=result_index)
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> None:
        jobs.delete_job(job_id)

    @app.get("/api/history")
    def history(limit: int = 100) -> list[dict[str, Any]]:
        return jobs.list_jobs(limit=limit)

    @app.get("/api/history/export")
    def export_history(id: str | None = None):
        if id:
            job = jobs.get_job(id)
            for output_format in ("txt", "csv", "batch_json", "json", "vtt", "srt", "tsv"):
                if output_format in job.get("downloads", {}):
                    path = jobs.get_download(id, output_format)
                    return FileResponse(path, filename=path.name, media_type="application/octet-stream")
        path = jobs.export_history_csv()
        return FileResponse(path, filename=path.name, media_type="text/csv")

    @app.delete("/api/history", status_code=204)
    def clear_history() -> None:
        jobs.clear_history()

    @app.post("/api/realtime/chunks", status_code=202)
    async def transcribe_realtime_chunk(
        file: UploadFile = File(...),
        model_id: str | None = Form(None),
        options: str | None = Form(None),
        output_formats: str | None = Form(None),
    ) -> dict[str, Any]:
        """Queue one self-contained microphone chunk for segmented realtime use."""
        suffix = _safe_audio_suffix(file.filename, file.content_type, fallback=".webm")
        destination = paths.workspace / "realtime" / f"chunk-{uuid.uuid4().hex}{suffix}"
        await _save_upload(file, destination)
        parsed_options, ui_formats = _prepare_transcription_options(_parse_json_object(options, "options"))
        settings = _load_settings(paths)
        return jobs.submit(
            [destination],
            model_id=model_id or settings["default_model"],
            options=_apply_transcription_defaults(parsed_options, settings),
            output_formats=output_formats or ui_formats or ["txt", "json"],
            kind="realtime",
            keep_uploads=settings["keep_uploads"],
        )

    @app.websocket("/api/realtime")
    async def realtime_socket(websocket: WebSocket) -> None:
        """Segmented realtime protocol.

        1. Send ``{"type":"configure","model_id":"base","options":{...},"format":"webm"}``.
        2. Send a binary, independently-decodable audio chunk (1-10 seconds is
           usually practical).
        3. Receive ``partial`` or ``error`` for that chunk.  Send ``close`` to
           end the session.  This is deliberately documented as chunked rather
           than native streaming because OpenAI Whisper has no streaming API.
        """
        await websocket.accept()
        session_id = uuid.uuid4().hex
        configuration: dict[str, Any] | None = None
        await websocket.send_json(
            {
                "type": "ready",
                "mode": "segmented",
                "session_id": session_id,
                "message": "Configure a loaded model, then send self-contained binary audio chunks.",
            }
        )
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                text = message.get("text")
                if text is not None:
                    try:
                        command = json.loads(text)
                    except json.JSONDecodeError:
                        await websocket.send_json({"type": "error", "error": {"code": "invalid_json", "message": "Realtime commands must be JSON."}})
                        continue
                    if not isinstance(command, dict):
                        await websocket.send_json({"type": "error", "error": {"code": "invalid_command", "message": "Realtime commands must be objects."}})
                        continue
                    command_type = command.get("type")
                    if command_type == "configure":
                        model_id = command.get("model_id")
                        options = command.get("options") or {}
                        if not isinstance(model_id, str) or not isinstance(options, dict):
                            await websocket.send_json({"type": "error", "error": {"code": "invalid_configuration", "message": "model_id and an options object are required."}})
                            continue
                        try:
                            model = models.get_model(model_id)
                            if not model["loaded"]:
                                raise WhisperDockError("Load the selected model before starting realtime transcription.", code="model_not_loaded", status_code=409)
                            format_name = str(command.get("format") or "webm").lower().lstrip(".")
                            if not format_name.isalnum() or len(format_name) > 10:
                                raise WhisperDockError("Realtime format must be a short file extension such as webm or wav.", code="invalid_audio_format")
                            configuration = {"model_id": model_id, "options": options, "format": format_name, "sequence": 0}
                            await websocket.send_json({"type": "configured", "model": model, "mode": "segmented"})
                        except WhisperDockError as exc:
                            await websocket.send_json({"type": "error", "error": exc.as_dict()})
                        continue
                    if command_type == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue
                    if command_type == "close":
                        await websocket.close()
                        return
                    await websocket.send_json({"type": "error", "error": {"code": "unknown_command", "message": "Use configure, ping, or close."}})
                    continue

                audio = message.get("bytes")
                if audio is None:
                    continue
                if configuration is None:
                    await websocket.send_json({"type": "error", "error": {"code": "not_configured", "message": "Send a configure command before audio bytes."}})
                    continue
                configuration["sequence"] += 1
                sequence = configuration["sequence"]
                destination = paths.workspace / "realtime" / f"{session_id}-{sequence:06d}.{configuration['format']}"
                destination.write_bytes(audio)
                try:
                    result = await asyncio.to_thread(models.transcribe, configuration["model_id"], destination, configuration["options"])
                    await websocket.send_json(
                        {
                            "type": "partial",
                            "sequence": sequence,
                            "text": result["text"],
                            "language": result.get("language"),
                            "segments": result.get("segments") or [],
                        }
                    )
                except WhisperDockError as exc:
                    await websocket.send_json({"type": "error", "sequence": sequence, "error": exc.as_dict()})
        except WebSocketDisconnect:
            return

    frontend_directory = paths.root / "frontend"
    if frontend_directory.is_dir():
        # This is registered after all API and WebSocket routes, so /api/* is
        # never mistaken for a static file while one-command launch serves the
        # actual Web UI at http://127.0.0.1:<port>/.
        app.mount("/frontend", StaticFiles(directory=str(frontend_directory)), name="frontend-assets")
        app.mount("/", StaticFiles(directory=str(frontend_directory), html=True), name="frontend")

    return app


try:
    # Importing the package for pure-Python tools should remain possible before
    # dependencies are installed.  ``python -m backend.main`` prints a precise
    # setup message in that case.
    app = create_app()
    _APP_IMPORT_ERROR: RuntimeError | None = None
except RuntimeError as error:
    app = None
    _APP_IMPORT_ERROR = error


def main() -> int:
    if app is None:
        print(_APP_IMPORT_ERROR or "WhisperDock could not create its API.", file=sys.stderr)
        return 1
    try:
        import uvicorn
    except ModuleNotFoundError:
        print("uvicorn is not installed. Install WhisperDock's requirements.txt in the project-local environment.", file=sys.stderr)
        return 1
    host = os.environ.get("WHISPERDOCK_HOST", "127.0.0.1")
    port = int(os.environ.get("WHISPERDOCK_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
