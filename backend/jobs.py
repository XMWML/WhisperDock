"""Persistent background transcription jobs and portable result writers."""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from .errors import WhisperDockError
from .model_manager import ModelManager
from .paths import ProjectPaths


SUPPORTED_OUTPUT_FORMATS = {"txt", "json", "srt", "vtt", "tsv", "csv"}
_JOB_ID = re.compile(r"[a-f0-9]{32}")


def _utc_timestamp() -> str:
    from time import gmtime, strftime

    return strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())


def _json_safe(value: Any) -> Any:
    """Convert common model return values into values accepted by json.dump."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _seconds_to_srt(value: Any) -> str:
    try:
        milliseconds = max(0, round(float(value) * 1000))
    except (TypeError, ValueError):
        milliseconds = 0
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _seconds_to_vtt(value: Any) -> str:
    return _seconds_to_srt(value).replace(",", ".")


def _segments(result: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for index, segment in enumerate(result.get("segments") or []):
        if not isinstance(segment, dict):
            continue
        start, end = segment.get("start"), segment.get("end")
        if start is None or end is None:
            continue
        yield {
            "id": segment.get("id", index),
            "start": start,
            "end": end,
            "text": str(segment.get("text") or "").strip(),
        }


def render_output(result: dict[str, Any], output_format: str) -> str:
    """Render the core result formats without relying on Whisper's private API."""
    output_format = output_format.lower()
    text = str(result.get("text") or "").strip()
    if output_format == "txt":
        return f"{text}\n"
    if output_format == "json":
        return json.dumps(_json_safe(result), ensure_ascii=False, indent=2) + "\n"
    if output_format == "tsv":
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
        writer.writerow(["start", "end", "text"])
        rows = list(_segments(result))
        if rows:
            for segment in rows:
                writer.writerow([segment["start"], segment["end"], segment["text"]])
        else:
            writer.writerow([0, "", text])
        return buffer.getvalue()
    if output_format == "srt":
        rows = list(_segments(result))
        if not rows:
            return f"1\n00:00:00,000 --> 00:00:00,000\n{text}\n"
        return "\n".join(
            f"{index + 1}\n{_seconds_to_srt(segment['start'])} --> {_seconds_to_srt(segment['end'])}\n{segment['text']}\n"
            for index, segment in enumerate(rows)
        )
    if output_format == "vtt":
        rows = list(_segments(result))
        if not rows:
            return f"WEBVTT\n\n00:00:00.000 --> 00:00:00.000\n{text}\n"
        cues = "\n".join(
            f"{_seconds_to_vtt(segment['start'])} --> {_seconds_to_vtt(segment['end'])}\n{segment['text']}\n"
            for segment in rows
        )
        return f"WEBVTT\n\n{cues}"
    raise WhisperDockError(f"Unsupported output format '{output_format}'.", code="invalid_output_format")


def normalize_output_formats(value: Any, *, batch: bool = False) -> list[str]:
    if value is None or value == "":
        formats = ["txt", "json", "srt", "vtt", "tsv"]
    elif isinstance(value, str):
        formats = [item.strip().lower() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        formats = [str(item).strip().lower() for item in value if str(item).strip()]
    else:
        raise WhisperDockError("output_formats must be a list or comma-separated string.", code="invalid_output_format")
    if batch and "csv" not in formats:
        formats.append("csv")
    invalid = set(formats) - SUPPORTED_OUTPUT_FORMATS
    if invalid:
        raise WhisperDockError(f"Unsupported output format(s): {', '.join(sorted(invalid))}.", code="invalid_output_format")
    # Preserve the user's order, without duplicate links in the UI.
    return list(dict.fromkeys(formats))


class JobManager:
    """Run serial inference jobs while retaining result metadata on disk."""

    def __init__(self, paths: ProjectPaths, models: ModelManager):
        self.paths = paths
        self.models = models
        self.paths.ensure()
        self.history_path = self.paths.outputs / "history.json"
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisperdock-job")
        self._jobs: dict[str, dict[str, Any]] = self._load_jobs()

    def _load_jobs(self) -> dict[str, dict[str, Any]]:
        saved = self.paths.read_json(self.history_path, {"version": 1, "jobs": []})
        records = saved.get("jobs", []) if isinstance(saved, dict) else []
        jobs: dict[str, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not _JOB_ID.fullmatch(record["id"]):
                continue
            # Jobs cannot survive a process restart mid-inference. Marking the
            # old entry failed makes that fact visible rather than pretending it
            # will resume in a new process.
            if record.get("status") in {"queued", "running"}:
                record["status"] = "interrupted"
                record["error"] = "The application restarted before this transcription finished."
                record["finished_at"] = _utc_timestamp()
            jobs[record["id"]] = record
        return jobs

    def _save_jobs(self) -> None:
        records = sorted(self._jobs.values(), key=lambda item: item.get("created_at", ""), reverse=True)
        self.paths.write_json(self.history_path, {"version": 1, "jobs": records})

    @staticmethod
    def _job_view(job: dict[str, Any]) -> dict[str, Any]:
        """Return a browser-friendly view without exposing filesystem paths."""
        view = _json_safe(job)
        job_id = view["id"]

        def download_urls(downloads: Any, *, result_index: int | None = None) -> dict[str, str]:
            if not isinstance(downloads, dict):
                return {}
            suffix = f"?result_index={result_index}" if result_index is not None else ""
            return {
                str(output_format): f"/api/jobs/{job_id}/download/{output_format}{suffix}"
                for output_format, relative_path in downloads.items()
                if isinstance(relative_path, str)
            }

        view["job_id"] = job_id
        progress = view.get("progress") if isinstance(view.get("progress"), dict) else {}
        completed = int(progress.get("completed") or 0)
        total = int(progress.get("total") or 0)
        view["percentage"] = round((completed / total) * 100) if total else 0
        view["message"] = {
            "queued": "Queued for transcription.",
            "running": "Transcribing audio.",
            "completed": "Transcription complete.",
            "completed_with_errors": "Transcription completed with some errors.",
            "failed": "Transcription failed.",
            "cancelled": "Transcription cancelled.",
            "interrupted": "Transcription was interrupted by restart.",
        }.get(str(view.get("status")), "Transcription status updated.")
        # Keep project-relative download paths in ``downloads`` for API clients
        # that need portable metadata; expose browser-ready routes separately.
        view["output_urls"] = download_urls(view.get("downloads"))
        view["completed_at"] = view.get("finished_at")

        public_results: list[dict[str, Any]] = []
        for result_index, item in enumerate(view.get("results") or []):
            if not isinstance(item, dict):
                continue
            item["output_urls"] = download_urls(item.get("downloads"), result_index=result_index)
            public_results.append(item)
        view["results"] = public_results

        successful = [item for item in public_results if item.get("status") == "completed" and isinstance(item.get("result"), dict)]
        if len(successful) == 1:
            item = successful[0]
            summary = dict(item["result"])
            summary.update(
                {
                    "filename": item.get("source", {}).get("name", "transcript"),
                    "downloads": item.get("downloads", {}),
                    "outputs": item.get("output_urls", {}),
                    "model_id": view.get("model_id"),
                }
            )
            view["result"] = summary
            view["filename"] = summary["filename"]
            view["text"] = summary.get("text", "")
            view["language"] = summary.get("language")
        elif public_results:
            texts = [str(item.get("text") or "") for item in successful]
            view["result"] = {
                "text": "\n\n".join(text for text in texts if text),
                "results": public_results,
                "downloads": view.get("downloads", {}),
                "outputs": view["output_urls"],
                "model_id": view.get("model_id"),
                "filename": "batch-results",
            }
            view["filename"] = "batch-results"
            view["text"] = view["result"]["text"]
        return view

    def list_jobs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            records = sorted(self._jobs.values(), key=lambda item: item.get("created_at", ""), reverse=True)
            return [self._job_view(item) for item in records[:max(1, min(limit, 500))]]

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                return self._job_view(self._jobs[job_id])
            except KeyError as exc:
                raise WhisperDockError("Transcription job was not found.", code="job_not_found", status_code=404) from exc

    def submit(
        self,
        files: list[Path],
        *,
        model_id: str,
        options: dict[str, Any] | None = None,
        output_formats: Any = None,
        kind: str = "single",
        keep_uploads: bool = True,
    ) -> dict[str, Any]:
        if not files:
            raise WhisperDockError("Choose at least one audio file.", code="audio_required")
        if kind not in {"single", "batch", "realtime"}:
            raise WhisperDockError("Unknown transcription job kind.", code="invalid_job")
        if not isinstance(keep_uploads, bool):
            raise WhisperDockError("keep_uploads must be true or false.", code="invalid_job")
        self.models.get_model(model_id)
        is_batch = kind == "batch" or len(files) > 1
        formats = normalize_output_formats(output_formats, batch=is_batch)
        source_files: list[dict[str, str]] = []
        for file_path in files:
            file_path = Path(file_path).resolve()
            if not file_path.is_file():
                raise WhisperDockError(f"Audio file '{file_path.name}' is unavailable.", code="audio_not_found", status_code=404)
            try:
                relative_path = self.paths.relative(file_path)
            except WhisperDockError as exc:
                raise WhisperDockError("Audio must be uploaded into the WhisperDock workspace first.", code="unsafe_audio_path") from exc
            source_files.append({"name": file_path.name, "path": relative_path})

        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "kind": "batch" if is_batch else kind,
            "model_id": model_id,
            "status": "queued",
            "created_at": _utc_timestamp(),
            "started_at": None,
            "finished_at": None,
            "progress": {"completed": 0, "total": len(source_files)},
            "files": source_files,
            "options": _json_safe(options or {}),
            "output_formats": formats,
            "keep_uploads": keep_uploads,
            "results": [],
            "downloads": {},
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._save_jobs()
        self._executor.submit(self._run_job, job_id)
        return self._job_view(job)

    def _update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            job = self._jobs[job_id]
            job.update(changes)
            self._save_jobs()
            return job

    def _run_job(self, job_id: str) -> None:
        self._update(job_id, status="running", started_at=_utc_timestamp())
        with self._lock:
            job = self._jobs[job_id]
            files = list(job["files"])
            model_id = job["model_id"]
            options = dict(job["options"])
            formats = list(job["output_formats"])
            keep_uploads = bool(job.get("keep_uploads", True))
        result_items: list[dict[str, Any]] = []
        for index, source in enumerate(files):
            try:
                # A history file can be manually edited or moved between
                # machines. Resolve it inside the exception boundary so a bad
                # stored path becomes a visible failed item, never a stranded
                # worker thread with a job stuck in "running".
                audio_path = self.paths.from_relative(source["path"])
                result = self.models.transcribe(model_id, audio_path, options)
                result_files = self._write_result_files(job_id, index, source["name"], result, formats)
                result_items.append(
                    {
                        "source": source,
                        "status": "completed",
                        "text": result["text"],
                        "language": result.get("language"),
                        "result": _json_safe(result),
                        "downloads": result_files,
                    }
                )
            except WhisperDockError as exc:
                result_items.append({"source": source, "status": "failed", "error": exc.as_dict(), "downloads": {}})
            except Exception as exc:  # Defensive boundary for a background worker.
                result_items.append(
                    {
                        "source": source,
                        "status": "failed",
                        "error": {"code": "unexpected_error", "message": str(exc)},
                        "downloads": {},
                    }
                )
            with self._lock:
                current = self._jobs[job_id]
                current["results"] = result_items
                current["progress"] = {"completed": index + 1, "total": len(files)}
                self._save_jobs()

        completed = [item for item in result_items if item["status"] == "completed"]
        if not completed:
            status = "failed"
            error = "No audio file could be transcribed."
        elif len(completed) < len(result_items):
            status = "completed_with_errors"
            error = "Some audio files could not be transcribed."
        else:
            status = "completed"
            error = None
        downloads: dict[str, str] = {}
        try:
            downloads = self._write_job_exports(job_id, result_items, formats, is_batch=len(files) > 1)
        except Exception as exc:  # A full disk must not strand the job as "running".
            status = "completed_with_errors" if completed else "failed"
            export_error = f"Transcription finished, but result exports could not be written: {exc}"
            error = f"{error} {export_error}" if error else export_error
        if not keep_uploads:
            self._remove_managed_uploads(files)
        self._update(
            job_id,
            status=status,
            finished_at=_utc_timestamp(),
            results=result_items,
            downloads=downloads,
            error=error,
        )

    def _remove_managed_uploads(self, files: list[dict[str, str]]) -> None:
        """Delete only server-created uploads after an explicit opt-out.

        Workspace paths submitted by an API user may be project-owned input
        files.  This guard limits cleanup to the two directories written by the
        upload and realtime endpoints.
        """
        managed_roots = (self.paths.workspace / "uploads", self.paths.workspace / "realtime")
        for source in files:
            try:
                path = self.paths.from_relative(source["path"])
            except (KeyError, WhisperDockError):
                continue
            if not path.is_file():
                continue
            try:
                if any(path.is_relative_to(root.resolve()) for root in managed_roots):
                    path.unlink()
            except OSError:
                continue

    def _write_result_files(
        self,
        job_id: str,
        index: int,
        source_name: str,
        result: dict[str, Any],
        formats: list[str],
    ) -> dict[str, str]:
        directory = self.paths.job_directory(job_id)
        stem = self.paths.safe_filename(Path(source_name).stem, f"audio-{index + 1}")
        prefix = f"{index + 1:03d}-{stem}"
        downloads: dict[str, str] = {}
        for output_format in formats:
            if output_format == "csv":
                continue
            destination = directory / f"{prefix}.{output_format}"
            destination.write_text(render_output(result, output_format), encoding="utf-8")
            downloads[output_format] = self.paths.relative(destination)
        return downloads

    def _write_job_exports(
        self,
        job_id: str,
        results: list[dict[str, Any]],
        formats: list[str],
        *,
        is_batch: bool,
    ) -> dict[str, str]:
        directory = self.paths.job_directory(job_id)
        downloads: dict[str, str] = {}
        successful = [item for item in results if item["status"] == "completed"]
        if not is_batch and len(successful) == 1:
            downloads.update(successful[0]["downloads"])
        if is_batch:
            if "csv" in formats:
                destination = directory / "batch-results.csv"
                with destination.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["file", "status", "language", "text", "error"])
                    for item in results:
                        error = item.get("error", {}).get("message", "") if isinstance(item.get("error"), dict) else ""
                        writer.writerow(
                            [
                                item["source"]["name"],
                                item["status"],
                                item.get("language") or "",
                                item.get("text") or "",
                                error,
                            ]
                        )
                downloads["csv"] = self.paths.relative(destination)
            destination = directory / "batch-results.json"
            destination.write_text(json.dumps(_json_safe(results), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            downloads["batch_json"] = self.paths.relative(destination)
        return downloads

    def get_download(self, job_id: str, output_format: str, *, result_index: int | None = None) -> Path:
        with self._lock:
            try:
                job = self._jobs[job_id]
            except KeyError as exc:
                raise WhisperDockError("Transcription job was not found.", code="job_not_found", status_code=404) from exc
            key = output_format.lower()
            if result_index is None:
                relative_path = job.get("downloads", {}).get(key)
            else:
                results = job.get("results", [])
                if result_index < 0 or not isinstance(results, list) or result_index >= len(results):
                    raise WhisperDockError("That result item is not available for this job.", code="download_not_found", status_code=404)
                result = results[result_index]
                relative_path = result.get("downloads", {}).get(key) if isinstance(result, dict) else None
        if not isinstance(relative_path, str):
            raise WhisperDockError("That result format is not available for this job.", code="download_not_found", status_code=404)
        path = self.paths.from_relative(relative_path)
        if not path.is_file():
            raise WhisperDockError("The result file is no longer available.", code="download_not_found", status_code=404)
        return path

    def export_history_csv(self) -> Path:
        destination = self.paths.outputs / "history-export.csv"
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["job_id", "created_at", "finished_at", "kind", "model_id", "file", "status", "language", "text", "error"])
            for job in self.list_jobs(limit=500):
                for item in job.get("results", []):
                    error = item.get("error", {}).get("message", "") if isinstance(item.get("error"), dict) else ""
                    writer.writerow(
                        [
                            job["id"],
                            job.get("created_at"),
                            job.get("finished_at"),
                            job.get("kind"),
                            job.get("model_id"),
                            item.get("source", {}).get("name", ""),
                            item.get("status", ""),
                            item.get("language") or "",
                            item.get("text") or "",
                            error,
                        ]
                    )
        return destination

    def delete_job(self, job_id: str) -> None:
        with self._lock:
            if job_id not in self._jobs:
                raise WhisperDockError("Transcription job was not found.", code="job_not_found", status_code=404)
            job = self._jobs[job_id]
            if job["status"] in {"queued", "running"}:
                raise WhisperDockError("A running job cannot be deleted.", code="job_running", status_code=409)
            del self._jobs[job_id]
            self._save_jobs()
        directory = self.paths.outputs / "jobs" / job_id
        if directory.exists():
            shutil.rmtree(directory)

    def clear_history(self) -> None:
        """Remove completed job metadata and output folders after an explicit API call."""
        with self._lock:
            active = [job for job in self._jobs.values() if job["status"] in {"queued", "running"}]
            if active:
                raise WhisperDockError("Wait for running jobs before clearing history.", code="job_running", status_code=409)
            self._jobs.clear()
            self._save_jobs()
        jobs_directory = self.paths.outputs / "jobs"
        if jobs_directory.exists():
            shutil.rmtree(jobs_directory)
        jobs_directory.mkdir(parents=True, exist_ok=True)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)
