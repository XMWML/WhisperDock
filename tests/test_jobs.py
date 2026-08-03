"""Portable job/result behavior without a real model download or GPU."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from backend.errors import WhisperDockError
from backend.jobs import JobManager, normalize_output_formats, render_output
from backend.paths import ProjectPaths


class FakeModels:
    def __init__(self, *, fail_names: set[str] | None = None):
        self.fail_names = fail_names or set()
        self.calls: list[tuple[str, Path, dict]] = []

    def get_model(self, model_id: str):
        if model_id != "base":
            raise WhisperDockError("unknown", code="model_not_found", status_code=404)
        return {"id": model_id}

    def transcribe(self, model_id: str, audio_path: Path, options: dict):
        self.calls.append((model_id, audio_path, options))
        if audio_path.name in self.fail_names:
            raise WhisperDockError("recognition failed", code="transcription_failed")
        return {
            "text": f"text for {audio_path.stem}",
            "language": "zh",
            "segments": [
                {"id": 0, "start": 0, "end": 1.25, "text": "hello"},
                {"id": 1, "start": 1.25, "end": 2.0, "text": "world"},
            ],
            "model_id": model_id,
        }


def make_jobs(tmp_path: Path, models: FakeModels | None = None) -> tuple[ProjectPaths, JobManager, FakeModels]:
    paths = ProjectPaths(tmp_path / "WhisperDock")
    fake_models = models or FakeModels()
    return paths, JobManager(paths, fake_models), fake_models


def wait_for_completion(manager: JobManager, job_id: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = manager.get_job(job_id)
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish within the test deadline")


def test_rendered_outputs_are_standard_and_unicode_safe():
    result = {
        "text": "潮汕话文本",
        "segments": [{"start": 1.25, "end": 2, "text": "第一句"}],
    }

    assert render_output(result, "txt") == "潮汕话文本\n"
    assert '"text": "潮汕话文本"' in render_output(result, "json")
    assert render_output(result, "srt") == "1\n00:00:01,250 --> 00:00:02,000\n第一句\n"
    assert render_output(result, "vtt") == "WEBVTT\n\n00:00:01.250 --> 00:00:02.000\n第一句\n"
    assert render_output(result, "tsv") == "start\tend\ttext\n1.25\t2\t第一句\n"


def test_output_format_normalization_is_deduplicated_and_batch_adds_csv():
    assert normalize_output_formats("txt, json, txt") == ["txt", "json"]
    assert normalize_output_formats(["vtt", "srt"], batch=True) == ["vtt", "srt", "csv"]
    with pytest.raises(WhisperDockError) as caught:
        normalize_output_formats("docx")
    assert caught.value.code == "invalid_output_format"


def test_job_rejects_audio_outside_the_portable_workspace(tmp_path):
    paths, manager, _ = make_jobs(tmp_path)
    external_audio = tmp_path / "outside.wav"
    external_audio.write_bytes(b"audio")

    with pytest.raises(WhisperDockError) as caught:
        manager.submit([external_audio], model_id="base")

    assert caught.value.code == "unsafe_audio_path"
    manager.shutdown()


def test_single_job_writes_each_requested_download_under_project_root(tmp_path):
    paths, manager, models = make_jobs(tmp_path)
    audio = paths.workspace / "uploads" / "clip.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"audio")

    submitted = manager.submit(
        [audio], model_id="base", options={"language": "zh"}, output_formats=["txt", "json", "srt", "vtt", "tsv"]
    )
    completed = wait_for_completion(manager, submitted["id"])

    assert completed["status"] == "completed"
    assert completed["progress"] == {"completed": 1, "total": 1}
    assert models.calls == [("base", audio.resolve(), {"language": "zh"})]
    assert set(completed["downloads"]) == {"txt", "json", "srt", "vtt", "tsv"}
    for output_format, relative_path in completed["downloads"].items():
        path = manager.get_download(completed["id"], output_format)
        assert path.is_file()
        assert path == paths.from_relative(relative_path)
        assert path.is_relative_to(paths.root)
    assert manager.get_download(completed["id"], "txt").read_text(encoding="utf-8") == "text for clip\n"
    manager.shutdown()


def test_batch_job_keeps_successful_results_when_one_file_fails(tmp_path):
    paths = ProjectPaths(tmp_path / "WhisperDock")
    models = FakeModels(fail_names={"bad.wav"})
    manager = JobManager(paths, models)
    uploads = paths.workspace / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    good, bad = uploads / "good.wav", uploads / "bad.wav"
    good.write_bytes(b"audio")
    bad.write_bytes(b"audio")

    submitted = manager.submit([good, bad], model_id="base", output_formats="txt")
    completed = wait_for_completion(manager, submitted["id"])

    assert completed["kind"] == "batch"
    assert completed["status"] == "completed_with_errors"
    assert [item["status"] for item in completed["results"]] == ["completed", "failed"]
    assert set(completed["downloads"]) == {"csv", "batch_json"}
    csv_text = manager.get_download(completed["id"], "csv").read_text(encoding="utf-8")
    assert "good.wav,completed,zh,text for good" in csv_text
    assert "bad.wav,failed" in csv_text
    assert manager.get_download(completed["id"], "txt", result_index=0).read_text(encoding="utf-8") == "text for good\n"
    assert completed["results"][0]["output_urls"]["txt"].endswith("?result_index=0")
    manager.shutdown()


def test_export_write_failure_completes_job_with_a_visible_error(tmp_path, monkeypatch):
    paths, manager, _ = make_jobs(tmp_path)
    audio = paths.workspace / "uploads" / "clip.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"audio")

    def fail_exports(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(manager, "_write_job_exports", fail_exports)
    submitted = manager.submit([audio], model_id="base", output_formats="txt")
    completed = wait_for_completion(manager, submitted["id"])

    assert completed["status"] == "completed_with_errors"
    assert "result exports could not be written: disk full" in completed["error"]
    assert completed["results"][0]["status"] == "completed"
    assert manager.get_download(completed["id"], "txt", result_index=0).is_file()
    manager.shutdown()


def test_restart_marks_in_progress_records_interrupted_and_ignores_bad_ids(tmp_path):
    paths = ProjectPaths(tmp_path / "WhisperDock")
    paths.ensure()
    paths.write_json(
        paths.outputs / "history.json",
        {
            "version": 1,
            "jobs": [
                {"id": "a" * 32, "status": "running", "created_at": "2026-08-03T00:00:00Z"},
                {"id": "../../not-a-job", "status": "completed", "created_at": "2026-08-03T00:00:01Z"},
            ],
        },
    )

    manager = JobManager(paths, FakeModels())

    restored = manager.get_job("a" * 32)
    assert restored["status"] == "interrupted"
    assert restored["error"].startswith("The application restarted")
    with pytest.raises(WhisperDockError) as caught:
        manager.get_job("../../not-a-job")
    assert caught.value.code == "job_not_found"
    manager.shutdown()


def test_history_export_is_project_local_and_contains_completed_results(tmp_path):
    paths, manager, _ = make_jobs(tmp_path)
    audio = paths.workspace / "uploads" / "clip.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"audio")
    submitted = manager.submit([audio], model_id="base", output_formats="txt")
    wait_for_completion(manager, submitted["id"])

    export = manager.export_history_csv()

    assert export == paths.outputs / "history-export.csv"
    assert "job_id,created_at" in export.read_text(encoding="utf-8")
    manager.shutdown()
