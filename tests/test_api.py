"""HTTP contract tests for the local FastAPI service."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.errors import WhisperDockError
from backend.main import create_app
from backend.model_manager import ENGINE_OPENAI, LoadedModel


class FakeModels:
    def get_model(self, model_id: str):
        if model_id != "base":
            raise WhisperDockError("Unknown model", code="model_not_found", status_code=404)
        return {"id": model_id, "loaded": True}

    def transcribe(self, model_id: str, audio_path: Path, options: dict):
        return {
            "text": f"recognized {audio_path.stem}",
            "language": options.get("language") or "zh",
            "segments": [{"id": 0, "start": 0, "end": 1, "text": "recognized"}],
            "model_id": model_id,
        }


def make_client(tmp_path: Path) -> TestClient:
    app = create_app(tmp_path / "WhisperDock")
    # Routes submit to the JobManager, so swapping this dependency preserves
    # the real upload, persistence, response, and result-export behavior.
    app.state.jobs.models = FakeModels()
    return TestClient(app)


def wait_for_job(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish within the test deadline")


def test_health_capabilities_and_whisper_schema_are_available(tmp_path):
    with make_client(tmp_path) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["workspace_path"].endswith("WhisperDock")

        capabilities = client.get("/api/capabilities")
        assert capabilities.status_code == 200
        assert capabilities.json()["realtime"]["mode"] == "segmented"
        assert capabilities.json()["storage"]["models"] == "models/"

        options = client.get("/api/whisper/options")
        assert options.status_code == 200
        keys = {field["key"] for field in options.json()["fields"]}
        assert {"task", "beam_size", "word_timestamps", "threads"} <= keys


def test_model_routes_publish_builtin_catalog_and_clean_errors(tmp_path):
    with make_client(tmp_path) as client:
        catalog = client.get("/api/models/catalog")
        assert catalog.status_code == 200
        assert any(item["id"] == "base" for item in catalog.json())

        invalid = client.post(
            "/api/models",
            json={"name": "Unsafe", "engine": "openai", "source_type": "url", "url": "http://example.test/model.pt"},
        )
        assert invalid.status_code == 400
        assert invalid.json()["error"]["code"] == "invalid_url"

        missing = client.get("/api/models/not-present")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "model_not_found"


def test_project_settings_persist_and_apply_to_uploaded_jobs(tmp_path):
    root = tmp_path / "WhisperDock"
    app = create_app(root)
    app.state.jobs.models = FakeModels()
    with TestClient(app) as client:
        changed = client.put(
            "/api/settings",
            json={
                "default_model": "base",
                "default_device": "cpu",
                "default_output_formats": ["json", "txt", "json"],
                "keep_uploads": False,
                "default_language": "nan",
                "default_prompt": "project default prompt",
            },
        )
        assert changed.status_code == 200
        settings = changed.json()
        assert settings["default_output_formats"] == ["json", "txt"]
        assert settings["default_device"] == "cpu"
        assert settings["keep_uploads"] is False

        # No model, language, prompt, or output-format fields: all resolve
        # from the project config and therefore survive browser/device moves.
        created = client.post(
            "/api/transcriptions",
            data={"options": "{}"},
            files={"file": ("defaults.wav", b"audio", "audio/wav")},
        )
        assert created.status_code == 202
        completed = wait_for_job(client, created.json()["id"])
        assert completed["status"] == "completed"
        assert completed["model_id"] == "base"
        assert completed["options"]["language"] == "nan"
        assert completed["options"]["initial_prompt"] == "project default prompt"
        assert set(completed["downloads"]) == {"json", "txt"}
        assert not (root / completed["files"][0]["path"]).exists()

        invalid = client.put("/api/settings", json={"default_output_formats": ["not-a-format"]})
        assert invalid.status_code == 400
        assert invalid.json()["error"]["code"] == "invalid_setting"

    # A new process instance reads exactly the project-local settings file.
    reloaded = create_app(root)
    assert reloaded.state.paths.read_json(root / "config" / "settings.json", {})["default_language"] == "nan"
    reloaded.state.jobs.shutdown()


def test_single_transcription_returns_pollable_job_and_browser_download_urls(tmp_path):
    with make_client(tmp_path) as client:
        created = client.post(
            "/api/transcriptions",
            data={"model_id": "base", "options": '{"language":"zh"}', "output_formats": "txt,json,srt"},
            files={"file": ("short.wav", b"fake wav bytes", "audio/wav")},
        )
        assert created.status_code == 202
        submitted = created.json()
        assert submitted["job_id"] == submitted["id"]

        completed = wait_for_job(client, submitted["id"])
        assert completed["status"] == "completed"
        assert completed["percentage"] == 100
        expected_text = completed["result"]["text"]
        assert expected_text.startswith("recognized audio-")
        assert completed["result"]["filename"].endswith("-short.wav")
        assert set(completed["result"]["outputs"]) == {"txt", "json", "srt"}

        download = client.get(completed["result"]["outputs"]["txt"])
        assert download.status_code == 200
        assert download.text == f"{expected_text}\n"


def test_batch_route_accepts_repeated_files_and_exposes_csv_export(tmp_path):
    with make_client(tmp_path) as client:
        created = client.post(
            "/api/transcriptions/batch",
            data={"model_id": "base", "options": "{}", "output_formats": "txt"},
            files=[
                ("files", ("one.wav", b"one", "audio/wav")),
                ("files", ("two.wav", b"two", "audio/wav")),
            ],
        )
        assert created.status_code == 202

        completed = wait_for_job(client, created.json()["id"])
        assert completed["status"] == "completed"
        assert completed["kind"] == "batch"
        assert completed["percentage"] == 100
        assert len(completed["results"]) == 2
        assert "csv" in completed["output_urls"]
        assert client.get(completed["output_urls"]["csv"]).status_code == 200
        item_output = completed["results"][0]["output_urls"]["txt"]
        assert "result_index=0" in item_output
        assert client.get(item_output).status_code == 200


def test_upload_and_workspace_routes_reject_path_traversal(tmp_path):
    with make_client(tmp_path) as client:
        uploaded = client.post(
            "/api/uploads",
            files={"file": ("clip.wav", b"audio", "audio/wav")},
        )
        assert uploaded.status_code == 201
        assert uploaded.json()["path"].startswith("workspace/uploads/")

        escaped = client.post(
            "/api/transcriptions/workspace",
            json={"model_id": "base", "files": ["../../outside.wav"]},
        )
        assert escaped.status_code == 400
        assert escaped.json()["error"]["code"] == "unsafe_audio_path"


def test_segmented_realtime_websocket_returns_a_partial_result(tmp_path):
    class FakeWhisper:
        def transcribe(self, _: str, **__: object) -> dict[str, object]:
            return {
                "text": "partial text",
                "language": "zh",
                "segments": [{"id": 0, "start": 0, "end": 1, "text": "partial text"}],
            }

    root = tmp_path / "WhisperDock"
    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"placeholder")
    app = create_app(root)
    record = app.state.models.create_custom_model(
        {"name": "Realtime fake", "engine": "openai", "source_type": "local", "local_path": str(checkpoint)}
    )
    app.state.models._loaded[record["id"]] = LoadedModel(record["id"], ENGINE_OPENAI, "cpu", FakeWhisper())

    with TestClient(app) as client:
        with client.websocket_connect("/api/realtime") as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_json({"type": "configure", "model_id": record["id"], "options": {}, "format": "wav"})
            assert socket.receive_json()["type"] == "configured"
            socket.send_bytes(b"fake audio chunk")
            partial = socket.receive_json()
            assert partial["type"] == "partial"
            assert partial["text"] == "partial text"
