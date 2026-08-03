"""Model management tests that do not download or load a real Whisper model."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.errors import WhisperDockError
from backend.model_manager import ENGINE_OPENAI, LoadedModel, ModelManager
from backend.paths import ProjectPaths


def make_manager(tmp_path: Path) -> ModelManager:
    return ModelManager(ProjectPaths(tmp_path / "WhisperDock"))


def url_payload(**overrides):
    payload = {
        "name": "Teochew checkpoint",
        "engine": "openai-whisper",
        "source_type": "url",
        "url": "https://models.example.test/teochew.pt",
    }
    payload.update(overrides)
    return payload


def test_custom_url_model_is_registered_but_never_downloaded_implicitly(tmp_path):
    manager = make_manager(tmp_path)

    record = manager.create_custom_model(url_payload())

    assert record["id"] == "teochew-checkpoint"
    assert record["engine"] == ENGINE_OPENAI
    assert record["state"] == "not_downloaded"
    assert record["path"] == "models/custom/teochew-checkpoint/teochew.pt"
    assert not manager.paths.from_relative(record["path"]).exists()
    assert manager.get_model(record["id"])["state"] == "not_downloaded"


def test_huggingface_transformers_model_keeps_repo_id_and_project_path(tmp_path):
    manager = make_manager(tmp_path)

    record = manager.create_custom_model(
        {
            "name": "Teochew fine tune",
            "id": "panlr-whisper-finetune-teochew",
            "engine": "transformers",
            "source_type": "huggingface",
            "hf_repo": "panlr/whisper-finetune-teochew",
        }
    )

    assert record["id"] == "panlr-whisper-finetune-teochew"
    assert record["engine"] == "transformers"
    assert record["source"] == {
        "kind": "huggingface",
        "repo_id": "panlr/whisper-finetune-teochew",
        "revision": "main",
    }
    assert record["path"] == "models/huggingface/panlr-whisper-finetune-teochew"
    assert record["state"] == "not_downloaded"


@pytest.mark.parametrize(
    "payload, error_code",
    [
        (url_payload(url="http://models.example.test/model.pt"), "invalid_url"),
        (url_payload(url="https://models.example.test/model.safetensors"), "invalid_checkpoint"),
        (url_payload(engine="transformers"), "invalid_source"),
        (url_payload(id="../../escape"), "invalid_model_id"),
        (url_payload(source="not-an-object"), "invalid_source"),
    ],
)
def test_invalid_remote_model_registrations_are_reported_as_clean_api_errors(tmp_path, payload, error_code):
    manager = make_manager(tmp_path)

    with pytest.raises(WhisperDockError) as caught:
        manager.create_custom_model(payload)

    assert caught.value.code == error_code


def test_local_checkpoint_is_copied_into_project_so_it_survives_source_removal(tmp_path):
    manager = make_manager(tmp_path)
    external_checkpoint = tmp_path / "external" / "model.pt"
    external_checkpoint.parent.mkdir()
    external_checkpoint.write_bytes(b"model-data")

    record = manager.create_custom_model(
        {
            "name": "Imported model",
            "engine": "openai",
            "source_type": "local",
            "local_path": str(external_checkpoint),
        }
    )
    destination = manager.paths.from_relative(record["path"])
    external_checkpoint.unlink()

    assert destination.read_bytes() == b"model-data"
    assert record["state"] == "downloaded"
    assert record["source"] == {"kind": "local", "imported": True}


def test_load_and_unload_change_only_the_in_memory_state(tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    checkpoint = tmp_path / "external.pt"
    checkpoint.write_bytes(b"model-data")
    record = manager.create_custom_model(
        {"name": "Memory model", "engine": "openai", "source_type": "local", "local_path": str(checkpoint)}
    )
    loaded_model = object()
    monkeypatch.setattr(manager, "_resolve_device", lambda requested: "cpu")
    monkeypatch.setattr(
        manager,
        "_load_openai",
        lambda model_id, location, device: LoadedModel(model_id, ENGINE_OPENAI, device, loaded_model),
    )

    loaded = manager.load(record["id"])
    assert loaded["state"] == "loaded"
    assert loaded["device"] == "cpu"

    unloaded = manager.unload(record["id"])
    assert unloaded["state"] == "downloaded"
    assert unloaded["device"] is None


def test_delete_removes_registered_model_and_project_local_copy(tmp_path):
    manager = make_manager(tmp_path)
    checkpoint = tmp_path / "external.pt"
    checkpoint.write_bytes(b"model-data")
    record = manager.create_custom_model(
        {"name": "Disposable model", "engine": "openai", "source_type": "local", "local_path": str(checkpoint)}
    )
    destination = manager.paths.from_relative(record["path"])

    manager.delete(record["id"])

    assert not destination.exists()
    with pytest.raises(WhisperDockError) as caught:
        manager.get_model(record["id"])
    assert caught.value.code == "model_not_found"


class _FakeWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path: str, **options):
        self.calls.append((audio_path, options))
        return {"text": "  recognized text  ", "language": "zh", "segments": [{"id": 0, "text": "x"}]}


def test_transcription_requires_loaded_model_and_normalizes_openai_result(tmp_path):
    manager = make_manager(tmp_path)
    checkpoint = tmp_path / "external.pt"
    checkpoint.write_bytes(b"model-data")
    record = manager.create_custom_model(
        {"name": "Inference model", "engine": "openai", "source_type": "local", "local_path": str(checkpoint)}
    )
    audio = manager.paths.workspace / "uploads" / "sample.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"not real audio - fake model ignores it")
    fake = _FakeWhisper()

    with pytest.raises(WhisperDockError) as caught:
        manager.transcribe(record["id"], audio)
    assert caught.value.code == "model_not_loaded"

    manager._loaded[record["id"]] = LoadedModel(record["id"], ENGINE_OPENAI, "cpu", fake)
    result = manager.transcribe(record["id"], audio, {"language": "auto", "temperature": "0"})

    assert result["text"] == "recognized text"
    assert result["language"] == "zh"
    assert result["engine"] == ENGINE_OPENAI
    assert fake.calls[0][1]["fp16"] is False
    assert fake.calls[0][1]["language"] is None
    assert fake.calls[0][1]["temperature"] == [0.0]
