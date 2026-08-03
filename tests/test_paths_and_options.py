"""Fast, dependency-free coverage for the portable backend primitives."""

from __future__ import annotations

import json

import pytest

from backend.errors import WhisperDockError
from backend.options import WHISPER_OPTION_METADATA, normalize_whisper_options, option_defaults
from backend.paths import ProjectPaths


def test_project_paths_are_entirely_rooted_in_portable_project(tmp_path):
    paths = ProjectPaths(tmp_path / "WhisperDock")
    paths.ensure()

    for directory in (
        paths.config,
        paths.models,
        paths.cache,
        paths.workspace,
        paths.outputs,
        paths.logs,
        paths.models / "openai-whisper",
        paths.models / "custom",
        paths.models / "huggingface",
        paths.workspace / "uploads",
        paths.workspace / "realtime",
        paths.outputs / "jobs",
    ):
        assert directory.is_dir()
        assert directory.is_relative_to(paths.root)


@pytest.mark.parametrize("relative_path", ["../outside", "/tmp/outside", "uploads/../../outside"])
def test_user_paths_cannot_escape_project_root(tmp_path, relative_path):
    paths = ProjectPaths(tmp_path / "WhisperDock")
    paths.ensure()

    with pytest.raises(WhisperDockError, match="stay inside"):
        paths.from_relative(relative_path)


def test_upload_names_are_sanitized_and_unique(tmp_path):
    paths = ProjectPaths(tmp_path / "WhisperDock")
    paths.ensure()

    first = paths.upload_destination("../../audio files/recording?.wav")
    second = paths.upload_destination("../../audio files/recording?.wav")

    assert first.parent == paths.workspace / "uploads"
    assert first.name.endswith("recording-.wav")
    assert first != second


def test_json_store_round_trips_and_recovers_from_invalid_json(tmp_path):
    paths = ProjectPaths(tmp_path / "WhisperDock")
    settings = paths.config / "settings.json"
    value = {"model": "base", "prompt": "潮汕话"}

    paths.write_json(settings, value)
    assert json.loads(settings.read_text(encoding="utf-8")) == value
    assert paths.read_json(settings, {}) == value

    settings.write_text("not json", encoding="utf-8")
    assert paths.read_json(settings, {"fallback": True}) == {"fallback": True}


def test_job_directories_require_canonical_uuid_hex(tmp_path):
    paths = ProjectPaths(tmp_path / "WhisperDock")
    paths.ensure()

    directory = paths.job_directory("a" * 32)
    assert directory.is_dir()

    for invalid in ("../" + "a" * 32, "A" * 32, "a" * 31, "a" * 33):
        with pytest.raises(WhisperDockError, match="Invalid job identifier"):
            paths.job_directory(invalid)


def test_capability_metadata_has_unique_keys_and_detached_defaults():
    keys = [entry["key"] for entry in WHISPER_OPTION_METADATA]
    assert len(keys) == len(set(keys))

    first = option_defaults()
    second = option_defaults()
    first["temperature"].append(9.9)
    assert 9.9 not in second["temperature"]


def test_options_are_normalized_for_forms_and_cpu_execution():
    options = normalize_whisper_options(
        {
            "language": "auto",
            "temperature": "0, 0.25",
            "clip_timestamps": [0, "30"],
            "threads": "4",
            "beam_size": "3",
            "fp16": True,
        },
        device="cpu",
    )

    assert options["language"] is None
    assert options["temperature"] == [0.0, 0.25]
    assert options["clip_timestamps"] == [0.0, 30.0]
    assert options["beam_size"] == 3
    assert "best_of" not in options
    assert options["threads"] == 4
    assert options["fp16"] is False


@pytest.mark.parametrize(
    "raw_options",
    [
        {"unknown_knob": True},
        {"fp16": "perhaps"},
        {"temperature": "warm"},
        {"clip_timestamps": {"not": "a list"}},
    ],
)
def test_invalid_options_are_rejected_before_whisper_is_called(raw_options):
    with pytest.raises(WhisperDockError):
        normalize_whisper_options(raw_options)


@pytest.mark.parametrize(
    "raw_options",
    [
        {"best_of": 0},
        {"beam_size": 0},
        {"sample_len": 0},
        {"patience": -0.5},
        {"max_initial_timestamp": -0.1},
        {"no_speech_threshold": 1.1},
        {"hallucination_silence_threshold": -0.1},
    ],
)
def test_option_values_honor_their_published_ui_bounds(raw_options):
    """The API must reject values its own GUI says are invalid."""
    with pytest.raises(WhisperDockError, match="must be|cannot be|at least|at most"):
        normalize_whisper_options(raw_options)
