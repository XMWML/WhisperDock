"""Whisper option validation and UI metadata.

OpenAI Whisper exposes a small set of transcription controls plus the decode
controls in ``DecodingOptions``.  Keeping this list here lets the Web UI render
every supported input without passing arbitrary Python kwargs to a model.
"""

from __future__ import annotations

from typing import Any

from .errors import WhisperDockError


# The fields below cover whisper.transcribe() and whisper.decoding.DecodingOptions.
WHISPER_OPTION_METADATA: list[dict[str, Any]] = [
    {"key": "task", "label": "Task", "type": "select", "default": "transcribe", "choices": ["transcribe", "translate"], "group": "General"},
    {"key": "language", "label": "Language code", "type": "text", "default": None, "placeholder": "auto, en, zh, yue...", "group": "General"},
    {"key": "verbose", "label": "Verbose progress", "type": "select", "default": False, "choices": [False, True, "live"], "group": "General"},
    {"key": "temperature", "label": "Temperature / fallback temperatures", "type": "temperature", "default": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "group": "Decoding"},
    {"key": "best_of", "label": "Best of", "type": "integer", "default": None, "min": 1, "group": "Decoding"},
    {"key": "beam_size", "label": "Beam size", "type": "integer", "default": None, "min": 1, "group": "Decoding"},
    {"key": "patience", "label": "Beam patience", "type": "number", "default": None, "min": 0, "group": "Decoding"},
    {"key": "length_penalty", "label": "Length penalty", "type": "number", "default": None, "group": "Decoding"},
    {"key": "sample_len", "label": "Sample length", "type": "integer", "default": None, "min": 1, "group": "Decoding"},
    {"key": "prompt", "label": "Prompt tokens", "type": "text", "default": None, "group": "Decoding", "advanced": True},
    {"key": "prefix", "label": "Prefix", "type": "text", "default": None, "group": "Decoding", "advanced": True},
    {"key": "suppress_tokens", "label": "Suppress token IDs", "type": "text", "default": "-1", "group": "Decoding", "advanced": True},
    {"key": "suppress_blank", "label": "Suppress blank", "type": "boolean", "default": True, "group": "Decoding", "advanced": True},
    {"key": "without_timestamps", "label": "Without timestamps", "type": "boolean", "default": False, "group": "Decoding"},
    {"key": "max_initial_timestamp", "label": "Max initial timestamp", "type": "number", "default": 1.0, "min": 0, "group": "Decoding", "advanced": True},
    {"key": "compression_ratio_threshold", "label": "Compression ratio threshold", "type": "number", "default": 2.4, "group": "Fallback"},
    {"key": "logprob_threshold", "label": "Log probability threshold", "type": "number", "default": -1.0, "group": "Fallback"},
    {"key": "no_speech_threshold", "label": "No-speech threshold", "type": "number", "default": 0.6, "min": 0, "max": 1, "group": "Fallback"},
    {"key": "condition_on_previous_text", "label": "Condition on previous text", "type": "boolean", "default": True, "group": "Context"},
    {"key": "initial_prompt", "label": "Initial prompt", "type": "text", "default": None, "group": "Context"},
    {"key": "carry_initial_prompt", "label": "Carry initial prompt", "type": "boolean", "default": False, "group": "Context", "advanced": True},
    {"key": "word_timestamps", "label": "Word timestamps", "type": "boolean", "default": False, "group": "Timestamps"},
    {"key": "prepend_punctuations", "label": "Prepend punctuations", "type": "text", "default": "\"'\u201c\u00bf([{-", "group": "Timestamps", "advanced": True},
    {"key": "append_punctuations", "label": "Append punctuations", "type": "text", "default": "\"'.\u3002,\uff0c!\uff01?\uff1f:\uff1a\u201d)]}\u3001", "group": "Timestamps", "advanced": True},
    {"key": "clip_timestamps", "label": "Clip timestamps", "type": "text", "default": "0", "placeholder": "0 or 0,30,60", "group": "Timestamps", "advanced": True},
    {"key": "hallucination_silence_threshold", "label": "Hallucination silence threshold", "type": "number", "default": None, "min": 0, "group": "Fallback", "advanced": True},
    {"key": "fp16", "label": "Use FP16", "type": "boolean", "default": True, "group": "Runtime"},
    {"key": "threads", "label": "CPU threads", "type": "integer", "default": 0, "min": 0, "group": "Runtime", "advanced": True},
]

_DEFAULTS = {entry["key"]: entry["default"] for entry in WHISPER_OPTION_METADATA}
_ALLOWED = set(_DEFAULTS)
_BOOLS = {entry["key"] for entry in WHISPER_OPTION_METADATA if entry["type"] == "boolean"}
_INTS = {entry["key"] for entry in WHISPER_OPTION_METADATA if entry["type"] == "integer"}
_NUMBERS = {entry["key"] for entry in WHISPER_OPTION_METADATA if entry["type"] == "number"}
_LIMITS = {
    entry["key"]: (entry.get("min"), entry.get("max"))
    for entry in WHISPER_OPTION_METADATA
    if "min" in entry or "max" in entry
}


def option_defaults() -> dict[str, Any]:
    """Return a detached copy suitable for JSON responses."""
    return {key: value.copy() if isinstance(value, list) else value for key, value in _DEFAULTS.items()}


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise WhisperDockError(f"{name} must be true or false.", code="invalid_option")


def _optional_number(value: Any, name: str, *, integer: bool = False) -> int | float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise WhisperDockError(f"{name} must be numeric.", code="invalid_option")
    try:
        parsed = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise WhisperDockError(f"{name} must be numeric.", code="invalid_option") from exc
    minimum, maximum = _LIMITS.get(name, (0 if integer else None, None))
    if minimum is not None and parsed < minimum:
        raise WhisperDockError(f"{name} must be at least {minimum}.", code="invalid_option")
    if maximum is not None and parsed > maximum:
        raise WhisperDockError(f"{name} must be at most {maximum}.", code="invalid_option")
    return parsed


def normalize_whisper_options(raw_options: dict[str, Any] | None, *, device: str = "auto") -> dict[str, Any]:
    """Validate options and convert form-friendly values for ``model.transcribe``."""
    raw_options = raw_options or {}
    if not isinstance(raw_options, dict):
        raise WhisperDockError("Whisper options must be a JSON object.", code="invalid_options")
    unknown = set(raw_options) - _ALLOWED
    if unknown:
        raise WhisperDockError(f"Unsupported Whisper option(s): {', '.join(sorted(unknown))}.", code="invalid_option")

    options = option_defaults()
    options.update(raw_options)
    if options["task"] not in {"transcribe", "translate"}:
        raise WhisperDockError("task must be 'transcribe' or 'translate'.", code="invalid_option")
    if options["verbose"] not in {True, False, "live"}:
        raise WhisperDockError("verbose must be true, false, or 'live'.", code="invalid_option")
    if options["language"] in {"", "auto"}:
        options["language"] = None

    for name in _BOOLS:
        options[name] = _as_bool(options[name], name)
    for name in _INTS:
        options[name] = _optional_number(options[name], name, integer=True)
    for name in _NUMBERS:
        options[name] = _optional_number(options[name], name)

    temperature = options["temperature"]
    if isinstance(temperature, str):
        try:
            temperature = [float(item.strip()) for item in temperature.split(",") if item.strip()]
        except ValueError as exc:
            raise WhisperDockError("temperature must be a number or comma-separated numbers.", code="invalid_option") from exc
    elif isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        temperature = float(temperature)
    elif isinstance(temperature, (list, tuple)):
        try:
            temperature = [float(item) for item in temperature]
        except (TypeError, ValueError) as exc:
            raise WhisperDockError("temperature values must be numeric.", code="invalid_option") from exc
    else:
        raise WhisperDockError("temperature must be a number or list of numbers.", code="invalid_option")
    options["temperature"] = temperature

    clip_timestamps = options["clip_timestamps"]
    if isinstance(clip_timestamps, str):
        clip_timestamps = clip_timestamps.strip() or "0"
    elif isinstance(clip_timestamps, (list, tuple)):
        try:
            clip_timestamps = [float(item) for item in clip_timestamps]
        except (TypeError, ValueError) as exc:
            raise WhisperDockError("clip_timestamps must contain numeric timestamps.", code="invalid_option") from exc
    else:
        raise WhisperDockError("clip_timestamps must be a string or list of numbers.", code="invalid_option")
    options["clip_timestamps"] = clip_timestamps

    if options["beam_size"] is not None and options["best_of"] is not None:
        # This is a Whisper limitation: best_of belongs to sampling, beam_size to beam search.
        options["best_of"] = None
    if device in {"cpu", "mps"}:
        # The official decoder's fp16 path is CUDA-oriented. MPS still
        # accelerates the model itself with a reliable fp32 decoder.
        options["fp16"] = False
    if options["threads"] in {None, 0}:
        options.pop("threads", None)
    # ``language=None`` is meaningful to Whisper: it explicitly asks the
    # decoder to auto-detect the language. Keep it instead of dropping the
    # field while still omitting unrelated unset optional controls.
    return {
        key: value
        for key, value in options.items()
        if value is not None or key == "language"
    }
