"""Decodes a line of the transcription worker's standard output into a transcript chunk."""

import json
from typing import Any

import ballroom_scribe.models

_NUMBER = (int, float)


def _require(value: Any, expected_type: type | tuple[type, ...], name: str) -> Any:
    """Returns `value` if it is an instance of `expected_type` and not a `bool`, else raises a `TypeError`."""
    if isinstance(value, bool) or not isinstance(value, expected_type):
        raise TypeError(f"{name} has the wrong type: {value!r}.")
    return value


def _decode_word(fields: Any) -> ballroom_scribe.models.Word:
    fields = _require(fields, dict, "A word")
    return ballroom_scribe.models.Word(
        word=_require(fields["word"], str, "word"),
        start=float(_require(fields["start"], _NUMBER, "start")),
        end=float(_require(fields["end"], _NUMBER, "end")),
        probability=float(_require(fields["probability"], _NUMBER, "probability")),
    )


def decode_transcript_chunk(line: str) -> ballroom_scribe.models.TranscriptChunk:
    """Returns the transcript chunk that `encode_transcript_chunk` encoded as `line`.

    Args:
        line: One line of the transcription worker's standard output, with or without its trailing newline.

    Raises:
        ValueError: If `line` is not an encoded transcript chunk, e.g. it is not JSON, a field is missing, or a field
            has the wrong type.
    """
    try:
        fields = _require(json.loads(line), dict, "A transcript chunk")
        return ballroom_scribe.models.TranscriptChunk(
            start_samples=_require(fields["start_samples"], int, "start_samples"),
            sample_rate=_require(fields["sample_rate"], int, "sample_rate"),
            duration_samples=_require(fields["duration_samples"], int, "duration_samples"),
            channel=_require(fields["channel"], int, "channel"),
            words=tuple(_decode_word(word) for word in _require(fields["words"], list, "words")),
        )
    except (KeyError, TypeError, ValueError) as err:  # `json.JSONDecodeError` is a `ValueError`.
        raise ValueError(f"Not an encoded transcript chunk: {line!r}") from err
