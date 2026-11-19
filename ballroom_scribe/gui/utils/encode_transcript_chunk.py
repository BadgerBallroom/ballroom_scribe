"""Encodes a transcript chunk as one line of text for the transcription worker's standard output."""

import dataclasses
import json

import ballroom_scribe.models


def encode_transcript_chunk(transcript_chunk: ballroom_scribe.models.TranscriptChunk) -> str:
    """Returns `transcript_chunk` as a single line of ASCII JSON, without a trailing newline.

    `decode_transcript_chunk` turns the line back into an equal transcript chunk.
    """
    return json.dumps(dataclasses.asdict(transcript_chunk))
