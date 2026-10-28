"""GUI model representing an input video open in an editor session."""

import pathlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoadedVideo:
    """Represents a video file loaded into the timestamp editor session.

    Attributes:
        input_video: Resolved filesystem path used to play and extract audio from the video.
        input_channel: Zero-based index of the audio channel selected for waveform display and pipeline runs.
        parent_cache_dir: Parent directory containing `cache_dir` (passed as `-c/--cache-dir` to the pipeline).
        cache_dir: Per-video cache directory (`<stem>-Ballroom_Cache`) holding extracted audio and cached envelopes.
        csv_file: Video path written to and matched against the `File` column of the events CSV.
        channel_count: Number of channels in the video's first audio stream, as reported by `ffprobe`.
    """

    input_video: pathlib.Path
    input_channel: int
    parent_cache_dir: pathlib.Path
    cache_dir: pathlib.Path
    csv_file: pathlib.Path
    channel_count: int
