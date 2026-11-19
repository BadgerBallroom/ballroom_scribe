"""Child process of the timestamp editor: transcribes one video and prints its speech for the editor.

This is not a command-line tool. `MainWindow` starts it with `python -m ballroom_scribe.gui.run_transcription_worker`
and reads its standard output. To transcribe videos from a terminal, run `find_timestamps.py`.
"""

import functools
import os
import sys
from collections.abc import Sequence
from typing import TextIO

import ballroom_scribe
import ballroom_scribe.cli.arguments
import ballroom_scribe.gui.utils
import ballroom_scribe.models

_STANDARD_OUTPUT = 1
_STANDARD_ERROR = 2


def _claim_standard_output() -> TextIO:
    """Returns a stream to the process's standard output and redirects everything else that writes there.

    The editor reads standard output as the transcript, so nothing else may be written to it. The pipeline prints its
    progress there, and so do the libraries and the child processes that it runs, such as `ffmpeg`. All of that goes
    to standard error instead: the file descriptor itself is redirected, not only `sys.stdout`.
    """
    transcript_stream = os.fdopen(os.dup(_STANDARD_OUTPUT), "w", encoding="utf-8", newline="\n")
    os.dup2(_STANDARD_ERROR, _STANDARD_OUTPUT)
    sys.stdout = sys.stderr
    return transcript_stream


def _send(
    transcript_stream: TextIO,
    video_input: ballroom_scribe.cli.arguments.InputCacheArgs,
    transcript_chunk: ballroom_scribe.models.TranscriptChunk,
) -> None:
    """Writes one transcript chunk to the editor as a line of text, at once."""
    del video_input  # The worker transcribes one video.
    transcript_stream.write(ballroom_scribe.gui.utils.encode_transcript_chunk(transcript_chunk) + "\n")
    transcript_stream.flush()


def run_transcription_worker(argv: Sequence[str] | None = None) -> int:
    """Runs the pipeline's stages through speech transcription on one video and prints each speech chunk.

    Every line of standard output is one speech chunk, encoded by `encode_transcript_chunk`, written as soon as that
    stretch of speech has been transcribed or read from the cache. Progress and errors go to standard error. The exit
    status `0` tells the editor that the transcript is complete.

    The arguments are those of `find_timestamps.py` without the output CSV and the language model, with the same
    defaults, so the worker reads and writes the same cache files as the pipeline.

    Args:
        argv: The command-line arguments, or `None` for those of the process.

    Returns:
        The exit status of the process.
    """
    transcript_stream = _claim_standard_output()
    input_cache_args, audio_classification_args, music_valley_detection_args, speech_transcription_args = (
        ballroom_scribe.cli.arguments.parse_args(
            (
                ballroom_scribe.cli.arguments.InputCacheGroup(),
                ballroom_scribe.cli.arguments.AudioClassificationGroup(),
                ballroom_scribe.cli.arguments.MusicValleyDetectionGroup(),
                ballroom_scribe.cli.arguments.SpeechTranscriptionGroup(),
            ),
            argv,
            description="Transcribes the speech of one video for the timestamp editor. Not meant to be run by hand.",
        )
    )
    pipeline = ballroom_scribe.TranscriptionPipeline(
        audio_classification_args=audio_classification_args,
        music_valley_detection_args=music_valley_detection_args,
        speech_transcription_args=speech_transcription_args,
    )
    pipeline.run([input_cache_args], on_speech_transcribed=functools.partial(_send, transcript_stream))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_transcription_worker())
