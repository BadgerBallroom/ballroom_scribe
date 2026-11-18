import pathlib
import unittest
from unittest.mock import ANY, MagicMock, call, patch

import ollama

import ballroom_scribe
import ballroom_scribe.cli.arguments
import ballroom_scribe.models
import ballroom_scribe.utils
import tests.helpers.transcript_chunk

_INPUT_VIDEO_A = pathlib.Path("fake_input_a.mp4")
_INPUT_CHANNEL_A = 1
_PARENT_CACHE_DIR_A = pathlib.Path("fake_cache_a")
_VIDEO_CACHE_DIR_A = pathlib.Path("fake_video_cache_a")

_INPUT_VIDEO_B = pathlib.Path("fake_input_b.mp4")
_INPUT_CHANNEL_B = 0
_PARENT_CACHE_DIR_B = pathlib.Path("fake_cache_b")
_VIDEO_CACHE_DIR_B = pathlib.Path("fake_video_cache_b")

_OUTPUT_CSV = pathlib.Path("fake_output.csv")


def _make_inputs() -> list[ballroom_scribe.cli.arguments.InputCacheArgs]:
    """Helper to create the inputs of videos A and B."""
    return [
        ballroom_scribe.cli.arguments.InputCacheArgs(
            input_video=_INPUT_VIDEO_A,
            input_channel=_INPUT_CHANNEL_A,
            parent_cache_dir=_PARENT_CACHE_DIR_A,
        ),
        ballroom_scribe.cli.arguments.InputCacheArgs(
            input_video=_INPUT_VIDEO_B,
            input_channel=_INPUT_CHANNEL_B,
            parent_cache_dir=_PARENT_CACHE_DIR_B,
        ),
    ]


def _make_audio_classification_args() -> ballroom_scribe.cli.arguments.AudioClassificationArgs:
    """Helper to create standard test arguments for audio loading and classification."""
    return ballroom_scribe.cli.arguments.AudioClassificationArgs(
        pann_device="cuda",
        pann_chunk_minutes=10.0,
        pann_chunk_overlap_seconds=2.0,
        pann_thresholds={
            ballroom_scribe.models.AudioChunk.Classification.SPEECH: 0.18,
            ballroom_scribe.models.AudioChunk.Classification.MUSIC: 0.45,
        },
        pann_min_chunk_seconds=1.0,
    )


def _make_music_valley_detection_args() -> ballroom_scribe.cli.arguments.MusicValleyDetectionArgs:
    """Helper to create standard test arguments for music valley detection."""
    return ballroom_scribe.cli.arguments.MusicValleyDetectionArgs(
        min_dance_seconds=35.0,
        max_song_dip_seconds=4.5,
        max_fade_speech_seconds=12.0,
        max_dip_speech_seconds=2.0,
    )


def _make_speech_transcription_args() -> ballroom_scribe.cli.arguments.SpeechTranscriptionArgs:
    """Helper to create standard test arguments for speech transcription."""
    return ballroom_scribe.cli.arguments.SpeechTranscriptionArgs(
        whisper_device="cpu",
        whisper_model="large-v3",
        whisper_compute_type="float16",
        whisper_cpu_threads=4,
        whisper_beam_size=5,
        whisper_init_kwargs={},
    )


def _make_pipeline() -> ballroom_scribe.TimestampPipeline:
    """Helper to create a `TimestampPipeline` with standard test arguments."""
    return ballroom_scribe.TimestampPipeline(
        audio_classification_args=_make_audio_classification_args(),
        music_valley_detection_args=_make_music_valley_detection_args(),
        speech_transcription_args=_make_speech_transcription_args(),
        language_processing_args=ballroom_scribe.cli.arguments.LanguageProcessingArgs(
            ollama_model="qwen2.5:7b",
            ollama_options=ollama.Options(temperature=0.0),
        ),
    )


class TestInit(unittest.TestCase):
    """Unit tests for the `TimestampPipeline.__init__` method."""

    def setUp(self):
        self.mock_transcription_pipeline_cls = self.enterContext(
            patch.object(ballroom_scribe, "TranscriptionPipeline", autospec=True),
        )
        self.mock_detector_cls = self.enterContext(
            patch.object(ballroom_scribe, "EventDetector", autospec=True),
        )
        self.mock_recorder_cls = self.enterContext(
            patch.object(ballroom_scribe, "EventRecorder", autospec=True),
        )

    def test_routes_arguments_to_each_part(self):
        _make_pipeline()

        self.mock_transcription_pipeline_cls.assert_called_once_with(
            audio_classification_args=_make_audio_classification_args(),
            music_valley_detection_args=_make_music_valley_detection_args(),
            speech_transcription_args=_make_speech_transcription_args(),
        )
        self.mock_detector_cls.assert_called_once_with(
            ollama_model="qwen2.5:7b",
            ollama_options=ollama.Options(temperature=0.0),
        )
        self.mock_recorder_cls.assert_called_once_with()


class TestRun(unittest.TestCase):
    """Unit tests for the `TimestampPipeline.run` method."""

    def setUp(self):
        self.mock_cache_dir_for_video = self.enterContext(
            patch.object(ballroom_scribe.utils, "cache_dir_for_video", autospec=True),
        )
        cache_dirs = {_INPUT_VIDEO_A: _VIDEO_CACHE_DIR_A, _INPUT_VIDEO_B: _VIDEO_CACHE_DIR_B}
        self.mock_cache_dir_for_video.side_effect = lambda parent_cache_dir, input_video: cache_dirs[input_video]

        self.mock_resolve = self.enterContext(patch.object(pathlib.Path, "resolve", autospec=True))
        self.mock_resolve.side_effect = lambda path: path if path.is_absolute() else pathlib.Path("/resolved") / path

        self.mock_transcription_pipeline_cls = self.enterContext(
            patch.object(ballroom_scribe, "TranscriptionPipeline", autospec=True),
        )
        self.mock_transcription_pipeline = self.mock_transcription_pipeline_cls.return_value

        self.mock_detector_cls = self.enterContext(
            patch.object(ballroom_scribe, "EventDetector", autospec=True),
        )
        self.mock_detector = self.mock_detector_cls.return_value

        self.mock_recorder_cls = self.enterContext(
            patch.object(ballroom_scribe, "EventRecorder", autospec=True),
        )
        self.mock_recorder = self.mock_recorder_cls.return_value
        self.recorded_events: list[tuple[pathlib.Path, ballroom_scribe.models.Event]] = []
        self.mock_recorder.record.side_effect = lambda events, target: self.recorded_events.extend(events)

    def test_run(self):
        transcript_chunk_maker = tests.helpers.transcript_chunk.Maker()
        spoken_transcript_chunk_a = transcript_chunk_maker.make(["hello", "world"])
        silent_transcript_chunk_a = transcript_chunk_maker.make([])
        transcript_chunk_b = transcript_chunk_maker.make(["welcome"])
        self.mock_transcription_pipeline.run.return_value = [
            [spoken_transcript_chunk_a, silent_transcript_chunk_a],
            [transcript_chunk_b],
        ]

        event_a = MagicMock(spec=ballroom_scribe.models.Event)
        event_b = MagicMock(spec=ballroom_scribe.models.Event)
        self.mock_detector.detect.side_effect = [iter([event_a]), iter([event_b])]

        manager = MagicMock()
        manager.attach_mock(self.mock_transcription_pipeline.run, "transcribe")
        manager.attach_mock(self.mock_recorder.record, "record")
        manager.attach_mock(self.mock_detector.detect, "detect")

        pipeline = _make_pipeline()
        pipeline.run(inputs=_make_inputs(), output_csv=_OUTPUT_CSV)

        # Each video's events are cached in the video's own cache directory.
        self.assertEqual(
            self.mock_cache_dir_for_video.call_args_list,
            [
                call(parent_cache_dir=_PARENT_CACHE_DIR_A, input_video=_INPUT_VIDEO_A),
                call(parent_cache_dir=_PARENT_CACHE_DIR_B, input_video=_INPUT_VIDEO_B),
            ],
        )
        self.assertEqual(
            self.recorded_events,
            [
                (pathlib.Path("/resolved") / _INPUT_VIDEO_A, event_a),
                (pathlib.Path("/resolved") / _INPUT_VIDEO_B, event_b),
            ],
        )

        # Every video is transcribed before any events are detected, and events stream lazily inside `record`.
        self.assertEqual(
            manager.mock_calls,
            [
                call.transcribe(_make_inputs()),
                call.record(events=ANY, target=_OUTPUT_CSV),
                # Every music valley's transcript chunk is passed on, even one without words, so that each produces an
                # event.
                call.detect(
                    chunks_of_words=[spoken_transcript_chunk_a, silent_transcript_chunk_a],
                    cache_dir=_VIDEO_CACHE_DIR_A,
                ),
                call.detect(
                    chunks_of_words=[transcript_chunk_b],
                    cache_dir=_VIDEO_CACHE_DIR_B,
                ),
            ],
        )

    def test_run_reads_inputs_that_can_be_iterated_only_once(self):
        transcript_chunk_maker = tests.helpers.transcript_chunk.Maker()
        transcript_chunk_a = transcript_chunk_maker.make(["hello"])
        transcript_chunk_b = transcript_chunk_maker.make(["welcome"])
        self.mock_transcription_pipeline.run.return_value = [[transcript_chunk_a], [transcript_chunk_b]]
        self.mock_detector.detect.side_effect = [iter([]), iter([])]

        pipeline = _make_pipeline()
        pipeline.run(inputs=iter(_make_inputs()), output_csv=_OUTPUT_CSV)

        self.mock_transcription_pipeline.run.assert_called_once_with(_make_inputs())
        self.assertEqual(
            self.mock_detector.detect.call_args_list,
            [
                call(chunks_of_words=[transcript_chunk_a], cache_dir=_VIDEO_CACHE_DIR_A),
                call(chunks_of_words=[transcript_chunk_b], cache_dir=_VIDEO_CACHE_DIR_B),
            ],
        )

    def test_run_detects_no_events_when_transcription_fails(self):
        self.mock_transcription_pipeline.run.side_effect = ValueError("out of range")

        pipeline = _make_pipeline()
        with self.assertRaisesRegex(ValueError, "out of range"):
            pipeline.run(inputs=_make_inputs(), output_csv=_OUTPUT_CSV)

        self.mock_detector.detect.assert_not_called()
        self.mock_recorder.record.assert_not_called()


if __name__ == "__main__":
    unittest.main()
