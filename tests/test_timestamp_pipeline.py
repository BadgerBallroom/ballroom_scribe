import pathlib
import unittest
from unittest.mock import ANY, MagicMock, call, patch

import ollama
from parameterized import param, parameterized

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


def _make_music_valley(start_samples: int, duration_samples: int) -> ballroom_scribe.models.MusicValley:
    """Helper to create a `MusicValley` without any speech chunks."""
    return ballroom_scribe.models.MusicValley(
        start_samples=start_samples,
        sample_rate=10,
        duration_samples=duration_samples,
        speech_chunks=(),
    )


def _make_pipeline() -> ballroom_scribe.TimestampPipeline:
    """Helper to create a `TimestampPipeline` with standard test arguments."""
    return ballroom_scribe.TimestampPipeline(
        audio_classification_args=ballroom_scribe.cli.arguments.AudioClassificationArgs(
            pann_device="cuda",
            pann_chunk_minutes=10.0,
            pann_chunk_overlap_seconds=2.0,
            pann_thresholds={
                ballroom_scribe.models.AudioChunk.Classification.SPEECH: 0.18,
                ballroom_scribe.models.AudioChunk.Classification.MUSIC: 0.45,
            },
            pann_min_chunk_seconds=1.0,
        ),
        music_valley_detection_args=ballroom_scribe.cli.arguments.MusicValleyDetectionArgs(
            min_dance_seconds=35.0,
            max_song_dip_seconds=4.5,
            max_fade_speech_seconds=12.0,
            max_dip_speech_seconds=2.0,
        ),
        speech_transcription_args=ballroom_scribe.cli.arguments.SpeechTranscriptionArgs(
            whisper_init_kwargs={"model_size_or_path": "large-v3"},
            whisper_beam_size=5,
        ),
        language_processing_args=ballroom_scribe.cli.arguments.LanguageProcessingArgs(
            ollama_model="qwen2.5:7b",
            ollama_options=ollama.Options(temperature=0.0),
        ),
    )


class TestPipeline(unittest.TestCase):
    """Unit tests for the `TimestampPipeline` class."""

    def setUp(self):
        self.mock_free_vram = self.enterContext(patch.object(ballroom_scribe.utils, "free_vram", autospec=True))

        self.mock_cache_dir_for_video = self.enterContext(
            patch.object(ballroom_scribe.utils, "cache_dir_for_video", autospec=True),
        )
        self.mock_cache_dir_for_video.side_effect = [_VIDEO_CACHE_DIR_A, _VIDEO_CACHE_DIR_B]

        self.mock_resolve = self.enterContext(patch.object(pathlib.Path, "resolve", autospec=True))
        self.mock_resolve.side_effect = lambda path: path if path.is_absolute() else pathlib.Path("/resolved") / path

        self.mock_validate_input_channel = self.enterContext(
            patch.object(ballroom_scribe.utils, "validate_input_channel", autospec=True),
        )

        self.mock_mkdir = self.enterContext(patch.object(pathlib.Path, "mkdir", autospec=True))

        self.mock_extract_audio = self.enterContext(patch.object(ballroom_scribe.utils, "extract_audio", autospec=True))

        self.mock_audio_loader_cls = self.enterContext(
            patch.object(ballroom_scribe, "AudioLoader", autospec=True),
        )
        self.mock_audio_loader = self.mock_audio_loader_cls.return_value

        self.mock_audio_classifier_cls = self.enterContext(
            patch.object(ballroom_scribe, "AudioClassifier", autospec=True),
        )
        self.mock_audio_classifier = self.mock_audio_classifier_cls.return_value

        self.mock_music_valley_detector_cls = self.enterContext(
            patch.object(ballroom_scribe, "MusicValleyDetector", autospec=True),
        )
        self.mock_music_valley_detector = self.mock_music_valley_detector_cls.return_value

        self.mock_transcriber_cls = self.enterContext(
            patch.object(ballroom_scribe, "SpeechTranscriber", autospec=True),
        )
        self.mock_transcriber = self.mock_transcriber_cls.return_value

        self.mock_detector_cls = self.enterContext(
            patch.object(ballroom_scribe, "EventDetector", autospec=True),
        )
        self.mock_detector = self.mock_detector_cls.return_value

        self.mock_recorder_cls = self.enterContext(
            patch.object(ballroom_scribe, "EventRecorder", autospec=True),
        )
        self.mock_recorder = self.mock_recorder_cls.return_value

    def test_run(self):
        extracted_audio_a = pathlib.Path("fake_video_cache_a/fake_extracted_a.wav")
        extracted_audio_b = pathlib.Path("fake_video_cache_b/fake_extracted_b.wav")
        self.mock_extract_audio.side_effect = [extracted_audio_a, extracted_audio_b]

        audio_chunks_a = MagicMock(spec=list[ballroom_scribe.models.AudioChunk])
        audio_chunks_b = MagicMock(spec=list[ballroom_scribe.models.AudioChunk])
        self.mock_audio_loader.load.side_effect = [audio_chunks_a, audio_chunks_b]

        classified_chunks_a = MagicMock(spec=list[ballroom_scribe.models.AudioChunk])
        classified_chunks_b = MagicMock(spec=list[ballroom_scribe.models.AudioChunk])
        self.mock_audio_classifier.classify.side_effect = [classified_chunks_a, classified_chunks_b]

        # Use generators to make sure that the pipeline consumes all of the music valleys before freeing the PANN.
        spoken_valley_a = _make_music_valley(start_samples=0, duration_samples=100)
        silent_valley_a = _make_music_valley(start_samples=500, duration_samples=100)
        valley_b = _make_music_valley(start_samples=200, duration_samples=100)
        self.mock_music_valley_detector.detect.side_effect = [
            iter([spoken_valley_a, silent_valley_a]),
            iter([valley_b]),
        ]

        transcript_chunk_maker = tests.helpers.transcript_chunk.Maker()
        spoken_transcript_chunk_a = transcript_chunk_maker.make(["hello", "world"])
        silent_transcript_chunk_a = transcript_chunk_maker.make([])
        transcript_chunk_b = transcript_chunk_maker.make(["welcome"])
        self.mock_transcriber.transcribe.side_effect = [
            spoken_transcript_chunk_a,
            silent_transcript_chunk_a,
            transcript_chunk_b,
        ]

        event_a = MagicMock(spec=ballroom_scribe.models.Event)
        event_b = MagicMock(spec=ballroom_scribe.models.Event)
        self.mock_detector.detect.side_effect = [iter([event_a]), iter([event_b])]

        recorded_events: list[tuple[pathlib.Path, ballroom_scribe.models.Event]] = []
        self.mock_recorder.record.side_effect = lambda events, target: recorded_events.extend(events)

        manager = MagicMock()
        manager.attach_mock(self.mock_validate_input_channel, "validate_input_channel")
        manager.attach_mock(self.mock_extract_audio, "extract_audio")
        manager.attach_mock(self.mock_audio_classifier.classify, "classify")
        manager.attach_mock(self.mock_free_vram, "free_vram")
        manager.attach_mock(self.mock_transcriber.transcribe, "transcribe")
        manager.attach_mock(self.mock_recorder.record, "record")
        manager.attach_mock(self.mock_detector.detect, "detect")

        pipeline = _make_pipeline()
        pipeline.run(
            inputs=[
                ballroom_scribe.cli.arguments.InputCacheArgs(
                    input_video=_INPUT_VIDEO_A,
                    input_channel=_INPUT_CHANNEL_A,
                    cache_dir=_PARENT_CACHE_DIR_A,
                ),
                ballroom_scribe.cli.arguments.InputCacheArgs(
                    input_video=_INPUT_VIDEO_B,
                    input_channel=_INPUT_CHANNEL_B,
                    cache_dir=_PARENT_CACHE_DIR_B,
                ),
            ],
            output_csv=_OUTPUT_CSV,
        )

        self.mock_audio_loader_cls.assert_called_once_with(
            chunk_minutes=10.0,
            chunk_overlap_seconds=2.0,
        )
        self.mock_audio_classifier_cls.assert_called_once_with(
            device="cuda",
            thresholds={
                ballroom_scribe.models.AudioChunk.Classification.SPEECH: 0.18,
                ballroom_scribe.models.AudioChunk.Classification.MUSIC: 0.45,
            },
            min_segment_seconds=1.0,
        )
        self.mock_music_valley_detector_cls.assert_called_once_with(
            min_dance_seconds=35.0,
            max_song_dip_seconds=4.5,
            max_fade_speech_seconds=12.0,
            max_dip_speech_seconds=2.0,
        )
        self.mock_transcriber_cls.assert_called_once_with(
            whisper_init_kwargs={"model_size_or_path": "large-v3"},
            whisper_beam_size=5,
        )
        self.mock_detector_cls.assert_called_once_with(
            ollama_model="qwen2.5:7b",
            ollama_options=ollama.Options(temperature=0.0),
        )
        self.mock_recorder_cls.assert_called_once_with()

        self.assertEqual(
            self.mock_cache_dir_for_video.call_args_list,
            [
                call(parent_cache_dir=_PARENT_CACHE_DIR_A, input_video=_INPUT_VIDEO_A),
                call(parent_cache_dir=_PARENT_CACHE_DIR_B, input_video=_INPUT_VIDEO_B),
            ],
        )
        self.assertEqual(
            self.mock_mkdir.call_args_list,
            [
                call(_VIDEO_CACHE_DIR_A, parents=True, exist_ok=True),
                call(_VIDEO_CACHE_DIR_B, parents=True, exist_ok=True),
            ],
        )
        self.assertEqual(
            self.mock_audio_loader.load.call_args_list,
            [
                call(input_audio=extracted_audio_a),
                call(input_audio=extracted_audio_b),
            ],
        )
        self.assertEqual(
            self.mock_music_valley_detector.detect.call_args_list,
            [
                call(classified_chunks=classified_chunks_a),
                call(classified_chunks=classified_chunks_b),
            ],
        )
        self.assertEqual(
            recorded_events,
            [
                (pathlib.Path("/resolved") / _INPUT_VIDEO_A, event_a),
                (pathlib.Path("/resolved") / _INPUT_VIDEO_B, event_b),
            ],
        )

        # Every video completes each stage before any video begins the next stage, and events stream lazily inside
        # `record`.
        self.assertEqual(
            manager.mock_calls,
            [
                call.validate_input_channel(input_video=_INPUT_VIDEO_A, input_channel=_INPUT_CHANNEL_A),
                call.validate_input_channel(input_video=_INPUT_VIDEO_B, input_channel=_INPUT_CHANNEL_B),
                call.extract_audio(
                    input_video=_INPUT_VIDEO_A,
                    input_channel=_INPUT_CHANNEL_A,
                    cache_dir=_VIDEO_CACHE_DIR_A,
                ),
                call.extract_audio(
                    input_video=_INPUT_VIDEO_B,
                    input_channel=_INPUT_CHANNEL_B,
                    cache_dir=_VIDEO_CACHE_DIR_B,
                ),
                call.classify(
                    audio_chunks=audio_chunks_a,
                    channel=_INPUT_CHANNEL_A,
                    cache_dir=_VIDEO_CACHE_DIR_A,
                ),
                call.classify(
                    audio_chunks=audio_chunks_b,
                    channel=_INPUT_CHANNEL_B,
                    cache_dir=_VIDEO_CACHE_DIR_B,
                ),
                call.free_vram(),
                call.transcribe(
                    music_valley=spoken_valley_a,
                    channel=_INPUT_CHANNEL_A,
                    cache_dir=_VIDEO_CACHE_DIR_A,
                ),
                call.transcribe(
                    music_valley=silent_valley_a,
                    channel=_INPUT_CHANNEL_A,
                    cache_dir=_VIDEO_CACHE_DIR_A,
                ),
                call.transcribe(
                    music_valley=valley_b,
                    channel=_INPUT_CHANNEL_B,
                    cache_dir=_VIDEO_CACHE_DIR_B,
                ),
                call.free_vram(),
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

    @parameterized.expand(
        [
            param(
                "identical_paths",
                cache_dirs=(pathlib.Path("shared_cache"), pathlib.Path("shared_cache")),
            ),
            param(
                "relative_and_absolute",
                cache_dirs=(pathlib.Path("shared_cache"), pathlib.Path("/resolved/shared_cache")),
            ),
        ]
    )
    def test_run_rejects_videos_sharing_a_cache_dir(self, _, cache_dirs):
        self.mock_cache_dir_for_video.side_effect = cache_dirs

        pipeline = _make_pipeline()
        with self.assertRaisesRegex(ValueError, "would share the cache directory"):
            pipeline.run(
                inputs=[
                    ballroom_scribe.cli.arguments.InputCacheArgs(
                        input_video=_INPUT_VIDEO_A,
                        input_channel=_INPUT_CHANNEL_A,
                        cache_dir=_PARENT_CACHE_DIR_A,
                    ),
                    ballroom_scribe.cli.arguments.InputCacheArgs(
                        input_video=_INPUT_VIDEO_B,
                        input_channel=_INPUT_CHANNEL_B,
                        cache_dir=_PARENT_CACHE_DIR_B,
                    ),
                ],
                output_csv=_OUTPUT_CSV,
            )

        self.mock_validate_input_channel.assert_not_called()
        self.mock_extract_audio.assert_not_called()

    def test_run_fails_before_extracting_when_a_channel_is_invalid(self):
        self.mock_validate_input_channel.side_effect = [None, ValueError("out of range")]

        pipeline = _make_pipeline()
        with self.assertRaisesRegex(ValueError, "out of range"):
            pipeline.run(
                inputs=[
                    ballroom_scribe.cli.arguments.InputCacheArgs(
                        input_video=_INPUT_VIDEO_A,
                        input_channel=_INPUT_CHANNEL_A,
                        cache_dir=_PARENT_CACHE_DIR_A,
                    ),
                    ballroom_scribe.cli.arguments.InputCacheArgs(
                        input_video=_INPUT_VIDEO_B,
                        input_channel=_INPUT_CHANNEL_B,
                        cache_dir=_PARENT_CACHE_DIR_B,
                    ),
                ],
                output_csv=_OUTPUT_CSV,
            )

        self.mock_extract_audio.assert_not_called()


if __name__ == "__main__":
    unittest.main()
