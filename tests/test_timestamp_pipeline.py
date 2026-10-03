import pathlib
import unittest
from unittest.mock import MagicMock, call, patch

import ollama

import ballroom_scribe
import ballroom_scribe.cli.arguments
import ballroom_scribe.models
import ballroom_scribe.utils
import tests.helpers.transcript_chunk

_INPUT_VIDEO = pathlib.Path("fake_input.mp4")
_INPUT_CHANNEL = 1
_OUTPUT_CSV = pathlib.Path("fake_output.csv")
_CACHE_DIR = pathlib.Path("fake_cache")


def _make_music_valley(start_samples: int, duration_samples: int) -> ballroom_scribe.models.MusicValley:
    """Helper to create a `MusicValley` without any speech chunks."""
    return ballroom_scribe.models.MusicValley(
        start_samples=start_samples,
        sample_rate=10,
        duration_samples=duration_samples,
        speech_chunks=(),
    )


class TestPipeline(unittest.TestCase):
    """Unit tests for the `TimestampPipeline` class."""

    def setUp(self):
        self.mock_free_vram = self.enterContext(patch.object(ballroom_scribe.utils, "free_vram", autospec=True))

        self.mock_cache_dir_for_video = self.enterContext(
            patch.object(ballroom_scribe.utils, "cache_dir_for_video", autospec=True),
        )
        self.mock_cache_dir_for_video.return_value = pathlib.Path("fake_video_cache")

        self.mock_mkdir = self.enterContext(patch.object(pathlib.Path, "mkdir", autospec=True))

        self.mock_resolve = self.enterContext(patch.object(pathlib.Path, "resolve", autospec=True))
        self.mock_resolve.return_value = pathlib.Path("/resolved/fake_input.mp4")

        self.mock_extract_audio = self.enterContext(patch.object(ballroom_scribe.utils, "extract_audio", autospec=True))
        self.mock_extract_audio.return_value = pathlib.Path("fake_video_cache/fake_extracted.wav")

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
        audio_chunks = MagicMock(spec=list[ballroom_scribe.models.AudioChunk])
        self.mock_audio_loader.load.return_value = audio_chunks

        classified_chunks = MagicMock(spec=list[ballroom_scribe.models.AudioChunk])
        self.mock_audio_classifier.classify.return_value = classified_chunks

        # Use a generator to make sure that the pipeline consumes all of the music valleys before freeing the PANN.
        spoken_valley = _make_music_valley(start_samples=0, duration_samples=100)
        silent_valley = _make_music_valley(start_samples=500, duration_samples=100)
        self.mock_music_valley_detector.detect.return_value = iter([spoken_valley, silent_valley])

        transcript_chunk_maker = tests.helpers.transcript_chunk.Maker()
        spoken_transcript_chunk = transcript_chunk_maker.make(["hello", "world"])
        silent_transcript_chunk = transcript_chunk_maker.make([])
        self.mock_transcriber.transcribe.side_effect = [spoken_transcript_chunk, silent_transcript_chunk]

        events = MagicMock(spec=list[ballroom_scribe.models.Event])
        self.mock_detector.detect.return_value = events

        audio_classification_args = ballroom_scribe.cli.arguments.AudioClassificationArgs(
            pann_device="cuda",
            pann_chunk_minutes=10.0,
            pann_chunk_overlap_seconds=2.0,
            pann_thresholds={
                ballroom_scribe.models.AudioChunk.Classification.SPEECH: 0.18,
                ballroom_scribe.models.AudioChunk.Classification.MUSIC: 0.45,
            },
            pann_min_chunk_seconds=1.0,
        )
        music_valley_detection_args = ballroom_scribe.cli.arguments.MusicValleyDetectionArgs(
            min_dance_seconds=35.0,
            max_song_dip_seconds=4.5,
            max_fade_speech_seconds=12.0,
            max_dip_speech_seconds=2.0,
        )
        speech_transcription_args = ballroom_scribe.cli.arguments.SpeechTranscriptionArgs(
            whisper_init_kwargs={"model_size_or_path": "large-v3"},
            whisper_beam_size=5,
        )
        language_processing_args = ballroom_scribe.cli.arguments.LanguageProcessingArgs(
            ollama_model="qwen2.5:7b",
            ollama_options=ollama.Options(temperature=0.0),
        )

        pipeline = ballroom_scribe.TimestampPipeline(
            audio_classification_args=audio_classification_args,
            music_valley_detection_args=music_valley_detection_args,
            speech_transcription_args=speech_transcription_args,
            language_processing_args=language_processing_args,
        )
        pipeline.run(
            input_video=_INPUT_VIDEO,
            input_channel=_INPUT_CHANNEL,
            output_csv=_OUTPUT_CSV,
            cache_dir=_CACHE_DIR,
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
        self.mock_cache_dir_for_video.assert_called_once_with(
            parent_cache_dir=_CACHE_DIR,
            input_video=_INPUT_VIDEO,
        )
        self.mock_mkdir.assert_called_once_with(self.mock_cache_dir_for_video.return_value, parents=True, exist_ok=True)
        self.mock_extract_audio.assert_called_once_with(
            input_video=_INPUT_VIDEO,
            input_channel=_INPUT_CHANNEL,
            cache_dir=self.mock_cache_dir_for_video.return_value,
        )
        self.mock_audio_loader.load.assert_called_once_with(input_audio=self.mock_extract_audio.return_value)
        self.mock_audio_classifier.classify.assert_called_once_with(
            audio_chunks=audio_chunks,
            channel=_INPUT_CHANNEL,
            cache_dir=self.mock_cache_dir_for_video.return_value,
        )
        self.mock_music_valley_detector.detect.assert_called_once_with(classified_chunks=classified_chunks)
        self.mock_transcriber.transcribe.assert_has_calls(
            [
                call(
                    music_valley=spoken_valley,
                    channel=_INPUT_CHANNEL,
                    cache_dir=self.mock_cache_dir_for_video.return_value,
                ),
                call(
                    music_valley=silent_valley,
                    channel=_INPUT_CHANNEL,
                    cache_dir=self.mock_cache_dir_for_video.return_value,
                ),
            ]
        )
        self.assertEqual(self.mock_transcriber.transcribe.call_count, 2)
        # Every music valley's transcript chunk is passed on, even one without words, so that each produces an event.
        self.mock_detector.detect.assert_called_once_with(
            chunks_of_words=[spoken_transcript_chunk, silent_transcript_chunk],
            cache_dir=self.mock_cache_dir_for_video.return_value,
        )
        self.mock_recorder_cls.assert_called_once_with()
        self.mock_resolve.assert_called_once_with(_INPUT_VIDEO)
        self.mock_recorder.record.assert_called_once_with(
            input_video=self.mock_resolve.return_value,
            events=events,
            target=_OUTPUT_CSV,
        )
        self.mock_free_vram.assert_called()


if __name__ == "__main__":
    unittest.main()
