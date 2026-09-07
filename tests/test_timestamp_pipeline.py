import pathlib
import unittest
from unittest.mock import MagicMock, patch

import ollama

import ballroom_scribe
import ballroom_scribe.cli.arguments
import ballroom_scribe.models
import ballroom_scribe.timestamp_pipeline
import ballroom_scribe.utils

_INPUT_VIDEO = pathlib.Path("fake_input.mp4")
_OUTPUT_CSV = pathlib.Path("fake_output.csv")
_CACHE_DIR = pathlib.Path("fake_cache")


class TestPipeline(unittest.TestCase):
    """Unit tests for the `TimestampPipeline` class."""

    def setUp(self):
        self.mock_free_vram = self.enterContext(patch.object(ballroom_scribe.timestamp_pipeline, "_free_vram"))

        self.mock_cache_dir_for_video = self.enterContext(
            patch.object(ballroom_scribe.utils, "cache_dir_for_video", autospec=True),
        )
        self.mock_cache_dir_for_video.return_value = pathlib.Path("fake_video_cache")

        self.mock_mkdir = self.enterContext(patch.object(pathlib.Path, "mkdir", autospec=True))

        self.mock_resolve = self.enterContext(patch.object(pathlib.Path, "resolve", autospec=True))

        self.mock_extract_audio = self.enterContext(patch.object(ballroom_scribe.utils, "extract_audio", autospec=True))

        self.mock_vocal_isolator_cls = self.enterContext(
            patch.object(ballroom_scribe, "VocalIsolator", autospec=True),
        )
        self.mock_vocal_isolator = self.mock_vocal_isolator_cls.return_value

        self.mock_transcriber_cls = self.enterContext(
            patch.object(ballroom_scribe, "SpeechTranscriber", autospec=True),
        )
        self.mock_transcriber = self.mock_transcriber_cls.return_value

        self.mock_chunker_cls = self.enterContext(
            patch.object(ballroom_scribe, "TranscriptChunker", autospec=True),
        )
        self.mock_chunker = self.mock_chunker_cls.return_value

        self.mock_detector_cls = self.enterContext(
            patch.object(ballroom_scribe, "EventDetector", autospec=True),
        )
        self.mock_detector = self.mock_detector_cls.return_value

        self.mock_recorder_cls = self.enterContext(
            patch.object(ballroom_scribe, "EventRecorder", autospec=True),
        )
        self.mock_recorder = self.mock_recorder_cls.return_value

    def test_run(self):
        extracted_audio = self.mock_cache_dir_for_video.return_value / "Audio.wav"
        isolated_vocals = self.mock_cache_dir_for_video.return_value / "Vocals.wav"

        words = MagicMock(spec=list[ballroom_scribe.models.Word])
        self.mock_transcriber.transcribe.return_value = words

        chunks_of_words = [MagicMock(spec=ballroom_scribe.models.TranscriptChunk) for _ in range(5)]
        self.mock_chunker.chunk.return_value = chunks_of_words

        events = MagicMock(spec=list[ballroom_scribe.models.Event])
        self.mock_detector.detect.return_value = events

        transcript_chunking_args = ballroom_scribe.cli.arguments.TranscriptChunkingArgs(
            split_on_silence_threshold=10.0,
        )
        vocal_isolation_args = ballroom_scribe.cli.arguments.VocalIsolationArgs(
            demucs_name="htdemucs",
            demucs_device="cuda",
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
            transcript_chunking_args=transcript_chunking_args,
            vocal_isolation_args=vocal_isolation_args,
            speech_transcription_args=speech_transcription_args,
            language_processing_args=language_processing_args,
        )
        pipeline.run(
            input_video=_INPUT_VIDEO,
            output_csv=_OUTPUT_CSV,
            cache_dir=_CACHE_DIR,
        )

        self.mock_vocal_isolator_cls.assert_called_once_with(
            demucs_name="htdemucs",
            demucs_device="cuda",
        )
        self.mock_transcriber_cls.assert_called_once_with(
            whisper_init_kwargs={"model_size_or_path": "large-v3"},
            whisper_beam_size=5,
        )
        self.mock_chunker_cls.assert_called_once_with(split_on_silence_threshold=10.0)
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
            output_audio=extracted_audio,
        )
        self.mock_vocal_isolator.separate.assert_called_once_with(
            input_audio=extracted_audio, output_vocals=isolated_vocals
        )
        self.mock_transcriber.transcribe.assert_called_once_with(
            input_audio=isolated_vocals,
            output_text=self.mock_cache_dir_for_video.return_value / "Speech.csv",
        )
        self.mock_chunker.chunk.assert_called_once_with(words=words)
        self.mock_detector.detect.assert_called_once_with(
            chunks_of_words=chunks_of_words,
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
