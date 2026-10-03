import pathlib
import unittest
from unittest.mock import MagicMock, call, mock_open, patch

import dataclass_csv
import faster_whisper
import numpy as np
from parameterized import param, parameterized

import ballroom_scribe
import ballroom_scribe.models
import ballroom_scribe.speech_transcriber
import ballroom_scribe.utils
import tests.helpers.whisper_segment

_CHANNEL = 1
_CACHE_DIR = pathlib.Path("fake/path")
_WHISPER_SAMPLE_RATE = 16_000


def _make_audio_chunk(start_samples: int, num_samples: int, sample_rate: int = 10) -> ballroom_scribe.models.AudioChunk:
    """Helper to create a speech `AudioChunk` with sequential data."""
    return ballroom_scribe.models.AudioChunk(
        data=np.arange(start_samples, start_samples + num_samples, dtype=np.float32),
        sample_rate=sample_rate,
        classification=ballroom_scribe.models.AudioChunk.Classification.SPEECH,
        start_samples=start_samples,
    )


class TestFlattenSegmentsToWords(unittest.TestCase):
    """Unit tests for the `_flatten_segments_to_words` function."""

    def setUp(self):
        self.segment_maker = tests.helpers.whisper_segment.Maker(0.0)
        self.segments = [
            self.segment_maker.make(["  hello ", "world  "]),
            self.segment_maker.make(None),
            self.segment_maker.make([]),
            self.segment_maker.make(["dance"]),
        ]

    @parameterized.expand(
        [
            param("no_offset", offset_seconds=0.0),
            param("with_offset", offset_seconds=10.0),
        ]
    )
    def test_flatten_segments_success(self, _, offset_seconds):
        results = list(
            ballroom_scribe.speech_transcriber._flatten_segments_to_words(self.segments, offset_seconds=offset_seconds)
        )

        expected = [
            ballroom_scribe.models.Word(
                word="hello", start=offset_seconds + 0.0, end=offset_seconds + 0.5, probability=0.0
            ),
            ballroom_scribe.models.Word(
                word="world", start=offset_seconds + 0.5, end=offset_seconds + 1.0, probability=0.5
            ),
            ballroom_scribe.models.Word(
                word="dance", start=offset_seconds + 2.0, end=offset_seconds + 2.5, probability=2.0
            ),
        ]
        self.assertEqual(results, expected)


class TestFileOperations(unittest.TestCase):
    """Unit tests for the `_save_words_to_file` and `_read_words_from_file` functions."""

    def setUp(self):
        self.mock_open = mock_open()
        self.enterContext(patch("builtins.open", self.mock_open))

        self.mock_writer_cls = self.enterContext(patch.object(dataclass_csv, "DataclassWriter"))
        self.mock_reader_cls = self.enterContext(patch.object(dataclass_csv, "DataclassReader"))

    def test_save_words_to_file(self):
        mock_writer_instance = self.mock_writer_cls.return_value
        fake_path = pathlib.Path("/tmp/test_transcription.csv")
        dummy_words = [MagicMock(), MagicMock()]

        ballroom_scribe.speech_transcriber._save_words_to_file(dummy_words, fake_path)

        self.mock_open.assert_called_once_with(fake_path, "w", encoding="utf-8", newline="")
        self.mock_writer_cls.assert_called_once_with(
            self.mock_open(),
            dummy_words,
            ballroom_scribe.speech_transcriber.ballroom_scribe.models.Word,
        )
        mock_writer_instance.write.assert_called_once()

    def test_read_words_from_file(self):
        expected_words = [MagicMock(), MagicMock()]
        self.mock_reader_cls.return_value = iter(expected_words)
        fake_path = pathlib.Path("/tmp/test_transcription.csv")

        results = list(ballroom_scribe.speech_transcriber._read_words_from_file(fake_path))

        self.mock_open.assert_called_once_with(fake_path, "r", encoding="utf-8")
        self.mock_reader_cls.assert_called_once_with(
            self.mock_open(),
            ballroom_scribe.speech_transcriber.ballroom_scribe.models.Word,
        )
        self.assertEqual(results, expected_words)


class TestInit(unittest.TestCase):
    """Unit tests for the `SpeechTranscriber.__init__` method."""

    def test_init_validates_whisper_init_kwargs(self):
        whisper_beam_size = MagicMock()
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
            ballroom_scribe.speech_transcriber.SpeechTranscriber(
                whisper_init_kwargs={"model_size_or_path": "large-v3", "invalid keyword": "value does not matter"},
                whisper_beam_size=whisper_beam_size,
            )
        with self.assertRaisesRegex(TypeError, "missing a required argument"):
            ballroom_scribe.speech_transcriber.SpeechTranscriber(
                whisper_init_kwargs={},
                whisper_beam_size=whisper_beam_size,
            )


class TestTranscribe(unittest.TestCase):
    """Unit tests for the `SpeechTranscriber.transcribe` method."""

    def setUp(self):
        self.transcriber = ballroom_scribe.SpeechTranscriber(
            whisper_init_kwargs={"model_size_or_path": "large-v3"},
            whisper_beam_size=MagicMock(),
        )
        self.mock_load_or_transcribe_chunk = self.enterContext(
            patch.object(self.transcriber, "_load_or_transcribe_chunk", autospec=True)
        )

    def test_transcribes_every_speech_chunk_into_one_transcript_chunk(self):
        first_chunk = _make_audio_chunk(start_samples=100, num_samples=40)
        second_chunk = _make_audio_chunk(start_samples=200, num_samples=50)
        music_valley = ballroom_scribe.models.MusicValley(
            start_samples=80,
            sample_rate=10,
            duration_samples=300,
            speech_chunks=(first_chunk, second_chunk),
        )
        first_words = (
            ballroom_scribe.models.Word(word="hello", start=10.0, end=10.5),
            ballroom_scribe.models.Word(word="world", start=10.5, end=11.0),
        )
        second_words = (ballroom_scribe.models.Word(word="dance", start=20.0, end=20.5),)
        self.mock_load_or_transcribe_chunk.side_effect = [iter(first_words), iter(second_words)]

        result = self.transcriber.transcribe(music_valley=music_valley, channel=_CHANNEL, cache_dir=_CACHE_DIR)

        self.mock_load_or_transcribe_chunk.assert_has_calls(
            [
                call(audio_chunk=first_chunk, channel=_CHANNEL, cache_dir=_CACHE_DIR),
                call(audio_chunk=second_chunk, channel=_CHANNEL, cache_dir=_CACHE_DIR),
            ]
        )
        self.assertEqual(self.mock_load_or_transcribe_chunk.call_count, 2)
        self.assertEqual(
            result,
            ballroom_scribe.models.TranscriptChunk(
                start_samples=80,
                sample_rate=10,
                duration_samples=300,
                words=first_words + second_words,
            ),
        )

    def test_valley_without_speech_yields_empty_transcript_chunk(self):
        music_valley = ballroom_scribe.models.MusicValley(
            start_samples=80,
            sample_rate=10,
            duration_samples=300,
            speech_chunks=(),
        )

        result = self.transcriber.transcribe(music_valley=music_valley, channel=_CHANNEL, cache_dir=_CACHE_DIR)

        self.mock_load_or_transcribe_chunk.assert_not_called()
        self.assertEqual(
            result,
            ballroom_scribe.models.TranscriptChunk(
                start_samples=80,
                sample_rate=10,
                duration_samples=300,
                words=(),
            ),
        )


class TestLoadOrTranscribeChunk(unittest.TestCase):
    """Unit tests for the `SpeechTranscriber._load_or_transcribe_chunk` method."""

    def setUp(self):
        self.mock_path_exists = self.enterContext(patch.object(pathlib.Path, "exists"))
        self.mock_read_words = self.enterContext(
            patch.object(ballroom_scribe.speech_transcriber, "_read_words_from_file", autospec=True)
        )
        self.mock_save_words = self.enterContext(
            patch.object(ballroom_scribe.speech_transcriber, "_save_words_to_file", autospec=True)
        )

        self.mock_atomic_path = self.enterContext(patch.object(ballroom_scribe.utils, "atomic_path", autospec=True))
        self.mock_partial_cache_file = MagicMock(spec=pathlib.Path)
        self.mock_atomic_path.return_value.__enter__.return_value = self.mock_partial_cache_file

        self.transcriber = ballroom_scribe.SpeechTranscriber(
            whisper_init_kwargs={"model_size_or_path": "large-v3"},
            whisper_beam_size=MagicMock(),
        )
        self.mock_call_whisper = self.enterContext(patch.object(self.transcriber, "_call_whisper", autospec=True))

        self.audio_chunk = _make_audio_chunk(start_samples=100, num_samples=40)
        self.cache_file = _CACHE_DIR / f"0000000100-0000000140-{_CHANNEL}.csv"

    def test_cache_hit_reads_file(self):
        self.mock_path_exists.return_value = True

        expected_words = [MagicMock(), MagicMock()]
        self.mock_read_words.return_value = iter(expected_words)

        results = list(
            self.transcriber._load_or_transcribe_chunk(
                audio_chunk=self.audio_chunk, channel=_CHANNEL, cache_dir=_CACHE_DIR
            )
        )

        self.mock_path_exists.assert_called_once_with()
        self.mock_call_whisper.assert_not_called()
        self.mock_save_words.assert_not_called()
        self.mock_atomic_path.assert_not_called()
        self.mock_read_words.assert_called_once_with(self.cache_file)
        self.assertEqual(results, expected_words)

    def test_cache_miss_runs_transcription(self):
        self.mock_path_exists.return_value = False

        transcribed_words = [MagicMock(), MagicMock()]
        self.mock_call_whisper.return_value = iter(transcribed_words)

        results = list(
            self.transcriber._load_or_transcribe_chunk(
                audio_chunk=self.audio_chunk, channel=_CHANNEL, cache_dir=_CACHE_DIR
            )
        )

        self.mock_path_exists.assert_called_once_with()
        self.mock_read_words.assert_not_called()
        self.mock_call_whisper.assert_called_once_with(self.audio_chunk)
        self.mock_atomic_path.assert_called_once_with(self.cache_file)
        self.mock_save_words.assert_called_once_with(transcribed_words, self.mock_partial_cache_file)
        self.assertEqual(results, transcribed_words)


class TestCallWhisper(unittest.TestCase):
    """Unit tests for the `SpeechTranscriber._call_whisper` method."""

    def setUp(self):
        self.audio_chunk = _make_audio_chunk(start_samples=100, num_samples=40)

        self.mock_whisper_cls = self.enterContext(patch.object(faster_whisper, "WhisperModel", autospec=True))
        self.mock_whisper = self.mock_whisper_cls.return_value
        # `feature_extractor` is set in `WhisperModel.__init__`, so the autospec of the class doesn't know about it.
        self.mock_whisper.feature_extractor = MagicMock(sampling_rate=_WHISPER_SAMPLE_RATE)

        self.mock_resample = self.enterContext(
            patch.object(ballroom_scribe.models.AudioChunk, "resample", autospec=True)
        )
        self.mock_flatten = self.enterContext(
            patch.object(ballroom_scribe.speech_transcriber, "_flatten_segments_to_words", autospec=True)
        )

        self.whisper_init_kwargs = {"model_size_or_path": "large-v3"}
        self.whisper_beam_size = MagicMock()
        self.transcriber = ballroom_scribe.SpeechTranscriber(
            whisper_init_kwargs=self.whisper_init_kwargs,
            whisper_beam_size=self.whisper_beam_size,
        )

    def test_lazy_loads_faster_whisper(self):
        self.mock_whisper.transcribe.return_value = ([MagicMock()], MagicMock())

        self.mock_whisper_cls.assert_not_called()
        self.transcriber._call_whisper(self.audio_chunk)
        self.transcriber._call_whisper(self.audio_chunk)
        self.mock_whisper_cls.assert_called_once_with(**self.whisper_init_kwargs)

    def test_resamples_calls_whisper_and_flattens_segments(self):
        raw_segments = [MagicMock()]
        flattened_words = [MagicMock(), MagicMock()]

        self.mock_whisper.transcribe.return_value = (raw_segments, MagicMock())
        self.mock_flatten.return_value = iter(flattened_words)

        results = list(self.transcriber._call_whisper(self.audio_chunk))

        self.mock_resample.assert_called_once_with(self.audio_chunk, new_sample_rate=_WHISPER_SAMPLE_RATE)
        self.mock_whisper.transcribe.assert_called_once_with(
            self.mock_resample.return_value.data,
            beam_size=self.whisper_beam_size,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        self.mock_flatten.assert_called_once_with(raw_segments, offset_seconds=self.audio_chunk.start_seconds)
        self.assertEqual(results, flattened_words)


if __name__ == "__main__":
    unittest.main()
