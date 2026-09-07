import pathlib
import unittest
from unittest.mock import MagicMock, mock_open, patch

import dataclass_csv
import faster_whisper

import ballroom_scribe
import ballroom_scribe.models
import ballroom_scribe.speech_transcriber
import ballroom_scribe.utils
import tests.helpers.whisper_segment


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

    def test_flatten_segments_success(self):
        results = list(ballroom_scribe.speech_transcriber._flatten_segments_to_words(self.segments))

        expected = [
            ballroom_scribe.models.Word(word="hello", start=0.0, end=0.5, probability=0.0),
            ballroom_scribe.models.Word(word="world", start=0.5, end=1.0, probability=0.5),
            ballroom_scribe.models.Word(word="dance", start=2.0, end=2.5, probability=2.0),
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
        self.mock_audio = MagicMock(spec=pathlib.Path)
        self.mock_output = MagicMock(spec=pathlib.Path)

        self.mock_read_words = self.enterContext(
            patch.object(ballroom_scribe.speech_transcriber, "_read_words_from_file", autospec=True)
        )

        self.transcriber = ballroom_scribe.SpeechTranscriber(
            whisper_init_kwargs={"model_size_or_path": "large-v3"},
            whisper_beam_size=MagicMock(),
        )
        self.mock_call_whisper = self.enterContext(patch.object(self.transcriber, "_call_whisper", autospec=True))

    def test_transcribe_cache_hit_reads_file(self):
        self.mock_output.exists.return_value = True

        expected_words = [MagicMock(), MagicMock()]
        self.mock_read_words.return_value = iter(expected_words)

        results = list(self.transcriber.transcribe(self.mock_audio, self.mock_output))

        self.mock_call_whisper.assert_not_called()
        self.mock_read_words.assert_called_once_with(self.mock_output)
        self.assertEqual(results, expected_words)

    def test_transcribe_cache_miss_runs_transcription(self):
        self.mock_output.exists.return_value = False

        final_words = [MagicMock(), MagicMock()]
        self.mock_read_words.return_value = iter(final_words)

        results = list(self.transcriber.transcribe(self.mock_audio, self.mock_output))

        self.mock_call_whisper.assert_called_once_with(self.mock_audio, self.mock_output)
        self.mock_read_words.assert_called_once_with(self.mock_output)
        self.assertEqual(results, final_words)


class TestCallWhisper(unittest.TestCase):
    """Unit tests for the `SpeechTranscriber._call_whisper` method."""

    def setUp(self):
        self.mock_audio = MagicMock(spec=pathlib.Path)
        self.mock_output = MagicMock(spec=pathlib.Path)

        self.mock_whisper_cls = self.enterContext(patch.object(faster_whisper, "WhisperModel", autospec=True))
        self.mock_whisper = self.mock_whisper_cls.return_value

        self.mock_save_words = self.enterContext(
            patch.object(ballroom_scribe.speech_transcriber, "_save_words_to_file", autospec=True)
        )
        self.mock_flatten = self.enterContext(
            patch.object(ballroom_scribe.speech_transcriber, "_flatten_segments_to_words", autospec=True)
        )

        self.mock_atomic_path = self.enterContext(patch.object(ballroom_scribe.utils, "atomic_path", autospec=True))
        self.mock_partial_output = MagicMock(spec=pathlib.Path)
        self.mock_atomic_path.return_value.__enter__.return_value = self.mock_partial_output

        self.whisper_init_kwargs = {"model_size_or_path": "large-v3"}
        self.whisper_beam_size = MagicMock()
        self.transcriber = ballroom_scribe.SpeechTranscriber(
            whisper_init_kwargs=self.whisper_init_kwargs,
            whisper_beam_size=self.whisper_beam_size,
        )

    def test_lazy_loads_faster_whisper(self):
        self.mock_whisper.transcribe.return_value = ([MagicMock()], MagicMock())

        self.mock_whisper_cls.assert_not_called()
        self.transcriber._call_whisper(self.mock_audio, self.mock_output)
        self.transcriber._call_whisper(self.mock_audio, self.mock_output)
        self.mock_whisper_cls.assert_called_once_with(**self.whisper_init_kwargs)

    def test_calls_whisper_and_saves_words(self):
        raw_segments = [MagicMock()]
        flattened_words = [MagicMock(), MagicMock()]

        self.mock_whisper.transcribe.return_value = (raw_segments, MagicMock())
        self.mock_flatten.return_value = iter(flattened_words)

        self.transcriber._call_whisper(self.mock_audio, self.mock_output)

        self.mock_whisper.transcribe.assert_called_once_with(
            str(self.mock_audio),
            beam_size=self.whisper_beam_size,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        self.mock_flatten.assert_called_once_with(raw_segments)
        self.mock_atomic_path.assert_called_once_with(self.mock_output)
        self.mock_save_words.assert_called_once_with(self.mock_flatten.return_value, self.mock_partial_output)


if __name__ == "__main__":
    unittest.main()
