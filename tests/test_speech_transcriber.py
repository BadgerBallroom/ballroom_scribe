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
_CPU_THREADS = 4


def _make_audio_chunk(start_samples: int, num_samples: int, sample_rate: int = 10) -> ballroom_scribe.models.AudioChunk:
    """Helper to create a speech `AudioChunk` with sequential data."""
    return ballroom_scribe.models.AudioChunk(
        data=np.arange(start_samples, start_samples + num_samples, dtype=np.float32),
        sample_rate=sample_rate,
        channel=_CHANNEL,
        classification=ballroom_scribe.models.AudioChunk.Classification.SPEECH,
        start_samples=start_samples,
    )


def _make_transcriber(**overrides) -> ballroom_scribe.SpeechTranscriber:
    """Helper to create a `SpeechTranscriber` with plain (non-mapping) arguments, overriding any of them."""
    arguments = {
        "whisper_device": "cpu",
        "whisper_model": "large-v3",
        "whisper_compute_type": "float16",
        "whisper_cpu_threads": _CPU_THREADS,
        "whisper_beam_size": MagicMock(),
        "whisper_init_kwargs": {},
    }
    arguments.update(overrides)
    return ballroom_scribe.SpeechTranscriber(**arguments)


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

    def setUp(self):
        self.mock_validate_torch_device = self.enterContext(
            patch.object(ballroom_scribe.utils, "validate_torch_device", autospec=True)
        )

    def test_init_validates_whisper_init_kwargs(self):
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
            _make_transcriber(whisper_init_kwargs={"invalid keyword": "value does not matter"})

    @parameterized.expand(
        [
            param("model", whisper_init_kwargs={"model_size_or_path": "large-v3"}),
            param("device", whisper_init_kwargs={"device": "cpu"}),
            param("compute_type", whisper_init_kwargs={"compute_type": "int8"}),
            param("cpu_threads", whisper_init_kwargs={"cpu_threads": 8}),
        ]
    )
    def test_init_rejects_dedicated_keys_in_whisper_init_kwargs(self, _, whisper_init_kwargs):
        with self.assertRaisesRegex(TypeError, "whisper_init_kwargs must not contain"):
            _make_transcriber(whisper_init_kwargs=whisper_init_kwargs)
        self.mock_validate_torch_device.assert_not_called()

    @parameterized.expand(
        [
            param("auto", whisper_device="auto"),
            param("indexed", whisper_device="cuda:1"),
        ]
    )
    def test_init_rejects_unknown_device(self, _, whisper_device):
        with self.assertRaisesRegex(
            ValueError, rf"whisper_device must be one of \['cpu', 'cuda'\], got '{whisper_device}'"
        ):
            _make_transcriber(whisper_device=whisper_device)
        self.mock_validate_torch_device.assert_not_called()

    def test_init_rejects_negative_cpu_threads(self):
        with self.assertRaisesRegex(ValueError, "whisper_cpu_threads must be 0 or positive, got -1"):
            _make_transcriber(whisper_cpu_threads=-1)

    @parameterized.expand(
        [
            param("cpu", whisper_device="cpu"),
            param("cuda", whisper_device="cuda"),
        ]
    )
    def test_init_validates_device(self, _, whisper_device):
        transcriber = _make_transcriber(whisper_device=whisper_device)

        self.mock_validate_torch_device.assert_called_once_with(whisper_device)
        self.assertEqual(
            transcriber._whisper_init_kwargs,
            {
                "model_size_or_path": "large-v3",
                "device": whisper_device,
                "compute_type": "float16",
                "cpu_threads": _CPU_THREADS,
            },
        )

    def test_init_propagates_device_validation_error(self):
        self.mock_validate_torch_device.side_effect = ValueError(
            "The device is set to CUDA, but CUDA is not available."
        )

        with self.assertRaisesRegex(ValueError, "CUDA is not available"):
            _make_transcriber(whisper_device="cuda")

    def test_init_passes_init_kwargs_through(self):
        whisper_init_kwargs = {"local_files_only": True}

        transcriber = _make_transcriber(
            whisper_model="turbo",
            whisper_compute_type="int8",
            whisper_cpu_threads=0,
            whisper_init_kwargs=whisper_init_kwargs,
        )

        self.assertEqual(
            transcriber._whisper_init_kwargs,
            {
                "model_size_or_path": "turbo",
                "device": "cpu",
                "compute_type": "int8",
                "cpu_threads": 0,
                "local_files_only": True,
            },
        )
        self.assertEqual(transcriber._model_tag, "turbo")
        # The caller's dictionary is not mutated.
        self.assertEqual(whisper_init_kwargs, {"local_files_only": True})

    def test_init_sanitizes_model_tag(self):
        transcriber = _make_transcriber(whisper_model="Systran/faster-whisper-large-v3")

        self.assertEqual(transcriber._model_tag, "Systran_faster-whisper-large-v3")


class TestTranscribe(unittest.TestCase):
    """Unit tests for the `SpeechTranscriber.transcribe` method."""

    def setUp(self):
        self.mock_validate_torch_device = self.enterContext(
            patch.object(ballroom_scribe.utils, "validate_torch_device", autospec=True)
        )

        self.transcriber = _make_transcriber()
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
            channel=_CHANNEL,
            speech_chunks=(first_chunk, second_chunk),
        )
        first_words = (
            ballroom_scribe.models.Word(word="hello", start=10.0, end=10.5),
            ballroom_scribe.models.Word(word="world", start=10.5, end=11.0),
        )
        second_words = (ballroom_scribe.models.Word(word="dance", start=20.0, end=20.5),)
        self.mock_load_or_transcribe_chunk.side_effect = [iter(first_words), iter(second_words)]

        result = self.transcriber.transcribe(music_valley=music_valley, cache_dir=_CACHE_DIR)

        self.mock_load_or_transcribe_chunk.assert_has_calls(
            [
                call(audio_chunk=first_chunk, cache_dir=_CACHE_DIR),
                call(audio_chunk=second_chunk, cache_dir=_CACHE_DIR),
            ]
        )
        self.assertEqual(self.mock_load_or_transcribe_chunk.call_count, 2)
        self.assertEqual(
            result,
            ballroom_scribe.models.TranscriptChunk(
                start_samples=80,
                sample_rate=10,
                duration_samples=300,
                channel=_CHANNEL,
                words=first_words + second_words,
            ),
        )

    def test_valley_without_speech_yields_empty_transcript_chunk(self):
        music_valley = ballroom_scribe.models.MusicValley(
            start_samples=80,
            sample_rate=10,
            duration_samples=300,
            channel=_CHANNEL,
            speech_chunks=(),
        )

        result = self.transcriber.transcribe(music_valley=music_valley, cache_dir=_CACHE_DIR)

        self.mock_load_or_transcribe_chunk.assert_not_called()
        self.assertEqual(
            result,
            ballroom_scribe.models.TranscriptChunk(
                start_samples=80,
                sample_rate=10,
                duration_samples=300,
                channel=_CHANNEL,
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

        self.mock_validate_torch_device = self.enterContext(
            patch.object(ballroom_scribe.utils, "validate_torch_device", autospec=True)
        )

        self.transcriber = _make_transcriber()
        self.mock_call_whisper = self.enterContext(patch.object(self.transcriber, "_call_whisper", autospec=True))

        self.audio_chunk = _make_audio_chunk(start_samples=100, num_samples=40)
        self.cache_file = _CACHE_DIR / f"0000000100-0000000140-{_CHANNEL}-large-v3.csv"

    def test_cache_hit_reads_file(self):
        self.mock_path_exists.return_value = True

        expected_words = [MagicMock(), MagicMock()]
        self.mock_read_words.return_value = iter(expected_words)

        results = list(self.transcriber._load_or_transcribe_chunk(audio_chunk=self.audio_chunk, cache_dir=_CACHE_DIR))

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

        results = list(self.transcriber._load_or_transcribe_chunk(audio_chunk=self.audio_chunk, cache_dir=_CACHE_DIR))

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
        # `feature_extractor` and `model` are set in `WhisperModel.__init__`, so the autospec of the class doesn't know
        # about them.
        self.mock_whisper.feature_extractor = MagicMock(sampling_rate=_WHISPER_SAMPLE_RATE)
        self.mock_whisper.model = MagicMock(device="cpu", compute_type="int8")

        self.mock_resample = self.enterContext(
            patch.object(ballroom_scribe.models.AudioChunk, "resample", autospec=True)
        )
        self.mock_flatten = self.enterContext(
            patch.object(ballroom_scribe.speech_transcriber, "_flatten_segments_to_words", autospec=True)
        )
        self.mock_validate_torch_device = self.enterContext(
            patch.object(ballroom_scribe.utils, "validate_torch_device", autospec=True)
        )

        self.whisper_beam_size = MagicMock()
        self.transcriber = _make_transcriber(whisper_beam_size=self.whisper_beam_size)

    def test_lazy_loads_faster_whisper(self):
        self.mock_whisper.transcribe.return_value = ([MagicMock()], MagicMock())

        self.mock_whisper_cls.assert_not_called()
        self.transcriber._call_whisper(self.audio_chunk)
        self.transcriber._call_whisper(self.audio_chunk)
        self.mock_whisper_cls.assert_called_once_with(
            model_size_or_path="large-v3", device="cpu", compute_type="float16", cpu_threads=_CPU_THREADS
        )

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
