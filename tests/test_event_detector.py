import json
import pathlib
import unittest
from unittest.mock import ANY, MagicMock, call, mock_open, patch

import ollama
from parameterized import param, parameterized

import ballroom_scribe
import ballroom_scribe.event_detector
import ballroom_scribe.models
import ballroom_scribe.utils
import tests.helpers.transcript_chunk

_OLLAMA_MODEL = "fake-model"
_OLLAMA_OPTIONS = MagicMock(spec=ollama.Options)
_CACHE_DIR = pathlib.Path("path/to/fake/dir")


class TestAlignQuote(unittest.TestCase):
    """Unit tests for the `_align_quote` function."""

    def setUp(self):
        self.mock_find_closest_matching = self.enterContext(
            patch.object(ballroom_scribe.utils, "find_closest_matching")
        )

    def test_calls_find_closest_matching(self):
        words = [
            "Thank",
            "you,",
            "dancers!",
            "We",
            "now",
            "welcome",
            "our",
            "gold",
            "dancers",
            "to",
            "the",
            "floor.",
            "This",
            "is",
            "event",
            "8.",
        ]
        chunk = tests.helpers.transcript_chunk.Maker(start_samples=200).make(words)
        self.mock_find_closest_matching.return_value = 3

        quote = "We now welcome our gold dancers"
        quote_start_time = ballroom_scribe.event_detector._align_quote(quote, chunk.words)

        self.mock_find_closest_matching.assert_called_once_with(
            needle=["we", "now", "welcome", "our", "gold", "dancers"],
            haystack=[
                "thank",
                "you",
                "dancers",
                "we",
                "now",
                "welcome",
                "our",
                "gold",
                "dancers",
                "to",
                "the",
                "floor",
                "this",
                "is",
                "event",
                "8",
            ],
        )
        self.assertEqual(
            quote_start_time,
            chunk.words[self.mock_find_closest_matching.return_value].start,
        )


class TestParseResponse(unittest.TestCase):
    """Unit tests for the `_parse_response` function."""

    def setUp(self):
        self.mock_align_quote = self.enterContext(patch.object(ballroom_scribe.event_detector, "_align_quote"))

    def test_parse_valid_json(self):
        quote = "Next onto the floor is the Bronze Smooth Waltz Final."
        response_json = json.dumps(
            {
                "events": [
                    {
                        "level": "Bronze",
                        "style": "Smooth",
                        "dance": "Waltz",
                        "round": "Final",
                        "quote": quote,
                    },
                ],
            },
        )
        words = MagicMock()
        self.mock_align_quote.return_value = 45.5

        events = ballroom_scribe.event_detector._parse_response(response_json, words)

        self.mock_align_quote.assert_called_once_with(quote=quote, words=words)
        self.assertEqual(
            events,
            [
                ballroom_scribe.models.Event(
                    level="Bronze",
                    style="Smooth",
                    dance="Waltz",
                    round="Final",
                    announcement_quote=quote,
                    announcement_start_seconds=45.5,
                    speech_detected=True,
                    event_detected=True,
                )
            ],
        )

    @parameterized.expand(
        [
            param("empty_json", input_json="{}"),
            param("malformed_json", input_json="{'level': 'Bronze', not valid json}"),
            param("non_dict_json", input_json='["not", "a", "dict"]'),
        ]
    )
    def test_parse_invalid_json_returns_empty(self, _, input_json):
        events = ballroom_scribe.event_detector._parse_response(input_json, ())
        self.assertEqual(events, [])

    def test_parse_partial_keys_populates_defaults(self):
        self.mock_align_quote.return_value = 25.0
        partial_json = json.dumps({"events": [{"dance": "Samba"}]})

        events = ballroom_scribe.event_detector._parse_response(partial_json, ())

        self.assertEqual(
            events,
            [
                ballroom_scribe.models.Event(
                    level="",
                    style="",
                    dance="Samba",
                    round="",
                    announcement_quote="",
                    announcement_start_seconds=25.0,
                    speech_detected=True,
                    event_detected=True,
                )
            ],
        )

    def test_parse_coerces_non_string_field_values_to_strings(self):
        self.mock_align_quote.return_value = 50.0
        mixed_types_json = json.dumps(
            {
                "events": [
                    {
                        "level": 1,
                        "style": None,
                        "dance": "Tango",
                        "round": 2,
                        "quote": False,
                    },
                ],
            },
        )

        events = ballroom_scribe.event_detector._parse_response(mixed_types_json, ())

        self.assertEqual(
            events,
            [
                ballroom_scribe.models.Event(
                    level="1",
                    style="None",
                    dance="Tango",
                    round="2",
                    announcement_quote="False",
                    announcement_start_seconds=50.0,
                    speech_detected=True,
                    event_detected=True,
                )
            ],
        )

    def test_parse_sorts_events_by_announcement_start_seconds(self):
        self.mock_align_quote.side_effect = [80.0, 20.0, 50.0]
        response_json = json.dumps(
            {
                "events": [
                    {
                        "level": "Gold",
                        "style": "Standard",
                        "dance": "Quickstep",
                        "round": "Final",
                        "quote": "Third announcement.",
                    },
                    {
                        "level": "Bronze",
                        "style": "Smooth",
                        "dance": "Waltz",
                        "round": "Final",
                        "quote": "First announcement.",
                    },
                    {
                        "level": "Silver",
                        "style": "Rhythm",
                        "dance": "Cha Cha",
                        "round": "Semifinal",
                        "quote": "Second announcement.",
                    },
                ],
            },
        )

        events = ballroom_scribe.event_detector._parse_response(response_json, ())

        self.assertEqual(
            events,
            [
                ballroom_scribe.models.Event(
                    level="Bronze",
                    style="Smooth",
                    dance="Waltz",
                    round="Final",
                    announcement_quote="First announcement.",
                    announcement_start_seconds=20.0,
                    speech_detected=True,
                    event_detected=True,
                ),
                ballroom_scribe.models.Event(
                    level="Silver",
                    style="Rhythm",
                    dance="Cha Cha",
                    round="Semifinal",
                    announcement_quote="Second announcement.",
                    announcement_start_seconds=50.0,
                    speech_detected=True,
                    event_detected=True,
                ),
                ballroom_scribe.models.Event(
                    level="Gold",
                    style="Standard",
                    dance="Quickstep",
                    round="Final",
                    announcement_quote="Third announcement.",
                    announcement_start_seconds=80.0,
                    speech_detected=True,
                    event_detected=True,
                ),
            ],
        )


class TestMakePlaceholderEvent(unittest.TestCase):
    """Unit tests for the `_make_placeholder_event` function."""

    @parameterized.expand(
        [
            param("speech_detected", speech_detected=True),
            param("no_speech_detected", speech_detected=False),
        ]
    )
    def test_makes_placeholder_event_at_chunk_midpoint(self, _, speech_detected):
        chunk = ballroom_scribe.models.TranscriptChunk(
            start_samples=100,
            sample_rate=10,
            duration_samples=100,
            words=(),
        )

        event = ballroom_scribe.event_detector._make_placeholder_event(
            chunk,
            speech_detected=speech_detected,
        )

        self.assertEqual(
            event,
            ballroom_scribe.models.Event(
                level="",
                style="",
                dance="",
                round="",
                announcement_quote="",
                announcement_start_seconds=15.0,
                speech_detected=speech_detected,
                event_detected=False,
            ),
        )


class TestModelSupportsThinking(unittest.TestCase):
    def setUp(self):
        self.mock_show = self.enterContext(patch.object(ollama, "show", autospec=True))
        self.model = MagicMock(spec=str)

    @parameterized.expand(
        [
            param("capabilities_none", capabilities=None, expected=False),
            param("without_thinking", capabilities=["pondering", "contemplating", "mulling"], expected=False),
            param("with_thinking", capabilities=["responding", "thinking"], expected=True),
        ]
    )
    def test_capabilities(self, _, capabilities, expected):
        self.mock_show.return_value.capabilities = capabilities

        result = ballroom_scribe.event_detector._model_supports_thinking(self.model)

        self.mock_show.assert_called_once_with(self.model)
        self.assertEqual(result, expected)


class TestEventDetectorDetect(unittest.TestCase):
    """Unit tests for `EventDetector.detect`."""

    def setUp(self):
        self.mock_model_supports_thinking = self.enterContext(
            patch.object(ballroom_scribe.event_detector, "_model_supports_thinking", return_value=True)
        )

        self.mock_process_chunk = self.enterContext(
            patch.object(
                ballroom_scribe.EventDetector,
                "_process_chunk",
                autospec=True,
            )
        )
        self.mock_make_placeholder_event = self.enterContext(
            patch.object(
                ballroom_scribe.event_detector,
                "_make_placeholder_event",
                autospec=True,
            )
        )

        transcript_chunk_maker = tests.helpers.transcript_chunk.Maker()
        self.chunks = [
            transcript_chunk_maker.make(["Housekeeping", "rules."]),
            transcript_chunk_maker.make(["Gold", "Latin", "Cha", "Cha", "Final."]),
            transcript_chunk_maker.make([]),
            transcript_chunk_maker.make(["Gold", "Latin", "Rumba", "Final."]),
        ]

    def test_detect_yields_at_least_one_event_per_chunk(self):
        cha_cha_event = ballroom_scribe.models.Event(
            level="Gold",
            style="Latin",
            dance="Cha Cha",
            round="Final",
            announcement_quote="Gold Latin Cha Cha Final.",
            announcement_start_seconds=10.0,
            speech_detected=True,
            event_detected=True,
        )
        rumba_event = ballroom_scribe.models.Event(
            level="Gold",
            style="Latin",
            dance="Rumba",
            round="Final",
            announcement_quote="Gold Latin Rumba Final.",
            announcement_start_seconds=30.0,
            speech_detected=True,
            event_detected=True,
        )
        no_event_placeholder = MagicMock(spec=ballroom_scribe.models.Event)
        no_speech_placeholder = MagicMock(spec=ballroom_scribe.models.Event)
        self.mock_process_chunk.side_effect = [[], [cha_cha_event], [rumba_event]]
        self.mock_make_placeholder_event.side_effect = [no_event_placeholder, no_speech_placeholder]

        detector = ballroom_scribe.EventDetector(
            ollama_model=_OLLAMA_MODEL,
            ollama_options=_OLLAMA_OPTIONS,
        )
        results = list(detector.detect(self.chunks, _CACHE_DIR))

        # The empty chunk should not be sent to the LLM.
        self.mock_process_chunk.assert_has_calls(
            [
                call(detector, self.chunks[0], _CACHE_DIR),
                call(detector, self.chunks[1], _CACHE_DIR),
                call(detector, self.chunks[3], _CACHE_DIR),
            ]
        )
        self.assertEqual(self.mock_process_chunk.call_count, 3)
        self.mock_make_placeholder_event.assert_has_calls(
            [
                call(self.chunks[0], speech_detected=True),
                call(self.chunks[2], speech_detected=False),
            ]
        )
        self.assertEqual(self.mock_make_placeholder_event.call_count, 2)
        self.assertEqual(
            results,
            [
                no_event_placeholder,
                cha_cha_event,
                no_speech_placeholder,
                rumba_event,
            ],
        )

    def test_detect_empty_chunk_sequence_yields_nothing(self):
        detector = ballroom_scribe.EventDetector(
            ollama_model=_OLLAMA_MODEL,
            ollama_options=_OLLAMA_OPTIONS,
        )
        generator = detector.detect([], _CACHE_DIR)
        self.assertEqual(list(generator), [])


class TestEventDetectorProcessChunk(unittest.TestCase):
    """Unit tests for `EventDetector._process_chunk."""

    def setUp(self):
        self.chunk = ballroom_scribe.models.TranscriptChunk(
            start_samples=100,
            sample_rate=10,
            duration_samples=10,
            words=(),
        )
        self.path_stem = "0000000100-0000000110"
        self.event = ballroom_scribe.models.Event(
            level="Newcomer",
            style="Smooth",
            dance="Viennese Waltz",
            round="Final",
            announcement_quote="Let's welcome our newcomers",
            announcement_start_seconds=2345.67,
            speech_detected=True,
            event_detected=True,
        )
        self.response = "<fake JSON string>"

        self.mock_model_supports_thinking = self.enterContext(
            patch.object(ballroom_scribe.event_detector, "_model_supports_thinking", return_value=True)
        )

        self.event_detector = ballroom_scribe.EventDetector(
            ollama_model=_OLLAMA_MODEL,
            ollama_options=_OLLAMA_OPTIONS,
        )

        self.mock_invoke_model = self.enterContext(
            patch.object(self.event_detector, "_invoke_model", return_value=self.response),
        )
        self.mock_parse_response = self.enterContext(
            patch.object(
                ballroom_scribe.event_detector,
                "_parse_response",
                return_value=[self.event],
            ),
        )
        self.mock_open = mock_open()
        self.enterContext(patch("builtins.open", self.mock_open))
        self.mock_path_exists = self.enterContext(patch.object(pathlib.Path, "exists"))

        self.mock_atomic_path = self.enterContext(patch.object(ballroom_scribe.utils, "atomic_path", autospec=True))
        self.partial_response_output = MagicMock(spec=pathlib.Path)
        self.mock_atomic_path.return_value.__enter__.return_value = self.partial_response_output

    def test_cache_hit(self):
        self.mock_path_exists.return_value = True
        self.mock_open.return_value.read.return_value = self.response

        events = self.event_detector._process_chunk(self.chunk, _CACHE_DIR)

        self.mock_open.assert_called_once_with(_CACHE_DIR / (self.path_stem + ".json"), "r", encoding="utf-8")
        self.mock_open().read.assert_called_once_with()
        self.mock_parse_response.assert_called_once_with(self.response, self.chunk.words)
        self.mock_invoke_model.assert_not_called()
        self.mock_atomic_path.assert_not_called()
        self.assertEqual(events, [self.event])

    def test_cache_miss(self):
        self.mock_path_exists.return_value = False

        events = self.event_detector._process_chunk(self.chunk, _CACHE_DIR)

        self.mock_atomic_path.assert_called_once_with(_CACHE_DIR / (self.path_stem + ".json"))
        self.mock_open.assert_called_once_with(self.partial_response_output, "w", encoding="utf-8")
        self.mock_invoke_model.assert_called_once_with(self.chunk, _CACHE_DIR / (self.path_stem + ".md"))
        self.mock_open.return_value.write.assert_called_once_with(self.response)
        self.mock_parse_response.assert_called_once_with(self.response, self.chunk.words)
        self.assertEqual(events, [self.event])

    def test_cache_miss_with_empty_response_does_not_write_cache(self):
        self.mock_path_exists.return_value = False
        self.mock_invoke_model.return_value = ""

        events = self.event_detector._process_chunk(self.chunk, _CACHE_DIR)

        self.mock_atomic_path.assert_not_called()
        self.mock_open.assert_not_called()
        self.mock_parse_response.assert_called_once_with("", self.chunk.words)
        self.assertEqual(events, [self.event])


class TestEventDetectorInvokeModel(unittest.TestCase):
    """Unit tests for `EventDetector._invoke_model` and Ollama interaction."""

    def setUp(self):
        self.thinking_output = _CACHE_DIR / "fake_path_stem.md"

        self.mock_open = mock_open()
        self.enterContext(patch("builtins.open", self.mock_open))

        self.mock_chat = self.enterContext(patch.object(ollama, "chat"))
        self.mock_parse_response = self.enterContext(patch.object(ballroom_scribe.event_detector, "_parse_response"))

        self.mock_model_supports_thinking = self.enterContext(
            patch.object(ballroom_scribe.event_detector, "_model_supports_thinking", return_value=True)
        )

        self.detector = ballroom_scribe.EventDetector(
            ollama_model=_OLLAMA_MODEL,
            ollama_options=_OLLAMA_OPTIONS,
        )
        self.chunk = tests.helpers.transcript_chunk.Maker(start_samples=120).make(
            [
                "Competitors",
                "on",
                "the",
                "floor",
                "for",
                "Silver",
                "Standard",
                "Quickstep",
                "Final.",
            ],
        )

    def test_processes_content(self):
        expected_content = ["<fake ", "JS", "ON string", ">"]
        self.mock_chat.return_value = []
        for piece in expected_content:
            mock_response = MagicMock()
            mock_response.message.content = piece
            mock_response.message.thinking = None
            self.mock_chat.return_value.append(mock_response)

        response = self.detector._invoke_model(self.chunk, self.thinking_output)

        self.mock_chat.assert_called_once_with(
            model=_OLLAMA_MODEL,
            messages=ANY,
            stream=True,
            think=ANY,
            format=ballroom_scribe.event_detector._SCHEMA,
            options=_OLLAMA_OPTIONS,
            keep_alive=ANY,
        )
        self.assertEqual(response, "".join(expected_content))

    def test_processes_thinking(self):
        expected_thinking = ["I am ", "an AI; ", "I ", "am think", "ing!"]
        self.mock_chat.return_value = []
        for piece in expected_thinking:
            mock_response = MagicMock()
            mock_response.message.content = None
            mock_response.message.thinking = piece
            self.mock_chat.return_value.append(mock_response)

        self.detector._invoke_model(self.chunk, self.thinking_output)

        self.mock_open.assert_called_once_with(self.thinking_output, "w", encoding="utf-8")
        self.mock_open.return_value.write.assert_has_calls([call(piece) for piece in expected_thinking])


if __name__ == "__main__":
    unittest.main()
