"""LLM-based ballroom competition event extraction using Ollama."""

import contextlib
import io
import json
import pathlib
import string
from collections.abc import Generator, Iterable, Mapping, Sequence
from enum import Enum
from typing import Any

import ollama

import ballroom_scribe
import ballroom_scribe.models
import ballroom_scribe.utils

_SYSTEM_PROMPT = """
You are a data extraction assistant for a ballroom dance competition. The spoken words from a video of a competition
have been transcribed into text. When you are given a chunk of the transcript, you must identify whether the emcee
announces any dance heats. For each event, extract the level, style, dance, round, and a quote of what the emcee
actually said:

*   The level is the skill level of the dancers, such as "Newcomer," "Bronze", "Silver", "Gold", "Novice", "Pre-Champ",
    "Championship," or "Syllabus." You may also encounter a "fun" or "showcase" level.
*   The style is the category of the type of dance, such as "Smooth," "Standard," "Rhythm," or "Latin."
*   The dance is the specific dance style being announced, such as "Waltz", "Tango", "Foxtrot," "Viennese Waltz,"
    "Quickstep," "Cha Cha," "Samba," "Rumba," "Jive," "Swing", or "Bolero."
    *   There often is more than one per event, in which case you should output all of them, separated by slashes (`/`).
    *   Especially for some of the more advanced levels like Gold and up, there might only be one multi-dance event for
        that level (e.g., "Gold Standard Semifinal"). In this case, the emcee won't mention a dance.
*   The round is the stage of the competition, such as "Preliminary", "Quarterfinal", "Semifinal", or "Final."
*   The quote is the exact sentence or phrase from the transcript that you are referencing.

The emcee may not always state every field. If a field is not mentioned, use `""`.

Ignore pure casual banter and judge introductions. However, if any phrase calls dancers to the floor or announces an
upcoming dance, it is not banter; extract it immediately.

If no announcements are found, return an empty list.

## Critical Policy - High Recall Over Precision

Transcription audio is noisy, and words will frequently be garbled. Bias heavily toward false positives over false
negatives:

1.  If the sentence structure indicates an announcement is occurring, extract the event.
2.  Do NOT validate whether the extracted dance/level/style exists in real life. If the speaker says "your first dance
    is the banana smoothie", you MUST extract "Banana smoothie" as the dance. Never reject an event because a field
    contains an unlisted or nonsensical word.
3.  Except for the known sound-alikes below, output the verbatim transcription words. Capitalize the first letter.

## Known ASR Sound-Alikes

| Transcribed  | Actual         |
|--------------|----------------|
| baby's waltz | Viennese waltz |
| broad        | bronze         |
| kick step    | quickstep      |
| lock drop    | foxtrot        |
| puddle       | couple         |
| puppet       | couple         |
| summer       | samba          |
| tangle       | tango          |
| Venus waltz  | Viennese waltz |
| walls        | waltz          |

## Announcement Cues (non-exhaustive)

*   "I'd like to call to the floor our nth heat..."
*   "Next is the <event>."
*   "We now return to event n, the <event>."
*   "Let's have a round of applause for the <event>."
*   "This is our <event>."
*   "This is the nth event of the day, the <event>"
*   "Dancers, find your places for the <event>."
*   "Judges, on the floor is the <event>."
*   "Your first / next dance is the <dance>."
*   "<event> to the floor."
""".strip()

_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "level": {"type": "string"},
                    "style": {"type": "string"},
                    "dance": {"type": "string"},
                    "round": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["level", "style", "dance", "round", "quote"],
            },
        },
    },
    "required": ["events"],
}

_EXAMPLES = (
    (
        "Judges, refresh your tablets. Newcomers, find a spot on the floor. Your first dance is the Boeing 747.",
        {
            "events": [
                {
                    "level": "Newcomer",
                    "style": "",
                    "dance": "Boeing 747",
                    "round": "Final",
                    "quote": "Newcomers, find a spot on the floor. Your first dance is the Boeing 747.",
                },
            ],
        },
    ),
    (
        "We now return to heat thirty-four, the Gold Standard Semifinal. Dancers, please take your places.",
        {
            "events": [
                {
                    "level": "Gold",
                    "style": "Standard",
                    "dance": "",
                    "round": "Semifinal",
                    "quote": "We now return to heat thirty-four, the Gold Standard Semifinal.",
                },
            ],
        },
    ),
    (
        "Moving along. We welcome to the floor the broad smooth Venus waltz quarter final.",
        {
            "events": [
                {
                    "level": "Bronze",
                    "style": "Smooth",
                    "dance": "Viennese waltz",
                    "round": "Quarterfinal",
                    "quote": "We welcome to the floor the broad smooth Venus waltz quarter final.",
                },
            ],
        },
    ),
    (
        "It's lunchtime. We will see you back here in one hour. Team captains, please see me up front.",
        {
            "events": [],
        },
    ),
)


class _ModelResponseMode(Enum):
    """Response modes from the LLM.

    Attributes:
        UNKNOWN: The mode is unknown.
        THINKING: The model is thinking.
        CONTENT: The model is generating is response.
    """

    UNKNOWN = 0
    THINKING = 1
    CONTENT = 2


def _chunk_to_path_stem(chunk: ballroom_scribe.models.TranscriptChunk) -> str:
    """Returns a string that can be used to name a file that corresponds to this chunk without a file extension."""
    return f"{chunk.start_time:08.2f}-{chunk.end_time:08.2f}"


def _strip_punctuation(s: str) -> str:
    """Removes punctuation from the given string."""
    return s.translate(str.maketrans("", "", string.punctuation))


def _align_quote(quote: str, words: Sequence[ballroom_scribe.models.Word]) -> float:
    """Finds the start time of the quote within the sequence of words.

    Args:
        quote: The quote to find within `words`.
        words: The text to find `quote` within.

    Returns:
        The start time of the quote.
    """
    index_in_words = ballroom_scribe.utils.find_closest_matching(
        needle=_strip_punctuation(quote).lower().split(),
        haystack=[_strip_punctuation(word.word).lower() for word in words],
    )
    return words[index_in_words].start


def _dict_to_event(d: dict, words: Sequence[ballroom_scribe.models.Word]) -> ballroom_scribe.models.Event:
    """Converts a `dict` representing one event from the LLM output to an `Event`.

    Args:
        d: The `dict`
        words: The transcribed text that the LLM was given.

    Returns:
        The event
    """
    quote = str(d.get("quote", ""))
    return ballroom_scribe.models.Event(
        level=str(d.get("level", "")),
        style=str(d.get("style", "")),
        dance=str(d.get("dance", "")),
        round=str(d.get("round", "")),
        announcement_quote=quote,
        announcement_start_seconds=_align_quote(quote=quote, words=words),
    )


def _parse_response(
    response_content: str,
    words: Sequence[ballroom_scribe.models.Word],
) -> list[ballroom_scribe.models.Event]:
    """Parses the LLM JSON response into a list of `Event`s if valid.

    Args:
        response_content: The response from the LLM.
        words: The transcribed text that the LLM was given.

    Returns:
        A list of the events, sorted by `announcement_start_seconds`.
    """
    if not response_content:
        print("The model returned an empty response.")
        return []

    try:
        model_output = json.loads(response_content)
    except json.JSONDecodeError as e:
        print("JSON decode error:", repr(response_content), e)
        return []

    if not isinstance(model_output, dict):
        print("Expected model output to be a dict, got", repr(model_output))
        return []

    events = model_output.get("events")
    if not isinstance(events, list):
        print("Expected events to be a list, got", repr(events))
        return []

    return sorted(
        (_dict_to_event(event, words) for event in events if isinstance(event, dict)),
        key=lambda event: event.announcement_start_seconds,
    )


def _model_supports_thinking(model: str) -> bool:
    """Queries Ollama for whether the specified model supports thinking."""
    capabilities = ollama.show(model).capabilities
    if not capabilities:
        return False
    return "thinking" in capabilities


class EventDetector:
    """Uses an LLM via Ollama to detect ballroom event announcements in transcripts."""

    def __init__(self, ollama_model: str, ollama_options: ollama.Options | Mapping[str, Any]):
        self._ollama_model = ollama_model
        self._ollama_options = ollama_options
        self._think = _model_supports_thinking(ollama_model)

    def detect(
        self,
        chunks_of_words: Iterable[ballroom_scribe.models.TranscriptChunk],
        cache_dir: pathlib.Path,
    ) -> Generator[ballroom_scribe.models.Event]:
        """Uses an LLM to extract information from a transcript.

        Args:
            chunks_of_words: The transcript to process, chunked by silence.
            cache_dir:
                The directory in which the LLM response will be cached in a .json file. If the model supports thinking,
                the thinking output will be stored in a .md file.

        Yields:
            The events that were found in the transcript.
        """
        for chunk in chunks_of_words:
            yield from self._process_chunk(chunk, cache_dir)

    def _process_chunk(
        self,
        chunk: ballroom_scribe.models.TranscriptChunk,
        cache_dir: pathlib.Path,
    ) -> list[ballroom_scribe.models.Event]:
        """Checks for a cached LLM response and if there isn't one, calls the LLM."""
        path_stem = _chunk_to_path_stem(chunk)
        response_output = cache_dir / (path_stem + ".json")

        if response_output.exists():
            with open(response_output, "r", encoding="utf-8") as f:
                return _parse_response(f.read(), chunk.words)

        response = self._invoke_model(chunk, cache_dir / (path_stem + ".md"))
        if response:
            with (
                ballroom_scribe.utils.atomic_path(response_output) as partial_response_output,
                open(partial_response_output, "w", encoding="utf-8") as f,
            ):
                f.write(response)

        return _parse_response(response, chunk.words)

    def _invoke_model(
        self,
        chunk: ballroom_scribe.models.TranscriptChunk,
        thinking_output: pathlib.Path,
    ) -> str:
        """Runs the LLM query on a transcript chunk and returns the response."""
        text = chunk.get_text()
        print("\nProcessing text:", text)

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for user, assistant in _EXAMPLES:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": json.dumps(assistant)})
        messages.append({"role": "user", "content": text})

        # The thinking output is only used for debugging, so it goes straight to the disk.
        # The content output is short enough that we can just buffer it in memory and write it to the disk a later.
        response_content_accumulator = io.StringIO()
        with open(thinking_output, "w", encoding="utf-8") if self._think else contextlib.nullcontext() as f_thinking:
            current_mode = _ModelResponseMode.UNKNOWN
            for response in ollama.chat(
                model=self._ollama_model,
                messages=messages,
                stream=True,
                think=self._think,
                format=_SCHEMA,
                options=self._ollama_options,
                keep_alive=30,
            ):
                if response.message.thinking:
                    if current_mode != _ModelResponseMode.THINKING:
                        print("Thinking:")
                        current_mode = _ModelResponseMode.THINKING
                    print(response.message.thinking, end="", flush=True)
                    if f_thinking:
                        f_thinking.write(response.message.thinking)
                    continue

                if response.message.content:
                    if current_mode != _ModelResponseMode.CONTENT:
                        print("\nResponse: ", end="")
                        current_mode = _ModelResponseMode.CONTENT
                    print(response.message.content, end="", flush=True)
                    response_content_accumulator.write(response.message.content)
                    continue
        print()

        return response_content_accumulator.getvalue()
