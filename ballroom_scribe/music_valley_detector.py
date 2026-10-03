"""Detects transitions (music valleys) between ballroom dance songs and extracts speech segments."""

import dataclasses
from collections.abc import Generator, Iterable, Sequence

import ballroom_scribe.models


@dataclasses.dataclass(frozen=True)
class _PeeledDance:
    """Represents a confirmed dance with boundary fade-in and fade-out speech separated from its core.

    Attributes:
        leading_speech: Consecutive chunks at the start of the dance that contain speech (e.g., emcee speaking over
            a music fade-in), to be appended to the preceding music valley.
        core_start_samples: Start sample of the dance's core, after any `leading_speech` has been removed.
        core_end_samples: End sample of the dance's core, before any `trailing_speech` begins.
        sample_rate: Sample rate of the dance's audio chunks in samples per second.
        trailing_speech: Consecutive chunks at the end of the dance that contain speech (e.g., emcee speaking over
            a music fade-out), to be prepended to the following music valley.
    """

    leading_speech: tuple[ballroom_scribe.models.AudioChunk, ...]
    core_start_samples: int
    core_end_samples: int
    sample_rate: int
    trailing_speech: tuple[ballroom_scribe.models.AudioChunk, ...]


def _extract_speech_chunks(
    chunks: Iterable[ballroom_scribe.models.AudioChunk],
) -> Generator[ballroom_scribe.models.AudioChunk]:
    """Yields chunks with the `SPEECH` bit set, normalized to `Classification.SPEECH`."""
    for chunk in chunks:
        if chunk.classification_is(ballroom_scribe.models.AudioChunk.Classification.SPEECH):
            yield dataclasses.replace(
                chunk,
                classification=ballroom_scribe.models.AudioChunk.Classification.SPEECH,
            )


def _merge_adjacent_chunks(
    chunks: Iterable[ballroom_scribe.models.AudioChunk],
) -> Generator[ballroom_scribe.models.AudioChunk]:
    """Merges temporally contiguous `AudioChunk`s of matching classification."""
    iterator = iter(chunks)
    try:
        current = next(iterator)
    except StopIteration:
        return

    for chunk in iterator:
        if current.end_samples == chunk.start_samples:
            # `AudioChunk.__add__` calls `np.concatenate`, which automatically creates a detached copy.
            current = current + chunk
        else:
            yield current
            current = chunk

    yield current


def _finalize_valley_speech(
    chunks: Iterable[ballroom_scribe.models.AudioChunk],
) -> tuple[ballroom_scribe.models.AudioChunk, ...]:
    """Gets `AudioChunk`s ready to be inserted into a `MusicValley`.

    Extracts chunks with the `SPEECH` bit set, merges adjacent speech chunks (e.g. `SPEECH` + `SPEECH|MUSIC`), and
    detaches them via `.copy()` so that parent 10-minute audio buffers can be garbage-collected.
    """
    speech_chunks = _extract_speech_chunks(chunks)
    merged_chunks = _merge_adjacent_chunks(speech_chunks)
    return tuple(chunk.copy_if_sharing_memory() for chunk in merged_chunks)


class MusicValleyDetector:
    """Detects transitions (music valleys) between dances and extracts their speech segments.

    A music valley is the transition interval between two competition dances (or before the first dance and after the
    last dance) where sustained dance music drops out. If competition dances are visualized on a timeline as plateaus
    of continuous music lasting at least `min_dance_seconds`, the spaces between those plateaus form valleys where the
    emcee makes announcements and couples enter or exit the floor.

    To accurately isolate these valleys from a stream of classified `AudioChunk`s, this detector:
    1.  Bridges brief non-music dropouts (`max_song_dip_seconds`) inside a song so dramatic pauses do not split a single
        dance into multiple plateaus.
    2.  Treats music segments shorter than `min_dance_seconds` (such as walk-on or bumper music) as part of the valley
        rather than as competition dances.
    3.  Peels overlapping speech (`SPEECH | MUSIC` up to `max_fade_speech_seconds`) from the leading fade-in and
        trailing fade-out edges of confirmed dances so emcee announcements over fading music are preserved in the
        adjacent valley.
    4.  Retains only detached copies of the valley's `SPEECH` chunks so the underlying multi-minute audio buffers for
        music and silence can be garbage-collected immediately.
    """

    def __init__(
        self,
        min_dance_seconds: float,
        max_song_dip_seconds: float,
        max_fade_speech_seconds: float,
        max_dip_speech_seconds: float,
    ):
        """Initializes the instance.

        Args:
            min_dance_seconds: Minimum cumulative duration in seconds required for a music run to be classified as a
                competition dance. Shorter music runs (such as brief walk-on or bumper music) are treated as part of
                the surrounding music valley.
            max_song_dip_seconds: Maximum cumulative duration in seconds of a non-music gap (such as a dramatic
                pause in a song) that can be bridged into an active music run if music resumes.
            max_fade_speech_seconds: Maximum cumulative duration in seconds of overlapping speech and music at the
                leading (fade-in) or trailing (fade-out) edge of a dance to peel off and include in the adjacent
                music valley.
            max_dip_speech_seconds: Duration in seconds of speech within a non-music gap at or above which the gap
                is no longer considered a mid-song dip, forcing the active music run to end.
        """
        self._min_dance_seconds = min_dance_seconds
        self._max_song_dip_seconds = max_song_dip_seconds
        self._max_fade_speech_seconds = max_fade_speech_seconds
        self._max_dip_speech_seconds = max_dip_speech_seconds

    def detect(
        self,
        classified_chunks: Iterable[ballroom_scribe.models.AudioChunk],
    ) -> Generator[ballroom_scribe.models.MusicValley]:
        """Finds music valleys in a stream of `AudioChunk`s.

        Args:
            classified_chunks:
                An iterable of `AudioChunk`s with the `classification` property set to something other than `UNKNOWN`.

        Yields: The music valleys.
        """
        valley_speech_chunks: list[ballroom_scribe.models.AudioChunk] = []
        valley_start_samples = 0
        has_unyielded_valley = False
        last_chunk_end_samples = 0
        sample_rate = 0

        for item in self._segment_stream(classified_chunks):
            sample_rate = item.sample_rate
            if isinstance(item, _PeeledDance):
                valley_speech_chunks.extend(item.leading_speech)
                if item.core_start_samples > valley_start_samples:
                    yield ballroom_scribe.models.MusicValley(
                        start_samples=valley_start_samples,
                        sample_rate=sample_rate,
                        duration_samples=item.core_start_samples - valley_start_samples,
                        speech_chunks=_finalize_valley_speech(valley_speech_chunks),
                    )
                # Detach trailing fade-out speech immediately so the dance's 10-min buffer can be freed
                valley_speech_chunks = [chunk.copy() for chunk in item.trailing_speech]
                valley_start_samples = item.core_end_samples
                last_chunk_end_samples = (
                    item.trailing_speech[-1].end_samples if item.trailing_speech else item.core_end_samples
                )
                has_unyielded_valley = bool(item.trailing_speech)
            else:
                has_unyielded_valley = True
                last_chunk_end_samples = item.end_samples
                if item.classification_is(ballroom_scribe.models.AudioChunk.Classification.SPEECH):
                    # Detach speech chunk immediately; non-speech chunks are dropped here, so RAM is freed!
                    valley_speech_chunks.append(item.copy())

        if has_unyielded_valley:
            yield ballroom_scribe.models.MusicValley(
                start_samples=valley_start_samples,
                sample_rate=sample_rate,
                duration_samples=last_chunk_end_samples - valley_start_samples,
                speech_chunks=_finalize_valley_speech(valley_speech_chunks),
            )

    def _segment_stream(
        self,
        classified_chunks: Iterable[ballroom_scribe.models.AudioChunk],
    ) -> Generator[ballroom_scribe.models.AudioChunk | _PeeledDance]:
        """Groups contiguous music runs and yields valley `AudioChunk`s interleaved with `_PeeledDance`s."""
        current_music_run: list[ballroom_scribe.models.AudioChunk] = []
        pending_dip: list[ballroom_scribe.models.AudioChunk] = []

        for chunk in classified_chunks:
            if chunk.classification_is(ballroom_scribe.models.AudioChunk.Classification.MUSIC):
                if pending_dip:
                    # Music resumed while the dip was still valid: bridge the dip into the music run.
                    current_music_run.extend(pending_dip)
                    pending_dip.clear()
                current_music_run.append(chunk)
                continue

            if current_music_run:
                pending_dip.append(chunk)
                if self._is_midsong_dip(pending_dip):
                    continue

                # The gap exceeded dip limits: the music run truly ended before `pending_dip`!
                yield from self._classify_music_run(current_music_run)
                current_music_run.clear()
                yield from pending_dip
                pending_dip.clear()
                continue

            yield chunk

        # Flush end of stream
        if current_music_run:
            yield from self._classify_music_run(current_music_run)
        if pending_dip:
            yield from pending_dip

    def _is_midsong_dip(self, chunks: Sequence[ballroom_scribe.models.AudioChunk]) -> bool:
        """Returns whether the cumulative non-music gap is still short enough to be a mid-song dip."""
        # Bridge brief mid-song dips (< 4.5s) unless they contain >= 2s of speech
        total_duration = sum(c.duration_seconds for c in chunks)
        if total_duration > self._max_song_dip_seconds:
            return False

        speech_duration = sum(
            chunk.duration_seconds
            for chunk in chunks
            if chunk.classification_is(ballroom_scribe.models.AudioChunk.Classification.SPEECH)
        )
        return speech_duration < self._max_dip_speech_seconds

    def _classify_music_run(
        self,
        music_run: Sequence[ballroom_scribe.models.AudioChunk],
    ) -> Generator[ballroom_scribe.models.AudioChunk | _PeeledDance]:
        """Yields a `_PeeledDance` if the run is a full dance, or yields its chunks as valley audio otherwise."""
        if self._is_dance_run(music_run):
            yield self._peel_fade_edges(music_run)
        else:
            # Short walk-on/bumper music (< 35s) stays in the valley
            yield from music_run

    def _is_dance_run(self, chunks: Sequence[ballroom_scribe.models.AudioChunk]) -> bool:
        """Returns whether a music run meets the minimum duration required to be considered a dance."""
        return sum(chunk.duration_seconds for chunk in chunks) >= self._min_dance_seconds

    def _peel_fade_edges(self, dance_chunks: Sequence[ballroom_scribe.models.AudioChunk]) -> _PeeledDance:
        """Separates overlapping speech at the fade-in and fade-out edges of a dance from its inner core.

        DJs frequently fade music in while the emcee is still introducing the heat, or fade music out while the emcee is
        already thanking the couples. Because those boundary segments have both `MUSIC` and `SPEECH` bits set, they are
        initially grouped into `dance_chunks`.

        This function "peels" those outer speech-containing chunks off the start and end of `dance_chunks`, scanning
        inward from both edges until it encounters a chunk without `SPEECH` or reaches `max_fade_speech_seconds`, so
        that they can be reassigned to the adjacent music valleys. Any speech chunks that occur in the interior of the
        dance (such as mid-heat time calls or sung vocals) remain inside the core and are ignored.

        Args:
            dance_chunks: The contiguous sequence of `AudioChunk`s making up a confirmed dance run.

        Returns:
            A `_PeeledDance` containing the peeled leading and trailing speech chunks along with the sample boundaries
            of the remaining core dance.
        """
        start_index = self._count_leading_fade_chunks(dance_chunks)
        trailing_count = self._count_leading_fade_chunks(reversed(dance_chunks[start_index:]))
        end_index = len(dance_chunks) - trailing_count

        core_start = (
            dance_chunks[start_index].start_samples
            if start_index < len(dance_chunks)
            else dance_chunks[0].start_samples
        )
        core_end = dance_chunks[end_index - 1].end_samples if end_index > start_index else dance_chunks[-1].end_samples

        return _PeeledDance(
            leading_speech=tuple(dance_chunks[:start_index]),
            core_start_samples=core_start,
            core_end_samples=core_end,
            sample_rate=dance_chunks[0].sample_rate,
            trailing_speech=tuple(dance_chunks[end_index:]),
        )

    def _count_leading_fade_chunks(self, chunks: Iterable[ballroom_scribe.models.AudioChunk]) -> int:
        """Counts consecutive leading `SPEECH` chunks whose cumulative duration is within `max_fade_speech_seconds`."""
        total_duration = 0.0
        count = 0
        for chunk in chunks:
            if not chunk.classification_is(ballroom_scribe.models.AudioChunk.Classification.SPEECH):
                break
            if total_duration + chunk.duration_seconds > self._max_fade_speech_seconds:
                break
            total_duration += chunk.duration_seconds
            count += 1
        return count
