"""Qt-free session state managing the videos loaded in the timestamp editor."""

import dataclasses
import pathlib
import re
from typing import TypeIs

import ballroom_scribe.cli.arguments
import ballroom_scribe.gui.models
import ballroom_scribe.utils

_CACHED_AUDIO_RE = re.compile(r"^Audio-Channel-(\d+)\.wav$")


def _is_loaded(video: ballroom_scribe.gui.models.SessionVideo) -> TypeIs[ballroom_scribe.gui.models.LoadedVideo]:
    """Returns whether `video`'s media opened (as opposed to an `OfflineVideo` placeholder)."""
    return isinstance(video, ballroom_scribe.gui.models.LoadedVideo)


class VideoSession:
    """Maintains an ordered collection of session videos keyed by `csv_file`.

    Entries are `LoadedVideo`s whose media opened, or `OfflineVideo`s for files the events CSV refers to that could
    not be opened; both keep their place in the order (the Video drop-down), and an offline entry can later be
    swapped for a loaded one in place with `replace`.
    """

    def __init__(self) -> None:
        self._videos: dict[pathlib.Path, ballroom_scribe.gui.models.SessionVideo] = {}

    @property
    def videos(self) -> tuple[ballroom_scribe.gui.models.SessionVideo, ...]:
        """Returns every session video, loaded or offline, in insertion order."""
        return tuple(self._videos.values())

    @property
    def loaded_videos(self) -> tuple[ballroom_scribe.gui.models.LoadedVideo, ...]:
        """Returns only the videos whose media could be opened, in insertion order."""
        return tuple(video for video in self._videos.values() if _is_loaded(video))

    def __len__(self) -> int:
        return len(self._videos)

    def __bool__(self) -> bool:
        return bool(self._videos)

    def get(self, csv_file: pathlib.Path) -> ballroom_scribe.gui.models.SessionVideo | None:
        """Returns the session video keyed by `csv_file`, or `None` if not present."""
        return self._videos.get(csv_file)

    @staticmethod
    def infer_channel(input_video: pathlib.Path, parent_cache_dir: pathlib.Path) -> int:
        """Infers the audio channel to select by default for `input_video`.

        Returns the channel index of the single `Audio-Channel-*.wav` already present in the video's cache directory,
        or `0` when none or multiple channels are cached.
        """
        cache_dir = ballroom_scribe.utils.cache_dir_for_video(
            parent_cache_dir=parent_cache_dir,
            input_video=input_video,
        )
        if not cache_dir.exists():
            return 0
        channels: list[int] = []
        for wav_path in cache_dir.glob("Audio-Channel-*.wav"):
            if (match := _CACHED_AUDIO_RE.match(wav_path.name)) is not None:
                channels.append(int(match.group(1)))
        return channels[0] if len(channels) == 1 else 0

    def add(
        self,
        input_video: pathlib.Path,
        input_channel: int,
        parent_cache_dir: pathlib.Path,
        *,
        csv_file: pathlib.Path | None = None,
    ) -> ballroom_scribe.gui.models.LoadedVideo:
        """Validates and adds a video to the session.

        An offline entry keyed by the same `csv_file` is replaced in place (keeping its position).

        Args:
            input_video: Path to the video file on disk.
            input_channel: Zero-based audio channel index.
            parent_cache_dir: Parent directory for the video's `<stem>-Ballroom_Cache` folder.
            csv_file: Path to store in the CSV `File` column. Defaults to `input_video.resolve()`.

        Returns:
            The newly added `LoadedVideo`.

        Raises:
            ValueError: If the video is already loaded, its cache directory collides with another loaded video, the
                video has no audio stream, or `input_channel` is out of range.
        """
        loaded = self.probe(input_video, input_channel, parent_cache_dir, csv_file=csv_file)
        self._videos[loaded.csv_file] = loaded
        return loaded

    def probe(
        self,
        input_video: pathlib.Path,
        input_channel: int,
        parent_cache_dir: pathlib.Path,
        *,
        csv_file: pathlib.Path | None = None,
    ) -> ballroom_scribe.gui.models.LoadedVideo:
        """Validates a video against the session and returns the `LoadedVideo` that `add` would store, without adding it.

        Args:
            input_video: Path to the video file on disk.
            input_channel: Zero-based audio channel index.
            parent_cache_dir: Parent directory for the video's `<stem>-Ballroom_Cache` folder.
            csv_file: Path to store in the CSV `File` column. Defaults to `input_video.resolve()`.

        Raises:
            ValueError: If the video is already loaded, its cache directory collides with another loaded video, the
                video has no audio stream, or `input_channel` is out of range.
        """
        resolved_video = input_video.resolve()
        target_csv_file = csv_file if csv_file is not None else resolved_video

        existing = self._videos.get(target_csv_file)
        if (existing is not None and _is_loaded(existing)) or any(
            loaded.input_video == resolved_video for loaded in self.loaded_videos
        ):
            raise ValueError(f"Video {input_video} is already loaded in the session.")

        cache_dir = ballroom_scribe.utils.cache_dir_for_video(
            parent_cache_dir=parent_cache_dir,
            input_video=input_video,
        )
        self._check_cache_dir_collision(input_video=input_video, cache_dir=cache_dir)

        channel_count = ballroom_scribe.utils.count_audio_channels(resolved_video)
        if channel_count <= 0:
            raise ValueError(f"Video {input_video} has no audio stream.")
        if not 0 <= input_channel < channel_count:
            raise ValueError(
                f"Audio channel index {input_channel} is out of range"
                f" (found {channel_count} channel(s) in {input_video})."
            )

        return ballroom_scribe.gui.models.LoadedVideo(
            input_video=resolved_video,
            input_channel=input_channel,
            parent_cache_dir=parent_cache_dir,
            cache_dir=cache_dir,
            csv_file=target_csv_file,
            channel_count=channel_count,
        )

    def add_offline(
        self,
        csv_file: pathlib.Path,
        input_video: pathlib.Path,
        reason: str,
    ) -> ballroom_scribe.gui.models.OfflineVideo:
        """Records a video the CSV refers to whose media could not be opened, keeping its place in the order.

        Args:
            csv_file: Path as written in the CSV's `File` column.
            input_video: Where the file was expected on disk.
            reason: Why it could not be opened.

        Raises:
            ValueError: If a loaded video already owns `csv_file`.
        """
        existing = self._videos.get(csv_file)
        if existing is not None and _is_loaded(existing):
            raise ValueError(f"Video {csv_file} is loaded; it cannot be marked offline.")
        offline = ballroom_scribe.gui.models.OfflineVideo(csv_file=csv_file, input_video=input_video, reason=reason)
        self._videos[csv_file] = offline
        return offline

    def replace(self, csv_file: pathlib.Path, entry: ballroom_scribe.gui.models.SessionVideo) -> None:
        """Swaps the entry keyed by `csv_file` for `entry` (keyed by `entry.csv_file`) at the same position.

        Raises:
            KeyError: If no entry is keyed by `csv_file`.
            ValueError: If `entry.csv_file` already keys a different entry.
        """
        if csv_file not in self._videos:
            raise KeyError(csv_file)
        if entry.csv_file != csv_file and entry.csv_file in self._videos:
            raise ValueError(f"Video {entry.csv_file} is already in the session.")
        self._videos = {
            (entry.csv_file if key == csv_file else key): (entry if key == csv_file else value)
            for key, value in self._videos.items()
        }

    def offline_match(self, video_path: pathlib.Path) -> ballroom_scribe.gui.models.OfflineVideo | None:
        """Returns the offline entry whose expected file has `video_path`'s name, if exactly one does."""
        matches = [
            video
            for video in self._videos.values()
            if not _is_loaded(video) and video.input_video.name == video_path.name
        ]
        return matches[0] if len(matches) == 1 else None

    def set_channel(
        self,
        csv_file: pathlib.Path,
        input_channel: int,
    ) -> ballroom_scribe.gui.models.LoadedVideo:
        """Updates the selected audio channel for a loaded video.

        Raises:
            ValueError: If `input_channel` is out of range or the video is offline.
        """
        existing = self._loaded(csv_file)
        if not 0 <= input_channel < existing.channel_count:
            raise ValueError(
                f"Audio channel index {input_channel} is out of range"
                f" (found {existing.channel_count} channel(s) in {existing.input_video})."
            )
        updated = dataclasses.replace(existing, input_channel=input_channel)
        self._videos[csv_file] = updated
        return updated

    def set_parent_cache_dir(
        self,
        csv_file: pathlib.Path,
        parent_cache_dir: pathlib.Path,
    ) -> ballroom_scribe.gui.models.LoadedVideo:
        """Updates the parent cache directory for a loaded video.

        Raises:
            ValueError: If the new cache directory collides with another video's or the video is offline.
        """
        existing = self._loaded(csv_file)
        new_cache_dir = ballroom_scribe.utils.cache_dir_for_video(
            parent_cache_dir=parent_cache_dir,
            input_video=existing.input_video,
        )
        self._check_cache_dir_collision(
            input_video=existing.input_video,
            cache_dir=new_cache_dir,
            ignore_csv_file=csv_file,
        )
        updated = dataclasses.replace(
            existing,
            parent_cache_dir=parent_cache_dir,
            cache_dir=new_cache_dir,
        )
        self._videos[csv_file] = updated
        return updated

    def remove(self, csv_file: pathlib.Path) -> None:
        """Removes the video keyed by `csv_file` from the session."""
        del self._videos[csv_file]

    def clear(self) -> None:
        """Removes all videos from the session."""
        self._videos.clear()

    def find_for_record(
        self,
        file: pathlib.Path,
        csv_dir: pathlib.Path | None = None,
    ) -> ballroom_scribe.gui.models.SessionVideo | None:
        """Matches a CSV `File` path to a session video, loaded or offline.

        Checks exact `csv_file` equality first, then resolved paths (relative to `csv_dir` when `file` is relative,
        followed by `file.resolve()`), and finally a unique filename match.
        """
        if (exact := self._videos.get(file)) is not None:
            return exact

        candidate_paths: list[pathlib.Path] = []
        if not file.is_absolute() and csv_dir is not None:
            candidate_paths.append((csv_dir / file).resolve())
        candidate_paths.append(file.resolve())

        for candidate in candidate_paths:
            for video in self._videos.values():
                if video.input_video == candidate:
                    return video

        by_name = [
            video
            for video in self._videos.values()
            if video.input_video.name == file.name or video.csv_file.name == file.name
        ]
        if len(by_name) == 1:
            return by_name[0]
        return None

    def inputs(self) -> list[ballroom_scribe.cli.arguments.InputCacheArgs]:
        """Returns `InputCacheArgs` entries for the loaded videos, suitable for `PipelineRunner`."""
        return [
            ballroom_scribe.cli.arguments.InputCacheArgs(
                input_video=video.input_video,
                input_channel=video.input_channel,
                cache_dir=video.parent_cache_dir,
            )
            for video in self.loaded_videos
        ]

    def _loaded(self, csv_file: pathlib.Path) -> ballroom_scribe.gui.models.LoadedVideo:
        """Returns the loaded video keyed by `csv_file`, raising `ValueError` if it is offline."""
        existing = self._videos[csv_file]
        if not _is_loaded(existing):
            raise ValueError(f"Video {csv_file} is offline ({existing.reason}).")
        return existing

    def _check_cache_dir_collision(
        self,
        *,
        input_video: pathlib.Path,
        cache_dir: pathlib.Path,
        ignore_csv_file: pathlib.Path | None = None,
    ) -> None:
        resolved_cache_dir = cache_dir.resolve()
        for existing in self.loaded_videos:
            if ignore_csv_file is not None and existing.csv_file == ignore_csv_file:
                continue
            if existing.cache_dir.resolve() == resolved_cache_dir:
                raise ValueError(
                    f"{input_video} and {existing.input_video} would share the cache directory {cache_dir}."
                    " Give one of them a different parent cache directory."
                )
