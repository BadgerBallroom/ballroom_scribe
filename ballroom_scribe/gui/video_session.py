"""Qt-free session state managing the videos loaded in the timestamp editor."""

import dataclasses
import pathlib
import re

import ballroom_scribe.cli.arguments
import ballroom_scribe.gui.models
import ballroom_scribe.utils

_CACHED_AUDIO_RE = re.compile(r"^Audio-Channel-(\d+)\.wav$")


class VideoSession:
    """Maintains an ordered collection of `LoadedVideo` instances keyed by `csv_file`."""

    def __init__(self) -> None:
        self._videos: dict[pathlib.Path, ballroom_scribe.gui.models.LoadedVideo] = {}

    @property
    def videos(self) -> tuple[ballroom_scribe.gui.models.LoadedVideo, ...]:
        """Returns the loaded videos in insertion order."""
        return tuple(self._videos.values())

    def __len__(self) -> int:
        return len(self._videos)

    def __bool__(self) -> bool:
        return bool(self._videos)

    def get(self, csv_file: pathlib.Path) -> ballroom_scribe.gui.models.LoadedVideo | None:
        """Returns the `LoadedVideo` keyed by `csv_file`, or `None` if not present."""
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
        resolved_video = input_video.resolve()
        target_csv_file = csv_file if csv_file is not None else resolved_video

        if target_csv_file in self._videos or any(
            existing.input_video == resolved_video for existing in self._videos.values()
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

        loaded = ballroom_scribe.gui.models.LoadedVideo(
            input_video=resolved_video,
            input_channel=input_channel,
            parent_cache_dir=parent_cache_dir,
            cache_dir=cache_dir,
            csv_file=target_csv_file,
            channel_count=channel_count,
        )
        self._videos[target_csv_file] = loaded
        return loaded

    def set_channel(
        self,
        csv_file: pathlib.Path,
        input_channel: int,
    ) -> ballroom_scribe.gui.models.LoadedVideo:
        """Updates the selected audio channel for a loaded video."""
        existing = self._videos[csv_file]
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
        """Updates the parent cache directory for a loaded video."""
        existing = self._videos[csv_file]
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
    ) -> ballroom_scribe.gui.models.LoadedVideo | None:
        """Matches a CSV `File` path to a loaded video.

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
        """Returns `InputCacheArgs` entries for all loaded videos, suitable for `PipelineRunner`."""
        return [
            ballroom_scribe.cli.arguments.InputCacheArgs(
                input_video=video.input_video,
                input_channel=video.input_channel,
                cache_dir=video.parent_cache_dir,
            )
            for video in self._videos.values()
        ]

    def _check_cache_dir_collision(
        self,
        *,
        input_video: pathlib.Path,
        cache_dir: pathlib.Path,
        ignore_csv_file: pathlib.Path | None = None,
    ) -> None:
        resolved_cache_dir = cache_dir.resolve()
        for existing in self._videos.values():
            if ignore_csv_file is not None and existing.csv_file == ignore_csv_file:
                continue
            if existing.cache_dir.resolve() == resolved_cache_dir:
                raise ValueError(
                    f"{input_video} and {existing.input_video} would share the cache directory {cache_dir}."
                    " Give one of them a different parent cache directory."
                )
