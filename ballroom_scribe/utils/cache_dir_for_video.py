import pathlib


def cache_dir_for_video(parent_cache_dir: pathlib.Path, input_video: pathlib.Path) -> pathlib.Path:
    """Returns the cache subdirectory for a given input video file.

    Args:
        parent_cache_dir: The parent directory in which video cache directories are stored.
        input_video: The path to the input video file.

    Returns:
        The path to the cache directory for `input_video`.
    """
    return parent_cache_dir / f"{input_video.stem}-Ballroom_Cache"
