"""Context manager for writing files atomically."""

import contextlib
import pathlib
from collections.abc import Generator


@contextlib.contextmanager
def atomic_path(final_path: pathlib.Path) -> Generator[pathlib.Path]:
    """Yields a temporary path to write to and renames it to `final_path` only if the `with` block succeeds.

    This guarantees that `final_path` never refers to a partially written file, so callers can treat its existence as
    proof that it was completely written (e.g., to skip regenerating a cache file).

    The temporary path is a sibling of `final_path` with `.partial` inserted before the suffix (e.g., `Audio.wav` ->
    `Audio.partial.wav`):

    *   It is in the same directory, so the final rename is atomic.
    *   It has the same suffix, so tools that infer the file format from the extension (e.g., FFmpeg and
        `demucs.audio.save_audio`) still work.
    *   It is deterministic, so a stale file left behind by a killed process is deleted on the next attempt instead of
        accumulating.

    If the body of the `with` statement raises any exception, including `KeyboardInterrupt`, the temporary file is
    deleted and `final_path` is left untouched.

    Because the temporary name is deterministic, callers must ensure only one writer targets a given `final_path` at
    a time: a second concurrent writer would delete (or, on Windows, fail to delete) the first writer's temporary
    file, and the first writer to finish would rename the other's partially written file into place. The GUI
    therefore de-duplicates in-flight waveform loads and stops its own extraction before launching the pipeline.

    Args:
        final_path: The path that the file should have once it is completely written.

    Yields:
        The temporary path that the body of the `with` statement should write the file to.

    Raises:
        OSError: If the temporary file can't be renamed to `final_path`, such as `FileNotFoundError` if the body of the
            `with` statement didn't create it. The temporary file is left in place in this case.
    """
    partial_path = final_path.with_stem(f"{final_path.stem}.partial")
    partial_path.unlink(missing_ok=True)

    try:
        yield partial_path
    except BaseException:
        # If the file can't be deleted (e.g., a killed subprocess still has it open on Windows), it is harmless because
        # it never gets the final name, so don't mask the original exception.
        with contextlib.suppress(OSError):
            partial_path.unlink(missing_ok=True)
        raise

    partial_path.replace(final_path)
