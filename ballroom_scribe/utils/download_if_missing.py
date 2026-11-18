"""Utility for atomically downloading a file when it is missing or incomplete."""

import pathlib
import urllib.request

import ballroom_scribe.utils

_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def download_if_missing(url: str, destination: pathlib.Path, *, min_bytes: int = 1) -> None:
    """Downloads `url` to `destination` atomically if `destination` is missing or smaller than `min_bytes`.

    The download is streamed to a sibling `.partial` file via `ballroom_scribe.utils.atomic_path` and renamed into
    place only once the response completes and satisfies both the `Content-Length` header (when present) and
    `min_bytes`, so an interrupted or truncated download never leaves an incomplete file at `destination`.

    Args:
        url: The HTTP or HTTPS URL to download from.
        destination: The final path at which to save the downloaded file. Parent directories are created if needed.
        min_bytes: Minimum valid file size in bytes. An existing file smaller than `min_bytes` is re-downloaded, and a
            completed download smaller than `min_bytes` is rejected.

    Raises:
        ValueError: If `min_bytes` is negative.
        OSError: If downloading or writing fails (including `urllib.error.URLError` and `urllib.error.HTTPError`), or
            if the downloaded payload is truncated relative to `Content-Length` or smaller than `min_bytes`.
    """
    if min_bytes < 0:
        raise ValueError(f"min_bytes must be non-negative, got {min_bytes}.")

    if destination.is_file() and destination.stat().st_size >= min_bytes:
        return

    print(f"Downloading {destination} from {url}...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ballroom_scribe.utils.atomic_path(destination) as partial_path:
        written_bytes = 0
        with urllib.request.urlopen(url) as response, open(partial_path, "wb") as out_file:
            expected_length = response.headers.get("Content-Length")
            while chunk := response.read(_DOWNLOAD_CHUNK_BYTES):
                out_file.write(chunk)
                written_bytes += len(chunk)

        if expected_length is not None and written_bytes < int(expected_length):
            raise OSError(f"Download of {url} was truncated: got {written_bytes} of {int(expected_length)} bytes.")
        if written_bytes < min_bytes:
            raise OSError(
                f"Downloaded file from {url} is unexpectedly small ({written_bytes} bytes; expected at least"
                f" {min_bytes})."
            )
