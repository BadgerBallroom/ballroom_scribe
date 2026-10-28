"""Background `QThreadPool` loader for extracting audio and caching `WaveformEnvelope`s."""

import concurrent.futures
import logging
import pathlib
import subprocess
import threading
import zipfile
from typing import override

import numpy as np
import soundfile
from PySide6 import QtCore

import ballroom_scribe.gui.models
import ballroom_scribe.gui.utils
import ballroom_scribe.utils

_LOGGER = logging.getLogger(__name__)
_MAX_CONCURRENT_LOADS = 2
_SHUTDOWN_TIMEOUT_SECONDS = 5.0


def _waveform_cache_path(video: ballroom_scribe.gui.models.LoadedVideo) -> pathlib.Path:
    return video.cache_dir / f"Waveform-Channel-{video.input_channel}.npz"


def _prepare_envelope(
    video: ballroom_scribe.gui.models.LoadedVideo,
    cancel: threading.Event | None = None,
) -> ballroom_scribe.gui.models.WaveformEnvelope:
    """Extracts the video's audio channel if needed and loads or computes its `WaveformEnvelope`.

    Args:
        video: The video whose waveform to prepare.
        cancel: Optional event that aborts the audio extraction and skips the envelope computation once set.

    Raises:
        concurrent.futures.CancelledError: If `cancel` was set before the envelope was ready.
    """
    video.cache_dir.mkdir(parents=True, exist_ok=True)
    extracted_wav = ballroom_scribe.utils.extract_audio(
        input_video=video.input_video,
        input_channel=video.input_channel,
        cache_dir=video.cache_dir,
        cancel=cancel,
    )
    cache_path = _waveform_cache_path(video)
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return ballroom_scribe.gui.models.WaveformEnvelope(
                mins=cached["mins"],
                maxs=cached["maxs"],
                bins_per_second=float(cached["bins_per_second"]),
            )

    if cancel is not None and cancel.is_set():
        raise concurrent.futures.CancelledError("Waveform computation was cancelled.")
    envelope = ballroom_scribe.gui.utils.compute_waveform_envelope(extracted_wav)
    with ballroom_scribe.utils.atomic_path(cache_path) as partial_cache_path:
        np.savez(
            partial_cache_path,
            mins=envelope.mins,
            maxs=envelope.maxs,
            bins_per_second=np.float64(envelope.bins_per_second),
        )
    return envelope


class _WorkerSignals(QtCore.QObject):
    """Signals emitted from a background `_WaveformTask` back to the GUI thread."""

    loaded = QtCore.Signal(object, object)
    failed = QtCore.Signal(object, str)
    cancelled = QtCore.Signal(object)


class _WaveformTask(QtCore.QRunnable):
    """Runnable task executing `_prepare_envelope` on a `QThreadPool` worker thread."""

    def __init__(self, video: ballroom_scribe.gui.models.LoadedVideo) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.video = video
        self.cancel = threading.Event()
        self.signals = _WorkerSignals()

    @override
    def run(self) -> None:
        try:
            envelope = _prepare_envelope(self.video, self.cancel)
        except concurrent.futures.CancelledError:
            self.signals.cancelled.emit(self)
            return
        except (
            ValueError,
            KeyError,
            subprocess.CalledProcessError,
            OSError,
            soundfile.LibsndfileError,
            zipfile.BadZipFile,
        ) as err:
            self.signals.failed.emit(self, str(err))
            return
        self.signals.loaded.emit(self, envelope)


class WaveformLoader(QtCore.QObject):
    """Loads `WaveformEnvelope`s asynchronously, reporting only results for the most recently requested video.

    Loads of videos that are no longer current keep running so their cache files get written, but a repeated request
    for a video whose load is still in flight re-adopts that load instead of starting a second one: cache files are
    written through `atomic_path`, which requires a single writer per file.
    """

    loaded = QtCore.Signal(object, object)
    failed = QtCore.Signal(object, str)

    def __init__(
        self,
        parent: QtCore.QObject | None = None,
        thread_pool: QtCore.QThreadPool | None = None,
    ) -> None:
        super().__init__(parent)
        if thread_pool is None:
            thread_pool = QtCore.QThreadPool(self)
            thread_pool.setMaxThreadCount(_MAX_CONCURRENT_LOADS)
        self._thread_pool = thread_pool
        self._current_video: ballroom_scribe.gui.models.LoadedVideo | None = None
        self._active_tasks: set[_WaveformTask] = set()

    @property
    def is_idle(self) -> bool:
        """Returns whether no load is in flight."""
        return not self._active_tasks

    def load(self, video: ballroom_scribe.gui.models.LoadedVideo) -> None:
        """Loads `video`'s waveform in the background, re-adopting an identical in-flight load if there is one."""
        self._current_video = video
        if any(task.video == video for task in self._active_tasks):
            return
        task = _WaveformTask(video)
        self._active_tasks.add(task)
        task.signals.loaded.connect(self._on_task_loaded)
        task.signals.failed.connect(self._on_task_failed)
        task.signals.cancelled.connect(self._on_task_cancelled)
        self._thread_pool.start(task)

    def cancel(self) -> None:
        """Forgets the current request and stops every in-flight load, killing its audio extraction."""
        self._current_video = None
        for task in self._active_tasks:
            task.cancel.set()

    def shutdown(self, timeout_seconds: float = _SHUTDOWN_TIMEOUT_SECONDS) -> bool:
        """Cancels in-flight loads and waits for their worker threads to finish.

        Args:
            timeout_seconds: Maximum time to wait for the workers.

        Returns:
            Whether all workers finished in time; otherwise a warning is logged and the caller may proceed anyway.
        """
        self.cancel()
        if self._thread_pool.waitForDone(round(timeout_seconds * 1000)):
            return True
        _LOGGER.warning("Background waveform loading did not stop within %.1f s.", timeout_seconds)
        return False

    def _on_task_loaded(
        self,
        task: _WaveformTask,
        envelope: ballroom_scribe.gui.models.WaveformEnvelope,
    ) -> None:
        self._active_tasks.discard(task)
        if task.video == self._current_video:
            self.loaded.emit(task.video, envelope)

    def _on_task_failed(self, task: _WaveformTask, message: str) -> None:
        self._active_tasks.discard(task)
        if task.video == self._current_video:
            self.failed.emit(task.video, message)

    def _on_task_cancelled(self, task: _WaveformTask) -> None:
        self._active_tasks.discard(task)
