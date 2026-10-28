import concurrent.futures
import logging
import pathlib
import subprocess
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

try:
    from PySide6 import QtCore
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.gui
import ballroom_scribe.gui.models
import ballroom_scribe.gui.utils
import ballroom_scribe.gui.waveform_loader
import ballroom_scribe.utils

_VIDEO = ballroom_scribe.gui.models.LoadedVideo(
    input_video=pathlib.Path("/videos/clip.mp4"),
    input_channel=1,
    parent_cache_dir=pathlib.Path("/videos"),
    cache_dir=pathlib.Path("/videos/clip-Ballroom_Cache"),
    csv_file=pathlib.Path("/videos/clip.mp4"),
    channel_count=2,
)
_OTHER_VIDEO = ballroom_scribe.gui.models.LoadedVideo(
    input_video=pathlib.Path("/videos/other.mp4"),
    input_channel=0,
    parent_cache_dir=pathlib.Path("/videos"),
    cache_dir=pathlib.Path("/videos/other-Ballroom_Cache"),
    csv_file=pathlib.Path("/videos/other.mp4"),
    channel_count=1,
)
_EXTRACTED_WAV = pathlib.Path("/videos/clip-Ballroom_Cache/Audio-Channel-1.wav")
_WAVEFORM_NPZ = pathlib.Path("/videos/clip-Ballroom_Cache/Waveform-Channel-1.npz")
_ENVELOPE = ballroom_scribe.gui.models.WaveformEnvelope(
    mins=np.array([-0.5], dtype=np.float32),
    maxs=np.array([0.5], dtype=np.float32),
    bins_per_second=100.0,
)


class TestPrepareEnvelope(unittest.TestCase):
    """Unit tests for `_prepare_envelope`."""

    def setUp(self):
        self.mock_mkdir = self.enterContext(patch.object(pathlib.Path, "mkdir", autospec=True))
        self.mock_exists = self.enterContext(patch.object(pathlib.Path, "exists", autospec=True))
        self.mock_extract_audio = self.enterContext(
            patch.object(ballroom_scribe.utils, "extract_audio", autospec=True, return_value=_EXTRACTED_WAV)
        )
        self.mock_compute_envelope = self.enterContext(
            patch.object(ballroom_scribe.gui.utils, "compute_waveform_envelope", autospec=True)
        )
        self.mock_np_load = self.enterContext(patch.object(np, "load", autospec=True))
        self.mock_np_savez = self.enterContext(patch.object(np, "savez", autospec=True))

        self.mock_atomic_path = self.enterContext(patch.object(ballroom_scribe.utils, "atomic_path", autospec=True))
        self.mock_partial_npz = MagicMock(spec=pathlib.Path)
        self.mock_atomic_path.return_value.__enter__.return_value = self.mock_partial_npz

    def test_loads_cached_npz_when_present(self):
        self.mock_exists.return_value = True
        npz_context = MagicMock()
        npz_context.__enter__.return_value = {
            "mins": np.array([-0.4, -0.2], dtype=np.float32),
            "maxs": np.array([0.4, 0.2], dtype=np.float32),
            "bins_per_second": np.float64(100.0),
        }
        self.mock_np_load.return_value = npz_context

        envelope = ballroom_scribe.gui.waveform_loader._prepare_envelope(_VIDEO)

        self.mock_mkdir.assert_called_once_with(_VIDEO.cache_dir, parents=True, exist_ok=True)
        self.mock_extract_audio.assert_called_once_with(
            input_video=_VIDEO.input_video,
            input_channel=_VIDEO.input_channel,
            cache_dir=_VIDEO.cache_dir,
            cancel=None,
        )
        self.mock_np_load.assert_called_once_with(_WAVEFORM_NPZ)
        self.mock_compute_envelope.assert_not_called()
        self.mock_np_savez.assert_not_called()
        self.mock_atomic_path.assert_not_called()
        self.assertEqual(
            envelope,
            ballroom_scribe.gui.models.WaveformEnvelope(
                mins=np.array([-0.4, -0.2], dtype=np.float32),
                maxs=np.array([0.4, 0.2], dtype=np.float32),
                bins_per_second=100.0,
            ),
        )

    def test_computes_and_saves_envelope_on_cache_miss(self):
        self.mock_exists.return_value = False
        self.mock_compute_envelope.return_value = _ENVELOPE

        envelope = ballroom_scribe.gui.waveform_loader._prepare_envelope(_VIDEO)

        self.assertEqual(envelope, _ENVELOPE)
        self.mock_compute_envelope.assert_called_once_with(_EXTRACTED_WAV)
        self.mock_atomic_path.assert_called_once_with(_WAVEFORM_NPZ)
        self.mock_np_savez.assert_called_once()
        self.assertEqual(self.mock_np_savez.call_args.args[0], self.mock_partial_npz)

    def test_passes_cancel_event_to_extraction(self):
        self.mock_exists.return_value = True
        self.mock_np_load.return_value.__enter__.return_value = {
            "mins": np.array([0.0], dtype=np.float32),
            "maxs": np.array([0.0], dtype=np.float32),
            "bins_per_second": np.float64(100.0),
        }
        cancel = threading.Event()

        ballroom_scribe.gui.waveform_loader._prepare_envelope(_VIDEO, cancel)

        self.assertIs(self.mock_extract_audio.call_args.kwargs["cancel"], cancel)

    def test_skips_envelope_computation_when_cancelled_after_extraction(self):
        self.mock_exists.return_value = False
        cancel = threading.Event()
        cancel.set()

        with self.assertRaises(concurrent.futures.CancelledError):
            ballroom_scribe.gui.waveform_loader._prepare_envelope(_VIDEO, cancel)

        self.mock_compute_envelope.assert_not_called()
        self.mock_atomic_path.assert_not_called()

    def test_propagates_extraction_failure(self):
        self.mock_extract_audio.side_effect = subprocess.CalledProcessError(returncode=1, cmd=["ffmpeg"])

        with self.assertRaises(subprocess.CalledProcessError):
            ballroom_scribe.gui.waveform_loader._prepare_envelope(_VIDEO)


class _DeferredThreadPool:
    """Stand-in for `QThreadPool` that records runnables so tests decide when (and whether) they run."""

    def __init__(self) -> None:
        self.started: list[QtCore.QRunnable] = []
        self.wait_result = True
        self.wait_timeouts_ms: list[int] = []

    def start(self, runnable: QtCore.QRunnable) -> None:
        self.started.append(runnable)

    def waitForDone(self, timeout_ms: int) -> bool:
        self.wait_timeouts_ms.append(timeout_ms)
        return self.wait_result


class TestWaveformLoader(unittest.TestCase):
    """Unit tests for `WaveformLoader` with a deferred thread pool and a mocked `_prepare_envelope`."""

    def setUp(self):
        self.mock_prepare_envelope = self.enterContext(
            patch.object(
                ballroom_scribe.gui.waveform_loader, "_prepare_envelope", autospec=True, return_value=_ENVELOPE
            )
        )
        self.pool = _DeferredThreadPool()
        self.loader = ballroom_scribe.gui.WaveformLoader(thread_pool=self.pool)
        self.loaded: list[tuple[object, object]] = []
        self.failed: list[tuple[object, str]] = []
        self.loader.loaded.connect(lambda video, envelope: self.loaded.append((video, envelope)))
        self.loader.failed.connect(lambda video, message: self.failed.append((video, message)))

    def _run_started_task(self, index: int = -1) -> None:
        self.pool.started[index].run()

    def test_emits_loaded_for_current_video(self):
        self.loader.load(_VIDEO)
        self._run_started_task()

        self.assertEqual(self.loaded, [(_VIDEO, _ENVELOPE)])
        self.assertTrue(self.loader.is_idle)
        task = self.pool.started[0]
        self.mock_prepare_envelope.assert_called_once_with(_VIDEO, task.cancel)

    def test_drops_result_of_superseded_video(self):
        self.loader.load(_VIDEO)
        self.loader.load(_OTHER_VIDEO)
        self._run_started_task(0)  # The stale load still finishes (and warms its cache) but is not reported.
        self._run_started_task(1)

        self.assertEqual(self.loaded, [(_OTHER_VIDEO, _ENVELOPE)])

    def test_reuses_in_flight_load_of_same_video(self):
        self.loader.load(_VIDEO)
        self.loader.load(_OTHER_VIDEO)
        self.loader.load(_VIDEO)  # Switching back re-adopts the first task instead of starting a second writer.

        self.assertEqual([task.video for task in self.pool.started], [_VIDEO, _OTHER_VIDEO])
        self._run_started_task(0)
        self.assertEqual(self.loaded, [(_VIDEO, _ENVELOPE)])

    def test_reports_failure_for_current_video(self):
        self.mock_prepare_envelope.side_effect = OSError("disk full")
        self.loader.load(_VIDEO)
        self._run_started_task()

        self.assertEqual(self.failed, [(_VIDEO, "disk full")])
        self.assertEqual(self.loaded, [])

    def test_cancel_sets_task_event_and_silences_result(self):
        self.loader.load(_VIDEO)
        task = self.pool.started[0]

        self.loader.cancel()

        self.assertTrue(task.cancel.is_set())
        self.mock_prepare_envelope.side_effect = concurrent.futures.CancelledError()
        self._run_started_task()
        self.assertEqual(self.loaded, [])
        self.assertEqual(self.failed, [])
        self.assertTrue(self.loader.is_idle)

    def test_shutdown_cancels_and_waits_for_workers(self):
        self.loader.load(_VIDEO)
        task = self.pool.started[0]

        self.assertTrue(self.loader.shutdown(timeout_seconds=2.5))

        self.assertTrue(task.cancel.is_set())
        self.assertEqual(self.pool.wait_timeouts_ms, [2500])

    def test_shutdown_warns_when_workers_do_not_finish(self):
        self.pool.wait_result = False
        self.loader.load(_VIDEO)

        with self.assertLogs(ballroom_scribe.gui.waveform_loader.__name__, level=logging.WARNING):
            self.assertFalse(self.loader.shutdown(timeout_seconds=0.1))


if __name__ == "__main__":
    unittest.main()
