import faulthandler
import unittest
from unittest.mock import MagicMock, call, patch

from parameterized import param, parameterized

try:
    from PySide6 import QtCore
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.gui

_FREEZE_SECONDS = 10.0


class TestFreezeWatchdog(unittest.TestCase):
    """Unit tests for `FreezeWatchdog` with a mocked `QTimer` and a mocked `faulthandler`."""

    def setUp(self):
        self.mock_timer_cls = self.enterContext(patch.object(QtCore, "QTimer", autospec=True))
        self.mock_timer = self.mock_timer_cls.return_value
        self.mock_timer.timeout = MagicMock()
        self.mock_dump_traceback_later = self.enterContext(
            patch.object(faulthandler, "dump_traceback_later", autospec=True)
        )
        self.mock_cancel_dump_traceback_later = self.enterContext(
            patch.object(faulthandler, "cancel_dump_traceback_later", autospec=True)
        )
        self.log = MagicMock()
        self.watchdog = ballroom_scribe.gui.FreezeWatchdog(self.log, freeze_seconds=_FREEZE_SECONDS)

    def _beat(self) -> None:
        """Does what the heartbeat timer does every time it fires."""
        (on_timeout,), _ = self.mock_timer.timeout.connect.call_args
        on_timeout()

    def test_heart_beats_several_times_per_freeze_time(self):
        self.mock_timer_cls.assert_called_once_with(self.watchdog)
        (interval_ms,), _ = self.mock_timer.setInterval.call_args
        self.assertGreater(interval_ms, 0)
        self.assertLessEqual(interval_ms, _FREEZE_SECONDS * 1000 / 2)
        self.mock_timer.timeout.connect.assert_called_once()

    def test_does_nothing_until_it_is_started(self):
        self.mock_dump_traceback_later.assert_not_called()
        self.mock_timer.start.assert_not_called()

    def test_start_schedules_the_stacks_and_starts_the_heartbeat(self):
        self.watchdog.start()

        self.mock_dump_traceback_later.assert_called_once_with(_FREEZE_SECONDS, file=self.log)
        self.mock_timer.start.assert_called_once_with()

    def test_every_heartbeat_postpones_the_stacks(self):
        self.watchdog.start()

        self._beat()
        self._beat()

        # Scheduling the stacks again replaces what was scheduled, so they are always a whole freeze time away.
        self.assertEqual(
            self.mock_dump_traceback_later.call_args_list,
            [call(_FREEZE_SECONDS, file=self.log)] * 3,
        )
        self.mock_cancel_dump_traceback_later.assert_not_called()

    def test_stop_ends_the_heartbeat_and_cancels_the_stacks(self):
        self.watchdog.start()

        self.watchdog.stop()

        self.mock_timer.stop.assert_called_once_with()
        self.mock_cancel_dump_traceback_later.assert_called_once_with()

    @parameterized.expand(
        [
            param("zero", freeze_seconds=0.0),
            param("negative", freeze_seconds=-1.0),
        ]
    )
    def test_rejects_a_freeze_time_that_is_not_positive(self, _, freeze_seconds):
        with self.assertRaisesRegex(ValueError, "freeze_seconds must be positive"):
            ballroom_scribe.gui.FreezeWatchdog(self.log, freeze_seconds=freeze_seconds)


if __name__ == "__main__":
    unittest.main()
