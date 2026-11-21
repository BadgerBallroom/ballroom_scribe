import signal
import unittest

from parameterized import param, parameterized

import ballroom_scribe.gui.utils

# A number that is no signal on any platform.
_NO_SIGNAL = 250


class TestDescribeProcessExit(unittest.TestCase):
    """Unit tests for the `describe_process_exit` function."""

    @parameterized.expand(
        [
            param("success", exit_code=0, expected="exited with status 0"),
            param("python_exception", exit_code=1, expected="exited with status 1"),
            param("usage_error", exit_code=2, expected="exited with status 2"),
            # The exit status that `QProcess.kill` gives a child on Windows.
            param("killed_by_qprocess_on_windows", exit_code=0xF291, expected="exited with status 62097"),
        ]
    )
    def test_process_that_exited_by_itself(self, _, exit_code, expected):
        self.assertEqual(ballroom_scribe.gui.utils.describe_process_exit(exit_code), expected)

    @parameterized.expand(
        [
            param(
                "segmentation_fault",
                signal_number=int(signal.SIGSEGV),
                expected=f"was ended by signal {int(signal.SIGSEGV)} (SIGSEGV)",
            ),
            param(
                "terminated",
                signal_number=int(signal.SIGTERM),
                expected=f"was ended by signal {int(signal.SIGTERM)} (SIGTERM)",
            ),
            param("unknown_signal", signal_number=_NO_SIGNAL, expected="was ended by signal 250 (unknown signal)"),
        ]
    )
    def test_process_that_a_signal_ended(self, _, signal_number, expected):
        # `QProcess` reports the signal as the exit code as well.
        self.assertEqual(
            ballroom_scribe.gui.utils.describe_process_exit(signal_number, signal_number=signal_number),
            expected,
        )

    @parameterized.expand(
        [
            param(
                "access_violation",
                exit_code=0xC0000005,
                expected="crashed with status 0xC0000005 (access violation)",
            ),
            param("aborted", exit_code=0xC0000409, expected="crashed with status 0xC0000409 (the program aborted)"),
            param(
                "dll_not_found", exit_code=0xC0000135, expected="crashed with status 0xC0000135 (a DLL was not found)"
            ),
            param("status_without_a_name", exit_code=0xC0000096, expected="crashed with status 0xC0000096"),
            # `QProcess.exitCode` is a signed 32-bit number, so it reports `0xC0000005` like this.
            param(
                "negative_from_qprocess",
                exit_code=-1073741819,
                expected="crashed with status 0xC0000005 (access violation)",
            ),
        ]
    )
    def test_process_that_crashed_on_windows(self, _, exit_code, expected):
        self.assertEqual(ballroom_scribe.gui.utils.describe_process_exit(exit_code), expected)


if __name__ == "__main__":
    unittest.main()
