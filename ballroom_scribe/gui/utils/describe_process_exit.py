"""Utility that says in words how a child process ended."""

import signal

_UNSIGNED_32_BIT_MASK = 0xFFFFFFFF
# Windows reports a crash as an `NTSTATUS` value, all of which are at least this large, instead of a small exit
# status.
_FIRST_WINDOWS_STATUS = 0x80000000
_WINDOWS_STATUS_NAMES = {
    0x80000003: "breakpoint",
    0xC0000005: "access violation",
    0xC000001D: "illegal instruction",
    0xC00000FD: "stack overflow",
    0xC0000135: "a DLL was not found",
    0xC000013A: "interrupted with Ctrl+C",
    0xC0000374: "heap corruption",
    0xC0000409: "the program aborted",
}
_UNKNOWN_SIGNAL = "unknown signal"


def _signal_name(signal_number: int) -> str:
    try:
        return signal.Signals(signal_number).name
    except ValueError:
        return _UNKNOWN_SIGNAL


def describe_process_exit(exit_code: int, *, signal_number: int | None = None) -> str:
    """Returns a phrase that completes "The process …", such as `exited with status 1`.

    Args:
        exit_code: The exit status of the process. A negative number is read as an unsigned 32-bit number, because
            `QProcess` reports the large values with which Windows marks a crash as negative numbers.
        signal_number: The signal that ended the process, on platforms that have signals, or `None` if the process
            exited by itself. `exit_code` is ignored when a signal is given.

    Returns:
        `exited with status 1`, `was ended by signal 11 (SIGSEGV)`, or `crashed with status 0xC0000005 (access
        violation)`.
    """
    if signal_number is not None:
        return f"was ended by signal {signal_number} ({_signal_name(signal_number)})"
    exit_code &= _UNSIGNED_32_BIT_MASK
    if exit_code < _FIRST_WINDOWS_STATUS:
        return f"exited with status {exit_code}"
    name = _WINDOWS_STATUS_NAMES.get(exit_code)
    return f"crashed with status 0x{exit_code:08X}" + (f" ({name})" if name else "")
