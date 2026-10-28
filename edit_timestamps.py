"""Desktop GUI for viewing, generating, and editing ballroom competition event timestamps."""

import ctypes
import sys

from PySide6 import QtCore, QtWidgets

import ballroom_scribe.cli.arguments
import ballroom_scribe.gui
import ballroom_scribe.gui.utils

_APP_NAME = "Ballroom Scribe Timestamp Editor"
_WINDOWS_APP_USER_MODEL_ID = "ballroom_scribe.timestamp_editor"
_K_CURRENT_PROCESS = 2
_K_PROCESS_TRANSFORM_TO_FOREGROUND_APPLICATION = 1
_K_LS_DEFAULT_SESSION_ID = -2
_K_CF_STRING_ENCODING_UTF8 = 0x08000100


class _ProcessSerialNumber(ctypes.Structure):
    _fields_ = (("high_long_of_psn", ctypes.c_uint32), ("low_long_of_psn", ctypes.c_uint32))


def _set_macos_process_name(name: str) -> None:
    """Registers `name` with macOS LaunchServices before `QApplication` checks in."""
    try:
        app_services = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        core_foundation = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")

        psn = _ProcessSerialNumber(0, _K_CURRENT_PROCESS)
        app_services.TransformProcessType.argtypes = (ctypes.POINTER(_ProcessSerialNumber), ctypes.c_uint32)
        app_services.TransformProcessType.restype = ctypes.c_int32
        app_services.TransformProcessType(ctypes.byref(psn), _K_PROCESS_TRANSFORM_TO_FOREGROUND_APPLICATION)

        app_services.CPSSetProcessName.argtypes = (ctypes.POINTER(_ProcessSerialNumber), ctypes.c_char_p)
        app_services.CPSSetProcessName.restype = ctypes.c_int32
        app_services.CPSSetProcessName(ctypes.byref(psn), name.encode("utf-8"))

        core_foundation.CFStringCreateWithCString.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32)
        core_foundation.CFStringCreateWithCString.restype = ctypes.c_void_p
        core_foundation.CFRelease.argtypes = (ctypes.c_void_p,)
        core_foundation.CFRelease.restype = None

        app_services._LSGetCurrentApplicationASN.argtypes = ()
        app_services._LSGetCurrentApplicationASN.restype = ctypes.c_void_p
        app_services._LSSetApplicationInformationItem.argtypes = (
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        app_services._LSSetApplicationInformationItem.restype = ctypes.c_int32

        display_key = ctypes.c_void_p.in_dll(app_services, "_kLSDisplayNameKey")
        cf_name = core_foundation.CFStringCreateWithCString(None, name.encode("utf-8"), _K_CF_STRING_ENCODING_UTF8)
        if cf_name:
            try:
                app_services._LSSetApplicationInformationItem(
                    _K_LS_DEFAULT_SESSION_ID,
                    app_services._LSGetCurrentApplicationASN(),
                    display_key,
                    cf_name,
                    None,
                )
            finally:
                core_foundation.CFRelease(cf_name)
    except AttributeError, OSError, ValueError:
        pass


def run_editor() -> int:
    """Parses CLI arguments and launches the Ballroom Scribe timestamp editor window."""
    cli_args, logging_args = ballroom_scribe.cli.arguments.parse_args(
        (ballroom_scribe.cli.arguments.InputCsvCacheGroup(), ballroom_scribe.cli.arguments.LoggingGroup()),
        description="Review, generate, and edit ballroom competition event CSV files.",
    )
    ballroom_scribe.gui.utils.configure_logging(logging_args["verbose"])
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_WINDOWS_APP_USER_MODEL_ID)
    elif sys.platform == "darwin":
        _set_macos_process_name(_APP_NAME)

    QtCore.QCoreApplication.setApplicationName(_APP_NAME)
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(ballroom_scribe.gui.utils.create_app_icon())

    window = ballroom_scribe.gui.MainWindow()
    window.show()
    if cli_args["inputs"]:
        window.add_videos(cli_args["inputs"])
    if cli_args["csv"] is not None:
        window.open_csv(cli_args["csv"])
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_editor())
