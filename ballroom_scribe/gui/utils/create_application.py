"""Creates the `QApplication` of the timestamp editor, named for the platform's task bar or Dock."""

import ctypes
import sys
from collections.abc import Sequence

from PySide6 import QtCore, QtWidgets

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


def create_application(argv: Sequence[str]) -> QtWidgets.QApplication:
    """Creates the application object, after telling the platform what this process is called.

    On Windows the process gets its own entry in the task bar instead of sharing Python's, and on macOS the Dock and
    the menu bar show the name of the editor instead of "Python". The application gets the editor's icon.

    Args:
        argv: The command line of the process, from which Qt takes the options that it knows.

    Returns:
        The application object. There can be only one per process.
    """
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_WINDOWS_APP_USER_MODEL_ID)
    elif sys.platform == "darwin":
        _set_macos_process_name(_APP_NAME)

    QtCore.QCoreApplication.setApplicationName(_APP_NAME)
    app = QtWidgets.QApplication(list(argv))
    app.setWindowIcon(ballroom_scribe.gui.utils.create_app_icon())
    return app
