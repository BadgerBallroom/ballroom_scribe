"""Utility that shows a file in the file manager of the platform."""

import pathlib
import subprocess
import sys

from PySide6 import QtCore, QtGui

_MACOS_OPEN = "/usr/bin/open"


def _select_in_file_manager(path: pathlib.Path) -> bool:
    """Asks the file manager to open the folder of `path` with the file selected; returns whether it could be asked.

    Explorer on Windows and Finder on macOS can do that. Elsewhere there is no common way to ask for it.
    """
    try:
        if sys.platform == "win32":
            # Explorer wants the path quoted inside this one argument, which a list of arguments cannot express. It
            # also exits with status 1 when it succeeds, so its exit status says nothing.
            subprocess.run(f'explorer /select,"{path}"', check=False)
            return True
        if sys.platform == "darwin":
            subprocess.run([_MACOS_OPEN, "-R", str(path)], check=False)
            return True
    except OSError:
        pass
    return False


def reveal_file(path: pathlib.Path) -> None:
    """Opens the folder of `path` in the file manager, with the file selected where the platform can do that.

    Explorer on Windows and Finder on macOS select the file, ready to be dragged somewhere. Elsewhere, or if they
    cannot be started, the folder is opened in whatever the desktop uses for folders.

    Args:
        path: The file to show.
    """
    if not _select_in_file_manager(path):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path.parent)))
