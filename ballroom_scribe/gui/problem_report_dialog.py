"""Dialog that offers to report a problem, with a filled-in GitHub issue and the report file one click away."""

import logging
import pathlib

from PySide6 import QtCore, QtGui, QtWidgets

import ballroom_scribe.gui.models
import ballroom_scribe.gui.utils
import ballroom_scribe.utils

_LOGGER = logging.getLogger(__name__)
_WINDOW_TITLE = "Report a Problem"
# What the report file says happened when the user asks for a report without the editor having noticed a failure.
_UNPROMPTED_SUMMARY = "The user asked for this report from the Help menu. The editor had not noticed a failure."
_INSTRUCTIONS = (
    "A report file has been saved. To report the problem, open a GitHub issue and attach the file, or send the file"
    " to the maintainer."
)
_INSTRUCTIONS_WITHOUT_FILE = (
    "The report file could not be saved ({error}). You can still open a GitHub issue and describe the problem there."
)
# The report file and the form of the GitHub issue say the same in their own words.
_PRIVACY_NOTICE = (
    "The report file contains file names and folder paths from this computer, and it may contain speech that was"
    " transcribed from your videos. Look through it before you share it: GitHub issues are public."
)
_HEADLINE_SCALE = 1.25
_SUMMARY_LINES = 3
_MIN_WIDTH = 580


class ProblemReportDialog(QtWidgets.QDialog):
    """Says what went wrong and offers to report it, by a GitHub issue or by sending the report file.

    Creating the dialog writes the report file of the session (see `write_problem_report`), so the file exists by
    the time the dialog tells the user about it. Nothing leaves the computer unless the user sends it: **Open GitHub
    Issue** opens the form in the browser with the gist filled in and shows the report file in the file manager,
    ready to be dragged into the form, and **Show Report File** only shows the file.
    """

    def __init__(
        self,
        session_logs: ballroom_scribe.gui.models.SessionLogs,
        *,
        headline: str,
        summary: str = "",
        advice: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initializes the instance.

        Args:
            session_logs: Where the session keeps its logs, which go into the report file.
            headline: What the dialog is about, in a few words.
            summary: What went wrong, in a sentence or a few lines. It goes into the report file and into the GitHub
                issue. Leave it empty if the user reports something that the editor did not notice.
            advice: What the user should do now, if anything. It is shown here only.
            parent: The Qt parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle(_WINDOW_TITLE)
        self.setMinimumWidth(_MIN_WIDTH)

        self._report_file: pathlib.Path | None = None
        instructions = _INSTRUCTIONS
        try:
            self._report_file = ballroom_scribe.gui.utils.write_problem_report(
                session_logs,
                summary or _UNPROMPTED_SUMMARY,
            )
        except OSError as err:
            _LOGGER.warning("The report file could not be written: %s", err)
            instructions = _INSTRUCTIONS_WITHOUT_FILE.format(error=err)
        self._issue_url = ballroom_scribe.gui.utils.build_issue_url(
            summary,
            ballroom_scribe.utils.describe_environment(),
        )

        headline_label = QtWidgets.QLabel(headline, self)
        headline_font = headline_label.font()
        headline_font.setBold(True)
        headline_font.setPointSizeF(headline_font.pointSizeF() * _HEADLINE_SCALE)
        headline_label.setFont(headline_font)
        headline_label.setWordWrap(True)

        summary_view = QtWidgets.QPlainTextEdit(summary, self)
        summary_view.setObjectName("summary")
        summary_view.setReadOnly(True)
        summary_view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        summary_view.setFixedHeight(
            summary_view.fontMetrics().lineSpacing() * _SUMMARY_LINES + 2 * summary_view.frameWidth()
        )
        summary_view.setVisible(bool(summary))

        advice_label = self._paragraph(advice, "advice")
        advice_label.setVisible(bool(advice))
        instructions_label = self._paragraph(instructions, "instructions")
        privacy_label = self._paragraph(_PRIVACY_NOTICE, "privacyNotice")
        # Without a report file, there is nothing to warn about.
        privacy_label.setVisible(self._report_file is not None)
        file_label = self._paragraph(str(self._report_file) if self._report_file is not None else "", "reportFile")
        file_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        file_label.setVisible(self._report_file is not None)

        button_box = QtWidgets.QDialogButtonBox(self)
        open_issue_button = button_box.addButton("Open GitHub Issue", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        open_issue_button.setDefault(True)
        open_issue_button.clicked.connect(self.open_issue)
        show_file_button = button_box.addButton("Show Report File", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        show_file_button.setEnabled(self._report_file is not None)
        show_file_button.clicked.connect(self.show_report_file)
        button_box.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(headline_label)
        layout.addWidget(summary_view)
        layout.addWidget(advice_label)
        layout.addWidget(instructions_label)
        layout.addWidget(privacy_label)
        layout.addWidget(file_label)
        layout.addWidget(button_box)

    @property
    def report_file(self) -> pathlib.Path | None:
        """Returns the report file that was written for this dialog, or `None` if it could not be written."""
        return self._report_file

    @property
    def issue_url(self) -> str:
        """Returns the address of the filled-in GitHub issue that **Open GitHub Issue** opens."""
        return self._issue_url

    def open_issue(self) -> None:
        """Opens the filled-in GitHub issue in the browser and shows the report file in the file manager.

        The file manager is asked last, so that its window is the one in front, with the file ready to be dragged
        into the issue.
        """
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._issue_url))
        self.show_report_file()

    def show_report_file(self) -> None:
        """Shows the report file in the file manager, if there is one."""
        if self._report_file is not None:
            ballroom_scribe.gui.utils.reveal_file(self._report_file)

    def _paragraph(self, text: str, object_name: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text, self)
        label.setObjectName(object_name)
        label.setWordWrap(True)
        return label
