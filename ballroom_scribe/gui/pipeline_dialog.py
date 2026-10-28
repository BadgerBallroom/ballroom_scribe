"""Modal dialog for configuring, running, and monitoring `find_timestamps.py`."""

import pathlib
import shlex
from collections.abc import Sequence
from typing import override

from PySide6 import QtCore, QtGui, QtWidgets

import ballroom_scribe.cli.arguments
import ballroom_scribe.gui


class PipelineDialog(QtWidgets.QDialog):
    """Configures output CSV and optional CLI flags, runs `PipelineRunner`, and streams output."""

    pipeline_starting = QtCore.Signal()
    pipeline_succeeded = QtCore.Signal(pathlib.Path)

    def __init__(
        self,
        inputs: Sequence[ballroom_scribe.cli.arguments.InputCacheArgs],
        *,
        default_output_csv: pathlib.Path | None = None,
        has_unsaved_edits: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto-Find Events")
        self.resize(740, 520)

        self._inputs = list(inputs)
        self._has_unsaved_edits = has_unsaved_edits
        self._target_csv: pathlib.Path | None = None
        self._runner = ballroom_scribe.gui.PipelineRunner(self)
        self._runner.output.connect(self._append_log)
        self._runner.finished.connect(self._on_runner_finished)

        videos_tree = QtWidgets.QTreeWidget(self)
        videos_tree.setHeaderLabels(["Video", "Channel", "Parent Cache Directory"])
        videos_tree.setRootIsDecorated(False)
        videos_tree.setMaximumHeight(130)
        for item_args in self._inputs:
            QtWidgets.QTreeWidgetItem(
                videos_tree,
                [
                    str(item_args["input_video"]),
                    str(item_args["input_channel"]),
                    str(item_args["cache_dir"]),
                ],
            )
        for col in range(3):
            videos_tree.resizeColumnToContents(col)

        initial_csv = default_output_csv
        if initial_csv is None and len(self._inputs) == 1:
            initial_csv = self._inputs[0]["input_video"].with_suffix(".csv")

        self._output_edit = QtWidgets.QLineEdit(str(initial_csv) if initial_csv is not None else "", self)
        self._output_edit.setPlaceholderText("Select output CSV path…")
        self._browse_button = QtWidgets.QPushButton("Browse…", self)
        self._browse_button.clicked.connect(self._browse_output_csv)

        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self._output_edit, stretch=1)
        output_row.addWidget(self._browse_button)

        self._extra_args_edit = QtWidgets.QLineEdit(self)
        self._extra_args_edit.setPlaceholderText(
            "Optional flags, e.g. --pann-threshold-speech 0.2 --ollama-model gpt-oss:20b"
        )

        form = QtWidgets.QFormLayout()
        form.addRow("Output CSV:", output_row)
        form.addRow("Extra CLI flags:", self._extra_args_edit)

        self._log_view = QtWidgets.QPlainTextEdit(self)
        self._log_view.setReadOnly(True)
        mono_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        self._log_view.setFont(mono_font)

        self._run_button = QtWidgets.QPushButton("Auto-Find Events", self)
        self._run_button.setDefault(True)
        self._run_button.clicked.connect(self._start_pipeline)

        self._cancel_button = QtWidgets.QPushButton("Cancel", self)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel_pipeline)

        self._close_button = QtWidgets.QPushButton("Close", self)
        self._close_button.clicked.connect(self.close)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self._run_button)
        button_row.addWidget(self._cancel_button)
        button_row.addWidget(self._close_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Input videos:", self))
        layout.addWidget(videos_tree)
        layout.addLayout(form)
        layout.addWidget(QtWidgets.QLabel("Pipeline output:", self))
        layout.addWidget(self._log_view, stretch=1)
        layout.addLayout(button_row)

    @override
    def reject(self) -> None:
        if self._runner.is_running():
            return
        super().reject()

    @override
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._runner.is_running():
            event.ignore()
            return
        super().closeEvent(event)

    def _browse_output_csv(self) -> None:
        start_dir = self._output_edit.text().strip() or str(self._inputs[0]["input_video"].parent)
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select Output CSV",
            start_dir,
            "CSV Files (*.csv);;All Files (*)",
        )
        if selected:
            self._output_edit.setText(selected)

    def _start_pipeline(self) -> None:
        raw_output = self._output_edit.text().strip()
        if not raw_output:
            QtWidgets.QMessageBox.warning(self, "Output CSV Required", "Please choose an output CSV path.")
            return
        output_csv = pathlib.Path(raw_output)

        raw_extra = self._extra_args_edit.text().strip()
        try:
            extra_args = shlex.split(raw_extra) if raw_extra else []
        except ValueError as err:
            QtWidgets.QMessageBox.warning(self, "Invalid Extra Flags", f"Could not parse extra CLI flags:\n{err}")
            return

        if self._has_unsaved_edits:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Unsaved Table Edits",
                "Loading the pipeline output will replace the current table and discard unsaved edits. Continue?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        if output_csv.exists():
            answer = QtWidgets.QMessageBox.question(
                self,
                "Overwrite Output CSV?",
                f"{output_csv} already exists.\nOverwrite it?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        self._target_csv = output_csv
        self._log_view.clear()
        self._set_running_ui(True)
        # Lets the main window stop its own audio extraction so the pipeline is the only writer of the cache.
        self.pipeline_starting.emit()
        self._runner.start(inputs=self._inputs, output_csv=output_csv, extra_args=extra_args)

    def _cancel_pipeline(self) -> None:
        self._append_log("\nCanceling pipeline…\n")
        self._runner.cancel()

    def _append_log(self, text: str) -> None:
        cursor = self._log_view.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self._log_view.setTextCursor(cursor)
        self._log_view.ensureCursorVisible()

    def _on_runner_finished(self, exit_code: int) -> None:
        self._set_running_ui(False)
        if exit_code == 0 and self._target_csv is not None:
            self._append_log(f"\nPipeline succeeded. Loaded {self._target_csv}.\n")
            self.pipeline_succeeded.emit(self._target_csv)
            self._has_unsaved_edits = False
        else:
            self._append_log(f"\nPipeline exited with status {exit_code}.\n")

    def _set_running_ui(self, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._browse_button.setEnabled(not running)
        self._output_edit.setEnabled(not running)
        self._extra_args_edit.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        self._close_button.setEnabled(not running)
