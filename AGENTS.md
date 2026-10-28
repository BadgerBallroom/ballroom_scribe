# Instructions for Agents

## Coding Conventions

### Imports and Package Exports

*   **Import modules, not symbols**: Always import `ballroom_scribe` packages
    and modules as `import ballroom_scribe`, `import ballroom_scribe.models`,
    `import ballroom_scribe.utils`, `import ballroom_scribe.cli.arguments`,
    `import ballroom_scribe.gui`, `import ballroom_scribe.gui.models`,
    `import ballroom_scribe.gui.utils`, or `import ballroom_scribe.<module>`
    (when accessing module-private symbols in tests), and reference symbols via
    their qualified names (e.g., `ballroom_scribe.AudioClassifier`,
    `ballroom_scribe.models.AudioChunk`, `ballroom_scribe.utils.extract_audio`,
    `ballroom_scribe.gui.models.WaveformEnvelope`). Never use `from
    ballroom_scribe... import ...` in application or test modules, and never
    `import` a same-named submodule directly (e.g., `import
    ballroom_scribe.utils.extract_audio`) — always access public symbols through
    the package.
    *   **Exceptions**: Package `__init__.pyi` stub files use relative imports
        (`from .foo import Foo`) to declare public re-exports in `__all__`, and
        modules within the same subpackage (`ballroom_scribe/models/`,
        `ballroom_scribe/gui/models/`, `ballroom_scribe/cli/arguments/`) may use
        relative imports (`from .word import Word`, `from .group_abc import
        Group, GroupArgs`) for sibling dependencies. A sibling symbol that a
        module needs *while it is being imported* — above all a base class, such
        as `CurrentCellDelegate` in `ballroom_scribe/gui/autocomplete_delegate.py`
        — must also be imported this way (`from .current_cell_delegate import
        CurrentCellDelegate`) rather than reached through the package
        (`ballroom_scribe.gui.CurrentCellDelegate`): under `EAGER_IMPORT=1` every
        submodule is loaded while its package is still initializing, so the
        package's own lazy exports do not resolve yet. Symbols used only inside
        function bodies keep the qualified `ballroom_scribe.gui.<Symbol>` form.
*   **Lazy package exports (`lazy-loader`)**: Every package `__init__.py`
    contains only a module docstring, `import lazy_loader`, and
    `__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)`.
    All relative `from .module import Symbol` imports and sorted `__all__` lists
    live in the adjacent `__init__.pyi` stub, which serves as the single source
    of truth for both runtime lazy loading and static type checking so
    lightweight consumers (such as the GUI editor and tests of light modules)
    never import PyTorch, PANNs, Faster Whisper, or Ollama. When a lightweight
    module needs a heavy third-party package only on certain code paths, bind it
    at module top with `name = lazy_loader.load("name")` (e.g.,
    `panns_inference` in `extract_audio.py`) rather than using a function-level
    import. Set `EAGER_IMPORT=1` when debugging or verifying stub exports.
*   **Test helper imports**: Import test helpers as `import
    tests.helpers.<module>` and reference symbols via
    `tests.helpers.<module>.Maker` rather than `from tests.helpers... import
    ...`.
*   **Standard library and third-party imports**: Prefer module imports (`import
    collections`, `import dataclasses`, `import functools`, `import pathlib`,
    `import unittest`, `import faster_whisper`, `import soundfile`, `import
    numpy as np`), using `from ... import ...` primarily for typing/ABC
    constructs (`from collections.abc import ...`, `from typing import ...`),
    Qt submodules (`from PySide6 import QtCore, QtGui, QtWidgets` with qualified
    names like `QtWidgets.QMainWindow`), `unittest.mock`, and `parameterized`.

### Package Structure and Models (`ballroom_scribe/models/` and `ballroom_scribe/gui/`)

*   **Shared domain models live in `ballroom_scribe/models/`**: Put domain data
    structures shared with the pipeline in `ballroom_scribe/models/` (one model
    class per file, snake_case filename matching the PascalCase class name) and
    re-export them in `ballroom_scribe/models/__init__.pyi` via `__all__`. Only
    strictly internal data structures coupled to a single module's algorithm
    (e.g., `_PeeledDance` in `music_valley_detector.py`, `_EditDistance` in
    `find_closest_matching.py`) should be defined as private `_`-prefixed
    classes outside `models/`.
*   **GUI-specific models and utilities live under `ballroom_scribe/gui/`**:
    Data structures and helper functions used only by the PySide6 timestamp
    editor go in `ballroom_scribe/gui/models/` and `ballroom_scribe/gui/utils/`
    (following the same one-symbol-per-file, frozen-dataclass, and lazy
    `__init__.pyi` re-export rules) and are referenced as
    `ballroom_scribe.gui.models.<Model>` and
    `ballroom_scribe.gui.utils.<function>`. Only code shared with the pipeline
    belongs in `ballroom_scribe/models/` and `ballroom_scribe/utils/`.
*   **Immutable models**: Define models with `@dataclass(frozen=True)`.
    *   Use `tuple[T, ...]` instead of `list[T]` for sequence fields.
    *   Mark NumPy array fields as read-only in `__post_init__`
        (`self.data.flags.writeable = False`) and compare them with
        `np.array_equal` in `__eq__`.
    *   Create updated instances with `dataclasses.replace(...)` rather than
        mutating state.
*   **One primary public symbol per module**: Each module in `ballroom_scribe/`,
    `ballroom_scribe/models/`, `ballroom_scribe/utils/`,
    `ballroom_scribe/gui/`, `ballroom_scribe/gui/models/`, and
    `ballroom_scribe/gui/utils/` (as well as `parse_args.py` in
    `ballroom_scribe/cli/arguments/`) defines one primary public class or
    function matching the module's filename and re-exports it in the package's
    `__init__.pyi` with `__all__`.

### Single Responsibility Principle (SRP) and Function Design

*   **Single responsibility**: Each function or method should have a single,
    well-defined responsibility. Decompose complex operations into small,
    composable steps (e.g., separating speech filtering, adjacent chunk merging,
    and buffer detachment in `music_valley_detector.py`, or quote alignment,
    dict conversion, and JSON parsing in `event_detector.py`).
*   **Stateless module-private helpers**: If a helper does not use `self`,
    define it as a module-private top-level function (`_helper_name`) rather
    than a method on the class.
*   **High-level orchestration**: Public methods (`run`, `classify`, `detect`,
    `transcribe`, `chunk`) should act as high-level orchestrators
    that compose these smaller private helpers and generator stages.
*   **Strict visibility**: Prefix all internal module helpers, constants
    (`_UPPER_SNAKE_CASE`), helper classes, private methods, and instance
    attributes (`self._attr`) with a single leading underscore `_`.

### Resource Management, Lazy Loading, and Streaming

*   **Fail-fast `__init__` validation + lazy `@functools.cached_property`
    loading**: Validate configuration and device arguments eagerly in `__init__`
    so misconfigurations fail immediately, and lazy-load heavy ML/GPU models
    (`SoundEventDetection`, `WhisperModel`) in private
    `@functools.cached_property` methods.
*   **Streaming generators and memory discipline**: Yield items through
    `Generator`s across pipeline stages to bound RAM and VRAM usage on long
    videos. When slicing large `AudioChunk` arrays (which creates zero-copy
    NumPy views) and retaining only a subset, explicitly call `.copy()` to
    detach views so parent buffers can be garbage-collected.
*   **Idempotent disk caching**: Check whether intermediate cache files already
    exist before running expensive extraction, inference,
    transcription, or LLM calls, and write them through
    `ballroom_scribe.utils.atomic_path` so that existence implies completeness.
    `atomic_path` uses a deterministic temporary name, so ensure a single writer
    per file at a time (the GUI de-duplicates in-flight waveform loads and stops
    its own extraction before launching the pipeline).
*   **Cancellable background subprocesses**: Long-running subprocesses started
    from the GUI (e.g., `ffmpeg` in `extract_audio`) take an optional
    `threading.Event` and poll it around `subprocess.Popen.wait(timeout=...)`,
    killing the process and raising `concurrent.futures.CancelledError` when it
    is set, so `MainWindow.closeEvent` can stop them via
    `WaveformLoader.shutdown()` instead of blocking at exit.
*   **GUI diagnostics via `logging`**: GUI modules report diagnostics through a
    module-private `_LOGGER = logging.getLogger(__name__)` (never `print`),
    using `INFO` for one-time milestones (media loaded, first frame rendered),
    `DEBUG` for chatty state transitions, and `WARNING`/`ERROR` for problems
    the user must see. `ballroom_scribe.gui.utils.configure_logging` (called by
    `edit_timestamps.py` and `smoke_test_editor.py`) owns `logging.basicConfig`,
    forces Qt's own handler to print to stderr in a matching format, and enables
    Qt multimedia debug categories only for `--verbose`.
*   **No Python on Qt's worker threads**: Never install a Python
    `qInstallMessageHandler`, and never connect plain Python callables to
    signals emitted from non-GUI threads (`QVideoSink.videoFrameChanged`,
    `QRunnable` helper signals). Both run Python on Qt Multimedia's decoder or
    renderer threads, which must acquire the GIL while the GUI thread may hold
    it inside a blocking Qt call and deadlock (observed on Windows). Connect
    such signals to bound methods of `QObject`s living in the GUI thread so Qt
    queues the delivery.

### Typing and Python 3.14 Idioms

*   **Python 3.14 syntax**: Use PEP 695 `type` aliases (`type ThresholdMapping =
    ...`) and generic type parameter syntax (`class RunLengthEncoding[T:
    np.generic]:`, `def find_closest_matching[T: HasEq](...):`).
*   **Modern type annotations**: Use built-in generics (`list`, `dict`, `tuple`,
    `frozenset`), `X | Y` union syntax, `typing.Self`, and `@typing.override`.
    Import `Generator`, `Iterable`, `Mapping`, and `Sequence` from
    `collections.abc` (using single-parameter `Generator[YieldType]` when send
    and return types are `None`).
*   **Paths**: Use `pathlib.Path` for all filesystem paths instead of `str`.
*   **Docstrings and comments**: Use Google-style docstrings (`Args:`,
    `Returns:`, `Yields:`, `Raises:`, `Attributes:`) for modules, classes, and
    non-trivial functions. Enclose code in docstrings and comments in backticks.

## Running Tests

Always run unit tests with Python's built-in `unittest` runner:

```pwsh
python -m unittest discover -v -s ./tests -p "test_*.py"
```

For an end-to-end check of the GUI on a real machine (real `ffmpeg`/`ffprobe`
subprocesses and Qt Multimedia decoding, which unit tests must never perform),
run the standalone smoke test, which is not discovered by `unittest` and drives
`MainWindow` only through its public API:

```pwsh
python smoke_test_editor.py
```

## What to Unit Test

The `tests/` directory mirrors the `ballroom_scribe/` package structure. Follow
these conventions for what does and doesn't have unit tests:

### Has Unit Tests

*   **Core services and pipeline classes (`ballroom_scribe/*.py` →
    `tests/test_*.py`)**: Every service, detector, loader, transcriber,
    recorder, and pipeline orchestrator in `ballroom_scribe/` has a
    corresponding `test_*.py` module.
*   **Utility functions (`ballroom_scribe/utils/*.py` →
    `tests/utils/test_*.py`, `ballroom_scribe/gui/utils/*.py` →
    `tests/gui/utils/test_*.py`)**: Every utility module has a corresponding
    unit test file.
*   **Models with custom logic (`ballroom_scribe/models/` →
    `tests/models/test_*.py`, `ballroom_scribe/gui/models/` →
    `tests/gui/models/test_*.py`)**: Dataclasses are unit-tested only when they
    contain non-trivial behavior—such as `__post_init__` validation or
    immutability enforcement, custom methods/classmethods, operator overloads
    (`__add__`, `__bool__`, `__eq__`, `__iter__`), or computed properties (e.g.,
    `AudioChunk`, `EventRecord`, `RunLengthEncoding`, `WaveformEnvelope`).
*   **GUI services and table models (`ballroom_scribe/gui/` →
    `tests/gui/test_*.py`)**: Non-widget GUI logic (`VideoSession`,
    `EventTableModel`, `PipelineRunner`, `WaveformLoader._prepare_envelope`) is
    unit-tested headlessly. Tests that import `PySide6` skip cleanly via
    `unittest.SkipTest` when the optional GUI dependency is not installed.
*   **Lazy exports and import footprint (`tests/test_lazy_exports.py`,
    `tests/gui/test_import_footprint.py`)**: Regression tests verify that every
    package's `__init__.pyi` exports are sorted and resolvable — both lazily and
    with `EAGER_IMPORT=1`, which catches modules that reach their own package's
    lazy exports at import time — and that importing the GUI never transitively
    loads heavy ML libraries.
*   **CLI argument parsing (`ballroom_scribe/cli/arguments/` →
    `tests/cli/test_arguments.py`)**: Argument groups and custom argument types
    are tested together through the public `parse_args` function in
    `tests/cli/test_arguments.py` (covering defaults, custom flags,
    cross-argument validation, and error handling) rather than in separate test
    files per group module.
*   **Non-trivial private functions and methods**: Complex module-private
    helpers and private methods (e.g., in `audio_classifier.py`,
    `event_detector.py`, `music_valley_detector.py`, `speech_transcriber.py`,
    `pipeline_runner.py`, `waveform_loader.py`) are unit-tested directly, and
    higher-level public methods mock them out to test orchestration in
    isolation.

### Does Not Have Unit Tests

*   **Top-level scripts and interactive UI widgets (`*.py` at repo root and Qt
    widgets)**: Thin CLI entry points (`find_timestamps.py`,
    `edit_timestamps.py`), interactive Streamlit apps
    (`tune_pann_thresholds.py`), the end-to-end GUI smoke test
    (`smoke_test_editor.py`), and thin Qt painting/wiring widgets, dialogs,
    and delegates (`MainWindow`, `VideoPlayerWidget`, `WaveformWidget`,
    `EventTableView`, `PipelineDialog`, `AutocompleteDelegate`,
    `CurrentCellDelegate`) do not have unit tests; the smoke test drives
    them, and the model operations they trigger (e.g.
    `EventTableModel.fill`) are unit-tested.
*   **Pure data containers (`ballroom_scribe/models/`,
    `ballroom_scribe/gui/models/`)**: Frozen dataclasses that only declare
    fields or trivial one-line formatting properties (e.g., `Event`,
    `LoadedVideo`, `MusicValley`, `Word`, `TranscriptChunk`) do not have
    dedicated unit test files; they are exercised through the components that
    produce and consume them.
*   **Trivial private helpers, internal data structures, and OS/hardware glue**:
    *   Simple one-line private helpers, internal algorithmic helper classes
        (such as `_SellersMatrix` and its debug `__str__` methods in
        `find_closest_matching.py`), and abstract base classes/TypedDicts
        (`group_abc.py`) are not tested directly when covered by their callers.
    *   Low-level OS hooks such as the Windows-only
        `_fix_windows_dll_search_path()` are not unit-tested.
*   **Real I/O, subprocesses, and ML model execution**: Unit tests never perform
    real disk I/O, run external binaries (`ffmpeg`), or execute real ML/GPU
    models (`panns_inference`, `faster_whisper`, `ollama`,
    `torch.cuda`, `librosa`, `soundfile`). Always mock external I/O and
    third-party libraries, and test eager `__init__` validation and
    `@cached_property` lazy loading instead.

### Test Structure and Style

*   **Mock registration**: Register patches in `setUp` using
    `self.enterContext(patch.object(target, "attr", autospec=True))` rather than
    nested `with patch(...)` blocks or stacked decorators. Prefer `patch.object`
    over string-path `patch("...")` (except for builtins like `builtins.open`).
*   **Parameterized tests**: Use `@parameterized.expand([param("case_name",
    ...), ...])` from `parameterized` for table-driven test cases, using `_` for
    the unused test name parameter in the test method signature.
*   **Focused `TestCase` classes**: When a module has multiple functions or
    methods under test, group tests into separate `unittest.TestCase` classes
    per function or method (e.g., `TestCutBeginning`, `TestInit`,
    `TestClassify`).
*   **Test helpers and entry point**: Place reusable test data builders in
    `tests/helpers/` and end every test file with `if __name__ == "__main__":
    unittest.main()`.

## Linting and Formatting

Always lint and format Python code using [Ruff](https://docs.astral.sh/ruff/):

```pwsh
ruff check --fix
ruff format
```

## Documentation

Keep `README.md` up to date whenever changes affect user-facing behavior or
project architecture, including:

*   **Architecture and pipeline stages**: Updates to `TimestampPipeline` stages,
    data flow, component responsibilities, or intermediate cache file formats
    (including the Mermaid flowchart and numbered stage breakdown).
*   **Models, hardware, and dependencies**: Changes to ML/LLM models, VRAM
    requirements, system tools (`ffmpeg`, `ollama`), or Python package Setup
    steps.
*   **CLI usage and flags**: New, renamed, or removed CLI arguments, entry-point
    scripts (`find_timestamps.py`, `edit_timestamps.py`,
    `smoke_test_editor.py`), or interactive tools (`tune_pann_thresholds.py`).
