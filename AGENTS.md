# Instructions for Agents

## Coding Conventions

### Imports and Package Exports

*   **Import modules, not symbols**: Always import `ballroom_scribe` packages
    and modules as `import ballroom_scribe`, `import ballroom_scribe.models`,
    `import ballroom_scribe.utils`, `import ballroom_scribe.cli.arguments`, or
    `import ballroom_scribe.<module>` (when accessing module-private symbols in
    tests), and reference symbols via their qualified names (e.g.,
    `ballroom_scribe.AudioClassifier`, `ballroom_scribe.models.AudioChunk`,
    `ballroom_scribe.utils.extract_audio`). Never use `from ballroom_scribe...
    import ...` in application or test modules.
    *   **Exceptions**: Package `__init__.py` files use relative imports (`from
        .foo import Foo`) to re-export public symbols in `__all__`, and modules
        within the same subpackage (`ballroom_scribe/models/`,
        `ballroom_scribe/cli/arguments/`) may use relative imports (`from .word
        import Word`, `from .group_abc import Group, GroupArgs`) for sibling
        dependencies.
*   **Test helper imports**: Import test helpers as `import
    tests.helpers.<module>` and reference symbols via
    `tests.helpers.<module>.Maker` rather than `from tests.helpers... import
    ...`.
*   **Standard library and third-party imports**: Prefer module imports (`import
    collections`, `import dataclasses`, `import functools`, `import pathlib`,
    `import unittest`, `import faster_whisper`, `import soundfile`, `import
    numpy as np`), using `from ... import ...` primarily for typing/ABC
    constructs (`from collections.abc import ...`, `from typing import ...`),
    `unittest.mock`, and `parameterized`.

### Package Structure and Models (`ballroom_scribe/models/`)

*   **Domain models live in `ballroom_scribe/models/`**: Put all shared domain
    data structures in `ballroom_scribe/models/` (one model class per file,
    snake_case filename matching the PascalCase class name) and re-export them
    in `ballroom_scribe/models/__init__.py` via `__all__`. Only strictly
    internal data structures coupled to a single module's algorithm (e.g.,
    `_PeeledDance` in `music_valley_detector.py`, `_EditDistance` in
    `find_closest_matching.py`) should be defined as private `_`-prefixed
    classes outside `ballroom_scribe/models/`.
*   **Immutable models**: Define models with `@dataclass(frozen=True)`.
    *   Use `tuple[T, ...]` instead of `list[T]` for sequence fields.
    *   Mark NumPy array fields as read-only in `__post_init__`
        (`self.data.flags.writeable = False`) and compare them with
        `np.array_equal` in `__eq__`.
    *   Create updated instances with `dataclasses.replace(...)` rather than
        mutating state.
*   **One primary public symbol per module**: Each module in `ballroom_scribe/`,
    `ballroom_scribe/models/`, and `ballroom_scribe/utils/` defines one primary
    public class or function matching the module's filename and re-exports it in
    the package's `__init__.py` with `__all__`.

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
    transcription, or LLM calls.

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

## What to Unit Test

The `tests/` directory mirrors the `ballroom_scribe/` package structure. Follow
these conventions for what does and doesn't have unit tests:

### Has Unit Tests

*   **Core services and pipeline classes (`ballroom_scribe/*.py` →
    `tests/test_*.py`)**: Every service, detector, loader, transcriber,
    recorder, and pipeline orchestrator in `ballroom_scribe/` has a
    corresponding `test_*.py` module.
*   **Utility functions (`ballroom_scribe/utils/*.py` →
    `tests/utils/test_*.py`)**: Every utility module in `ballroom_scribe/utils/`
    has a corresponding unit test file.
*   **Models with custom logic (`ballroom_scribe/models/` →
    `tests/models/test_*.py`)**: Dataclasses in `ballroom_scribe/models/` are
    unit-tested only when they contain non-trivial behavior—such as
    `__post_init__` validation or immutability enforcement, custom
    methods/classmethods, operator overloads (`__add__`, `__bool__`, `__eq__`,
    `__iter__`), or computed properties (e.g., `AudioChunk`,
    `RunLengthEncoding`).
*   **CLI argument parsing (`ballroom_scribe/cli/arguments/` →
    `tests/cli/test_arguments.py`)**: Argument groups and custom argument types
    are tested together through the public `parse_args` function in
    `tests/cli/test_arguments.py` (covering defaults, custom flags,
    cross-argument validation, and error handling) rather than in separate test
    files per group module.
*   **Non-trivial private functions and methods**: Complex module-private
    helpers and private methods (e.g., in `audio_classifier.py`,
    `event_detector.py`, `music_valley_detector.py`, `speech_transcriber.py`)
    are unit-tested directly, and higher-level public methods mock them out to
    test orchestration in isolation.

### Does Not Have Unit Tests

*   **Top-level scripts and interactive UI tools (`*.py` at repo root)**: Thin
    CLI entry points (`find_timestamps.py`) and interactive
    developer/visualization apps (`tune_pann_thresholds.py` with Streamlit,
    local HTTP streaming, and embedded HTML/JS) do not have unit tests.
*   **Pure data containers (`ballroom_scribe/models/`)**: Frozen dataclasses
    that only declare fields or trivial one-line formatting properties (e.g.,
    `Event`, `MusicValley`, `Word`, `TranscriptChunk`) do not have dedicated
    unit test files; they are exercised through the components that produce and
    consume them.
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
    scripts (`find_timestamps.py`), or interactive tools
    (`tune_pann_thresholds.py`).
