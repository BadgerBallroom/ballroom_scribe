import importlib
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

from parameterized import param, parameterized

_PYSIDE6_AVAILABLE = importlib.util.find_spec("PySide6") is not None
_PACKAGES = [
    param("root", package_name="ballroom_scribe", requires_pyside6=False),
    param("models", package_name="ballroom_scribe.models", requires_pyside6=False),
    param("utils", package_name="ballroom_scribe.utils", requires_pyside6=False),
    param("cli_arguments", package_name="ballroom_scribe.cli.arguments", requires_pyside6=False),
    param("gui", package_name="ballroom_scribe.gui", requires_pyside6=True),
    param("gui_models", package_name="ballroom_scribe.gui.models", requires_pyside6=False),
    param("gui_utils", package_name="ballroom_scribe.gui.utils", requires_pyside6=True),
]


def _is_ballroom_scribe_module(name: str) -> bool:
    return name == "ballroom_scribe" or name.startswith("ballroom_scribe.")


def _assert_exports_resolve(case: unittest.TestCase, package_name: str) -> None:
    """Imports `package_name` and checks that its `__all__` is sorted and that every export resolves."""
    package = importlib.import_module(package_name)
    exports = list(package.__all__)

    case.assertGreater(len(exports), 0)
    case.assertEqual(exports, sorted(exports))

    for symbol_name in exports:
        symbol = getattr(package, symbol_name)
        case.assertTrue(
            symbol.__module__.startswith(f"{package_name}."),
            f"{package_name}.{symbol_name} resolved to unexpected module {symbol.__module__!r}",
        )


class TestLazyExports(unittest.TestCase):
    """Tests that every `ballroom_scribe` package lazily re-exports all symbols in its `__init__.pyi` stub."""

    @parameterized.expand(_PACKAGES)
    def test_exports_resolve_and_are_sorted(self, _, package_name: str, requires_pyside6: bool):
        if requires_pyside6 and not _PYSIDE6_AVAILABLE:
            self.skipTest("PySide6 is not installed.")

        _assert_exports_resolve(self, package_name)


class TestEagerImports(unittest.TestCase):
    """Tests that every package also imports with `EAGER_IMPORT=1`.

    In that mode `lazy_loader` imports every submodule while the package itself is still initializing, so a module
    that reaches its own package's lazy exports at import time (e.g. `class Foo(ballroom_scribe.gui.Bar)`) raises
    `AttributeError` here even though it works under the default lazy loading.
    """

    @parameterized.expand(_PACKAGES)
    def test_package_imports_eagerly(self, _, package_name: str, requires_pyside6: bool):
        # Importing any GUI subpackage eagerly initializes `ballroom_scribe.gui`, i.e. the whole Qt editor.
        if (requires_pyside6 or package_name.startswith("ballroom_scribe.gui")) and not _PYSIDE6_AVAILABLE:
            self.skipTest("PySide6 is not installed.")

        # Only `ballroom_scribe` modules are re-imported; third-party modules stay in `sys.modules` so no extension
        # module is initialized twice.
        saved_modules = {name: module for name, module in sys.modules.items() if _is_ballroom_scribe_module(name)}
        for name in saved_modules:
            del sys.modules[name]
        try:
            with patch.dict(os.environ, {"EAGER_IMPORT": "1"}):
                _assert_exports_resolve(self, package_name)
        finally:
            for name in [name for name in sys.modules if _is_ballroom_scribe_module(name)]:
                del sys.modules[name]
            sys.modules.update(saved_modules)


if __name__ == "__main__":
    unittest.main()
