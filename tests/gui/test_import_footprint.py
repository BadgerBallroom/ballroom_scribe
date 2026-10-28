import importlib
import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

if importlib.util.find_spec("PySide6") is None:
    raise unittest.SkipTest("PySide6 is not installed.")

_FORBIDDEN_ML_MODULES = (
    "torch",
    "torchaudio",
    "faster_whisper",
    "ollama",
    "librosa",
)
_PURGE_ROOTS = ("ballroom_scribe", "panns_inference", *_FORBIDDEN_ML_MODULES)
_PURGE_PREFIXES = tuple(f"{root}." for root in _PURGE_ROOTS)


class TestGuiImportFootprint(unittest.TestCase):
    """Verifies that importing and resolving all GUI modules never imports heavy ML packages."""

    def test_gui_imports_do_not_load_ml_stack(self):
        saved_modules = dict(sys.modules)
        blocked_overrides: dict[str, None] = dict.fromkeys(_FORBIDDEN_ML_MODULES, None)
        clean_env = {k: v for k, v in os.environ.items() if k != "EAGER_IMPORT"}

        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch.dict(sys.modules, saved_modules, clear=True),
        ):
            # Remove `ballroom_scribe` and `panns_inference` so the GUI import chain runs from scratch.
            for name in list(sys.modules):
                if name in _PURGE_ROOTS or name.startswith(_PURGE_PREFIXES):
                    sys.modules.pop(name, None)

            sys.modules.update(blocked_overrides)

            for package_name in (
                "ballroom_scribe.gui",
                "ballroom_scribe.gui.models",
                "ballroom_scribe.gui.utils",
            ):
                package = importlib.import_module(package_name)
                for symbol_name in package.__all__:
                    self.assertIsNotNone(getattr(package, symbol_name))

            cli_pkg = importlib.import_module("ballroom_scribe.cli.arguments")
            self.assertIsNotNone(cli_pkg.InputCsvCacheGroup)
            self.assertIsNotNone(cli_pkg.parse_args)

            for forbidden in _FORBIDDEN_ML_MODULES:
                self.assertIsNone(
                    sys.modules.get(forbidden),
                    f"{forbidden!r} was unexpectedly imported by the GUI.",
                )

            if "panns_inference" in sys.modules:
                self.assertIsInstance(
                    sys.modules["panns_inference"],
                    importlib.util._LazyModule,
                    "'panns_inference' was eagerly executed instead of staying a lazy module proxy.",
                )


if __name__ == "__main__":
    unittest.main()
