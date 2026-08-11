"""Load the sibling exp101 implementation under an unambiguous package name."""

import importlib.util
import sys
from pathlib import Path


def load_exp101():
    name = "exp101_certified_src"
    if name not in sys.modules:
        package = Path(__file__).resolve().parents[2] / "exp101" / "src"
        spec = importlib.util.spec_from_file_location(
            name, package / "__init__.py",
            submodule_search_locations=[str(package)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
