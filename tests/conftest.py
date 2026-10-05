"""Shared fixtures: paper-canonical parameters and module loading for the reference repo."""

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execalgo import ExecParams  # noqa: E402


@pytest.fixture(scope="session")
def params():
    return ExecParams()


@pytest.fixture(scope="session")
def small_params():
    return ExecParams(grid_size=151, max_order=20)


def load_reference(name):
    path = ROOT / "reference" / "perturbative-impact" / f"{name}.py"
    if not path.exists():
        pytest.skip("reference repository not cloned")
    spec = importlib.util.spec_from_file_location(f"ref_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
