import importlib.util
import os
from pathlib import Path


def load_problem_module(relative_path: str):
    """
    Load a problem python file as a module by relative path from repo root.
    Returns the imported module object.
    """
    root = Path(__file__).resolve().parents[1]
    file_path = root / relative_path
    if not file_path.exists():
        raise FileNotFoundError(f"Problem file not found: {file_path}")

    module_name = "problem_" + file_path.name.replace(".", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Failed to create module spec"
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def env_flag(name: str, default: str = "0") -> bool:
    """Return True if environment variable equals "1"."""
    return os.getenv(name, default) == "1"
