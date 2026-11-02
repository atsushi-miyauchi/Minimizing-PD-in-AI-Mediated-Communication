"""Runtime selection between the C++ and preserved Python numerical cores."""

from __future__ import annotations

import importlib
import os
from types import ModuleType

_VALID_BACKENDS = {"cpp", "python"}
_override: str | None = None
_cpp_import_error: str | None = None
_cached_request: str | None = None
_cached_backend: ModuleType | None = None


def _load(name: str) -> ModuleType:
    global _cpp_import_error
    if name == "python":
        return importlib.import_module("polar_ai.backends.python_core")
    try:
        module = importlib.import_module("polar_ai.backends.cpp_core")
        _cpp_import_error = None
        return module
    except ImportError as exc:
        _cpp_import_error = str(exc)
        return importlib.import_module("polar_ai.backends.python_core")


def _requested_backend() -> str:
    name = (_override or os.environ.get("POLAR_AI_BACKEND", "cpp")).lower()
    if name not in _VALID_BACKENDS:
        choices = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"POLAR_AI_BACKEND must be one of: {choices}.")
    return name


def get_backend() -> ModuleType:
    """Return the requested backend, falling back to Python if C++ is absent."""

    global _cached_request, _cached_backend
    requested = _requested_backend()
    if _cached_backend is None or _cached_request != requested:
        _cached_backend = _load(requested)
        _cached_request = requested
    return _cached_backend


def get_backend_name() -> str:
    """Return the backend that will actually execute numerical kernels."""

    backend = get_backend()
    return "cpp" if backend.__name__.endswith("cpp_core") else "python"


def set_backend(name: str | None) -> None:
    """Override backend selection in-process; None restores environment/default."""

    global _override, _cached_request, _cached_backend
    if name is not None and name.lower() not in _VALID_BACKENDS:
        choices = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"backend must be one of: {choices}.")
    _override = None if name is None else name.lower()
    _cached_request = None
    _cached_backend = None


def available_backends() -> tuple[str, ...]:
    """Return backends available in the current installation."""

    available = ["python"]
    try:
        importlib.import_module("polar_ai.backends.cpp_core")
    except ImportError:
        pass
    else:
        available.insert(0, "cpp")
    return tuple(available)


def backend_status() -> dict[str, str | None]:
    """Return requested/active backend and any C++ loading error."""

    requested = _requested_backend()
    active = get_backend_name()
    return {
        "requested": requested,
        "active": active,
        "cpp_import_error": _cpp_import_error,
    }
