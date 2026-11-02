"""Backend-dispatched equilibrium computation for the AI-mediated model."""

from __future__ import annotations

from typing import Any

import numpy as np

from polar_ai.backends.selector import get_backend
from polar_ai.instance import Instance


def extremize(z: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Compute f_alpha_i(z_i) using the selected numerical backend."""

    return get_backend().extremize(z, alpha)


def compute_equilibrium(
    instance: Instance,
    *,
    tol: float = 1e-10,
    max_iter: int = 10_000,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute a fixed point using the selected numerical backend."""

    return get_backend().compute_equilibrium(
        instance, tol=tol, max_iter=max_iter, initial=initial
    )
