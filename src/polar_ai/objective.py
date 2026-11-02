"""Backend-dispatched polarization-disagreement objective functions."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from polar_ai.backends.selector import get_backend
from polar_ai.instance import Instance


def polarization(z: np.ndarray) -> float:
    """Return ||z||_2^2 using the selected backend."""

    return get_backend().polarization(z)


def disagreement(z: np.ndarray, Wbar: sparse.spmatrix) -> float:
    """Return weighted disagreement using the selected backend."""

    return get_backend().disagreement(z, Wbar)


def polarization_disagreement(z: np.ndarray, Wbar: sparse.spmatrix) -> float:
    """Return PD(z) using the selected backend."""

    return get_backend().polarization_disagreement(z, Wbar)


def pd_gradient(z: np.ndarray, Wbar: sparse.spmatrix) -> np.ndarray:
    """Return the gradient of PD using the selected backend."""

    return get_backend().pd_gradient(z, Wbar)


def objective_components(instance: Instance, z: np.ndarray) -> dict[str, float]:
    """Return polarization, disagreement, and total PD."""

    return get_backend().objective_components(instance, z)
