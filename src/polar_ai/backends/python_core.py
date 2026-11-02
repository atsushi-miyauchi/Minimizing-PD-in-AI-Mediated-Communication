"""Preserved pure-Python/SciPy numerical core.

This module contains the original implementations that predate the C++
backend. It remains independently selectable for rollback and equivalence
testing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from polar_ai.instance import Instance


def extremize(z: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Compute f_alpha_i(z_i) = clip(alpha_i z_i, -1, 1)."""

    z = np.asarray(z, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if z.shape != alpha.shape:
        raise ValueError("z and alpha must have the same shape.")
    return np.clip(alpha * z, -1.0, 1.0)


def compute_equilibrium(
    instance: Instance,
    *,
    tol: float = 1e-10,
    max_iter: int = 10_000,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute a fixed point using the preserved Python implementation."""

    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    z = instance.s.copy() if initial is None else np.asarray(initial, dtype=float).copy()
    if z.shape != (instance.n,):
        raise ValueError(f"initial must have shape ({instance.n},).")

    base = (1.0 - instance.lambda_vec) * instance.s
    residual = np.inf
    converged = False

    for iteration in range(1, max_iter + 1):
        z_next = base + instance.lambda_vec * (
            instance.P @ extremize(z, instance.alpha)
        )
        residual = float(np.linalg.norm(z_next - z, ord=np.inf))
        z = z_next
        if residual < tol:
            converged = True
            break

    diagnostics = {
        "converged": converged,
        "iterations": iteration,
        "residual_inf": residual,
        "tol": tol,
        "max_iter": max_iter,
    }
    return z, diagnostics


def polarization(z: np.ndarray) -> float:
    """Return ||z||_2^2."""

    z = np.asarray(z, dtype=float)
    return float(np.dot(z, z))


def disagreement(z: np.ndarray, Wbar: sparse.spmatrix) -> float:
    """Return sum over unordered pairs Wbar_uv (z_u - z_v)^2."""

    z = np.asarray(z, dtype=float)
    Wbar_upper = sparse.triu(Wbar.tocsr(), k=1).tocoo()
    if Wbar_upper.shape[0] != z.shape[0]:
        raise ValueError("z length must match Wbar dimensions.")
    diffs = z[Wbar_upper.row] - z[Wbar_upper.col]
    return float(np.sum(Wbar_upper.data * diffs * diffs))


def polarization_disagreement(z: np.ndarray, Wbar: sparse.spmatrix) -> float:
    """Return PD(z) = polarization(z) + disagreement(z, Wbar)."""

    return polarization(z) + disagreement(z, Wbar)


def pd_gradient(z: np.ndarray, Wbar: sparse.spmatrix) -> np.ndarray:
    """Return the gradient of PD with respect to z."""

    z = np.asarray(z, dtype=float)
    Wbar = Wbar.tocsr().astype(float)
    if Wbar.shape[0] != z.shape[0]:
        raise ValueError("z length must match Wbar dimensions.")

    weighted_degrees = np.asarray(Wbar.sum(axis=1)).ravel()
    laplacian_z = weighted_degrees * z - Wbar @ z
    return 2.0 * z + 2.0 * laplacian_z


def objective_components(instance: Instance, z: np.ndarray) -> dict[str, float]:
    """Return polarization, disagreement, and total PD."""

    pol = polarization(z)
    dis = disagreement(z, instance.Wbar)
    return {"polarization": pol, "disagreement": dis, "pd": pol + dis}
