"""Spectral-radius helpers for sparse nonnegative influence matrices."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import sparse


def spectral_radius_p_gamma(
    P: sparse.spmatrix,
    gamma: np.ndarray,
    *,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> float:
    """Estimate rho(P Gamma) by deterministic power iteration."""

    P = P.tocsr().astype(float)
    gamma = np.asarray(gamma, dtype=float)
    _validate_vector("gamma", gamma, P.shape[0])

    def matvec(x: np.ndarray) -> np.ndarray:
        return P @ (gamma * x)

    return _power_iteration_nonnegative(matvec, P.shape[0], tol=tol, max_iter=max_iter)


def spectral_radius_lambda_p_gamma(
    P: sparse.spmatrix,
    lambda_vec: np.ndarray,
    gamma: np.ndarray,
    *,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> float:
    """Estimate rho(Lambda P Gamma) by deterministic power iteration."""

    P = P.tocsr().astype(float)
    lambda_vec = np.asarray(lambda_vec, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    n = P.shape[0]
    _validate_vector("lambda_vec", lambda_vec, n)
    _validate_vector("gamma", gamma, n)

    def matvec(x: np.ndarray) -> np.ndarray:
        return lambda_vec * (P @ (gamma * x))

    return _power_iteration_nonnegative(matvec, n, tol=tol, max_iter=max_iter)


def homogeneous_lambda_from_p_gamma(
    P: sparse.spmatrix,
    gamma: np.ndarray,
    *,
    eps: float = 0.01,
    cap: float = 0.5,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> float:
    """Return min((1 - eps) / rho(P Gamma), cap)."""

    if not 0.0 < eps < 1.0:
        raise ValueError("eps must lie in (0, 1).")
    if cap <= 0.0:
        raise ValueError("cap must be positive.")

    rho = spectral_radius_p_gamma(P, gamma, tol=tol, max_iter=max_iter)
    if rho <= 0.0:
        return float(cap)
    return float(min((1.0 - eps) / rho, cap))


def homogeneous_lambda_vector(
    P: sparse.spmatrix,
    gamma: np.ndarray,
    *,
    eps: float = 0.01,
    cap: float = 0.5,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> np.ndarray:
    """Return a constant lambda vector using the stability rule."""

    lambda_value = homogeneous_lambda_from_p_gamma(
        P, gamma, eps=eps, cap=cap, tol=tol, max_iter=max_iter
    )
    return np.full(P.shape[0], lambda_value, dtype=float)


def _power_iteration_nonnegative(
    matvec: Callable[[np.ndarray], np.ndarray],
    n: int,
    *,
    tol: float,
    max_iter: int,
) -> float:
    if n == 0:
        return 0.0
    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    x = np.full(n, 1.0 / n, dtype=float)
    rho = 0.0

    for _ in range(max_iter):
        y = np.asarray(matvec(x), dtype=float)
        if np.any(y < -1e-14):
            raise ValueError("Power iteration requires a nonnegative matrix.")
        y = np.maximum(y, 0.0)
        y_sum = float(y.sum())
        if y_sum == 0.0:
            return 0.0

        x_next = y / y_sum
        rho_next = y_sum
        if (
            abs(rho_next - rho) <= tol * max(1.0, abs(rho_next))
            and np.linalg.norm(x_next - x, ord=np.inf) <= tol
        ):
            return float(rho_next)
        x = x_next
        rho = rho_next

    return float(rho)


def _validate_vector(name: str, vector: np.ndarray, n: int) -> None:
    if vector.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},).")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must be nonnegative.")

