"""Linear-system helpers for AIS-Greedy screening."""

from __future__ import annotations

import inspect
import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

CLIPPING_TOLERANCE = 1e-12


def clipping_unsaturated_mask(
    alpha: np.ndarray,
    z: np.ndarray,
    *,
    tolerance: float = CLIPPING_TOLERANCE,
) -> np.ndarray:
    """Return coordinates strictly inside the clipping boundary."""

    alpha = np.asarray(alpha, dtype=float)
    z = np.asarray(z, dtype=float)
    if alpha.shape != z.shape:
        raise ValueError("alpha and z must have the same shape.")
    return np.abs(alpha * z) < (1.0 - float(tolerance))


def build_adjoint_matrix(
    P: sparse.spmatrix, lambda_vec: np.ndarray, d: np.ndarray
) -> sparse.csr_matrix:
    """Build ``(I - Lambda P Diag(d))^T`` as a sparse matrix."""

    n = P.shape[0]
    P_jz = P.tocsr().multiply(d)
    lambda_p_jz = P_jz.multiply(lambda_vec[:, None])
    return (sparse.eye(n, format="csr") - lambda_p_jz).T.tocsr()


def solve_adjoint(
    A_transpose: sparse.spmatrix,
    rhs: np.ndarray,
    *,
    tol: float,
    max_iter: int | None,
    solver: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    iterations = 0

    def callback(_: Any) -> None:
        nonlocal iterations
        iterations += 1

    start = time.perf_counter()
    solver = solver.lower()
    if solver == "gmres":
        solution, info = _call_iterative_solver(
            spla.gmres,
            A_transpose,
            rhs,
            tol=tol,
            max_iter=max_iter,
            callback=callback,
            callback_type="legacy",
        )
    elif solver == "bicgstab":
        solution, info = _call_iterative_solver(
            spla.bicgstab,
            A_transpose,
            rhs,
            tol=tol,
            max_iter=max_iter,
            callback=callback,
        )
    else:
        raise ValueError("solver must be 'gmres' or 'bicgstab'.")

    residual = float(np.linalg.norm(A_transpose @ solution - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    relative_residual = residual / max(rhs_norm, np.finfo(float).eps)
    diagnostics = {
        "solver": solver,
        "info": int(info),
        "converged": bool(info == 0),
        "iterations": int(iterations),
        "residual_2": residual,
        "relative_residual_2": relative_residual,
        "large_residual": bool(relative_residual > max(10.0 * tol, 1e-10)),
        "elapsed_sec": time.perf_counter() - start,
    }
    return np.asarray(solution, dtype=float), diagnostics


def _call_iterative_solver(
    solver_func: Any,
    A: sparse.spmatrix,
    b: np.ndarray,
    *,
    tol: float,
    max_iter: int | None,
    callback: Any,
    callback_type: str | None = None,
) -> tuple[np.ndarray, int]:
    signature = inspect.signature(solver_func)
    kwargs: dict[str, Any] = {"callback": callback}
    if "rtol" in signature.parameters:
        kwargs["rtol"] = tol
        kwargs["atol"] = 0.0
    else:
        kwargs["tol"] = tol
    if max_iter is not None:
        kwargs["maxiter"] = max_iter
    if callback_type is not None and "callback_type" in signature.parameters:
        kwargs["callback_type"] = callback_type
    return solver_func(A, b, **kwargs)
