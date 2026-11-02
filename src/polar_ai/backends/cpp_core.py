"""Python frontend for the compiled C++ numerical core."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from polar_ai.instance import Instance

try:
    from polar_ai import _cpp_core
except ImportError as exc:  # pragma: no cover - exercised by selector fallback
    raise ImportError(
        "The polar_ai C++ extension is not built. Run "
        "`python -m pip install -e .` or `python setup.py build_ext --inplace`."
    ) from exc


def _float64_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _csr(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    csr = matrix.tocsr().astype(np.float64, copy=False)
    if not csr.has_sorted_indices:
        csr = csr.copy()
        csr.sort_indices()
    return csr


def extremize(z: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    z = _float64_vector(z, name="z")
    alpha = _float64_vector(alpha, name="alpha")
    if z.shape != alpha.shape:
        raise ValueError("z and alpha must have the same shape.")
    out = np.empty_like(z)
    _cpp_core.extremize(z, alpha, out)
    return out


def compute_equilibrium(
    instance: Instance,
    *,
    tol: float = 1e-10,
    max_iter: int = 10_000,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if tol <= 0.0:
        raise ValueError("tol must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    P = _csr(instance.P)
    s = _float64_vector(instance.s, name="s")
    lambda_vec = _float64_vector(instance.lambda_vec, name="lambda_vec")
    alpha = _float64_vector(instance.alpha, name="alpha")
    z0 = s.copy() if initial is None else _float64_vector(initial, name="initial").copy()
    if z0.shape != (instance.n,):
        raise ValueError(f"initial must have shape ({instance.n},).")
    out = np.empty_like(z0)
    iterations, converged, residual = _cpp_core.compute_equilibrium(
        P.indptr,
        P.indices,
        P.data,
        s,
        lambda_vec,
        alpha,
        z0,
        out,
        float(tol),
        int(max_iter),
    )
    diagnostics = {
        "converged": bool(converged),
        "iterations": int(iterations),
        "residual_inf": float(residual),
        "tol": float(tol),
        "max_iter": int(max_iter),
    }
    return out, diagnostics


def polarization(z: np.ndarray) -> float:
    return float(_cpp_core.polarization(_float64_vector(z, name="z")))


def disagreement(z: np.ndarray, Wbar: sparse.spmatrix) -> float:
    z = _float64_vector(z, name="z")
    matrix = _csr(Wbar)
    if matrix.shape[0] != z.shape[0]:
        raise ValueError("z length must match Wbar dimensions.")
    return float(
        _cpp_core.disagreement(matrix.indptr, matrix.indices, matrix.data, z)
    )


def polarization_disagreement(z: np.ndarray, Wbar: sparse.spmatrix) -> float:
    return polarization(z) + disagreement(z, Wbar)


def pd_gradient(z: np.ndarray, Wbar: sparse.spmatrix) -> np.ndarray:
    z = _float64_vector(z, name="z")
    matrix = _csr(Wbar)
    if matrix.shape[0] != z.shape[0]:
        raise ValueError("z length must match Wbar dimensions.")
    out = np.empty_like(z)
    _cpp_core.pd_gradient(
        matrix.indptr, matrix.indices, matrix.data, z, out
    )
    return out


def objective_components(instance: Instance, z: np.ndarray) -> dict[str, float]:
    pol = polarization(z)
    dis = disagreement(z, instance.Wbar)
    return {"polarization": pol, "disagreement": dis, "pd": pol + dis}
