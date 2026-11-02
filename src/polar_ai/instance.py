"""Experiment instance container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Instance:
    """All data needed by algorithms for one polarization experiment."""

    W: sparse.spmatrix
    P: sparse.spmatrix
    Wbar: sparse.spmatrix
    s: np.ndarray
    lambda_vec: np.ndarray
    alpha: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        W = self.W.tocsr().astype(float)
        P = self.P.tocsr().astype(float)
        Wbar = self.Wbar.tocsr().astype(float)
        s = np.asarray(self.s, dtype=float)
        lambda_vec = np.asarray(self.lambda_vec, dtype=float)
        alpha = np.asarray(self.alpha, dtype=float)

        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be a square sparse matrix.")
        if P.shape != W.shape:
            raise ValueError("P must have the same shape as W.")
        if Wbar.shape != W.shape:
            raise ValueError("Wbar must have the same shape as W.")

        n = W.shape[0]
        for name, vec in (("s", s), ("lambda_vec", lambda_vec), ("alpha", alpha)):
            if vec.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},).")

        object.__setattr__(self, "W", W)
        object.__setattr__(self, "P", P)
        object.__setattr__(self, "Wbar", Wbar)
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "lambda_vec", lambda_vec)
        object.__setattr__(self, "alpha", alpha)

    @property
    def n(self) -> int:
        """Number of vertices."""

        return int(self.W.shape[0])

