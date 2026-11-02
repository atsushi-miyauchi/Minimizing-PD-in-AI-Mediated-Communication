"""Alpha parameter generation."""

from __future__ import annotations

import numpy as np
from typing import Union


RandomLike = Union[int, np.random.Generator, None]


def generate_homogeneous_alpha(n: int, *, value: float = 2.0) -> np.ndarray:
    """Generate constant alpha_v."""

    if value < 1.0:
        raise ValueError("alpha must be at least 1.")
    return np.full(_validate_n(n), float(value), dtype=float)


def generate_heterogeneous_alpha(
    n: int,
    *,
    seed: RandomLike = None,
    low: float = 1.0,
    high: float = 3.0,
) -> np.ndarray:
    """Generate alpha_v uniformly on [1, 3)."""

    if low < 1.0 or high <= low:
        raise ValueError("Require 1 <= low < high for heterogeneous alpha.")
    rng = _rng(seed)
    return rng.uniform(low, high, size=_validate_n(n))


def _rng(seed: RandomLike) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _validate_n(n: int) -> int:
    if n < 0:
        raise ValueError("n must be nonnegative.")
    return int(n)
