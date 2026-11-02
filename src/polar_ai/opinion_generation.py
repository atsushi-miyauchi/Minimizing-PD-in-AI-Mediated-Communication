"""Innate opinion generation."""

from __future__ import annotations

import numpy as np
from scipy.stats import truncnorm
from typing import Union


RandomLike = Union[int, np.random.Generator, None]


def generate_uniform_opinions(n: int, *, seed: RandomLike = None) -> np.ndarray:
    """Generate s_v uniformly on [-1, 1]."""

    rng = _rng(seed)
    return rng.uniform(-1.0, 1.0, size=_validate_n(n))


def generate_polarized_opinions(
    n: int,
    *,
    seed: RandomLike = None,
    mean: float = 0.5,
    std: float = 0.15,
) -> np.ndarray:
    """Generate signed truncated-normal polarized opinions in [-1, 1]."""

    if std <= 0:
        raise ValueError("std must be positive.")
    n = _validate_n(n)
    rng = _rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    magnitudes = truncnorm.rvs(
        lower,
        upper,
        loc=mean,
        scale=std,
        size=n,
        random_state=rng,
    )
    return signs * np.asarray(magnitudes, dtype=float)


def _rng(seed: RandomLike) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _validate_n(n: int) -> int:
    if n < 0:
        raise ValueError("n must be nonnegative.")
    return int(n)
