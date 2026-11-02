"""Shared helpers for the paper experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from polar_ai.equilibrium import compute_equilibrium, extremize
from polar_ai.instance import Instance
from polar_ai.objective import objective_components


@dataclass(frozen=True)
class AlgorithmResult:
    """Common return object for intervention algorithms."""

    solution: Any
    objective: float
    z_star: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)


def alpha_for_deactivation(alpha: np.ndarray, targets: Iterable[int]) -> np.ndarray:
    """Return alpha(T), where target coordinates are set to one."""

    alpha_eff = np.asarray(alpha, dtype=float).copy()
    for idx in targets:
        alpha_eff[int(idx)] = 1.0
    return alpha_eff


def instance_with_alpha(instance: Instance, alpha: np.ndarray) -> Instance:
    """Return a copy of an instance with a different alpha vector."""

    return Instance(
        W=instance.W,
        P=instance.P,
        Wbar=instance.Wbar,
        s=instance.s,
        lambda_vec=instance.lambda_vec,
        alpha=alpha,
        metadata=dict(instance.metadata),
    )


def evaluate_alpha(
    instance: Instance,
    alpha: np.ndarray,
    *,
    eq_tol: float = 1e-10,
    eq_max_iter: int = 10_000,
    eq_initial: np.ndarray | None = None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Evaluate PD at the equilibrium for a modified alpha vector."""

    start = time.perf_counter()
    modified = instance_with_alpha(instance, alpha)
    z_star, eq_diag = compute_equilibrium(
        modified, tol=eq_tol, max_iter=eq_max_iter, initial=eq_initial
    )
    components = objective_components(modified, z_star)
    diagnostics = {
        "objective": components["pd"],
        "polarization": components["polarization"],
        "disagreement": components["disagreement"],
        "equilibrium_iterations": int(eq_diag["iterations"]),
        "equilibrium_converged": bool(eq_diag["converged"]),
        "equilibrium_residual_inf": float(eq_diag["residual_inf"]),
        "elapsed_sec": time.perf_counter() - start,
    }
    return components["pd"], z_star, diagnostics


def evaluate_deactivation(
    instance: Instance,
    targets: Iterable[int],
    *,
    eq_tol: float = 1e-10,
    eq_max_iter: int = 10_000,
    eq_initial: np.ndarray | None = None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Evaluate a deactivation target set."""

    return evaluate_alpha(
        instance,
        alpha_for_deactivation(instance.alpha, targets),
        eq_tol=eq_tol,
        eq_max_iter=eq_max_iter,
        eq_initial=eq_initial,
    )


def local_pd_share_scores(instance: Instance, z: np.ndarray) -> np.ndarray:
    """Return node-level polarization-disagreement shares at ``z``.

    Each unordered disagreement contribution is split equally between its two
    endpoints and added to the endpoint's squared opinion.  Iterating
    over the upper triangle of the symmetric disagreement matrix visits every
    unordered pair exactly once, so the computation is O(|V| + |E|).
    """

    z = np.asarray(z, dtype=float)
    if z.shape != (instance.n,):
        raise ValueError(f"z must have shape ({instance.n},).")

    scores = z * z
    W_upper = sparse.triu(instance.Wbar.tocsr(), k=1).tocoo()
    diffs = z[W_upper.row] - z[W_upper.col]
    half_edge_values = 0.5 * W_upper.data * diffs * diffs
    np.add.at(scores, W_upper.row, half_edge_values)
    np.add.at(scores, W_upper.col, half_edge_values)
    return scores


def neighbor_pd_greedy_scores(
    instance: Instance,
    z: np.ndarray,
    selected: Iterable[int] = (),
) -> np.ndarray:
    """Return influence-weighted NeighborPD-Greedy screening scores.

    The local PD shares of the nodes directly influenced by each candidate are
    aggregated with the equilibrium weights ``lambda_i P_iv``.  The aggregate
    is multiplied by the magnitude of the candidate's expressed-opinion change
    under deactivation.  This is a gradient-free, one-hop score computable in
    O(|V| + |E|).
    """

    z = np.asarray(z, dtype=float)
    if z.shape != (instance.n,):
        raise ValueError(f"z must have shape ({instance.n},).")

    local_pd_shares = local_pd_share_scores(instance, z)
    affected_pd = np.asarray(
        instance.P.T @ (instance.lambda_vec * local_pd_shares),
        dtype=float,
    ).ravel()
    alpha_eff = alpha_for_deactivation(instance.alpha, selected)
    finite_gap_magnitudes = np.abs(z - extremize(z, alpha_eff))
    return finite_gap_magnitudes * affected_pd


def top_k_indices(scores: np.ndarray, k: int, *, largest: bool = True) -> list[int]:
    """Return up to k indices sorted by score, with deterministic tie-breaking.

    The selection step is linear-time on average via ``np.partition``. Only the
    selected strict-better part is sorted, and boundary ties are taken by index.
    """

    scores = np.asarray(scores, dtype=float)
    k = max(0, min(int(k), len(scores)))
    if k == 0:
        return []
    if k == len(scores):
        primary = -scores if largest else scores
        order = np.lexsort((np.arange(len(scores)), primary))
        return [int(i) for i in order]

    values = scores if largest else -scores
    threshold = float(np.partition(values, len(values) - k)[len(values) - k])
    better = np.flatnonzero(values > threshold)
    ties = np.flatnonzero(values == threshold)
    remaining = k - len(better)

    if better.size:
        better_primary = -scores[better] if largest else scores[better]
        better = better[np.lexsort((better, better_primary))]
    selected = np.concatenate([better, ties[:remaining]])
    return [int(i) for i in selected]


def score_exact_objective_spearman_correlation(
    scores: np.ndarray, exact_objectives: np.ndarray
) -> float | None:
    """Compare descending score ranks with ascending exact-objective ranks.

    A value of +1 means that the score and exact objective induce the same
    best-to-worst ranking. Ties receive their average rank. The correlation is
    undefined when fewer than two candidates are present or either ranking is
    constant.
    """

    scores = np.asarray(scores, dtype=float)
    exact_objectives = np.asarray(exact_objectives, dtype=float)
    if (
        scores.size < 2
        or exact_objectives.shape != scores.shape
        or not np.all(np.isfinite(scores))
        or not np.all(np.isfinite(exact_objectives))
    ):
        return None

    score_ranks = rankdata(-scores, method="average")
    objective_ranks = rankdata(exact_objectives, method="average")
    if np.ptp(score_ranks) == 0.0 or np.ptp(objective_ranks) == 0.0:
        return None
    return float(np.corrcoef(score_ranks, objective_ranks)[0, 1])
