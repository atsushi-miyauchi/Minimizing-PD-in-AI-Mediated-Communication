"""Deactivation algorithms used by the paper experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from polar_ai.algorithms.common import (
    AlgorithmResult,
    alpha_for_deactivation,
    evaluate_deactivation,
    neighbor_pd_greedy_scores,
    score_exact_objective_spearman_correlation,
    top_k_indices,
)
from polar_ai.equilibrium import extremize
from polar_ai.instance import Instance
from polar_ai.objective import pd_gradient
from polar_ai.sensitivity import (
    CLIPPING_TOLERANCE,
    build_adjoint_matrix,
    clipping_unsaturated_mask,
    solve_adjoint,
)


@dataclass(frozen=True)
class ScreeningScoreResult:
    """Screening scores and their intermediate quantities."""

    scores: np.ndarray
    finite_gaps: np.ndarray
    propagation: np.ndarray
    adjoint: np.ndarray
    unsaturated: np.ndarray
    diagnostics: dict[str, Any]


class CandidateEvaluationTimeout(TimeoutError):
    """Raised when a candidate scan reaches its deadline mid-iteration."""

    def __init__(
        self,
        evaluated_count: int,
        equilibrium_iterations: list[int],
    ) -> None:
        super().__init__("Candidate evaluation deadline exceeded.")
        self.evaluated_count = int(evaluated_count)
        self.equilibrium_iterations = list(equilibrium_iterations)


def no_intervention(
    instance: Instance,
    *,
    eq_tol: float = 1e-10,
    eq_max_iter: int = 10_000,
) -> AlgorithmResult:
    """Select no deactivated nodes."""

    return _result_for_set(
        instance, set(), "No-Intervention", eq_tol=eq_tol, eq_max_iter=eq_max_iter
    )


def greedy(
    instance: Instance,
    b: int,
    *,
    eq_tol: float = 1e-10,
    eq_max_iter: int = 10_000,
) -> AlgorithmResult:
    """Naive greedy deactivation."""

    selected: set[int] = set()
    best_set: set[int] = set()
    best_obj, best_z, best_diag = evaluate_deactivation(
        instance, best_set, eq_tol=eq_tol, eq_max_iter=eq_max_iter
    )
    current_z = best_z.copy()
    logs = [
        _log_row("Greedy", best_obj, best_diag, iteration=-1, set_size=0)
    ]

    for iteration in range(min(max(int(b), 0), instance.n)):
        candidates = [v for v in range(instance.n) if v not in selected]
        if not candidates:
            break

        obj, v, z, diag, _ = evaluate_best_deactivation_candidate(
            instance, selected, candidates, eq_initial=current_z,
            eq_tol=eq_tol, eq_max_iter=eq_max_iter,
        )
        selected.add(v)
        current_z = z
        logs.append(
            _log_row(
                "Greedy",
                obj,
                diag,
                iteration=iteration,
                selected_node=v,
                set_size=len(selected),
            )
        )
        if obj < best_obj:
            best_obj, best_z, best_diag = obj, z, diag
            best_set = set(selected)

    return AlgorithmResult(
        best_set,
        best_obj,
        best_z,
        {"method": "Greedy", "best": best_diag},
        logs,
    )


def approximate_improvement_scores(
    instance: Instance,
    z_star: np.ndarray,
    selected: set[int],
    *,
    clipping_tolerance: float = CLIPPING_TOLERANCE,
    linear_tol: float = 1e-8,
    linear_max_iter: int | None = None,
    solver: str = "gmres",
) -> ScreeningScoreResult:
    """Compute AIS scores with the mediation Jacobian approximated by I."""

    return _screening_scores(
        instance,
        z_star,
        selected,
        clipping_tolerance=clipping_tolerance,
        linear_tol=linear_tol,
        linear_max_iter=linear_max_iter,
        solver=solver,
    )


def neighbor_pd_greedy_screening_scores(
    instance: Instance,
    z_star: np.ndarray,
    selected: set[int],
    *,
    clipping_tolerance: float = CLIPPING_TOLERANCE,
    linear_tol: float = 1e-8,
    linear_max_iter: int | None = None,
    solver: str = "gmres",
) -> ScreeningScoreResult:
    """Compute gradient-free NeighborPD scores for candidate screening."""

    del clipping_tolerance, linear_tol, linear_max_iter, solver
    start = time.perf_counter()
    scores = neighbor_pd_greedy_scores(instance, z_star, selected)
    zeros = np.zeros(instance.n, dtype=float)
    linear_diag = {
        "solver": "none",
        "info": 0,
        "converged": True,
        "iterations": 0,
        "residual_2": 0.0,
        "relative_residual_2": 0.0,
        "large_residual": False,
        "elapsed_sec": 0.0,
    }
    return ScreeningScoreResult(
        scores=scores,
        finite_gaps=np.abs(
            z_star
            - extremize(
                z_star,
                alpha_for_deactivation(instance.alpha, selected),
            )
        ),
        propagation=zeros.copy(),
        adjoint=zeros,
        unsaturated=np.ones(instance.n, dtype=bool),
        diagnostics={
            "linear_solver": linear_diag,
            "adjoint_converged": True,
            "adjoint_residual_2": 0.0,
            "adjoint_relative_residual_2": 0.0,
            "adjoint_large_residual": False,
            "saturated_count": 0,
            "screening_score_norm_inf": (
                float(np.linalg.norm(scores, ord=np.inf))
                if scores.size
                else 0.0
            ),
            "positive_screening_score_count": int(
                np.count_nonzero(scores > 0.0)
            ),
            "score_variant": "neighbor_pd_greedy",
            "elapsed_sec": time.perf_counter() - start,
        },
    )


def _screening_scores(
    instance: Instance,
    z_star: np.ndarray,
    selected: set[int],
    *,
    clipping_tolerance: float,
    linear_tol: float,
    linear_max_iter: int | None,
    solver: str,
) -> ScreeningScoreResult:
    """Compute AIS approximate-improvement scores."""

    start = time.perf_counter()
    z_star = np.asarray(z_star, dtype=float)
    if z_star.shape != (instance.n,):
        raise ValueError(f"z_star must have shape ({instance.n},).")

    alpha_t = alpha_for_deactivation(instance.alpha, selected)
    unsaturated = clipping_unsaturated_mask(
        alpha_t, z_star, tolerance=clipping_tolerance
    )
    gradient = pd_gradient(z_star, instance.Wbar)
    finite_gaps = z_star - extremize(z_star, alpha_t)

    adjoint_matrix = build_adjoint_matrix(
        instance.P,
        instance.lambda_vec,
        np.ones(instance.n, dtype=float),
    )
    adjoint, linear_diag = solve_adjoint(
        adjoint_matrix,
        gradient,
        tol=linear_tol,
        max_iter=linear_max_iter,
        solver=solver,
    )
    propagation = np.asarray(
        instance.P.T @ (instance.lambda_vec * adjoint), dtype=float
    ).ravel()

    scores = -finite_gaps * propagation
    remaining = np.array(
        [v for v in range(instance.n) if v not in selected], dtype=int
    )
    remaining_scores = scores[remaining]
    diagnostics = {
        "linear_solver": linear_diag,
        "adjoint_converged": bool(linear_diag["converged"]),
        "adjoint_residual_2": float(linear_diag["residual_2"]),
        "adjoint_relative_residual_2": float(
            linear_diag["relative_residual_2"]
        ),
        "adjoint_large_residual": bool(linear_diag["large_residual"]),
        "saturated_count": int(np.count_nonzero(~unsaturated)),
        "screening_score_norm_inf": (
            float(np.linalg.norm(remaining_scores, ord=np.inf))
            if remaining_scores.size
            else 0.0
        ),
        "positive_screening_score_count": int(
            np.count_nonzero(remaining_scores > 0.0)
        ),
        "score_variant": "ais_greedy",
        "elapsed_sec": time.perf_counter() - start,
    }
    return ScreeningScoreResult(
        scores=np.asarray(scores, dtype=float),
        finite_gaps=np.asarray(finite_gaps, dtype=float),
        propagation=propagation,
        adjoint=np.asarray(adjoint, dtype=float),
        unsaturated=np.asarray(unsaturated, dtype=bool),
        diagnostics=diagnostics,
    )


def screening_candidate_set(
    scores: np.ndarray,
    remaining: np.ndarray,
    k: int,
) -> list[int]:
    """Return the top-k remaining nodes by largest screening score."""

    if int(k) < 1:
        raise ValueError("k must be at least one.")
    remaining = np.sort(np.asarray(remaining, dtype=int))
    candidate_count = min(int(k), len(remaining))
    return _top_k_from_remaining_scores(
        np.asarray(scores, dtype=float),
        remaining,
        candidate_count,
        largest=True,
    )


def ais_greedy(
    instance: Instance,
    b: int,
    *,
    k: int = 10,
    eq_tol: float = 1e-10,
    eq_max_iter: int = 10_000,
    linear_tol: float = 1e-8,
    linear_max_iter: int | None = None,
    solver: str = "gmres",
    clipping_tolerance: float = CLIPPING_TOLERANCE,
    timeout_sec: float | None = None,
) -> AlgorithmResult:
    """Approximate-improvement-screened greedy deactivation."""

    return _approximate_improvement_screened_greedy(
        instance,
        b,
        method="AIS-Greedy",
        internal_identifier="ais_greedy",
        score_function=approximate_improvement_scores,
        k=k,
        eq_tol=eq_tol,
        eq_max_iter=eq_max_iter,
        linear_tol=linear_tol,
        linear_max_iter=linear_max_iter,
        solver=solver,
        clipping_tolerance=clipping_tolerance,
        timeout_sec=timeout_sec,
    )


def neighbor_pd_greedy(
    instance: Instance,
    b: int,
    *,
    k: int = 10,
    eq_tol: float = 1e-10,
    eq_max_iter: int = 10_000,
    linear_tol: float = 1e-8,
    linear_max_iter: int | None = None,
    solver: str = "gmres",
    clipping_tolerance: float = CLIPPING_TOLERANCE,
    timeout_sec: float | None = None,
) -> AlgorithmResult:
    """NeighborPD-screened exact greedy deactivation."""

    return _approximate_improvement_screened_greedy(
        instance,
        b,
        method="NeighborPD-Greedy",
        internal_identifier="neighbor_pd_greedy",
        score_function=neighbor_pd_greedy_screening_scores,
        k=k,
        eq_tol=eq_tol,
        eq_max_iter=eq_max_iter,
        linear_tol=linear_tol,
        linear_max_iter=linear_max_iter,
        solver=solver,
        clipping_tolerance=clipping_tolerance,
        timeout_sec=timeout_sec,
    )


def _approximate_improvement_screened_greedy(
    instance: Instance,
    b: int,
    *,
    method: str,
    internal_identifier: str,
    score_function: Any,
    k: int,
    eq_tol: float,
    eq_max_iter: int,
    linear_tol: float,
    linear_max_iter: int | None,
    solver: str,
    clipping_tolerance: float,
    timeout_sec: float | None,
) -> AlgorithmResult:
    """Shared exact screened-greedy loop for deterministic score oracles."""

    if int(k) < 1:
        raise ValueError("k must be at least one.")
    start = time.perf_counter()
    selected: set[int] = set()
    best_set: set[int] = set()
    prefix_obj, current_z, initial_diag = evaluate_deactivation(
        instance, selected, eq_tol=eq_tol, eq_max_iter=eq_max_iter
    )
    best_obj = float(prefix_obj)
    best_z = current_z.copy()
    best_diag: dict[str, Any] = dict(initial_diag)
    cumulative_eq_iters = int(initial_diag.get("equilibrium_iterations", 0))
    cumulative_linear_iters = 0
    logs: list[dict[str, Any]] = [
        {
            "method": method,
            "iteration": 0,
            "budget": 0,
            "set_size": 0,
            "prefix_set": [],
            "incumbent_set": [],
            "objective": float(prefix_obj),
            "incumbent_objective": float(best_obj),
            "elapsed_sec": time.perf_counter() - start,
            "equilibrium_iterations": cumulative_eq_iters,
            "candidate_equilibrium_iterations": [],
            "linear_solver_iterations": 0,
            "adjoint_converged": None,
            "adjoint_residual_2": 0.0,
            "adjoint_relative_residual_2": 0.0,
            "adjoint_large_residual": False,
            "saturated_count": 0,
            "candidate_set_size": 0,
            "candidate_nodes": [],
            "selected_node": None,
            "selected_node_approximate_improvement_score": 0.0,
            "selected_node_finite_gap": 0.0,
            "selected_node_approximate_improvement_rank": None,
            "approximate_improvement_score_norm_inf": 0.0,
            "positive_approximate_improvement_count": 0,
            "selected_node_local_pd_score": None,
            "selected_node_local_pd_rank": None,
            "local_pd_score_norm_inf": 0.0,
            "positive_local_pd_count": 0,
            "selected_node_neighbor_pd_score": None,
            "selected_node_neighbor_pd_rank": None,
            "neighbor_pd_score_norm_inf": 0.0,
            "positive_neighbor_pd_count": 0,
            "screened_saturated_fraction": 0.0,
            "score_gain_correlation": None,
            "score_exact_objective_spearman_correlation": None,
            "feasibility_violation": 0.0,
            "timed_out": False,
        }
    ]

    for iteration in range(min(max(int(b), 0), instance.n)):
        remaining = np.array(
            [v for v in range(instance.n) if v not in selected], dtype=int
        )
        if remaining.size == 0:
            break
        score_result = score_function(
            instance,
            current_z,
            selected,
            clipping_tolerance=clipping_tolerance,
            linear_tol=linear_tol,
            linear_max_iter=linear_max_iter,
            solver=solver,
        )
        linear_diag = score_result.diagnostics["linear_solver"]
        cumulative_linear_iters += int(linear_diag["iterations"])
        candidates = screening_candidate_set(score_result.scores, remaining, k)
        common_initial = current_z
        evaluations = evaluate_deactivation_candidates(
            instance,
            selected,
            candidates,
            eq_initial=common_initial,
            eq_tol=eq_tol,
            eq_max_iter=eq_max_iter,
        )
        candidate_eq_iters = [
            int(diag.get("equilibrium_iterations", 0))
            for _, _, _, diag in evaluations
        ]
        cumulative_eq_iters += sum(candidate_eq_iters)
        prefix_obj, selected_node, selected_z, selected_diag = min(
            evaluations, key=lambda item: (item[0], item[1])
        )
        exact_gains = np.asarray(
            [float(logs[-1]["objective"]) - objective
             for objective, _, _, _ in evaluations],
            dtype=float,
        )
        screened_scores = np.asarray(
            [score_result.scores[v] for _, v, _, _ in evaluations], dtype=float
        )
        exact_objectives = np.asarray(
            [objective for objective, _, _, _ in evaluations], dtype=float
        )
        correlation = _safe_correlation(screened_scores, exact_gains)
        spearman_correlation = score_exact_objective_spearman_correlation(
            screened_scores, exact_objectives
        )
        selected.add(selected_node)
        current_z = selected_z
        if prefix_obj < best_obj:
            best_obj = float(prefix_obj)
            best_z = current_z.copy()
            best_diag = dict(selected_diag)
            best_set = set(selected)

        ranked_remaining = _top_k_from_remaining_scores(
            score_result.scores, remaining, len(remaining), largest=True
        )
        screened_saturated = np.count_nonzero(
            ~score_result.unsaturated[np.asarray(candidates, dtype=int)]
        )
        elapsed_sec = time.perf_counter() - start
        timed_out = (
            timeout_sec is not None
            and float(timeout_sec) > 0.0
            and elapsed_sec > float(timeout_sec)
        )
        logs.append(
            {
                "method": method,
                "iteration": iteration + 1,
                "budget": iteration + 1,
                "set_size": len(selected),
                "prefix_set": sorted(selected),
                "incumbent_set": sorted(best_set),
                "objective": float(prefix_obj),
                "incumbent_objective": float(best_obj),
                "elapsed_sec": elapsed_sec,
                "equilibrium_iterations": cumulative_eq_iters,
                "candidate_equilibrium_iterations": candidate_eq_iters,
                "linear_solver_iterations": cumulative_linear_iters,
                "adjoint_converged": bool(linear_diag["converged"]),
                "adjoint_residual_2": float(linear_diag["residual_2"]),
                "adjoint_relative_residual_2": float(
                    linear_diag["relative_residual_2"]
                ),
                "adjoint_large_residual": bool(linear_diag["large_residual"]),
                "saturated_count": int(
                    score_result.diagnostics["saturated_count"]
                ),
                "candidate_set_size": len(candidates),
                "candidate_nodes": list(candidates),
                "selected_node": int(selected_node),
                "selected_node_approximate_improvement_score": float(
                    score_result.scores[selected_node]
                ),
                "selected_node_finite_gap": float(
                    score_result.finite_gaps[selected_node]
                ),
                "selected_node_approximate_improvement_rank": (
                    ranked_remaining.index(selected_node) + 1
                ),
                "approximate_improvement_score_norm_inf": float(
                    score_result.diagnostics["screening_score_norm_inf"]
                ),
                "positive_approximate_improvement_count": int(
                    score_result.diagnostics["positive_screening_score_count"]
                ),
                "selected_node_neighbor_pd_score": (
                    float(score_result.scores[selected_node])
                    if internal_identifier == "neighbor_pd_greedy"
                    else None
                ),
                "selected_node_neighbor_pd_rank": (
                    ranked_remaining.index(selected_node) + 1
                    if internal_identifier == "neighbor_pd_greedy"
                    else None
                ),
                "neighbor_pd_score_norm_inf": (
                    float(
                        score_result.diagnostics[
                            "screening_score_norm_inf"
                        ]
                    )
                    if internal_identifier == "neighbor_pd_greedy"
                    else 0.0
                ),
                "positive_neighbor_pd_count": (
                    int(
                        score_result.diagnostics[
                            "positive_screening_score_count"
                        ]
                    )
                    if internal_identifier == "neighbor_pd_greedy"
                    else 0
                ),
                "screened_saturated_fraction": (
                    float(screened_saturated / len(candidates))
                    if candidates
                    else 0.0
                ),
                "score_gain_correlation": correlation,
                "score_exact_objective_spearman_correlation": (
                    spearman_correlation
                ),
                "feasibility_violation": 0.0,
                "timed_out": timed_out,
            }
        )
        if timed_out:
            break

    return AlgorithmResult(
        best_set,
        best_obj,
        best_z,
        {
            "method": method,
            "internal_identifier": internal_identifier,
            "short_name": (
                "AIS" if internal_identifier == "ais_greedy"
                else method.removesuffix("-Greedy")
            ),
            "k": int(k),
            "clipping_tolerance": float(clipping_tolerance),
            "best": best_diag,
        },
        logs,
    )


def usampling_greedy(
    instance: Instance,
    b: int,
    *,
    k: int = 10,
    seed: int | np.random.Generator | None = None,
    eq_tol: float = 1e-10,
    eq_max_iter: int = 10_000,
) -> AlgorithmResult:
    """Uniform-sampling-screened greedy deactivation."""

    rng = _rng(seed)
    selected: set[int] = set()
    best_set: set[int] = set()
    best_obj, best_z, best_diag = evaluate_deactivation(
        instance, best_set, eq_tol=eq_tol, eq_max_iter=eq_max_iter
    )
    current_z = best_z.copy()
    logs = [
        _log_row(
            "USampling-Greedy", best_obj, best_diag, iteration=-1, set_size=0
        )
    ]

    for iteration in range(min(max(int(b), 0), instance.n)):
        remaining = np.array([v for v in range(instance.n) if v not in selected])
        if remaining.size == 0:
            break

        candidate_count = min(max(int(k), 1), len(remaining))
        screened = [
            int(v)
            for v in rng.choice(remaining, size=candidate_count, replace=False)
        ]

        obj, v, z, diag, _ = evaluate_best_deactivation_candidate(
            instance, selected, screened, eq_initial=current_z,
            eq_tol=eq_tol, eq_max_iter=eq_max_iter,
        )
        selected.add(v)
        current_z = z
        logs.append(
            _log_row(
                "USampling-Greedy",
                obj,
                diag,
                iteration=iteration,
                selected_node=v,
                set_size=len(selected),
            )
        )
        if obj < best_obj:
            best_obj, best_z, best_diag = obj, z, diag
            best_set = set(selected)

    return AlgorithmResult(
        best_set,
        best_obj,
        best_z,
        {"method": "USampling-Greedy", "k": k, "best": best_diag},
        logs,
    )


def evaluate_best_deactivation_candidate(
    instance: Instance,
    selected: set[int],
    candidates: list[int],
    *,
    eq_initial: np.ndarray,
    eq_tol: float,
    eq_max_iter: int,
    deadline: float | None = None,
) -> tuple[float, int, np.ndarray, dict[str, Any], list[int]]:
    """Evaluate candidates while retaining only the current best equilibrium."""

    best: tuple[float, int, np.ndarray, dict[str, Any]] | None = None
    equilibrium_iterations: list[int] = []
    for candidate in candidates:
        if deadline is not None and time.perf_counter() >= deadline:
            raise CandidateEvaluationTimeout(
                len(equilibrium_iterations), equilibrium_iterations
            )
        v = int(candidate)
        objective, z_star, diagnostics = evaluate_deactivation(
            instance,
            selected | {v},
            eq_tol=eq_tol,
            eq_max_iter=eq_max_iter,
            eq_initial=eq_initial,
        )
        equilibrium_iterations.append(
            int(diagnostics.get("equilibrium_iterations", 0))
        )
        evaluation = (objective, v, z_star, diagnostics)
        if best is None or (objective, v) < (best[0], best[1]):
            best = evaluation
    if best is None:
        raise ValueError("At least one candidate is required.")
    return (*best, equilibrium_iterations)


def evaluate_deactivation_candidates(
    instance: Instance,
    selected: set[int],
    candidates: list[int],
    *,
    eq_initial: np.ndarray,
    eq_tol: float,
    eq_max_iter: int,
) -> list[tuple[float, int, np.ndarray, dict[str, Any]]]:
    """Evaluate every candidate from the same equilibrium warm start."""

    evaluations: list[tuple[float, int, np.ndarray, dict[str, Any]]] = []
    for candidate in candidates:
        v = int(candidate)
        objective, z_star, diagnostics = evaluate_deactivation(
            instance, selected | {v}, eq_tol=eq_tol, eq_max_iter=eq_max_iter,
            eq_initial=eq_initial,
        )
        evaluations.append((objective, v, z_star, diagnostics))
    if not evaluations:
        raise ValueError("At least one candidate is required.")
    return evaluations


def _result_for_set(
    instance: Instance,
    selected: set[int],
    method: str,
    *,
    eq_tol: float,
    eq_max_iter: int,
) -> AlgorithmResult:
    start = time.perf_counter()
    objective, z_star, diag = evaluate_deactivation(
        instance, selected, eq_tol=eq_tol, eq_max_iter=eq_max_iter
    )
    elapsed = time.perf_counter() - start
    log = _log_row(method, objective, diag, iteration=0, elapsed_sec=elapsed, set_size=len(selected))
    return AlgorithmResult(selected, objective, z_star, {"method": method}, [log])


def _log_row(
    method: str,
    objective: float,
    diag: dict[str, Any],
    *,
    iteration: int,
    elapsed_sec: float | None = None,
    set_size: int,
    selected_node: int | None = None,
    linear_solver_iterations: int = 0,
    sensitivity_norm_inf: float = 0.0,
    global_influence_norm_inf: float = 0.0,
    dgi_score_norm_inf: float = 0.0,
    candidate_nodes: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "method": method,
        "iteration": iteration,
        "objective": float(objective),
        "elapsed_sec": float(diag.get("elapsed_sec", 0.0) if elapsed_sec is None else elapsed_sec),
        "equilibrium_iterations": int(diag.get("equilibrium_iterations", 0)),
        "linear_solver_iterations": int(linear_solver_iterations),
        "sensitivity_norm_inf": float(sensitivity_norm_inf),
        "global_influence_norm_inf": float(global_influence_norm_inf),
        "dgi_score_norm_inf": float(dgi_score_norm_inf),
        "feasibility_violation": 0.0,
        "set_size": int(set_size),
        "selected_node": selected_node,
        "candidate_nodes": [] if candidate_nodes is None else list(candidate_nodes),
    }


def _top_k_from_remaining_scores(
    scores: np.ndarray, remaining: np.ndarray, count: int, *, largest: bool
) -> list[int]:
    local_indices = top_k_indices(scores[remaining], count, largest=largest)
    return [int(remaining[idx]) for idx in local_indices]


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return Pearson correlation when both vectors have nonzero variance."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size != x.size:
        return None
    if np.ptp(x) <= np.finfo(float).eps or np.ptp(y) <= np.finfo(float).eps:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)

__all__ = [
    "CandidateEvaluationTimeout",
    "ais_greedy",
    "evaluate_best_deactivation_candidate",
    "greedy",
    "neighbor_pd_greedy",
    "no_intervention",
    "usampling_greedy",
]
