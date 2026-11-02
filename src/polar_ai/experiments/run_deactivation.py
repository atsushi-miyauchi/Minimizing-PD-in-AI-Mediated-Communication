"""Run the deactivation experiment and write its CSV output."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from polar_ai.algorithms import deactivation
from polar_ai.algorithms.common import (
    AlgorithmResult,
    evaluate_alpha,
    evaluate_deactivation,
)
from polar_ai.experiments.utils import (
    algorithm_timeout_sec,
    build_instance,
    budget_values,
    common_params,
    dataset_configs,
    default_output_path,
    dumps_json,
    load_config,
    load_dataset,
    method_allowed,
    note_timeout,
    parse_args,
    save_rows,
    sensitivity_params,
    setting_pairs,
    solution_indices,
    solution_labels,
    spectral_params,
    stable_seed,
    timed_call,
)
from polar_ai.instance import Instance


def main(argv: list[str] | None = None) -> int:
    args = parse_args("Run deactivation experiments.", argv)
    config, base_dir = load_config(args.config)
    output_path = default_output_path(config, base_dir, "deactivation.csv", args.output)
    run_config(config, base_dir, output_path)
    print(output_path)
    return 0


def run_config(
    config: dict[str, Any], base_dir: Path, output_path: Path
) -> Path:
    """Run the configured deactivation experiments into one CSV."""

    rows: list[dict[str, Any]] = []
    seed = int(config.get("seed", 0))
    eps, lambda_cap = spectral_params(config)
    for dataset in dataset_configs(config, base_dir):
        graph = load_dataset(dataset)
        for opinion_setting, alpha_setting in setting_pairs(config):
            instance = build_instance(
                graph,
                dataset_name=dataset.name,
                opinion_setting=opinion_setting,
                alpha_setting=alpha_setting,
                seed=seed,
                eps=eps,
                lambda_cap=lambda_cap,
            )
            _print_progress(
                dataset.name,
                opinion_setting,
                alpha_setting,
                "No-Intervention",
            )
            baseline, baseline_runtime = timed_call(
                deactivation.no_intervention, instance, **common_params(config)
            )
            rows.extend(
                _run_budgets(
                    config,
                    instance,
                    dataset.name,
                    opinion_setting,
                    alpha_setting,
                    budget_values(config, instance.n),
                    baseline.objective,
                    baseline_runtime,
                    seed,
                    {},
                )
            )

    return save_rows(rows, output_path)


def _run_budgets(
    config: dict[str, Any],
    instance: Instance,
    dataset_name: str,
    opinion_setting: str,
    alpha_setting: str,
    budgets: list[int],
    baseline_objective: float,
    baseline_runtime: float,
    seed: int,
    timeout_cutoffs: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not budgets:
        return rows

    common = common_params(config)
    sens = sensitivity_params(config)
    usampling_k_values = _usampling_greedy_k_values(config)
    random_repetitions = _usampling_random_repetitions(config)
    neighbor_pd_k_values = _neighbor_pd_greedy_k_values(config)
    ais_k_values = _ais_greedy_k_values(config)
    timeout_sec = algorithm_timeout_sec(config)

    screened_rows: list[dict[str, Any]] = []
    for usampling_k in usampling_k_values:
        for repetition in range(random_repetitions):
            screened_rows.extend(
                _usampling_greedy_prefix_rows(
                    instance,
                    budgets,
                    usampling_k,
                    method=f"USampling-Greedy (k={usampling_k})",
                    baseline_objective=baseline_objective,
                    common=common,
                    timeout_sec=timeout_sec,
                    timeout_cutoffs=timeout_cutoffs,
                    dataset=dataset_name,
                    opinion_setting=opinion_setting,
                    alpha_setting=alpha_setting,
                    repetition=repetition,
                    seed=stable_seed(
                        seed,
                        dataset_name,
                        opinion_setting,
                        alpha_setting,
                        usampling_k,
                        repetition,
                        "usampling_greedy",
                    ),
                )
            )
    deterministic_screened_rows: list[dict[str, Any]] = []
    for neighbor_pd_k in neighbor_pd_k_values:
        deterministic_screened_rows.extend(
            _ais_greedy_prefix_rows(
                instance,
                budgets,
                neighbor_pd_k,
                method=f"NeighborPD-Greedy (k={neighbor_pd_k})",
                algorithm=deactivation.neighbor_pd_greedy,
                baseline_objective=baseline_objective,
                common=common,
                sens=sens,
                timeout_sec=timeout_sec,
                timeout_cutoffs=timeout_cutoffs,
                dataset=dataset_name,
                opinion_setting=opinion_setting,
                alpha_setting=alpha_setting,
            )
        )
    for ais_k in ais_k_values:
        deterministic_screened_rows.extend(
            _ais_greedy_prefix_rows(
                instance,
                budgets,
                ais_k,
                method=f"AIS-Greedy (k={ais_k})",
                algorithm=deactivation.ais_greedy,
                baseline_objective=baseline_objective,
                common=common,
                sens=sens,
                timeout_sec=timeout_sec,
                timeout_cutoffs=timeout_cutoffs,
                dataset=dataset_name,
                opinion_setting=opinion_setting,
                alpha_setting=alpha_setting,
            )
        )
    greedy_rows = _greedy_prefix_rows(
        instance,
        budgets,
        "Greedy",
        baseline_objective=baseline_objective,
        common=common,
        timeout_sec=timeout_sec,
        timeout_cutoffs=timeout_cutoffs,
        dataset=dataset_name,
        opinion_setting=opinion_setting,
        alpha_setting=alpha_setting,
    )

    _print_progress(
        dataset_name,
        opinion_setting,
        alpha_setting,
        "No-AI-Mediation",
    )
    no_ai_result, no_ai_runtime = _evaluate_no_ai_mediation(instance, common)
    for b in budgets:
        rows.append(
            _base_row(
                instance,
                dataset_name,
                opinion_setting,
                alpha_setting,
                b,
                "No-AI-Mediation",
                baseline_objective,
                no_ai_result.objective,
                no_ai_runtime,
                instance.n,
                repetition=-1,
                diagnostics=no_ai_result.diagnostics,
                selected_indices=list(range(instance.n)),
                log_summary=_summarize_logs(no_ai_result.logs),
            )
        )

    rows.extend(screened_rows)
    rows.extend(deterministic_screened_rows)
    rows.extend(greedy_rows)
    return rows


def _greedy_prefix_rows(
    instance: Instance,
    budgets: list[int],
    method: str,
    *,
    baseline_objective: float,
    common: dict[str, Any],
    timeout_sec: float,
    timeout_cutoffs: dict[str, dict[str, int]],
    dataset: str,
    opinion_setting: str,
    alpha_setting: str,
) -> list[dict[str, Any]]:
    _print_progress(
        dataset, opinion_setting, alpha_setting, method, budget=max(budgets)
    )
    rows: list[dict[str, Any]] = []
    selected: set[int] = set()
    best_set: set[int] = set()
    best_obj, current_z, baseline_diag = evaluate_deactivation(
        instance, best_set, **common
    )
    cumulative_runtime = 0.0
    cumulative_eq_iters = int(baseline_diag.get("equilibrium_iterations", 0))

    for step in range(1, min(max(budgets), instance.n) + 1):
        if not method_allowed(timeout_cutoffs, dataset, method, step):
            break
        step_start = time.perf_counter()
        candidates = [v for v in range(instance.n) if v not in selected]
        deadline = (
            step_start + max(0.0, timeout_sec - cumulative_runtime)
            if timeout_sec > 0.0
            else None
        )
        try:
            obj, v, selected_z, _, candidate_eq_iters = (
                deactivation.evaluate_best_deactivation_candidate(
                    instance,
                    selected,
                    candidates,
                    eq_initial=current_z,
                    deadline=deadline,
                    **common,
                )
            )
        except deactivation.CandidateEvaluationTimeout as exc:
            cumulative_runtime += time.perf_counter() - step_start
            cumulative_runtime = max(
                cumulative_runtime,
                float(np.nextafter(timeout_sec, np.inf)),
            )
            cumulative_eq_iters += sum(exc.equilibrium_iterations)
            rows.append(
                _prefix_row(
                    instance,
                    dataset,
                    opinion_setting,
                    alpha_setting,
                    step,
                    method,
                    baseline_objective,
                    best_obj,
                    cumulative_runtime,
                    best_set,
                    cumulative_eq_iters,
                    linear_solver_iterations=0,
                    timed_out=True,
                    extra_log_summary={
                        "candidate_set_size": exc.evaluated_count,
                    },
                    extra_diagnostics={
                        "iteration_completed": False,
                        "evaluated_candidate_count": exc.evaluated_count,
                    },
                )
            )
            note_timeout(
                timeout_cutoffs,
                dataset,
                method,
                step,
                cumulative_runtime,
                timeout_sec,
            )
            break
        cumulative_eq_iters += sum(candidate_eq_iters)
        current_z = selected_z
        selected.add(v)
        cumulative_runtime += time.perf_counter() - step_start
        if obj < best_obj:
            best_obj = obj
            best_set = set(selected)
        timed_out = timeout_sec > 0.0 and cumulative_runtime > timeout_sec
        if step in budgets:
            rows.append(
                _prefix_row(
                    instance,
                    dataset,
                    opinion_setting,
                    alpha_setting,
                    step,
                    method,
                    baseline_objective,
                    best_obj,
                    cumulative_runtime,
                    best_set,
                    cumulative_eq_iters,
                    linear_solver_iterations=0,
                    timed_out=timed_out,
                )
            )
        if note_timeout(
            timeout_cutoffs, dataset, method, step, cumulative_runtime, timeout_sec
        ):
            break
    return rows


def _usampling_greedy_prefix_rows(
    instance: Instance,
    budgets: list[int],
    ss_k: int,
    method: str,
    *,
    baseline_objective: float,
    common: dict[str, Any],
    timeout_sec: float,
    timeout_cutoffs: dict[str, dict[str, int]],
    dataset: str,
    opinion_setting: str,
    alpha_setting: str,
    repetition: int,
    seed: int,
) -> list[dict[str, Any]]:
    _print_progress(
        dataset,
        opinion_setting,
        alpha_setting,
        method,
        budget=max(budgets),
        repetition=repetition,
    )
    timeout_method = f"{method} [repetition={repetition}]"
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    best_set: set[int] = set()
    best_obj, current_z, baseline_diag = evaluate_deactivation(
        instance, best_set, **common
    )
    cumulative_runtime = 0.0
    cumulative_eq_iters = int(baseline_diag.get("equilibrium_iterations", 0))

    for step in range(1, min(max(budgets), instance.n) + 1):
        if not method_allowed(timeout_cutoffs, dataset, timeout_method, step):
            break
        step_start = time.perf_counter()
        remaining = np.array([v for v in range(instance.n) if v not in selected])
        candidate_count = min(max(int(ss_k), 1), len(remaining))
        candidates = [
            int(v)
            for v in rng.choice(remaining, size=candidate_count, replace=False)
        ]

        obj, v, current_z, _, candidate_eq_iters = (
            deactivation.evaluate_best_deactivation_candidate(
                instance, selected, candidates, eq_initial=current_z, **common
            )
        )
        cumulative_eq_iters += sum(candidate_eq_iters)
        selected.add(v)
        cumulative_runtime += time.perf_counter() - step_start
        if obj < best_obj:
            best_obj = obj
            best_set = set(selected)
        timed_out = timeout_sec > 0.0 and cumulative_runtime > timeout_sec
        if step in budgets:
            rows.append(
                _prefix_row(
                    instance,
                    dataset,
                    opinion_setting,
                    alpha_setting,
                    step,
                    method,
                    baseline_objective,
                    best_obj,
                    cumulative_runtime,
                    best_set,
                    cumulative_eq_iters,
                    linear_solver_iterations=0,
                    timed_out=timed_out,
                    repetition=repetition,
                )
            )
        if note_timeout(
            timeout_cutoffs,
            dataset,
            timeout_method,
            step,
            cumulative_runtime,
            timeout_sec,
        ):
            break
    return rows


def _ais_greedy_prefix_rows(
    instance: Instance,
    budgets: list[int],
    k: int,
    method: str,
    *,
    algorithm: Any = deactivation.ais_greedy,
    baseline_objective: float,
    common: dict[str, Any],
    sens: dict[str, Any],
    timeout_sec: float,
    timeout_cutoffs: dict[str, dict[str, int]],
    dataset: str,
    opinion_setting: str,
    alpha_setting: str,
) -> list[dict[str, Any]]:
    if not any(method_allowed(timeout_cutoffs, dataset, method, b) for b in budgets):
        return []
    _print_progress(
        dataset, opinion_setting, alpha_setting, method, budget=max(budgets)
    )
    result = algorithm(
        instance,
        min(max(budgets), instance.n),
        k=k,
        **common,
        linear_tol=float(sens.get("linear_tol", 1e-8)),
        linear_max_iter=sens.get("linear_max_iter"),
        solver=str(sens.get("solver", "gmres")),
        timeout_sec=timeout_sec,
    )
    logs = {int(log["budget"]): log for log in result.logs}
    last_logged_budget = max(logs)
    rows: list[dict[str, Any]] = []
    for b in budgets:
        if not method_allowed(timeout_cutoffs, dataset, method, b):
            continue
        if b > last_logged_budget:
            break
        log = logs[min(b, instance.n)]
        runtime = float(log["elapsed_sec"])
        timed_out = note_timeout(
            timeout_cutoffs, dataset, method, b, runtime, timeout_sec
        ) or bool(log.get("timed_out", False))
        incumbent = set(int(v) for v in log["incumbent_set"])
        rows.append(
            _prefix_row(
                instance,
                dataset,
                opinion_setting,
                alpha_setting,
                b,
                method,
                baseline_objective,
                float(log["incumbent_objective"]),
                runtime,
                incumbent,
                int(log["equilibrium_iterations"]),
                linear_solver_iterations=int(log["linear_solver_iterations"]),
                timed_out=timed_out,
                extra_log_summary=log,
                extra_diagnostics={
                    "k": int(k),
                    "prefix_set": log["prefix_set"],
                    "incumbent_set": log["incumbent_set"],
                    "candidate_nodes": log["candidate_nodes"],
                    "selected_node": log["selected_node"],
                    "selected_node_approximate_improvement_score": log[
                        "selected_node_approximate_improvement_score"
                    ],
                    "selected_node_finite_gap": log[
                        "selected_node_finite_gap"
                    ],
                    "selected_node_approximate_improvement_rank": log[
                        "selected_node_approximate_improvement_rank"
                    ],
                    "prefix_objective": log["objective"],
                    "candidate_equilibrium_iterations": log[
                        "candidate_equilibrium_iterations"
                    ],
                },
            )
        )
    return rows


def _print_progress(
    dataset: str,
    opinion_setting: str,
    alpha_setting: str,
    method: str,
    *,
    budget: int | None = None,
    repetition: int | None = None,
) -> None:
    """Print the deactivation method that is about to run."""

    fields = [
        f"dataset={dataset}",
        f"opinion={opinion_setting}",
        f"alpha={alpha_setting}",
        f"method={method}",
    ]
    if budget is not None:
        fields.append(f"budget={budget}")
    if repetition is not None:
        fields.append(f"repetition={repetition}")
    print("[deactivation] " + " ".join(fields), flush=True)


def _prefix_row(
    instance: Instance,
    dataset: str,
    opinion_setting: str,
    alpha_setting: str,
    b: int,
    method: str,
    baseline_objective: float,
    objective: float,
    runtime: float,
    selected: set[int],
    equilibrium_iterations: int,
    *,
    linear_solver_iterations: int,
    timed_out: bool = False,
    repetition: int = -1,
    extra_log_summary: dict[str, Any] | None = None,
    extra_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    log_summary = {
        "equilibrium_iterations": equilibrium_iterations,
        "linear_solver_iterations": linear_solver_iterations,
        "feasibility_violation": 0.0,
    }
    if extra_log_summary:
        log_summary.update(extra_log_summary)
    diagnostics = {"method": method, "prefix_reused": True}
    if extra_diagnostics:
        diagnostics.update(extra_diagnostics)
    return _base_row(
        instance,
        dataset,
        opinion_setting,
        alpha_setting,
        b,
        method,
        baseline_objective,
        objective,
        runtime,
        len(selected),
        repetition=repetition,
        diagnostics=diagnostics,
        selected_indices=solution_indices(selected),
        timed_out=timed_out,
        log_summary=log_summary,
    )


def _evaluate_no_ai_mediation(
    instance: Instance,
    common: dict[str, Any],
) -> tuple[AlgorithmResult, float]:
    start = time.perf_counter()
    objective, z_star, diag = evaluate_alpha(
        instance, np.ones(instance.n, dtype=float), **common
    )
    runtime = time.perf_counter() - start
    log = {
        "method": "No-AI-Mediation",
        "objective": objective,
        "elapsed_sec": runtime,
        "equilibrium_iterations": int(diag.get("equilibrium_iterations", 0)),
        "linear_solver_iterations": 0,
        "feasibility_violation": 0.0,
        "set_size": instance.n,
    }
    return (
        AlgorithmResult(
            set(range(instance.n)),
            objective,
            z_star,
            {"method": "No-AI-Mediation", **diag},
            [log],
        ),
        runtime,
    )


def _base_row(
    instance: Instance,
    dataset: str,
    opinion_setting: str,
    alpha_setting: str,
    b: int,
    method: str,
    baseline_objective: float,
    objective: float,
    runtime: float,
    selected_set_size: int,
    *,
    repetition: int,
    diagnostics: Any,
    selected_indices: list[int] | None = None,
    log_summary: dict[str, Any] | None = None,
    timed_out: bool = False,
) -> dict[str, Any]:
    selected_indices = [] if selected_indices is None else selected_indices
    log_summary = {} if log_summary is None else log_summary
    return {
        "dataset": dataset,
        "n": instance.n,
        "m": instance.W.nnz,
        "opinion_setting": opinion_setting,
        "alpha_setting": alpha_setting,
        "budget": b,
        "method": method,
        "repetition": repetition,
        "objective": float(objective),
        "normalized_objective": float(objective / baseline_objective),
        "runtime_sec": float(runtime),
        "timed_out": bool(timed_out),
        "selected_set_size": int(selected_set_size),
        "equilibrium_iterations": int(log_summary.get("equilibrium_iterations", 0)),
        "linear_solver_iterations": int(
            log_summary.get("linear_solver_iterations", 0)
        ),
        "approximate_improvement_score_norm_inf": float(
            log_summary.get("approximate_improvement_score_norm_inf", 0.0)
        ),
        "neighbor_pd_score_norm_inf": float(
            log_summary.get("neighbor_pd_score_norm_inf", 0.0)
        ),
        "adjoint_converged": log_summary.get("adjoint_converged"),
        "adjoint_residual_2": float(
            log_summary.get("adjoint_residual_2", 0.0)
        ),
        "adjoint_relative_residual_2": float(
            log_summary.get("adjoint_relative_residual_2", 0.0)
        ),
        "adjoint_large_residual": bool(
            log_summary.get("adjoint_large_residual", False)
        ),
        "saturated_count": int(log_summary.get("saturated_count", 0)),
        "candidate_set_size": int(log_summary.get("candidate_set_size", 0)),
        "selected_approximate_improvement_score": float(
            log_summary.get(
                "selected_node_approximate_improvement_score", 0.0
            )
        ),
        "selected_finite_gap": float(
            log_summary.get("selected_node_finite_gap", 0.0)
        ),
        "selected_approximate_improvement_rank": log_summary.get(
            "selected_node_approximate_improvement_rank"
        ),
        "selected_neighbor_pd_score": log_summary.get(
            "selected_node_neighbor_pd_score"
        ),
        "selected_neighbor_pd_rank": log_summary.get(
            "selected_node_neighbor_pd_rank"
        ),
        "positive_approximate_improvement_count": int(
            log_summary.get("positive_approximate_improvement_count", 0)
        ),
        "positive_neighbor_pd_count": int(
            log_summary.get("positive_neighbor_pd_count", 0)
        ),
        "screened_saturated_fraction": float(
            log_summary.get("screened_saturated_fraction", 0.0)
        ),
        "score_gain_correlation": log_summary.get("score_gain_correlation"),
        "score_exact_objective_spearman_correlation": log_summary.get(
            "score_exact_objective_spearman_correlation"
        ),
        "feasibility_violation": float(log_summary.get("feasibility_violation", 0.0)),
        "selected_indices": dumps_json(selected_indices),
        "selected_labels": dumps_json(solution_labels(instance, selected_indices)),
        "diagnostics": dumps_json(diagnostics),
    }


def _summarize_logs(logs: list[dict[str, Any]]) -> dict[str, Any]:
    if not logs:
        return {}
    return {
        "equilibrium_iterations": sum(
            int(row.get("equilibrium_iterations", 0)) for row in logs
        ),
        "linear_solver_iterations": sum(
            int(row.get("linear_solver_iterations", 0)) for row in logs
        ),
        "feasibility_violation": max(
            float(row.get("feasibility_violation", 0.0)) for row in logs
        ),
    }


def _usampling_greedy_k_values(config: dict[str, Any]) -> list[int]:
    if "usampling_greedy_k_values" in config:
        return [int(value) for value in config["usampling_greedy_k_values"]]
    if "usampling_greedy_k" in config:
        return [int(config["usampling_greedy_k"])]
    return [10]


def _usampling_random_repetitions(config: dict[str, Any]) -> int:
    """Return the number of independent runs for every USampling k value."""

    value = config.get("random_repetitions", 1)
    repetitions = int(value)
    if repetitions < 1:
        raise ValueError("random_repetitions must be at least one.")
    return repetitions


def _ais_greedy_k_values(config: dict[str, Any]) -> list[int]:
    if "ais_greedy_k_values" in config:
        return [int(value) for value in config["ais_greedy_k_values"]]
    if "ais_greedy_k" in config:
        return [int(config["ais_greedy_k"])]
    return [10]


def _neighbor_pd_greedy_k_values(config: dict[str, Any]) -> list[int]:
    if "neighbor_pd_greedy_k_values" in config:
        return [int(value) for value in config["neighbor_pd_greedy_k_values"]]
    if "neighbor_pd_greedy_k" in config:
        return [int(config["neighbor_pd_greedy_k"])]
    return [10]

if __name__ == "__main__":
    raise SystemExit(main())
