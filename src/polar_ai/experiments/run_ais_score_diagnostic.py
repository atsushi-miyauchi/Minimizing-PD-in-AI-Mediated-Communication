"""Diagnose AIS scores against all remaining deactivation candidates.

This module is intentionally separate from the paper's original experiment
runner.  Full-candidate evaluations are used only for diagnostics; each AIS
trajectory still selects the exact-best node from its score-based top-k
shortlist.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from polar_ai.algorithms import deactivation
from polar_ai.algorithms.common import (
    evaluate_deactivation,
    score_exact_objective_spearman_correlation,
)
from polar_ai.backends.selector import get_backend_name
from polar_ai.experiments.plot_ais_score_diagnostic import (
    plot_ais_score_diagnostics,
    write_ais_score_summary,
)
from polar_ai.experiments.utils import (
    build_instance,
    common_params,
    dataset_configs,
    dumps_json,
    load_config,
    load_dataset,
    resolve_path,
    sensitivity_params,
    setting_pairs,
    solution_labels,
    spectral_params,
)
from polar_ai.instance import Instance


RESULT_COLUMNS = [
    "experiment",
    "dataset",
    "n",
    "m",
    "opinion_setting",
    "alpha_setting",
    "seed",
    "backend",
    "k",
    "iteration",
    "budget",
    "selected_set_size_before",
    "selected_set_size_after",
    "num_remaining_candidates",
    "current_objective",
    "selected_objective",
    "selected_gain",
    "selected_node",
    "selected_node_label",
    "selected_node_ais_rank",
    "exact_best_objective",
    "exact_best_gain",
    "exact_best_node",
    "exact_best_node_label",
    "exact_best_ais_rank",
    "hit_at_k",
    "full_candidate_spearman",
    "score_runtime_sec",
    "candidate_scan_runtime_sec",
    "iteration_runtime_sec",
    "trajectory_runtime_sec",
    "candidate_equilibrium_iterations",
    "nonconverged_candidate_count",
    "adjoint_converged",
    "adjoint_residual_2",
    "adjoint_relative_residual_2",
    "adjoint_large_residual",
    "selected_indices",
    "selected_labels",
]

RUN_KEY_COLUMNS = ["dataset", "opinion_setting", "alpha_setting", "k"]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config, base_dir = load_config(args.config)
    output_root = resolve_path(
        args.results_dir
        or config.get("results_dir", "../results/additional/ais_score_diagnostic"),
        base_dir,
    )
    selected_datasets = set(args.dataset or [])

    backend = get_backend_name()
    if bool(config.get("require_cpp_backend", False)) and backend != "cpp":
        raise RuntimeError(
            "The diagnostic config requires the C++ backend, but the active "
            "backend is Python. Install the project in editable mode first."
        )
    print(f"[ais-score-diagnostic] backend={backend}", flush=True)

    for dataset in dataset_configs(config, base_dir):
        if selected_datasets and dataset.name not in selected_datasets:
            continue
        dataset_dir = output_root / dataset.name
        result_path = dataset_dir / "ais_score_diagnostic.csv"
        summary_path = dataset_dir / "ais_score_diagnostic_summary.csv"
        figures_dir = dataset_dir / "figures"

        if args.plot_only:
            _write_outputs(result_path, summary_path, figures_dir)
            continue

        if not bool(config.get("resume", True)) and result_path.exists():
            result_path.unlink()

        graph = load_dataset(dataset)
        seed = int(config.get("seed", 0))
        eps, lambda_cap = spectral_params(config)
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
            for k in _k_values(config):
                completed = _completed_rows(
                    result_path,
                    dataset=dataset.name,
                    opinion_setting=opinion_setting,
                    alpha_setting=alpha_setting,
                    k=k,
                    seed=seed,
                    backend=backend,
                )
                start_selected, runtime_offset = _resume_state(completed)
                budget = min(_budget_max(config), instance.n)
                if len(start_selected) >= budget:
                    print(
                        "[ais-score-diagnostic] "
                        f"dataset={dataset.name} opinion={opinion_setting} "
                        f"alpha={alpha_setting} k={k} status=complete",
                        flush=True,
                    )
                    continue

                print(
                    "[ais-score-diagnostic] "
                    f"dataset={dataset.name} opinion={opinion_setting} "
                    f"alpha={alpha_setting} k={k} "
                    f"resume_iteration={len(start_selected) + 1} "
                    f"budget={budget}",
                    flush=True,
                )
                for row in diagnose_ais_trajectory(
                    instance,
                    dataset=dataset.name,
                    opinion_setting=opinion_setting,
                    alpha_setting=alpha_setting,
                    seed=seed,
                    backend=backend,
                    k=k,
                    budget=budget,
                    start_selected=start_selected,
                    runtime_offset=runtime_offset,
                    fail_on_nonconvergence=bool(
                        config.get("fail_on_nonconvergence", True)
                    ),
                    fail_on_adjoint_nonconvergence=bool(
                        config.get("fail_on_adjoint_nonconvergence", True)
                    ),
                    **common_params(config),
                    **_score_params(config),
                ):
                    _append_result_row(result_path, row)
                    print(
                        "[ais-score-diagnostic] "
                        f"dataset={dataset.name} opinion={opinion_setting} "
                        f"alpha={alpha_setting} k={k} "
                        f"iteration={row['iteration']}/{budget} "
                        f"rho={_format_optional_float(row['full_candidate_spearman'])} "
                        f"rank={row['exact_best_ais_rank']} "
                        f"hit={row['hit_at_k']}",
                        flush=True,
                    )

        if bool(config.get("generate_plots", True)):
            _write_outputs(result_path, summary_path, figures_dir)

    return 0


def diagnose_ais_trajectory(
    instance: Instance,
    *,
    dataset: str,
    opinion_setting: str,
    alpha_setting: str,
    seed: int,
    backend: str,
    k: int,
    budget: int,
    eq_tol: float,
    eq_max_iter: int,
    linear_tol: float,
    linear_max_iter: int | None,
    solver: str,
    start_selected: set[int] | None = None,
    runtime_offset: float = 0.0,
    fail_on_nonconvergence: bool = True,
    fail_on_adjoint_nonconvergence: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield one full-candidate diagnostic row per AIS iteration."""

    if int(k) < 1:
        raise ValueError("k must be at least one.")
    selected = set() if start_selected is None else set(start_selected)
    if any(v < 0 or v >= instance.n for v in selected):
        raise ValueError("start_selected contains a node outside the instance.")
    if len(selected) > int(budget):
        raise ValueError("start_selected is larger than the requested budget.")

    current_objective, current_z, initial_diag = evaluate_deactivation(
        instance,
        selected,
        eq_tol=eq_tol,
        eq_max_iter=eq_max_iter,
    )
    if fail_on_nonconvergence and not bool(
        initial_diag.get("equilibrium_converged", False)
    ):
        raise RuntimeError("The initial/resumed equilibrium did not converge.")

    trajectory_start = time.perf_counter()
    maximum_step = min(max(int(budget), 0), instance.n)
    for step in range(len(selected) + 1, maximum_step + 1):
        iteration_start = time.perf_counter()
        remaining = np.array(
            [v for v in range(instance.n) if v not in selected], dtype=int
        )
        if remaining.size == 0:
            break

        score_start = time.perf_counter()
        score_result = deactivation.approximate_improvement_scores(
            instance,
            current_z,
            selected,
            linear_tol=linear_tol,
            linear_max_iter=linear_max_iter,
            solver=solver,
        )
        score_runtime = time.perf_counter() - score_start
        linear_diag = score_result.diagnostics["linear_solver"]
        if fail_on_adjoint_nonconvergence and not bool(
            linear_diag.get("converged", False)
        ):
            raise RuntimeError(
                f"The AIS adjoint solve did not converge at iteration {step}."
            )

        ranked_remaining = deactivation.screening_candidate_set(
            score_result.scores,
            remaining,
            len(remaining),
        )
        shortlist = set(ranked_remaining[: min(int(k), len(ranked_remaining))])

        scan_start = time.perf_counter()
        evaluations = deactivation.evaluate_deactivation_candidates(
            instance,
            selected,
            [int(v) for v in remaining],
            eq_initial=current_z,
            eq_tol=eq_tol,
            eq_max_iter=eq_max_iter,
        )
        scan_runtime = time.perf_counter() - scan_start
        nonconverged = [
            int(node)
            for _, node, _, diagnostics in evaluations
            if not bool(diagnostics.get("equilibrium_converged", False))
        ]
        if fail_on_nonconvergence and nonconverged:
            raise RuntimeError(
                f"{len(nonconverged)} candidate equilibria did not converge "
                f"at iteration {step}; first node={nonconverged[0]}."
            )

        exact_best = min(evaluations, key=lambda item: (item[0], item[1]))
        shortlist_evaluations = [
            evaluation for evaluation in evaluations if evaluation[1] in shortlist
        ]
        selected_evaluation = min(
            shortlist_evaluations, key=lambda item: (item[0], item[1])
        )
        exact_best_objective, exact_best_node, _, _ = exact_best
        selected_objective, selected_node, selected_z, _ = selected_evaluation

        exact_best_rank = ranked_remaining.index(int(exact_best_node)) + 1
        selected_rank = ranked_remaining.index(int(selected_node)) + 1
        exact_objectives = np.asarray(
            [objective for objective, _, _, _ in evaluations], dtype=float
        )
        remaining_scores = np.asarray(
            [score_result.scores[node] for _, node, _, _ in evaluations],
            dtype=float,
        )
        spearman = score_exact_objective_spearman_correlation(
            remaining_scores,
            exact_objectives,
        )
        candidate_equilibrium_iterations = sum(
            int(diagnostics.get("equilibrium_iterations", 0))
            for _, _, _, diagnostics in evaluations
        )

        selected_before = len(selected)
        selected.add(int(selected_node))
        current_z = selected_z
        previous_objective = float(current_objective)
        current_objective = float(selected_objective)
        iteration_runtime = time.perf_counter() - iteration_start
        trajectory_runtime = (
            float(runtime_offset) + time.perf_counter() - trajectory_start
        )
        yield {
            "experiment": "ais_score_diagnostic",
            "dataset": dataset,
            "n": instance.n,
            "m": instance.W.nnz,
            "opinion_setting": opinion_setting,
            "alpha_setting": alpha_setting,
            "seed": int(seed),
            "backend": backend,
            "k": int(k),
            "iteration": int(step),
            "budget": int(step),
            "selected_set_size_before": selected_before,
            "selected_set_size_after": len(selected),
            "num_remaining_candidates": len(remaining),
            "current_objective": previous_objective,
            "selected_objective": current_objective,
            "selected_gain": previous_objective - current_objective,
            "selected_node": int(selected_node),
            "selected_node_label": _node_label(instance, selected_node),
            "selected_node_ais_rank": int(selected_rank),
            "exact_best_objective": float(exact_best_objective),
            "exact_best_gain": previous_objective - float(exact_best_objective),
            "exact_best_node": int(exact_best_node),
            "exact_best_node_label": _node_label(instance, exact_best_node),
            "exact_best_ais_rank": int(exact_best_rank),
            "hit_at_k": bool(exact_best_rank <= int(k)),
            "full_candidate_spearman": spearman,
            "score_runtime_sec": score_runtime,
            "candidate_scan_runtime_sec": scan_runtime,
            "iteration_runtime_sec": iteration_runtime,
            "trajectory_runtime_sec": trajectory_runtime,
            "candidate_equilibrium_iterations": (
                candidate_equilibrium_iterations
            ),
            "nonconverged_candidate_count": len(nonconverged),
            "adjoint_converged": bool(linear_diag.get("converged", False)),
            "adjoint_residual_2": float(linear_diag.get("residual_2", 0.0)),
            "adjoint_relative_residual_2": float(
                linear_diag.get("relative_residual_2", 0.0)
            ),
            "adjoint_large_residual": bool(
                linear_diag.get("large_residual", False)
            ),
            "selected_indices": dumps_json(sorted(selected)),
            "selected_labels": dumps_json(
                solution_labels(instance, sorted(selected))
            ),
        }


def _write_outputs(
    result_path: Path, summary_path: Path, figures_dir: Path
) -> None:
    summary = write_ais_score_summary(result_path, summary_path)
    print(summary, flush=True)
    for path in plot_ais_score_diagnostics(result_path, figures_dir):
        print(path, flush=True)


def _append_result_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unexpected = set(row) - set(RESULT_COLUMNS)
    if unexpected:
        raise ValueError(f"Unexpected result columns: {sorted(unexpected)}")
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _completed_rows(
    path: Path,
    *,
    dataset: str,
    opinion_setting: str,
    alpha_setting: str,
    k: int,
    seed: int,
    backend: str,
) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = set(RUN_KEY_COLUMNS + ["iteration", "selected_indices"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Existing diagnostic CSV is missing columns: {sorted(missing)}"
        )
    subset = frame[
        (frame["dataset"].astype(str) == str(dataset))
        & (frame["opinion_setting"].astype(str) == str(opinion_setting))
        & (frame["alpha_setting"].astype(str) == str(alpha_setting))
        & (frame["k"].astype(int) == int(k))
    ].copy()
    if subset.empty:
        return subset
    if "seed" in subset and set(subset["seed"].astype(int)) != {int(seed)}:
        raise ValueError("Existing rows use a different random seed.")
    if "backend" in subset and set(subset["backend"].astype(str)) != {backend}:
        raise ValueError("Existing rows use a different numerical backend.")
    subset = subset.sort_values("iteration")
    iterations = list(subset["iteration"].astype(int))
    expected = list(range(1, len(iterations) + 1))
    if iterations != expected:
        raise ValueError(
            "Existing rows are not a contiguous trajectory from iteration 1."
        )
    return subset


def _resume_state(rows: pd.DataFrame) -> tuple[set[int], float]:
    if rows.empty:
        return set(), 0.0
    last = rows.iloc[-1]
    raw_selected = json.loads(str(last["selected_indices"]))
    selected = {int(node) for node in raw_selected}
    if len(selected) != int(last["iteration"]):
        raise ValueError(
            "Stored selected_indices do not match the completed iteration."
        )
    runtime = float(last.get("trajectory_runtime_sec", 0.0))
    return selected, runtime


def _score_params(config: dict[str, Any]) -> dict[str, Any]:
    sensitivity = sensitivity_params(config)
    return {
        "linear_tol": float(sensitivity.get("linear_tol", 1e-8)),
        "linear_max_iter": sensitivity.get("linear_max_iter"),
        "solver": str(sensitivity.get("solver", "gmres")),
    }


def _k_values(config: dict[str, Any]) -> list[int]:
    values = [int(value) for value in config.get("ais_k_values", [1, 10, 100])]
    if not values or any(value < 1 for value in values):
        raise ValueError("ais_k_values must contain positive integers.")
    if len(values) != len(set(values)):
        raise ValueError("ais_k_values must not contain duplicates.")
    return values


def _budget_max(config: dict[str, Any]) -> int:
    value = int(config.get("budget_max", 500))
    if value < 1:
        raise ValueError("budget_max must be at least one.")
    return value


def _node_label(instance: Instance, node: int) -> str:
    labels = instance.metadata.get("node_labels")
    return str(node) if not labels else str(labels[int(node)])


def _format_optional_float(value: Any) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full-remaining-candidate AIS score diagnostics and generate "
            "publication-style figures."
        )
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Optional output-root override (resolved relative to the config).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Run only this configured dataset; may be supplied repeatedly.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate summaries and figures from existing CSV files.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
