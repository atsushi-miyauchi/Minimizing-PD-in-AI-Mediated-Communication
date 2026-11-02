"""Compare Greedy with a simple backtracking extension on small networks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from polar_ai.algorithms.common import evaluate_deactivation
from polar_ai.backends.selector import get_backend_name
from polar_ai.experiments.plot_backtracking_greedy import (
    plot_backtracking_greedy,
    write_backtracking_summary,
)
from polar_ai.experiments.run_exact_optimum_small_networks import (
    greedy_budget_results,
)
from polar_ai.experiments.utils import (
    DatasetConfig,
    budget_values,
    build_instance,
    common_params,
    dataset_configs,
    load_config,
    load_dataset,
    resolve_path,
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
    "seed",
    "backend",
    "opinion_setting",
    "alpha_setting",
    "budget",
    "baseline_objective",
    "greedy_objective",
    "backtracking_objective",
    "greedy_normalized_objective",
    "backtracking_normalized_objective",
    "objective_gain_over_greedy",
    "normalized_gain_over_greedy",
    "backtracking_beats_greedy",
    "objectives_equal",
    "greedy_set_size",
    "backtracking_set_size",
    "greedy_indices",
    "backtracking_indices",
    "greedy_labels",
    "backtracking_labels",
    "sets_equal",
    "set_jaccard",
    "greedy_num_evaluations",
    "backtracking_num_evaluations",
    "greedy_equilibrium_iterations",
    "backtracking_equilibrium_iterations",
    "greedy_runtime_sec",
    "backtracking_runtime_sec",
    "accepted_additions",
    "accepted_deletions",
    "accepted_replacements",
    "forced_additions",
    "switched_to_addition_only",
    "switch_move",
    "path_length",
    "terminal_set_size",
]


@dataclass(frozen=True)
class Move:
    """One exactly evaluated neighboring intervention set."""

    action: str
    objective: float
    selected: frozenset[int]
    equilibrium: np.ndarray
    diagnostics: dict[str, Any]
    added: int | None = None
    removed: int | None = None


@dataclass(frozen=True)
class BacktrackingResult:
    """Best incumbent and diagnostics from one budget-specific run."""

    selected: frozenset[int]
    objective: float
    equilibrium: np.ndarray
    runtime_sec: float
    num_evaluations: int
    equilibrium_iterations: int
    accepted_additions: int
    accepted_deletions: int
    accepted_replacements: int
    forced_additions: int
    switched_to_addition_only: bool
    switch_move: int | None
    path_length: int
    terminal_set_size: int


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config, base_dir = load_config(args.config)
    output_root = resolve_path(
        args.results_dir
        or config.get(
            "results_dir", "../../results/additional/backtracking_greedy"
        ),
        base_dir,
    )
    selected_datasets = set(args.dataset or [])
    backend = get_backend_name()
    if bool(config.get("require_cpp_backend", False)) and backend != "cpp":
        raise RuntimeError(
            "The backtracking experiment requires the C++ backend, but the "
            "active backend is Python. Install the project before running it."
        )
    print(f"[backtracking-greedy] backend={backend}", flush=True)

    matched = False
    for dataset in dataset_configs(config, base_dir):
        if selected_datasets and dataset.name not in selected_datasets:
            continue
        matched = True
        graph = load_dataset(dataset)
        budgets = budget_values(config, int(graph.P.shape[0]))
        if not budgets:
            raise ValueError("At least one positive budget is required.")
        dataset_dir = output_root / dataset.name
        raw_path = dataset_dir / "backtracking_greedy_results.csv"
        summary_path = dataset_dir / "backtracking_greedy_summary.csv"
        figures_dir = dataset_dir / "figures"

        if args.plot_only:
            write_backtracking_summary(raw_path, summary_path)
            for path in plot_backtracking_greedy(raw_path, figures_dir):
                print(path, flush=True)
            continue

        if args.rerun_existing and raw_path.exists():
            raw_path.unlink()
        completed = (
            read_completed_keys(raw_path)
            if config.get("resume", True)
            else set()
        )
        if not config.get("resume", True) and raw_path.exists():
            raw_path.unlink()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not raw_path.exists() or raw_path.stat().st_size == 0

        with raw_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
            if write_header:
                writer.writeheader()
                handle.flush()
            _run_dataset(
                config,
                dataset,
                graph,
                budgets,
                backend=backend,
                completed=completed,
                writer=writer,
                handle=handle,
            )

        write_backtracking_summary(raw_path, summary_path)
        outputs = [raw_path, summary_path]
        if bool(config.get("generate_plots", True)):
            outputs.extend(plot_backtracking_greedy(raw_path, figures_dir))
        for path in outputs:
            print(path, flush=True)

    if selected_datasets and not matched:
        raise ValueError(
            "None of the requested datasets occur in the config: "
            + ", ".join(sorted(selected_datasets))
        )
    return 0


def backtracking_greedy(
    instance: Instance,
    budget: int,
    *,
    eq_tol: float = 1.0e-10,
    eq_max_iter: int = 10_000,
    improvement_tol: float = 1.0e-9,
) -> BacktrackingResult:
    """Run improving local moves until the first non-improving addition.

    Before the switch, the method accepts the best strictly improving
    addition, deletion, or one-for-one replacement.  If no such move exists
    below the budget, it takes the best addition even when non-improving and
    permanently switches to ordinary addition-only Greedy.  The best set seen
    along the path is returned.
    """

    budget = min(max(int(budget), 0), instance.n)
    params = {"eq_tol": float(eq_tol), "eq_max_iter": int(eq_max_iter)}
    start = time.perf_counter()
    current_obj, current_z, baseline_diag = evaluate_deactivation(
        instance, set(), **params
    )
    _require_convergence(baseline_diag, selected=frozenset())

    selected: frozenset[int] = frozenset()
    best_set = selected
    best_obj = float(current_obj)
    best_z = np.asarray(current_z, dtype=float).copy()
    locked = False
    switch_move: int | None = None
    num_evaluations = 1
    equilibrium_iterations = int(
        baseline_diag.get("equilibrium_iterations", 0)
    )
    counts = {"addition": 0, "deletion": 0, "replacement": 0}
    forced_additions = 0
    path_length = 0

    while True:
        additions, additions_iterations = _evaluate_additions(
            instance, selected, current_z, budget=budget, **params
        )
        num_evaluations += len(additions)
        equilibrium_iterations += additions_iterations

        if locked:
            if not additions:
                break
            move = min(additions, key=_move_key)
            forced_additions += 1
        else:
            deletions, deletion_iterations = _evaluate_deletions(
                instance, selected, current_z, **params
            )
            replacements, replacement_iterations = _evaluate_replacements(
                instance, selected, current_z, **params
            )
            num_evaluations += len(deletions) + len(replacements)
            equilibrium_iterations += (
                deletion_iterations + replacement_iterations
            )
            neighbors = additions + deletions + replacements
            if not neighbors:
                break
            candidate = min(neighbors, key=_move_key)
            if _strictly_improves(
                candidate.objective, current_obj, improvement_tol
            ):
                move = candidate
            elif additions:
                move = min(additions, key=_move_key)
                locked = True
                switch_move = path_length + 1
                forced_additions += 1
            else:
                break

        selected = move.selected
        current_obj = float(move.objective)
        current_z = np.asarray(move.equilibrium, dtype=float)
        counts[move.action] += 1
        path_length += 1
        if current_obj < best_obj:
            best_obj = current_obj
            best_set = selected
            best_z = current_z.copy()

    return BacktrackingResult(
        selected=best_set,
        objective=best_obj,
        equilibrium=best_z,
        runtime_sec=time.perf_counter() - start,
        num_evaluations=num_evaluations,
        equilibrium_iterations=equilibrium_iterations,
        accepted_additions=counts["addition"],
        accepted_deletions=counts["deletion"],
        accepted_replacements=counts["replacement"],
        forced_additions=forced_additions,
        switched_to_addition_only=locked,
        switch_move=switch_move,
        path_length=path_length,
        terminal_set_size=len(selected),
    )


def _evaluate_additions(
    instance: Instance,
    selected: frozenset[int],
    current_z: np.ndarray,
    *,
    budget: int,
    eq_tol: float,
    eq_max_iter: int,
) -> tuple[list[Move], int]:
    if len(selected) >= budget:
        return [], 0
    moves = []
    for added in range(instance.n):
        if added in selected:
            continue
        target = selected | {added}
        moves.append(
            _evaluate_move(
                instance,
                target,
                current_z,
                action="addition",
                added=added,
                eq_tol=eq_tol,
                eq_max_iter=eq_max_iter,
            )
        )
    return moves, _iteration_sum(moves)


def _evaluate_deletions(
    instance: Instance,
    selected: frozenset[int],
    current_z: np.ndarray,
    *,
    eq_tol: float,
    eq_max_iter: int,
) -> tuple[list[Move], int]:
    moves = [
        _evaluate_move(
            instance,
            selected - {removed},
            current_z,
            action="deletion",
            removed=removed,
            eq_tol=eq_tol,
            eq_max_iter=eq_max_iter,
        )
        for removed in sorted(selected)
    ]
    return moves, _iteration_sum(moves)


def _evaluate_replacements(
    instance: Instance,
    selected: frozenset[int],
    current_z: np.ndarray,
    *,
    eq_tol: float,
    eq_max_iter: int,
) -> tuple[list[Move], int]:
    remaining = [v for v in range(instance.n) if v not in selected]
    moves = [
        _evaluate_move(
            instance,
            (selected - {removed}) | {added},
            current_z,
            action="replacement",
            added=added,
            removed=removed,
            eq_tol=eq_tol,
            eq_max_iter=eq_max_iter,
        )
        for removed in sorted(selected)
        for added in remaining
    ]
    return moves, _iteration_sum(moves)


def _evaluate_move(
    instance: Instance,
    selected: frozenset[int],
    current_z: np.ndarray,
    *,
    action: str,
    eq_tol: float,
    eq_max_iter: int,
    added: int | None = None,
    removed: int | None = None,
) -> Move:
    objective, equilibrium, diagnostics = evaluate_deactivation(
        instance,
        set(selected),
        eq_initial=current_z,
        eq_tol=eq_tol,
        eq_max_iter=eq_max_iter,
    )
    _require_convergence(diagnostics, selected=selected)
    return Move(
        action=action,
        objective=float(objective),
        selected=selected,
        equilibrium=np.asarray(equilibrium, dtype=float),
        diagnostics=diagnostics,
        added=added,
        removed=removed,
    )


def _run_dataset(
    config: dict[str, Any],
    dataset: DatasetConfig,
    graph: Any,
    budgets: list[int],
    *,
    backend: str,
    completed: set[tuple[int, str, str, int]],
    writer: csv.DictWriter,
    handle: Any,
) -> None:
    seed = int(config.get("seed", 0))
    eps, lambda_cap = spectral_params(config)
    params = common_params(config)
    improvement_tol = float(config.get("improvement_tol", 1.0e-9))
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
        greedy = greedy_budget_results(instance, [0, *budgets], **params)
        baseline = float(greedy[0]["objective"])
        for budget in budgets:
            key = (seed, opinion_setting, alpha_setting, int(budget))
            if key in completed:
                print(
                    "[backtracking-greedy] "
                    f"dataset={dataset.name} opinion={opinion_setting} "
                    f"alpha={alpha_setting} budget={budget} status=skip",
                    flush=True,
                )
                continue
            print(
                "[backtracking-greedy] "
                f"dataset={dataset.name} opinion={opinion_setting} "
                f"alpha={alpha_setting} budget={budget} status=run",
                flush=True,
            )
            result = backtracking_greedy(
                instance,
                budget,
                improvement_tol=improvement_tol,
                **params,
            )
            writer.writerow(
                result_row(
                    instance,
                    dataset=dataset,
                    seed=seed,
                    backend=backend,
                    opinion_setting=opinion_setting,
                    alpha_setting=alpha_setting,
                    budget=budget,
                    baseline=baseline,
                    greedy=greedy[budget],
                    backtracking=result,
                    comparison_tol=improvement_tol,
                )
            )
            handle.flush()
            completed.add(key)


def result_row(
    instance: Instance,
    *,
    dataset: DatasetConfig,
    seed: int,
    backend: str,
    opinion_setting: str,
    alpha_setting: str,
    budget: int,
    baseline: float,
    greedy: dict[str, Any],
    backtracking: BacktrackingResult,
    comparison_tol: float,
) -> dict[str, Any]:
    greedy_obj = float(greedy["objective"])
    backtracking_obj = float(backtracking.objective)
    greedy_set = frozenset(int(v) for v in greedy["selected_indices"])
    backtracking_set = backtracking.selected
    union = greedy_set | backtracking_set
    gain = greedy_obj - backtracking_obj
    scale = max(1.0, abs(greedy_obj), abs(backtracking_obj))
    baseline_scale = max(abs(float(baseline)), np.finfo(float).eps)
    return {
        "experiment": "backtracking_greedy",
        "dataset": dataset.name,
        "n": instance.n,
        "m": _edge_count(instance),
        "seed": int(seed),
        "backend": backend,
        "opinion_setting": opinion_setting,
        "alpha_setting": alpha_setting,
        "budget": int(budget),
        "baseline_objective": baseline,
        "greedy_objective": greedy_obj,
        "backtracking_objective": backtracking_obj,
        "greedy_normalized_objective": greedy_obj / baseline_scale,
        "backtracking_normalized_objective": (
            backtracking_obj / baseline_scale
        ),
        "objective_gain_over_greedy": gain,
        "normalized_gain_over_greedy": gain / baseline_scale,
        "backtracking_beats_greedy": bool(
            gain > float(comparison_tol) * scale
        ),
        "objectives_equal": bool(
            math.isclose(
                greedy_obj,
                backtracking_obj,
                rel_tol=float(comparison_tol),
                abs_tol=float(comparison_tol),
            )
        ),
        "greedy_set_size": len(greedy_set),
        "backtracking_set_size": len(backtracking_set),
        "greedy_indices": json.dumps(sorted(greedy_set)),
        "backtracking_indices": json.dumps(sorted(backtracking_set)),
        "greedy_labels": json.dumps(
            solution_labels(instance, sorted(greedy_set))
        ),
        "backtracking_labels": json.dumps(
            solution_labels(instance, sorted(backtracking_set))
        ),
        "sets_equal": greedy_set == backtracking_set,
        "set_jaccard": (
            len(greedy_set & backtracking_set) / len(union) if union else 1.0
        ),
        "greedy_num_evaluations": int(greedy["num_evaluations"]),
        "backtracking_num_evaluations": backtracking.num_evaluations,
        "greedy_equilibrium_iterations": int(greedy["equilibrium_iterations"]),
        "backtracking_equilibrium_iterations": (
            backtracking.equilibrium_iterations
        ),
        "greedy_runtime_sec": float(greedy["runtime_sec"]),
        "backtracking_runtime_sec": backtracking.runtime_sec,
        "accepted_additions": backtracking.accepted_additions,
        "accepted_deletions": backtracking.accepted_deletions,
        "accepted_replacements": backtracking.accepted_replacements,
        "forced_additions": backtracking.forced_additions,
        "switched_to_addition_only": backtracking.switched_to_addition_only,
        "switch_move": backtracking.switch_move,
        "path_length": backtracking.path_length,
        "terminal_set_size": backtracking.terminal_set_size,
    }


def read_completed_keys(path: Path) -> set[tuple[int, str, str, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(
        path,
        usecols=["seed", "opinion_setting", "alpha_setting", "budget"],
    )
    return {
        (
            int(row.seed),
            str(row.opinion_setting),
            str(row.alpha_setting),
            int(row.budget),
        )
        for row in frame.itertuples(index=False)
    }


def _strictly_improves(new: float, current: float, tolerance: float) -> bool:
    scale = max(1.0, abs(float(current)))
    return float(new) < float(current) - float(tolerance) * scale


def _move_key(move: Move) -> tuple[float, int, int, int]:
    priority = {"addition": 0, "deletion": 1, "replacement": 2}
    return (
        float(move.objective),
        priority[move.action],
        -1 if move.removed is None else int(move.removed),
        -1 if move.added is None else int(move.added),
    )


def _iteration_sum(moves: Iterable[Move]) -> int:
    return sum(
        int(move.diagnostics.get("equilibrium_iterations", 0)) for move in moves
    )


def _require_convergence(
    diagnostics: dict[str, Any], *, selected: frozenset[int]
) -> None:
    if bool(diagnostics.get("equilibrium_converged", True)):
        return
    raise RuntimeError(
        "Equilibrium computation did not converge for intervention set "
        f"{sorted(selected)}."
    )


def _edge_count(instance: Instance) -> int:
    if bool(instance.metadata.get("directed_input", True)):
        return int(instance.W.nnz)
    diagonal = int(np.count_nonzero(instance.W.diagonal()))
    return int((instance.W.nnz - diagonal) // 2 + diagonal)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Greedy with improving backtracking moves."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Run only this configured dataset; may be repeated.",
    )
    parser.add_argument(
        "--results-dir", default=None, help="Optional output directory."
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="Only rebuild summaries/plots."
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Discard this experiment's existing raw CSV before running.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
