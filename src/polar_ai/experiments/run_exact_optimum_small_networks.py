"""Compare Greedy with exhaustive exact optima on small real networks."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from polar_ai.algorithms import deactivation
from polar_ai.algorithms.common import evaluate_deactivation
from polar_ai.backends.selector import get_backend_name
from polar_ai.experiments.plot_exact_optimum_small_networks import (
    plot_exact_optimum_small_networks,
    write_exact_optimum_summary,
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


RAW_COLUMNS = [
    "experiment",
    "dataset",
    "n",
    "m",
    "seed",
    "backend",
    "opinion_setting",
    "alpha_setting",
    "max_budget",
    "subset_size",
    "subset_indices",
    "subset_labels",
    "objective",
    "polarization",
    "disagreement",
    "equilibrium_iterations",
    "equilibrium_converged",
    "equilibrium_residual_inf",
    "runtime_sec",
]

CHECKPOINT_VERSION = 1

COMPARISON_COLUMNS = [
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
    "optimal_objective",
    "greedy_objective",
    "optimal_normalized_objective",
    "greedy_normalized_objective",
    "greedy_to_optimal_ratio",
    "greedy_relative_optimality_gap",
    "greedy_fraction_of_optimal_improvement",
    "optimal_set_size",
    "greedy_set_size",
    "optimal_indices",
    "greedy_indices",
    "optimal_labels",
    "greedy_labels",
    "sets_equal",
    "set_jaccard",
    "exact_num_evaluations",
    "greedy_num_evaluations",
    "exact_equilibrium_iterations",
    "greedy_equilibrium_iterations",
    "exact_runtime_sec",
    "greedy_runtime_sec",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config, base_dir = load_config(args.config)
    output_root = resolve_path(
        args.results_dir
        or config.get(
            "results_dir",
            "../../results/additional/exact_optimum_small_networks",
        ),
        base_dir,
    )
    selected_datasets = set(args.dataset or [])
    backend = get_backend_name()
    if bool(config.get("require_cpp_backend", False)) and backend != "cpp":
        raise RuntimeError(
            "The exhaustive exact-optimum experiment requires the C++ "
            "backend, but the active backend is Python. Install the project "
            "before running it."
        )
    print(f"[exact-optimum] backend={backend}", flush=True)

    matched = False
    for dataset in dataset_configs(config, base_dir):
        if selected_datasets and dataset.name not in selected_datasets:
            continue
        matched = True
        graph = load_dataset(dataset)
        budgets = budget_values(config, int(graph.P.shape[0]))
        if not budgets:
            raise ValueError("At least one positive budget is required.")
        max_budget = max(budgets)
        # Keep checkpoints from different maximum budgets independent.  A
        # smaller run must never scan or mix with a potentially enormous CSV
        # produced for a larger maximum budget.
        dataset_dir = output_root / dataset.name / f"b_max_{max_budget}"
        # ``exact_subset_evaluations.csv`` is the legacy, one-row-per-subset
        # checkpoint.  It is kept only for a one-time streaming migration;
        # new runs store compact per-size aggregate state in JSON files.
        legacy_raw_path = dataset_dir / "exact_subset_evaluations.csv"
        checkpoint_dir = dataset_dir / "checkpoints"
        comparison_path = dataset_dir / "exact_optimum_comparison.csv"
        summary_path = dataset_dir / "exact_optimum_summary.csv"
        figures_dir = dataset_dir / "figures"

        if args.plot_only:
            write_exact_optimum_summary(comparison_path, summary_path)
            for path in plot_exact_optimum_small_networks(
                comparison_path, figures_dir
            ):
                print(path, flush=True)
            continue

        resume = bool(config.get("resume", True))
        if not resume:
            clear_compact_checkpoints(checkpoint_dir)
        elif not any(checkpoint_dir.glob("*.json")):
            migrate_legacy_raw_checkpoint(
                legacy_raw_path,
                checkpoint_dir,
                dataset=dataset,
                n=int(graph.P.shape[0]),
                max_budget=max_budget,
                seeds=experiment_seeds(config),
                settings=setting_pairs(config),
                progress_every=int(
                    config.get("migration_progress_every", 1_000_000)
                ),
            )

        eps, lambda_cap = spectral_params(config)
        for seed in experiment_seeds(config):
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
                run_compact_exact_search(
                    instance,
                    dataset=dataset,
                    opinion_setting=opinion_setting,
                    alpha_setting=alpha_setting,
                    seed=seed,
                    backend=backend,
                    max_budget=max_budget,
                    checkpoint_path=compact_checkpoint_path(
                        checkpoint_dir,
                        seed=seed,
                        opinion_setting=opinion_setting,
                        alpha_setting=alpha_setting,
                    ),
                    eq_tol=float(config.get("eq_tol", 1.0e-10)),
                    eq_max_iter=int(config.get("eq_max_iter", 10_000)),
                    checkpoint_every=int(config.get("checkpoint_every", 250)),
                    progress_every=int(config.get("progress_every", 1000)),
                    fail_on_nonconvergence=bool(
                        config.get("fail_on_nonconvergence", True)
                    ),
                )

        write_comparison_from_checkpoints(
            config,
            dataset,
            graph,
            checkpoint_dir,
            comparison_path,
            backend=backend,
        )
        write_exact_optimum_summary(comparison_path, summary_path)
        if bool(config.get("generate_plots", True)):
            for path in plot_exact_optimum_small_networks(
                comparison_path, figures_dir
            ):
                print(path, flush=True)

    if selected_datasets and not matched:
        raise ValueError(
            "None of the requested datasets occur in the config: "
            + ", ".join(sorted(selected_datasets))
        )
    return 0


def compact_checkpoint_path(
    checkpoint_dir: Path,
    *,
    seed: int,
    opinion_setting: str,
    alpha_setting: str,
) -> Path:
    """Return the compact checkpoint path for one generated instance."""

    return checkpoint_dir / (
        f"seed_{int(seed)}__opinion_{opinion_setting}__alpha_{alpha_setting}.json"
    )


def clear_compact_checkpoints(checkpoint_dir: Path) -> None:
    """Remove only compact checkpoints for an explicitly fresh run."""

    if not checkpoint_dir.exists():
        return
    for path in checkpoint_dir.glob("*.json"):
        path.unlink()


def initial_compact_state(
    *,
    dataset: str,
    n: int,
    seed: int,
    opinion_setting: str,
    alpha_setting: str,
    max_budget: int,
    backend: str,
) -> dict[str, Any]:
    """Create a constant-memory exact-search checkpoint."""

    return {
        "version": CHECKPOINT_VERSION,
        "dataset": str(dataset),
        "n": int(n),
        "seed": int(seed),
        "opinion_setting": str(opinion_setting),
        "alpha_setting": str(alpha_setting),
        "max_budget": int(max_budget),
        "backend": str(backend),
        "next_subset_size": 0,
        "next_combination_rank": 0,
        "complete": False,
        "sizes": {},
    }


def run_compact_exact_search(
    instance: Instance,
    *,
    dataset: DatasetConfig,
    opinion_setting: str,
    alpha_setting: str,
    seed: int,
    backend: str,
    max_budget: int,
    checkpoint_path: Path,
    eq_tol: float,
    eq_max_iter: int,
    checkpoint_every: int = 250,
    progress_every: int = 1000,
    fail_on_nonconvergence: bool = True,
) -> int:
    """Enumerate all feasible subsets with compact per-size checkpoint state."""

    max_budget = min(max(int(max_budget), 0), instance.n)
    expected = initial_compact_state(
        dataset=dataset.name,
        n=instance.n,
        seed=seed,
        opinion_setting=opinion_setting,
        alpha_setting=alpha_setting,
        max_budget=max_budget,
        backend=backend,
    )
    if checkpoint_path.exists():
        state = load_compact_checkpoint(checkpoint_path)
        validate_compact_state(state, expected)
    else:
        state = expected

    total = sum(math.comb(instance.n, size) for size in range(max_budget + 1))
    completed = sum(
        int(stats.get("num_evaluations", 0))
        for stats in state["sizes"].values()
    )
    print(
        "[exact-optimum] "
        f"dataset={dataset.name} seed={seed} opinion={opinion_setting} "
        f"alpha={alpha_setting} subsets={total} completed={completed}",
        flush=True,
    )
    if bool(state.get("complete", False)):
        return 0

    params = {"eq_tol": float(eq_tol), "eq_max_iter": int(eq_max_iter)}
    baseline_obj, baseline_z, baseline_diag = evaluate_deactivation(
        instance, set(), **params
    )
    _require_convergence(
        baseline_diag,
        dataset=dataset.name,
        opinion_setting=opinion_setting,
        alpha_setting=alpha_setting,
        subset=(),
        fail=fail_on_nonconvergence,
    )

    newly_evaluated = 0
    size = int(state["next_subset_size"])
    while size <= max_budget:
        total_at_size = math.comb(instance.n, size)
        rank = (
            int(state["next_combination_rank"])
            if size == int(state["next_subset_size"])
            else 0
        )
        stats = state["sizes"].setdefault(
            str(size), empty_size_statistics(size)
        )
        if int(stats["num_evaluations"]) != rank:
            raise ValueError(
                f"Invalid checkpoint {checkpoint_path}: size {size} has "
                f"{stats['num_evaluations']} evaluations but next rank {rank}."
            )

        for combination_rank, subset in iter_combinations_from_rank(
            instance.n, size, rank
        ):
            if subset:
                start = time.perf_counter()
                objective, _, diagnostics = evaluate_deactivation(
                    instance,
                    subset,
                    eq_initial=baseline_z,
                    **params,
                )
                runtime = time.perf_counter() - start
            else:
                objective = baseline_obj
                diagnostics = baseline_diag
                runtime = float(baseline_diag.get("elapsed_sec", 0.0))
            _require_convergence(
                diagnostics,
                dataset=dataset.name,
                opinion_setting=opinion_setting,
                alpha_setting=alpha_setting,
                subset=subset,
                fail=fail_on_nonconvergence,
            )
            update_size_statistics(
                stats,
                subset=subset,
                objective=float(objective),
                diagnostics=diagnostics,
                runtime_sec=runtime,
            )
            newly_evaluated += 1
            completed += 1
            state["next_subset_size"] = size
            state["next_combination_rank"] = combination_rank + 1

            if checkpoint_every > 0 and newly_evaluated % checkpoint_every == 0:
                save_compact_checkpoint(checkpoint_path, state)
            if progress_every > 0 and completed % progress_every == 0:
                print(
                    "[exact-optimum] "
                    f"dataset={dataset.name} opinion={opinion_setting} "
                    f"alpha={alpha_setting} size={size} "
                    f"size_progress={combination_rank + 1}/{total_at_size} "
                    f"progress={completed}/{total}",
                    flush=True,
                )

        stats["complete"] = True
        size += 1
        state["next_subset_size"] = size
        state["next_combination_rank"] = 0
        save_compact_checkpoint(checkpoint_path, state)

    state["complete"] = True
    save_compact_checkpoint(checkpoint_path, state)
    print(
        "[exact-optimum] "
        f"dataset={dataset.name} opinion={opinion_setting} "
        f"alpha={alpha_setting} progress={total}/{total} "
        f"newly_evaluated={newly_evaluated}",
        flush=True,
    )
    return newly_evaluated


def empty_size_statistics(subset_size: int) -> dict[str, Any]:
    return {
        "subset_size": int(subset_size),
        "num_evaluations": 0,
        "equilibrium_iterations": 0,
        "runtime_sec": 0.0,
        "best_objective": None,
        "best_indices": [],
        "complete": False,
    }


def update_size_statistics(
    stats: dict[str, Any],
    *,
    subset: tuple[int, ...],
    objective: float,
    diagnostics: dict[str, Any],
    runtime_sec: float,
) -> None:
    """Update exact-size aggregates after one subset evaluation."""

    stats["num_evaluations"] = int(stats["num_evaluations"]) + 1
    stats["equilibrium_iterations"] = int(
        stats["equilibrium_iterations"]
    ) + int(diagnostics.get("equilibrium_iterations", 0))
    stats["runtime_sec"] = float(stats["runtime_sec"]) + float(runtime_sec)
    previous = stats.get("best_objective")
    previous_indices = tuple(int(v) for v in stats.get("best_indices", []))
    if previous is None or (float(objective), subset) < (
        float(previous),
        previous_indices,
    ):
        stats["best_objective"] = float(objective)
        stats["best_indices"] = [int(v) for v in subset]


def save_compact_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically save a compact checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def load_compact_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid compact checkpoint: {path}")
    return state


def validate_compact_state(
    state: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Reject checkpoints belonging to a different experiment instance."""

    fields = [
        "version",
        "dataset",
        "n",
        "seed",
        "opinion_setting",
        "alpha_setting",
        "max_budget",
    ]
    mismatches = [
        field
        for field in fields
        if state.get(field) != expected.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Compact checkpoint does not match the requested experiment: "
            + ", ".join(mismatches)
        )


def iter_combinations_from_rank(
    n: int, size: int, start_rank: int
) -> Iterator[tuple[int, tuple[int, ...]]]:
    """Yield lexicographic combinations without replaying earlier ranks."""

    n = int(n)
    size = int(size)
    total = math.comb(n, size)
    rank = int(start_rank)
    if rank < 0 or rank > total:
        raise ValueError(f"Combination rank {rank} is outside [0, {total}].")
    if rank == total:
        return
    combination = list(combination_at_rank(n, size, rank))
    while True:
        yield rank, tuple(combination)
        rank += 1
        if rank >= total:
            return
        position = size - 1
        while position >= 0 and combination[position] == n - size + position:
            position -= 1
        if position < 0:
            return
        combination[position] += 1
        for index in range(position + 1, size):
            combination[index] = combination[index - 1] + 1


def combination_at_rank(n: int, size: int, rank: int) -> tuple[int, ...]:
    """Unrank a zero-based lexicographic combination."""

    n = int(n)
    size = int(size)
    rank = int(rank)
    total = math.comb(n, size)
    if rank < 0 or rank >= total:
        raise ValueError(f"Combination rank {rank} is outside [0, {total}).")
    if size == 0:
        return ()
    result: list[int] = []
    lower = 0
    for position in range(size):
        remaining = size - position - 1
        maximum = n - remaining
        for candidate in range(lower, maximum):
            block = math.comb(n - candidate - 1, remaining)
            if rank < block:
                result.append(candidate)
                lower = candidate + 1
                break
            rank -= block
        else:  # pragma: no cover - guarded by the rank validation above
            raise RuntimeError("Failed to unrank combination.")
    return tuple(result)


def migrate_legacy_raw_checkpoint(
    raw_path: Path,
    checkpoint_dir: Path,
    *,
    dataset: DatasetConfig,
    n: int,
    max_budget: int,
    seeds: Iterable[int],
    settings: Iterable[tuple[str, str]],
    progress_every: int = 1_000_000,
) -> None:
    """Stream a legacy row-per-subset CSV into compact aggregate states."""

    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return
    expected_keys = {
        (int(seed), str(opinion), str(alpha))
        for seed in seeds
        for opinion, alpha in settings
    }
    states = {
        key: initial_compact_state(
            dataset=dataset.name,
            n=n,
            seed=key[0],
            opinion_setting=key[1],
            alpha_setting=key[2],
            max_budget=max_budget,
            backend="legacy",
        )
        for key in expected_keys
    }
    rows_read = 0
    print(
        f"[exact-optimum] migrating legacy checkpoint {raw_path} "
        f"({raw_path.stat().st_size} bytes)",
        flush=True,
    )
    with raw_path.open("r", newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                key = (
                    int(row["seed"]),
                    str(row["opinion_setting"]),
                    str(row["alpha_setting"]),
                )
                if key not in expected_keys:
                    continue
                row_budget = int(row["max_budget"])
                if row_budget != int(max_budget):
                    raise ValueError(
                        f"row max_budget={row_budget}, expected {max_budget}"
                    )
                subset = tuple(
                    int(value) for value in json.loads(row["subset_indices"])
                )
                size = int(row["subset_size"])
                if len(subset) != size or size < 0 or size > max_budget:
                    raise ValueError("invalid subset size or indices")
                state = states[key]
                stats = state["sizes"].setdefault(
                    str(size), empty_size_statistics(size)
                )
                update_size_statistics(
                    stats,
                    subset=subset,
                    objective=float(row["objective"]),
                    diagnostics={
                        "equilibrium_iterations": int(
                            row["equilibrium_iterations"]
                        )
                    },
                    runtime_sec=float(row["runtime_sec"]),
                )
                if str(row.get("backend", "")):
                    state["backend"] = str(row["backend"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                print(
                    "[exact-optimum] stopped legacy migration at malformed "
                    f"line {line_number}: {exc}",
                    flush=True,
                )
                break
            rows_read += 1
            if progress_every > 0 and rows_read % progress_every == 0:
                print(
                    f"[exact-optimum] legacy migration rows={rows_read}",
                    flush=True,
                )

    for key, state in states.items():
        if not state["sizes"]:
            continue
        first_incomplete: int | None = None
        for size in range(max_budget + 1):
            stats = state["sizes"].get(str(size))
            observed = 0 if stats is None else int(stats["num_evaluations"])
            expected = math.comb(int(n), size)
            if observed > expected:
                raise ValueError(
                    f"Legacy checkpoint has too many size-{size} rows for {key}."
                )
            if first_incomplete is None and observed < expected:
                first_incomplete = size
            elif first_incomplete is not None and observed > 0:
                raise ValueError(
                    "Legacy checkpoint is not a contiguous enumeration for "
                    f"{key}: found size {size} after incomplete size "
                    f"{first_incomplete}."
                )
            if stats is not None:
                stats["complete"] = observed == expected

        if first_incomplete is None:
            state["next_subset_size"] = max_budget + 1
            state["next_combination_rank"] = 0
            state["complete"] = True
        else:
            state["next_subset_size"] = first_incomplete
            current = state["sizes"].setdefault(
                str(first_incomplete), empty_size_statistics(first_incomplete)
            )
            state["next_combination_rank"] = int(current["num_evaluations"])
            state["complete"] = False
        path = compact_checkpoint_path(
            checkpoint_dir,
            seed=key[0],
            opinion_setting=key[1],
            alpha_setting=key[2],
        )
        save_compact_checkpoint(path, state)
        print(f"[exact-optimum] migrated {path}", flush=True)


def enumerate_exact_subsets(
    instance: Instance,
    *,
    dataset: DatasetConfig,
    opinion_setting: str,
    alpha_setting: str,
    seed: int,
    backend: str,
    max_budget: int,
    completed_keys: set[tuple[int, str, str, tuple[int, ...]]],
    writer: csv.DictWriter,
    handle: Any,
    eq_tol: float,
    eq_max_iter: int,
    checkpoint_every: int = 250,
    progress_every: int = 1000,
    fail_on_nonconvergence: bool = True,
) -> int:
    """Evaluate every subset up to ``max_budget`` and append missing rows."""

    max_budget = min(max(int(max_budget), 0), instance.n)
    total = sum(math.comb(instance.n, size) for size in range(max_budget + 1))
    prefix = (int(seed), str(opinion_setting), str(alpha_setting))
    already_done = sum(
        1
        for key in completed_keys
        if key[:3] == prefix and len(key[3]) <= max_budget
    )
    print(
        "[exact-optimum] "
        f"dataset={dataset.name} seed={seed} opinion={opinion_setting} "
        f"alpha={alpha_setting} subsets={total} completed={already_done}",
        flush=True,
    )

    params = {"eq_tol": float(eq_tol), "eq_max_iter": int(eq_max_iter)}
    baseline_obj, baseline_z, baseline_diag = evaluate_deactivation(
        instance, set(), **params
    )
    _require_convergence(
        baseline_diag,
        dataset=dataset.name,
        opinion_setting=opinion_setting,
        alpha_setting=alpha_setting,
        subset=(),
        fail=fail_on_nonconvergence,
    )

    written = 0
    visited = 0
    for subset in iter_subsets(instance.n, max_budget):
        visited += 1
        key = (*prefix, subset)
        if key in completed_keys:
            continue
        if subset:
            start = time.perf_counter()
            objective, _, diagnostics = evaluate_deactivation(
                instance,
                subset,
                eq_initial=baseline_z,
                **params,
            )
            runtime = time.perf_counter() - start
        else:
            objective = baseline_obj
            diagnostics = baseline_diag
            runtime = float(baseline_diag.get("elapsed_sec", 0.0))

        _require_convergence(
            diagnostics,
            dataset=dataset.name,
            opinion_setting=opinion_setting,
            alpha_setting=alpha_setting,
            subset=subset,
            fail=fail_on_nonconvergence,
        )
        row = exact_evaluation_row(
            instance,
            dataset=dataset.name,
            opinion_setting=opinion_setting,
            alpha_setting=alpha_setting,
            seed=seed,
            backend=backend,
            max_budget=max_budget,
            subset=subset,
            objective=float(objective),
            diagnostics=diagnostics,
            runtime_sec=runtime,
        )
        writer.writerow(row)
        completed_keys.add(key)
        written += 1

        if checkpoint_every > 0 and written % checkpoint_every == 0:
            handle.flush()
        if progress_every > 0 and visited % progress_every == 0:
            print(
                "[exact-optimum] "
                f"dataset={dataset.name} opinion={opinion_setting} "
                f"alpha={alpha_setting} progress={visited}/{total}",
                flush=True,
            )

    handle.flush()
    print(
        "[exact-optimum] "
        f"dataset={dataset.name} opinion={opinion_setting} "
        f"alpha={alpha_setting} progress={total}/{total} "
        f"newly_written={written}",
        flush=True,
    )
    return written


def iter_subsets(n: int, max_budget: int) -> Iterator[tuple[int, ...]]:
    """Yield all subsets in increasing cardinality and lexicographic order."""

    for size in range(min(max(int(max_budget), 0), int(n)) + 1):
        yield from itertools.combinations(range(int(n)), size)


def exact_evaluation_row(
    instance: Instance,
    *,
    dataset: str,
    opinion_setting: str,
    alpha_setting: str,
    seed: int,
    backend: str,
    max_budget: int,
    subset: tuple[int, ...],
    objective: float,
    diagnostics: dict[str, Any],
    runtime_sec: float,
) -> dict[str, Any]:
    """Build one raw exhaustive-evaluation row."""

    return {
        "experiment": "exact_optimum_small_networks",
        "dataset": dataset,
        "n": instance.n,
        "m": _edge_count(instance),
        "seed": int(seed),
        "backend": backend,
        "opinion_setting": opinion_setting,
        "alpha_setting": alpha_setting,
        "max_budget": int(max_budget),
        "subset_size": len(subset),
        "subset_indices": json.dumps(list(subset)),
        "subset_labels": json.dumps(solution_labels(instance, subset)),
        "objective": float(objective),
        "polarization": float(diagnostics.get("polarization", np.nan)),
        "disagreement": float(diagnostics.get("disagreement", np.nan)),
        "equilibrium_iterations": int(
            diagnostics.get("equilibrium_iterations", 0)
        ),
        "equilibrium_converged": bool(
            diagnostics.get("equilibrium_converged", False)
        ),
        "equilibrium_residual_inf": float(
            diagnostics.get("equilibrium_residual_inf", np.nan)
        ),
        "runtime_sec": float(runtime_sec),
    }


def write_comparison_from_checkpoints(
    config: dict[str, Any],
    dataset: DatasetConfig,
    graph: Any,
    checkpoint_dir: Path,
    output_path: Path,
    *,
    backend: str,
) -> Path:
    """Combine compact exact-search states and Greedy trajectories."""

    budgets = budget_values(config, int(graph.P.shape[0]))
    eps, lambda_cap = spectral_params(config)
    rows: list[dict[str, Any]] = []

    for seed in experiment_seeds(config):
        for opinion_setting, alpha_setting in setting_pairs(config):
            checkpoint_path = compact_checkpoint_path(
                checkpoint_dir,
                seed=seed,
                opinion_setting=opinion_setting,
                alpha_setting=alpha_setting,
            )
            state = load_compact_checkpoint(checkpoint_path)
            if not bool(state.get("complete", False)):
                raise ValueError(f"Exact search is incomplete: {checkpoint_path}")
            instance = build_instance(
                graph,
                dataset_name=dataset.name,
                opinion_setting=opinion_setting,
                alpha_setting=alpha_setting,
                seed=seed,
                eps=eps,
                lambda_cap=lambda_cap,
            )
            greedy = greedy_budget_results(
                instance,
                budgets,
                **common_params(config),
            )
            baseline_stats = state["sizes"].get("0")
            if baseline_stats is None or int(
                baseline_stats.get("num_evaluations", 0)
            ) != 1:
                raise ValueError(f"Missing exact baseline: {checkpoint_path}")
            baseline = float(baseline_stats["best_objective"])

            for budget in budgets:
                exact = compact_budget_result(state, int(budget), instance.n)
                greedy_row = greedy[int(budget)]
                optimal_indices = tuple(exact["best_indices"])
                greedy_indices = tuple(greedy_row["selected_indices"])
                optimal_obj = float(exact["best_objective"])
                greedy_obj = float(greedy_row["objective"])
                rows.append(
                    comparison_row(
                        instance,
                        dataset=dataset.name,
                        opinion_setting=opinion_setting,
                        alpha_setting=alpha_setting,
                        seed=seed,
                        backend=backend,
                        budget=int(budget),
                        baseline_objective=baseline,
                        optimal_objective=optimal_obj,
                        greedy_objective=greedy_obj,
                        optimal_indices=optimal_indices,
                        greedy_indices=greedy_indices,
                        exact_num_evaluations=int(exact["num_evaluations"]),
                        greedy_num_evaluations=int(
                            greedy_row["num_evaluations"]
                        ),
                        exact_equilibrium_iterations=int(
                            exact["equilibrium_iterations"]
                        ),
                        greedy_equilibrium_iterations=int(
                            greedy_row["equilibrium_iterations"]
                        ),
                        exact_runtime_sec=float(exact["runtime_sec"]),
                        greedy_runtime_sec=float(greedy_row["runtime_sec"]),
                    )
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COMPARISON_COLUMNS).to_csv(
        output_path, index=False
    )
    print(output_path, flush=True)
    return output_path


def compact_budget_result(
    state: dict[str, Any], budget: int, n: int
) -> dict[str, Any]:
    """Return the exact optimum and cumulative work through one budget."""

    budget = min(max(int(budget), 0), int(n))
    candidates: list[tuple[float, int, tuple[int, ...]]] = []
    evaluations = 0
    iterations = 0
    runtime = 0.0
    for size in range(budget + 1):
        stats = state["sizes"].get(str(size))
        if stats is None or not bool(stats.get("complete", False)):
            raise ValueError(f"Exact size {size} is incomplete in checkpoint.")
        expected = math.comb(int(n), size)
        observed = int(stats.get("num_evaluations", 0))
        if observed != expected:
            raise ValueError(
                f"Exact size {size} has {observed} evaluations; "
                f"expected {expected}."
            )
        indices = tuple(int(v) for v in stats["best_indices"])
        candidates.append((float(stats["best_objective"]), size, indices))
        evaluations += observed
        iterations += int(stats["equilibrium_iterations"])
        runtime += float(stats["runtime_sec"])
    objective, _, indices = min(candidates)
    return {
        "best_objective": objective,
        "best_indices": indices,
        "num_evaluations": evaluations,
        "equilibrium_iterations": iterations,
        "runtime_sec": runtime,
    }


def greedy_budget_results(
    instance: Instance,
    budgets: Iterable[int],
    *,
    eq_tol: float,
    eq_max_iter: int,
) -> dict[int, dict[str, Any]]:
    """Run one Greedy path and return its incumbent at each requested budget."""

    requested = sorted(
        {min(instance.n, max(0, int(budget))) for budget in budgets}
    )
    if not requested:
        return {}
    selected: set[int] = set()
    best_set: set[int] = set()
    start = time.perf_counter()
    best_obj, current_z, baseline_diag = evaluate_deactivation(
        instance,
        set(),
        eq_tol=eq_tol,
        eq_max_iter=eq_max_iter,
    )
    cumulative_iterations = int(
        baseline_diag.get("equilibrium_iterations", 0)
    )
    num_evaluations = 1
    results: dict[int, dict[str, Any]] = {}
    if 0 in requested:
        results[0] = _greedy_result_row(
            best_obj,
            best_set,
            num_evaluations,
            cumulative_iterations,
            time.perf_counter() - start,
        )

    for step in range(1, max(requested) + 1):
        candidates = [v for v in range(instance.n) if v not in selected]
        objective, node, current_z, _, candidate_iterations = (
            deactivation.evaluate_best_deactivation_candidate(
                instance,
                selected,
                candidates,
                eq_initial=current_z,
                eq_tol=eq_tol,
                eq_max_iter=eq_max_iter,
            )
        )
        selected.add(int(node))
        num_evaluations += len(candidates)
        cumulative_iterations += sum(int(v) for v in candidate_iterations)
        if float(objective) < float(best_obj):
            best_obj = float(objective)
            best_set = set(selected)
        if step in requested:
            results[step] = _greedy_result_row(
                best_obj,
                best_set,
                num_evaluations,
                cumulative_iterations,
                time.perf_counter() - start,
            )
    return results


def _greedy_result_row(
    objective: float,
    selected: set[int],
    num_evaluations: int,
    equilibrium_iterations: int,
    runtime_sec: float,
) -> dict[str, Any]:
    return {
        "objective": float(objective),
        "selected_indices": tuple(sorted(int(v) for v in selected)),
        "num_evaluations": int(num_evaluations),
        "equilibrium_iterations": int(equilibrium_iterations),
        "runtime_sec": float(runtime_sec),
    }


def select_optimal_row(frame: pd.DataFrame) -> pd.Series:
    """Select the minimum objective with deterministic smaller-set ties."""

    records = list(frame.to_dict(orient="records"))
    if not records:
        raise ValueError("Cannot select an optimum from an empty frame.")
    best = min(
        records,
        key=lambda row: (
            float(row["objective"]),
            int(row["subset_size"]),
            tuple(int(v) for v in json.loads(str(row["subset_indices"]))),
        ),
    )
    return pd.Series(best)


def comparison_row(
    instance: Instance,
    *,
    dataset: str,
    opinion_setting: str,
    alpha_setting: str,
    seed: int,
    backend: str,
    budget: int,
    baseline_objective: float,
    optimal_objective: float,
    greedy_objective: float,
    optimal_indices: tuple[int, ...],
    greedy_indices: tuple[int, ...],
    exact_num_evaluations: int,
    greedy_num_evaluations: int,
    exact_equilibrium_iterations: int,
    greedy_equilibrium_iterations: int,
    exact_runtime_sec: float,
    greedy_runtime_sec: float,
) -> dict[str, Any]:
    """Build one budget-level exact-versus-Greedy comparison row."""

    scale = max(abs(float(baseline_objective)), np.finfo(float).eps)
    objective_scale = max(
        1.0,
        abs(float(optimal_objective)),
        abs(float(greedy_objective)),
    )
    objective_tolerance = 1e-9 * objective_scale
    raw_gap = float(greedy_objective - optimal_objective)
    if raw_gap < -objective_tolerance:
        raise ValueError(
            "Greedy objective is smaller than the certified optimum beyond "
            f"numerical tolerance: greedy={greedy_objective}, "
            f"optimal={optimal_objective}."
        )
    objective_gap = 0.0 if abs(raw_gap) <= objective_tolerance else raw_gap
    improvement_denominator = float(baseline_objective - optimal_objective)
    if improvement_denominator > np.finfo(float).eps * scale:
        fraction = float(
            np.clip(
                1.0 - objective_gap / improvement_denominator,
                0.0,
                1.0,
            )
        )
    else:
        fraction = 1.0 if objective_gap == 0.0 else np.nan
    optimal_set = set(optimal_indices)
    greedy_set = set(greedy_indices)
    union = optimal_set | greedy_set
    return {
        "experiment": "exact_optimum_small_networks",
        "dataset": dataset,
        "n": instance.n,
        "m": _edge_count(instance),
        "seed": int(seed),
        "backend": backend,
        "opinion_setting": opinion_setting,
        "alpha_setting": alpha_setting,
        "budget": int(budget),
        "baseline_objective": float(baseline_objective),
        "optimal_objective": float(optimal_objective),
        "greedy_objective": float(greedy_objective),
        "optimal_normalized_objective": float(
            optimal_objective / baseline_objective
        ),
        "greedy_normalized_objective": float(
            greedy_objective / baseline_objective
        ),
        "greedy_to_optimal_ratio": (
            (
                1.0
                if objective_gap == 0.0
                else float(greedy_objective / optimal_objective)
            )
            if optimal_objective != 0.0
            else np.nan
        ),
        "greedy_relative_optimality_gap": float(objective_gap / scale),
        "greedy_fraction_of_optimal_improvement": fraction,
        "optimal_set_size": len(optimal_indices),
        "greedy_set_size": len(greedy_indices),
        "optimal_indices": json.dumps(list(optimal_indices)),
        "greedy_indices": json.dumps(list(greedy_indices)),
        "optimal_labels": json.dumps(
            solution_labels(instance, optimal_indices)
        ),
        "greedy_labels": json.dumps(solution_labels(instance, greedy_indices)),
        "sets_equal": optimal_set == greedy_set,
        "set_jaccard": (
            float(len(optimal_set & greedy_set) / len(union)) if union else 1.0
        ),
        "exact_num_evaluations": int(exact_num_evaluations),
        "greedy_num_evaluations": int(greedy_num_evaluations),
        "exact_equilibrium_iterations": int(exact_equilibrium_iterations),
        "greedy_equilibrium_iterations": int(greedy_equilibrium_iterations),
        "exact_runtime_sec": float(exact_runtime_sec),
        "greedy_runtime_sec": float(greedy_runtime_sec),
    }


def read_completed_keys(
    path: Path,
) -> set[tuple[int, str, str, tuple[int, ...]]]:
    """Read checkpoint keys from a possibly partial prior raw CSV."""

    if not path.exists() or path.stat().st_size == 0:
        return set()
    keys: set[tuple[int, str, str, tuple[int, ...]]] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("subset_indices"):
                continue
            subset = tuple(int(v) for v in json.loads(row["subset_indices"]))
            keys.add(
                (
                    int(row["seed"]),
                    str(row["opinion_setting"]),
                    str(row["alpha_setting"]),
                    subset,
                )
            )
    return keys


def experiment_seeds(config: dict[str, Any]) -> list[int]:
    """Return configured instance seeds, preserving their input order."""

    values = config.get("seeds", [config.get("seed", 0)])
    return list(dict.fromkeys(int(value) for value in values))


def _edge_count(instance: Instance) -> int:
    """Return the conventional edge count for directed or undirected input."""

    if bool(instance.metadata.get("directed_input", True)):
        return int(instance.W.nnz)
    diagonal = int(np.count_nonzero(instance.W.diagonal()))
    return int((instance.W.nnz - diagonal) // 2 + diagonal)


def _require_convergence(
    diagnostics: dict[str, Any],
    *,
    dataset: str,
    opinion_setting: str,
    alpha_setting: str,
    subset: tuple[int, ...],
    fail: bool,
) -> None:
    if bool(diagnostics.get("equilibrium_converged", False)) or not fail:
        return
    raise RuntimeError(
        "Equilibrium failed to converge for "
        f"dataset={dataset}, opinion={opinion_setting}, alpha={alpha_setting}, "
        f"subset={subset}."
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Greedy with exact optima on small real networks."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name to run; repeat to select multiple datasets.",
    )
    parser.add_argument(
        "--results-dir", default=None, help="Optional results root override."
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from an existing comparison CSV.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
