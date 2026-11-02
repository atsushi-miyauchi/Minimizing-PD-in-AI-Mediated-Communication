"""Utilities shared by experiment scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from polar_ai.alpha_generation import (
    generate_heterogeneous_alpha,
    generate_homogeneous_alpha,
)
from polar_ai.graph_io import GraphMatrices, load_edge_list
from polar_ai.instance import Instance
from polar_ai.opinion_generation import (
    generate_polarized_opinions,
    generate_uniform_opinions,
)
from polar_ai.spectral import homogeneous_lambda_vector


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset entry parsed from YAML."""

    name: str
    path: Path
    directed: bool = True
    weighted: bool | None = None
    delimiter: str | None = None
    comments: tuple[str, ...] = ("#", "%")
    has_header: bool = False


def parse_args(description: str, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    return parser.parse_args(argv)


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping.")
    return config, config_path.parent


def dataset_configs(config: dict[str, Any], base_dir: Path) -> list[DatasetConfig]:
    entries = config.get("datasets", [])
    if not entries:
        raise ValueError("Config must contain at least one dataset entry.")

    datasets: list[DatasetConfig] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each dataset entry must be a mapping.")
        path = resolve_path(entry["path"], base_dir)
        datasets.append(
            DatasetConfig(
                name=str(entry.get("name", path.stem)),
                path=path,
                directed=bool(entry.get("directed", True)),
                weighted=entry.get("weighted"),
                delimiter=entry.get("delimiter"),
                comments=tuple(entry.get("comments", ("#", "%"))),
                has_header=bool(entry.get("has_header", False)),
            )
        )
    return datasets


def load_dataset(dataset: DatasetConfig) -> GraphMatrices:
    return load_edge_list(
        dataset.path,
        directed=dataset.directed,
        weighted=dataset.weighted,
        delimiter=dataset.delimiter,
        comments=dataset.comments,
        has_header=dataset.has_header,
    )


def build_instance(
    graph: GraphMatrices,
    *,
    dataset_name: str,
    opinion_setting: str,
    alpha_setting: str,
    seed: int,
    eps: float,
    lambda_cap: float,
) -> Instance:
    n = graph.P.shape[0]
    if opinion_setting == "uniform":
        s = generate_uniform_opinions(
            n, seed=stable_seed(seed, dataset_name, opinion_setting, "s")
        )
    elif opinion_setting == "polarized":
        s = generate_polarized_opinions(
            n, seed=stable_seed(seed, dataset_name, opinion_setting, "s")
        )
    else:
        raise ValueError(f"Unknown opinion setting: {opinion_setting}")

    if alpha_setting == "homogeneous":
        alpha = generate_homogeneous_alpha(n, value=2.0)
    elif alpha_setting == "heterogeneous":
        alpha = generate_heterogeneous_alpha(
            n, seed=stable_seed(seed, dataset_name, alpha_setting, "alpha")
        )
    else:
        raise ValueError(f"Unknown alpha setting: {alpha_setting}")

    lambda_vec = homogeneous_lambda_vector(graph.P, alpha, eps=eps, cap=lambda_cap)
    metadata = dict(graph.metadata)
    metadata.update(
        {
            "dataset": dataset_name,
            "node_labels": list(graph.nodes),
            "opinion_setting": opinion_setting,
            "alpha_setting": alpha_setting,
            "seed": int(seed),
        }
    )
    return Instance(
        W=graph.W,
        P=graph.P,
        Wbar=graph.Wbar,
        s=s,
        lambda_vec=lambda_vec,
        alpha=alpha,
        metadata=metadata,
    )


def setting_pairs(config: dict[str, Any]) -> list[tuple[str, str]]:
    opinions = config.get("opinion_settings", ["uniform", "polarized"])
    alphas = config.get("alpha_settings", ["homogeneous", "heterogeneous"])
    return [(str(op), str(al)) for op in opinions for al in alphas]


def budget_values(config: dict[str, Any], n: int) -> list[int]:
    if "budgets" in config:
        return [int(b) for b in config["budgets"] if int(b) <= n]
    if "budget_max" in config:
        max_budget = min(n, int(config["budget_max"]))
    else:
        max_budget = int(math.floor(n * float(config.get("budget_fraction", 0.1))))
    return list(range(1, max_budget + 1))


def deactivation_marker_budgets(b_max: int) -> list[int]:
    """Return the budgets used for markers and repeated Random runs."""

    if b_max < 1:
        return []
    return sorted(
        {1, b_max}
        | {
            (i * b_max) // 10
            for i in range(1, 11)
            if (i * b_max) // 10 >= 1
        }
    )


def algorithm_timeout_sec(config: dict[str, Any]) -> float:
    """Return the per-algorithm timeout threshold in seconds."""

    return float(config.get("algorithm_timeout_sec", 1200.0))


def method_allowed(
    cutoffs: dict[str, dict[str, int]],
    dataset: str,
    method: str,
    budget: int | float,
) -> bool:
    """Return whether a method should still be run at this budget."""

    cutoff = cutoffs.get(dataset, {}).get(method)
    return cutoff is None or int(budget) <= cutoff


def note_timeout(
    cutoffs: dict[str, dict[str, int]],
    dataset: str,
    method: str,
    budget: int | float,
    runtime: float,
    timeout_sec: float,
) -> bool:
    """Record the first budget where a method exceeds the timeout."""

    if timeout_sec <= 0.0 or runtime <= timeout_sec:
        return False
    dataset_cutoffs = cutoffs.setdefault(dataset, {})
    budget_int = int(budget)
    previous = dataset_cutoffs.get(method)
    if previous is None or budget_int < previous:
        dataset_cutoffs[method] = budget_int
    return True


def default_output_path(
    config: dict[str, Any], base_dir: Path, filename: str, output: str | None
) -> Path:
    if output is not None:
        return resolve_path(output, base_dir)
    output_dir = resolve_path(config.get("output_dir", "results/raw"), base_dir)
    return output_dir / filename


def save_rows(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def stable_seed(base_seed: int, *parts: Any) -> int:
    text = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(v) for v in value]
    if isinstance(value, set):
        return [sanitize_json(v) for v in sorted(value)]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dumps_json(value: Any) -> str:
    return json.dumps(sanitize_json(value), sort_keys=True)


def solution_indices(solution: Any) -> list[int]:
    if isinstance(solution, np.ndarray):
        return [int(i) for i in np.flatnonzero(solution > 1e-12)]
    return [int(i) for i in sorted(solution)]


def solution_labels(instance: Instance, indices: Iterable[int]) -> list[str]:
    labels = instance.metadata.get("node_labels")
    if not labels:
        return [str(i) for i in indices]
    return [str(labels[int(i)]) for i in indices]


def timed_call(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def common_params(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "eq_tol": float(config.get("eq_tol", 1e-10)),
        "eq_max_iter": int(config.get("eq_max_iter", 10_000)),
    }


def sensitivity_params(config: dict[str, Any]) -> dict[str, Any]:
    params = common_params(config)
    params.update(
        {
            "linear_tol": float(config.get("linear_tol", 1e-8)),
            "linear_max_iter": config.get("linear_max_iter"),
        }
    )
    if "linear_solver" in config:
        params["solver"] = str(config["linear_solver"])
    if params["linear_max_iter"] is not None:
        params["linear_max_iter"] = int(params["linear_max_iter"])
    return params


def spectral_params(config: dict[str, Any]) -> tuple[float, float]:
    return float(config.get("eps", 0.01)), float(config.get("lambda_cap", 0.5))


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()
