"""Benchmark deactivation with network-correlated innate opinions.

This additional experiment keeps the complete empirical distribution of the
paper's uniform and polarized opinion settings, while ordering those opinions
by the Fiedler vector of the symmetrized interaction graph.  It repeats the
benchmark over multiple instance seeds and uses the same aggregation and
plotting pipeline as the multi-seed performance experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from polar_ai.alpha_generation import (
    generate_heterogeneous_alpha,
    generate_homogeneous_alpha,
)
from polar_ai.algorithms import deactivation
from polar_ai.backends.selector import get_backend_name
from polar_ai.experiments import run_deactivation
from polar_ai.experiments.run_multi_seed_performance import (
    experiment_seeds,
    refresh_outputs,
)
from polar_ai.experiments.utils import (
    budget_values,
    common_params,
    dataset_configs,
    load_config,
    load_dataset,
    resolve_path,
    save_rows,
    spectral_params,
    stable_seed,
    timed_call,
)
from polar_ai.graph_io import GraphMatrices
from polar_ai.instance import Instance
from polar_ai.opinion_generation import (
    generate_polarized_opinions,
    generate_uniform_opinions,
)
from polar_ai.spectral import homogeneous_lambda_vector


DEFAULT_RESULTS_DIR = "../../results/additional/fiedler_correlated_performance"
FIEDLER_PREFIX = "fiedler-"
OPINION_CONSTRUCTION = "fiedler_rank_preserving"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config, base_dir = load_config(args.config)
    output_root = resolve_path(
        args.results_dir or config.get("results_dir", DEFAULT_RESULTS_DIR),
        base_dir,
    )
    seeds = experiment_seeds(config)
    selected_datasets = set(args.dataset or [])

    backend = get_backend_name()
    if bool(config.get("require_cpp_backend", False)) and backend != "cpp":
        raise RuntimeError(
            "The Fiedler-correlated performance experiment requires the C++ "
            "backend, but the active backend is Python. Install the project "
            "before running it."
        )
    print(f"[fiedler-correlated-performance] backend={backend}", flush=True)

    entries = config.get("datasets", [])
    if not isinstance(entries, list):
        raise ValueError("Config 'datasets' must be a list.")
    datasets = dataset_configs(config, base_dir)
    matched = False

    for dataset in datasets:
        if selected_datasets and dataset.name not in selected_datasets:
            continue
        matched = True
        _validate_dataset_folder_name(dataset.name)
        dataset_dir = output_root / dataset.name

        if not args.plot_only:
            resume = bool(config.get("resume", True))
            pending = [
                seed
                for seed in seeds
                if args.rerun_existing
                or not resume
                or not _seed_result_is_current(
                    _completed_seed_path(dataset_dir, seed)
                )
            ]
            if pending:
                graph = load_dataset(dataset)
                fiedler, fiedler_diag = compute_fiedler_vector(
                    graph.Wbar,
                    tol=float(config.get("fiedler_tol", 1.0e-6)),
                    max_iter=_optional_positive_int(
                        config.get("fiedler_max_iter", None),
                        "fiedler_max_iter",
                    ),
                )
                _write_fiedler_diagnostics(
                    dataset_dir,
                    dataset.name,
                    graph,
                    fiedler_diag,
                )
                for seed in pending:
                    seed_path = _completed_seed_path(dataset_dir, seed)
                    print(
                        "[fiedler-correlated-performance] "
                        f"dataset={dataset.name} seed={seed} status=run",
                        flush=True,
                    )
                    run_seed(
                        config,
                        graph=graph,
                        fiedler=fiedler,
                        fiedler_diagnostics=fiedler_diag,
                        dataset_name=dataset.name,
                        seed=int(seed),
                        output_path=seed_path,
                    )
                    refresh_outputs(
                        dataset_dir,
                        dataset_name=dataset.name,
                        seeds=seeds,
                        generate_plots=False,
                    )
            for seed in seeds:
                if seed not in pending:
                    print(
                        "[fiedler-correlated-performance] "
                        f"dataset={dataset.name} seed={seed} status=skip",
                        flush=True,
                    )

        outputs = refresh_outputs(
            dataset_dir,
            dataset_name=dataset.name,
            seeds=seeds,
            generate_plots=bool(config.get("generate_plots", True)),
        )
        for path in outputs:
            print(path, flush=True)

    if selected_datasets and not matched:
        raise ValueError(
            "None of the requested datasets occur in the config: "
            + ", ".join(sorted(selected_datasets))
        )
    return 0


def run_seed(
    config: dict[str, Any],
    *,
    graph: GraphMatrices,
    fiedler: np.ndarray,
    fiedler_diagnostics: dict[str, float | int],
    dataset_name: str,
    seed: int,
    output_path: Path,
) -> Path:
    """Run every configured structure-correlated setting for one seed."""

    rows: list[dict[str, Any]] = []
    eps, lambda_cap = spectral_params(config)
    for base_opinion_setting in base_opinion_settings(config):
        opinion_setting = f"{FIEDLER_PREFIX}{base_opinion_setting}"
        for alpha_setting in alpha_settings(config):
            instance = build_fiedler_instance(
                graph,
                fiedler=fiedler,
                dataset_name=dataset_name,
                base_opinion_setting=base_opinion_setting,
                alpha_setting=alpha_setting,
                seed=seed,
                eps=eps,
                lambda_cap=lambda_cap,
            )
            print(
                "[fiedler-correlated-performance] "
                f"dataset={dataset_name} seed={seed} "
                f"opinion={opinion_setting} alpha={alpha_setting}",
                flush=True,
            )
            baseline, baseline_runtime = timed_call(
                deactivation.no_intervention,
                instance,
                **common_params(config),
            )
            setting_rows = run_deactivation._run_budgets(
                config,
                instance,
                dataset_name,
                opinion_setting,
                alpha_setting,
                budget_values(config, instance.n),
                baseline.objective,
                baseline_runtime,
                seed,
                {},
            )
            opinion_diag = opinion_diagnostics(
                instance.s,
                graph.Wbar,
                fiedler,
            )
            for row in setting_rows:
                row.update(
                    {
                        "base_opinion_setting": base_opinion_setting,
                        "opinion_construction": OPINION_CONSTRUCTION,
                        "fiedler_eigenvalue": float(
                            fiedler_diagnostics["fiedler_eigenvalue"]
                        ),
                        "fiedler_residual_2": float(
                            fiedler_diagnostics["fiedler_residual_2"]
                        ),
                        **opinion_diag,
                    }
                )
            rows.extend(setting_rows)
    return save_rows(rows, output_path)


def compute_fiedler_vector(
    Wbar: sparse.spmatrix,
    *,
    tol: float = 1.0e-6,
    max_iter: int | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Return a canonical Fiedler vector of the symmetrized graph Laplacian."""

    Wbar = Wbar.tocsr().astype(float)
    n = Wbar.shape[0]
    if Wbar.shape != (n, n):
        raise ValueError("Wbar must be square.")
    if n < 2:
        raise ValueError("At least two nodes are required for a Fiedler vector.")
    if tol <= 0.0:
        raise ValueError("fiedler_tol must be positive.")
    component_count = int(
        connected_components(Wbar, directed=False, return_labels=False)
    )
    if component_count != 1:
        raise ValueError(
            "The symmetrized graph must be connected to define a unique "
            f"nonzero Fiedler mode; found {component_count} components."
        )

    degree = np.asarray(Wbar.sum(axis=1)).ravel()
    laplacian = (sparse.diags(degree, format="csr") - Wbar).tocsr()
    if n == 2:
        dense_values, dense_vectors = np.linalg.eigh(laplacian.toarray())
        eigenvalue = float(dense_values[1])
        vector = np.asarray(dense_vectors[:, 1], dtype=float)
    else:
        v0 = np.linspace(-1.0, 1.0, n, dtype=float)
        v0 -= float(v0.mean())
        try:
            values, vectors = eigsh(
                laplacian,
                k=2,
                which="SM",
                tol=tol,
                maxiter=max_iter,
                v0=v0,
            )
        except ArpackNoConvergence as exc:
            converged = 0 if exc.eigenvalues is None else len(exc.eigenvalues)
            raise RuntimeError(
                "Fiedler eigensolver did not converge to two eigenpairs "
                f"(converged={converged}). Increase fiedler_max_iter or "
                "relax fiedler_tol."
            ) from exc
        order = np.argsort(values)
        eigenvalue = float(values[order[1]])
        vector = np.asarray(vectors[:, order[1]], dtype=float)

    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise RuntimeError("The computed Fiedler vector has zero or invalid norm.")
    vector /= norm

    # Eigenvectors are defined only up to a global sign.  Fix that ambiguity so
    # repeated runs and saved selected-node labels remain reproducible.
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0.0:
        vector = -vector

    residual = laplacian @ vector - eigenvalue * vector
    zero_tol = max(1.0e-12, 10.0 * np.finfo(float).eps)
    diagnostics: dict[str, float | int] = {
        "fiedler_eigenvalue": eigenvalue,
        "fiedler_residual_2": float(np.linalg.norm(residual)),
        "fiedler_mean": float(vector.mean()),
        "fiedler_positive_count": int(np.count_nonzero(vector > zero_tol)),
        "fiedler_negative_count": int(np.count_nonzero(vector < -zero_tol)),
        "fiedler_near_zero_count": int(np.count_nonzero(np.abs(vector) <= zero_tol)),
    }
    return vector, diagnostics


def build_fiedler_instance(
    graph: GraphMatrices,
    *,
    fiedler: np.ndarray,
    dataset_name: str,
    base_opinion_setting: str,
    alpha_setting: str,
    seed: int,
    eps: float,
    lambda_cap: float,
) -> Instance:
    """Construct an instance by ordering base opinions along the Fiedler mode."""

    n = graph.P.shape[0]
    fiedler = np.asarray(fiedler, dtype=float)
    if fiedler.shape != (n,):
        raise ValueError(f"fiedler must have shape ({n},).")

    opinion_seed = stable_seed(
        seed,
        dataset_name,
        base_opinion_setting,
        "s",
    )
    if base_opinion_setting == "uniform":
        base = generate_uniform_opinions(n, seed=opinion_seed)
    elif base_opinion_setting == "polarized":
        base = generate_polarized_opinions(n, seed=opinion_seed)
    else:
        raise ValueError(
            f"Unknown base opinion setting: {base_opinion_setting}"
        )
    # Preserve the complete multiset of generated opinions—including its mean,
    # variance, sign balance, and magnitude distribution—while making opinion
    # order monotone in the Fiedler coordinate. Stable sorting supplies a
    # deterministic index-based tie break if Fiedler components coincide.
    vertex_order = np.argsort(fiedler, kind="stable")
    sorted_opinions = np.sort(base, kind="stable")
    s = np.empty_like(sorted_opinions)
    s[vertex_order] = sorted_opinions

    if alpha_setting == "homogeneous":
        alpha = generate_homogeneous_alpha(n, value=2.0)
    elif alpha_setting == "heterogeneous":
        alpha = generate_heterogeneous_alpha(
            n,
            seed=stable_seed(seed, dataset_name, alpha_setting, "alpha"),
        )
    else:
        raise ValueError(f"Unknown alpha setting: {alpha_setting}")

    lambda_vec = homogeneous_lambda_vector(
        graph.P,
        alpha,
        eps=eps,
        cap=lambda_cap,
    )
    opinion_setting = f"{FIEDLER_PREFIX}{base_opinion_setting}"
    metadata = dict(graph.metadata)
    metadata.update(
        {
            "dataset": dataset_name,
            "node_labels": list(graph.nodes),
            "opinion_setting": opinion_setting,
            "base_opinion_setting": base_opinion_setting,
            "opinion_construction": OPINION_CONSTRUCTION,
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


def opinion_diagnostics(
    opinions: np.ndarray,
    Wbar: sparse.spmatrix,
    fiedler: np.ndarray,
) -> dict[str, float]:
    """Return compact checks of the generated network-correlated opinions."""

    opinions = np.asarray(opinions, dtype=float)
    fiedler = np.asarray(fiedler, dtype=float)
    opinion_ranks = _average_ranks(opinions)
    fiedler_ranks = _average_ranks(fiedler)
    rank_correlation = float(np.corrcoef(opinion_ranks, fiedler_ranks)[0, 1])
    degree = np.asarray(Wbar.sum(axis=1)).ravel()
    laplacian = sparse.diags(degree, format="csr") - Wbar
    squared_norm = float(opinions @ opinions)
    rayleigh = (
        float(opinions @ (laplacian @ opinions)) / squared_norm
        if squared_norm > 0.0
        else 0.0
    )
    return {
        "opinion_fiedler_spearman_correlation": rank_correlation,
        "opinion_mean": float(opinions.mean()),
        "opinion_std": float(opinions.std()),
        "opinion_positive_fraction": float(np.mean(opinions >= 0.0)),
        "opinion_graph_rayleigh_quotient": rayleigh,
    }


def base_opinion_settings(config: dict[str, Any]) -> list[str]:
    values = config.get("base_opinion_settings", ["uniform", "polarized"])
    settings = [str(value) for value in values]
    if not settings:
        raise ValueError("base_opinion_settings must not be empty.")
    unknown = set(settings) - {"uniform", "polarized"}
    if unknown:
        raise ValueError(
            "Unknown base opinion settings: " + ", ".join(sorted(unknown))
        )
    return list(dict.fromkeys(settings))


def alpha_settings(config: dict[str, Any]) -> list[str]:
    values = config.get("alpha_settings", ["homogeneous", "heterogeneous"])
    settings = [str(value) for value in values]
    if not settings:
        raise ValueError("alpha_settings must not be empty.")
    unknown = set(settings) - {"homogeneous", "heterogeneous"}
    if unknown:
        raise ValueError("Unknown alpha settings: " + ", ".join(sorted(unknown)))
    return list(dict.fromkeys(settings))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks, using zero-based rank values."""

    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _write_fiedler_diagnostics(
    dataset_dir: Path,
    dataset_name: str,
    graph: GraphMatrices,
    diagnostics: dict[str, float | int],
) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / "fiedler_diagnostics.csv"
    row = {
        "dataset": dataset_name,
        "n": int(graph.W.shape[0]),
        "m": int(graph.W.nnz),
        "laplacian": "combinatorial_of_Wbar",
        "directed_graph_handling": "symmetrized_Wbar",
        **diagnostics,
    }
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def _completed_seed_path(dataset_dir: Path, seed: int) -> Path:
    return dataset_dir / "seeds" / f"seed_{int(seed)}" / "deactivation.csv"


def _seed_result_is_current(path: Path) -> bool:
    """Return whether a completed seed uses the current opinion construction."""

    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        constructions = pd.read_csv(
            path,
            usecols=["opinion_construction"],
        )["opinion_construction"]
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False
    return bool(
        not constructions.empty
        and constructions.astype(str).eq(OPINION_CONSTRUCTION).all()
    )


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be positive when specified.")
    return parsed


def _validate_dataset_folder_name(name: str) -> None:
    path = Path(name)
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or path.name != name
    ):
        raise ValueError(
            f"Dataset name must be a single safe folder name: {name!r}"
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat the paper's deactivation benchmark over multiple seeds "
            "with opinions ordered by the graph Fiedler vector."
        )
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name to run; repeat to select multiple datasets.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Optional results root override.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate aggregate CSVs and plots from completed seed CSVs.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Rerun and overwrite completed per-seed CSVs.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
