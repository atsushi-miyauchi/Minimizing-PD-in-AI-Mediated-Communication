"""Compare fixed-lambda and recomputed-lambda alpha sweeps.

This is an additional experiment kept separate from the original paper suite.
For the fixed regime, lambda is computed once at a configured reference alpha
and then held constant while alpha varies.  The comparison regime recomputes
lambda at every alpha value using the original stability rule.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from polar_ai.algorithms.common import evaluate_deactivation
from polar_ai.backends.selector import get_backend_name
from polar_ai.experiments.plot_fixed_lambda_alpha_sweep import (
    plot_fixed_lambda_alpha_sweep,
    write_fixed_lambda_comparison,
)
from polar_ai.experiments.utils import (
    DatasetConfig,
    common_params,
    dataset_configs,
    load_config,
    load_dataset,
    resolve_path,
    spectral_params,
    stable_seed,
)
from polar_ai.instance import Instance
from polar_ai.opinion_generation import (
    generate_polarized_opinions,
    generate_uniform_opinions,
)
from polar_ai.spectral import (
    homogeneous_lambda_vector,
    spectral_radius_lambda_p_gamma,
)


FIXED_REGIME = "fixed_at_reference"
RECOMPUTED_REGIME = "recomputed_each_alpha"
VALID_REGIMES = {FIXED_REGIME, RECOMPUTED_REGIME}

RESULT_COLUMNS = [
    "experiment",
    "dataset",
    "n",
    "m",
    "opinion_setting",
    "alpha_setting",
    "seed",
    "backend",
    "lambda_regime",
    "alpha",
    "lambda_reference_alpha",
    "lambda_rule_alpha",
    "lambda_value",
    "stability_spectral_radius",
    "pd_active",
    "pd_deactivated",
    "ratio",
    "active_equilibrium_iterations",
    "deactivated_equilibrium_iterations",
    "active_equilibrium_converged",
    "deactivated_equilibrium_converged",
    "active_equilibrium_residual_inf",
    "deactivated_equilibrium_residual_inf",
    "runtime_sec",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config, base_dir = load_config(args.config)
    output_root = resolve_path(
        args.results_dir
        or config.get(
            "results_dir", "../results/additional/fixed_lambda_alpha_sweep"
        ),
        base_dir,
    )
    selected_datasets = set(args.dataset or [])
    backend = get_backend_name()
    if bool(config.get("require_cpp_backend", False)) and backend != "cpp":
        raise RuntimeError(
            "The fixed-lambda sweep requires the C++ backend, but the active "
            "backend is Python. Install the project before running the sweep."
        )
    print(f"[fixed-lambda-alpha-sweep] backend={backend}", flush=True)

    for dataset in dataset_configs(config, base_dir):
        if selected_datasets and dataset.name not in selected_datasets:
            continue
        dataset_dir = output_root / dataset.name
        result_path = dataset_dir / "fixed_lambda_alpha_sweep.csv"
        comparison_path = dataset_dir / "fixed_lambda_alpha_comparison.csv"
        figures_dir = dataset_dir / "figures"

        if args.plot_only:
            _write_outputs(result_path, comparison_path, figures_dir)
            continue
        if not bool(config.get("resume", True)) and result_path.exists():
            result_path.unlink()

        graph = load_dataset(dataset)
        rows = run_dataset(
            config,
            dataset,
            graph.W,
            graph.P,
            graph.Wbar,
            list(graph.nodes),
            backend=backend,
            completed_keys=_completed_keys(result_path),
        )
        for row in rows:
            _append_result_row(result_path, row)
            print(
                "[fixed-lambda-alpha-sweep] "
                f"dataset={row['dataset']} opinion={row['opinion_setting']} "
                f"regime={row['lambda_regime']} alpha={row['alpha']:.1f} "
                f"lambda={row['lambda_value']:.6f} "
                f"ratio={row['ratio']:.6f}",
                flush=True,
            )

        if bool(config.get("generate_plots", True)):
            _write_outputs(result_path, comparison_path, figures_dir)
    return 0


def run_dataset(
    config: dict[str, Any],
    dataset: DatasetConfig,
    W: Any,
    P: Any,
    Wbar: Any,
    node_labels: list[Any],
    *,
    backend: str,
    completed_keys: set[tuple[str, str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Return all missing sweep rows for one already-loaded graph."""

    completed = set() if completed_keys is None else set(completed_keys)
    alpha_values = _alpha_values(config)
    reference_alpha = float(
        config.get("lambda_reference_alpha", max(alpha_values))
    )
    if reference_alpha < max(alpha_values):
        raise ValueError(
            "lambda_reference_alpha must be at least the largest swept alpha."
        )
    regimes = _lambda_regimes(config)
    seed = int(config.get("seed", 0))
    eps, lambda_cap = spectral_params(config)
    eq_params = common_params(config)
    n = int(P.shape[0])
    alpha_reference = np.full(n, reference_alpha, dtype=float)
    fixed_lambda = homogeneous_lambda_vector(
        P,
        alpha_reference,
        eps=eps,
        cap=lambda_cap,
    )

    rows: list[dict[str, Any]] = []
    for opinion_setting in _opinion_settings(config):
        s = _opinions(opinion_setting, n, seed, dataset)
        fixed_deactivated: tuple[float, dict[str, Any]] | None = None
        for regime in regimes:
            for alpha_value in alpha_values:
                key = (opinion_setting, regime, float(alpha_value))
                if key in completed:
                    continue
                row_start = time.perf_counter()
                alpha_vector = np.full(n, alpha_value, dtype=float)
                if regime == FIXED_REGIME:
                    lambda_vec = fixed_lambda
                    lambda_rule_alpha = reference_alpha
                else:
                    lambda_vec = homogeneous_lambda_vector(
                        P,
                        alpha_vector,
                        eps=eps,
                        cap=lambda_cap,
                    )
                    lambda_rule_alpha = alpha_value

                active_instance = Instance(
                    W=W,
                    P=P,
                    Wbar=Wbar,
                    s=s,
                    lambda_vec=lambda_vec,
                    alpha=alpha_vector,
                    metadata={
                        "dataset": dataset.name,
                        "node_labels": node_labels,
                        "opinion_setting": opinion_setting,
                    },
                )
                pd_active, _, active_diag = evaluate_deactivation(
                    active_instance,
                    set(),
                    **eq_params,
                )

                if regime == FIXED_REGIME and fixed_deactivated is not None:
                    pd_deactivated, deactivated_diag = fixed_deactivated
                else:
                    deactivated_instance = Instance(
                        W=W,
                        P=P,
                        Wbar=Wbar,
                        s=s,
                        lambda_vec=lambda_vec,
                        alpha=np.ones(n, dtype=float),
                        metadata=dict(active_instance.metadata),
                    )
                    pd_deactivated, _, deactivated_diag = evaluate_deactivation(
                        deactivated_instance,
                        set(),
                        **eq_params,
                    )
                    if regime == FIXED_REGIME:
                        fixed_deactivated = (
                            float(pd_deactivated),
                            dict(deactivated_diag),
                        )

                _require_convergence(
                    config,
                    active_diag,
                    deactivated_diag,
                    dataset=dataset.name,
                    opinion_setting=opinion_setting,
                    regime=regime,
                    alpha=alpha_value,
                )
                stability_radius = spectral_radius_lambda_p_gamma(
                    P,
                    lambda_vec,
                    alpha_vector,
                )
                ratio = (
                    float(pd_active / pd_deactivated)
                    if pd_deactivated != 0.0
                    else float("nan")
                )
                rows.append(
                    {
                        "experiment": "fixed_lambda_alpha_sweep",
                        "dataset": dataset.name,
                        "n": n,
                        "m": int(W.nnz),
                        "opinion_setting": opinion_setting,
                        "alpha_setting": "homogeneous",
                        "seed": seed,
                        "backend": backend,
                        "lambda_regime": regime,
                        "alpha": float(alpha_value),
                        "lambda_reference_alpha": reference_alpha,
                        "lambda_rule_alpha": float(lambda_rule_alpha),
                        "lambda_value": (
                            float(lambda_vec[0]) if lambda_vec.size else 0.0
                        ),
                        "stability_spectral_radius": float(stability_radius),
                        "pd_active": float(pd_active),
                        "pd_deactivated": float(pd_deactivated),
                        "ratio": ratio,
                        "active_equilibrium_iterations": int(
                            active_diag["equilibrium_iterations"]
                        ),
                        "deactivated_equilibrium_iterations": int(
                            deactivated_diag["equilibrium_iterations"]
                        ),
                        "active_equilibrium_converged": bool(
                            active_diag["equilibrium_converged"]
                        ),
                        "deactivated_equilibrium_converged": bool(
                            deactivated_diag["equilibrium_converged"]
                        ),
                        "active_equilibrium_residual_inf": float(
                            active_diag["equilibrium_residual_inf"]
                        ),
                        "deactivated_equilibrium_residual_inf": float(
                            deactivated_diag["equilibrium_residual_inf"]
                        ),
                        "runtime_sec": time.perf_counter() - row_start,
                    }
                )
    return rows


def _write_outputs(
    result_path: Path, comparison_path: Path, figures_dir: Path
) -> None:
    comparison = write_fixed_lambda_comparison(result_path, comparison_path)
    print(comparison, flush=True)
    for path in plot_fixed_lambda_alpha_sweep(result_path, figures_dir):
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


def _completed_keys(path: Path) -> set[tuple[str, str, float]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path)
    required = {"opinion_setting", "lambda_regime", "alpha"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Existing sweep CSV is missing columns: {sorted(missing)}"
        )
    return {
        (str(row.opinion_setting), str(row.lambda_regime), float(row.alpha))
        for row in frame.itertuples(index=False)
    }


def _require_convergence(
    config: dict[str, Any],
    active_diag: dict[str, Any],
    deactivated_diag: dict[str, Any],
    *,
    dataset: str,
    opinion_setting: str,
    regime: str,
    alpha: float,
) -> None:
    if not bool(config.get("fail_on_nonconvergence", True)):
        return
    if bool(active_diag["equilibrium_converged"]) and bool(
        deactivated_diag["equilibrium_converged"]
    ):
        return
    raise RuntimeError(
        "Equilibrium did not converge: "
        f"dataset={dataset}, opinion={opinion_setting}, regime={regime}, "
        f"alpha={alpha}."
    )


def _alpha_values(config: dict[str, Any]) -> list[float]:
    if "alpha_values" in config:
        values = [float(value) for value in config["alpha_values"]]
    else:
        start = float(config.get("alpha_start", 1.0))
        stop = float(config.get("alpha_stop", 3.0))
        step = float(config.get("alpha_step", 0.1))
        if step <= 0.0 or stop < start:
            raise ValueError("Require alpha_step > 0 and alpha_stop >= alpha_start.")
        count = int(round((stop - start) / step))
        values = [round(start + index * step, 10) for index in range(count + 1)]
    if not values or any(value < 1.0 for value in values):
        raise ValueError("All alpha values must be at least one.")
    if values != sorted(set(values)):
        raise ValueError("Alpha values must be strictly increasing and unique.")
    return values


def _lambda_regimes(config: dict[str, Any]) -> list[str]:
    regimes = [
        str(value)
        for value in config.get(
            "lambda_regimes", [FIXED_REGIME, RECOMPUTED_REGIME]
        )
    ]
    invalid = set(regimes) - VALID_REGIMES
    if invalid:
        raise ValueError(f"Unknown lambda regimes: {sorted(invalid)}")
    if set(regimes) != VALID_REGIMES:
        raise ValueError(
            "The comparison experiment requires both fixed_at_reference and "
            "recomputed_each_alpha."
        )
    return regimes


def _opinion_settings(config: dict[str, Any]) -> list[str]:
    settings = [
        str(value)
        for value in config.get("opinion_settings", ["uniform", "polarized"])
    ]
    invalid = set(settings) - {"uniform", "polarized"}
    if invalid:
        raise ValueError(f"Unknown opinion settings: {sorted(invalid)}")
    return settings


def _opinions(
    opinion_setting: str,
    n: int,
    seed: int,
    dataset: DatasetConfig,
) -> np.ndarray:
    opinion_seed = stable_seed(seed, dataset.name, opinion_setting, "s")
    if opinion_setting == "uniform":
        return generate_uniform_opinions(n, seed=opinion_seed)
    return generate_polarized_opinions(n, seed=opinion_seed)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an alpha sweep with lambda fixed at a reference alpha "
            "against a sweep that recomputes lambda at every alpha."
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
        help="Regenerate the comparison CSV and figures from existing rows.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
