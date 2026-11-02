"""Decompose AI-driven polarization changes into dispersion and mean terms.

This additional experiment reproduces the fixed-lambda alpha sweep used for
the transformation-effect figure.  At every equilibrium it uses

    ||z||^2 = ||z - mean(z) 1||^2 + n mean(z)^2

to distinguish increased dispersion around the population mean from a change
in the squared population mean.
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
from polar_ai.experiments.plot_polarization_decomposition import (
    plot_polarization_decomposition,
    write_polarization_decomposition_summary,
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
from polar_ai.spectral import homogeneous_lambda_vector


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
    "lambda_value",
    "mean_active",
    "mean_baseline",
    "mean_shift",
    "absolute_mean_shift",
    "dispersion_active",
    "dispersion_baseline",
    "mean_component_active",
    "mean_component_baseline",
    "polarization_active",
    "polarization_baseline",
    "disagreement_active",
    "disagreement_baseline",
    "pd_active",
    "pd_baseline",
    "pd_ratio",
    "polarization_ratio",
    "dispersion_ratio",
    "mean_component_ratio",
    "dispersion_change",
    "mean_component_change",
    "polarization_change",
    "dispersion_change_over_baseline_polarization",
    "mean_component_change_over_baseline_polarization",
    "polarization_change_over_baseline_polarization",
    "dispersion_fraction_of_polarization_change",
    "mean_component_fraction_of_polarization_change",
    "decomposition_residual_active",
    "decomposition_residual_baseline",
    "active_equilibrium_iterations",
    "baseline_equilibrium_iterations",
    "active_equilibrium_converged",
    "baseline_equilibrium_converged",
    "active_equilibrium_residual_inf",
    "baseline_equilibrium_residual_inf",
    "runtime_sec",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config, base_dir = load_config(args.config)
    output_root = resolve_path(
        args.results_dir
        or config.get(
            "results_dir", "../results/additional/polarization_decomposition"
        ),
        base_dir,
    )
    selected_datasets = set(args.dataset or [])
    backend = get_backend_name()
    if bool(config.get("require_cpp_backend", False)) and backend != "cpp":
        raise RuntimeError(
            "The polarization-decomposition experiment requires the C++ "
            "backend, but the active backend is Python. Install the project "
            "before running the experiment."
        )
    print(f"[polarization-decomposition] backend={backend}", flush=True)

    for dataset in dataset_configs(config, base_dir):
        if selected_datasets and dataset.name not in selected_datasets:
            continue
        dataset_dir = output_root / dataset.name
        result_path = dataset_dir / "polarization_decomposition.csv"
        summary_path = dataset_dir / "polarization_decomposition_summary.csv"
        figures_dir = dataset_dir / "figures"

        if args.plot_only:
            _write_outputs(result_path, summary_path, figures_dir, config)
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
                "[polarization-decomposition] "
                f"dataset={row['dataset']} opinion={row['opinion_setting']} "
                f"alpha={row['alpha']:.1f} pd_ratio={row['pd_ratio']:.6f} "
                "dispersion_delta/P1="
                f"{row['dispersion_change_over_baseline_polarization']:.6e} "
                "mean_delta/P1="
                f"{row['mean_component_change_over_baseline_polarization']:.6e}",
                flush=True,
            )

        if bool(config.get("generate_plots", True)):
            _write_outputs(result_path, summary_path, figures_dir, config)
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
    completed_keys: set[tuple[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Return all missing decomposition rows for one loaded graph."""

    completed = set() if completed_keys is None else set(completed_keys)
    alpha_values = _alpha_values(config)
    reference_alpha = float(
        config.get("lambda_reference_alpha", max(alpha_values))
    )
    if reference_alpha < max(alpha_values):
        raise ValueError(
            "lambda_reference_alpha must be at least the largest swept alpha."
        )

    seed = int(config.get("seed", 0))
    eps, lambda_cap = spectral_params(config)
    eq_params = common_params(config)
    n = int(P.shape[0])
    lambda_vec = homogeneous_lambda_vector(
        P,
        np.full(n, reference_alpha, dtype=float),
        eps=eps,
        cap=lambda_cap,
    )
    baseline_alpha = np.ones(n, dtype=float)

    rows: list[dict[str, Any]] = []
    for opinion_setting in _opinion_settings(config):
        missing_alphas = [
            alpha
            for alpha in alpha_values
            if (opinion_setting, float(alpha)) not in completed
        ]
        if not missing_alphas:
            continue

        s = _opinions(opinion_setting, n, seed, dataset)
        metadata = {
            "dataset": dataset.name,
            "node_labels": node_labels,
            "opinion_setting": opinion_setting,
        }
        baseline_instance = Instance(
            W=W,
            P=P,
            Wbar=Wbar,
            s=s,
            lambda_vec=lambda_vec,
            alpha=baseline_alpha,
            metadata=metadata,
        )
        pd_baseline, z_baseline, baseline_diag = evaluate_deactivation(
            baseline_instance,
            set(),
            **eq_params,
        )
        _require_convergence(
            config,
            baseline_diag,
            dataset=dataset.name,
            opinion_setting=opinion_setting,
            alpha=1.0,
            role="baseline",
        )
        baseline_decomp = polarization_decomposition(z_baseline)

        for alpha_value in missing_alphas:
            row_start = time.perf_counter()
            active_instance = Instance(
                W=W,
                P=P,
                Wbar=Wbar,
                s=s,
                lambda_vec=lambda_vec,
                alpha=np.full(n, alpha_value, dtype=float),
                metadata=metadata,
            )
            pd_active, z_active, active_diag = evaluate_deactivation(
                active_instance,
                set(),
                **eq_params,
            )
            _require_convergence(
                config,
                active_diag,
                dataset=dataset.name,
                opinion_setting=opinion_setting,
                alpha=alpha_value,
                role="active",
            )
            active_decomp = polarization_decomposition(z_active)

            dispersion_change = (
                active_decomp["dispersion"] - baseline_decomp["dispersion"]
            )
            mean_component_change = (
                active_decomp["mean_component"]
                - baseline_decomp["mean_component"]
            )
            polarization_change = (
                active_decomp["polarization"]
                - baseline_decomp["polarization"]
            )
            baseline_polarization = baseline_decomp["polarization"]
            change_scale = max(1.0, abs(baseline_polarization))
            change_tol = float(config.get("change_fraction_tolerance", 1e-12))
            has_nonzero_change = abs(polarization_change) > (
                change_tol * change_scale
            )

            rows.append(
                {
                    "experiment": "polarization_decomposition",
                    "dataset": dataset.name,
                    "n": n,
                    "m": int(W.nnz),
                    "opinion_setting": opinion_setting,
                    "alpha_setting": "homogeneous",
                    "seed": seed,
                    "backend": backend,
                    "lambda_regime": "fixed_at_reference",
                    "alpha": float(alpha_value),
                    "lambda_reference_alpha": reference_alpha,
                    "lambda_value": (
                        float(lambda_vec[0]) if lambda_vec.size else 0.0
                    ),
                    "mean_active": active_decomp["mean"],
                    "mean_baseline": baseline_decomp["mean"],
                    "mean_shift": (
                        active_decomp["mean"] - baseline_decomp["mean"]
                    ),
                    "absolute_mean_shift": abs(
                        active_decomp["mean"] - baseline_decomp["mean"]
                    ),
                    "dispersion_active": active_decomp["dispersion"],
                    "dispersion_baseline": baseline_decomp["dispersion"],
                    "mean_component_active": active_decomp["mean_component"],
                    "mean_component_baseline": baseline_decomp[
                        "mean_component"
                    ],
                    "polarization_active": active_decomp["polarization"],
                    "polarization_baseline": baseline_polarization,
                    "disagreement_active": float(active_diag["disagreement"]),
                    "disagreement_baseline": float(
                        baseline_diag["disagreement"]
                    ),
                    "pd_active": float(pd_active),
                    "pd_baseline": float(pd_baseline),
                    "pd_ratio": _safe_ratio(pd_active, pd_baseline),
                    "polarization_ratio": _safe_ratio(
                        active_decomp["polarization"], baseline_polarization
                    ),
                    "dispersion_ratio": _safe_ratio(
                        active_decomp["dispersion"],
                        baseline_decomp["dispersion"],
                    ),
                    "mean_component_ratio": _safe_ratio(
                        active_decomp["mean_component"],
                        baseline_decomp["mean_component"],
                    ),
                    "dispersion_change": dispersion_change,
                    "mean_component_change": mean_component_change,
                    "polarization_change": polarization_change,
                    "dispersion_change_over_baseline_polarization": _safe_ratio(
                        dispersion_change, baseline_polarization
                    ),
                    "mean_component_change_over_baseline_polarization": _safe_ratio(
                        mean_component_change, baseline_polarization
                    ),
                    "polarization_change_over_baseline_polarization": _safe_ratio(
                        polarization_change, baseline_polarization
                    ),
                    "dispersion_fraction_of_polarization_change": (
                        dispersion_change / polarization_change
                        if has_nonzero_change
                        else float("nan")
                    ),
                    "mean_component_fraction_of_polarization_change": (
                        mean_component_change / polarization_change
                        if has_nonzero_change
                        else float("nan")
                    ),
                    "decomposition_residual_active": active_decomp["residual"],
                    "decomposition_residual_baseline": baseline_decomp[
                        "residual"
                    ],
                    "active_equilibrium_iterations": int(
                        active_diag["equilibrium_iterations"]
                    ),
                    "baseline_equilibrium_iterations": int(
                        baseline_diag["equilibrium_iterations"]
                    ),
                    "active_equilibrium_converged": bool(
                        active_diag["equilibrium_converged"]
                    ),
                    "baseline_equilibrium_converged": bool(
                        baseline_diag["equilibrium_converged"]
                    ),
                    "active_equilibrium_residual_inf": float(
                        active_diag["equilibrium_residual_inf"]
                    ),
                    "baseline_equilibrium_residual_inf": float(
                        baseline_diag["equilibrium_residual_inf"]
                    ),
                    "runtime_sec": time.perf_counter() - row_start,
                }
            )
    return rows


def polarization_decomposition(z: np.ndarray) -> dict[str, float]:
    """Return the exact dispersion/mean decomposition of ``||z||^2``."""

    values = np.asarray(z, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("z must be a nonempty one-dimensional vector.")
    mean = float(np.mean(values))
    centered = values - mean
    dispersion = float(np.dot(centered, centered))
    mean_component = float(values.size * mean * mean)
    polarization = float(np.dot(values, values))
    residual = polarization - dispersion - mean_component
    tolerance = 1e-10 * max(1.0, abs(polarization))
    if abs(residual) > tolerance:
        raise RuntimeError(
            "Polarization decomposition failed numerically: "
            f"residual={residual:.6e}, tolerance={tolerance:.6e}."
        )
    return {
        "mean": mean,
        "dispersion": dispersion,
        "mean_component": mean_component,
        "polarization": polarization,
        "residual": residual,
    }


def _write_outputs(
    result_path: Path,
    summary_path: Path,
    figures_dir: Path,
    config: dict[str, Any],
) -> None:
    summary = write_polarization_decomposition_summary(
        result_path,
        summary_path,
        min_relative_change=float(
            config.get("summary_min_relative_polarization_change", 1e-10)
        ),
    )
    print(summary, flush=True)
    for path in plot_polarization_decomposition(result_path, figures_dir):
        print(path, flush=True)


def _append_result_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unexpected = set(row) - set(RESULT_COLUMNS)
    if unexpected:
        raise ValueError(f"Unexpected result columns: {sorted(unexpected)}")
    missing = set(RESULT_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _completed_keys(path: Path) -> set[tuple[str, float]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path)
    required = {"opinion_setting", "alpha"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Existing decomposition CSV is missing columns: {sorted(missing)}"
        )
    return {
        (str(row.opinion_setting), float(row.alpha))
        for row in frame.itertuples(index=False)
    }


def _require_convergence(
    config: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    dataset: str,
    opinion_setting: str,
    alpha: float,
    role: str,
) -> None:
    if not bool(config.get("fail_on_nonconvergence", True)):
        return
    if bool(diagnostics["equilibrium_converged"]):
        return
    raise RuntimeError(
        "Equilibrium did not converge: "
        f"dataset={dataset}, opinion={opinion_setting}, alpha={alpha}, "
        f"role={role}."
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    tolerance = np.finfo(float).eps * max(1.0, abs(float(numerator)))
    if abs(float(denominator)) <= tolerance:
        return float("nan")
    return float(numerator / denominator)


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
    if 1.0 not in values:
        raise ValueError("The alpha sweep must include alpha=1 as its baseline.")
    return values


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
            "Decompose the polarization change in the fixed-lambda alpha "
            "sweep into dispersion and mean components."
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
        help="Regenerate the summary CSV and figures from existing rows.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
