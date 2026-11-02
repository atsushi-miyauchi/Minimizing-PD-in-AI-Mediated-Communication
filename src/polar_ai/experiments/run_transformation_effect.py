"""Run the transformation-effect experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from polar_ai.algorithms.common import evaluate_deactivation, instance_with_alpha
from polar_ai.alpha_generation import generate_homogeneous_alpha
from polar_ai.instance import Instance
from polar_ai.opinion_generation import (
    generate_polarized_opinions,
    generate_uniform_opinions,
)
from polar_ai.spectral import homogeneous_lambda_vector
from polar_ai.experiments.utils import (
    DatasetConfig,
    common_params,
    dataset_configs,
    default_output_path,
    load_config,
    load_dataset,
    parse_args,
    save_rows,
    spectral_params,
    stable_seed,
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args("Run transformation-effect sweep.", argv)
    config, base_dir = load_config(args.config)
    output_path = default_output_path(
        config, base_dir, "transformation_effect.csv", args.output
    )
    run_config(config, base_dir, output_path)
    print(output_path)
    return 0


def run_config(
    config: dict[str, Any], base_dir: Path, output_path: Path
) -> Path:
    """Run the configured transformation-effect experiments into one CSV."""

    rows: list[dict[str, object]] = []
    seed = int(config.get("seed", 0))
    eps, lambda_cap = spectral_params(config)
    eq_params = common_params(config)

    for dataset in dataset_configs(config, base_dir):
        graph = load_dataset(dataset)
        alpha_values = _alpha_values(config)
        max_alpha = max(alpha_values)
        lambda_vec = homogeneous_lambda_vector(
            graph.P,
            np.full(graph.P.shape[0], max_alpha, dtype=float),
            eps=eps,
            cap=lambda_cap,
        )

        opinion_settings = [
            str(value)
            for value in config.get("opinion_settings", ["uniform", "polarized"])
        ]
        for opinion_setting in opinion_settings:
            s = _opinions(opinion_setting, graph.P.shape[0], seed, dataset)
            alpha_deactivated = np.ones(graph.P.shape[0], dtype=float)
            base_instance = Instance(
                W=graph.W,
                P=graph.P,
                Wbar=graph.Wbar,
                s=s,
                lambda_vec=lambda_vec,
                alpha=alpha_deactivated,
                metadata={
                    "dataset": dataset.name,
                    "node_labels": list(graph.nodes),
                    "opinion_setting": opinion_setting,
                },
            )
            pd_deactivated, _, deactivated_diag = evaluate_deactivation(
                base_instance,
                set(),
                **eq_params,
            )

            for alpha_value in alpha_values:
                active = instance_with_alpha(
                    base_instance,
                    generate_homogeneous_alpha(graph.P.shape[0], value=alpha_value),
                )
                pd_active, _, active_diag = evaluate_deactivation(
                    active,
                    set(),
                    **eq_params,
                )
                ratio = pd_active / pd_deactivated if pd_deactivated != 0 else np.nan
                rows.append(
                    {
                        "dataset": dataset.name,
                        "n": graph.P.shape[0],
                        "m": graph.W.nnz,
                        "opinion_setting": opinion_setting,
                        "alpha": alpha_value,
                        "lambda": float(lambda_vec[0]) if lambda_vec.size else 0.0,
                        "pd_active": pd_active,
                        "pd_deactivated": pd_deactivated,
                        "ratio": ratio,
                        "ratio_average_over_opinions": False,
                        "active_equilibrium_iterations": active_diag[
                            "equilibrium_iterations"
                        ],
                        "deactivated_equilibrium_iterations": deactivated_diag[
                            "equilibrium_iterations"
                        ],
                    }
                )

    return save_rows(rows, output_path)


def _alpha_values(config: dict[str, object]) -> list[float]:
    if "alpha_values" in config:
        return [float(value) for value in config["alpha_values"]]  # type: ignore[index]
    start = float(config.get("alpha_start", 1.0))
    stop = float(config.get("alpha_stop", 3.0))
    step = float(config.get("alpha_step", 0.1))
    count = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(count + 1)]


def _opinions(
    opinion_setting: str, n: int, seed: int, dataset: DatasetConfig
) -> np.ndarray:
    if opinion_setting == "uniform":
        return generate_uniform_opinions(
            n, seed=stable_seed(seed, dataset.name, opinion_setting, "s")
        )
    return generate_polarized_opinions(
        n, seed=stable_seed(seed, dataset.name, opinion_setting, "s")
    )


if __name__ == "__main__":
    raise SystemExit(main())
