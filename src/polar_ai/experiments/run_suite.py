"""Run the selected experiment suite for every configured dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from polar_ai.experiments import (
    plot_results,
    run_deactivation,
    run_transformation_effect,
)
from polar_ai.experiments.utils import (
    dataset_configs,
    load_config,
    resolve_path,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config, base_dir = load_config(args.config)
    output_root = resolve_path(
        config.get("results_dir", config.get("output_dir", "../results")),
        base_dir,
    )
    entries = config.get("datasets", [])
    if not isinstance(entries, list):
        raise ValueError("Config 'datasets' must be a list.")
    datasets = dataset_configs(config, base_dir)

    names: set[str] = set()
    for entry, dataset in zip(entries, datasets):
        _validate_dataset_folder_name(dataset.name)
        if dataset.name in names:
            raise ValueError(f"Duplicate dataset name: {dataset.name}")
        names.add(dataset.name)

        dataset_dir = output_root / dataset.name
        figures_dir = dataset_dir / "figures"
        dataset_config: dict[str, Any] = dict(config)
        dataset_config["datasets"] = [entry]

        print(f"dataset={dataset.name} output={dataset_dir}")
        transformation_path = dataset_dir / "transformation_effect.csv"
        deactivation_path = dataset_dir / "deactivation.csv"
        run_transformation_effect.run_config(
            dataset_config, base_dir, transformation_path
        )
        print(transformation_path)
        run_deactivation.run_config(
            dataset_config, base_dir, deactivation_path
        )
        print(deactivation_path)

        figure_paths = plot_results.plot_deactivation_and_transformation_results(
            dataset_dir,
            figures_dir,
            timeout_sec=float(config.get("algorithm_timeout_sec", 1200.0)),
        )
        for path in figure_paths:
            print(path)

    return 0


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
            "Run deactivation and transformation-effect experiments for every "
            "configured dataset, then generate their figures."
        )
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
