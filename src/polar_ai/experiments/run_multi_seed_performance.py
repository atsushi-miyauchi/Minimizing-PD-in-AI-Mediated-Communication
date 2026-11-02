"""Repeat the paper's deactivation benchmark over multiple instance seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from polar_ai.backends.selector import get_backend_name
from polar_ai.experiments import run_deactivation
from polar_ai.experiments.plot_multi_seed_performance import (
    plot_multi_seed_performance,
)
from polar_ai.experiments.utils import (
    dataset_configs,
    load_config,
    resolve_path,
)


GROUP_COLUMNS = [
    "dataset",
    "n",
    "m",
    "seed",
    "opinion_setting",
    "alpha_setting",
    "budget",
    "method",
]

METRIC_COLUMNS = [
    "objective",
    "normalized_objective",
    "runtime_sec",
    "selected_set_size",
    "equilibrium_iterations",
    "linear_solver_iterations",
    "feasibility_violation",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config, base_dir = load_config(args.config)
    output_root = resolve_path(
        args.results_dir
        or config.get(
            "results_dir", "../../results/additional/multi_seed_performance"
        ),
        base_dir,
    )
    seeds = experiment_seeds(config)
    selected_datasets = set(args.dataset or [])
    backend = get_backend_name()
    if bool(config.get("require_cpp_backend", False)) and backend != "cpp":
        raise RuntimeError(
            "The multi-seed performance experiment requires the C++ backend, "
            "but the active backend is Python. Install the project before "
            "running it."
        )
    print(f"[multi-seed-performance] backend={backend}", flush=True)

    entries = config.get("datasets", [])
    if not isinstance(entries, list):
        raise ValueError("Config 'datasets' must be a list.")
    datasets = dataset_configs(config, base_dir)
    matched = False
    for entry, dataset in zip(entries, datasets):
        if selected_datasets and dataset.name not in selected_datasets:
            continue
        matched = True
        _validate_dataset_folder_name(dataset.name)
        dataset_dir = output_root / dataset.name

        if not args.plot_only:
            for seed in seeds:
                seed_path = (
                    dataset_dir
                    / "seeds"
                    / f"seed_{seed}"
                    / "deactivation.csv"
                )
                if (
                    bool(config.get("resume", True))
                    and seed_path.exists()
                    and seed_path.stat().st_size > 0
                    and not args.rerun_existing
                ):
                    print(
                        "[multi-seed-performance] "
                        f"dataset={dataset.name} seed={seed} status=skip",
                        flush=True,
                    )
                else:
                    print(
                        "[multi-seed-performance] "
                        f"dataset={dataset.name} seed={seed} status=run",
                        flush=True,
                    )
                    seed_config: dict[str, Any] = dict(config)
                    seed_config["datasets"] = [entry]
                    seed_config["seed"] = int(seed)
                    run_deactivation.run_config(
                        seed_config,
                        base_dir,
                        seed_path,
                    )
                refresh_outputs(
                    dataset_dir,
                    dataset_name=dataset.name,
                    seeds=seeds,
                    generate_plots=False,
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


def refresh_outputs(
    dataset_dir: Path,
    *,
    dataset_name: str,
    seeds: Iterable[int],
    generate_plots: bool,
) -> list[Path]:
    """Rebuild combined data, summaries, and optionally figures."""

    configured_seeds = [int(seed) for seed in seeds]
    combined = collect_seed_results(
        dataset_dir,
        dataset_name=dataset_name,
        seeds=configured_seeds,
    )
    if combined.empty:
        raise FileNotFoundError(
            f"No completed seed results found under {dataset_dir / 'seeds'}"
        )

    combined_path = dataset_dir / "multi_seed_deactivation.csv"
    seed_level_path = dataset_dir / "multi_seed_seed_level.csv"
    summary_path = dataset_dir / "multi_seed_summary.csv"
    overall_path = dataset_dir / "multi_seed_overall_summary.csv"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)

    seed_level = aggregate_within_seed(combined)
    seed_level.to_csv(seed_level_path, index=False)
    summary = aggregate_across_seeds(seed_level)
    summary.to_csv(summary_path, index=False)
    overall = aggregate_over_settings_and_seeds(seed_level)
    overall.to_csv(overall_path, index=False)

    outputs = [combined_path, seed_level_path, summary_path, overall_path]
    if generate_plots:
        completed_seed_count = int(combined["seed"].nunique())
        outputs.extend(
            plot_multi_seed_performance(
                summary_path,
                dataset_dir / "figures",
                expected_num_seeds=completed_seed_count,
            )
        )
    return outputs


def collect_seed_results(
    dataset_dir: Path,
    *,
    dataset_name: str,
    seeds: Iterable[int],
) -> pd.DataFrame:
    """Read completed per-seed CSV files and attach their instance seeds."""

    frames: list[pd.DataFrame] = []
    required = {
        "dataset",
        "n",
        "m",
        "opinion_setting",
        "alpha_setting",
        "budget",
        "method",
        "repetition",
        "objective",
        "normalized_objective",
        "runtime_sec",
        "timed_out",
        "selected_set_size",
    }
    for seed in seeds:
        path = dataset_dir / "seeds" / f"seed_{seed}" / "deactivation.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Seed result {path} is missing columns: {sorted(missing)}"
            )
        observed = set(frame["dataset"].astype(str).unique())
        if observed != {str(dataset_name)}:
            raise ValueError(
                f"Seed result {path} contains datasets {sorted(observed)}, "
                f"expected only {dataset_name!r}."
            )
        frame = frame.drop(columns=["seed"], errors="ignore")
        frame.insert(3, "seed", int(seed))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.sort_values(
        [
            "dataset",
            "seed",
            "opinion_setting",
            "alpha_setting",
            "method",
            "repetition",
            "budget",
        ],
        kind="stable",
    ).reset_index(drop=True)


def aggregate_within_seed(frame: pd.DataFrame) -> pd.DataFrame:
    """Average stochastic repetitions before comparing different seeds."""

    working = frame.copy()
    for column in METRIC_COLUMNS:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working["timed_out"] = _boolean_series(working["timed_out"])

    rows: list[dict[str, Any]] = []
    for key, group in working.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        row = dict(zip(GROUP_COLUMNS, key))
        row["num_repetitions"] = int(len(group))
        row["timed_out"] = bool(group["timed_out"].any())
        row["timed_out_repetition_rate"] = float(group["timed_out"].mean())
        for column in METRIC_COLUMNS:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[column] = float(values.mean()) if not values.empty else np.nan
            row[f"within_seed_std_{column}"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [
            "dataset",
            "seed",
            "opinion_setting",
            "alpha_setting",
            "method",
            "budget",
        ],
        kind="stable",
    ).reset_index(drop=True)


def aggregate_across_seeds(seed_level: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and sample standard deviation across instance seeds."""

    group_columns = [
        "dataset",
        "n",
        "m",
        "opinion_setting",
        "alpha_setting",
        "budget",
        "method",
    ]
    return _summarize_seed_rows(seed_level, group_columns)


def aggregate_over_settings_and_seeds(
    seed_level: pd.DataFrame,
) -> pd.DataFrame:
    """Average four settings within seed, then summarize across seeds."""

    per_seed_columns = ["dataset", "n", "m", "seed", "budget", "method"]
    rows: list[dict[str, Any]] = []
    for key, group in seed_level.groupby(
        per_seed_columns, sort=True, dropna=False
    ):
        row = dict(zip(per_seed_columns, key))
        row["num_settings"] = int(
            group[["opinion_setting", "alpha_setting"]]
            .drop_duplicates()
            .shape[0]
        )
        row["timed_out"] = bool(group["timed_out"].any())
        row["num_repetitions"] = float(group["num_repetitions"].mean())
        for column in METRIC_COLUMNS:
            if column in group.columns:
                row[column] = float(group[column].mean())
        rows.append(row)
    averaged = pd.DataFrame(rows)
    summary = _summarize_seed_rows(
        averaged,
        ["dataset", "n", "m", "budget", "method"],
    )
    settings = (
        averaged.groupby(
            ["dataset", "n", "m", "budget", "method"], sort=True
        )["num_settings"]
        .agg(["min", "max"])
        .reset_index()
        .rename(
            columns={
                "min": "min_num_settings_per_seed",
                "max": "max_num_settings_per_seed",
            }
        )
    )
    return summary.merge(
        settings,
        on=["dataset", "n", "m", "budget", "method"],
        how="left",
    )


def _summarize_seed_rows(
    frame: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, key))
        seeds = sorted(int(value) for value in group["seed"].unique())
        row["num_seeds"] = len(seeds)
        row["seeds"] = json.dumps(seeds)
        row["timed_out_seed_rate"] = float(
            _boolean_series(group["timed_out"]).mean()
        )
        row["mean_num_repetitions"] = float(
            pd.to_numeric(group["num_repetitions"], errors="coerce").mean()
        )
        for column in METRIC_COLUMNS:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"mean_{column}"] = (
                float(values.mean()) if not values.empty else np.nan
            )
            row[f"std_{column}"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [*group_columns[:-1], "method"], kind="stable"
    ).reset_index(drop=True)


def experiment_seeds(config: dict[str, Any]) -> list[int]:
    """Return distinct configured instance seeds in their listed order."""

    values = config.get("seeds")
    if values is None:
        raise ValueError("Multi-seed config must define a nonempty 'seeds' list.")
    seeds = list(dict.fromkeys(int(value) for value in values))
    if not seeds:
        raise ValueError("Multi-seed config must define at least one seed.")
    return seeds


def _boolean_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


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
            "and aggregate mean and standard deviation across seeds."
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
        "--results-dir", default=None, help="Optional results root override."
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
