#!/usr/bin/env python3
"""Derive manuscript robustness metrics from completed experiment outputs.

The script processes the five-seed and Fiedler-correlated experiments so that
summary values can be regenerated without rerunning any optimization
algorithm.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASETS = ["wiki-Vote", "hamster", "advogato", "epinions"]
K_VALUES = [1, 10, 100]
INPUT_FILENAME = "multi_seed_seed_level.csv"

PAIR_KEYS = ["seed", "opinion_setting", "alpha_setting", "budget"]
REQUIRED_COLUMNS = set(
    PAIR_KEYS
    + ["method", "normalized_objective", "timed_out", "num_repetitions"]
)


@dataclass(frozen=True)
class ExperimentSpec:
    directory: str
    label: str
    expected_seeds: int


EXPERIMENTS = {
    "multi-seed": ExperimentSpec(
        directory="multi_seed_performance",
        label="Five independent parameter draws",
        expected_seeds=5,
    ),
    "fiedler": ExperimentSpec(
        directory="fiedler_correlated_performance",
        label="Network-correlated innate opinions",
        expected_seeds=1,
    ),
}


def main() -> int:
    args = _parse_args()
    additional_root = Path(args.additional_results_dir).resolve()
    selected = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]

    for experiment_name in selected:
        spec = EXPERIMENTS[experiment_name]
        output_root = additional_root / spec.directory
        seed_metrics, network_metrics, overall_metrics = summarize_experiment(
            output_root,
            spec=spec,
        )
        outputs = {
            "paper_metrics_by_seed_k.csv": seed_metrics,
            "paper_metrics_by_network_k.csv": network_metrics,
            "paper_metrics_overall.csv": overall_metrics,
        }
        for filename, frame in outputs.items():
            path = output_root / filename
            frame.to_csv(path, index=False)
            print(path)

        report_path = output_root / "paper_metrics.md"
        report_path.write_text(
            render_report(
                experiment_name,
                spec,
                network_metrics,
                overall_metrics,
            ),
            encoding="utf-8",
        )
        print(report_path)
    return 0


def summarize_experiment(
    output_root: Path,
    *,
    spec: ExperimentSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return seed-, network-, and experiment-level manuscript metrics."""

    seed_frames: list[pd.DataFrame] = []
    for dataset in DATASETS:
        path = output_root / dataset / INPUT_FILENAME
        frame = _read_input(path, expected_dataset=dataset)
        seed_frames.append(_seed_metrics_for_dataset(frame, dataset=dataset))

    seed_metrics = pd.concat(seed_frames, ignore_index=True)
    observed_seed_counts = seed_metrics.groupby("dataset")["seed"].nunique()
    if not (observed_seed_counts == spec.expected_seeds).all():
        raise ValueError(
            f"{spec.directory} does not contain {spec.expected_seeds} seeds "
            f"for every dataset: {observed_seed_counts.to_dict()}"
        )

    network_rows: list[dict[str, Any]] = []
    for (dataset, k), group in seed_metrics.groupby(
        ["dataset", "k"], sort=False
    ):
        ratios = group["mean_reduction_ratio_percent"].to_numpy(float)
        network_rows.append(
            {
                "experiment": spec.directory,
                "dataset": str(dataset),
                "k": int(k),
                "num_seeds": int(group["seed"].nunique()),
                "seeds": ";".join(str(int(seed)) for seed in sorted(group["seed"])),
                "num_matched_points": int(group["num_matched_points"].sum()),
                "min_matched_budget": int(group["min_matched_budget"].min()),
                "max_matched_budget": int(group["max_matched_budget"].max()),
                "min_parameter_settings_per_seed": int(
                    group["num_parameter_settings"].min()
                ),
                "max_parameter_settings_per_seed": int(
                    group["num_parameter_settings"].max()
                ),
                "mean_reduction_ratio_percent": float(np.mean(ratios)),
                "std_reduction_ratio_percentage_points": (
                    float(np.std(ratios, ddof=1))
                    if len(ratios) > 1
                    else float("nan")
                ),
                "min_seed_reduction_ratio_percent": float(np.min(ratios)),
                "max_seed_reduction_ratio_percent": float(np.max(ratios)),
            }
        )
    network_metrics = pd.DataFrame(network_rows)
    network_metrics["dataset"] = pd.Categorical(
        network_metrics["dataset"], categories=DATASETS, ordered=True
    )
    network_metrics = network_metrics.sort_values(["dataset", "k"]).reset_index(
        drop=True
    )
    network_metrics["dataset"] = network_metrics["dataset"].astype(str)

    overall_rows: list[dict[str, Any]] = []
    for k, group in network_metrics.groupby("k", sort=True):
        minimum_index = group["mean_reduction_ratio_percent"].idxmin()
        maximum_index = group["mean_reduction_ratio_percent"].idxmax()
        standard_deviations = group[
            "std_reduction_ratio_percentage_points"
        ].dropna()
        overall_rows.append(
            {
                "experiment": spec.directory,
                "k": int(k),
                "num_networks": int(group["dataset"].nunique()),
                "min_network_mean_reduction_ratio_percent": float(
                    group.loc[minimum_index, "mean_reduction_ratio_percent"]
                ),
                "min_network": str(group.loc[minimum_index, "dataset"]),
                "max_network_mean_reduction_ratio_percent": float(
                    group.loc[maximum_index, "mean_reduction_ratio_percent"]
                ),
                "max_network": str(group.loc[maximum_index, "dataset"]),
                "max_network_std_percentage_points": (
                    float(standard_deviations.max())
                    if not standard_deviations.empty
                    else float("nan")
                ),
            }
        )
    return seed_metrics, network_metrics, pd.DataFrame(overall_rows)


def _seed_metrics_for_dataset(
    frame: pd.DataFrame,
    *,
    dataset: str,
) -> pd.DataFrame:
    """Pair AIS and Greedy rows, then average ratios within each seed."""

    valid = frame.loc[~frame["timed_out"]].copy()
    greedy = valid.loc[valid["method"] == "Greedy", PAIR_KEYS + ["normalized_objective"]]
    _require_unique_rows(greedy, dataset=dataset, method="Greedy")
    greedy = greedy.rename(columns={"normalized_objective": "greedy_objective"})

    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        method = f"AIS-Greedy (k={k})"
        ais = valid.loc[
            valid["method"] == method,
            PAIR_KEYS + ["normalized_objective"],
        ]
        _require_unique_rows(ais, dataset=dataset, method=method)
        ais = ais.rename(columns={"normalized_objective": "ais_objective"})
        paired = ais.merge(greedy, on=PAIR_KEYS, how="inner", validate="one_to_one")
        paired["greedy_reduction"] = 1.0 - paired["greedy_objective"]
        paired["ais_reduction"] = 1.0 - paired["ais_objective"]
        finite = np.isfinite(paired["greedy_reduction"]) & np.isfinite(
            paired["ais_reduction"]
        )
        paired = paired.loc[finite & (paired["greedy_reduction"] > 0.0)].copy()
        if paired.empty:
            raise ValueError(
                f"No valid matched {method}/Greedy rows remain for {dataset}."
            )
        paired["reduction_ratio_percent"] = (
            100.0 * paired["ais_reduction"] / paired["greedy_reduction"]
        )

        for seed, group in paired.groupby("seed", sort=True):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "k": int(k),
                    "num_matched_points": int(len(group)),
                    "num_parameter_settings": int(
                        group[["opinion_setting", "alpha_setting"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "min_matched_budget": int(group["budget"].min()),
                    "max_matched_budget": int(group["budget"].max()),
                    "mean_reduction_ratio_percent": float(
                        group["reduction_ratio_percent"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def render_report(
    experiment_name: str,
    spec: ExperimentSpec,
    network_metrics: pd.DataFrame,
    overall_metrics: pd.DataFrame,
) -> str:
    """Render exact tables plus the rounded manuscript sentence."""

    lines = [
        f"# {spec.label}: manuscript metrics",
        "",
        "The script excludes rows marked `timed_out`, pairs AIS-Greedy and "
        "Greedy by seed, opinion setting, alpha setting, and budget, and "
        "calculates",
        "",
        "```text",
        "100 * (1 - AIS normalized objective) / (1 - Greedy normalized objective).",
        "```",
        "",
        "Ratios are averaged over all matched budgets and four parameter "
        "settings within each seed. Network means and sample standard "
        "deviations then give equal weight to each seed.",
        "",
        "| Network | k | Seeds | Matched points | Mean ratio | Seed SD (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in network_metrics.itertuples(index=False):
        standard_deviation = (
            "n.a."
            if pd.isna(row.std_reduction_ratio_percentage_points)
            else f"{row.std_reduction_ratio_percentage_points:.6f}"
        )
        lines.append(
            f"| {row.dataset} | {int(row.k)} | {int(row.num_seeds)} | "
            f"{int(row.num_matched_points)} | "
            f"{row.mean_reduction_ratio_percent:.6f}% | "
            f"{standard_deviation} |"
        )

    ranges = []
    for row in overall_metrics.sort_values("k").itertuples(index=False):
        ranges.append(
            f"{_format_half_up(row.min_network_mean_reduction_ratio_percent, 2)}%"
            f"--{_format_half_up(row.max_network_mean_reduction_ratio_percent, 2)}%"
        )
    lines.extend(["", "## Values quoted in the manuscript", ""])
    if experiment_name == "multi-seed":
        maximum_std = float(
            overall_metrics["max_network_std_percentage_points"].max()
        )
        lines.append(
            "For $k=1,10,100$, respectively, the ranges are "
            f"{ranges[0]}, {ranges[1]}, and {ranges[2]}, with standard "
            "deviations across seeds at most "
            f"{_format_half_up(maximum_std, 2)} percentage points."
        )
    else:
        lines.append(
            "For $k=1,10,100$, respectively, the ranges are "
            f"{ranges[0]}, {ranges[1]}, and {ranges[2]}."
        )
    return "\n".join(lines) + "\n"


def _read_input(path: Path, *, expected_dataset: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Input is empty: {path}")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "dataset" in frame.columns:
        observed = set(frame["dataset"].astype(str))
        if observed != {expected_dataset}:
            raise ValueError(
                f"{path} contains datasets {sorted(observed)}, expected "
                f"{expected_dataset!r}."
            )
    frame = frame.copy()
    frame["timed_out"] = frame["timed_out"].map(_parse_bool)
    for column in ["seed", "budget", "normalized_objective", "num_repetitions"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _require_unique_rows(
    frame: pd.DataFrame,
    *,
    dataset: str,
    method: str,
) -> None:
    if frame.empty:
        raise ValueError(f"Missing {method} rows for {dataset}.")
    duplicates = frame.duplicated(PAIR_KEYS, keep=False)
    if duplicates.any():
        raise ValueError(
            f"Duplicate {method} comparison keys occur for {dataset}."
        )


def _parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def _format_half_up(value: float, decimal_places: int) -> str:
    quantum = Decimal(1).scaleb(-decimal_places)
    return format(
        Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP),
        f".{decimal_places}f",
    )


def _parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Recompute manuscript robustness metrics from saved results."
    )
    parser.add_argument(
        "--additional-results-dir",
        type=Path,
        default=repository_root / "results/additional",
        help="Root containing the additional experiment result directories.",
    )
    parser.add_argument(
        "--experiment",
        choices=["all", *EXPERIMENTS],
        default="all",
        help="Process both experiments or only one of them.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
