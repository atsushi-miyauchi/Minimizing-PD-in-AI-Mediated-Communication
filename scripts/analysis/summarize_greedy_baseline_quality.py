#!/usr/bin/env python3
"""Recompute manuscript metrics for the empirical quality of Greedy.

This script combines the exhaustive-optimum comparisons on soc-karate and
soc-dolphins with the Backtracking-Greedy comparisons on soc-karate,
soc-dolphins, and wiki-Vote.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd


EXACT_SOURCES = {
    "soc-karate": Path(
        "exact_optimum_small_networks/soc-karate/b_max_8/"
        "exact_optimum_comparison.csv"
    ),
    "soc-dolphins": Path(
        "exact_optimum_small_networks/soc-dolphins/b_max_6/"
        "exact_optimum_comparison.csv"
    ),
}
EXACT_BUDGETS = {
    "soc-karate": set(range(1, 9)),
    "soc-dolphins": set(range(1, 7)),
}

LOCAL_SEARCH_SOURCES = {
    dataset: Path(f"backtracking_greedy/{dataset}/backtracking_greedy_results.csv")
    for dataset in ["soc-karate", "soc-dolphins", "wiki-Vote"]
}
LOCAL_SEARCH_BUDGETS = {
    "soc-karate": {10, 20, 30},
    "soc-dolphins": {10, 20, 30, 40, 50, 60},
    "wiki-Vote": {100, 200, 300, 400, 500},
}
EXPECTED_SETTINGS = {
    ("uniform", "homogeneous"),
    ("uniform", "heterogeneous"),
    ("polarized", "homogeneous"),
    ("polarized", "heterogeneous"),
}

EXACT_REQUIRED_COLUMNS = {
    "dataset",
    "seed",
    "opinion_setting",
    "alpha_setting",
    "budget",
    "optimal_objective",
    "greedy_objective",
    "greedy_relative_optimality_gap",
}
LOCAL_SEARCH_REQUIRED_COLUMNS = {
    "dataset",
    "seed",
    "opinion_setting",
    "alpha_setting",
    "budget",
    "baseline_objective",
    "greedy_objective",
    "backtracking_objective",
    "objective_gain_over_greedy",
    "backtracking_beats_greedy",
    "objectives_equal",
    "greedy_runtime_sec",
    "backtracking_runtime_sec",
}

EXACT_OUTPUT = "paper_metrics_exact_by_dataset.csv"
LOCAL_SEARCH_OUTPUT = "paper_metrics_local_search_by_dataset.csv"
OVERALL_OUTPUT = "paper_metrics_overall.csv"
REPORT_OUTPUT = "paper_metrics.md"


def main() -> int:
    args = _parse_args()
    additional_root = Path(args.additional_results_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    exact = _read_exact_results(additional_root)
    local_search = _read_local_search_results(additional_root)
    exact_by_dataset, exact_overall = summarize_exact(exact)
    local_by_dataset, local_overall = summarize_local_search(local_search)
    overall = pd.DataFrame([{**exact_overall, **local_overall}])

    output_root.mkdir(parents=True, exist_ok=True)
    exact_path = output_root / EXACT_OUTPUT
    local_path = output_root / LOCAL_SEARCH_OUTPUT
    overall_path = output_root / OVERALL_OUTPUT
    report_path = output_root / REPORT_OUTPUT
    exact_by_dataset.to_csv(exact_path, index=False)
    local_by_dataset.to_csv(local_path, index=False)
    overall.to_csv(overall_path, index=False)
    report_path.write_text(
        render_report(exact_by_dataset, local_by_dataset, overall.iloc[0]),
        encoding="utf-8",
    )
    print(exact_path)
    print(local_path)
    print(overall_path)
    print(report_path)
    return 0


def summarize_exact(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    """Summarize Greedy's relative objective excess over the exact optimum."""

    rows: list[dict[str, int | float | str]] = []
    for dataset, group in frame.groupby("dataset", sort=True):
        rows.append(_exact_summary_row(str(dataset), group))
    by_dataset = pd.DataFrame(rows)
    overall = _exact_summary_row("all", frame)
    exact_mean = float(overall["mean_relative_objective_excess_percent"])
    exact_worst = float(overall["max_relative_objective_excess_percent"])
    overall.update(
        {
            "manuscript_mean_excess_standard_rounding_percent": (
                _round_half_up(exact_mean, 3)
            ),
            "manuscript_mean_excess_upper_bound_percent": _round_up(
                exact_mean, 3
            ),
            "manuscript_worst_excess_percent": _round_half_up(
                exact_worst, 3
            ),
        }
    )
    return by_dataset, {f"exact_{key}": value for key, value in overall.items()}


def summarize_local_search(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    """Summarize Backtracking-Greedy's gains over ordinary Greedy."""

    rows: list[dict[str, int | float | str]] = []
    for dataset, group in frame.groupby("dataset", sort=True):
        rows.append(_local_search_summary_row(str(dataset), group))
    by_dataset = pd.DataFrame(rows)
    overall = _local_search_summary_row("all", frame)
    overall["manuscript_max_additional_reduction_percent"] = _round_half_up(
        float(overall["max_additional_reduction_over_greedy_reduction_percent"]),
        2,
    )
    return by_dataset, {f"local_search_{key}": value for key, value in overall.items()}


def render_report(
    exact_by_dataset: pd.DataFrame,
    local_by_dataset: pd.DataFrame,
    overall: pd.Series,
) -> str:
    """Render the derivation and manuscript-facing values."""

    runtime_ratios = {
        str(row.dataset): float(
            row.manuscript_total_runtime_ratio_over_greedy
        )
        for row in local_by_dataset.itertuples(index=False)
    }

    lines = [
        "# Quality of the Greedy baseline: manuscript metrics",
        "",
        "All values are reconstructed by `summarize_paper_metrics.py` from "
        "the saved exact-optimum comparison rows and Backtracking-Greedy "
        "result rows.",
        "",
        "## Exhaustive optimum comparison",
        "",
        "For each instance, the relative objective excess is",
        "",
        "$$",
        "100\\,\\frac{F(T_{\\mathrm{Greedy}})-F(T^*)}{F(T^*)}.",
        "$$",
        "",
        "Numerically equal objectives, as determined by the tolerance used "
        "when the comparison CSV was generated, have zero excess.",
        "",
        "| Dataset | Instances | Exactly optimal | Mean excess | Worst excess |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in exact_by_dataset.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {int(row.num_instances)} | "
            f"{int(row.num_exactly_optimal)} | "
            f"{row.mean_relative_objective_excess_percent:.9f}% | "
            f"{row.max_relative_objective_excess_percent:.9f}% |"
        )
    lines.extend(
        [
            "",
            f"Overall, Greedy is exactly optimal in "
            f"**{int(overall['exact_num_exactly_optimal'])}/"
            f"{int(overall['exact_num_instances'])}** instances. Its exact "
            "mean and maximum relative objective excesses are "
            f"**{overall['exact_mean_relative_objective_excess_percent']:.9f}%** "
            "and "
            f"**{overall['exact_max_relative_objective_excess_percent']:.9f}%**, "
            "respectively.",
            "",
            "Conventional half-up rounding to three decimal places gives "
            f"**{overall['exact_manuscript_mean_excess_standard_rounding_percent']:.3f}%** "
            "for the mean and "
            f"**{overall['exact_manuscript_worst_excess_percent']:.3f}%** "
            "for the maximum. The draft value **0.026%** is instead a "
            "conservative upward bound on the exact mean.",
            "",
            "## Local-search comparison",
            "",
            "The additional objective reduction relative to that achieved by "
            "Greedy is",
            "",
            "$$",
            "100\\,\\frac{F(T_{\\mathrm{Greedy}})-F(T_{\\mathrm{BT}})}"
            "{F(\\emptyset)-F(T_{\\mathrm{Greedy}})}.",
            "$$",
            "",
            "| Dataset | Instances | Improved | Maximum additional reduction | Total runtime ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in local_by_dataset.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {int(row.num_instances)} | "
            f"{int(row.num_improved)} | "
            f"{row.max_additional_reduction_over_greedy_reduction_percent:.9f}% | "
            f"{row.total_runtime_ratio_over_greedy:.9f}$\\times$ |"
        )
    lines.extend(
        [
            "",
            "The largest relative increase in objective reduction is "
            f"**{overall['local_search_max_additional_reduction_over_greedy_reduction_percent']:.9f}%**, "
            "which gives the manuscript value "
            f"**{overall['local_search_manuscript_max_additional_reduction_percent']:.2f}%**.",
            "",
            "Aggregating runtime over all tested budgets and the four settings "
            "within each network gives manuscript runtime multipliers of "
            f"**{runtime_ratios['soc-karate']:.1f}$\\times$**, "
            f"**{runtime_ratios['soc-dolphins']:.1f}$\\times$**, and "
            f"**{runtime_ratios['wiki-Vote']:.1f}$\\times$** relative to "
            "Greedy on soc-karate, soc-dolphins, and wiki-Vote, respectively. "
            "These results support the conclusion that the local-search "
            "variant is substantially less scalable than Greedy.",
        ]
    )
    return "\n".join(lines) + "\n"


def _exact_summary_row(
    dataset: str, group: pd.DataFrame
) -> dict[str, int | float | str]:
    excess = group["relative_objective_excess_percent"].to_numpy(dtype=float)
    return {
        "dataset": dataset,
        "num_instances": int(len(group)),
        "num_exactly_optimal": int(group["objective_match"].sum()),
        "objective_match_rate": float(group["objective_match"].mean()),
        "mean_relative_objective_excess_percent": float(np.mean(excess)),
        "max_relative_objective_excess_percent": float(np.max(excess)),
    }


def _local_search_summary_row(
    dataset: str, group: pd.DataFrame
) -> dict[str, int | float | str]:
    gains = group[
        "additional_reduction_over_greedy_reduction_percent"
    ].to_numpy(dtype=float)
    runtime_ratio = float(
        group["backtracking_runtime_sec"].sum()
        / group["greedy_runtime_sec"].sum()
    )
    return {
        "dataset": dataset,
        "num_instances": int(len(group)),
        "num_improved": int(group["backtracking_beats_greedy"].sum()),
        "improvement_rate": float(group["backtracking_beats_greedy"].mean()),
        "mean_additional_reduction_over_greedy_reduction_percent": float(
            np.mean(gains)
        ),
        "max_additional_reduction_over_greedy_reduction_percent": float(
            np.max(gains)
        ),
        "total_greedy_runtime_sec": float(group["greedy_runtime_sec"].sum()),
        "total_local_search_runtime_sec": float(
            group["backtracking_runtime_sec"].sum()
        ),
        "total_runtime_ratio_over_greedy": runtime_ratio,
        "manuscript_total_runtime_ratio_over_greedy": _round_half_up(
            runtime_ratio, 1
        ),
    }


def _read_exact_results(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for expected_dataset, relative_path in EXACT_SOURCES.items():
        path = root / relative_path
        frame = _read_csv(path, EXACT_REQUIRED_COLUMNS)
        _validate_design(
            frame,
            path=path,
            expected_dataset=expected_dataset,
            expected_budgets=EXACT_BUDGETS[expected_dataset],
        )
        numeric = [
            "budget",
            "optimal_objective",
            "greedy_objective",
            "greedy_relative_optimality_gap",
        ]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if (frame["optimal_objective"] <= 0.0).any():
            raise ValueError(f"{path} contains a nonpositive optimum.")
        if (frame["greedy_relative_optimality_gap"] < 0.0).any():
            raise ValueError(f"{path} contains a negative optimality gap.")
        frame["objective_match"] = (
            frame["greedy_relative_optimality_gap"] == 0.0
        )
        raw_excess = 100.0 * (
            frame["greedy_objective"] - frame["optimal_objective"]
        ) / frame["optimal_objective"]
        frame["relative_objective_excess_percent"] = np.where(
            frame["objective_match"], 0.0, raw_excess
        )
        if (frame["relative_objective_excess_percent"] < 0.0).any():
            raise ValueError(f"{path} contains a negative objective excess.")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _read_local_search_results(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for expected_dataset, relative_path in LOCAL_SEARCH_SOURCES.items():
        path = root / relative_path
        frame = _read_csv(path, LOCAL_SEARCH_REQUIRED_COLUMNS)
        _validate_design(
            frame,
            path=path,
            expected_dataset=expected_dataset,
            expected_budgets=LOCAL_SEARCH_BUDGETS[expected_dataset],
        )
        numeric = [
            "budget",
            "baseline_objective",
            "greedy_objective",
            "backtracking_objective",
            "objective_gain_over_greedy",
            "greedy_runtime_sec",
            "backtracking_runtime_sec",
        ]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        for column in ["backtracking_beats_greedy", "objectives_equal"]:
            frame[column] = frame[column].map(_parse_bool)
        expected_gain = frame["greedy_objective"] - frame["backtracking_objective"]
        if not np.allclose(
            frame["objective_gain_over_greedy"],
            expected_gain,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(f"{path} has inconsistent objective gains.")
        greedy_reduction = (
            frame["baseline_objective"] - frame["greedy_objective"]
        )
        if (greedy_reduction <= 0.0).any():
            raise ValueError(
                f"{path} contains a nonpositive Greedy objective reduction."
            )
        if (frame["objective_gain_over_greedy"] < -1e-12).any():
            raise ValueError(f"{path} contains a local-search loss.")
        if (
            (frame["greedy_runtime_sec"] <= 0.0).any()
            or (frame["backtracking_runtime_sec"] <= 0.0).any()
        ):
            raise ValueError(f"{path} contains a nonpositive runtime.")
        if not (
            frame["backtracking_beats_greedy"] | frame["objectives_equal"]
        ).all():
            raise ValueError(
                f"{path} contains a row classified as neither win nor tie."
            )
        frame["additional_reduction_over_greedy_reduction_percent"] = (
            100.0 * frame["objective_gain_over_greedy"] / greedy_reduction
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} is empty.")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame.copy()


def _validate_design(
    frame: pd.DataFrame,
    *,
    path: Path,
    expected_dataset: str,
    expected_budgets: set[int],
) -> None:
    if set(frame["dataset"].astype(str)) != {expected_dataset}:
        raise ValueError(f"Unexpected dataset in {path}.")
    observed_settings = set(
        zip(
            frame["opinion_setting"].astype(str),
            frame["alpha_setting"].astype(str),
        )
    )
    if observed_settings != EXPECTED_SETTINGS:
        raise ValueError(
            f"{path} has unexpected parameter settings: {observed_settings}."
        )
    observed_budgets = set(pd.to_numeric(frame["budget"], errors="raise"))
    if observed_budgets != expected_budgets:
        raise ValueError(
            f"{path} has budgets {sorted(observed_budgets)}; expected "
            f"{sorted(expected_budgets)}."
        )
    keys = ["seed", "opinion_setting", "alpha_setting", "budget"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{path} contains duplicate instance rows.")
    expected_rows = len(EXPECTED_SETTINGS) * len(expected_budgets)
    if len(frame) != expected_rows:
        raise ValueError(
            f"{path} contains {len(frame)} rows; expected {expected_rows}."
        )


def _parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def _round_half_up(value: float, decimal_places: int) -> float:
    quantum = Decimal(1).scaleb(-decimal_places)
    return float(
        Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP)
    )


def _round_up(value: float, decimal_places: int) -> float:
    quantum = Decimal(1).scaleb(-decimal_places)
    return float(
        Decimal(str(float(value))).quantize(quantum, rounding=ROUND_CEILING)
    )


def _parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Recompute manuscript metrics for Greedy baseline quality."
    )
    parser.add_argument(
        "--additional-results-dir",
        type=Path,
        default=repository_root / "results/additional",
        help="Root containing the exact-optimum and backtracking results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "results/additional/greedy_baseline_quality",
        help="Directory for derived CSV and Markdown files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
