#!/usr/bin/env python3
"""Recompute the manuscript AIS-score diagnostics from the raw result CSVs.

It reconstructs the summary tables without rerunning the expensive
equilibrium evaluations.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd


RAW_FILENAME = "ais_score_diagnostic.csv"
OUTPUT_FILENAME = "paper_metrics_by_dataset_k.csv"
REPORT_FILENAME = "paper_metrics.md"

REQUIRED_COLUMNS = {
    "dataset",
    "opinion_setting",
    "alpha_setting",
    "k",
    "iteration",
    "current_objective",
    "exact_best_gain",
    "hit_at_k",
    "full_candidate_spearman",
}


def main() -> int:
    args = _parse_args()
    root = Path(args.results_dir).resolve()
    frames = [
        _read_raw_csv(path)
        for path in sorted(root.glob(f"*/{RAW_FILENAME}"))
    ]
    if not frames:
        raise FileNotFoundError(
            f"No dataset-level {RAW_FILENAME} files were found under {root}."
        )

    raw = pd.concat(frames, ignore_index=True)
    metrics = summarize(raw)
    csv_path = root / OUTPUT_FILENAME
    report_path = root / REPORT_FILENAME
    metrics.to_csv(csv_path, index=False)
    report_path.write_text(render_report(metrics), encoding="utf-8")
    print(csv_path)
    print(report_path)
    return 0


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the exact dataset-by-k quantities used in the manuscript."""

    rows: list[dict[str, int | float | str]] = []
    for (dataset, k), group in frame.groupby(["dataset", "k"], sort=True):
        setting_sizes = group.groupby(
            ["opinion_setting", "alpha_setting"], sort=True
        ).size()
        missed = group.loc[~group["hit_at_k"]]
        positive_missed = missed.loc[missed["exact_best_gain"] > 0.0].copy()
        positive_missed["exact_best_gain_pct_current_objective"] = (
            100.0
            * positive_missed["exact_best_gain"]
            / positive_missed["current_objective"]
        )
        miss_percentages = positive_missed[
            "exact_best_gain_pct_current_objective"
        ].to_numpy(dtype=float)

        rows.append(
            {
                "dataset": str(dataset),
                "k": int(k),
                "num_parameter_settings": int(len(setting_sizes)),
                "num_iterations": int(len(group)),
                "min_iterations_per_setting": int(setting_sizes.min()),
                "max_iterations_per_setting": int(setting_sizes.max()),
                "num_defined_spearman": int(
                    group["full_candidate_spearman"].notna().sum()
                ),
                "mean_full_candidate_spearman": float(
                    group["full_candidate_spearman"].mean()
                ),
                "hit_rate_at_k": float(group["hit_at_k"].mean()),
                "hit_rate_at_k_percent": float(
                    100.0 * group["hit_at_k"].mean()
                ),
                "num_misses": int(len(missed)),
                "num_positive_exact_gain_misses": int(len(positive_missed)),
                "mean_positive_miss_exact_gain_pct_current_objective": (
                    _mean_or_nan(miss_percentages)
                ),
                "max_positive_miss_exact_gain_pct_current_objective": (
                    _max_or_nan(miss_percentages)
                ),
            }
        )
    return pd.DataFrame(rows)


def render_report(metrics: pd.DataFrame) -> str:
    """Render a compact, human-readable record of manuscript-ready values."""

    lines = [
        "# AIS-score diagnostic: manuscript metrics",
        "",
        "All values below are recomputed from each dataset's "
        f"`{RAW_FILENAME}` by `summarize_paper_metrics.py`.",
        "Spearman and Hit@$k$ are averaged over all iterations and the four "
        "parameter settings for each independent $k$ trajectory.",
        "The last two columns condition on iterations where the exact-best "
        "candidate was not in the top-$k$ shortlist and its exact marginal "
        "improvement was positive. The improvement percentage is "
        "$100 \\times$ `exact_best_gain / current_objective`.",
        "",
        "| Dataset | $k$ | Iterations | Mean Spearman | Hit@$k$ | Positive-gain misses | Mean missed gain / current objective | Maximum |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {int(row.k)} | {int(row.num_iterations)} | "
            f"{row.mean_full_candidate_spearman:.6f} | "
            f"{_format_half_up(row.hit_rate_at_k_percent, 1)}% | "
            f"{int(row.num_positive_exact_gain_misses)} | "
            f"{row.mean_positive_miss_exact_gain_pct_current_objective:.6f}% | "
            f"{row.max_positive_miss_exact_gain_pct_current_objective:.6f}% |"
        )

    lines.extend(
        [
            "",
            "## Values quoted in the manuscript",
            "",
        ]
    )
    for dataset, group in metrics.groupby("dataset", sort=True):
        ordered = group.sort_values("k")
        spearman = ", ".join(
            _format_half_up(value, 3)
            for value in ordered["mean_full_candidate_spearman"]
        )
        hits = ", ".join(
            f"{_format_half_up(value, 1)}%"
            for value in ordered["hit_rate_at_k_percent"]
        )
        lines.append(
            f"- **{dataset}:** mean Spearman for $k=1,10,100$: {spearman}; "
            f"Hit@$k$: {hits}."
        )

    k100 = metrics.loc[metrics["k"] == 100].sort_values("dataset")
    if not k100.empty:
        lines.extend(["", "For positive-gain Hit@100 misses:", ""])
        for row in k100.itertuples(index=False):
            lines.append(
                f"- **{row.dataset}:** mean "
                f"{_format_half_up(row.mean_positive_miss_exact_gain_pct_current_objective, 4)}%, "
                f"maximum "
                f"{_format_half_up(row.max_positive_miss_exact_gain_pct_current_objective, 4)}%."
            )
    return "\n".join(lines) + "\n"


def _read_raw_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} is empty.")

    frame = frame.copy()
    frame["hit_at_k"] = frame["hit_at_k"].map(_parse_bool)
    numeric_columns = [
        "k",
        "iteration",
        "current_objective",
        "exact_best_gain",
        "full_candidate_spearman",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame["current_objective"] <= 0.0).any():
        raise ValueError(
            f"{path} contains a nonpositive current_objective, so the "
            "relative missed-gain statistic is undefined."
        )
    return frame


def _parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _max_or_nan(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else float("nan")


def _format_half_up(value: float, decimal_places: int) -> str:
    """Use conventional half-up rounding for manuscript-facing values."""

    quantum = Decimal(1).scaleb(-decimal_places)
    return format(
        Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP),
        f".{decimal_places}f",
    )


def _parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Recompute manuscript metrics from raw AIS diagnostic CSVs."
    )
    parser.add_argument(
        "--results-dir",
        default=repository_root / "results/additional/ais_score_diagnostic",
        type=Path,
        help="AIS diagnostic results root.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
