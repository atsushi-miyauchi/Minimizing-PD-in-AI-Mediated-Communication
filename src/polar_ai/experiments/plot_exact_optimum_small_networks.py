"""Publication-style plots for the small-network exact-optimum experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from polar_ai.experiments import plot_results


def write_exact_optimum_summary(
    comparison_path: Path, output_path: Path
) -> Path:
    """Write budget-level aggregate statistics for rebuttal reporting."""

    frame = _read_comparison(comparison_path)
    rows: list[dict[str, float | int | str]] = []
    for budget, group in frame.groupby("budget", sort=True):
        fraction = group["greedy_fraction_of_optimal_improvement"].dropna()
        relative_gap = group["greedy_relative_optimality_gap"].dropna()
        rows.append(
            {
                "dataset": " / ".join(
                    sorted(str(value) for value in group["dataset"].unique())
                ),
                "budget": int(budget),
                "num_instances": int(len(group)),
                "mean_fraction_of_optimal_improvement": float(fraction.mean()),
                "min_fraction_of_optimal_improvement": float(fraction.min()),
                "mean_relative_optimality_gap": float(relative_gap.mean()),
                "max_relative_optimality_gap": float(relative_gap.max()),
                "objective_match_rate": float(
                    np.isclose(
                        group["greedy_objective"],
                        group["optimal_objective"],
                        rtol=1e-9,
                        atol=1e-12,
                    ).mean()
                ),
                "selected_set_match_rate": float(
                    group["sets_equal"].astype(str).str.lower().isin(
                        {"true", "1", "yes"}
                    ).mean()
                ),
                "mean_optimal_set_size": float(
                    group["optimal_set_size"].mean()
                ),
                "mean_greedy_set_size": float(
                    group["greedy_set_size"].mean()
                ),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def plot_exact_optimum_small_networks(
    comparison_path: Path, figures_dir: Path
) -> list[Path]:
    """Generate objective and optimal-improvement comparison plots."""

    frame = _read_comparison(comparison_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_results._set_publication_style()
    return [
        _plot_normalized_objective(
            frame, figures_dir / "exact_optimum_objective.pdf"
        ),
        _plot_fraction_of_improvement(
            frame, figures_dir / "greedy_fraction_of_optimal_improvement.pdf"
        ),
    ]


def _plot_normalized_objective(frame: pd.DataFrame, path: Path) -> Path:
    fig, ax = plot_results.plt.subplots(figsize=(3.8, 2.7))
    specs = [
        (
            "optimal_normalized_objective",
            "Optimal",
            {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
        ),
        (
            "greedy_normalized_objective",
            "Greedy",
            {"color": "#000000", "linestyle": "--", "marker": "*"},
        ),
    ]
    for column, label, style in specs:
        summary = _mean_std(frame, column)
        ax.plot(
            summary["budget"],
            summary["mean"],
            label=label,
            markersize=4.0,
            linewidth=1.6,
            **style,
        )
        _fill_std(ax, summary)
    ax.set_xlabel("Budget")
    ax.set_ylabel(
        r"$\mathrm{PD}(z^*(T)) / \mathrm{PD}(z^*(\emptyset))$"
    )
    ax.set_xticks(sorted(int(value) for value in frame["budget"].unique()))
    ax.legend(frameon=False, loc="best")
    plot_results._set_dataset_title(ax, frame)
    return plot_results._save(fig, path)


def _plot_fraction_of_improvement(frame: pd.DataFrame, path: Path) -> Path:
    fig, ax = plot_results.plt.subplots(figsize=(3.8, 2.7))
    summary = _mean_std(frame, "greedy_fraction_of_optimal_improvement")
    ax.plot(
        summary["budget"],
        summary["mean"],
        color="#000000",
        linestyle="-",
        marker="*",
        markersize=4.0,
        linewidth=1.6,
        label="Greedy",
    )
    _fill_std(ax, summary)
    ax.axhline(1.0, color="#1f77b4", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Budget")
    ax.set_ylabel("Fraction of optimal improvement")
    ax.set_xticks(sorted(int(value) for value in frame["budget"].unique()))
    lower = min(0.0, float(frame["greedy_fraction_of_optimal_improvement"].min()))
    ax.set_ylim(lower, 1.05)
    ax.legend(frameon=False, loc="best")
    plot_results._set_dataset_title(ax, frame)
    return plot_results._save(fig, path)


def _mean_std(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        frame.groupby("budget", as_index=False)[column]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("budget")
    )


def _fill_std(ax: object, summary: pd.DataFrame) -> None:
    if "std" not in summary or summary["std"].isna().all():
        return
    line = ax.lines[-1]
    x = summary["budget"].to_numpy(dtype=float)
    mean = summary["mean"].to_numpy(dtype=float)
    std = summary["std"].fillna(0.0).to_numpy(dtype=float)
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=line.get_color(),
        alpha=0.12,
        linewidth=0,
    )


def _read_comparison(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison CSV: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Comparison CSV is empty: {path}")
    required = {
        "dataset",
        "budget",
        "optimal_objective",
        "greedy_objective",
        "optimal_normalized_objective",
        "greedy_normalized_objective",
        "greedy_fraction_of_optimal_improvement",
        "greedy_relative_optimality_gap",
        "optimal_set_size",
        "greedy_set_size",
        "sets_equal",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Comparison CSV is missing columns: {sorted(missing)}"
        )
    numeric_columns = required - {"dataset", "sets_equal"}
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(
        frame[
            [
                "budget",
                "optimal_normalized_objective",
                "greedy_normalized_objective",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Comparison CSV contains non-finite required values.")
    return frame


__all__ = [
    "plot_exact_optimum_small_networks",
    "write_exact_optimum_summary",
]
