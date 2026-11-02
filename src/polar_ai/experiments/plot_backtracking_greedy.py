"""Publication-style plots for the Backtracking-Greedy experiment."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "polar_ai_matplotlib")
)

from polar_ai.experiments import plot_results


def write_backtracking_summary(raw_path: Path, output_path: Path) -> Path:
    """Aggregate paired Greedy comparisons over parameter settings."""

    frame = _read_results(raw_path)
    rows: list[dict[str, float | int | str]] = []
    for budget, group in frame.groupby("budget", sort=True):
        beats = _boolean_series(group["backtracking_beats_greedy"])
        equal = _boolean_series(group["objectives_equal"])
        rows.append(
            {
                "dataset": " / ".join(
                    sorted(str(value) for value in group["dataset"].unique())
                ),
                "budget": int(budget),
                "num_instances": int(len(group)),
                "mean_greedy_normalized_objective": float(
                    group["greedy_normalized_objective"].mean()
                ),
                "std_greedy_normalized_objective": float(
                    group["greedy_normalized_objective"].std(ddof=1)
                ),
                "mean_backtracking_normalized_objective": float(
                    group["backtracking_normalized_objective"].mean()
                ),
                "std_backtracking_normalized_objective": float(
                    group["backtracking_normalized_objective"].std(ddof=1)
                ),
                "mean_normalized_gain_over_greedy": float(
                    group["normalized_gain_over_greedy"].mean()
                ),
                "max_normalized_gain_over_greedy": float(
                    group["normalized_gain_over_greedy"].max()
                ),
                "backtracking_win_rate": float(beats.mean()),
                "objective_tie_rate": float(equal.mean()),
                "backtracking_loss_rate": float((~beats & ~equal).mean()),
                "selected_set_match_rate": float(
                    _boolean_series(group["sets_equal"]).mean()
                ),
                "mean_greedy_set_size": float(group["greedy_set_size"].mean()),
                "mean_backtracking_set_size": float(
                    group["backtracking_set_size"].mean()
                ),
                "mean_accepted_deletions": float(
                    group["accepted_deletions"].mean()
                ),
                "mean_accepted_replacements": float(
                    group["accepted_replacements"].mean()
                ),
                "addition_only_switch_rate": float(
                    _boolean_series(
                        group["switched_to_addition_only"]
                    ).mean()
                ),
                "mean_greedy_runtime_sec": float(
                    group["greedy_runtime_sec"].mean()
                ),
                "mean_backtracking_runtime_sec": float(
                    group["backtracking_runtime_sec"].mean()
                ),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def plot_backtracking_greedy(
    raw_path: Path, figures_dir: Path
) -> list[Path]:
    """Plot objective quality and gain over ordinary Greedy."""

    frame = _read_results(raw_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_results._set_publication_style()
    return [
        _plot_objective(frame, figures_dir / "backtracking_objective.pdf"),
        _plot_gain(frame, figures_dir / "backtracking_gain_over_greedy.pdf"),
    ]


def _plot_objective(frame: pd.DataFrame, path: Path) -> Path:
    fig, ax = plot_results.plt.subplots(figsize=(3.8, 2.7))
    specs = [
        (
            "greedy_normalized_objective",
            "Greedy",
            {"color": "#000000", "linestyle": "--", "marker": "*"},
        ),
        (
            "backtracking_normalized_objective",
            "Backtracking-Greedy",
            {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
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


def _plot_gain(frame: pd.DataFrame, path: Path) -> Path:
    fig, ax = plot_results.plt.subplots(figsize=(3.8, 2.7))
    summary = _mean_std(frame, "normalized_gain_over_greedy")
    ax.plot(
        summary["budget"],
        summary["mean"],
        color="#1f77b4",
        linestyle="-",
        marker="o",
        markersize=4.0,
        linewidth=1.6,
    )
    _fill_std(ax, summary)
    ax.axhline(0.0, color="#000000", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Budget")
    ax.set_ylabel("Additional reduction over Greedy")
    ax.set_xticks(sorted(int(value) for value in frame["budget"].unique()))
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


def _read_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Backtracking-Greedy CSV: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Backtracking-Greedy CSV is empty: {path}")
    required = {
        "dataset",
        "budget",
        "greedy_normalized_objective",
        "backtracking_normalized_objective",
        "normalized_gain_over_greedy",
        "backtracking_beats_greedy",
        "objectives_equal",
        "sets_equal",
        "greedy_set_size",
        "backtracking_set_size",
        "accepted_deletions",
        "accepted_replacements",
        "switched_to_addition_only",
        "greedy_runtime_sec",
        "backtracking_runtime_sec",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Result CSV is missing columns: {sorted(missing)}")
    booleans = {
        "backtracking_beats_greedy",
        "objectives_equal",
        "sets_equal",
        "switched_to_addition_only",
    }
    for column in required - {"dataset"} - booleans:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(
        frame[
            [
                "budget",
                "greedy_normalized_objective",
                "backtracking_normalized_objective",
                "normalized_gain_over_greedy",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Result CSV contains non-finite required values.")
    return frame


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


__all__ = ["plot_backtracking_greedy", "write_backtracking_summary"]
