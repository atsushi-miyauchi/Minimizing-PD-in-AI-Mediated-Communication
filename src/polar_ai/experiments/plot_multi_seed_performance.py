"""Publication-style plots for the multi-seed performance experiment."""

from __future__ import annotations

import re
from pathlib import Path

from polar_ai.experiments import plot_results

import numpy as np
import pandas as pd

from polar_ai.experiments.utils import deactivation_marker_budgets


def plot_multi_seed_performance(
    summary_path: Path,
    figures_dir: Path,
    *,
    expected_num_seeds: int,
) -> list[Path]:
    """Plot per-setting seed means with one-standard-deviation bands."""

    frame = _read_summary(summary_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_results._set_publication_style()

    outputs: list[Path] = []
    settings = (
        frame[["opinion_setting", "alpha_setting"]]
        .drop_duplicates()
        .sort_values(["opinion_setting", "alpha_setting"])
    )
    for setting in settings.itertuples(index=False):
        opinion_setting = str(setting.opinion_setting)
        alpha_setting = str(setting.alpha_setting)
        subset = frame[
            (frame["opinion_setting"].astype(str) == opinion_setting)
            & (frame["alpha_setting"].astype(str) == alpha_setting)
        ].copy()
        suffix = f"{_slug(opinion_setting)}_{_slug(alpha_setting)}"
        outputs.append(
            _plot_metric(
                subset,
                figures_dir / f"multi_seed_quality_{suffix}.pdf",
                mean_column="mean_normalized_objective",
                std_column="std_normalized_objective",
                ylabel="Normalized objective",
                expected_num_seeds=expected_num_seeds,
                logarithmic=False,
            )
        )
        outputs.append(
            _plot_metric(
                subset,
                figures_dir / f"multi_seed_runtime_{suffix}.pdf",
                mean_column="mean_runtime_sec",
                std_column="std_runtime_sec",
                ylabel="Runtime (s)",
                expected_num_seeds=expected_num_seeds,
                logarithmic=True,
            )
        )

    order = plot_results._extend_method_order(
        plot_results.DEACTIVATION_ORDER,
        frame["method"].astype(str),
    )
    if order:
        outputs.append(
            plot_results.plot_method_legend(
                figures_dir,
                order,
                "multi_seed_legend.pdf",
                "AIS-Greedy",
            )
        )
    return outputs


def _plot_metric(
    frame: pd.DataFrame,
    path: Path,
    *,
    mean_column: str,
    std_column: str,
    ylabel: str,
    expected_num_seeds: int,
    logarithmic: bool,
) -> Path:
    fig, ax = plot_results.plt.subplots(figsize=(3.8, 2.7))
    working = frame[
        (frame["num_seeds"] >= int(expected_num_seeds))
        & np.isclose(frame["timed_out_seed_rate"], 0.0)
    ].copy()
    order = plot_results._extend_method_order(
        plot_results.DEACTIVATION_ORDER,
        working["method"].astype(str),
    )
    b_max = int(working["budget"].max()) if not working.empty else 0
    marker_budgets = set(deactivation_marker_budgets(b_max))

    for method in order:
        group = working[
            working["method"].astype(str) == str(method)
        ].sort_values("budget")
        if group.empty:
            continue
        x = group["budget"].to_numpy(dtype=float)
        mean = group[mean_column].to_numpy(dtype=float)
        std = group[std_column].fillna(0.0).to_numpy(dtype=float)
        markevery = [
            index
            for index, budget in enumerate(group["budget"])
            if int(budget) in marker_budgets
        ]
        style = plot_results._style_for(method, order)
        ax.plot(
            x,
            mean,
            label=plot_results._legend_label(method),
            linewidth=1.8,
            markersize=3.0,
            markevery=markevery,
            alpha=0.65,
            **style,
        )
        lower = mean - std
        if logarithmic:
            positive = mean[mean > 0.0]
            floor = float(np.min(positive)) * 0.1 if positive.size else 1e-12
            lower = np.maximum(lower, floor)
        else:
            lower = np.maximum(lower, 0.0)
        ax.fill_between(
            x,
            lower,
            mean + std,
            color=ax.lines[-1].get_color(),
            alpha=0.12,
            linewidth=0,
        )

    if mean_column == "mean_normalized_objective":
        _annotate_no_ai(ax, frame, expected_num_seeds)
    ax.set_xlabel(r"Budget $b$")
    ax.set_ylabel(ylabel)
    if logarithmic:
        ax.set_yscale("log")
    plot_results._set_dataset_title(ax, frame)
    return plot_results._save(fig, path)


def _annotate_no_ai(
    ax: object, frame: pd.DataFrame, expected_num_seeds: int
) -> None:
    group = frame[
        (frame["method"].astype(str) == "No-AI-Mediation")
        & (frame["num_seeds"] >= int(expected_num_seeds))
    ].sort_values("budget")
    if group.empty:
        return
    row = group.iloc[0]
    mean = float(row["mean_normalized_objective"])
    std = float(row["std_normalized_objective"])
    ax.text(
        0.02,
        0.06,
        f"No-AI-mediation = {mean:.3f} ± {std:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
            "pad": 1.5,
        },
        zorder=8,
    )


def _read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing multi-seed summary CSV: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Multi-seed summary CSV is empty: {path}")
    required = {
        "dataset",
        "opinion_setting",
        "alpha_setting",
        "budget",
        "method",
        "num_seeds",
        "timed_out_seed_rate",
        "mean_normalized_objective",
        "std_normalized_objective",
        "mean_runtime_sec",
        "std_runtime_sec",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Multi-seed summary CSV is missing columns: {sorted(missing)}"
        )
    for column in required - {
        "dataset",
        "opinion_setting",
        "alpha_setting",
        "method",
    }:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


__all__ = ["plot_multi_seed_performance"]
