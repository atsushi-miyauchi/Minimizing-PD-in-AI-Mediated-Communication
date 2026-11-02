"""Publication-style plots for the selected-set-size audit."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from polar_ai.experiments import plot_results
from polar_ai.experiments.utils import deactivation_marker_budgets


def plot_selected_set_size_audit(
    by_budget_path: Path,
    figures_dir: Path,
    *,
    methods: Iterable[str] = (),
) -> list[Path]:
    """Plot selected-budget fraction and full-budget frequency."""

    frame = _read_summary(by_budget_path)
    requested = [str(value) for value in methods]
    if requested:
        frame = frame[frame["method"].astype(str).isin(requested)].copy()
        if frame.empty:
            raise ValueError("No requested plotting methods occur in the summary.")
    order = [method for method in requested if method in set(frame["method"])]
    if not order:
        order = sorted(str(value) for value in frame["method"].unique())

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_results._set_publication_style()
    return [
        _plot_metric(
            frame,
            figures_dir / "selected_set_fraction.pdf",
            order=order,
            value="mean_selected_fraction",
            spread="std_selected_fraction",
            ylabel=r"Selected fraction $|T_{\mathrm{best}}|/b$",
        ),
        _plot_metric(
            frame,
            figures_dir / "full_budget_rate.pdf",
            order=order,
            value="full_budget_rate",
            spread=None,
            ylabel=r"Fraction with $|T_{\mathrm{best}}|=b$",
        ),
    ]


def _plot_metric(
    frame: pd.DataFrame,
    path: Path,
    *,
    order: list[str],
    value: str,
    spread: str | None,
    ylabel: str,
) -> Path:
    fig, ax = plot_results.plt.subplots(figsize=(3.8, 3.05))
    maximum_budget = int(frame["budget"].max())
    marker_budgets = set(deactivation_marker_budgets(maximum_budget))

    for method in order:
        group = frame[frame["method"].astype(str) == method].sort_values(
            "budget"
        )
        if group.empty:
            continue
        markevery = [
            index
            for index, budget in enumerate(group["budget"])
            if int(budget) in marker_budgets
        ]
        ax.plot(
            group["budget"],
            group[value],
            label=plot_results._legend_label(method),
            markersize=3.0,
            linewidth=1.6,
            markevery=markevery,
            **plot_results._style_for(method, order),
        )
        if spread is not None:
            mean = group[value].to_numpy(dtype=float)
            std = group[spread].fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(
                group["budget"].to_numpy(dtype=float),
                np.maximum(0.0, mean - std),
                np.minimum(1.0, mean + std),
                color=ax.lines[-1].get_color(),
                alpha=0.10,
                linewidth=0,
            )

    ax.set_xlabel("Budget")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=2,
        handlelength=1.8,
        columnspacing=1.0,
    )
    plot_results._set_dataset_title(ax, frame)
    return plot_results._save(fig, path)


def _read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing selected-size summary CSV: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Selected-size summary CSV is empty: {path}")
    required = {
        "dataset",
        "method",
        "budget",
        "mean_selected_fraction",
        "std_selected_fraction",
        "full_budget_rate",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Summary CSV is missing columns: {sorted(missing)}")
    for column in required - {"dataset", "method"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(
        frame[
            ["budget", "mean_selected_fraction", "full_budget_rate"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Summary CSV contains non-finite required values.")
    return frame


__all__ = ["plot_selected_set_size_audit"]
