"""Publication-style plots for the full-candidate AIS score diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from polar_ai.experiments import plot_results
from polar_ai.experiments.utils import deactivation_marker_budgets


def write_ais_score_summary(csv_path: Path, output_path: Path) -> Path:
    """Write setting-level and dataset-level numerical summaries."""

    frame = _read_diagnostic_csv(csv_path)
    setting_columns = [
        "dataset",
        "opinion_setting",
        "alpha_setting",
        "k",
    ]
    setting_summary = _summarize(frame, setting_columns)
    setting_summary.insert(0, "aggregation", "parameter_setting")

    dataset_summary = _summarize(frame, ["dataset", "k"])
    dataset_summary.insert(0, "aggregation", "dataset")
    dataset_summary.insert(2, "opinion_setting", "all")
    dataset_summary.insert(3, "alpha_setting", "all")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([setting_summary, dataset_summary], ignore_index=True).to_csv(
        output_path, index=False
    )
    return output_path


def plot_ais_score_diagnostics(
    csv_path: Path, figures_dir: Path
) -> list[Path]:
    """Generate aggregate and parameter-setting diagnostic figures."""

    frame = _read_diagnostic_csv(csv_path)
    plot_results._set_publication_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = _plot_frame(frame, figures_dir, suffix="")

    settings = (
        frame[["opinion_setting", "alpha_setting"]]
        .drop_duplicates()
        .sort_values(["opinion_setting", "alpha_setting"])
    )
    for setting in settings.itertuples(index=False):
        opinion = str(setting.opinion_setting)
        alpha = str(setting.alpha_setting)
        subset = frame[
            (frame["opinion_setting"].astype(str) == opinion)
            & (frame["alpha_setting"].astype(str) == alpha)
        ].copy()
        suffix = (
            f"_{plot_results._slug(opinion)}_{plot_results._slug(alpha)}"
        )
        outputs.extend(_plot_frame(subset, figures_dir, suffix=suffix))
    return outputs


def _plot_frame(
    frame: pd.DataFrame, figures_dir: Path, *, suffix: str
) -> list[Path]:
    working = _add_cumulative_hit_rate(frame)
    specs = [
        (
            "full_candidate_spearman",
            "Full-candidate Spearman coefficient",
            f"ais_score_full_spearman{suffix}.pdf",
            (-1.0, 1.0),
            False,
        ),
        (
            "exact_best_ais_rank",
            "Exact-best AIS rank",
            f"ais_score_exact_best_rank{suffix}.pdf",
            None,
            True,
        ),
        (
            "cumulative_hit_rate",
            r"Cumulative Hit@$k$ rate",
            f"ais_score_hit_rate{suffix}.pdf",
            (0.0, 1.0),
            False,
        ),
    ]
    return [
        _plot_metric(
            working,
            figures_dir / filename,
            value=value,
            ylabel=ylabel,
            ylim=ylim,
            rank_axis=rank_axis,
        )
        for value, ylabel, filename, ylim, rank_axis in specs
    ]


def _plot_metric(
    frame: pd.DataFrame,
    path: Path,
    *,
    value: str,
    ylabel: str,
    ylim: tuple[float, float] | None,
    rank_axis: bool,
) -> Path:
    fig, ax = plot_results.plt.subplots(figsize=(3.8, 2.7))
    methods = [f"AIS-Greedy (k={k})" for k in sorted(frame["k"].unique())]
    maximum_budget = int(frame["iteration"].max())
    marker_budgets = set(deactivation_marker_budgets(maximum_budget))

    for k, group in frame.groupby("k", sort=True):
        summary = (
            group.groupby("iteration", as_index=False)[value]
            .mean()
            .sort_values("iteration")
        )
        method = f"AIS-Greedy (k={int(k)})"
        markevery = [
            index
            for index, iteration in enumerate(summary["iteration"])
            if int(iteration) in marker_budgets
        ]
        ax.plot(
            summary["iteration"],
            summary[value],
            label=plot_results._legend_label(method),
            linewidth=1.8,
            markersize=3.0,
            markevery=markevery,
            alpha=0.55,
            **plot_results._style_for(method, methods),
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if rank_axis:
        ax.set_yscale("log")
        maximum_rank = float(frame[value].max())
        ax.set_ylim(max(10.0, maximum_rank * 1.05), 1.0)
    ax.legend(frameon=False, loc="best", handlelength=1.8)
    plot_results._set_dataset_title(ax, frame)
    return plot_results._save(fig, path)


def _add_cumulative_hit_rate(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.sort_values(
        [
            "dataset",
            "opinion_setting",
            "alpha_setting",
            "k",
            "iteration",
        ]
    ).copy()
    keys = ["dataset", "opinion_setting", "alpha_setting", "k"]
    hit = working["hit_at_k"].astype(float)
    working["cumulative_hit_rate"] = (
        hit.groupby([working[column] for column in keys]).cumsum()
        / (
            working.groupby(keys).cumcount().to_numpy(dtype=float)
            + 1.0
        )
    )
    return working


def _summarize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(columns, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(columns, key_tuple))
        spearman = group["full_candidate_spearman"].dropna().to_numpy(float)
        ranks = group["exact_best_ais_rank"].to_numpy(float)
        row.update(
            {
                "num_iterations": int(len(group)),
                "num_defined_spearman": int(len(spearman)),
                "mean_full_candidate_spearman": _mean_or_nan(spearman),
                "median_full_candidate_spearman": _quantile_or_nan(
                    spearman, 0.5
                ),
                "q25_full_candidate_spearman": _quantile_or_nan(
                    spearman, 0.25
                ),
                "q75_full_candidate_spearman": _quantile_or_nan(
                    spearman, 0.75
                ),
                "mean_exact_best_ais_rank": float(np.mean(ranks)),
                "median_exact_best_ais_rank": float(np.median(ranks)),
                "hit_rate_at_k": float(group["hit_at_k"].astype(float).mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _read_diagnostic_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing diagnostic CSV: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Diagnostic CSV is empty: {path}")
    required = {
        "dataset",
        "opinion_setting",
        "alpha_setting",
        "k",
        "iteration",
        "full_candidate_spearman",
        "exact_best_ais_rank",
        "hit_at_k",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Diagnostic CSV is missing columns: {sorted(missing)}")
    frame["hit_at_k"] = frame["hit_at_k"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    for column in [
        "k",
        "iteration",
        "full_candidate_spearman",
        "exact_best_ais_rank",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _quantile_or_nan(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")
