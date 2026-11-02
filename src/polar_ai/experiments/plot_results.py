"""Plot transformation-effect and deactivation experiment results."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "polar_ai_matplotlib")
)
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from polar_ai.experiments.utils import deactivation_marker_budgets  # noqa: E402

METHOD_LABELS = {
    "No-AI-Mediation": "No-AI-Mediation",
    "Greedy": "Greedy",
    "USampling-Greedy": "USampling-Greedy",
    "NeighborPD-Greedy": "NeighborPD-Greedy",
    "AIS-Greedy": "AIS-Greedy",
}
DEACTIVATION_ORDER = [
    "USampling-Greedy (k=10)",
    "USampling-Greedy (k=50)",
    "USampling-Greedy (k=200)",
    "NeighborPD-Greedy (k=10)",
    "NeighborPD-Greedy (k=50)",
    "NeighborPD-Greedy (k=200)",
    "AIS-Greedy (k=10)",
    "AIS-Greedy (k=50)",
    "AIS-Greedy (k=200)",
    "Greedy",
]
LINE_STYLES = {
    "No-AI-Mediation": {"color": "#000000", "linestyle": "-", "marker": None},
    "Greedy": {"color": "#000000", "linestyle": "-", "marker": "*"},
}
PLOT_CSV_DTYPES = {
    "timed_out": "boolean",
    "adjoint_converged": "boolean",
    "adjoint_large_residual": "boolean",
    "selected_indices": "string",
    "selected_labels": "string",
    "diagnostics": "string",
}
SPEARMAN_COLUMN = "score_exact_objective_spearman_correlation"


class PlotDataUnavailable(ValueError):
    """Raised when a plot has no generated input data yet."""


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    raw_dir = Path(args.raw_dir)
    figures_dir = Path(args.figures_dir)
    outputs = plot_deactivation_and_transformation_results(
        raw_dir,
        figures_dir,
        timeout_sec=float(args.timeout_sec),
    )
    for path in outputs:
        print(path)
    return 0


def plot_deactivation_and_transformation_results(
    raw_dir: Path,
    figures_dir: Path,
    *,
    timeout_sec: float = 1200.0,
) -> list[Path]:
    """Plot only the experiment families produced by the suite runner."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    _set_publication_style()
    filters: dict[str, str | None] = {
        "dataset": None,
        "opinion_setting": None,
        "alpha_setting": None,
    }
    outputs = [
        plot_transformation_effect(raw_dir, figures_dir),
        plot_deactivation_quality(
            raw_dir, figures_dir, filters, timeout_sec
        ),
    ]
    spearman_path = _maybe_plot(
        plot_deactivation_spearman,
        raw_dir,
        figures_dir,
        filters,
        timeout_sec,
    )
    if spearman_path is not None:
        outputs.append(spearman_path)
    outputs.append(
        plot_deactivation_runtime(
            raw_dir, figures_dir, filters, timeout_sec
        )
    )
    order = _deactivation_order_from_csv(raw_dir, filters)
    if order:
        outputs.append(
            plot_method_legend(
                figures_dir,
                order,
                "deactivation_legend.pdf",
                "AIS-Greedy",
            )
        )
    outputs.extend(
        plot_parameter_setting_figures(
            raw_dir,
            figures_dir,
            filters,
            timeout_sec,
        )
    )
    return outputs


def _maybe_plot(plotter: Callable[..., Path], *args: object, **kwargs: object) -> Path | None:
    """Generate one plot, or skip it when its input data has not been produced."""

    try:
        return plotter(*args, **kwargs)
    except (FileNotFoundError, PlotDataUnavailable):
        return None


def plot_transformation_effect(raw_dir: Path, figures_dir: Path) -> Path:
    df = _read_csv(raw_dir / "transformation_effect.csv")
    if "ratio_average_over_opinions" in df.columns:
        df = df[~df["ratio_average_over_opinions"].astype(bool)].copy()

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    styles = {
        "uniform": {"color": "#1f77b4", "marker": "o"},
        "polarized": {"color": "#d62728", "marker": "s"},
    }
    for opinion_setting, group in df.groupby("opinion_setting", sort=True):
        summary = _mean_by(group, ["alpha"], "ratio").sort_values("alpha")
        style = styles.get(
            str(opinion_setting), {"color": "#000000", "marker": "o"}
        )
        ax.plot(
            summary["alpha"],
            summary["mean"],
            label=str(opinion_setting),
            linestyle="-",
            markersize=3.0,
            linewidth=1.6,
            zorder=2,
            **style,
        )
    ax.set_xlabel(r"Extremizing parameter $\alpha$")
    ax.set_ylabel(r"$\mathrm{PD}(z^*(\emptyset)) / \mathrm{PD}(z^*(V))$")
    _set_dataset_title(ax, df)
    ax.legend(frameon=False, loc="best")
    return _save(fig, figures_dir / "transformation_effect.pdf")


def plot_deactivation_quality(
    raw_dir: Path,
    figures_dir: Path,
    filters: dict[str, str | None],
    timeout_sec: float = 1200.0,
    filename: str = "deactivation_quality.pdf",
    *,
    show_usampling_std: bool = False,
) -> Path:
    df = _filter_frame(_read_csv(raw_dir / "deactivation.csv"), filters)
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    order = _extend_method_order(DEACTIVATION_ORDER, _label_methods(df, "method"))
    _plot_grouped_methods(
        ax,
        df,
        value="normalized_objective",
        order=order,
        method_column="method",
        timeout_sec=timeout_sec,
        marker_at_deactivation_budgets=True,
        show_usampling_std=show_usampling_std,
    )
    _annotate_no_ai_value(ax, df, "method", timeout_sec)
    ax.set_xlabel(r"Budget $b$")
    ax.set_ylabel("Normalized objective")
    _set_dataset_title(ax, df)
    return _save(fig, figures_dir / filename)


def plot_deactivation_runtime(
    raw_dir: Path,
    figures_dir: Path,
    filters: dict[str, str | None],
    timeout_sec: float = 1200.0,
    filename: str = "deactivation_runtime.pdf",
    *,
    show_usampling_std: bool = False,
) -> Path:
    df = _filter_frame(_read_csv(raw_dir / "deactivation.csv"), filters)
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    order = _extend_method_order(DEACTIVATION_ORDER, _label_methods(df, "method"))
    _plot_grouped_methods(
        ax,
        df,
        value="runtime_sec",
        order=order,
        method_column="method",
        timeout_sec=timeout_sec,
        marker_at_deactivation_budgets=True,
        show_usampling_std=show_usampling_std,
    )
    ax.set_xlabel(r"Budget $b$")
    ax.set_ylabel("Runtime (s)")
    ax.set_yscale("log")
    _set_dataset_title(ax, df)
    return _save(fig, figures_dir / filename)


def plot_deactivation_spearman(
    raw_dir: Path,
    figures_dir: Path,
    filters: dict[str, str | None],
    timeout_sec: float = 1200.0,
    filename: str = "deactivation_spearman.pdf",
) -> Path:
    """Plot per-iteration Spearman coefficients for each family's largest k."""

    df = _filter_frame(_read_csv(raw_dir / "deactivation.csv"), filters)
    df = _prepare_spearman_plot_data(df)
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    order = _extend_method_order(DEACTIVATION_ORDER, df["method_label"])
    _plot_grouped_methods(
        ax,
        df,
        value=SPEARMAN_COLUMN,
        order=order,
        method_column="method_label",
        timeout_sec=timeout_sec,
        marker_at_deactivation_budgets=True,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Spearman coefficient")
    ax.set_ylim(-1.0, 1.0)
    ax.legend(
        frameon=False,
        loc="best",
        handlelength=1.8,
    )
    _set_dataset_title(ax, df)
    return _save(fig, figures_dir / filename)


def plot_parameter_setting_figures(
    raw_dir: Path,
    figures_dir: Path,
    filters: dict[str, str | None],
    timeout_sec: float,
) -> list[Path]:
    """Write quality/runtime figures for each opinion x alpha setting."""

    if filters.get("opinion_setting") is not None or filters.get("alpha_setting") is not None:
        return []

    try:
        df = _filter_frame(
            _read_csv(raw_dir / "deactivation.csv"),
            {
                "dataset": filters.get("dataset"),
                "opinion_setting": None,
                "alpha_setting": None,
            },
        )
    except (FileNotFoundError, PlotDataUnavailable):
        return []

    if "opinion_setting" not in df.columns or "alpha_setting" not in df.columns:
        return []

    settings = (
        df[["opinion_setting", "alpha_setting"]]
        .drop_duplicates()
        .sort_values(["opinion_setting", "alpha_setting"])
    )
    outputs: list[Path] = []
    for row in settings.itertuples(index=False):
        opinion_setting = str(row.opinion_setting)
        alpha_setting = str(row.alpha_setting)
        setting_filters = {
            **filters,
            "opinion_setting": opinion_setting,
            "alpha_setting": alpha_setting,
        }
        suffix = f"{_slug(opinion_setting)}_{_slug(alpha_setting)}"
        plot_specs = [
            (
                plot_deactivation_quality,
                f"deactivation_quality_{suffix}.pdf",
            ),
            (
                plot_deactivation_spearman,
                f"deactivation_spearman_{suffix}.pdf",
            ),
            (
                plot_deactivation_runtime,
                f"deactivation_runtime_{suffix}.pdf",
            ),
        ]
        for plotter, filename in plot_specs:
            try:
                plot_kwargs = {"filename": filename}
                if plotter in {
                    plot_deactivation_quality,
                    plot_deactivation_runtime,
                }:
                    plot_kwargs["show_usampling_std"] = True
                outputs.append(
                    plotter(
                        raw_dir,
                        figures_dir,
                        setting_filters,
                        timeout_sec,
                        **plot_kwargs,
                    )
                )
            except (FileNotFoundError, PlotDataUnavailable):
                continue
    return outputs


def plot_method_legend(
    figures_dir: Path,
    order: list[str],
    filename: str,
    break_before: str | None = None,
) -> Path:
    entries = []
    fig, ax = plt.subplots(figsize=(2.85, 0.17 * len(order) + 0.02))
    for method in order:
        (handle,) = ax.plot(
            [],
            [],
            label=method,
            linewidth=1.6,
            markersize=3.0,
            alpha=0.55,
            **_style_for(method, order),
        )
        entries.append((handle, method, _legend_label(method)))

    break_index = next(
        (
            index
            for index, (_, method, _) in enumerate(entries)
            if break_before is not None and method.startswith(break_before)
        ),
        None,
    )
    if break_index is not None and 0 < break_index < len(entries):
        first_row = entries[:break_index]
        second_row = entries[break_index:]
        columns = max(len(first_row), len(second_row))
        (empty_handle,) = ax.plot([], [], linewidth=0.0, alpha=0.0)
        empty_entry = (empty_handle, "", "")
        first_row.extend([empty_entry] * (columns - len(first_row)))
        second_row.extend([empty_entry] * (columns - len(second_row)))
        # Matplotlib fills each legend column from top to bottom.
        entries = [
            entry
            for column_entries in zip(first_row, second_row)
            for entry in column_entries
        ]
        rows = 2
    else:
        columns = 1
        rows = len(entries)

    fig.set_size_inches(max(2.85, 1.9 * columns), 0.17 * rows + 0.02)
    handles = [handle for handle, _, _ in entries]
    labels = [label for _, _, label in entries]
    ax.axis("off")
    legend = ax.legend(
        handles,
        labels,
        frameon=False,
        loc="center",
        ncol=columns,
        borderaxespad=0.0,
        borderpad=0.0,
        handlelength=1.45,
        handletextpad=0.32,
        labelspacing=0.12,
        columnspacing=0.8 if columns > 1 else 0.0,
    )
    return _save_legend(fig, legend, figures_dir / filename)


def _deactivation_order_from_csv(
    raw_dir: Path, filters: dict[str, str | None]
) -> list[str]:
    try:
        df = _filter_frame(_read_csv(raw_dir / "deactivation.csv"), filters)
    except (FileNotFoundError, PlotDataUnavailable):
        return []
    return _extend_method_order(DEACTIVATION_ORDER, _label_methods(df, "method"))


def _plot_grouped_methods(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    value: str,
    order: list[str],
    method_column: str,
    timeout_sec: float,
    marker_every: int | None = None,
    marker_at_deactivation_budgets: bool = False,
    show_usampling_std: bool = False,
) -> None:
    working = df.copy()
    if method_column == "method":
        working["method_label"] = _label_methods(working, "method")
        method_column = "method_label"
    working = working[working[method_column] != "No-AI-Mediation"].copy()
    working = _drop_timeout_and_later_rows(working, method_column, timeout_sec)

    for method in order:
        group = working[working[method_column] == method]
        if group.empty:
            continue
        summary = _mean_by(group, ["budget"], value).sort_values("budget")
        style = _style_for(method, order)
        markevery: int | list[int] | None = marker_every
        if marker_at_deactivation_budgets:
            b_max = int(working["budget"].max())
            marker_budgets = set(deactivation_marker_budgets(b_max))
            markevery = [
                index
                for index, budget in enumerate(summary["budget"])
                if int(budget) in marker_budgets
            ]
        ax.plot(
            summary["budget"],
            summary["mean"],
            label=_legend_label(method),
            linewidth=1.8
            if (
                method.startswith("USampling-Greedy")
                or method.startswith("NeighborPD-Greedy")
                or method.startswith("AIS-Greedy")
                or method == "Greedy"
            )
            else 1.3,
            markersize=3.0,
            markevery=markevery,
            alpha=0.55,
            **style,
        )
        if show_usampling_std and method.startswith("USampling-Greedy"):
            _add_error_band(
                ax,
                summary,
                "budget",
                lower_bound=0.0 if value == "runtime_sec" else None,
            )


def _prepare_spearman_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep recorded Spearman series at the largest available k per family."""

    if SPEARMAN_COLUMN not in df.columns:
        raise PlotDataUnavailable(
            f"CSV has no Spearman coefficient column: {SPEARMAN_COLUMN}"
        )

    working = df.copy()
    working[SPEARMAN_COLUMN] = pd.to_numeric(
        working[SPEARMAN_COLUMN], errors="coerce"
    )
    working = working[working[SPEARMAN_COLUMN].notna()].copy()
    if working.empty:
        raise PlotDataUnavailable("No recorded Spearman coefficients are available.")

    working["method_label"] = _label_methods(working, "method")
    working["_spearman_family"] = working["method_label"].map(
        lambda method: _screening_family(str(method)) or str(method)
    )
    working["_spearman_k"] = working["method_label"].map(_method_k_value)
    maximum_k = working.groupby("_spearman_family")["_spearman_k"].transform(
        "max"
    )
    return working[working["_spearman_k"] == maximum_k].copy()


def _label_methods(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].map(METHOD_LABELS).fillna(df[column]).astype(str)


def _extend_method_order(base_order: list[str], labels: pd.Series) -> list[str]:
    present = set(labels.astype(str))
    present.discard("No-AI-Mediation")
    return sorted(
        present,
        key=lambda method: _method_order_key(base_order, method),
    )


def _method_order_key(base_order: list[str], method: str) -> tuple[int, int, str]:
    if method == "Greedy":
        return (3, 0, method)
    if method.startswith("USampling-Greedy"):
        return (0, _method_k_value(method), method)
    if method.startswith("NeighborPD-Greedy"):
        return (1, _method_k_value(method), method)
    if method.startswith("AIS-Greedy"):
        return (2, _method_k_value(method), method)
    return (4, _method_k_value(method), method)


def _method_k_value(method: str) -> int:
    match = re.search(r"k=(\d+)", method)
    return int(match.group(1)) if match else 10**9


def _legend_label(method: str) -> str:
    """Render parenthesized screening-size parameters with a math italic k."""

    return re.sub(r"\(k=(\d+)\)", r"($k=\1$)", method)


def _screening_color(
    k: int,
    colors: dict[int, str],
    available_ks: Iterable[int] | None = None,
) -> str:
    """Select a darker shade based on k's rank among the displayed series."""

    if available_ks is not None:
        displayed_ks = sorted(set(available_ks))
        if k not in displayed_ks:
            displayed_ks.append(k)
            displayed_ks.sort()
        palette = [colors[threshold] for threshold in sorted(colors)]
        if len(displayed_ks) <= len(palette):
            # Use the dark end of the palette.  A plot with only a few k values
            # should not inherit the almost-white shades reserved for plots
            # that need all five levels.
            offset = len(palette) - len(displayed_ks)
            return palette[offset + displayed_ks.index(k)]

        # More k values than predefined shades: spread them over the complete
        # palette while preserving monotonic darkness.
        rank = displayed_ks.index(k)
        color_index = round(rank * (len(palette) - 1) / (len(displayed_ks) - 1))
        return palette[color_index]

    thresholds = sorted(colors)
    threshold = next((value for value in thresholds if k <= value), thresholds[-1])
    return colors[threshold]


def _screening_family(method: str) -> str | None:
    """Return the shared color family for methods whose shades encode k."""

    families = {
        "ais": ("AIS-Greedy",),
        "neighbor_pd": ("NeighborPD-Greedy",),
        "usampling": ("USampling-Greedy",),
    }
    return next(
        (
            family
            for family, prefixes in families.items()
            if method.startswith(prefixes)
        ),
        None,
    )


def _displayed_family_ks(
    method: str, available_methods: Iterable[str] | None
) -> list[int] | None:
    if available_methods is None:
        return None
    family = _screening_family(method)
    if family is None:
        return None
    return [
        _method_k_value(candidate)
        for candidate in available_methods
        if _screening_family(candidate) == family
        and _method_k_value(candidate) != 10**9
    ]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _annotate_no_ai_value(
    ax: plt.Axes,
    df: pd.DataFrame,
    method_column: str,
    timeout_sec: float,
) -> None:
    working = _drop_timeout_and_later_rows(df.copy(), method_column, timeout_sec)
    if method_column not in working.columns:
        return
    group = working[working[method_column] == "No-AI-Mediation"]
    if group.empty or "normalized_objective" not in group.columns:
        return
    keys = [
        column
        for column in ["dataset", "opinion_setting", "alpha_setting"]
        if column in group.columns
    ]
    if keys:
        values = group.groupby(keys)["normalized_objective"].mean()
        value = float(values.mean())
    else:
        value = float(group["normalized_objective"].mean())
    ax.text(
        0.02,
        0.06,
        f"No-AI-mediation = {value:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        zorder=8,
    )


def _mean_by(df: pd.DataFrame, columns: list[str], value: str) -> pd.DataFrame:
    grouped = df.groupby(columns, as_index=False)[value].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def _drop_timeout_and_later_rows(
    df: pd.DataFrame, method_column: str, timeout_sec: float
) -> pd.DataFrame:
    if df.empty or "budget" not in df.columns:
        return df

    working = df.copy()
    if "timed_out" in working.columns:
        timeout_mask = working["timed_out"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
    elif timeout_sec > 0.0 and "runtime_sec" in working.columns:
        timeout_mask = working["runtime_sec"].astype(float) > timeout_sec
    else:
        return working

    key_columns = [
        column
        for column in ["dataset", method_column]
        if column in working.columns
    ]
    if not key_columns:
        return working[~timeout_mask].copy()

    cutoff_rows = working[timeout_mask]
    if cutoff_rows.empty:
        return working

    cutoffs = (
        cutoff_rows.groupby(key_columns, as_index=False)["budget"]
        .min()
        .rename(columns={"budget": "_timeout_budget"})
    )
    merged = working.merge(cutoffs, on=key_columns, how="left")
    keep = merged["_timeout_budget"].isna() | (
        merged["budget"] < merged["_timeout_budget"]
    )
    return merged.loc[keep, working.columns].copy()


def _add_error_band(
    ax: plt.Axes,
    summary: pd.DataFrame,
    x_col: str,
    *,
    alpha: float = 0.16,
    lower_bound: float | None = None,
) -> None:
    if "std" not in summary.columns or np.allclose(summary["std"], 0.0):
        return
    line = ax.lines[-1]
    color = line.get_color()
    x = summary[x_col].to_numpy(dtype=float)
    y = summary["mean"].to_numpy(dtype=float)
    std = summary["std"].to_numpy(dtype=float)
    lower = y - std
    if lower_bound is not None:
        positive_y = y[y > lower_bound]
        if positive_y.size:
            # Keep a finite, readable lower edge on logarithmic runtime axes.
            floor = max(lower_bound, float(np.min(positive_y)) * 0.1)
            lower = np.maximum(lower, floor)
    ax.fill_between(x, lower, y + std, color=color, alpha=alpha, linewidth=0)


def _filter_frame(df: pd.DataFrame, filters: dict[str, str | None]) -> pd.DataFrame:
    filtered = df.copy()
    for column, value in filters.items():
        if value is not None and column in filtered.columns:
            filtered = filtered[filtered[column].astype(str) == str(value)]
    if filtered.empty:
        raise PlotDataUnavailable("No rows remain after applying plotting filters.")
    return filtered


def _set_dataset_title(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Place the graph/dataset name above a plot."""

    if "dataset" not in df.columns:
        return
    datasets = sorted({str(value) for value in df["dataset"].dropna().unique()})
    if datasets:
        ax.set_title(" / ".join(datasets), pad=4.0)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    df = pd.read_csv(path, dtype=PLOT_CSV_DTYPES)
    if df.empty:
        raise PlotDataUnavailable(f"CSV is empty: {path}")
    return df


def _style_for(
    method: str, available_methods: Iterable[str] | None = None
) -> dict[str, object]:
    available_ks = _displayed_family_ks(method, available_methods)
    if method.startswith("NeighborPD-Greedy"):
        k = _method_k_value(method)
        local_pd_colors = {
            10: "#eff3ff",
            50: "#c6dbef",
            100: "#9ecae1",
            200: "#4292c6",
            500: "#08519c",
        }
        return {
            "color": _screening_color(k, local_pd_colors, available_ks),
            "linestyle": "-",
            "marker": "^",
        }
    if method.startswith("AIS-Greedy"):
        k = _method_k_value(method)
        ais_colors = {
            10: "#fee5d9",
            50: "#fcae91",
            100: "#fb6a4a",
            200: "#de2d26",
            500: "#a50f15",
        }
        return {
            "color": _screening_color(k, ais_colors, available_ks),
            "linestyle": "-",
            "marker": "v",
        }
    if method.startswith("USampling-Greedy"):
        k = _method_k_value(method)
        usample_colors = {
            10: "#ffffcc",
            50: "#d9f0a3",
            100: "#addd8e",
            200: "#41ab5d",
            500: "#006837",
        }
        return {
            "color": _screening_color(k, usample_colors, available_ks),
            "linestyle": "-",
            "marker": "P",
        }
    return dict(LINE_STYLES.get(method, {"marker": "o"}))


def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _save(fig: plt.Figure, path: Path, *, pad_inches: float = 0.02) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.set_axisbelow(True)
        if getattr(ax, "_polar_ai_skip_grid", False):
            ax.grid(False)
        else:
            ax.grid(True, color="#e6e6e6", linewidth=0.5, zorder=0)
            for gridline in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
                gridline.set_zorder(0)
        for spine in ax.spines.values():
            spine.set_visible(True)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_zorder(100)
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_alpha(1.0)
            legend.get_frame().set_zorder(99)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    return path


def _save_legend(fig: plt.Figure, legend: object, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, format="pdf", bbox_inches=bbox, pad_inches=0.0)
    plt.close(fig)
    return path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PDF figures from CSV logs.")
    parser.add_argument("--raw-dir", default="results/raw", help="Directory containing CSV logs.")
    parser.add_argument(
        "--figures-dir", default="results/figures", help="Directory for output PDF figures."
    )
    parser.add_argument("--dataset", default=None, help="Optional dataset filter.")
    parser.add_argument(
        "--opinion-setting", default=None, help="Optional opinion_setting filter."
    )
    parser.add_argument("--alpha-setting", default=None, help="Optional alpha_setting filter.")
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=1200.0,
        help="Drop timeout budgets and larger budgets from quality/runtime plots.",
    )
    return parser.parse_args(argv)

if __name__ == "__main__":
    raise SystemExit(main())
