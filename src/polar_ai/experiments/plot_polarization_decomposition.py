"""Publication-style outputs for the polarization-decomposition experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from polar_ai.experiments import plot_results


COMPONENT_STYLES = {
    "dispersion_change_over_baseline_polarization": {
        "label": "Dispersion component",
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
    },
    "mean_component_change_over_baseline_polarization": {
        "label": "Mean component",
        "color": "#d62728",
        "marker": "s",
        "linestyle": "-",
    },
    "polarization_change_over_baseline_polarization": {
        "label": "Total polarization",
        "color": "#000000",
        "marker": None,
        "linestyle": "--",
    },
}


def plot_polarization_decomposition(
    csv_path: Path, figures_dir: Path
) -> list[Path]:
    """Plot normalized component changes and the equilibrium mean."""

    frame = _read_decomposition_csv(csv_path)
    plot_results._set_publication_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    return [
        _plot_component_changes(
            frame,
            figures_dir / "polarization_decomposition.pdf",
        ),
        _plot_equilibrium_means(
            frame,
            figures_dir / "equilibrium_mean.pdf",
        ),
    ]


def write_polarization_decomposition_summary(
    csv_path: Path,
    output_path: Path,
    *,
    min_relative_change: float = 1e-10,
) -> Path:
    """Write endpoint statistics used to answer the reviewer's question."""

    frame = _read_decomposition_csv(csv_path)
    endpoint_alpha = float(frame["alpha"].max())
    endpoint = frame[np.isclose(frame["alpha"], endpoint_alpha)].copy()
    if endpoint.empty:
        raise ValueError(f"No rows found at endpoint alpha={endpoint_alpha}.")

    valid_change = (
        endpoint["polarization_change_over_baseline_polarization"].abs()
        > float(min_relative_change)
    )
    endpoint.loc[
        ~valid_change, "dispersion_fraction_of_polarization_change"
    ] = np.nan
    endpoint.loc[
        ~valid_change, "mean_component_fraction_of_polarization_change"
    ] = np.nan
    endpoint["endpoint_alpha"] = endpoint_alpha
    endpoint["absolute_baseline_mean"] = endpoint["mean_baseline"].abs()
    endpoint["absolute_active_mean"] = endpoint["mean_active"].abs()
    endpoint["absolute_mean_magnitude_change"] = (
        endpoint["absolute_active_mean"] - endpoint["absolute_baseline_mean"]
    )

    columns = [
        "dataset",
        "opinion_setting",
        "seed",
        "n",
        "endpoint_alpha",
        "lambda_value",
        "mean_baseline",
        "mean_active",
        "mean_shift",
        "absolute_mean_shift",
        "absolute_baseline_mean",
        "absolute_active_mean",
        "absolute_mean_magnitude_change",
        "dispersion_change",
        "mean_component_change",
        "polarization_change",
        "dispersion_change_over_baseline_polarization",
        "mean_component_change_over_baseline_polarization",
        "polarization_change_over_baseline_polarization",
        "dispersion_fraction_of_polarization_change",
        "mean_component_fraction_of_polarization_change",
        "pd_ratio",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    endpoint[columns].sort_values(
        ["dataset", "opinion_setting"]
    ).to_csv(output_path, index=False)
    return output_path


def _plot_component_changes(frame: pd.DataFrame, path: Path) -> Path:
    settings = _ordered_opinion_settings(frame)
    fig, axes = plot_results.plt.subplots(
        1,
        len(settings),
        figsize=(3.35 * len(settings), 2.55),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for index, opinion_setting in enumerate(settings):
        ax = axes[0, index]
        subset = frame[
            frame["opinion_setting"].astype(str) == opinion_setting
        ].sort_values("alpha")
        for column, style in COMPONENT_STYLES.items():
            ax.plot(
                subset["alpha"],
                subset[column],
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=3.0,
                linewidth=1.6,
                markevery=2,
                zorder=3,
            )
        ax.axhline(0.0, color="#777777", linewidth=0.7, zorder=1)
        ax.set_xlabel(r"Extremizing parameter $\alpha$")
        ax.set_title(opinion_setting, pad=4.0)
        if index == 0:
            ax.set_ylabel(r"Component change normalized by $P_1$")
        if index == len(settings) - 1:
            ax.legend(frameon=False, loc="best")
    _add_dataset_suptitle(fig, frame)
    return plot_results._save(fig, path)


def _plot_equilibrium_means(frame: pd.DataFrame, path: Path) -> Path:
    fig, ax = plot_results.plt.subplots(figsize=(3.4, 2.5))
    styles = {
        "uniform": {"color": "#1f77b4", "marker": "o"},
        "polarized": {"color": "#d62728", "marker": "s"},
    }
    for opinion_setting, group in frame.groupby("opinion_setting", sort=True):
        group = group.sort_values("alpha")
        style = styles.get(
            str(opinion_setting), {"color": "#000000", "marker": "o"}
        )
        ax.plot(
            group["alpha"],
            group["mean_active"],
            label=str(opinion_setting),
            linestyle="-",
            markersize=3.0,
            linewidth=1.6,
            markevery=2,
            zorder=3,
            **style,
        )
    ax.axhline(0.0, color="#777777", linewidth=0.7, zorder=1)
    ax.set_xlabel(r"Extremizing parameter $\alpha$")
    ax.set_ylabel(r"Equilibrium mean $\bar{z}^*$")
    plot_results._set_dataset_title(ax, frame)
    ax.legend(frameon=False, loc="best")
    return plot_results._save(fig, path)


def _ordered_opinion_settings(frame: pd.DataFrame) -> list[str]:
    present = {str(value) for value in frame["opinion_setting"].unique()}
    preferred = [value for value in ["uniform", "polarized"] if value in present]
    return preferred + sorted(present - set(preferred))


def _add_dataset_suptitle(fig: object, frame: pd.DataFrame) -> None:
    datasets = sorted({str(value) for value in frame["dataset"].dropna().unique()})
    if datasets:
        fig.suptitle(" / ".join(datasets), fontsize=8, y=1.01)


def _read_decomposition_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing decomposition CSV: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Decomposition CSV is empty: {path}")
    required = {
        "dataset",
        "opinion_setting",
        "alpha",
        "mean_active",
        "mean_baseline",
        "dispersion_change_over_baseline_polarization",
        "mean_component_change_over_baseline_polarization",
        "polarization_change_over_baseline_polarization",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Decomposition CSV is missing columns: {sorted(missing)}"
        )
    return frame
