"""Publication-style plots for the fixed-lambda alpha-sweep experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from polar_ai.experiments import plot_results


FIXED_REGIME = "fixed_at_reference"
RECOMPUTED_REGIME = "recomputed_each_alpha"
REGIME_LABELS = {
    FIXED_REGIME: r"Fixed $\lambda$",
    RECOMPUTED_REGIME: r"Recomputed $\lambda$",
}
REGIME_STYLES = {
    FIXED_REGIME: {
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
    },
    RECOMPUTED_REGIME: {
        "color": "#d62728",
        "marker": "s",
        "linestyle": "--",
    },
}
OPINION_STYLES = {
    "uniform": {"color": "#1f77b4", "marker": "o"},
    "polarized": {"color": "#d62728", "marker": "s"},
}


def write_fixed_lambda_comparison(
    csv_path: Path, output_path: Path
) -> Path:
    """Write one row per alpha with fixed/recomputed ratios side by side."""

    frame = _read_sweep_csv(csv_path)
    index = ["dataset", "opinion_setting", "alpha"]
    comparison = (
        frame.pivot_table(
            index=index,
            columns="lambda_regime",
            values=["ratio", "lambda_value"],
            aggfunc="first",
        )
        .reset_index()
    )
    comparison.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in comparison.columns
    ]
    fixed_ratio = f"ratio_{FIXED_REGIME}"
    recomputed_ratio = f"ratio_{RECOMPUTED_REGIME}"
    comparison["ratio_difference_fixed_minus_recomputed"] = (
        comparison[fixed_ratio] - comparison[recomputed_ratio]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    return output_path


def plot_fixed_lambda_alpha_sweep(
    csv_path: Path, figures_dir: Path
) -> list[Path]:
    """Generate the fixed-only, direct-comparison, and lambda-schedule plots."""

    frame = _read_sweep_csv(csv_path)
    plot_results._set_publication_style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        _plot_fixed_only(
            frame,
            figures_dir / "fixed_lambda_alpha_sweep.pdf",
        )
    ]
    for opinion_setting in sorted(frame["opinion_setting"].unique()):
        outputs.append(
            _plot_regime_comparison(
                frame,
                str(opinion_setting),
                figures_dir
                / f"lambda_regime_comparison_{plot_results._slug(str(opinion_setting))}.pdf",
            )
        )
    outputs.append(
        _plot_lambda_schedule(frame, figures_dir / "lambda_schedule.pdf")
    )
    return outputs


def _plot_fixed_only(frame: pd.DataFrame, path: Path) -> Path:
    fixed = frame[frame["lambda_regime"] == FIXED_REGIME]
    fig, ax = plot_results.plt.subplots(figsize=(3.4, 2.5))
    for opinion_setting, group in fixed.groupby("opinion_setting", sort=True):
        summary = (
            group.groupby("alpha", as_index=False)["ratio"]
            .mean()
            .sort_values("alpha")
        )
        style = OPINION_STYLES.get(
            str(opinion_setting), {"color": "#000000", "marker": "o"}
        )
        ax.plot(
            summary["alpha"],
            summary["ratio"],
            label=str(opinion_setting),
            linestyle="-",
            markersize=3.0,
            linewidth=1.6,
            zorder=2,
            **style,
        )
    _format_ratio_axis(ax, fixed)
    ax.legend(frameon=False, loc="best")
    return plot_results._save(fig, path)


def _plot_regime_comparison(
    frame: pd.DataFrame, opinion_setting: str, path: Path
) -> Path:
    subset = frame[frame["opinion_setting"] == opinion_setting]
    fig, ax = plot_results.plt.subplots(figsize=(3.4, 2.5))
    for regime in [FIXED_REGIME, RECOMPUTED_REGIME]:
        group = subset[subset["lambda_regime"] == regime].sort_values("alpha")
        ax.plot(
            group["alpha"],
            group["ratio"],
            label=REGIME_LABELS[regime],
            markersize=3.0,
            linewidth=1.6,
            zorder=2,
            **REGIME_STYLES[regime],
        )
    _format_ratio_axis(ax, subset)
    ax.legend(frameon=False, loc="best")
    return plot_results._save(fig, path)


def _plot_lambda_schedule(frame: pd.DataFrame, path: Path) -> Path:
    schedule = frame.drop_duplicates(["lambda_regime", "alpha"])
    fig, ax = plot_results.plt.subplots(figsize=(3.4, 2.5))
    for regime in [FIXED_REGIME, RECOMPUTED_REGIME]:
        group = schedule[schedule["lambda_regime"] == regime].sort_values(
            "alpha"
        )
        ax.plot(
            group["alpha"],
            group["lambda_value"],
            label=REGIME_LABELS[regime],
            markersize=3.0,
            linewidth=1.6,
            zorder=2,
            **REGIME_STYLES[regime],
        )
    ax.set_xlabel(r"Extremizing parameter $\alpha$")
    ax.set_ylabel(r"Susceptibility $\lambda$")
    ax.legend(frameon=False, loc="best")
    plot_results._set_dataset_title(ax, frame)
    return plot_results._save(fig, path)


def _format_ratio_axis(ax: object, frame: pd.DataFrame) -> None:
    ax.set_xlabel(r"Extremizing parameter $\alpha$")
    ax.set_ylabel(r"$\mathrm{PD}(z^*(\emptyset)) / \mathrm{PD}(z^*(V))$")
    plot_results._set_dataset_title(ax, frame)


def _read_sweep_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed-lambda sweep CSV: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Fixed-lambda sweep CSV is empty: {path}")
    required = {
        "dataset",
        "opinion_setting",
        "lambda_regime",
        "alpha",
        "lambda_value",
        "ratio",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Fixed-lambda sweep CSV is missing: {sorted(missing)}")
    return frame
