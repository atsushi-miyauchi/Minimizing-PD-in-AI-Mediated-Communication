#!/usr/bin/env python3
"""Recompute polarization-decomposition metrics from raw CSVs.

The expensive experiment stores one row for each dataset, opinion setting,
and alpha value. This public-bundle version validates the six included main
networks and records both full-precision extrema and conservative bounds.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

import numpy as np
import pandas as pd


RAW_FILENAME = "polarization_decomposition.csv"
BY_DATASET_FILENAME = "paper_metrics_by_dataset.csv"
OVERALL_FILENAME = "paper_metrics_overall.csv"
REPORT_FILENAME = "paper_metrics.md"

EXPECTED_DATASETS = {
    "wiki-Vote",
    "hamster",
    "advogato",
    "epinions",
    "Slashdot0902",
    "youtube",
}
EXPECTED_OPINION_SETTINGS = {"uniform", "polarized"}
EXPECTED_NONBASELINE_ALPHAS = np.round(np.arange(1.1, 3.0 + 0.05, 0.1), 10)

REQUIRED_COLUMNS = {
    "dataset",
    "opinion_setting",
    "alpha",
    "absolute_mean_shift",
    "polarization_change",
    "dispersion_fraction_of_polarization_change",
    "mean_component_fraction_of_polarization_change",
    "active_equilibrium_converged",
    "baseline_equilibrium_converged",
    "decomposition_residual_active",
    "decomposition_residual_baseline",
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
    cases = _select_and_validate_manuscript_cases(raw)
    by_dataset = summarize_by_dataset(cases)
    overall = summarize_overall(cases)

    by_dataset_path = root / BY_DATASET_FILENAME
    overall_path = root / OVERALL_FILENAME
    report_path = root / REPORT_FILENAME
    by_dataset.to_csv(by_dataset_path, index=False)
    overall.to_csv(overall_path, index=False)
    report_path.write_text(
        render_report(by_dataset, overall.iloc[0]), encoding="utf-8"
    )
    print(by_dataset_path)
    print(overall_path)
    print(report_path)
    return 0


def summarize_by_dataset(cases: pd.DataFrame) -> pd.DataFrame:
    """Return full-precision extrema for each bundled network."""

    rows: list[dict[str, int | float | str]] = []
    for dataset, group in cases.groupby("dataset", sort=True):
        mean_shift_row = group.loc[group["absolute_mean_shift"].idxmax()]
        mean_fraction_row = group.loc[
            group["mean_component_fraction_percent"].idxmax()
        ]
        rows.append(
            {
                "dataset": str(dataset),
                "num_cases": int(len(group)),
                "max_absolute_mean_shift": float(
                    mean_shift_row["absolute_mean_shift"]
                ),
                "max_absolute_mean_shift_opinion_setting": str(
                    mean_shift_row["opinion_setting"]
                ),
                "max_absolute_mean_shift_alpha": float(mean_shift_row["alpha"]),
                "max_mean_component_fraction_percent": float(
                    mean_fraction_row["mean_component_fraction_percent"]
                ),
                "max_mean_component_fraction_opinion_setting": str(
                    mean_fraction_row["opinion_setting"]
                ),
                "max_mean_component_fraction_alpha": float(
                    mean_fraction_row["alpha"]
                ),
                "min_dispersion_fraction_percent": float(
                    group["dispersion_fraction_percent"].min()
                ),
                "min_polarization_change": float(
                    group["polarization_change"].min()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_overall(cases: pd.DataFrame) -> pd.DataFrame:
    """Return exact extrema and manuscript-facing conservative bounds."""

    mean_shift_row = cases.loc[cases["absolute_mean_shift"].idxmax()]
    mean_fraction_row = cases.loc[
        cases["mean_component_fraction_percent"].idxmax()
    ]
    exact_max_mean_shift = float(mean_shift_row["absolute_mean_shift"])
    exact_max_mean_fraction = float(
        mean_fraction_row["mean_component_fraction_percent"]
    )
    exact_min_dispersion_fraction = float(
        cases["dispersion_fraction_percent"].min()
    )

    row = {
        "num_datasets": int(cases["dataset"].nunique()),
        "num_opinion_settings": int(cases["opinion_setting"].nunique()),
        "num_nonbaseline_alpha_values": int(cases["alpha"].nunique()),
        "num_cases": int(len(cases)),
        "alpha_min_exclusive_baseline": float(cases["alpha"].min()),
        "alpha_max": float(cases["alpha"].max()),
        "max_absolute_mean_shift": exact_max_mean_shift,
        "max_absolute_mean_shift_dataset": str(mean_shift_row["dataset"]),
        "max_absolute_mean_shift_opinion_setting": str(
            mean_shift_row["opinion_setting"]
        ),
        "max_absolute_mean_shift_alpha": float(mean_shift_row["alpha"]),
        "manuscript_upper_bound_absolute_mean_shift": _round_up(
            exact_max_mean_shift, 4
        ),
        "max_mean_component_fraction_percent": exact_max_mean_fraction,
        "max_mean_component_fraction_dataset": str(
            mean_fraction_row["dataset"]
        ),
        "max_mean_component_fraction_opinion_setting": str(
            mean_fraction_row["opinion_setting"]
        ),
        "max_mean_component_fraction_alpha": float(
            mean_fraction_row["alpha"]
        ),
        "manuscript_upper_bound_mean_component_fraction_percent": _round_up(
            exact_max_mean_fraction, 2
        ),
        "min_dispersion_fraction_percent": exact_min_dispersion_fraction,
        "manuscript_lower_bound_dispersion_fraction_percent": _round_down(
            exact_min_dispersion_fraction, 2
        ),
        "min_polarization_change": float(cases["polarization_change"].min()),
        "max_fraction_sum_error": float(
            np.max(
                np.abs(
                    cases["dispersion_fraction_of_polarization_change"]
                    + cases["mean_component_fraction_of_polarization_change"]
                    - 1.0
                )
            )
        ),
        "max_absolute_decomposition_residual": float(
            np.max(
                np.abs(
                    cases[
                        [
                            "decomposition_residual_active",
                            "decomposition_residual_baseline",
                        ]
                    ].to_numpy(dtype=float)
                )
            )
        ),
    }
    return pd.DataFrame([row])


def render_report(by_dataset: pd.DataFrame, overall: pd.Series) -> str:
    """Render an auditable record of the derived values."""

    num_datasets = len(EXPECTED_DATASETS)
    num_cases = len(EXPECTED_DATASETS) * len(EXPECTED_OPINION_SETTINGS) * len(
        EXPECTED_NONBASELINE_ALPHAS
    )

    lines = [
        "# Polarization decomposition: public-bundle metrics",
        "",
        "The values below are recomputed from "
        f"{num_datasets} dataset-level `{RAW_FILENAME}` files. The baseline "
        f"rows at $\\alpha=1$ are excluded, leaving {num_cases} cases.",
        "",
        "For each case, the change in polarization is decomposed as",
        "",
        "$$",
        "P(z_\\alpha)-P(z_1) = "
        "\\bigl(D(z_\\alpha)-D(z_1)\\bigr) + "
        "n(\\bar z_\\alpha^2-\\bar z_1^2),",
        "$$",
        "",
        "where $D(z)=\\lVert z-\\bar z\\mathbf{1}\\rVert_2^2$. "
        f"All {num_cases} polarization changes are positive. Component shares are "
        "their respective changes divided by the polarization change.",
        "",
        "| Dataset | Cases | Max. $|\\bar z_\\alpha-\\bar z_1|$ | "
        "Max. mean-component share | Min. dispersion share |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in by_dataset.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {int(row.num_cases)} | "
            f"{row.max_absolute_mean_shift:.10f} | "
            f"{row.max_mean_component_fraction_percent:.8f}% | "
            f"{row.min_dispersion_fraction_percent:.8f}% |"
        )

    lines.extend(
        [
            "",
            "## Aggregate values for the bundled networks",
            "",
            f"- Number of cases: **{int(overall['num_cases'])}**.",
            "- Maximum absolute change in the mean equilibrium opinion: "
            f"**{overall['max_absolute_mean_shift']:.10f}** "
            f"({overall['max_absolute_mean_shift_dataset']}, "
            f"{overall['max_absolute_mean_shift_opinion_setting']}, "
            f"$\\alpha={overall['max_absolute_mean_shift_alpha']:.1f}$).",
            "- Maximum mean-component share of the polarization increase: "
            f"**{overall['max_mean_component_fraction_percent']:.8f}%** "
            f"({overall['max_mean_component_fraction_dataset']}, "
            f"{overall['max_mean_component_fraction_opinion_setting']}, "
            f"$\\alpha={overall['max_mean_component_fraction_alpha']:.1f}$).",
            "- Corresponding minimum dispersion share: "
            f"**{overall['min_dispersion_fraction_percent']:.8f}%**.",
            "",
            "Because the prose states upper and lower bounds, it uses "
            "conservative outward rounding:",
            "",
            f"- $|\\bar z_\\alpha-\\bar z_1|\\leq "
            f"{overall['manuscript_upper_bound_absolute_mean_shift']:.4f}$;",
            "- the mean component accounts for at most "
            f"${overall['manuscript_upper_bound_mean_component_fraction_percent']:.2f}\\%$;",
            "- the dispersion component accounts for at least "
            f"${overall['manuscript_lower_bound_dispersion_fraction_percent']:.2f}\\%$.",
            "",
        ]
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
    for column in REQUIRED_COLUMNS - {
        "dataset",
        "opinion_setting",
        "active_equilibrium_converged",
        "baseline_equilibrium_converged",
    }:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in [
        "active_equilibrium_converged",
        "baseline_equilibrium_converged",
    ]:
        frame[column] = frame[column].map(_parse_bool)
    return frame


def _select_and_validate_manuscript_cases(raw: pd.DataFrame) -> pd.DataFrame:
    datasets = set(raw["dataset"].astype(str))
    if datasets != EXPECTED_DATASETS:
        raise ValueError(
            "Expected exactly the six Figure 1 datasets; "
            f"found {sorted(datasets)}."
        )
    settings = set(raw["opinion_setting"].astype(str))
    if settings != EXPECTED_OPINION_SETTINGS:
        raise ValueError(
            "Expected uniform and polarized opinion settings; "
            f"found {sorted(settings)}."
        )

    cases = raw.loc[raw["alpha"] > 1.0 + 1e-12].copy()
    observed_alphas = np.sort(cases["alpha"].unique())
    if not np.allclose(
        observed_alphas, EXPECTED_NONBASELINE_ALPHAS, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "Expected nonbaseline alpha values 1.1, 1.2, ..., 3.0; "
            f"found {observed_alphas.tolist()}."
        )
    duplicate_key = ["dataset", "opinion_setting", "alpha"]
    if cases.duplicated(duplicate_key).any():
        raise ValueError("Duplicate dataset/opinion/alpha rows were found.")
    expected_cases = (
        len(EXPECTED_DATASETS)
        * len(EXPECTED_OPINION_SETTINGS)
        * len(EXPECTED_NONBASELINE_ALPHAS)
    )
    if len(cases) != expected_cases:
        raise ValueError(f"Expected {expected_cases} cases; found {len(cases)}.")
    if not (
        cases["active_equilibrium_converged"].all()
        and cases["baseline_equilibrium_converged"].all()
    ):
        raise ValueError("At least one equilibrium is marked nonconverged.")
    if not (cases["polarization_change"] > 0.0).all():
        raise ValueError(
            "The manuscript share statement requires a positive polarization "
            "increase in every selected case."
        )
    fraction_columns = [
        "dispersion_fraction_of_polarization_change",
        "mean_component_fraction_of_polarization_change",
    ]
    if not np.isfinite(cases[fraction_columns].to_numpy(dtype=float)).all():
        raise ValueError("At least one selected component share is undefined.")

    cases["dispersion_fraction_percent"] = (
        100.0 * cases["dispersion_fraction_of_polarization_change"]
    )
    cases["mean_component_fraction_percent"] = (
        100.0 * cases["mean_component_fraction_of_polarization_change"]
    )
    return cases


def _parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def _round_up(value: float, decimal_places: int) -> float:
    quantum = Decimal(1).scaleb(-decimal_places)
    return float(
        Decimal(str(float(value))).quantize(quantum, rounding=ROUND_CEILING)
    )


def _round_down(value: float, decimal_places: int) -> float:
    quantum = Decimal(1).scaleb(-decimal_places)
    return float(
        Decimal(str(float(value))).quantize(quantum, rounding=ROUND_FLOOR)
    )


def _parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Recompute Figure 1 polarization-decomposition manuscript "
            "metrics from raw result CSVs."
        )
    )
    parser.add_argument(
        "--results-dir",
        default=repository_root / "results/additional/polarization_decomposition",
        type=Path,
        help="Polarization-decomposition results root.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
