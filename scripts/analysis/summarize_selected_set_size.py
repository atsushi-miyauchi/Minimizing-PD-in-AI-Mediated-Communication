#!/usr/bin/env python3
"""Recompute manuscript selected-set-size metrics from compact raw rows."""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd


RAW_FILENAME = "selected_set_size_rows.csv"
NON_WIKI_OUTPUT = "paper_metrics_non_wiki_by_k.csv"
WIKI_OUTPUT = "paper_metrics_wiki_by_budget.csv"
OVERALL_OUTPUT = "paper_metrics_overall.csv"
REPORT_OUTPUT = "paper_metrics.md"

EXPECTED_DATASETS = {
    "wiki-Vote",
    "hamster",
    "advogato",
    "epinions",
    "Slashdot0902",
    "youtube",
}
METHOD_TO_K = {
    "AIS-Greedy (k=1)": 1,
    "AIS-Greedy (k=10)": 10,
    "AIS-Greedy (k=100)": 100,
}
WIKI_REPORT_BUDGETS = {100, 200, 300, 400, 500}
REQUIRED_COLUMNS = {
    "dataset",
    "opinion_setting",
    "alpha_setting",
    "budget",
    "method",
    "repetition",
    "timed_out",
    "selected_set_size",
}


def main() -> int:
    args = _parse_args()
    root = Path(args.results_dir).resolve()
    frames = [
        _read_compact_rows(path)
        for path in sorted(root.glob(f"*/{RAW_FILENAME}"))
    ]
    if not frames:
        raise FileNotFoundError(
            f"No dataset-level {RAW_FILENAME} files were found under {root}."
        )
    rows = _select_and_validate(pd.concat(frames, ignore_index=True))
    non_wiki = summarize_non_wiki(rows)
    wiki = summarize_wiki(rows)
    overall = summarize_overall(non_wiki, wiki, rows)

    non_wiki_path = root / NON_WIKI_OUTPUT
    wiki_path = root / WIKI_OUTPUT
    overall_path = root / OVERALL_OUTPUT
    report_path = root / REPORT_OUTPUT
    non_wiki.to_csv(non_wiki_path, index=False)
    wiki.to_csv(wiki_path, index=False)
    overall.to_csv(overall_path, index=False)
    report_path.write_text(
        render_report(non_wiki, wiki, overall.iloc[0]), encoding="utf-8"
    )
    print(non_wiki_path)
    print(wiki_path)
    print(overall_path)
    print(report_path)
    return 0


def summarize_non_wiki(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize all completed AIS-Greedy rows outside wiki-Vote."""

    subset = rows.loc[rows["dataset"] != "wiki-Vote"].copy()
    output: list[dict[str, int | float]] = []
    for k, group in subset.groupby("k", sort=True):
        final = group.loc[group["budget"] == 500]
        output.append(
            {
                "k": int(k),
                "num_completed_runs": int(len(group)),
                "num_full_budget_runs": int(group["uses_full_budget"].sum()),
                "full_budget_rate": float(group["uses_full_budget"].mean()),
                "full_budget_rate_percent": float(
                    100.0 * group["uses_full_budget"].mean()
                ),
                "manuscript_full_budget_rate_percent": _round_half_up(
                    100.0 * group["uses_full_budget"].mean(), 1
                ),
                "max_unused_budget": int(group["unused_budget"].max()),
                "num_completed_runs_at_b500": int(len(final)),
                "num_full_budget_runs_at_b500": int(
                    final["uses_full_budget"].sum()
                ),
                "all_completed_runs_at_b500_use_full_budget": bool(
                    len(final) > 0 and final["uses_full_budget"].all()
                ),
            }
        )
    return pd.DataFrame(output)


def summarize_wiki(rows: pd.DataFrame) -> pd.DataFrame:
    """Return selected sizes at the five manuscript-reporting budgets."""

    subset = rows.loc[
        (rows["dataset"] == "wiki-Vote")
        & rows["budget"].isin(WIKI_REPORT_BUDGETS)
    ].copy()
    output: list[dict[str, int | float]] = []
    for (k, budget), group in subset.groupby(["k", "budget"], sort=True):
        output.append(
            {
                "k": int(k),
                "budget": int(budget),
                "num_parameter_settings": int(len(group)),
                "mean_selected_set_size": float(
                    group["selected_set_size"].mean()
                ),
                "min_selected_set_size": int(group["selected_set_size"].min()),
                "max_selected_set_size": int(group["selected_set_size"].max()),
                "full_budget_count": int(group["uses_full_budget"].sum()),
                "full_budget_rate": float(group["uses_full_budget"].mean()),
            }
        )
    result = pd.DataFrame(output)
    expected_rows = len(METHOD_TO_K) * len(WIKI_REPORT_BUDGETS)
    if len(result) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} wiki-Vote summary rows; found {len(result)}."
        )
    if not (result["num_parameter_settings"] == 4).all():
        raise ValueError(
            "Every reported wiki-Vote budget must contain all four settings."
        )
    return result


def summarize_overall(
    non_wiki: pd.DataFrame,
    wiki: pd.DataFrame,
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return the compact values quoted in the manuscript."""

    values: dict[str, int | float | bool] = {
        "num_datasets": int(rows["dataset"].nunique()),
        "num_non_wiki_datasets": int(
            rows.loc[rows["dataset"] != "wiki-Vote", "dataset"].nunique()
        ),
    }
    for row in non_wiki.itertuples(index=False):
        k = int(row.k)
        values[f"non_wiki_k{k}_num_completed_runs"] = int(
            row.num_completed_runs
        )
        values[f"non_wiki_k{k}_num_full_budget_runs"] = int(
            row.num_full_budget_runs
        )
        values[f"non_wiki_k{k}_full_budget_rate_percent"] = float(
            row.full_budget_rate_percent
        )
        values[f"non_wiki_k{k}_manuscript_full_budget_rate_percent"] = float(
            row.manuscript_full_budget_rate_percent
        )
        values[f"non_wiki_k{k}_max_unused_budget"] = int(
            row.max_unused_budget
        )
        values[f"non_wiki_k{k}_all_completed_b500_full"] = bool(
            row.all_completed_runs_at_b500_use_full_budget
        )

    wiki_b500 = wiki.loc[wiki["budget"] == 500]
    for row in wiki_b500.itertuples(index=False):
        values[f"wiki_b500_k{int(row.k)}_mean_selected_set_size"] = float(
            row.mean_selected_set_size
        )
    return pd.DataFrame([values])


def render_report(
    non_wiki: pd.DataFrame,
    wiki: pd.DataFrame,
    overall: pd.Series,
) -> str:
    """Render an auditable record and manuscript-ready summary."""

    lines = [
        "# Selected-set size: manuscript metrics",
        "",
        "The values below are recomputed from the compact result rows created "
        "from the original deactivation CSVs. Timeout rows are excluded, and "
        "only AIS-Greedy with $k=1,10,100$ is retained.",
        "",
        "## Networks other than wiki-Vote",
        "",
        "| $k$ | Completed runs | Full-budget runs | Full-budget rate | Maximum unused budget | Completed $b=500$ runs all full? |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in non_wiki.itertuples(index=False):
        lines.append(
            f"| {int(row.k)} | {int(row.num_completed_runs)} | "
            f"{int(row.num_full_budget_runs)} | "
            f"{row.full_budget_rate_percent:.4f}% | "
            f"{int(row.max_unused_budget)} | "
            f"{'yes' if row.all_completed_runs_at_b500_use_full_budget else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## wiki-Vote: selected-set size versus budget",
            "",
            "Each entry is the mean over the four parameter settings; the "
            "parenthesized range gives the minimum and maximum.",
            "",
            "| $k$ | $b=100$ | $b=200$ | $b=300$ | $b=400$ | $b=500$ |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for k, group in wiki.groupby("k", sort=True):
        by_budget = {int(row.budget): row for row in group.itertuples(index=False)}
        cells = []
        for budget in sorted(WIKI_REPORT_BUDGETS):
            row = by_budget[budget]
            cells.append(
                f"{_format_size(row.mean_selected_set_size)} "
                f"({int(row.min_selected_set_size)}--{int(row.max_selected_set_size)})"
            )
        lines.append(f"| {int(k)} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Manuscript-ready summary",
            "",
            "Outside wiki-Vote, AIS-Greedy used the full budget in every "
            "completed run for $k=10,100$ and in "
            f"{overall['non_wiki_k1_manuscript_full_budget_rate_percent']:.1f}% "
            "of completed runs for $k=1$; every completed run at $b=500$ "
            "used all 500 nodes. On wiki-Vote, the selected-set size equaled "
            "$b$ for all four settings at $b=100,200,300$, but then "
            "plateaued: its mean at both $b=400$ and $b=500$ was "
            f"{overall['wiki_b500_k1_mean_selected_set_size']:.1f}, "
            f"{overall['wiki_b500_k10_mean_selected_set_size']:.1f}, and "
            f"{overall['wiki_b500_k100_mean_selected_set_size']:.2f} for "
            "$k=1,10,100$, respectively.",
            "",
            "The sequential selections are computational steps of an offline "
            "optimization procedure for determining the final intervention "
            "set; they do not represent interventions applied sequentially "
            "during opinion evolution.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_compact_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} is empty.")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    for column in ["budget", "repetition", "selected_set_size"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["timed_out"] = frame["timed_out"].map(_parse_bool)
    return frame


def _select_and_validate(raw: pd.DataFrame) -> pd.DataFrame:
    datasets = set(raw["dataset"].astype(str))
    if datasets != EXPECTED_DATASETS:
        raise ValueError(
            f"Expected the six paper datasets; found {sorted(datasets)}."
        )
    rows = raw.loc[
        raw["method"].astype(str).isin(METHOD_TO_K) & ~raw["timed_out"]
    ].copy()
    rows["k"] = rows["method"].map(METHOD_TO_K).astype(int)
    keys = [
        "dataset",
        "opinion_setting",
        "alpha_setting",
        "budget",
        "method",
        "repetition",
    ]
    if rows.duplicated(keys).any():
        raise ValueError("Duplicate AIS-Greedy result rows were found.")
    if (
        (rows["budget"] <= 0).any()
        or (rows["selected_set_size"] < 0).any()
        or (rows["selected_set_size"] > rows["budget"]).any()
    ):
        raise ValueError("Invalid budget or selected-set size was found.")
    rows["unused_budget"] = rows["budget"] - rows["selected_set_size"]
    rows["uses_full_budget"] = rows["unused_budget"] == 0
    return rows


def _parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def _round_half_up(value: float, decimal_places: int) -> float:
    quantum = Decimal(1).scaleb(-decimal_places)
    return float(
        Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP)
    )


def _format_size(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Recompute manuscript selected-set-size metrics."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repository_root / "results/additional/selected_set_size",
        help="Selected-set-size results root.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
