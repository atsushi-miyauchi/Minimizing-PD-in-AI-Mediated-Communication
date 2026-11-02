"""Audit incumbent selected-set sizes from existing deactivation CSV files.

The original result files can be very large because their final diagnostics
column stores detailed JSON.  This module therefore streams only the simple
prefix ending at ``selected_set_size`` and writes a compact, resumable CSV.
No optimization algorithm or equilibrium computation is rerun.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np
import pandas as pd

from polar_ai.experiments.plot_selected_set_size_audit import (
    plot_selected_set_size_audit,
)
from polar_ai.experiments.utils import load_config, resolve_path


COMPACT_COLUMNS = [
    "experiment",
    "dataset",
    "n",
    "m",
    "opinion_setting",
    "alpha_setting",
    "budget",
    "method",
    "repetition",
    "timed_out",
    "selected_set_size",
    "selected_fraction",
    "unused_budget",
    "uses_full_budget",
    "source_row_number",
    "source_next_offset",
]

BY_BUDGET_COLUMNS = [
    "dataset",
    "method",
    "budget",
    "num_rows",
    "mean_selected_set_size",
    "std_selected_set_size",
    "min_selected_set_size",
    "max_selected_set_size",
    "mean_selected_fraction",
    "std_selected_fraction",
    "min_selected_fraction",
    "full_budget_count",
    "full_budget_rate",
    "mean_unused_budget",
    "max_unused_budget",
]

SUMMARY_COLUMNS = [
    "dataset",
    "method",
    "num_rows",
    "num_parameter_settings",
    "min_budget",
    "max_budget",
    "full_budget_count",
    "full_budget_rate",
    "mean_selected_fraction",
    "min_selected_fraction",
    "mean_unused_budget",
    "max_unused_budget",
    "final_budget",
    "final_num_rows",
    "final_mean_selected_set_size",
    "final_mean_selected_fraction",
    "final_min_selected_fraction",
    "final_full_budget_count",
    "final_full_budget_rate",
]

REQUIRED_SOURCE_COLUMNS = [
    "dataset",
    "n",
    "m",
    "opinion_setting",
    "alpha_setting",
    "budget",
    "method",
    "repetition",
    "timed_out",
    "selected_set_size",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config, base_dir = load_config(args.config)
    output_root = resolve_path(
        args.results_dir
        or config.get("results_dir", "../results/additional/selected_set_size"),
        base_dir,
    )
    selected_datasets = set(args.dataset or [])
    entries = input_entries(config, base_dir)
    matched = False
    combined_by_budget: list[pd.DataFrame] = []
    combined_summary: list[pd.DataFrame] = []

    for dataset, source_path in entries:
        if selected_datasets and dataset not in selected_datasets:
            continue
        matched = True
        dataset_dir = output_root / dataset
        compact_path = dataset_dir / "selected_set_size_rows.csv"
        by_budget_path = dataset_dir / "selected_set_size_by_budget.csv"
        summary_path = dataset_dir / "selected_set_size_summary.csv"
        figures_dir = dataset_dir / "figures"

        if not args.plot_only:
            if not bool(config.get("resume", True)) and compact_path.exists():
                compact_path.unlink()
            stream_compact_results(
                source_path,
                compact_path,
                expected_dataset=dataset,
                checkpoint_every=int(config.get("checkpoint_every", 1000)),
                progress_every=int(config.get("progress_every", 5000)),
            )

        rows = read_and_filter_compact(compact_path, config)
        by_budget, summary = aggregate_selected_set_sizes(rows)
        by_budget.to_csv(by_budget_path, index=False)
        summary.to_csv(summary_path, index=False)
        combined_by_budget.append(by_budget)
        combined_summary.append(summary)
        print(by_budget_path, flush=True)
        print(summary_path, flush=True)

        if bool(config.get("generate_plots", True)):
            outputs = plot_selected_set_size_audit(
                by_budget_path,
                figures_dir,
                methods=[str(value) for value in config.get("plot_methods", [])],
            )
            for path in outputs:
                print(path, flush=True)

    if selected_datasets and not matched:
        raise ValueError(
            "None of the requested datasets occur in the config: "
            + ", ".join(sorted(selected_datasets))
        )
    if combined_by_budget:
        output_root.mkdir(parents=True, exist_ok=True)
        pd.concat(combined_by_budget, ignore_index=True).to_csv(
            output_root / "selected_set_size_by_budget.csv", index=False
        )
        pd.concat(combined_summary, ignore_index=True).to_csv(
            output_root / "selected_set_size_summary.csv", index=False
        )
    return 0


def stream_compact_results(
    source_path: Path,
    output_path: Path,
    *,
    expected_dataset: str,
    checkpoint_every: int = 1000,
    progress_every: int = 5000,
) -> int:
    """Stream the source prefix into a compact, resumable CSV."""

    if not source_path.exists():
        raise FileNotFoundError(f"Missing source deactivation CSV: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resume_offset, resume_row = compact_checkpoint(output_path)
    mode = "a" if resume_offset > 0 else "w"
    written = 0

    with source_path.open("rb") as source, output_path.open(
        mode, newline="", encoding="utf-8"
    ) as output:
        header_line = source.readline()
        if not header_line:
            raise ValueError(f"Source CSV is empty: {source_path}")
        header = header_line.decode("utf-8").rstrip("\r\n").split(",")
        indices = source_column_indices(header)
        maximum_index = max(indices.values())

        writer = csv.DictWriter(output, fieldnames=COMPACT_COLUMNS)
        if mode == "w":
            writer.writeheader()
            output.flush()
        else:
            if resume_offset < source.tell() or resume_offset > source_path.stat().st_size:
                raise ValueError(
                    f"Invalid resume offset {resume_offset} for {source_path}."
                )
            source.seek(resume_offset)

        source_row = resume_row
        for line in source:
            source_row += 1
            values = parse_simple_prefix(line, maximum_index + 1)
            row = {name: values[index] for name, index in indices.items()}
            dataset = str(row["dataset"])
            if dataset != expected_dataset:
                raise ValueError(
                    f"Expected dataset {expected_dataset!r}, found {dataset!r} "
                    f"at source row {source_row}."
                )
            budget = int(row["budget"])
            selected_size = int(row["selected_set_size"])
            if budget < 0 or selected_size < 0:
                raise ValueError(
                    f"Negative budget or selected_set_size at source row {source_row}."
                )
            unused_budget = budget - selected_size
            writer.writerow(
                {
                    "experiment": "selected_set_size_audit",
                    "dataset": dataset,
                    "n": int(row["n"]),
                    "m": int(row["m"]),
                    "opinion_setting": str(row["opinion_setting"]),
                    "alpha_setting": str(row["alpha_setting"]),
                    "budget": budget,
                    "method": str(row["method"]),
                    "repetition": int(row["repetition"]),
                    "timed_out": parse_bool(row["timed_out"]),
                    "selected_set_size": selected_size,
                    "selected_fraction": (
                        float(selected_size / budget) if budget > 0 else np.nan
                    ),
                    "unused_budget": unused_budget,
                    "uses_full_budget": selected_size == budget,
                    "source_row_number": source_row,
                    "source_next_offset": source.tell(),
                }
            )
            written += 1
            if checkpoint_every > 0 and written % checkpoint_every == 0:
                output.flush()
            if progress_every > 0 and source_row % progress_every == 0:
                print(
                    "[selected-set-size] "
                    f"dataset={expected_dataset} source_rows={source_row}",
                    flush=True,
                )
        output.flush()

    print(
        "[selected-set-size] "
        f"dataset={expected_dataset} newly_compacted={written} "
        f"output={output_path}",
        flush=True,
    )
    return written


def parse_simple_prefix(line: bytes, count: int) -> list[str]:
    """Parse ``count`` leading comma-free CSV fields without copying the tail."""

    values: list[str] = []
    start = 0
    for field_index in range(int(count)):
        comma = line.find(b",", start)
        if comma < 0:
            if field_index != count - 1:
                raise ValueError("CSV row ended before the required prefix.")
            comma = len(line)
        raw = line[start:comma].strip()
        if raw.startswith(b'"') or raw.endswith(b'"'):
            raw = raw.strip(b'"')
        values.append(raw.decode("utf-8"))
        start = comma + 1
    return values


def source_column_indices(header: list[str]) -> dict[str, int]:
    """Validate that required fields form a simple prefix of the large CSV."""

    missing = set(REQUIRED_SOURCE_COLUMNS) - set(header)
    if missing:
        raise ValueError(f"Source CSV is missing columns: {sorted(missing)}")
    indices = {name: header.index(name) for name in REQUIRED_SOURCE_COLUMNS}
    if max(indices.values()) >= header.index("selected_indices"):
        raise ValueError(
            "Required selected-set fields must precede selected_indices."
        )
    return indices


def compact_checkpoint(path: Path) -> tuple[int, int]:
    """Return the source byte offset and row number from the last compact row."""

    if not path.exists() or path.stat().st_size == 0:
        return 0, 0
    last: dict[str, str] | None = None
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            last = row
    if last is None:
        return 0, 0
    return int(last["source_next_offset"]), int(last["source_row_number"])


def read_and_filter_compact(
    path: Path, config: dict[str, Any]
) -> pd.DataFrame:
    """Read compact rows and apply method/timeout filters from the config."""

    if not path.exists():
        raise FileNotFoundError(f"Missing compact selected-set CSV: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Compact selected-set CSV is empty: {path}")
    excluded = {str(value) for value in config.get("exclude_methods", [])}
    included = {str(value) for value in config.get("include_methods", [])}
    if excluded:
        frame = frame[~frame["method"].astype(str).isin(excluded)]
    if included:
        frame = frame[frame["method"].astype(str).isin(included)]
    if bool(config.get("exclude_timed_out", True)):
        timed_out = frame["timed_out"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
        frame = frame[~timed_out]
    if frame.empty:
        raise ValueError("No selected-set rows remain after filtering.")
    numeric = [
        "budget",
        "selected_set_size",
        "selected_fraction",
        "unused_budget",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["uses_full_budget"] = (
        frame["selected_set_size"] == frame["budget"]
    )
    return frame


def aggregate_selected_set_sizes(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return budget-level and method-level selected-size summaries."""

    budget_rows: list[dict[str, Any]] = []
    for (dataset, method, budget), group in rows.groupby(
        ["dataset", "method", "budget"], sort=True
    ):
        budget_rows.append(
            {
                "dataset": str(dataset),
                "method": str(method),
                "budget": int(budget),
                "num_rows": int(len(group)),
                "mean_selected_set_size": float(
                    group["selected_set_size"].mean()
                ),
                "std_selected_set_size": float(
                    group["selected_set_size"].std(ddof=0)
                ),
                "min_selected_set_size": int(
                    group["selected_set_size"].min()
                ),
                "max_selected_set_size": int(
                    group["selected_set_size"].max()
                ),
                "mean_selected_fraction": float(
                    group["selected_fraction"].mean()
                ),
                "std_selected_fraction": float(
                    group["selected_fraction"].std(ddof=0)
                ),
                "min_selected_fraction": float(
                    group["selected_fraction"].min()
                ),
                "full_budget_count": int(group["uses_full_budget"].sum()),
                "full_budget_rate": float(group["uses_full_budget"].mean()),
                "mean_unused_budget": float(group["unused_budget"].mean()),
                "max_unused_budget": int(group["unused_budget"].max()),
            }
        )
    by_budget = pd.DataFrame(budget_rows, columns=BY_BUDGET_COLUMNS)

    summary_rows: list[dict[str, Any]] = []
    for (dataset, method), group in rows.groupby(
        ["dataset", "method"], sort=True
    ):
        final_budget = int(group["budget"].max())
        final = group[group["budget"] == final_budget]
        settings = group[["opinion_setting", "alpha_setting"]].drop_duplicates()
        summary_rows.append(
            {
                "dataset": str(dataset),
                "method": str(method),
                "num_rows": int(len(group)),
                "num_parameter_settings": int(len(settings)),
                "min_budget": int(group["budget"].min()),
                "max_budget": int(group["budget"].max()),
                "full_budget_count": int(group["uses_full_budget"].sum()),
                "full_budget_rate": float(group["uses_full_budget"].mean()),
                "mean_selected_fraction": float(
                    group["selected_fraction"].mean()
                ),
                "min_selected_fraction": float(
                    group["selected_fraction"].min()
                ),
                "mean_unused_budget": float(group["unused_budget"].mean()),
                "max_unused_budget": int(group["unused_budget"].max()),
                "final_budget": final_budget,
                "final_num_rows": int(len(final)),
                "final_mean_selected_set_size": float(
                    final["selected_set_size"].mean()
                ),
                "final_mean_selected_fraction": float(
                    final["selected_fraction"].mean()
                ),
                "final_min_selected_fraction": float(
                    final["selected_fraction"].min()
                ),
                "final_full_budget_count": int(
                    final["uses_full_budget"].sum()
                ),
                "final_full_budget_rate": float(
                    final["uses_full_budget"].mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    return by_budget, summary


def input_entries(
    config: dict[str, Any], base_dir: Path
) -> list[tuple[str, Path]]:
    """Parse configured existing-result inputs."""

    entries = config.get("inputs", [])
    if not entries:
        raise ValueError("Config must contain at least one input entry.")
    parsed: list[tuple[str, Path]] = []
    for entry in entries:
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError("Each input must provide dataset and path.")
        path = resolve_path(entry["path"], base_dir)
        parsed.append((str(entry.get("dataset", path.parent.name)), path))
    return parsed


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit selected-set sizes from existing deactivation CSVs."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name to process; repeat to select multiple datasets.",
    )
    parser.add_argument(
        "--results-dir", default=None, help="Optional results root override."
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate summaries and plots from compact CSVs.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
