from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polar_ai.experiments.plot_selected_set_size_audit import (  # noqa: E402
    plot_selected_set_size_audit,
)
from polar_ai.experiments.run_selected_set_size_audit import (  # noqa: E402
    aggregate_selected_set_sizes,
    read_and_filter_compact,
    stream_compact_results,
)


class SelectedSetSizeAuditTests(unittest.TestCase):
    def _write_source(self, path: Path) -> None:
        columns = [
            "dataset",
            "n",
            "m",
            "opinion_setting",
            "alpha_setting",
            "budget",
            "method",
            "repetition",
            "objective",
            "normalized_objective",
            "runtime_sec",
            "timed_out",
            "selected_set_size",
            "selected_indices",
            "diagnostics",
        ]
        rows = [
            (1, "Greedy", -1, False, 1),
            (2, "Greedy", -1, False, 1),
            (1, "AIS-Greedy (k=100)", -1, False, 1),
            (2, "AIS-Greedy (k=100)", -1, False, 2),
            (2, "No-AI-Mediation", -1, False, 4),
            (2, "Greedy", -1, True, 0),
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for budget, method, repetition, timed_out, size in rows:
                writer.writerow(
                    {
                        "dataset": "toy",
                        "n": 4,
                        "m": 5,
                        "opinion_setting": "uniform",
                        "alpha_setting": "homogeneous",
                        "budget": budget,
                        "method": method,
                        "repetition": repetition,
                        "objective": 1.0,
                        "normalized_objective": 1.0,
                        "runtime_sec": 0.1,
                        "timed_out": timed_out,
                        "selected_set_size": size,
                        "selected_indices": "[0, 1]",
                        "diagnostics": '{"large": [1, 2, 3], "text": "a,b"}',
                    }
                )

    def test_streaming_prefix_resume_and_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "deactivation.csv"
            compact = root / "selected_set_size_rows.csv"
            self._write_source(source)

            self.assertEqual(
                stream_compact_results(
                    source,
                    compact,
                    expected_dataset="toy",
                    checkpoint_every=1,
                    progress_every=0,
                ),
                6,
            )
            self.assertEqual(
                stream_compact_results(
                    source,
                    compact,
                    expected_dataset="toy",
                    checkpoint_every=1,
                    progress_every=0,
                ),
                0,
            )
            rows = read_and_filter_compact(
                compact,
                {
                    "exclude_methods": ["No-AI-Mediation"],
                    "exclude_timed_out": True,
                },
            )
            self.assertEqual(len(rows), 4)
            by_budget, summary = aggregate_selected_set_sizes(rows)
            greedy_two = by_budget[
                (by_budget["method"] == "Greedy")
                & (by_budget["budget"] == 2)
            ].iloc[0]
            self.assertEqual(float(greedy_two["mean_selected_fraction"]), 0.5)
            self.assertEqual(float(greedy_two["full_budget_rate"]), 0.0)
            self.assertEqual(len(summary), 2)

    def test_publication_plots_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = []
            for method in ["AIS-Greedy (k=100)", "Greedy"]:
                for budget in [1, 2, 3]:
                    rows.append(
                        {
                            "dataset": "toy",
                            "method": method,
                            "budget": budget,
                            "mean_selected_fraction": 0.8,
                            "std_selected_fraction": 0.1,
                            "full_budget_rate": 0.75,
                        }
                    )
            summary_path = root / "by_budget.csv"
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            outputs = plot_selected_set_size_audit(
                summary_path,
                root / "figures",
                methods=["AIS-Greedy (k=100)", "Greedy"],
            )
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.exists() for path in outputs))


if __name__ == "__main__":
    unittest.main()
