from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polar_ai.experiments import plot_results, run_deactivation, run_suite  # noqa: E402
from polar_ai.experiments.utils import deactivation_marker_budgets  # noqa: E402


class ExperimentScriptTests(unittest.TestCase):
    def test_deactivation_marker_budgets_use_tenths_of_maximum(self) -> None:
        self.assertEqual(
            deactivation_marker_budgets(100),
            [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        )
        self.assertEqual(deactivation_marker_budgets(5), [1, 2, 3, 4, 5])

    def test_only_current_method_parameters_are_read(self) -> None:
        config = {
            "usampling_greedy_k_values": [1, 10],
            "neighbor_pd_greedy_k_values": [2, 20],
            "ais_greedy_k_values": [3, 30],
            "random_repetitions": 4,
        }
        self.assertEqual(
            run_deactivation._usampling_greedy_k_values(config),
            [1, 10],
        )
        self.assertEqual(
            run_deactivation._neighbor_pd_greedy_k_values(config),
            [2, 20],
        )
        self.assertEqual(
            run_deactivation._ais_greedy_k_values(config),
            [3, 30],
        )
        self.assertEqual(
            run_deactivation._usampling_random_repetitions(config),
            4,
        )

    def test_suite_runs_only_two_paper_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "results"
            config = {
                "results_dir": str(output_root),
                "datasets": [{"name": "sample", "path": "sample.edges"}],
            }
            dataset = SimpleNamespace(name="sample")
            with (
                mock.patch.object(
                    run_suite,
                    "load_config",
                    return_value=(config, Path(tmpdir)),
                ),
                mock.patch.object(
                    run_suite,
                    "dataset_configs",
                    return_value=[dataset],
                ),
                mock.patch.object(
                    run_suite.run_transformation_effect,
                    "run_config",
                ) as transformation,
                mock.patch.object(
                    run_suite.run_deactivation,
                    "run_config",
                ) as deactivation,
                mock.patch.object(
                    run_suite.plot_results,
                    "plot_deactivation_and_transformation_results",
                    return_value=[],
                ) as plots,
            ):
                result = run_suite.main(["--config", "config.yaml"])

        self.assertEqual(result, 0)
        transformation.assert_called_once()
        deactivation.assert_called_once()
        plots.assert_called_once()

    def test_deactivation_plot_marks_publication_budgets(self) -> None:
        df = pd.DataFrame(
            {
                "budget": range(1, 101),
                "method": ["Greedy"] * 100,
                "normalized_objective": [1.0] * 100,
            }
        )
        fig, ax = plot_results.plt.subplots()
        try:
            plot_results._plot_grouped_methods(
                ax,
                df,
                value="normalized_objective",
                order=["Greedy"],
                method_column="method",
                timeout_sec=1200.0,
                marker_at_deactivation_budgets=True,
            )
            self.assertEqual(
                ax.lines[0].get_markevery(),
                [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99],
            )
        finally:
            plot_results.plt.close(fig)

    def test_plot_order_contains_only_current_families(self) -> None:
        labels = pd.Series(
            [
                "AIS-Greedy (k=10)",
                "Greedy",
                "NeighborPD-Greedy (k=10)",
                "USampling-Greedy (k=10)",
                "No-AI-Mediation",
            ]
        )
        self.assertEqual(
            plot_results._extend_method_order(
                plot_results.DEACTIVATION_ORDER,
                labels,
            ),
            [
                "USampling-Greedy (k=10)",
                "NeighborPD-Greedy (k=10)",
                "AIS-Greedy (k=10)",
                "Greedy",
            ],
        )

    def test_repetition_count_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            run_deactivation._usampling_random_repetitions(
                {"random_repetitions": 0}
            )


if __name__ == "__main__":
    unittest.main()
