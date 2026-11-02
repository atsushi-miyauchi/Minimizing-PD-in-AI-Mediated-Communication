from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polar_ai.experiments.plot_multi_seed_performance import (  # noqa: E402
    plot_multi_seed_performance,
)
from polar_ai.experiments.run_multi_seed_performance import (  # noqa: E402
    aggregate_across_seeds,
    aggregate_over_settings_and_seeds,
    aggregate_within_seed,
    experiment_seeds,
)


class MultiSeedPerformanceTests(unittest.TestCase):
    def _raw_rows(self) -> pd.DataFrame:
        rows = []
        for seed, deterministic_value, random_values in [
            (11, 0.8, [0.7, 0.9]),
            (22, 1.0, [0.8, 1.2]),
        ]:
            for opinion in ["uniform", "polarized"]:
                for alpha in ["homogeneous", "heterogeneous"]:
                    rows.append(
                        self._row(
                            seed,
                            opinion,
                            alpha,
                            "AIS-Greedy (k=1)",
                            -1,
                            deterministic_value,
                        )
                    )
                    for repetition, value in enumerate(random_values):
                        rows.append(
                            self._row(
                                seed,
                                opinion,
                                alpha,
                                "USampling-Greedy (k=1)",
                                repetition,
                                value,
                            )
                        )
                    rows.append(
                        self._row(
                            seed,
                            opinion,
                            alpha,
                            "No-AI-Mediation",
                            -1,
                            0.6,
                        )
                    )
        return pd.DataFrame(rows)

    @staticmethod
    def _row(
        seed: int,
        opinion: str,
        alpha: str,
        method: str,
        repetition: int,
        value: float,
    ) -> dict[str, object]:
        return {
            "dataset": "toy",
            "n": 5,
            "m": 8,
            "seed": seed,
            "opinion_setting": opinion,
            "alpha_setting": alpha,
            "budget": 1,
            "method": method,
            "repetition": repetition,
            "objective": value * 10.0,
            "normalized_objective": value,
            "runtime_sec": value,
            "timed_out": False,
            "selected_set_size": 1,
            "equilibrium_iterations": 2,
            "linear_solver_iterations": 3,
            "feasibility_violation": 0.0,
        }

    def test_repetitions_are_averaged_before_seed_statistics(self) -> None:
        seed_level = aggregate_within_seed(self._raw_rows())
        summary = aggregate_across_seeds(seed_level)

        random_summary = summary[
            (summary["opinion_setting"] == "uniform")
            & (summary["alpha_setting"] == "homogeneous")
            & (summary["method"] == "USampling-Greedy (k=1)")
        ].iloc[0]
        self.assertEqual(int(random_summary["num_seeds"]), 2)
        self.assertAlmostEqual(
            float(random_summary["mean_normalized_objective"]), 0.9
        )
        self.assertAlmostEqual(
            float(random_summary["std_normalized_objective"]),
            0.1 * (2.0**0.5),
        )
        self.assertEqual(float(random_summary["mean_num_repetitions"]), 2.0)

    def test_overall_summary_averages_settings_within_each_seed(self) -> None:
        seed_level = aggregate_within_seed(self._raw_rows())
        overall = aggregate_over_settings_and_seeds(seed_level)
        ais = overall[overall["method"] == "AIS-Greedy (k=1)"].iloc[0]
        self.assertEqual(int(ais["num_seeds"]), 2)
        self.assertEqual(int(ais["min_num_settings_per_seed"]), 4)
        self.assertEqual(int(ais["max_num_settings_per_seed"]), 4)
        self.assertAlmostEqual(float(ais["mean_normalized_objective"]), 0.9)

    def test_publication_plots_are_written(self) -> None:
        seed_level = aggregate_within_seed(self._raw_rows())
        summary = aggregate_across_seeds(seed_level)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.csv"
            summary.to_csv(summary_path, index=False)
            outputs = plot_multi_seed_performance(
                summary_path,
                root / "figures",
                expected_num_seeds=2,
            )
            self.assertEqual(len(outputs), 9)
            self.assertTrue(all(path.exists() for path in outputs))

    def test_seed_list_is_required_and_deduplicated(self) -> None:
        self.assertEqual(experiment_seeds({"seeds": [3, 3, 5]}), [3, 5])
        with self.assertRaises(ValueError):
            experiment_seeds({})


if __name__ == "__main__":
    unittest.main()
