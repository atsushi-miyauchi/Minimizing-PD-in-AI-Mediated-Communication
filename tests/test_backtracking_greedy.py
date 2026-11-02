from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polar_ai.experiments.plot_backtracking_greedy import (
    plot_backtracking_greedy,
    write_backtracking_summary,
)  # noqa: E402
from polar_ai.experiments.run_backtracking_greedy import (
    backtracking_greedy,
)  # noqa: E402
from polar_ai.graph_io import build_graph_matrices  # noqa: E402
from polar_ai.instance import Instance  # noqa: E402


class BacktrackingGreedyTests(unittest.TestCase):
    def setUp(self) -> None:
        graph = build_graph_matrices(
            [("a", "b", 1.0), ("b", "c", 1.0), ("c", "d", 1.0)],
            directed=False,
        )
        self.instance = Instance(
            W=graph.W,
            P=graph.P,
            Wbar=graph.Wbar,
            s=np.array([-0.8, -0.2, 0.3, 0.7]),
            lambda_vec=np.full(4, 0.25),
            alpha=np.full(4, 2.0),
            metadata={"node_labels": list(graph.nodes)},
        )

    def test_replacement_then_nonimproving_addition_locks_search(self) -> None:
        objectives = {
            frozenset(): 10.0,
            frozenset({0}): 5.0,
            frozenset({1}): 6.0,
            frozenset({2}): 7.0,
            frozenset({3}): 8.0,
            frozenset({0, 1}): 4.0,
            frozenset({0, 2}): 6.0,
            frozenset({0, 3}): 7.0,
            frozenset({1, 2}): 3.0,
            frozenset({1, 3}): 7.0,
            frozenset({2, 3}): 9.0,
            frozenset({0, 1, 2}): 5.0,
            frozenset({0, 1, 3}): 6.0,
            frozenset({1, 2, 3}): 4.0,
        }

        def fake_evaluate(instance, selected, **kwargs):
            del kwargs
            key = frozenset(selected)
            objective = objectives[key]
            return (
                objective,
                np.full(instance.n, objective),
                {
                    "equilibrium_converged": True,
                    "equilibrium_iterations": 1,
                },
            )

        with patch(
            "polar_ai.experiments.run_backtracking_greedy.evaluate_deactivation",
            side_effect=fake_evaluate,
        ):
            result = backtracking_greedy(self.instance, 3)

        self.assertEqual(result.selected, frozenset({1, 2}))
        self.assertEqual(result.objective, 3.0)
        self.assertEqual(result.accepted_additions, 3)
        self.assertEqual(result.accepted_deletions, 0)
        self.assertEqual(result.accepted_replacements, 1)
        self.assertEqual(result.forced_additions, 1)
        self.assertTrue(result.switched_to_addition_only)
        self.assertEqual(result.switch_move, 4)
        self.assertEqual(result.path_length, 4)
        self.assertEqual(result.terminal_set_size, 3)

    def test_zero_budget_returns_baseline(self) -> None:
        result = backtracking_greedy(
            self.instance,
            0,
            eq_tol=1.0e-8,
            eq_max_iter=1000,
        )
        self.assertEqual(result.selected, frozenset())
        self.assertEqual(result.num_evaluations, 1)
        self.assertEqual(result.path_length, 0)
        self.assertFalse(result.switched_to_addition_only)

    def test_summary_and_plots_are_written(self) -> None:
        rows = []
        for budget in [1, 2]:
            for setting in range(2):
                rows.append(
                    {
                        "dataset": "toy",
                        "budget": budget,
                        "greedy_normalized_objective": 1.0 - 0.1 * budget,
                        "backtracking_normalized_objective": (
                            1.0 - 0.11 * budget - 0.001 * setting
                        ),
                        "normalized_gain_over_greedy": (
                            0.01 * budget + 0.001 * setting
                        ),
                        "backtracking_beats_greedy": True,
                        "objectives_equal": False,
                        "sets_equal": False,
                        "greedy_set_size": budget,
                        "backtracking_set_size": budget,
                        "accepted_deletions": 0,
                        "accepted_replacements": 1,
                        "switched_to_addition_only": True,
                        "greedy_runtime_sec": 0.1,
                        "backtracking_runtime_sec": 0.2,
                    }
                )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = root / "raw.csv"
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            summary_path = root / "summary.csv"
            write_backtracking_summary(raw_path, summary_path)
            outputs = plot_backtracking_greedy(raw_path, root / "figures")
            summary = pd.read_csv(summary_path)
            self.assertEqual(len(summary), 2)
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.exists() for path in outputs))


if __name__ == "__main__":
    unittest.main()
