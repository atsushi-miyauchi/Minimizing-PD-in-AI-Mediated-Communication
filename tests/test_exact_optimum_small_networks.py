from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polar_ai.experiments.plot_exact_optimum_small_networks import (  # noqa: E402
    plot_exact_optimum_small_networks,
    write_exact_optimum_summary,
)
from polar_ai.experiments.run_exact_optimum_small_networks import (  # noqa: E402
    RAW_COLUMNS,
    combination_at_rank,
    compact_budget_result,
    compact_checkpoint_path,
    comparison_row,
    enumerate_exact_subsets,
    greedy_budget_results,
    iter_subsets,
    load_compact_checkpoint,
    migrate_legacy_raw_checkpoint,
    run_compact_exact_search,
    select_optimal_row,
)
from polar_ai.experiments.utils import DatasetConfig  # noqa: E402
from polar_ai.graph_io import build_graph_matrices  # noqa: E402
from polar_ai.instance import Instance  # noqa: E402
from polar_ai.spectral import homogeneous_lambda_vector  # noqa: E402


class ExactOptimumSmallNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_graph_matrices(
            [
                ("a", "b", 1.0),
                ("b", "c", 1.0),
                ("c", "d", 1.0),
                ("d", "a", 1.0),
                ("a", "c", 1.0),
            ],
            directed=False,
        )
        alpha = np.full(4, 2.0)
        self.instance = Instance(
            W=self.graph.W,
            P=self.graph.P,
            Wbar=self.graph.Wbar,
            s=np.array([-0.8, -0.2, 0.3, 0.7]),
            lambda_vec=homogeneous_lambda_vector(self.graph.P, alpha),
            alpha=alpha,
            metadata={"node_labels": list(self.graph.nodes)},
        )
        self.dataset = DatasetConfig(name="toy", path=Path("toy.edges"))

    def test_iter_subsets_includes_all_smaller_sets(self) -> None:
        subsets = list(iter_subsets(4, 2))
        self.assertEqual(len(subsets), 1 + 4 + 6)
        self.assertEqual(subsets[0], ())
        self.assertIn((0, 3), subsets)

    def test_combination_unranking_matches_itertools_order(self) -> None:
        expected = list(iter_subsets(6, 3))[-20:]
        size_three = [subset for subset in expected if len(subset) == 3]
        for rank, subset in enumerate(size_three):
            self.assertEqual(combination_at_rank(6, 3, rank), subset)

    def test_compact_search_is_exact_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint.json"
            written = run_compact_exact_search(
                self.instance,
                dataset=self.dataset,
                opinion_setting="uniform",
                alpha_setting="homogeneous",
                seed=1,
                backend="python",
                max_budget=2,
                checkpoint_path=checkpoint,
                eq_tol=1e-8,
                eq_max_iter=1000,
                checkpoint_every=2,
                progress_every=0,
            )
            self.assertEqual(written, 11)
            state = load_compact_checkpoint(checkpoint)
            self.assertTrue(state["complete"])
            result = compact_budget_result(state, 2, self.instance.n)
            self.assertEqual(result["num_evaluations"], 11)
            self.assertEqual(
                run_compact_exact_search(
                    self.instance,
                    dataset=self.dataset,
                    opinion_setting="uniform",
                    alpha_setting="homogeneous",
                    seed=1,
                    backend="python",
                    max_budget=2,
                    checkpoint_path=checkpoint,
                    eq_tol=1e-8,
                    eq_max_iter=1000,
                    checkpoint_every=2,
                    progress_every=0,
                ),
                0,
            )

    def test_exhaustive_rows_and_greedy_are_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw.csv"
            completed: set[tuple[int, str, str, tuple[int, ...]]] = set()
            with raw_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
                writer.writeheader()
                written = enumerate_exact_subsets(
                    self.instance,
                    dataset=self.dataset,
                    opinion_setting="uniform",
                    alpha_setting="homogeneous",
                    seed=1,
                    backend="python",
                    max_budget=2,
                    completed_keys=completed,
                    writer=writer,
                    handle=handle,
                    eq_tol=1e-8,
                    eq_max_iter=1000,
                    checkpoint_every=2,
                    progress_every=0,
                )
            frame = pd.read_csv(raw_path)
            self.assertEqual(written, 11)
            self.assertEqual(len(frame), 11)
            optimum = select_optimal_row(frame[frame["subset_size"] <= 2])
            greedy = greedy_budget_results(
                self.instance, [1, 2], eq_tol=1e-8, eq_max_iter=1000
            )
            self.assertGreaterEqual(
                greedy[2]["objective"] + 1e-10, float(optimum["objective"])
            )
            self.assertLessEqual(len(greedy[2]["selected_indices"]), 2)

            checkpoint_dir = Path(tmpdir) / "checkpoints"
            migrate_legacy_raw_checkpoint(
                raw_path,
                checkpoint_dir,
                dataset=self.dataset,
                n=self.instance.n,
                max_budget=2,
                seeds=[1],
                settings=[("uniform", "homogeneous")],
                progress_every=0,
            )
            migrated = load_compact_checkpoint(
                compact_checkpoint_path(
                    checkpoint_dir,
                    seed=1,
                    opinion_setting="uniform",
                    alpha_setting="homogeneous",
                )
            )
            migrated_result = compact_budget_result(
                migrated, 2, self.instance.n
            )
            self.assertAlmostEqual(
                migrated_result["best_objective"],
                float(optimum["objective"]),
            )

    def test_comparison_plots_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = []
            for budget in [1, 2, 3]:
                row = comparison_row(
                    self.instance,
                    dataset="toy",
                    opinion_setting="uniform",
                    alpha_setting="homogeneous",
                    seed=1,
                    backend="python",
                    budget=budget,
                    baseline_objective=10.0,
                    optimal_objective=10.0 - budget,
                    greedy_objective=10.0 - 0.9 * budget,
                    optimal_indices=tuple(range(budget)),
                    greedy_indices=tuple(range(budget)),
                    exact_num_evaluations=budget + 1,
                    greedy_num_evaluations=budget + 1,
                    exact_equilibrium_iterations=10,
                    greedy_equilibrium_iterations=10,
                    exact_runtime_sec=1.0,
                    greedy_runtime_sec=0.1,
                )
                rows.append(row)
            comparison_path = root / "comparison.csv"
            pd.DataFrame(rows).to_csv(comparison_path, index=False)
            summary_path = root / "summary.csv"
            write_exact_optimum_summary(comparison_path, summary_path)
            outputs = plot_exact_optimum_small_networks(
                comparison_path, root / "figures"
            )
            self.assertTrue(summary_path.exists())
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.exists() for path in outputs))


if __name__ == "__main__":
    unittest.main()
