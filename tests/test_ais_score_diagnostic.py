from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polar_ai.alpha_generation import generate_homogeneous_alpha  # noqa: E402
from polar_ai.experiments.plot_ais_score_diagnostic import (  # noqa: E402
    plot_ais_score_diagnostics,
    write_ais_score_summary,
)
from polar_ai.experiments.run_ais_score_diagnostic import (  # noqa: E402
    diagnose_ais_trajectory,
)
from polar_ai.graph_io import build_graph_matrices  # noqa: E402
from polar_ai.instance import Instance  # noqa: E402
from polar_ai.spectral import homogeneous_lambda_vector  # noqa: E402


class AISScoreDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        graph = build_graph_matrices(
            [
                ("a", "b", 1.0),
                ("b", "c", 1.0),
                ("c", "d", 1.0),
                ("d", "a", 1.0),
                ("a", "c", 0.5),
                ("b", "d", 0.5),
            ],
            directed=True,
        )
        alpha = generate_homogeneous_alpha(graph.P.shape[0], value=2.0)
        self.instance = Instance(
            W=graph.W,
            P=graph.P,
            Wbar=graph.Wbar,
            s=np.array([-0.8, -0.2, 0.3, 0.7]),
            lambda_vec=homogeneous_lambda_vector(graph.P, alpha),
            alpha=alpha,
            metadata={"node_labels": list(graph.nodes)},
        )

    def test_full_shortlist_always_hits_exact_best(self) -> None:
        rows = list(
            diagnose_ais_trajectory(
                self.instance,
                dataset="toy",
                opinion_setting="uniform",
                alpha_setting="homogeneous",
                seed=1,
                backend="python",
                k=self.instance.n,
                budget=2,
                eq_tol=1e-6,
                eq_max_iter=1000,
                linear_tol=1e-6,
                linear_max_iter=1000,
                solver="bicgstab",
            )
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [row["num_remaining_candidates"] for row in rows], [4, 3]
        )
        for row in rows:
            self.assertTrue(row["hit_at_k"])
            self.assertEqual(row["selected_node"], row["exact_best_node"])
            self.assertEqual(
                row["selected_objective"], row["exact_best_objective"]
            )
            self.assertGreaterEqual(row["exact_best_ais_rank"], 1)
            self.assertLessEqual(
                row["exact_best_ais_rank"], row["num_remaining_candidates"]
            )

    def test_hit_uses_the_trajectory_specific_k(self) -> None:
        row = next(
            diagnose_ais_trajectory(
                self.instance,
                dataset="toy",
                opinion_setting="uniform",
                alpha_setting="homogeneous",
                seed=1,
                backend="python",
                k=1,
                budget=1,
                eq_tol=1e-6,
                eq_max_iter=1000,
                linear_tol=1e-6,
                linear_max_iter=1000,
                solver="bicgstab",
            )
        )
        self.assertEqual(
            row["hit_at_k"], row["exact_best_ais_rank"] <= row["k"]
        )
        self.assertEqual(row["selected_node_ais_rank"], 1)

    def test_summary_and_publication_style_plots_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "ais_score_diagnostic.csv"
            rows = []
            for k in [1, 10, 100]:
                for iteration in [1, 2, 3]:
                    rows.append(
                        {
                            "dataset": "toy",
                            "opinion_setting": "uniform",
                            "alpha_setting": "homogeneous",
                            "k": k,
                            "iteration": iteration,
                            "full_candidate_spearman": 0.5,
                            "exact_best_ais_rank": min(k, iteration),
                            "hit_at_k": True,
                        }
                    )
            pd.DataFrame(rows).to_csv(csv_path, index=False)

            summary_path = root / "summary.csv"
            self.assertEqual(
                write_ais_score_summary(csv_path, summary_path), summary_path
            )
            outputs = plot_ais_score_diagnostics(csv_path, root / "figures")

            self.assertTrue(summary_path.exists())
            self.assertEqual(len(outputs), 6)
            self.assertTrue(all(path.exists() for path in outputs))


if __name__ == "__main__":
    unittest.main()
