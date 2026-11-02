from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polar_ai.algorithms import DEACTIVATION_ALGORITHMS, deactivation  # noqa: E402
from polar_ai.algorithms.common import (  # noqa: E402
    alpha_for_deactivation,
    evaluate_deactivation,
    neighbor_pd_greedy_scores,
    score_exact_objective_spearman_correlation,
)
from polar_ai.alpha_generation import generate_homogeneous_alpha  # noqa: E402
from polar_ai.graph_io import build_graph_matrices  # noqa: E402
from polar_ai.instance import Instance  # noqa: E402
from polar_ai.objective import pd_gradient  # noqa: E402
from polar_ai.spectral import homogeneous_lambda_vector  # noqa: E402


class AlgorithmTests(unittest.TestCase):
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
            metadata={},
        )

    def test_registry_contains_only_executed_methods(self) -> None:
        self.assertEqual(
            set(DEACTIVATION_ALGORITHMS),
            {
                "No-Intervention",
                "USampling-Greedy",
                "NeighborPD-Greedy",
                "AIS-Greedy",
                "Greedy",
            },
        )

    def test_alpha_for_deactivation(self) -> None:
        alpha = np.array([1.5, 2.0, 3.0])
        np.testing.assert_allclose(
            alpha_for_deactivation(alpha, {0, 2}),
            np.array([1.0, 2.0, 1.0]),
        )

    def test_current_algorithms_respect_budget(self) -> None:
        budget = 2
        results = [
            deactivation.no_intervention(self.instance),
            deactivation.usampling_greedy(
                self.instance, budget, k=2, seed=3
            ),
            deactivation.neighbor_pd_greedy(
                self.instance, budget, k=2
            ),
            deactivation.ais_greedy(
                self.instance, budget, k=2, linear_max_iter=50
            ),
            deactivation.greedy(self.instance, budget),
        ]
        for result in results:
            self.assertLessEqual(len(result.solution), budget)
            self.assertTrue(np.isfinite(result.objective))

    def test_usampling_is_deterministic_for_fixed_seed(self) -> None:
        first = deactivation.usampling_greedy(
            self.instance, 3, k=2, seed=17
        )
        second = deactivation.usampling_greedy(
            self.instance, 3, k=2, seed=17
        )
        self.assertEqual(
            [row["selected_node"] for row in first.logs[1:]],
            [row["selected_node"] for row in second.logs[1:]],
        )
        self.assertEqual(first.solution, second.solution)

    def test_ais_score_matches_identity_jacobian_formula(self) -> None:
        _, z_star, _ = evaluate_deactivation(self.instance, set())
        result = deactivation.approximate_improvement_scores(
            self.instance,
            z_star,
            set(),
            linear_max_iter=50,
        )
        b_matrix = (
            np.diag(self.instance.lambda_vec) @ self.instance.P.toarray()
        )
        gradient = pd_gradient(z_star, self.instance.Wbar)
        adjoint = np.linalg.solve(
            (np.eye(self.instance.n) - b_matrix).T,
            gradient,
        )
        expected = -result.finite_gaps * (b_matrix.T @ adjoint)
        np.testing.assert_allclose(result.scores, expected, atol=1e-7)
        self.assertEqual(result.diagnostics["score_variant"], "ais_greedy")

    def test_neighbor_pd_scores_are_finite(self) -> None:
        _, z_star, _ = evaluate_deactivation(self.instance, set())
        scores = neighbor_pd_greedy_scores(self.instance, z_star)
        self.assertEqual(scores.shape, (self.instance.n,))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(scores >= 0.0))

    def test_spearman_correlation_uses_best_to_worst_ranks(self) -> None:
        scores = np.array([3.0, 2.0, 1.0])
        self.assertAlmostEqual(
            score_exact_objective_spearman_correlation(
                scores,
                np.array([1.0, 2.0, 3.0]),
            ),
            1.0,
        )
        self.assertAlmostEqual(
            score_exact_objective_spearman_correlation(
                scores,
                np.array([3.0, 2.0, 1.0]),
            ),
            -1.0,
        )


if __name__ == "__main__":
    unittest.main()
