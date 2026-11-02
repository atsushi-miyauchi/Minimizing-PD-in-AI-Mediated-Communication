from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polar_ai.alpha_generation import (  # noqa: E402
    generate_heterogeneous_alpha,
    generate_homogeneous_alpha,
)
from polar_ai.equilibrium import compute_equilibrium  # noqa: E402
from polar_ai.graph_io import build_graph_matrices, load_edge_list  # noqa: E402
from polar_ai.instance import Instance  # noqa: E402
from polar_ai.objective import polarization_disagreement  # noqa: E402
from polar_ai.opinion_generation import (  # noqa: E402
    generate_polarized_opinions,
    generate_uniform_opinions,
)
from polar_ai.spectral import (  # noqa: E402
    homogeneous_lambda_vector,
    spectral_radius_lambda_p_gamma,
)


class CoreFrameworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_graph_matrices(
            [("a", "b", 2.0), ("b", "c", 1.0), ("c", "a", 1.0)],
            directed=True,
        )

    def test_load_edge_list_undirected_expands_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "edges.txt"
            path.write_text("a b\nb c 2\nx y\n", encoding="utf-8")
            graph = load_edge_list(path, directed=False)

        self.assertEqual(graph.nodes, ("a", "b", "c"))
        self.assertEqual(graph.W[0, 1], 1.0)
        self.assertEqual(graph.W[1, 0], 1.0)
        self.assertEqual(graph.W[1, 2], 2.0)
        self.assertEqual(graph.W[2, 1], 2.0)

    def test_p_is_row_stochastic(self) -> None:
        row_sums = np.asarray(self.graph.P.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, np.ones(self.graph.P.shape[0]))

    def test_wbar_is_symmetric(self) -> None:
        diff = (self.graph.Wbar - self.graph.Wbar.T).tocoo()
        self.assertTrue(np.allclose(diff.data, 0.0))

    def test_generated_opinions_are_in_unit_interval(self) -> None:
        uniform = generate_uniform_opinions(200, seed=1)
        polarized = generate_polarized_opinions(200, seed=2)
        self.assertTrue(np.all(uniform >= -1.0))
        self.assertTrue(np.all(uniform <= 1.0))
        self.assertTrue(np.all(polarized >= -1.0))
        self.assertTrue(np.all(polarized <= 1.0))

    def test_generated_alpha_is_at_least_one(self) -> None:
        homogeneous = generate_homogeneous_alpha(10)
        heterogeneous = generate_heterogeneous_alpha(100, seed=3)
        self.assertTrue(np.all(homogeneous >= 1.0))
        self.assertTrue(np.all(heterogeneous >= 1.0))

    def test_lambda_makes_lambda_p_gamma_stable(self) -> None:
        alpha = generate_heterogeneous_alpha(self.graph.P.shape[0], seed=4)
        lambda_vec = homogeneous_lambda_vector(self.graph.P, alpha)
        rho = spectral_radius_lambda_p_gamma(self.graph.P, lambda_vec, alpha)
        self.assertLess(rho, 1.0)

    def test_equilibrium_remains_in_unit_box(self) -> None:
        n = self.graph.P.shape[0]
        s = generate_uniform_opinions(n, seed=5)
        alpha = generate_homogeneous_alpha(n)
        lambda_vec = homogeneous_lambda_vector(self.graph.P, alpha)
        instance = Instance(
            W=self.graph.W,
            P=self.graph.P,
            Wbar=self.graph.Wbar,
            s=s,
            lambda_vec=lambda_vec,
            alpha=alpha,
            metadata={},
        )

        z, diagnostics = compute_equilibrium(instance, tol=1e-12, max_iter=500)

        self.assertTrue(diagnostics["converged"])
        self.assertTrue(np.all(z >= -1.0 - 1e-12))
        self.assertTrue(np.all(z <= 1.0 + 1e-12))

    def test_pd_is_nonnegative(self) -> None:
        z = np.array([-0.5, 0.0, 0.75])
        pd = polarization_disagreement(z, self.graph.Wbar)
        self.assertGreaterEqual(pd, 0.0)


if __name__ == "__main__":
    unittest.main()
