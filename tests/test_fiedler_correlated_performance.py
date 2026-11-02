from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polar_ai.experiments.run_fiedler_correlated_performance import (  # noqa: E402
    base_opinion_settings,
    build_fiedler_instance,
    compute_fiedler_vector,
    opinion_diagnostics,
    _seed_result_is_current,
)
from polar_ai.experiments.utils import stable_seed  # noqa: E402
from polar_ai.graph_io import build_graph_matrices  # noqa: E402
from polar_ai.opinion_generation import (  # noqa: E402
    generate_polarized_opinions,
    generate_uniform_opinions,
)


class FiedlerCorrelatedPerformanceTests(unittest.TestCase):
    @staticmethod
    def _path_graph(n: int = 6):
        return build_graph_matrices(
            [(str(i), str(i + 1), 1.0) for i in range(n - 1)],
            directed=False,
        )

    def test_fiedler_vector_satisfies_second_eigenpair(self) -> None:
        graph = self._path_graph()
        vector, diagnostics = compute_fiedler_vector(
            graph.Wbar,
            tol=1.0e-10,
        )
        self.assertEqual(vector.shape, (6,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0)
        self.assertAlmostEqual(float(vector.mean()), 0.0, places=8)
        self.assertGreater(float(diagnostics["fiedler_eigenvalue"]), 0.0)
        self.assertLess(float(diagnostics["fiedler_residual_2"]), 1.0e-8)
        self.assertGreater(int(diagnostics["fiedler_positive_count"]), 0)
        self.assertGreater(int(diagnostics["fiedler_negative_count"]), 0)

    def test_two_node_graph_is_supported(self) -> None:
        graph = self._path_graph(2)
        vector, diagnostics = compute_fiedler_vector(graph.Wbar)
        self.assertTrue(np.allclose(np.abs(vector), 1.0 / np.sqrt(2.0)))
        self.assertAlmostEqual(
            float(diagnostics["fiedler_eigenvalue"]),
            2.0,
        )

    def test_generated_opinions_preserve_values_and_follow_fiedler_order(
        self,
    ) -> None:
        graph = self._path_graph()
        fiedler, _ = compute_fiedler_vector(graph.Wbar)
        for setting, generator in [
            ("uniform", generate_uniform_opinions),
            ("polarized", generate_polarized_opinions),
        ]:
            instance = build_fiedler_instance(
                graph,
                fiedler=fiedler,
                dataset_name="path",
                base_opinion_setting=setting,
                alpha_setting="homogeneous",
                seed=17,
                eps=0.01,
                lambda_cap=0.5,
            )
            expected = generator(
                instance.n,
                seed=stable_seed(17, "path", setting, "s"),
            )
            self.assertTrue(np.allclose(np.sort(instance.s), np.sort(expected)))
            self.assertTrue(
                np.all(np.diff(instance.s[np.argsort(fiedler)]) >= 0.0)
            )
            self.assertAlmostEqual(float(instance.s.mean()), float(expected.mean()))
            self.assertAlmostEqual(float(instance.s.std()), float(expected.std()))
            self.assertEqual(
                instance.metadata["opinion_setting"],
                f"fiedler-{setting}",
            )
            self.assertEqual(
                instance.metadata["opinion_construction"],
                "fiedler_rank_preserving",
            )

    def test_opinion_diagnostics_report_perfect_rank_correlation(self) -> None:
        graph = self._path_graph()
        fiedler, _ = compute_fiedler_vector(graph.Wbar)
        opinions = np.empty_like(fiedler)
        order = np.argsort(fiedler)
        opinions[order] = np.linspace(-1.0, 1.0, fiedler.size)
        diagnostics = opinion_diagnostics(opinions, graph.Wbar, fiedler)
        self.assertAlmostEqual(
            diagnostics["opinion_fiedler_spearman_correlation"],
            1.0,
        )
        self.assertGreaterEqual(
            diagnostics["opinion_graph_rayleigh_quotient"],
            0.0,
        )

    def test_invalid_and_duplicate_settings(self) -> None:
        self.assertEqual(
            base_opinion_settings(
                {"base_opinion_settings": ["uniform", "uniform", "polarized"]}
            ),
            ["uniform", "polarized"],
        )
        with self.assertRaises(ValueError):
            base_opinion_settings({"base_opinion_settings": ["unknown"]})
        with self.assertRaises(ValueError):
            compute_fiedler_vector(sparse.csr_matrix((1, 1)))

    def test_old_sign_based_seed_result_is_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deactivation.csv"
            pd.DataFrame(
                {
                    "opinion_construction": [
                        "fiedler_sign_base_magnitude",
                    ]
                }
            ).to_csv(path, index=False)
            self.assertFalse(_seed_result_is_current(path))

            pd.DataFrame(
                {"opinion_construction": ["fiedler_rank_preserving"]}
            ).to_csv(path, index=False)
            self.assertTrue(_seed_result_is_current(path))


if __name__ == "__main__":
    unittest.main()
