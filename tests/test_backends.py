from __future__ import annotations

import sys
import unittest
import os
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polar_ai.algorithms import deactivation  # noqa: E402
from polar_ai.backends import (  # noqa: E402
    available_backends,
    backend_status,
    get_backend_name,
    set_backend,
)
from polar_ai.equilibrium import compute_equilibrium, extremize  # noqa: E402
from polar_ai.graph_io import build_graph_matrices  # noqa: E402
from polar_ai.instance import Instance  # noqa: E402
from polar_ai.objective import (  # noqa: E402
    objective_components,
    pd_gradient,
    polarization_disagreement,
)


@unittest.skipUnless("cpp" in available_backends(), "C++ extension is not built")
class BackendEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        graph = build_graph_matrices(
            [
                ("a", "b", 1.0),
                ("b", "c", 0.7),
                ("c", "d", 1.2),
                ("d", "a", 0.8),
                ("a", "c", 0.4),
            ],
            directed=True,
        )
        self.instance = Instance(
            W=graph.W,
            P=graph.P,
            Wbar=graph.Wbar,
            s=np.array([-0.7, -0.1, 0.25, 0.8]),
            lambda_vec=np.full(4, 0.2),
            alpha=np.array([1.4, 1.8, 2.0, 1.2]),
            metadata={},
        )

    def tearDown(self) -> None:
        set_backend(None)

    def _equilibrium(self, backend: str):
        set_backend(backend)
        return compute_equilibrium(self.instance, tol=1e-12, max_iter=1000)

    def test_default_backend_is_cpp(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("POLAR_AI_BACKEND", None)
            set_backend(None)
            self.assertEqual(get_backend_name(), "cpp")
            self.assertEqual(backend_status()["active"], "cpp")

    def test_extremize_equilibrium_and_diagnostics_match(self) -> None:
        values = np.array([-1.0, -0.3, 0.4, 0.9])
        alpha = np.array([2.0, 1.5, 2.5, 1.1])
        set_backend("python")
        python_extremized = extremize(values, alpha)
        python_z, python_diag = self._equilibrium("python")

        set_backend("cpp")
        cpp_extremized = extremize(values, alpha)
        cpp_z, cpp_diag = self._equilibrium("cpp")

        np.testing.assert_array_equal(cpp_extremized, python_extremized)
        np.testing.assert_allclose(cpp_z, python_z, atol=1e-14, rtol=1e-14)
        self.assertEqual(cpp_diag["converged"], python_diag["converged"])
        self.assertEqual(cpp_diag["iterations"], python_diag["iterations"])
        self.assertAlmostEqual(
            cpp_diag["residual_inf"], python_diag["residual_inf"], places=14
        )

    def test_objective_components_and_gradient_match(self) -> None:
        z, _ = self._equilibrium("python")
        set_backend("python")
        python_pd = polarization_disagreement(z, self.instance.Wbar)
        python_components = objective_components(self.instance, z)
        python_gradient = pd_gradient(z, self.instance.Wbar)

        set_backend("cpp")
        cpp_pd = polarization_disagreement(z, self.instance.Wbar)
        cpp_components = objective_components(self.instance, z)
        cpp_gradient = pd_gradient(z, self.instance.Wbar)

        self.assertAlmostEqual(cpp_pd, python_pd, places=14)
        for key in python_components:
            self.assertAlmostEqual(
                cpp_components[key], python_components[key], places=14
            )
        np.testing.assert_allclose(
            cpp_gradient, python_gradient, atol=1e-14, rtol=1e-14
        )

    def test_deactivation_trajectory_matches(self) -> None:
        set_backend("python")
        python_result = deactivation.ais_greedy(
            self.instance, 3, k=3, linear_max_iter=100
        )
        set_backend("cpp")
        cpp_result = deactivation.ais_greedy(
            self.instance, 3, k=3, linear_max_iter=100
        )

        self.assertEqual(
            [row["selected_node"] for row in cpp_result.logs[1:]],
            [row["selected_node"] for row in python_result.logs[1:]],
        )
        self.assertEqual(cpp_result.solution, python_result.solution)
        self.assertAlmostEqual(cpp_result.objective, python_result.objective, places=12)

    def test_python_backend_remains_explicitly_selectable(self) -> None:
        set_backend("python")
        self.assertEqual(get_backend_name(), "python")
        z, diagnostics = compute_equilibrium(self.instance)
        self.assertEqual(z.shape, (self.instance.n,))
        self.assertTrue(diagnostics["converged"])


if __name__ == "__main__":
    unittest.main()
