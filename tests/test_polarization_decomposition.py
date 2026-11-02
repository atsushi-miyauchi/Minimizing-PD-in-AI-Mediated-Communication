from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polar_ai.experiments.plot_polarization_decomposition import (  # noqa: E402
    plot_polarization_decomposition,
    write_polarization_decomposition_summary,
)
from polar_ai.experiments.run_polarization_decomposition import (  # noqa: E402
    polarization_decomposition,
    run_dataset,
)
from polar_ai.experiments.utils import DatasetConfig  # noqa: E402
from polar_ai.graph_io import build_graph_matrices  # noqa: E402


class PolarizationDecompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = build_graph_matrices(
            [
                ("a", "b", 1.0),
                ("b", "c", 1.0),
                ("c", "a", 1.0),
            ],
            directed=True,
        )
        self.dataset = DatasetConfig(name="toy", path=Path("toy.edges"))
        self.config = {
            "seed": 7,
            "eps": 0.01,
            "lambda_cap": 0.5,
            "eq_tol": 1e-8,
            "eq_max_iter": 1000,
            "opinion_settings": ["uniform", "polarized"],
            "alpha_values": [1.0, 2.0, 3.0],
            "lambda_reference_alpha": 3.0,
            "fail_on_nonconvergence": True,
        }

    def _rows(self) -> list[dict[str, object]]:
        return run_dataset(
            self.config,
            self.dataset,
            self.graph.W,
            self.graph.P,
            self.graph.Wbar,
            list(self.graph.nodes),
            backend="python",
        )

    def test_exact_vector_decomposition(self) -> None:
        values = np.array([-0.6, 0.2, 0.8], dtype=float)
        result = polarization_decomposition(values)
        self.assertAlmostEqual(
            result["polarization"],
            result["dispersion"] + result["mean_component"],
        )
        self.assertAlmostEqual(result["residual"], 0.0)

    def test_sweep_reproduces_baseline_and_component_identity(self) -> None:
        frame = pd.DataFrame(self._rows())
        self.assertEqual(len(frame), 6)
        baseline = frame[np.isclose(frame["alpha"], 1.0)]
        np.testing.assert_allclose(baseline["pd_ratio"], 1.0, atol=1e-8)
        np.testing.assert_allclose(
            baseline["polarization_change_over_baseline_polarization"],
            0.0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            frame["dispersion_change"] + frame["mean_component_change"],
            frame["polarization_change"],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            frame["dispersion_change_over_baseline_polarization"]
            + frame["mean_component_change_over_baseline_polarization"],
            frame["polarization_change_over_baseline_polarization"],
            atol=1e-10,
        )
        self.assertEqual(frame["lambda_value"].nunique(), 1)

    def test_summary_and_two_plots_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "polarization_decomposition.csv"
            pd.DataFrame(self._rows()).to_csv(csv_path, index=False)
            summary_path = root / "summary.csv"
            write_polarization_decomposition_summary(csv_path, summary_path)
            outputs = plot_polarization_decomposition(
                csv_path, root / "figures"
            )

            summary = pd.read_csv(summary_path)
            self.assertEqual(len(summary), 2)
            self.assertTrue((summary["endpoint_alpha"] == 3.0).all())
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.exists() for path in outputs))


if __name__ == "__main__":
    unittest.main()
