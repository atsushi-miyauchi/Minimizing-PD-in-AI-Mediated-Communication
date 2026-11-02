from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polar_ai.experiments.plot_fixed_lambda_alpha_sweep import (  # noqa: E402
    plot_fixed_lambda_alpha_sweep,
    write_fixed_lambda_comparison,
)
from polar_ai.experiments.run_fixed_lambda_alpha_sweep import (  # noqa: E402
    FIXED_REGIME,
    RECOMPUTED_REGIME,
    run_dataset,
)
from polar_ai.experiments.utils import DatasetConfig  # noqa: E402
from polar_ai.graph_io import build_graph_matrices  # noqa: E402


class FixedLambdaAlphaSweepTests(unittest.TestCase):
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
            "eq_tol": 1e-6,
            "eq_max_iter": 1000,
            "opinion_settings": ["uniform", "polarized"],
            "alpha_values": [1.0, 2.0, 3.0],
            "lambda_reference_alpha": 3.0,
            "lambda_regimes": [FIXED_REGIME, RECOMPUTED_REGIME],
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

    def test_fixed_lambda_is_constant_and_endpoints_match(self) -> None:
        frame = pd.DataFrame(self._rows())
        self.assertEqual(len(frame), 12)
        for _, group in frame.groupby("opinion_setting"):
            fixed = group[group["lambda_regime"] == FIXED_REGIME]
            recomputed = group[group["lambda_regime"] == RECOMPUTED_REGIME]
            self.assertEqual(fixed["lambda_value"].nunique(), 1)
            np.testing.assert_allclose(
                fixed.loc[fixed["alpha"] == 1.0, "ratio"], 1.0
            )
            np.testing.assert_allclose(
                recomputed.loc[recomputed["alpha"] == 1.0, "ratio"], 1.0
            )
            fixed_end = float(fixed.loc[fixed["alpha"] == 3.0, "ratio"].iloc[0])
            recomputed_end = float(
                recomputed.loc[recomputed["alpha"] == 3.0, "ratio"].iloc[0]
            )
            self.assertAlmostEqual(fixed_end, recomputed_end)
            self.assertTrue(
                np.all(np.diff(recomputed.sort_values("alpha")["lambda_value"]) <= 0)
            )
        self.assertTrue(
            (frame["stability_spectral_radius"] < 1.0).all()
        )

    def test_comparison_csv_and_four_plots_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "fixed_lambda_alpha_sweep.csv"
            pd.DataFrame(self._rows()).to_csv(csv_path, index=False)
            comparison_path = root / "comparison.csv"
            write_fixed_lambda_comparison(csv_path, comparison_path)
            outputs = plot_fixed_lambda_alpha_sweep(
                csv_path, root / "figures"
            )

            self.assertTrue(comparison_path.exists())
            self.assertEqual(len(outputs), 4)
            self.assertTrue(all(path.exists() for path in outputs))


if __name__ == "__main__":
    unittest.main()
