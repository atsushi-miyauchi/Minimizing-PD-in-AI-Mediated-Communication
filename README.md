# Minimizing Polarization and Disagreement in AI-Mediated Communication

This repository contains the public reproduction bundle for the
paper *Minimizing Polarization and Disagreement in AI-Mediated
Communication*. It includes the implementation of all algorithms, the main
experiments, the additional experiments reported in the paper, and scripts
that derive manuscript-facing statistics from raw CSV files.

All six main-experiment networks are bundled: wiki-Vote, hamster, advogato,
epinions, Slashdot0902, and youtube. The exact-search experiments additionally
use soc-karate and soc-dolphins. The much larger soc-pokec network is not used
in the paper and is intentionally excluded.

## Repository layout

```text
configs/quick.yaml              Short smoke test (not a paper experiment)
configs/main.yaml               Full main transformation and deactivation runs
configs/additional/             One independent config per additional run
cpp/                            Optional high-performance C++17 backend
data/raw/                       Bundled Network Repository instances
scripts/analysis/               Raw-results-to-paper-metrics scripts
src/polar_ai/                   Algorithms and experiment drivers
tests/                          Unit and integration tests
results/                        Generated CSV, checkpoint, and PDF files
```

There is intentionally one README at the repository root. Experiment-specific
settings and output paths are documented directly in the corresponding YAML
files.

## Setup

Python 3.9 or later and a C++17 compiler are required for the full experiments.
From the repository root, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -c "from polar_ai.backends.selector import get_backend_name; print(get_backend_name())"
pytest
```

The editable installation compiles the C++ numerical backend with `-O3` on
macOS/Linux (`/O2` on Windows). If compilation is unavailable, ordinary model
operations fall back to Python/SciPy. The backend check above should print
`cpp`; the expensive additional-experiment configs deliberately require it.

## Quick smoke test

Before starting the full runs, verify the complete pipeline on reduced
wiki-Vote settings:

```bash
python -m polar_ai.experiments.run_suite --config configs/quick.yaml
```

This checks data loading, equilibrium computation, all deactivation methods,
CSV output, and plotting. Its reduced budgets, alpha grid, repetitions, and
timeout do not reproduce a paper result.

## Main experiments

Run the transformation-effect experiment and the deactivation benchmark, then
generate their paper-style plots:

```bash
python -m polar_ai.experiments.run_suite --config configs/main.yaml
```

Outputs are written to `results/main/<dataset>/`. The config uses the four
combinations of uniform/polarized innate opinions and homogeneous/heterogeneous
amplification parameters, budgets up to 500, the 1,800-second per-method time
limit, and the paper's $k\in\{1,10,100\}$ settings on all six networks. A full
run is computationally expensive and can take several days; completed datasets
remain on disk if a later dataset is interrupted.

## Additional experiments

Each command below has an independent config and writes below
`results/additional/`. Long-running drivers append checkpoints and can be
resumed by rerunning the same command.

### Approximate improvement score

```bash
python -m polar_ai.experiments.run_ais_score_diagnostic \
  --config configs/additional/ais_score_diagnostic.yaml
python scripts/analysis/summarize_ais_score.py
```

This evaluates every remaining candidate exactly at every AIS-Greedy
iteration on wiki-Vote and hamster. It is substantially more expensive than
an ordinary AIS-Greedy run.

### Polarization decomposition and fixed-lambda diagnostic

```bash
python -m polar_ai.experiments.run_polarization_decomposition \
  --config configs/additional/polarization_decomposition.yaml
python scripts/analysis/summarize_polarization_decomposition.py

python -m polar_ai.experiments.run_fixed_lambda_alpha_sweep \
  --config configs/additional/fixed_lambda_alpha_sweep.yaml
```

The first experiment separates the polarization change into dispersion and
mean-shift terms. The second compares the fixed-lambda sweep used in the paper
with recomputing lambda for every alpha.

### Robustness to generated parameters and network-correlated opinions

```bash
python -m polar_ai.experiments.run_multi_seed_performance \
  --config configs/additional/multi_seed_performance.yaml
python -m polar_ai.experiments.run_fiedler_correlated_performance \
  --config configs/additional/fiedler_correlated_performance.yaml
python scripts/analysis/summarize_robustness.py
```

Both configs contain wiki-Vote, hamster, advogato, and epinions, matching the
networks reported for these robustness checks. To run only one, append
`--dataset wiki-Vote` (or another configured name). The five-seed experiment
can take several days with the full settings.

### Exact optimum and Greedy local-search refinement

```bash
python -m polar_ai.experiments.run_exact_optimum_small_networks \
  --config configs/additional/exact_optimum_karate.yaml
python -m polar_ai.experiments.run_exact_optimum_small_networks \
  --config configs/additional/exact_optimum_dolphins.yaml

python -m polar_ai.experiments.run_backtracking_greedy \
  --config configs/additional/backtracking_karate.yaml
python -m polar_ai.experiments.run_backtracking_greedy \
  --config configs/additional/backtracking_dolphins.yaml
python -m polar_ai.experiments.run_backtracking_greedy \
  --config configs/additional/backtracking_wiki_vote.yaml

python scripts/analysis/summarize_greedy_baseline_quality.py
```

Exact search enumerates every subset up to the configured budget. Despite the
small graphs, the soc-dolphins run through budget 6 is computationally
expensive. The local-search runs are also much less scalable than Greedy.

### Selected-set size

After the main experiment has completed, run:

```bash
python -m polar_ai.experiments.run_selected_set_size_audit \
  --config configs/additional/selected_set_size.yaml
python scripts/analysis/summarize_selected_set_size.py
```

This is a derived analysis of the main CSVs and performs no new optimization.

## Results and reproducibility

Raw results, resumable checkpoints, derived summaries, and figures are kept
under `results/`; generated files are ignored by Git. Random seeds and all
numerical tolerances are recorded in YAML. Runtime measurements are
hardware-dependent; the paper used an Apple M1 Mac mini with 16 GB RAM.

The software implementation is Python with performance-critical routines in
C++. The C++ extension is compiled with optimization enabled as described
above.

## Data

The bundled graph files were obtained from Network Repository (Rossi and
Ahmed, 2015). Network Repository states that its data are distributed under a
Creative Commons Attribution-ShareAlike License. Dataset URLs, requested
citations, the license-policy link, and the preprocessing disclosure are in
[`DATA_LICENSE.md`](DATA_LICENSE.md). Keep that notice and all headers inside
the data files when redistributing the repository. File integrity can be
checked with `cd data && shasum -a 256 -c SHA256SUMS`.

## License

The Python and C++ software is released under the MIT License; see
[`LICENSE`](LICENSE). The graph files under `data/raw/` are separately covered
by the terms described in [`DATA_LICENSE.md`](DATA_LICENSE.md). The MIT License
does not replace or relicense those data.
