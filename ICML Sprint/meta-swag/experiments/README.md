# Meta-SWAG Experiments

This folder contains the current executable experiment stack for Meta-SWAG.

The project now has three empirical tracks:

- matrix-game experiments aligned to `resources/technical_report.pdf`
- Kim-style iterated IPD and RPS experiments using the bundled Meta-MAPG personas
- an AxBench and AlpacaEval benchmark runner for LoRA and PreferenceLoRA checkpoint trajectories

The original matrix-game setup is aligned to the configurations described in `resources/technical_report.pdf`, especially:

- Experiment 1: EW-PG variance reduction on Matching Pennies and a 3-action zero-sum variant
- heterogeneity levels `V = [1,1], [5,1], [20,1]`
- posterior fitting and geometry diagnostics intended to support the Meta-SWAG paper draft

## What is implemented

- exact two-player matrix games:
  - matching pennies
  - rock-paper-scissors as a 3-action zero-sum variant
  - plus extra toy games available for later extensions
- Kim et al. 2021 reference persona loader:
  - loads bundled IPD and RPS test personas from `external/meta-mapg/pretrain_model`
  - summarizes state-wise action bias and entropy
- noisy policy-gradient trajectories over 4-dimensional logits
- Meta-SWAG fitting with:
  - standard softmax weighting
  - ESS-constrained softmax weighting
  - thresholded satisficing weighting
  - diagonal-plus-low-rank covariance
- two draft metrics aligned with the paper:
  - variance reduction against the HM/AM prediction from the report
  - posterior geometry against a finite-difference Hessian basin proxy
- one runner script that writes CSV summaries, eigenvalue diagnostics, and a paper-ready placeholder plot
- one Kim-style iterated-game runner that evaluates bundled IPD/RPS personas
- one AxBench benchmark runner that:
  - retains adapter-only checkpoints over the tail of training
  - evaluates retained checkpoints on validation factor sweeps
  - aggregates them with `map`, `uniform`, `softmax`, `ess`, or `threshold`
  - writes checkpoint-level metrics, factor sweeps, final summaries, and dependency manifests
- one Posterior HyperSteer benchmark runner that:
  - trains the shared HyperSteer hypernetwork with late-checkpoint retention
  - posteriorizes retained hypernetwork checkpoints with `map`, `uniform`, `softmax`, `ess`, or `threshold`
  - supports optional posterior-sampled factor statistics for risk-aware factor selection
  - writes HyperSteer-specific checkpoint metrics, factor sweeps, final summaries, and optional transfer outputs

## Run

```bash
cd /Users/meuge/coding/maynard/ICML\ Sprint/meta-swag
python3 experiments/run_matrix_games.py --output-dir experiments/artifacts
python3 experiments/inspect_kim_personas.py
python3 experiments/run_kim_iterated.py --output-dir experiments/artifacts
python3 experiments/run_axbench_meta_swag.py --help
python3 experiments/run_hypersteer_posterior.py --help
```

## Outputs

- `matrix_games_metrics.csv`
- `matrix_games_summary.csv`
- `matrix_games_summary.png`
- `matrix_games_eigenvalues.csv`
- `kim_persona_summary.csv`
- `kim_iterated_metrics.csv`
- `kim_iterated_summary.csv`
- `kim_iterated_eigenvalues.csv`
- AxBench run bundles under a user-provided output directory:
  - `dependency_manifest.json`
  - `checkpoint_validation_metrics.csv`
  - `factor_sweeps.csv`
  - `final_summary.csv`
  - `summary_by_scheme.csv`
  - `alpacaeval_summary.json`
- Posterior HyperSteer run bundles under a user-provided output directory:
  - `hypersteer_manifest.json`
  - `hypersteer_checkpoint_validation_metrics.csv`
  - `hypersteer_factor_sweeps.csv`
  - `hypersteer_final_summary.csv`
  - `hypersteer_summary_by_scheme.csv`
  - `hypersteer_posterior_factor_stats.csv` when posterior factor sampling is enabled
  - `hypersteer_alpacaeval_summary.json`

## Notes

- This is a research scaffold, not a final benchmark implementation.
- The current simulator uses exact expected-payoff objectives plus injected Gaussian gradient noise as a controlled stand-in for estimator variance.
- The current runner compares `softmax`, `ess`, and `threshold` weighting schemes so we can measure posterior collapse versus Goodhart-resilient weighting directly.
- The report mentions original code at `simulations/{full_experiment, fixed_point_ne, iterated_games}.py`; those files are not present in this workspace, so this folder is a clean reconstruction rather than a direct port.
- The cloned Meta-MAPG repo is now available under `external/meta-mapg`, and its bundled persona files can be loaded directly from this scaffold.
- The iterated runner now evaluates exact discounted adaptation against bundled Kim-style personas for IPD and RPS, logging objective values, ESS, and low-rank eigenvalue snapshots.
- The AxBench runner lives at [run_axbench_meta_swag.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/run_axbench_meta_swag.py:1).
- The Posterior HyperSteer runner lives at [run_hypersteer_posterior.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/run_hypersteer_posterior.py:1).
- The AxBench utilities live under [experiments/meta_swag](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag).
- HyperSteer-specific posterior utilities live at [experiments/meta_swag/hypersteer_posterior.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/hypersteer_posterior.py:1).
- The AxBench path is unit-tested and CLI-verified, but a full benchmark run still depends on completing the external runtime environment and model setup.
