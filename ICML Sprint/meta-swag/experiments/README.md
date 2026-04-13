# Meta-SWAG Experiments

This folder contains a first executable scaffold for the paper's matrix-game experiments.

The current setup is aligned to the configurations described in `resources/technical_report.pdf`, especially:

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

## Run

```bash
cd /Users/meuge/coding/maynard/ICML\ Sprint/meta-swag
python3 experiments/run_matrix_games.py --output-dir experiments/artifacts
python3 experiments/inspect_kim_personas.py
python3 experiments/run_kim_iterated.py --output-dir experiments/artifacts
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

## Notes

- This is a research scaffold, not a final benchmark implementation.
- The current simulator uses exact expected-payoff objectives plus injected Gaussian gradient noise as a controlled stand-in for estimator variance.
- The current runner compares `softmax`, `ess`, and `threshold` weighting schemes so we can measure posterior collapse versus Goodhart-resilient weighting directly.
- The report mentions original code at `simulations/{full_experiment, fixed_point_ne, iterated_games}.py`; those files are not present in this workspace, so this folder is a clean reconstruction rather than a direct port.
- The cloned Meta-MAPG repo is now available under `external/meta-mapg`, and its bundled persona files can be loaded directly from this scaffold.
- The iterated runner now evaluates exact discounted adaptation against bundled Kim-style personas for IPD and RPS, logging objective values, ESS, and low-rank eigenvalue snapshots.
- The next natural step after this is LoRA checkpoint collection plus posterior fitting over saved adapters.
