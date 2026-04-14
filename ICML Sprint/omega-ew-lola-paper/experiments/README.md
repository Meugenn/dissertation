# EW-LOLA Experiments

This folder is the first paper-focused experiment scaffold for the EW-LOLA project.

It currently contains:

- `ew_lola_core.py`: shared utilities for two-player matrix and iterated games
- `run_ew_lola_matrix.py`: baseline runner for standard, EW, LOLA, and EW-LOLA on matrix games
- `run_basin_mapping_matrix.py`: deterministic basin-mapping runner for matching pennies near the mixed Nash point
- `run_ew_lola_iterated.py`: baseline runner for the same four methods on iterated IPD and iterated RPS
- `run_ew_lola_kim_personas.py`: persona-conditioned iterated-game runner using the bundled Meta-MAPG IPD and RPS personas
- `LARGE_SCALE_EXPERIMENT_PLAN.md`: staged roadmap from paper-core runs to a `$10k` benchmark campaign

## Current Design

The implementation is intentionally small and explicit.

- Gradients are computed with finite differences.
- The standard policy-gradient part is perturbed with Gaussian noise to model heterogeneous estimator quality.
- EW rescales each player's update by the oracle variance ratio `min(V) / V_i`.
- LOLA is approximated by differentiating through one anticipated opponent update.
- EW-LOLA combines both modifications in the same per-player step.

This is a research scaffold, not the final benchmark stack.

## Run

```bash
cd /Users/meuge/coding/maynard/ICML\ Sprint/omega-ew-lola-paper/experiments
python3 run_ew_lola_matrix.py --output-dir artifacts/matrix
python3 run_basin_mapping_matrix.py --output-dir artifacts/basin_matrix --grid-size 17 --radius 1.6 --steps 100 --lr 0.35 --lambdas 0.0 0.4 0.8
python3 run_ew_lola_iterated.py --output-dir artifacts/iterated
python3 run_ew_lola_kim_personas.py --output-dir artifacts/kim_personas
python3 run_ew_lola_kim_personas.py --envs ipd --methods standard ew lola ew_lola --output-dir artifacts/kim_ipd
```

## Expected Outputs

Matrix runner:

- `matrix_summary.csv`
- `matrix_trace.csv`
- `matrix_summary.png`

Basin runner:

- `basin_map.csv`
- `basin_summary.csv`
- `basin_map.png`

Iterated runner:

- `iterated_summary.csv`
- `iterated_trace.csv`

Kim persona runner:

- `kim_persona_summary.csv`
- `kim_persona_trace.csv`
- `kim_persona_summary.partial.csv`
- `kim_persona_trace.partial.csv`

## Next Steps

- replace oracle EW weights with estimated online variance weights
- extend the Kim-style persona runner with richer persona-level summaries and paper-ready plots
- use the `--envs` and `--methods` switches to shard bigger campaigns across machines or jobs
- push the final figure generation into a paper-ready plotting script
