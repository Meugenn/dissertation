# Matching GFM Experiments

This folder contains the first executable experiment stack for the matching GFM
paper.

## What is implemented

- synthetic temporal bipartite market generation with latent buyer and seller
  utilities
- typed edge histories with recency weighting
- Sinkhorn soft matching and Gale-Shapley hard matching
- pointwise gradient-boosted baseline
- compact graph-matching model trained by reconstruction, stability, and
  welfare terms
- ranking, preference-recovery, stability, and welfare metrics

## Run

```bash
cd /Users/meuge/coding/maynard/ICML\ Sprint/matching-gfm-paper
python3 experiments/run_synthetic_matching.py --output-dir experiments/artifacts
python3 experiments/render_empirical_note.py --artifacts-dir experiments/artifacts/default_check
```

## Expected outputs

- `summary.csv`
- `matchings.csv`
- `metadata.json`
- `paper/generated_results.tex`
- `paper/current_results.png`

## Design notes

- The compact model is deliberately small so we can run it with the system
  Python stack already available in this workspace.
- Optimization uses `scipy.optimize.minimize` over a low-parameter scorer rather
  than a full neural implementation.
- The code is structured so a future temporal GNN can replace the compact model
  without changing the evaluation path.
