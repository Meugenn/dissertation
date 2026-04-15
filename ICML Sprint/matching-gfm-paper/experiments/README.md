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
- real-data adapters for live Polymarket and local H&M CSVs

## Run

```bash
cd /Users/meuge/coding/maynard/ICML\ Sprint/matching-gfm-paper
python3 experiments/run_synthetic_matching.py --output-dir experiments/artifacts
python3 experiments/run_real_market_experiment.py --source polymarket --output-dir experiments/artifacts/polymarket_live
python3 experiments/run_real_market_experiment.py --source hm_local --data-dir /path/to/hm --output-dir experiments/artifacts/hm_local
python3 experiments/render_empirical_note.py --artifacts-dir experiments/artifacts/default_check
```

## Expected outputs

- `summary.csv`
- `matchings.csv`
- `metadata.json`
- `paper/generated_results.tex`
- `paper/current_results.png`

Polymarket runs also write:

- `snapshot/train_edges.csv`
- `snapshot/eval_edges.csv`
- `snapshot/buyers.csv`
- `snapshot/sellers.csv`

## Design notes

- The compact model is deliberately small so we can run it with the system
  Python stack already available in this workspace.
- Optimization uses `scipy.optimize.minimize` over a low-parameter scorer rather
  than a full neural implementation.
- The code is structured so a future temporal GNN can replace the compact model
  without changing the evaluation path.
- The H&M adapter expects the official CSV filenames `customers.csv`,
  `articles.csv`, and `transactions_train.csv`.
