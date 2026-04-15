# Matching GFM Paper

This folder is the first paper-focused scaffold for the proposal in
`/Users/meuge/Downloads/matching_gfm_proposal (1).pdf`.

The current milestone turns the proposal into a runnable synthetic validation
track built around:

- temporal bipartite multigraph generation with typed edges
- preference recovery from observed interaction history
- one-to-one matching with Gale-Shapley and Sinkhorn relaxations
- stability, welfare, and ranking metrics aligned to the paper
- two initial model families:
  - a pointwise gradient-boosted baseline
  - a compact temporal graph-matching model with stability regularization

This is intentionally the smallest end-to-end scaffold that exercises the
paper's main loop:

`interaction graph -> recovered preferences -> matching -> stability metrics`

## Layout

- `paper-draft.md`: working paper outline and implementation notes
- `paper/main.tex`: full working paper in LaTeX
- `paper/empirical_validation_note.tex`: short empirical validation note in LaTeX
- `paper/references.bib`: bibliography for both LaTeX documents
- `experiments/README.md`: experiment-specific usage notes
- `experiments/matching_gfm/`: synthetic market, matching, metrics, and model code
- `experiments/run_synthetic_matching.py`: end-to-end runner
- `experiments/run_real_market_experiment.py`: real-data runner for Polymarket and local H&M
- `experiments/render_empirical_note.py`: render LaTeX tables and a figure from run artifacts
- `tests/test_matching_gfm.py`: small regression suite using `unittest`
- `validation_datasets.md`: ranked survey of candidate validation datasets

## Run

```bash
cd /Users/meuge/coding/maynard/ICML\ Sprint/matching-gfm-paper
python3 experiments/run_synthetic_matching.py --output-dir experiments/artifacts
python3 experiments/run_real_market_experiment.py --source polymarket --output-dir experiments/artifacts/polymarket_live
python3 experiments/run_real_market_experiment.py --source hm_local --data-dir /path/to/hm --output-dir experiments/artifacts/hm_local
python3 experiments/render_empirical_note.py --artifacts-dir experiments/artifacts/default_check
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd paper && pdflatex empirical_validation_note.tex && bibtex empirical_validation_note && pdflatex empirical_validation_note.tex && pdflatex empirical_validation_note.tex
python3 -m unittest tests/test_matching_gfm.py
```

## Outputs

The runner writes:

- `summary.csv`: model-level comparison table
- `matchings.csv`: buyer-to-seller assignments by model
- `metadata.json`: market configuration and run settings
- `paper/generated_results.tex`: auto-rendered LaTeX tables and commentary
- `paper/current_results.png`: auto-rendered summary figure

## Real Data

The scaffold now supports:

- `polymarket`: live ingestion from the official Polymarket Gamma and Data APIs
- `hm_local`: local H&M CSV ingestion from `customers.csv`, `articles.csv`, and `transactions_train.csv`

For real data, evaluation focuses on held-out future interactions and predicted
matching stability under the learned scores, since latent utilities are not
directly observed.

## Notes

- The current scaffold covers a synthetic preference-recovery setting, not the
  full ChronoGraph pretraining programme from the proposal.
- The compact model is a small research prototype, not yet a full temporal GFM.
  It uses recency-weighted typed message aggregation, a Sinkhorn relaxation, and
  a blocking-pair loss so we can test the paper's stability story early.
- The next natural extensions are:
  - add static and temporal GNN baselines
  - strengthen the current Polymarket slice into a richer temporal graph dataset
  - add many-to-one and many-to-many matching variants
