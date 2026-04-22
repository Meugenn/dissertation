# Meta-MAPG Restart Paper

This folder contains a fresh ICML-style workshop draft for the Meta-MAPG convergence and restart-globalisation thesis.

Canonical files:

- `main.tex`: LaTeX source for the paper.
- `main.pdf`: compiled PDF.
- `references.bib`: bibliography.
- `experiments/run_meta_mapg_experiments.py`: sample-based tabular experiments.
- `artifacts/main/`: 20-seed experiment outputs used in the paper.
- `figures/` and `tables/`: paper-ready copies of the generated figures and summary table.

Regenerate the experiments from this directory with:

```bash
python3 experiments/run_meta_mapg_experiments.py \
  --outdir artifacts/main \
  --seeds 20 \
  --steps 260 \
  --restart-steps 120 \
  --max-restarts 12 \
  --batch-size 256 \
  --basin-batch-size 128 \
  --grid-size 21 \
  --basin-steps 120 \
  --reference-batch-size 120000 \
  --sanity-reps 80 \
  --own-coef 0.05 \
  --peer-coef 1.5 \
  --basin-peer-coef 2.0
```

Recompile the paper with:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```
