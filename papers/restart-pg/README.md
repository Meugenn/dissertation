# Restart-PG: Global Policy Gradient Convergence via Random Restarts

**Target**: Arxiv preprint → NeurIPS 2026 / AAMAS 2027  
**Status**: Draft complete (10 pages NeurIPS format). Pending: NN experiment figure, LLM experiment polish.

## Paper Summary

Lifts local policy gradient convergence (Giannou et al. 2022) to global almost-sure convergence via random restarts. Three contributions:

| Contribution | Result |
|---|---|
| Global convergence (Thm 1) | Almost-sure convergence, E[K*] ≤ 1/(p(1-δ)) |
| Equilibrium selection (Prop 2) | Pareto-optimal Nash found w.p. 1-(1-p_max(1-δ))^K |
| Experiments | Tabular ✓, Neural network (running), LLM alignment (simulated) |

## Directory Structure

```
restart-pg/
├── src/
│   ├── main.tex              # Paper source (10 pages NeurIPS format)
│   └── neurips_2024.sty      # Style file approximation
├── figures/
│   ├── basins_of_attraction.pdf     # 3-game basin visualization
│   ├── equilibrium_selection.pdf    # Welfare vs restarts (Stag Hunt)
│   ├── dimension_scaling.pdf        # Blessing of dimensionality
│   ├── geometric_convergence.pdf    # Geometric failure decay
│   ├── stochastic_pg.pdf            # |S|=2 state-contingent result
│   ├── nn_restart_results.pdf       # (generating...)
│   └── llm_experiment.pdf           # LLM alignment game (simulated)
├── experiments/
│   ├── restart_pg.py          # Original tabular experiments (from dissertation)
│   ├── stochastic_pg.py       # Stochastic game experiments
│   ├── nn_restart_pg.py       # NEW: neural network policy experiment
│   └── llm_alignment_game.py  # NEW: LLM alignment game
└── references/
    └── references.bib
```

## Compile

```bash
cd src/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Key Prior Art Check (April 2026)

No prior work uses random restarts for global convergence in general stochastic games:

| Method | Mechanism | Selection? |
|---|---|---|
| **Restart-PG (ours)** | Spatial random restarts | **Yes** |
| Nash-PG (2510.18183) | Regularization refinement | No |
| n-sided PL (2602.11835) | PL condition assumption | No |
| Markov Potential (2106.01969) | Potential game structure | No |
| Alternating PG (2501.07022) | Proximal-PL | No |

## Running Experiments

```bash
# Tabular experiments (already run, figures in figures/)
cd experiments/
python3 restart_pg.py          # Re-generates tabular figures

# Neural network experiment (NEW)
python3 nn_restart_pg.py       # ~20 min, produces nn_restart_results.pdf

# LLM alignment (simulated, quick)
python3 llm_alignment_game.py --quick    # ~5 min
python3 llm_alignment_game.py            # Full, ~30 min
```

## Submission Checklist (Arxiv)

- [ ] Finalize NN experiment figure (running)
- [ ] Re-run LLM experiment with more steps for cleaner results
- [ ] Add author names and LSE affiliation
- [ ] Supervisor co-authorship discussion (Prof. Galit Ashkenazi-Golan)
- [ ] Final proofread of all theorem statements
- [ ] Create arxiv-compatible source package (remove .sty conflicts)
- [ ] Upload to arxiv cs.GT + cs.LG categories

## Arxiv Category

Primary: cs.GT (Computer Science - Computer Science and Game Theory)  
Secondary: cs.LG (Machine Learning), stat.ML

## Target Venues (Post-Arxiv)

1. NeurIPS 2026 (submission ~May 2026)
2. AAMAS 2027
3. ICLR 2027
