# EW-LOLA-PG: Precision Weighting and Opponent-Shaping Basin Enlargement

**Target**: NExT-Game: New Frontiers in Game-Theoretic Learning @ ICML 2026  
**Deadline**: April 24, 2026  
**Status**: Draft ready for author review

## Paper Summary

Three orthogonal improvements to policy gradient convergence in general stochastic games (Giannou et al. 2022 foundation):

| Contribution | Theorem | Result |
|---|---|---|
| Precision-Weighted PG | Thm 1 | Variance constant × HM(σ²)/AM(σ²) ≤ 1 |
| LOLA Basin Enlargement | Thm 2 | μ_LOLA = μ + λμ_H > μ |
| Combined PW-LOLA-PG | Thm 3 | Orthogonal composition of both |

## Directory Structure

```
ew-lola-pg/
├── src/
│   └── main.tex          # Paper source (6 pages, compiles cleanly)
├── figures/
│   ├── convergence_matching_pennies.png   # PW-PG convergence speedup
│   ├── variance_ratio_matching_pennies.png # AM-HM bound validation
│   ├── basin_matching_pennies.png          # LOLA basin enlargement
│   └── trajectories_matching_pennies.png  # LOLA trajectory comparison
├── references/
│   └── references.bib    # Full bibliography
└── experiments/          # (see dissertation/simulations/ for source code)
```

## Compile

```bash
cd src/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires: pdflatex, bibtex, standard TeX Live packages (times, natbib, booktabs, algorithm2e)

## Key Theorems (self-contained)

### Theorem 1 (PW-PG Variance Reduction)
With precision weights $w_{i,n} \propto 1/\sigma_{i,n}^2$:
$$\sigma_{w,n}^2 = \frac{\mathrm{HM}(\sigma^2)}{\mathrm{AM}(\sigma^2)} \cdot \sigma_n^2 \leq \sigma_n^2$$
Maximum improvement: 67% for 2-agent, 10:1 heterogeneity ratio.

### Theorem 2 (LOLA Basin Enlargement)
Under spectral reinforcement ($S_H \prec 0$, $\mu_H > 0$):
$$\mu_\mathrm{LOLA} = \mu + \lambda\mu_H > \mu$$
Holds in zero-sum games and negative cross-Hessian games. Fails in potential games.

### Theorem 3 (Combined)
$$C_\mathrm{combined} = \frac{\mathrm{HM}}{\mathrm{AM}} \cdot C_\mathrm{std}, \quad \mu_\mathrm{eff} = \mu + \lambda\mu_H$$
Orthogonal because HM/AM enters the noise term, μ_H enters the drift term in the Lyapunov recursion.

## Figures Source

Figures are from the extended dissertation simulations in:
- `dissertation/simulations/lola_basin.py` → LOLA basin/trajectory figures
- `dissertation/simulations/evidence_weighted_pg.py` → PW-PG figures

To regenerate:
```bash
cd dissertation/simulations/
python3 lola_basin.py
python3 evidence_weighted_pg.py
```

## Submission Checklist

- [ ] Author names and affiliations (currently anonymous)
- [ ] Verify NExT-Game page limit (4 or 6 pages?)
- [ ] Check CFP for specific formatting requirements
- [ ] Supervisor (Prof. Galit Ashkenazi-Golan) coauthorship discussion
- [ ] Proofread all theorem statements against dissertation source
- [ ] Verify all citations render correctly
- [ ] Submit to OpenReview (check CFP for portal)
- [ ] Upload to arxiv simultaneously for credential establishment

## Positioning Against CFP

The NExT-Game CFP asks for:
- "theoretical frontiers" → ✓ (three new theorems building on Giannou 2022)
- "high-dimensional, non-convex landscapes" → ✓ (spectral analysis of LOLA Hessian, effective dimension)
- "principal-agent dynamics among boundedly rational learners" → ✓ (heterogeneous data = bounded rationality; LOLA = principal models agent)
- "regret-minimizing learners" → ✓ (PG is regret-minimizing in the SOS regime)

## Notes on Future Work

The "alignment component" of the full gradient (Hayekian consensus penalties) is deferred to future work and not mentioned in the submission. Focus is tight on the two proven results.
