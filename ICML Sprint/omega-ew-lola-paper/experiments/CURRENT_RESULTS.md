# Current Results

This file records the runs that completed successfully during the initial EW-LOLA experiment push.

## Completed Runs

### Medium Matrix Sweep

Artifacts:

- [matrix_summary.csv](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/medium_matrix/matrix_summary.csv)
- [matrix_trace.csv](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/medium_matrix/matrix_trace.csv)
- [matrix_summary.png](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/medium_matrix/matrix_summary.png)

Settings:

- games: `matching_pennies`, `rock_paper_scissors`
- methods: `standard`, `ew`, `lola`, `ew_lola`
- seeds: `8`
- steps: `40`
- noise std pairs: `1,1`, `4,1`, `16,1`

Takeaways:

- under equal noise, `ew` is usually as good as or slightly better than the baseline, with `ew_lola` not yet clearly dominant;
- under heterogeneous noise, `ew_lola` becomes the best method on both matrix families;
- pure `lola` degrades badly in the highest-noise asymmetric settings, which is consistent with the idea that shaping alone does not address estimator variance.

### Matching-Pennies Basin Map

Artifacts:

- [basin_map.csv](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/basin_matrix/basin_map.csv)
- [basin_summary.csv](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/basin_matrix/basin_summary.csv)
- [basin_map.png](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/basin_matrix/basin_map.png)

Settings:

- game: `matching_pennies`
- methods: `standard`, `lola`
- grid: `17 x 17` symmetric logit initialisations in `[-1.6, 1.6]^2`
- steps: `100`
- learning rate: `0.35`
- LOLA strengths: `0.0`, `0.4`, `0.8`
- convergence threshold: final distance to Nash `<= 0.15`

Takeaways:

- standard PG converges from only `0.3%` of the tested initialisations;
- constant-`lambda` LOLA at `0.4` raises that to `14.2%`;
- constant-`lambda` LOLA at `0.8` raises it further to `51.6%`;
- this is the first direct empirical support in the scaffold for the basin-enlargement story, and it matches the expected behaviour in a rotational zero-sum game.

### Medium Kim-Persona IPD Sweep

Artifacts:

- [kim_persona_summary.csv](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/medium_kim_ipd/kim_persona_summary.csv)
- [kim_persona_trace.csv](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/medium_kim_ipd/kim_persona_trace.csv)
- [kim_persona_summary.png](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/medium_kim_ipd/kim_persona_summary.png)

Settings:

- environment: `ipd`
- personas per group: `2`
- groups: cooperative and defective
- seeds: `2`
- steps: `10`
- noise std pairs: `0.2,0.2`, `0.5,0.2`

Takeaways:

- at the lower-noise setting, `ew_lola` has the highest mean learner return;
- at the noisier asymmetric setting, `lola` remains strongest while `ew_lola` is slightly above `ew` and below pure `lola`;
- persona-conditioned evaluation is now working end-to-end, which is more meaningful than symmetric self-play for the paper.

## Completed Smoke Tests

Artifacts:

- [smoke_matrix](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/smoke_matrix)
- [smoke_iterated](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/smoke_iterated)
- [smoke_kim_personas_plot](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/smoke_kim_personas_plot)
- [smoke_kim_partial](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/smoke_kim_partial)

These confirm:

- both matrix and iterated runners execute successfully;
- the Kim persona runner executes successfully;
- the partial-save path writes `.partial.csv` files during long runs.

## Known Bottleneck

The current persona-conditioned iterated RPS runs are slow because the LOLA correction is implemented with nested finite differences through a stateful iterated policy. This is scientifically acceptable for small runs, but it is the main blocker before a serious large-scale campaign.

## Next Suggested Runs

1. Replace oracle EW weights with online variance estimates.
2. Speed up the iterated LOLA path before launching larger RPS or mixed-environment runs.
3. Replace the current constant-`lambda` basin map with an annealed-vs-constant comparison if we want one figure that jointly speaks to geometry and asymptotic correctness.
4. Move to one medium-scale benchmark family from the large-scale plan.
