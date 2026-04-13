# Meta-SWAG Technical Report

This file is the living technical report for the Meta-SWAG project. Unlike the paper draft, this document is intended to track ongoing implementation, experiment status, intermediate results, issues, and next actions.

Primary companion files:

- Paper draft: [paper-draft.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/paper-draft.md:1)
- LaTeX paper: [main.tex](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/main.tex:1)
- Current PDF: [main.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/main.pdf)
- Literature review: [literature-review.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/literature-review.md:1)
- AxBench benchmark note: [axbench-benchmark-plan.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/axbench-benchmark-plan.md:1)
- Experiment scaffold: [experiments/README.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/README.md:1)
- Bibliography seed: [references.bib](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/references.bib:1)

## 1. Project Status

### 1.1 Current objective

Build a first credible empirical and theoretical package for **Meta-SWAG**:

- a workshop-style paper draft built around Markovian learning trajectories;
- a literature-grounded Bayesian positioning story;
- executable first experiments for matrix games;
- a path toward Kim et al. 2021-style iterated-game experiments and later LLM alignment experiments.

### 1.2 Current status summary

- Paper draft exists in markdown and LaTeX.
- ICML-style PDF compiles locally with a local workshop-style class scaffold.
- The LaTeX draft now uses real in-text citations and BibTeX rather than a hand-written reference list.
- Literature review and local paper archive have been created.
- Matrix-game experiment scaffold is implemented and runs end-to-end.
- Goodhart-resilient weighting schemes are now part of the experiment scaffold.
- Kim et al. persona bundles are integrated locally and ready for iterated-game follow-up runs.
- Initial results exist, but **should be treated as exploratory only**.

### 1.3 What is done

- Drafted the main paper with:
  - Markovian learning trajectory framing for MARL and LLM alignment;
  - Meta-SWAG posterior definition;
  - three theorem blocks and alignment corollaries;
  - placeholder-ready experiments section.
- Created a LaTeX version and compiled a PDF.
- Reviewed literature and downloaded the core adjacent papers into [papers](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers).
- Built a first experiment package under [experiments](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments).

### 1.4 What is not done

- No original dissertation simulation code has been recovered from this workspace.
- No benchmark-scale iterated-game study is implemented yet; only a first exact-dynamics runner exists.
- No LoRA checkpoint collection or LLM posterior fitting is implemented yet.
- No final paper-quality empirical claim is justified by current numbers.

## 2. Source Grounding

### 2.1 Key external references currently downloaded

- [papers/maddox19-swag.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/maddox19-swag.pdf)
- [papers/kim21g-meta-mapg.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/kim21g-meta-mapg.pdf)
- [papers/giannou22-policy-gradient-stochastic-games.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/giannou22-policy-gradient-stochastic-games.pdf)
- [papers/yang24-laplace-lora.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/yang24-laplace-lora.pdf)
- [papers/yang24-bayesian-reward-models.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/yang24-bayesian-reward-models.pdf)
- [papers/gao23h-reward-overoptimization.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/gao23h-reward-overoptimization.pdf)
- [papers/rafailov23-dpo.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/rafailov23-dpo.pdf)
- [papers/rafailov24-daa-overoptimization.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/rafailov24-daa-overoptimization.pdf)
- [papers/schulman17-ppo.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/schulman17-ppo.pdf)

### 2.2 Local project references used so far

- [resources/meta-swag-plan-v2.docx](/Users/meuge/coding/maynard/ICML%20Sprint/resources/meta-swag-plan-v2.docx)
- [resources/technical_report.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/resources/technical_report.pdf)
- [resources/dissertation extended.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/resources/dissertation%20extended.pdf)

### 2.3 Relevant technical report notes

From `technical_report.pdf`, the most relevant experiment references for Meta-SWAG follow-up are:

- Experiment 1: EW-PG variance reduction
  - Matching Pennies and 3-action variants
  - heterogeneity settings `V = [1,1], [5,1], [20,1]`
- Experiments 9-12:
  - iterated Prisoner's Dilemma adaptation
  - iterated Rock-Paper-Scissors exploitation
  - strategy identification
  - scaling to `N > 2`
- Report note:
  - code is said to exist at `simulations/{full_experiment, fixed_point_ne, iterated_games}.py`
  - these files are **not present in this workspace**

## 3. Current Experiment Code

### 3.1 Files

- [experiments/run_matrix_games.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/run_matrix_games.py:1)
- [experiments/meta_swag/games.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/games.py:1)
- [experiments/meta_swag/configs.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/configs.py:1)
- [experiments/meta_swag/simulate.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/simulate.py:1)
- [experiments/meta_swag/posterior.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/posterior.py:1)
- [experiments/meta_swag/metrics.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/metrics.py:1)
- [experiments/meta_swag/policies.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/policies.py:1)

### 3.2 What the scaffold currently does

- simulates two-player matrix games with policy logits;
- injects controlled Gaussian noise to emulate heteroskedastic gradient estimates;
- retains late checkpoints;
- computes softmax, ESS-constrained, or thresholded Meta-SWAG mean and covariance;
- samples posterior checkpoints;
- writes per-seed and averaged summaries;
- logs low-rank eigenvalue diagnostics;
- emits one summary plot for the paper scaffold.

### 3.3 Games currently implemented

- Matching Pennies
- Rock-Paper-Scissors
- Stag Hunt
- Prisoner's Dilemma

The current runner focuses on Matching Pennies and Rock-Paper-Scissors because these are closest to the report's Experiment 1 description.

## 4. Current Artifacts

Generated artifacts:

- [experiments/artifacts/matrix_games_metrics.csv](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/artifacts/matrix_games_metrics.csv)
- [experiments/artifacts/matrix_games_summary.csv](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/artifacts/matrix_games_summary.csv)
- [experiments/artifacts/matrix_games_summary.png](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/artifacts/matrix_games_summary.png)
- [experiments/artifacts/matrix_games_eigenvalues.csv](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/artifacts/matrix_games_eigenvalues.csv)
- [experiments/artifacts/kim_iterated_metrics.csv](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/artifacts/kim_iterated_metrics.csv)
- [experiments/artifacts/kim_iterated_summary.csv](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/artifacts/kim_iterated_summary.csv)
- [experiments/artifacts/kim_iterated_eigenvalues.csv](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/artifacts/kim_iterated_eigenvalues.csv)

Current exploratory summary table from the first pass:

| Game | Variance Setting | HM/AM Ratio | Point Variance | Posterior Variance | Variance Ratio | Basin Proxy | Top Posterior Eigenvalue |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| matching_pennies | `[1,1]` | 1.0000 | 0.02298 | 0.04728 | 2.1784 | 0.0000 | 0.8741 |
| matching_pennies | `[20,1]` | 0.1814 | 0.02927 | 0.03961 | 2.1783 | 0.0000 | 24.0200 |
| matching_pennies | `[5,1]` | 0.5556 | 0.03173 | 0.04326 | 2.4489 | 0.0000 | 5.5037 |
| rock_paper_scissors | `[1,1]` | 1.0000 | 0.01374 | 0.02371 | 3.8971 | 2.1013 | 1.5535 |
| rock_paper_scissors | `[20,1]` | 0.1814 | 0.01390 | 0.02059 | 1021.4890 | 6.5383 | 25.9331 |
| rock_paper_scissors | `[5,1]` | 0.5556 | 0.01334 | 0.02009 | 260.7981 | 4.1646 | 6.4279 |

## 5. Interpretation of Current Results

### 5.1 What the current run tells us

- The code path works end-to-end.
- The posterior covariance reacts strongly to increasing heterogeneity.
- The experiment artifact generation is usable for iteration.

### 5.2 What the current run does **not** justify

- It does **not** currently support the paper's desired variance-reduction claim.
- The empirical variance ratios are not tracking the expected HM/AM direction.
- The geometry metric is still too crude to serve as a theorem validation.
- The zero-sum settings remain fragile under the current objective and metric choices.

### 5.3 Honest assessment

These are **pipeline-validation numbers**, not paper numbers.

The current scaffold is useful because it establishes:

- a reproducible simulation layout;
- a Meta-SWAG fitting path;
- artifact generation;
- a place to iterate.

It is not yet a faithful reproduction of the dissertation technical-report experiments.

The main value of the current run is methodological:

- weighting-scheme plumbing now works end-to-end;
- ESS and threshold rules expose posterior-collapse tradeoffs directly;
- low-rank eigenvalue logging is in place for later subspace diagnostics.

### 4.1 Kim-style iterated-game first pass

A first exact-dynamics adaptation runner is now implemented for bundled Kim personas:

- IPD adaptation against cooperative and defective persona families;
- RPS adaptation against rock, paper, and scissors persona families;
- comparisons across `map`, `softmax`, `ess`, and `threshold` weighting;
- artifact logging for posterior ESS, posterior objective mean/std, and top-5 low-rank eigenvalues at 5 trajectory snapshots.

The current defaults are intentionally lightweight for iteration speed rather than benchmark scale. They are sufficient to validate the code path and generate honest tables, but not yet to support final paper claims.

## 6. Known Issues

### 6.1 Missing original code

The technical report references simulation files that are absent from this workspace:

- `simulations/full_experiment.py`
- `simulations/fixed_point_ne.py`
- `simulations/iterated_games.py`

Recovering those would materially improve fidelity.

### 6.1a Kim et al. reference repo is now available

The Meta-MAPG GitHub repository has now been cloned into:

- [external/meta-mapg](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/external/meta-mapg)

Useful findings:

- bundled IPD persona files exist locally;
- bundled RPS persona files exist locally;
- environment definitions for iterated IPD and iterated RPS are directly inspectable;
- this gives us a much stronger base for implementing report Experiments 9-12 faithfully.

### 6.2 Metric mismatch

Current variance metrics are based on posterior-sampled predictive variability over a simplified objective. That is likely too far from the report's actual distance-to-Nash measurement and may be why the qualitative pattern is off.

### 6.3 Geometry proxy is weak

The current eigenvalue and basin-proxy measurements are useful as engineering diagnostics, but they are not yet aligned to a benchmark-grade LLM robustness story. This is one reason AxBench is attractive: its factor sweeps and component metrics give us a cleaner place to connect posterior concentration with observed oversteering.

## 7. AxBench Opportunity

### 7.1 Why it matters

The earlier HyperSteer-Weight project suggests a very relevant empirical pattern for Meta-SWAG:

- benchmark-specific gains do not necessarily transfer to broader instruction quality;
- factors below `1.0` can outperform nominal full-strength steering;
- train loss and perplexity can decouple from benchmark performance;
- the unsteered baseline can already score nontrivially due to prompt-concept leakage.

Those observations line up unusually well with the current Meta-SWAG story around posterior collapse, Goodhart pressure, and LoRA-only posteriors.

### 7.2 Why AxBench fits this project

AxBench is a strong candidate first LLM benchmark because it already provides:

- steering-factor sweeps;
- concept relevance, instruction relevance, and fluency metrics;
- harmonic-mean composite scoring;
- LoRA and preference-LoRA model hooks;
- a direct route to AlpacaEval-style generalization checks.

In other words, it gives us an empirical arena where Meta-SWAG can be tested as a method for reducing over-concentration in adaptive training trajectories, not just as another benchmark optimizer.

### 7.3 Recommended first target

The best first AxBench experiment is likely:

- run Meta-SWAG over `PreferenceLoRA` checkpoint trajectories;
- compare `map`, uniform late-checkpoint averaging, naive softmax weighting, ESS-constrained weighting, and thresholded weighting;
- evaluate on AxBench first and then AlpacaEval for transfer.

If that path proves too heavy initially, the fallback smoke test is the same pipeline over standard `LoRA`.

### 7.4 Working note

See the dedicated note at [axbench-benchmark-plan.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/axbench-benchmark-plan.md:1) for the detailed rationale, metrics, and implementation path.

## 8. Next Recommended Actions

1. Add checkpoint-retention hooks to an AxBench LoRA or PreferenceLoRA training run.
2. Export retained adapter parameters into the existing Meta-SWAG posterior-fitting code path.
3. Compare `map`, `softmax`, `ess`, and `threshold` on AxBench with explicit factor sweeps and leakage-aware baselines.
4. Carry the best variants to AlpacaEval to measure generalization rather than only benchmark fit.

## 9. Additional Gaps

### 9.1 No iterated-game benchmark yet

The most paper-relevant empirical next step is still missing:

- iterated Prisoner's Dilemma adaptation from Kim et al.
- iterated RPS exploitation from Kim et al.

### 9.2 ICML class file

The paper PDF uses a local ICML-style scaffold, not the official upstream workshop template.

## 10. Proposed Experiment Roadmap

### 10.1 Immediate next experiments

1. **Report-faithful Experiment 1**
   - Rework the matrix-game metric to mirror the report's distance-to-Nash tracking.
   - Add 50-run aggregation and episode-based traces.
   - Plot convergence curves instead of only end-state summary scatter plots.

2. **Kim-style iterated IPD**
   - Increase the current first-pass runner from a smoke test to a benchmark-scale run.
   - Add adaptation AUC and trajectory-level cooperation traces, not just endpoint summaries.
   - Compare MAP, vanilla softmax Meta-SWAG, and Goodhart-resilient weighting under larger seed counts.

3. **Kim-style iterated RPS**
   - Increase the current first-pass runner from a smoke test to a benchmark-scale run.
   - Compare posterior sampling versus point estimates for exploitation stability.
   - Extend diagnostics from endpoint reward to time-resolved exploitability curves.

### 10.2 Medium-term experiments

4. **Strategy identification**
   - Map learned IPD policies to named strategies through state-conditional cooperation probabilities.

5. **Scaling to more players**
   - Extend zero-sum and coordination-style games to `N = 3, 4`.

6. **LoRA posterior tracking**
   - Create a checkpoint collection utility for aligned LLM LoRA weights.
   - Fit Meta-SWAG over retained adapters.

### 10.3 LLM alignment experiments

7. **DPO first**
   - Save LoRA checkpoints along a DPO run.
   - Score checkpoints on held-out preference or reward metrics.
   - Fit Meta-SWAG in LoRA space.
   - Compare MAP versus posterior averaging under best-of-`n`.

8. **Algorithm-agnostic extension**
   - Repeat with PPO and GRPO after DPO works.

## 11. Action Log

### 2026-04-13

- Created the first paper draft in markdown.
- Created the LaTeX paper and compiled the PDF.
- Performed a literature review and downloaded adjacent papers.
- Added a matrix-game experiment scaffold.
- Aligned the scaffold more closely to `technical_report.pdf` Experiment 1.
- Ran the first end-to-end matrix-game experiment.
- Created this technical report.
- Cloned the Kim et al. 2021 Meta-MAPG repository into `external/meta-mapg`.
- Added a local loader for bundled IPD and RPS personas from the Meta-MAPG repo.
- Added Goodhart-resilient ESS and threshold weighting to the Meta-SWAG posterior fitter.
- Added low-rank eigenvalue logging for matrix-game runs.
- Added a first Kim-style iterated IPD/RPS adaptation runner with artifact export.
- Performed a critical paper pass and started implementing fixes: softened theorem claims, clarified Goodhart evaluation separation, and replaced the manual bibliography with BibTeX citations.

## 12. Decisions and Defaults

- The paper remains **theory-first**.
- The core trajectory framing is now **Markovian learning trajectories**, not optimizer-as-agent language.
- The LLM experiment remains **placeholder-ready** until real runs exist.
- The technical report should prefer **honest state tracking** over polished language.
- Intermediate failures and negative results should be recorded here rather than hidden.

## 13. Open Requests / Dependencies

- Recover any dissertation simulation code if available.
- If available, import the Kim et al. 2021 reference GitHub implementation.
- If available, replace the local ICML scaffold with the official workshop template.
- Add a direct GRPO citation before making GRPO a headline empirical claim in the paper.
