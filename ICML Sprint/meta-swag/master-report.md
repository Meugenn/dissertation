# Meta-SWAG Master Report

This document is the unlimited-length, non-ICML master document for the Meta-SWAG project. It is intended to be the single cross-referenced source of truth across theory, literature, experiments, implementation status, open problems, and paper-writing decisions.

It is not a polished submission draft. It is a working research record.

## 1. How to use this document

Use this file for four purposes:

- to understand the current state of the project without piecing together multiple notes;
- to check what has been implemented versus what is still conceptual;
- to locate the exact files, artifacts, and benchmarks relevant to each claim;
- to support future compression into ICML-style paper drafts, workshop submissions, or a longer technical report.

## 2. Companion documents

- Workshop-style paper draft: [paper-draft.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/paper-draft.md:1)
- HyperSteer-focused parallel paper draft: [hypersteer_main.tex](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/hypersteer_main.tex:1)
- HyperSteer drafting note: [hypersteer-paper-draft.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/hypersteer-paper-draft.md:1)
- ICML-style LaTeX draft: [main.tex](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/main.tex:1)
- Current PDF: [main.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/main.pdf)
- Living implementation log: [technical-report.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/technical-report.md:1)
- Literature note: [literature-review.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/literature-review.md:1)
- AxBench benchmark note: [axbench-benchmark-plan.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/axbench-benchmark-plan.md:1)
- Experiment scaffold README: [experiments/README.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/README.md:1)
- Bibliography seed: [references.bib](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/references.bib:1)

## 3. Current project snapshot

### 3.1 One-sentence thesis

Meta-SWAG adapts SWAG-style posterior approximation to adaptive, non-stationary Markovian learning trajectories, with the goal of preserving useful uncertainty and reducing over-concentration in both multi-agent learning and LLM alignment.

### 3.2 Current state in one page

- The theory and paper framing now center on a Markovian learning trajectory rather than an optimizer-as-agent or strict meta-MDP claim.
- The ICML-style paper exists and compiles, but it remains a constrained workshop draft rather than the full project record.
- A second paper track now exists that focuses specifically on improving HyperSteer on AxBench rather than on the full Meta-SWAG unification story.
- Matrix-game experiments exist and run end-to-end, but they are exploratory and do not yet validate the desired variance-reduction theorem.
- Kim-style iterated IPD and RPS experiments exist as an exact-dynamics scaffold and produce meaningful exploratory comparisons across weighting rules.
- A serious AxBench and AlpacaEval implementation path now exists in code, including adapter checkpoint retention, posterior fitting over LoRA parameters, aggregation rules, diagnostics, and a benchmark runner.
- A first Posterior HyperSteer implementation path now exists in code, including retained hypernetwork checkpoints, posterior aggregation, and a dedicated benchmark runner.
- The AxBench runner is unit-tested and CLI-verified, but it has not yet been validated through a full real-model benchmark run because the external dependency stack and model/runtime setup are still incomplete.

### 3.3 Maturity breakdown

| Area | Status | Notes |
| --- | --- | --- |
| Core idea | strong | The method concept and motivating problem are coherent. |
| Paper framing | medium-strong | The Markovian trajectory pivot materially improved rigor and defensibility. |
| Formal theory | medium | The setup is more formal now, but some theorem layers remain provisional. |
| Matrix-game experiments | exploratory | Good for plumbing and diagnostics, not final evidence. |
| Kim-style iterated experiments | exploratory but promising | Stronger than the matrix scaffold, still smoke-test scale. |
| AxBench implementation | substantial code complete | Not yet benchmark-validated end-to-end. |
| Posterior HyperSteer implementation | early but real | Runner and tests exist; benchmark-grade runs still pending. |
| LLM empirical evidence | incomplete | The code path exists, but no benchmark-grade results yet. |

## 4. Conceptual framing

### 4.1 Naming and framing

The current project keeps the method name `Meta-SWAG`, but the framing has shifted away from “the optimizer is an agent” language.

The recommended interpretation is:

- the training process induces a sequence of checkpoints `\theta_t`;
- these checkpoints form a Markovian learning trajectory under an update operator `\mathcal{T}`;
- vanilla SWAG assumes a locally stationary regime that is often too optimistic in adaptive multi-agent and alignment settings;
- Meta-SWAG modifies checkpoint retention and weighting so the posterior remains meaningful under adaptive training dynamics.

### 4.2 Why this pivot matters

The earlier meta-MDP wording invited objections of the form:

- DPO or PPO is a fixed update rule, not an optimizing agent;
- the training process is not obviously an MDP in the standard RL sense;
- the paper was leaning on analogy more heavily than it needed to.

The current Markovian trajectory framing keeps the useful mathematics while removing that unnecessary vulnerability.

### 4.3 Central posterior object

The central object is still a Gaussian posterior approximation:

`q_T(\theta) = \mathcal{N}(\mu_T, \Sigma_T)`.

For LLM experiments, `\theta` refers only to retained LoRA adapter parameters, not the full base-model parameter vector.

### 4.4 Weighting rules now under consideration

The project currently supports five aggregation rules:

- `map`: final retained checkpoint only;
- `uniform`: uniform average across retained checkpoints;
- `softmax`: `w_t \propto \exp(\beta m_t)` with fixed `\beta`;
- `ess`: softmax weighting with `\beta` chosen to satisfy a minimum effective sample size target;
- `threshold`: uniform weighting over checkpoints whose validation metric exceeds a quantile threshold.

The current default theoretical and empirical preference is:

- use ESS-constrained weighting as the main Goodhart-resilient method;
- keep thresholding as a simple ablation or fallback variant;
- treat naive softmax weighting as the collapse-prone baseline.

## 5. Formal setup status

### 5.1 What is already in the paper

The formal setup in [main.tex](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/main.tex:1) now includes:

- a checkpoint state space `\Theta`;
- a Markov update rule `\theta_{t+1} = \mathcal{T}(\theta_t, \xi_{t+1})`;
- a retained checkpoint set `\mathcal{K}_T`;
- a weighting map over retained checkpoints;
- local assumptions on mixing, stability, and evidence regularity.

### 5.2 What is mathematically strongest right now

The strongest parts of the current formalism are:

- the trajectory-based setup itself;
- the posterior construction over retained checkpoints;
- the ESS-constrained weighting rule as a clean anti-collapse mechanism;
- the diagonal-plus-low-rank covariance representation over LoRA trajectories.

### 5.3 What is still weaker than the rest

The weaker parts of the current formal layer are:

- Theorem 1 still depends on an explicit assumption linking evidence scores to predictive variance;
- Theorem 2 is strongest as a local geometry claim, not yet as a universal basin theorem;
- the self-knowledge theorem is best interpreted as a conceptual limitation result rather than a directly operational theorem about the implemented procedure.

### 5.4 Recommended formal direction

If this is going to become a mathematically serious RL paper rather than only a workshop draft, the best next move is:

1. define theorem-specific primitives in the appendix rather than relying on one global setup;
2. turn the strongest theorem into a full proposition with explicit assumptions;
3. move weaker claims into conjectures, limitations, or discussion if full proof quality is not yet there.

## 6. Literature positioning

### 6.1 Direct technical lineage

The project currently sits at the intersection of:

- SWAG and subspace posterior approximation;
- policy-gradient dynamics in stochastic games and meta-learning;
- Bayesian or uncertainty-aware alignment methods for LoRA and reward modeling;
- reward overoptimization and Goodhart-style failure modes in alignment.

Core references already collected locally include:

- [papers/maddox19-swag.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/maddox19-swag.pdf)
- [papers/kim21g-meta-mapg.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/kim21g-meta-mapg.pdf)
- [papers/giannou22-policy-gradient-stochastic-games.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/giannou22-policy-gradient-stochastic-games.pdf)
- [papers/yang24-laplace-lora.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/yang24-laplace-lora.pdf)
- [papers/yang24-bayesian-reward-models.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/yang24-bayesian-reward-models.pdf)
- [papers/gao23h-reward-overoptimization.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/gao23h-reward-overoptimization.pdf)
- [papers/rafailov23-dpo.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/rafailov23-dpo.pdf)
- [papers/rafailov24-daa-overoptimization.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/rafailov24-daa-overoptimization.pdf)
- [papers/schulman17-ppo.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/schulman17-ppo.pdf)

### 6.2 What seems most novel

The project’s most interesting potential novelty is not simply “Bayesian LoRA” and not simply “SWAG for RL.”

The strongest version of the claim is:

- use SWAG-style posterior approximation over adaptive training trajectories;
- make the weighting rule explicitly Goodhart-aware;
- operationalize the posterior in LoRA space for alignment-style fine-tuning;
- study posterior geometry and collapse diagnostics as part of the method rather than as after-the-fact analysis.

### 6.3 What needs to stay cautious

The project should still avoid overclaiming on:

- “first” claims unless checked very carefully;
- any statement that LLM alignment training is literally a meta-MDP in the strict game-theoretic sense;
- any theorem language that is stronger than the appendix can support;
- any empirical claim about LLM robustness until AxBench or related benchmark results are complete.

## 7. Source references from the broader project

This Meta-SWAG work is drawing on three local source documents in particular:

- project plan: [resources/meta-swag-plan-v2.docx](/Users/meuge/coding/maynard/ICML%20Sprint/resources/meta-swag-plan-v2.docx)
- original technical report reference: [resources/technical_report.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/resources/technical_report.pdf)
- dissertation reference material: [resources/dissertation%20extended.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/resources/dissertation%20extended.pdf)

The older technical report matters especially because it anchors:

- the original variance-reduction and EW-PG motivation;
- the matrix-game experimental shape;
- the later Kim-style iterated-game follow-up experiments;
- the expectation that some original simulation code once lived at `simulations/{full_experiment, fixed_point_ne, iterated_games}.py`.

## 8. Experiment program overview

### 8.1 Experiment families currently in scope

The project now spans three empirical tiers:

- matrix-game experiments inspired by the original technical report;
- Kim-style iterated IPD and RPS experiments using bundled persona files from the Meta-MAPG repo;
- an in-progress AxBench and AlpacaEval benchmark track for LoRA-based LLM evaluation.

### 8.2 Artifact locations

- Matrix and iterated-game artifacts: [experiments/artifacts](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/artifacts)
- Experiment runners: [experiments](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments)

## 9. Matrix-game experiment scaffold

### 9.1 Purpose

The matrix-game scaffold exists to test the basic Meta-SWAG plumbing under controlled conditions before moving into more complex adaptive settings.

### 9.2 Relevant code

- Runner: [experiments/run_matrix_games.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/run_matrix_games.py:1)
- Game definitions: [experiments/meta_swag/games.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/games.py:1)
- Simulator: [experiments/meta_swag/simulate.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/simulate.py:1)
- Posterior utilities: [experiments/meta_swag/posterior.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/posterior.py:1)
- Metrics: [experiments/meta_swag/metrics.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/metrics.py:1)

### 9.3 Current results

Primary result table from [matrix_games_summary.csv](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/artifacts/matrix_games_summary.csv:1):

| Game | Variance | Scheme | HM/AM Ratio | Point Var | Posterior Var | Variance Ratio | Basin Proxy | Top Eigenvalue | ESS | Beta | Threshold |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| matching_pennies | `[1,1]` | `ess` | 1.0000 | 0.022984 | 0.046257 | 2.0808 | 0.000122 | 0.848601 | 100.0000 | 7.9489 |  |
| matching_pennies | `[1,1]` | `softmax` | 1.0000 | 0.022984 | 0.047284 | 2.1784 | 0.000000 | 0.874119 | 164.7244 | 3.0000 |  |
| matching_pennies | `[1,1]` | `threshold` | 1.0000 | 0.022984 | 0.044182 | 1.9971 | 0.000000 | 0.806511 | 50.0000 |  | -0.1657 |
| matching_pennies | `[20,1]` | `ess` | 0.1814 | 0.029270 | 0.037086 | 2.1499 | 0.000000 | 24.668017 | 100.0000 | 9.9688 |  |
| matching_pennies | `[20,1]` | `softmax` | 0.1814 | 0.029270 | 0.039608 | 2.1783 | 0.000000 | 24.019993 | 150.3076 | 3.0000 |  |
| matching_pennies | `[20,1]` | `threshold` | 0.1814 | 0.029270 | 0.034080 | 1.9022 | 0.001141 | 20.441557 | 50.0000 |  | -0.7152 |
| matching_pennies | `[5,1]` | `ess` | 0.5556 | 0.031727 | 0.041607 | 2.3254 | 0.000000 | 5.493212 | 100.0000 | 7.8487 |  |
| matching_pennies | `[5,1]` | `softmax` | 0.5556 | 0.031727 | 0.043263 | 2.4489 | 0.000000 | 5.503706 | 142.0088 | 3.0000 |  |
| matching_pennies | `[5,1]` | `threshold` | 0.5556 | 0.031727 | 0.040089 | 2.2026 | 0.001319 | 4.718776 | 50.0000 |  | -0.5990 |
| rock_paper_scissors | `[1,1]` | `ess` | 1.0000 | 0.013736 | 0.022143 | 3.8143 | 2.113212 | 1.485971 | 100.0000 | 6.2138 |  |
| rock_paper_scissors | `[1,1]` | `softmax` | 1.0000 | 0.013736 | 0.023712 | 3.8971 | 2.101348 | 1.553496 | 154.9413 | 3.0000 |  |
| rock_paper_scissors | `[1,1]` | `threshold` | 1.0000 | 0.013736 | 0.018270 | 4.0904 | 55.079720 | 1.345274 | 50.0000 |  | -0.5717 |
| rock_paper_scissors | `[20,1]` | `ess` | 0.1814 | 0.013898 | 0.019577 | 1521.6329 | 20.043153 | 24.477003 | 100.0000 | 6.0667 |  |
| rock_paper_scissors | `[20,1]` | `softmax` | 0.1814 | 0.013898 | 0.020594 | 1021.4890 | 6.538281 | 25.933051 | 156.1967 | 3.0000 |  |
| rock_paper_scissors | `[20,1]` | `threshold` | 0.1814 | 0.013898 | 0.018646 | 1691.2569 | 35403.492093 | 17.353851 | 50.0000 |  | -1.0295 |
| rock_paper_scissors | `[5,1]` | `ess` | 0.5556 | 0.013342 | 0.021413 | 578.4790 | 48.640754 | 6.234432 | 100.0000 | 6.4783 |  |
| rock_paper_scissors | `[5,1]` | `softmax` | 0.5556 | 0.013342 | 0.020086 | 260.7981 | 4.164568 | 6.427904 | 159.6464 | 3.0000 |  |
| rock_paper_scissors | `[5,1]` | `threshold` | 0.5556 | 0.013342 | 0.016887 | 1555.7828 | 29523.410260 | 4.450528 | 50.0000 |  | -0.9678 |

### 9.4 Interpretation

The matrix-game scaffold is useful mainly as an engineering and diagnostic artifact.

What it does show:

- the full retention and posterior-fitting path works;
- weighting rules matter;
- the posterior geometry reacts strongly to heterogeneity;
- low-rank eigenvalue diagnostics are behaving nontrivially.

What it does not yet show:

- a convincing empirical validation of the intended variance-reduction theorem;
- a stable geometry metric that can be used as final paper evidence;
- faithful reproduction of the original dissertation technical-report experiments.

### 9.5 Relevant diagnostics

The eigenvalue logs are stored in [matrix_games_eigenvalues.csv](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/artifacts/matrix_games_eigenvalues.csv:1).

The first few rows already show that the posterior geometry is strongly concentrated in a narrow subspace for some settings, but the current interpretation remains diagnostic rather than theorem-validating.

## 10. Kim-style iterated-game experiments

### 10.1 Purpose

The iterated-game path is closer to the project’s adaptive-learning story than the static matrix-game scaffold.

It uses the bundled persona files from the Kim et al. Meta-MAPG repository and therefore gives a more grounded benchmark for adaptive behavior.

### 10.2 Relevant code

- Persona integration: [experiments/meta_swag/kim_reference.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/kim_reference.py:1)
- Iterated game dynamics: [experiments/meta_swag/iterated_games.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/iterated_games.py:1)
- Runner: [experiments/run_kim_iterated.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/run_kim_iterated.py:1)
- External reference repo: [external/meta-mapg](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/external/meta-mapg)

### 10.3 Current results

Primary result table from [kim_iterated_summary.csv](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/artifacts/kim_iterated_summary.csv:1):

| Env | Scheme | Final Objective | Best Checkpoint | Objective Mean | Objective Std | ESS | Beta | Threshold | Top Eigenvalue | Cooperation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ipd | `ess` | 8.015747 | 8.022176 | 8.011932 | 0.005469 | 30.000000 | 60.638439 |  | 0.023663 | 0.997058 |
| ipd | `map` | 8.015747 | 8.022176 | 8.022176 | 0.000000 | 1.000000 |  |  | 0.000000 | 0.997543 |
| ipd | `softmax` | 8.015747 | 8.022176 | 7.986889 | 0.011374 | 59.910365 | 1.000000 |  | 0.051518 | 0.995900 |
| ipd | `threshold` | 8.015747 | 8.022176 | 8.015879 | 0.002577 | 15.000000 |  | 8.010392 | 0.014402 | 0.997252 |
| rps | `ess` | 12.995768 | 12.998468 | 12.966773 | 0.011018 | 30.000000 | 22.822222 |  | 0.043764 |  |
| rps | `map` | 12.995768 | 12.998468 | 12.998468 | 0.000000 | 1.000000 |  |  | 0.000000 |  |
| rps | `softmax` | 12.995768 | 12.998468 | 12.898719 | 0.025566 | 59.076213 | 1.000000 |  | 0.119742 |  |
| rps | `threshold` | 12.995768 | 12.998468 | 12.982681 | 0.005870 | 15.000000 |  | 12.964601 | 0.018083 |  |

### 10.4 Interpretation

These results are more encouraging than the matrix-game scaffold.

The most important qualitative pattern is:

- `softmax` is the most collapse-prone weighting rule in the current smoke tests;
- `ess` and `threshold` stay much closer to MAP objective values;
- `softmax` also produces larger posterior variance and larger top eigenvalues;
- in IPD, cooperation remains very high across methods, but `softmax` is still the weakest of the posterior schemes on objective mean.

This is not yet final evidence, but it is consistent with the central Goodhart-resilience story.

### 10.5 Remaining gaps

The current iterated-game runs are still smoke-test scale.

They still need:

- larger seed counts;
- benchmark-scale trajectory logging;
- adaptation AUC metrics;
- time-resolved traces rather than only endpoint summaries;
- closer alignment with the original report’s experiment definitions.

## 11. AxBench and AlpacaEval benchmark track

### 11.1 Why this benchmark matters

AxBench is currently the best candidate benchmark for the LLM part of the project because it naturally exposes:

- steering-factor sensitivity;
- component metrics beyond a single scalar;
- oversteering behavior;
- LoRA and preference-LoRA training paths;
- a natural route to AlpacaEval transfer testing.

See [axbench-benchmark-plan.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/axbench-benchmark-plan.md:1) for the more focused planning note.

### 11.2 Current implementation status

The AxBench Meta-SWAG implementation now has substantial code behind it.

Core files:

- Adapter state flattening and restore: [experiments/meta_swag/adapter_state.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/adapter_state.py:1)
- Posterior aggregation and ESS solver: [experiments/meta_swag/adapter_posterior.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/adapter_posterior.py:1)
- External repo tracking and import helpers: [experiments/meta_swag/axbench_runtime.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/axbench_runtime.py:1)
- Retention-aware LoRA and preference-LoRA training: [experiments/meta_swag/axbench_meta_swag.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/meta_swag/axbench_meta_swag.py:1)
- Benchmark runner: [experiments/run_axbench_meta_swag.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/run_axbench_meta_swag.py:1)
- Tests: [tests/test_meta_swag_axbench.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/tests/test_meta_swag_axbench.py:1)

### 11.3 What is implemented

The current AxBench path already supports:

- adapter-only checkpoint retention;
- retention over the final fraction of optimizer steps;
- checkpoint metadata collection;
- post hoc validation factor sweeps for retained checkpoints;
- five aggregation rules: `map`, `uniform`, `softmax`, `ess`, `threshold`;
- posterior diagnostics including ESS, max weight, trace, and top eigenvalues;
- AxBench validation/test summaries with delta over unsteered;
- optional AlpacaEval transfer outputs for promoted methods;
- external-repo provenance logging by URL and git SHA.

### 11.4 What has been verified so far

The following verification has already been completed:

- the runner CLI parses and exposes the expected options;
- unit tests for adapter flattening/restoration, ESS solving, threshold weighting, retention scheduling, and checkpoint metadata all pass;
- the dependency manifest and concept-selection path were fixed to be robust to AxBench’s `concept_id = -1` negative rows;
- the runner now writes concept-local artifact bundles rather than re-dumping global tables into each concept directory.

### 11.5 What remains blocked

The remaining blockers are external, not conceptual:

- the external AxBench stack has several runtime dependencies beyond the local smoke environment;
- a full real-model run still needs the right HF model/runtime configuration;
- the full import path is heavier than the isolated unit-tested Meta-SWAG modules.

So the status is:

- implementation path exists;
- unit-tested core logic exists;
- benchmark-grade end-to-end run has not yet been completed.

### 11.6 Benchmark design currently intended

The intended benchmark structure remains:

- smoke test: standard `LoRA`, 10-concept slice, 1 seed;
- main run: `PreferenceLoRA`, `concept500`, 3 seeds, Gemma-2B-style configuration;
- transfer: AlpacaEval for the best AxBench methods plus MAP and unsteered baselines.

### 11.7 Why this is still exciting

This benchmark path is especially interesting because it would let us test the strongest version of the LLM claim:

- standard checkpoint weighting collapses toward over-optimized proxy behavior;
- ESS-constrained or thresholded Meta-SWAG keeps support over multiple competent checkpoints;
- that flatter posterior support improves the robustness side of the accuracy-robustness tradeoff.

## 12. Current code verification summary

### 12.1 Passing tests

The AxBench-specific unit test file is:

- [tests/test_meta_swag_axbench.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/tests/test_meta_swag_axbench.py:1)

Covered checks include:

- adapter flatten/unflatten is lossless;
- frozen base-model parameters are excluded from retained state;
- retention schedules stay within the intended tail of training;
- ESS target solving is stable within tolerance;
- threshold weighting remains normalized and non-empty;
- posterior aggregation returns the expected shapes and diagnostics;
- factor selection and checkpoint metric attachment behave as intended.

### 12.2 Current interpretation

This is strong engineering evidence for the core Meta-SWAG machinery, but not yet empirical evidence for the full benchmark claim.

## 13. Key project-wide findings so far

### 13.1 Findings that already seem real

- The Markovian learning trajectory framing is better than the earlier meta-MDP wording for this paper.
- Goodhart-aware checkpoint weighting is not an optional detail; it is central to the method.
- ESS-constrained weighting is the strongest current candidate for the main method.
- Threshold weighting is a useful ablation because it makes anti-collapse behavior very explicit.
- In the current iterated-game smoke tests, naive softmax weighting is meaningfully worse than ESS or thresholding.
- Posterior geometry diagnostics are worth keeping because they capture something the scalar metrics miss.

### 13.2 Findings that are not yet secure

- The variance-reduction theorem is not yet empirically validated by the matrix scaffold.
- The basin-geometry theorem is not yet in a final proof-ready state.
- No strong LLM-alignment claim should be made until AxBench or related runs are complete.
- The current matrix-game results should not be used as headline evidence in a final paper.

## 14. Main weaknesses and risks

### 14.1 Theory risks

- theorem strength may still exceed proof maturity in places;
- some assumptions are still more reasonable than canonical;
- the self-knowledge result may remain better as a limitation principle than as a central theorem.

### 14.2 Empirical risks

- matrix-game evidence may never become the cleanest validation path;
- AxBench setup cost may be high enough to slow iteration;
- benchmark-specific gains may fail to transfer to broader instruction quality;
- if weighting and evaluation are not kept separate, the Goodhart story can become circular.

### 14.3 Paper-writing risks

- the current workshop paper can still overstate maturity if not carefully toned;
- too many ideas can make the submission feel broader than the evidence supports;
- the project may be strongest as a theory-plus-preliminary-evidence workshop piece before it becomes a fully mature conference paper.

## 15. Paper improvement priorities

### 15.1 Best immediate paper changes

1. keep the workshop paper narrow and honest;
2. use the master document as the full source of truth;
3. move only the strongest theorem into fully assertive language;
4. replace placeholder figure boxes with compact real tables where possible;
5. make the LLM section explicit about what is implemented versus what is still pending.

### 15.2 Best immediate empirical changes

1. scale the Kim-style iterated runs first;
2. keep the AxBench path moving in parallel;
3. treat matrix games as a diagnostic sandbox, not the centerpiece;
4. add factor-sensitivity and eigenvalue plots once real LLM runs begin.

## 16. Recommended next actions

### 16.1 Research direction

1. decide whether the next public-facing paper should be:
   - a focused workshop paper with completed MARL evidence and LLM implementation status, or
   - a broader paper that waits for real AxBench results.
2. formalize one theorem fully rather than stretching three to equal status prematurely.

### 16.2 Experimental direction

1. push the Kim-style iterated benchmark from smoke-test scale to report-scale runs;
2. complete the first genuine AxBench smoke test with a real model/runtime setup;
3. export exact arrays and diagnostics before investing in polished plots;
4. add AlpacaEval only after AxBench outputs are stable.

### 16.3 Writing direction

1. keep the ICML paper as the compressed submission artifact;
2. maintain this master document as the exhaustive project record;
3. use the technical report as the execution log rather than the primary narrative.

## 17. Cross-reference index

### 17.1 Theory and paper

- [paper-draft.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/paper-draft.md:1)
- [hypersteer_main.tex](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/hypersteer_main.tex:1)
- [hypersteer-paper-draft.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/hypersteer-paper-draft.md:1)
- [main.tex](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/main.tex:1)
- [appendix.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/appendix.md:1)

### 17.2 Experiments

- [experiments/README.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/README.md:1)
- [experiments/run_matrix_games.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/run_matrix_games.py:1)
- [experiments/run_kim_iterated.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/run_kim_iterated.py:1)
- [experiments/run_axbench_meta_swag.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/run_axbench_meta_swag.py:1)

### 17.3 Results

- [experiments/artifacts/matrix_games_summary.csv](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/artifacts/matrix_games_summary.csv:1)
- [experiments/artifacts/matrix_games_eigenvalues.csv](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/artifacts/matrix_games_eigenvalues.csv:1)
- [experiments/artifacts/kim_iterated_summary.csv](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/artifacts/kim_iterated_summary.csv:1)
- [experiments/artifacts/kim_iterated_eigenvalues.csv](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/experiments/artifacts/kim_iterated_eigenvalues.csv:1)

### 17.4 Planning and literature

- [technical-report.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/technical-report.md:1)
- [literature-review.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/literature-review.md:1)
- [axbench-benchmark-plan.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/axbench-benchmark-plan.md:1)
- [references.bib](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/references.bib:1)

## 18. Bottom line

The project is no longer just a paper outline. It now has:

- a coherent theory-first framing;
- real experiment scaffolds and exploratory results;
- a promising adaptive-games benchmark path;
- a serious AxBench implementation path for LLM evaluation;
- enough structure to support either a workshop paper now or a stronger later paper after the benchmark track matures.

This master report should now be treated as the primary long-form reference document for the project.
