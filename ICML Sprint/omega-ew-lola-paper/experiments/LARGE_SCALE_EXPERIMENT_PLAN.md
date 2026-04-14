# Large-Scale Experiment Plan

This document scales the current EW-LOLA experiment scaffold into a realistic benchmark programme with up to `$10k` in compute credits.

The goal is not to spend the whole budget by default. The goal is to spend it in stages, with each stage answering a different scientific question and unlocking the next one only if the previous stage succeeds.

## 1. Strategic Principle

The current paper has three empirical needs:

- show that EW matches the variance story;
- show that LOLA matches the basin story;
- show that EW-LOLA is better than either component alone in adaptive multi-agent settings.

Small matrix and iterated games are enough to establish that logic.

Large-scale compute should therefore be used for only one thing:

- testing whether the same qualitative story survives in richer MARL benchmarks.

That means the right spend pattern is:

1. use cheap runs to select the right design;
2. use mid-scale runs to check transfer;
3. use expensive runs only on the strongest benchmark families.

## 2. Benchmark Tiers

### Tier A: Paper-Core Validation

Purpose:

- theorem-facing validation;
- cheap ablations;
- quick iteration on schedules, noise, and weighting.

Benchmarks:

- current matrix games;
- current Kim-style IPD and iterated RPS persona runs.

Current implementation status:

- already in this folder.

Budget:

- `$0` to `$200`.

Why keep it:

- this is where theorem debugging and figure iteration happen;
- this tier should be run before every other tier.

### Tier B: Structured PettingZoo Tasks

Purpose:

- move beyond toy games without paying the price of the hardest social benchmarks;
- test whether EW and EW-LOLA still help in environments with richer observation and coordination structure.

Suggested benchmark family:

- PettingZoo environments, especially its maintained reference suites and simultaneous-action support.

Why PettingZoo:

- the official docs position it as a standard API for MARL with a wide variety of reference environments and both AEC and Parallel interfaces;
- it is a natural bridge from our small hand-built games to larger environments.

Candidate tasks:

- cooperative or mixed-incentive MPE-style tasks;
- simple communication tasks;
- a small competitive subset for shaping stress tests.

Spend target:

- `$500` to `$1,500`.

Success criterion:

- EW-LOLA beats standard PG and at least one ablation on several tasks with multiple seeds;
- runtime is manageable enough to support schedule sweeps.

### Tier C: Many-Agent Scaling

Purpose:

- test whether the method still behaves sensibly when agent count becomes part of the challenge.

Suggested benchmark family:

- MAgent2.

Why MAgent2:

- the official docs describe it as a maintained environment library for large numbers of agents in gridworld battles and related scenarios;
- it is a natural stress test for evidence heterogeneity because agent experiences become much less uniform at scale.

Candidate tasks:

- battlefield-style competitive tasks;
- gathering or resource-control variants if available.

Spend target:

- `$1,000` to `$2,500`.

Success criterion:

- EW or EW-LOLA shows a measurable robustness gain as agent count increases;
- the method remains trainable without fragile per-scenario retuning.

### Tier D: Social Generalisation

Purpose:

- test whether opponent shaping and evidence weighting help in broad social-interaction suites rather than one-off games.

Suggested benchmark family:

- Melting Pot.

Why Melting Pot:

- the official repository describes it as a suite for MARL social generalisation, with over 50 substrates and over 256 test scenarios spanning cooperation, competition, deception, reciprocation, and trust;
- it is the most convincing place to test whether EW-LOLA is doing something generally useful about adaptive social dynamics.

Caution:

- this is the expensive tier;
- it should only be attempted after the method is stable on cheaper suites.

Spend target:

- `$3,000` to `$6,000`.

Success criterion:

- clear transfer from training settings to held-out social scenarios;
- evidence weighting reduces instability or variance across seeds;
- shaping helps in at least a meaningful subset of social substrates.

### Tier E: Cooperative Micromanagement

Purpose:

- evaluate on a benchmark that the wider cooperative MARL community already recognises as serious.

Suggested benchmark family:

- SMACv2.

Why SMACv2:

- the official repository describes it as an updated cooperative MARL benchmark built around decentralised StarCraft II micromanagement;
- compared to SMAC, it adds more randomness through randomised start positions, unit types, and other environment variability.

Caution:

- setup complexity is higher than Tier B and Tier C because of StarCraft dependencies;
- this tier is best reserved for a later validation round or for a stronger camera-ready version.

Spend target:

- `$2,000` to `$4,000`.

Success criterion:

- EW-LOLA remains competitive with strong baselines under stochastic scenario variation;
- the added randomness actually matters for our variance and stability story.

## 3. Recommended Spend Sequence

Do not allocate the full budget up front.

Recommended staged release:

### Stage 1: `$500`

- finish Tier A cleanly;
- get matrix and Kim persona figures into paper-ready shape;
- run a few larger CPU or small-GPU sweeps if helpful.

Decision gate:

- if the theory-facing figures are still noisy or ambiguous, do not move on.

### Stage 2: `$1,500`

- implement Tier B with a small number of PettingZoo tasks;
- compare standard, EW, LOLA, and EW-LOLA with fixed training budgets;
- identify which environments actually separate the four methods.

Decision gate:

- if EW-LOLA does not beat at least one ablation consistently, stop and revisit method design.

### Stage 3: `$2,000`

- run many-agent scaling on MAgent2;
- use this to answer whether evidence weighting matters more as the population grows.

Decision gate:

- continue only if the method shows a scaling story that is not already obvious from Tier B.

### Stage 4: `$4,000` to `$6,000`

- commit to either Melting Pot or SMACv2 as the flagship large-scale benchmark, not both at once.

Preferred choice:

- Melting Pot if the paper's story is primarily about adaptive social interaction and generalisation;
- SMACv2 if the audience wants a more standard cooperative MARL benchmark.

## 4. Preferred Scientific Narrative

If the budget is used well, the final narrative becomes:

1. small games validate the theorem-level claims;
2. PettingZoo tasks show the method is not confined to toy settings;
3. one flagship large-scale benchmark shows the method remains useful in a serious modern MARL suite.

That is enough for a strong story. It is not necessary to benchmark every MARL environment that exists.

## 5. What To Measure At Scale

At large scale, do not limit metrics to raw return.

Track:

- mean return across seeds;
- variance across seeds;
- time to threshold performance;
- instability rate or collapse rate;
- sensitivity to initialisation;
- optional evidence diagnostics if the implementation exposes them cleanly.

Why:

- EW is fundamentally a variance-control story;
- LOLA is fundamentally a dynamics and basin story;
- pure mean return hides both.

## 6. Baseline Set

Minimum baseline set for every tier:

- standard PG or the closest baseline implementation;
- EW only;
- LOLA only;
- EW-LOLA.

Optional strong baselines for later tiers:

- MAPPO-style strong cooperative baseline where appropriate;
- environment-standard baselines provided by the benchmark suite;
- parameter-sharing ablations if agent homogeneity is relevant.

Do not add many baselines until the four-method comparison is stable.

## 7. Implementation Priorities

Before starting Tier B and above, the current scaffold should improve in this order:

1. online variance estimation instead of oracle EW weights;
2. more efficient LOLA gradients for iterated games;
3. proper plotting scripts and result tables;
4. a simple job launcher and result manifest;
5. benchmark adapters for one larger framework.

That order matters because otherwise large-scale compute will be spent on a prototype with the wrong weighting and poor observability.

## 8. Practical Recommendation

Given the current state of the code, the best use of the next block of time is:

1. finish the paper-core sweeps with the Kim persona runner;
2. improve the current scaffold so EW weights are estimated online;
3. choose one medium benchmark family from PettingZoo first;
4. only then commit serious budget to Melting Pot or SMACv2.

This keeps the large-scale campaign evidence-driven instead of aspirational.

## 9. Source Notes

The benchmark recommendations above were checked against current official sources:

- PettingZoo docs: [pettingzoo.farama.org](https://pettingzoo.farama.org/index.html)
- BenchMARL repository: [github.com/facebookresearch/BenchMARL](https://github.com/facebookresearch/BenchMARL)
- Melting Pot repository: [github.com/google-deepmind/meltingpot](https://github.com/google-deepmind/meltingpot)
- SMACv2 repository: [github.com/oxwhirl/smacv2](https://github.com/oxwhirl/smacv2)
- MAgent2 docs: [magent2.farama.org](https://magent2.farama.org/)
