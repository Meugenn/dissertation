# Meta-SWAG on AxBench: Notes and Experiment Plan

This note records what we can reuse from the earlier HyperSteer-Weight project and how it translates into a credible Meta-SWAG benchmark program.

## 1. What the earlier project teaches us

The old project is useful less for its exact method and more for its empirical instincts.

### 1.1 Separate benchmark fit from generalization

The strongest design choice in the earlier project was:

- evaluate in-domain behavior on AxBench;
- evaluate out-of-domain or broader instruction quality on AlpacaEval.

That separation is directly relevant to Meta-SWAG. If Meta-SWAG improves only the AxBench composite score but hurts broader response quality, it will look like another benchmark optimizer rather than a robustness method.

### 1.2 Oversteering is real

The project reports that steering factors below `1.0` often outperform the raw learned intervention. That is a concrete warning sign:

- the nominally "best" steering update is often too sharp;
- stronger interventions can degrade instruction following and fluency;
- post-training selection needs an anti-collapse mechanism.

This is almost exactly the empirical phenomenon our Goodhart-resilient Meta-SWAG weighting is designed to address.

### 1.3 Train loss is not enough

The old writeup also observed that lower CE loss and lower perplexity did not reliably translate into better AxBench or AlpacaEval performance. That is important for Meta-SWAG because:

- checkpoint quality should not be defined by training loss alone;
- posterior weighting should use held-out validation signals;
- we should log both optimization metrics and benchmark metrics, then show they diverge.

### 1.4 Baseline leakage must be measured explicitly

The previous project noted that unsteered models can receive nontrivial AxBench scores because concepts partially overlap with prompts. This means all Meta-SWAG runs should include:

- unsteered baseline;
- MAP checkpoint baseline;
- relative improvement over unsteered;
- absolute score.

Otherwise, small gains are hard to interpret.

### 1.5 LoRA is the right parameterization

The old project operated in a LoRA-style low-rank parameter space. That matches our current Meta-SWAG design choice very well:

- the posterior can be defined over adapter parameters only;
- checkpoint storage becomes practical;
- covariance diagnostics become interpretable and computationally feasible.

## 2. Why AxBench is a particularly good fit for Meta-SWAG

AxBench is attractive because it is not just a leaderboard. It gives us a controlled environment for testing posterior selection under adaptive training trajectories.

### 2.1 It already has the right metrics

AxBench already evaluates:

- concept relevance;
- instruction relevance;
- fluency;
- harmonic-mean composite score;
- perplexity.

This matters because Meta-SWAG is meant to balance competence and robustness, not just maximize one scalar reward.

### 2.2 It already supports steering-factor sweeps

The benchmark code is built around steering factors and explicit factor selection. That is very useful for us because it creates a natural place to compare:

- point estimate / MAP;
- naive softmax-weighted Meta-SWAG;
- ESS-constrained Meta-SWAG;
- thresholded Meta-SWAG.

If the naive weighting collapses toward oversteered behavior while ESS or thresholding stays flatter, that becomes direct empirical evidence for the Goodhart-resilience story.

### 2.3 It already has LoRA and preference-training hooks

The AxBench codebase already includes:

- a generic `BaseModel` interface in [external/axbench/axbench/models/model.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/external/axbench/axbench/models/model.py:1);
- LoRA steering models in [external/axbench/axbench/models/lora.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/external/axbench/axbench/models/lora.py:1);
- preference-based LoRA models in [external/axbench/axbench/models/preference_lora.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/external/axbench/axbench/models/preference_lora.py:1).

This means we do not need to invent a benchmark harness from scratch. We can adapt Meta-SWAG as an AxBench-compatible method.

### 2.4 It matches the current paper's LoRA posterior story

Our paper already says the LLM posterior lives over LoRA parameters. AxBench gives us:

- a benchmark where LoRA is already a first-class object;
- concept-level and preference-style training modes;
- a ready-made evaluation layer.

That makes it one of the cleanest empirical bridges between the theory paper and actual LLM experiments.

## 3. Most promising Meta-SWAG experiment variants

There are three realistic ways to use AxBench.

### 3.1 Variant A: Meta-SWAG over preference-LoRA trajectories

Train AxBench's `PreferenceLoRA` model on DPO-style data, save the final `K` checkpoints, fit a posterior over LoRA parameters, and compare:

- MAP;
- uniform late-checkpoint averaging;
- naive softmax weighting;
- ESS-constrained weighting;
- thresholded weighting.

This is the best first target because:

- it is close to our alignment story;
- it plugs naturally into the Goodhart section of the paper;
- it uses the exact parameter subset we want to model.

### 3.2 Variant B: Meta-SWAG over plain LoRA steering trajectories

Train AxBench's `LoRA` baseline, retain late checkpoints, and fit Meta-SWAG over the adapter trajectory.

This is weaker conceptually than Variant A, but it is operationally simpler and could become the first smoke test.

### 3.3 Variant C: Meta-SWAG over HyperSteer-style hypernetwork weights

Apply Meta-SWAG to the hypernetwork or low-rank bases in a HyperSteer-Weight-style model.

This is interesting but should not be the first implementation:

- the parameterization is more complex;
- it introduces more moving parts;
- it makes it harder to isolate whether improvements come from Meta-SWAG or from the steering architecture itself.

## 4. Recommended benchmark stack

The cleanest stack is:

- Primary benchmark: AxBench
- Generalization check: AlpacaEval
- Optional later extension: Best-of-`N` reward-overoptimization on alignment-style data

### 4.1 AxBench should answer

- Does Meta-SWAG improve the composite steering score over the MAP checkpoint?
- Does it reduce oversteering sensitivity across factor sweeps?
- Does Goodhart-resilient weighting preserve instruction relevance and fluency better than naive weighting?
- Does posterior geometry reveal narrow brittle trajectories versus broad robust ones?

### 4.2 AlpacaEval should answer

- Do gains on AxBench transfer to broader instruction-following quality?
- Does Meta-SWAG keep generalization flatter than the point estimate?

## 5. Concrete metrics to log

For every method, log:

- AxBench composite score
- concept relevance
- instruction relevance
- fluency
- perplexity
- absolute score
- delta over unsteered baseline
- best factor on validation split
- test score at chosen factor

For Meta-SWAG specifically, also log:

- retained checkpoint count
- effective sample size
- max normalized checkpoint weight
- posterior trace
- top-5 low-rank eigenvalues
- top-eigenvalue / trace ratio
- score variance across posterior samples

These extra diagnostics are important because our method claims to manage posterior concentration, not just improve a single downstream metric.

## 6. Goodhart-resilient evaluation design

To keep the experiment honest, we should separate three roles:

- weighting metric: the signal used to define checkpoint weights;
- model-selection metric: the signal used to choose steering factor or stopping point;
- final evaluation metric: the reported benchmark score.

Recommended default:

- weighting metric: held-out instruction relevance or a capped composite validation score;
- model-selection metric: AxBench composite score on a validation split;
- final evaluation metric: AxBench test composite score plus component metrics.

This avoids the failure mode where we weight and evaluate on the same proxy with no guardrail against collapse.

## 7. Immediate implementation path

### 7.1 Phase 1: low-risk smoke test

Use AxBench's existing LoRA model and:

- save late training checkpoints;
- export LoRA parameters at each retained step;
- fit Meta-SWAG over those parameters;
- evaluate MAP vs uniform vs softmax vs ESS vs threshold.

### 7.2 Phase 2: preferred main experiment

Repeat the same pipeline for `PreferenceLoRA`, which better matches the DPO-alignment framing in the paper.

### 7.3 Phase 3: generalization check

Take the best AxBench methods and run AlpacaEval on the corresponding steered models to measure whether Meta-SWAG buys robustness rather than benchmark gaming.

## 8. Code hooks that look promising

The most useful integration points appear to be:

- AxBench model interface: [external/axbench/axbench/models/model.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/external/axbench/axbench/models/model.py:1)
- standard LoRA model: [external/axbench/axbench/models/lora.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/external/axbench/axbench/models/lora.py:1)
- preference LoRA model: [external/axbench/axbench/models/preference_lora.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/external/axbench/axbench/models/preference_lora.py:1)
- steering evaluation logic: [external/axbench/axbench/scripts/evaluate.py](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/external/axbench/axbench/scripts/evaluate.py:1)

The evaluator already aggregates:

- judge-based concept relevance;
- judge-based instruction relevance;
- fluency;
- harmonic means over those metrics.

That means the shortest path is probably to implement Meta-SWAG as a new AxBench model wrapper rather than inventing a separate evaluation pipeline.

## 9. Recommended paper angle if this works

If the results are good, the strongest claim is not:

- "Meta-SWAG gets a higher benchmark score."

The stronger claim is:

- "Meta-SWAG improves steering quality while reducing posterior collapse and oversteering sensitivity on AxBench, with better transfer to AlpacaEval than naive checkpoint selection."

That is much more distinctive and much closer to the paper's actual thesis.
