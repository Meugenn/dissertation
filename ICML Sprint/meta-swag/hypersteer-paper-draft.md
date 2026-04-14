# Posterior HyperSteer: Goodhart-Resilient Bayesian Hypernetwork Steering at Scale

This is a parallel paper draft. It does not replace the current Meta-SWAG paper. The strategic purpose of this draft is different: rather than positioning Meta-SWAG as a broad unification story across MARL and LLM alignment, it positions our method as a concrete improvement to the current AxBench state of the art, HyperSteer.

## Abstract

HyperSteer is the current state of the art activation-steering method on AxBench, but it remains a point-estimate method: a single hypernetwork checkpoint predicts a single steering vector, and steering strength is selected by a mean-score sweep without any explicit notion of posterior uncertainty, evidence weighting, or robustness to oversteering. We introduce **Posterior HyperSteer**, a Bayesian extension of HyperSteer that places a SWAG-style posterior over the late hypernetwork trajectory and uses Goodhart-resilient weighting to prevent collapse onto proxy-maximizing checkpoints. The method combines three ingredients: a posterior over retained HyperSteer checkpoints, ESS-constrained or thresholded evidence weighting over held-out AxBench validation scores, and an omega-inspired training view in which evidence quality and alignment stability explicitly shape updates. At inference time, Posterior HyperSteer performs uncertainty-aware factor selection by maximizing a risk-sensitive objective over posterior predictive steering performance. This changes the role of uncertainty from a diagnostic to a control signal. We argue that this is the right next step for HyperSteer because AxBench already exposes the exact failure mode that Bayesian steering should address: stronger steering often improves concept relevance while degrading instruction following and fluency. The resulting paper is benchmark-first, empirically tractable, and directly comparable to the current state of the art.

## 1. Introduction

Activation steering has become one of the most compelling ways to control language models without full fine-tuning. Among steering methods on AxBench, HyperSteer is currently the strongest activation-steering baseline: a shared hypernetwork maps a natural-language steering prompt, optionally together with base-prompt context and base-model activations, to a steering vector that is injected into the residual stream. This gives HyperSteer a crucial advantage over per-concept supervised steering methods. It amortizes steering-vector generation across many concepts and generalizes to held-out steering prompts.

That scaling result is real, but HyperSteer still behaves like a point-estimate system. A single trained hypernetwork checkpoint generates one steering vector per prompt pair. Steering strength is then chosen by sweeping a scalar factor and selecting the best aggregate validation score. This means HyperSteer inherits exactly the weakness that now shows up across many alignment and steering settings: the same mechanism that increases task-target relevance can also push the model into brittle or oversteered behavior. In AxBench terms, higher concept relevance can come at the cost of instruction following and fluency. HyperSteer wins the benchmark anyway because its mean behavior is strong, but the method has no explicit way to represent uncertainty over steering vectors, no anti-collapse checkpoint weighting rule, and no uncertainty-aware mechanism for choosing steering strength.

This paper argues that the next strategic move is not to replace HyperSteer, but to **posteriorize** it. We introduce Posterior HyperSteer, a Bayesian extension that applies a Meta-SWAG-style posterior approximation to the HyperSteer training trajectory. The core object is a Gaussian posterior over late hypernetwork checkpoints. Instead of selecting one final checkpoint and one best factor by mean validation score alone, Posterior HyperSteer aggregates retained checkpoints with Goodhart-resilient weighting and then chooses steering strength by balancing posterior mean performance against posterior uncertainty. This makes the method directly responsive to the failure mode that AxBench exposes.

The framing here is intentionally narrower than the broader Meta-SWAG paper. We are not trying to unify MARL and LLM alignment in one move. We are trying to improve the strongest existing AxBench method in a way that is benchmark-relevant, measurable, and plausibly publishable on its own. That narrower framing has three advantages. First, it gives the paper a clean empirical anchor: HyperSteer is the incumbent SOTA. Second, it turns Meta-SWAG from an abstract posterior story into a concrete model improvement. Third, it creates a natural bridge from uncertainty estimation to steering robustness, which is exactly where the current benchmark is most interesting.

Our proposal combines three ideas. First, we place a SWAG-style posterior over retained HyperSteer checkpoints. Second, we use Goodhart-resilient evidence weighting, with ESS-constrained softmax weighting as the main method and thresholded weighting as the main ablation. Third, we reinterpret part of the broader omega-gradient perspective for the single-model steering setting: evidence quality should scale updates, and alignment stability should enter the training signal directly rather than appearing only at evaluation time. The outcome is a method that is still recognizably HyperSteer, but no longer purely point-estimate.

## 2. Why This Is a Strategic Paper Pivot

Focusing a paper around improving HyperSteer is strategically attractive for five reasons.

### 2.1 It starts from a strong benchmark target

HyperSteer is already established as the top activation-steering baseline on AxBench. That means an improvement paper is immediately legible: reviewers do not need to be convinced that the baseline matters.

### 2.2 It makes the empirical story much sharper

The current broad Meta-SWAG paper tries to do several things at once: unify MARL and alignment, introduce a posterior method, and support a theory story. A HyperSteer-focused paper can ask one clean question:

> Can Bayesian posteriorization and uncertainty-aware control make the strongest existing activation-steering method more robust?

That is a much easier paper to explain.

### 2.3 It directly targets the benchmark’s most interesting failure mode

AxBench does not just reward concept relevance. It explicitly scores concept relevance, instruction relevance, and fluency, then combines them by a strict harmonic mean. That means it already operationalizes a Goodhart tradeoff. If a method improves concept steering by destroying fluency, AxBench will expose it. HyperSteer is therefore an unusually good place to test whether posterior methods reduce brittle oversteering rather than only improving an average score.

### 2.4 It is easier to complete empirically

A HyperSteer paper is benchmark-first rather than universe-first. It can win by:

- improving held-out steering scores,
- improving robustness across steering factors,
- improving transfer to AlpacaEval,
- and adding meaningful uncertainty diagnostics.

It does not need to carry the full theoretical burden of the broader Meta-SWAG paper on its own.

### 2.5 It still preserves the broader program

This is not a retreat from Meta-SWAG. It is a stronger first flagship application. If Posterior HyperSteer works, it becomes the cleanest empirical chapter of the broader Meta-SWAG story.

## 3. HyperSteer Setup

We follow the HyperSteer setting closely. Let:

- `x` be a base prompt,
- `s` be a natural-language steering prompt,
- `B` be the frozen base language model,
- `H_\phi` be a hypernetwork with parameters `\phi`,
- `\Delta_\phi(x,s)` be the steering vector predicted by `H_\phi`.

HyperSteer intervenes on a hidden state `h` of `B` by

```math
h \leftarrow h + a \, \Delta_\phi(x,s),
```

where `a > 0` is the steering factor. The model then produces a steered response

```math
\hat{y}_{a,\phi}(x,s) = B\!\left(x \mid h \leftarrow h + a \Delta_\phi(x,s)\right).
```

AxBench evaluates this response using three discrete scores:

- concept relevance `C(a,\phi)`,
- instruction relevance `I(a,\phi)`,
- fluency `F(a,\phi)`.

The benchmark score is the strict harmonic mean

```math
S(a,\phi) = \operatorname{HM}\big(C(a,\phi), I(a,\phi), F(a,\phi)\big).
```

This setup already contains the core steering tradeoff: increasing concept strength can raise `C` while lowering `I` or `F`.

## 4. Where HyperSteer Is Still Weak

HyperSteer’s strengths are clear. It scales well, generalizes to held-out steering prompts, and is compute-efficient compared with training one steering mechanism per concept. But the current method has four important limitations.

### 4.1 It is a point-estimate method

The final system is one hypernetwork checkpoint. There is no posterior over hypernetwork weights, no posterior over generated steering vectors, and no notion of epistemic uncertainty in steering behavior.

### 4.2 Factor selection is mean-only

The steering factor is chosen by whichever value gives the best aggregate validation score. This ignores uncertainty. A factor may have slightly higher mean score but much larger variance across prompts or posterior samples.

### 4.3 It has no anti-collapse checkpoint weighting

If validation metrics improve late in training only because the hypernetwork has overfit to benchmark idiosyncrasies, the current training pipeline has no explicit protection against selecting that collapse-prone region.

### 4.4 Its training objective does not directly encode the full robustness target

HyperSteer is trained with a language-modeling objective on AxBench-generated outputs. That is sensible, but it does not explicitly encode the evidence quality of updates or the stability of steering under factor variation.

## 5. Posterior HyperSteer

We define a posterior approximation over the retained HyperSteer training trajectory.

### 5.1 Retained checkpoint process

Let `\phi_1,\dots,\phi_T` be hypernetwork checkpoints after burn-in, retained from the final portion of training. Each checkpoint `\phi_t` induces a steering map `(x,s) \mapsto \Delta_{\phi_t}(x,s)`.

We retain `K` late checkpoints and define a posterior

```math
q_T(\phi) = \mathcal{N}(\mu_T,\Sigma_T),
```

fit in hypernetwork parameter space by a SWAG-style diagonal-plus-low-rank approximation. When full-weight posteriorization is too heavy, the practical fallback is to posteriorize the output steering vectors or low-rank subspace parameters only.

### 5.2 Evidence scores

Each retained checkpoint receives an evidence score `m_t` computed on a held-out validation slice. The default choice is the checkpoint’s own validation harmonic mean after selecting its own best steering factor on the validation split.

This choice intentionally separates:

- checkpoint weighting,
- factor selection,
- and final test reporting.

That separation is important because otherwise the Goodhart story becomes circular.

### 5.3 Goodhart-resilient weights

We consider five aggregation rules:

- `map`: final retained checkpoint;
- `uniform`: uniform average over retained checkpoints;
- `softmax`: `w_t \propto \exp(\beta m_t)`;
- `ess`: choose `\beta` so that `ESS(w)` stays above a floor;
- `threshold`: keep checkpoints above a quantile threshold and weight uniformly.

Our main recommendation is to make `ess` the primary method and `threshold` the primary ablation.

### 5.4 Posterior predictive steering

Posterior HyperSteer can be used in two ways:

1. **Parameter posterior mode**:
   sample `\phi \sim q_T` and generate `\Delta_\phi(x,s)`.
2. **Vector posterior mode**:
   aggregate or sample steering vectors directly:
   ```math
   \bar{\Delta}(x,s) = \mathbb{E}_{\phi \sim q_T}[\Delta_\phi(x,s)].
   ```

The second route may be cheaper and simpler in practice, but the first is the cleaner Bayesian object.

## 6. Omega-Inspired HyperSteer Training

The broader omega-gradient framework decomposes updates into exploration, exploitation, evidence, alignment, and opponent-shaping terms. Not all of these transfer literally to the single-model steering setting, but two parts transfer cleanly and one part transfers approximately.

### 6.1 Evidence weighting transfers cleanly

Updates should not all be trusted equally. For hypernetwork training, we can attach an evidence weight `w_t` to a batch, concept, or checkpoint based on gradient variance or held-out validation reliability. The simplest update is

```math
\phi_{t+1}
=
\phi_t
-
\eta_t \, w_t \, \nabla_\phi L_t.
```

This is the direct analog of evidence-weighted policy gradient for HyperSteer training.

### 6.2 Alignment transfers cleanly

HyperSteer’s real deployment objective is not raw concept activation. It is a tradeoff among concept relevance, instruction following, and fluency. So the training objective should expose at least part of that tradeoff explicitly:

```math
L_{\text{total}}
=
L_{\text{LM}}
+
\lambda_{\text{align}} L_{\text{align}}
+
\lambda_{\text{stab}} L_{\text{stab}}.
```

Here:

- `L_LM` is the original HyperSteer language-modeling loss;
- `L_align` penalizes instruction/fluency degradation or KL drift;
- `L_stab` penalizes excessive sensitivity to factor scaling.

### 6.3 Opponent-shaping transfers approximately

There is no literal second agent in standard HyperSteer. But there is a bilevel conflict between:

- the steering objective pushing toward stronger concept activation,
- and the base-task objective pushing toward faithful instruction following.

So the relevant transfer is not literal opponent shaping, but **lookahead shaping**: updates should account for how improving concept steering now changes the best attainable instruction-following and fluency after factor selection.

This gives an omega-style update:

```math
\phi_{t+1}
=
\phi_t
-
\eta_t \, w_t
\Big(
\nabla L_{\text{LM}}
+
\lambda \nabla L_{\text{align}}
+
\rho \nabla L_{\text{lookahead}}
\Big).
```

This is the strongest version of the paper’s training story. It is also the part that will need the most care in implementation.

## 7. Uncertainty-Aware Factor Selection

Once we have a posterior over HyperSteer, the steering factor should no longer be chosen only by mean validation score.

For each factor `a` in a finite sweep, define the posterior predictive mean and standard deviation:

```math
\mu(a) = \mathbb{E}_{\phi \sim q_T}[S(a,\phi)],
\qquad
\sigma(a) = \sqrt{\operatorname{Var}_{\phi \sim q_T}[S(a,\phi)]}.
```

Then select

```math
a^*(x,s)
=
\arg\max_{a \in \mathcal{A}}
\bigl(\mu(a) - \tau \sigma(a)\bigr),
```

where `\tau \ge 0` controls the robustness penalty.

This gives a simple interpretation:

- `\tau = 0` recovers mean-only factor selection;
- larger `\tau` prefers factors that are stable across posterior samples;
- high-uncertainty prompts receive gentler steering automatically.

This is, in my view, the single most promising empirical addition to HyperSteer.

## 8. Formal Results We Can State Cleanly

The strongest formal statements for this paper are not broad convergence claims. They are targeted results about posterior collapse and robust factor selection.

### Theorem 1: ESS floors prevent checkpoint collapse

Let `w_1,\dots,w_K` be normalized checkpoint weights with `\sum_k w_k = 1`, and suppose

```math
ESS(w) = \frac{1}{\sum_k w_k^2} \ge B.
```

Then

```math
\max_k w_k \le \frac{1}{\sqrt{B}}.
```

**Proof sketch.** Since `\sum_k w_k^2 \le 1/B` and `\max_k w_k^2 \le \sum_k w_k^2`, we obtain the bound immediately.

**Interpretation.** An ESS floor gives an explicit anti-collapse guarantee. This is much easier to defend than saying “the softmax temperature looked reasonable.”

### Proposition 2: Risk-sensitive factor selection never underperforms its own uncertainty budget

Let

```math
a^* = \arg\max_{a \in \mathcal{A}} \bigl(\mu(a) - \tau \sigma(a)\bigr).
```

Then for every `a \in \mathcal{A}`,

```math
\mu(a^*) \ge \mu(a) - \tau\bigl(\sigma(a) - \sigma(a^*)\bigr).
```

**Interpretation.** Any mean-score loss of the chosen factor is controlled by the uncertainty reduction it buys.

### Proposition 3: Posterior score variance is a first-class steering diagnostic

For any factor `a`,

```math
\operatorname{Var}_{\phi \sim q_T}[S(a,\phi)]
=
\mathbb{E}_{\phi}[\operatorname{Var}(S(a,\phi)\mid \phi)]
+
\operatorname{Var}_{\phi}(\mathbb{E}[S(a,\phi)\mid \phi]).
```

In this setting the second term is the practically interesting one: it measures sensitivity of benchmark performance to checkpoint choice. HyperSteer currently has no analog of this diagnostic.

## 9. Main Experimental Program

### 9.1 Core comparison set

The paper should compare:

- HyperSteer (published baseline);
- HyperSteer with our reproduction code path;
- Posterior HyperSteer `map`;
- Posterior HyperSteer `uniform`;
- Posterior HyperSteer `softmax`;
- Posterior HyperSteer `ess`;
- Posterior HyperSteer `threshold`;
- omega-style HyperSteer without posteriorization;
- combined omega + posterior HyperSteer.

### 9.2 Main benchmarks

Primary benchmark:

- AxBench Concept500 held-out steering prompts.

Secondary benchmarks:

- AxBench held-in;
- Concept10 smoke runs for rapid iteration;
- AlpacaEval transfer for top methods.

### 9.3 Main metrics

Report all existing AxBench metrics:

- steering composite score,
- concept relevance,
- instruction relevance,
- fluency,
- perplexity.

Add new metrics that matter for the Bayesian story:

- factor-robust AUC across steering factors,
- posterior trace,
- top-5 eigenvalues,
- top-eigenvalue / trace ratio,
- ESS,
- max normalized checkpoint weight,
- validation-test factor mismatch,
- delta over unsteered baseline,
- transfer score on AlpacaEval.

### 9.4 Main hypotheses

The key empirical hypotheses are:

1. Posterior HyperSteer improves held-out robustness even if raw mean held-in score changes little.
2. `softmax` weighting is more collapse-prone than `ess` and `threshold`.
3. Uncertainty-aware factor selection reduces oversteering.
4. Omega-style evidence and alignment weighting help held-out generalization and transfer.
5. The combined method improves the robustness side of the AxBench tradeoff without giving away all of HyperSteer’s strength on concept relevance.

## 10. Why This Paper Could Be Stronger Than the Current One

This paper is narrower, but that is a feature.

It is stronger than the current broad Meta-SWAG paper in three ways:

### 10.1 The baseline is concrete

Instead of arguing that Meta-SWAG should matter in principle across many domains, this paper starts from a specific incumbent SOTA method and tries to improve it.

### 10.2 The benchmark is aligned with the method

AxBench is already a structured Goodhart benchmark. It measures the exact tradeoff Posterior HyperSteer is designed to improve.

### 10.3 The contribution is easier to validate

Even a workshop paper can land if it shows:

- a clear improvement or robustness gain over HyperSteer,
- a compelling uncertainty-aware factor-selection result,
- and one or two clean diagnostic figures.

That is a more direct path to impact than trying to close the entire broader Meta-SWAG story in one paper.

## 11. Risks and Caveats

This strategic pivot is strong, but it has risks.

### 11.1 It is less theoretically grand

A HyperSteer-focused paper will be more benchmark-centered and less universal than the current Meta-SWAG draft.

### 11.2 It depends on AxBench execution

The paper becomes much more dependent on the benchmark implementation actually running at scale.

### 11.3 Full omega transfer is not automatic

Evidence weighting and alignment transfer naturally. Full opponent-shaping language does not. The paper should be precise about what parts of omega are literal and what parts are analogical or bilevel reinterpretations.

## 12. Recommended Positioning

The best title direction is something like:

- **Posterior HyperSteer: Goodhart-Resilient Bayesian Hypernetwork Steering at Scale**
- **Bayesian HyperSteer: Uncertainty-Aware Activation Steering with Posterior Factor Selection**
- **HyperSteer-Ω-SWAG: Posterior Hypernetwork Steering for Robust AxBench Control**

My recommendation is the first title. It is the clearest and easiest to review.

The best positioning sentence is:

> We improve HyperSteer, the current strongest activation-steering method on AxBench, by adding a Bayesian posterior over late hypernetwork checkpoints and uncertainty-aware steering-factor selection.

That is simple, concrete, and benchmark-relevant.

## 13. Conclusion

Yes, focusing a parallel paper around HyperSteer is a strategically strong decision. It gives us a sharper baseline, a stronger empirical target, and a more direct path from Meta-SWAG ideas to a result the field can immediately understand. The right version of the paper is not “Meta-SWAG but with HyperSteer mentioned more often.” It is a new paper with a new center of gravity: HyperSteer is the object, Posterior HyperSteer is the method, and AxBench robustness is the test.

If this paper works, it does more than produce one benchmark win. It gives the broader Meta-SWAG program its first truly natural flagship application.

## References

- HyperSteer paper: [2506.03292v1.pdf](</Users/meuge/Downloads/2506.03292v1.pdf>)
- AxBench repo and implementation context: [external/axbench/README.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/external/axbench/README.md:1)
- Current broad Meta-SWAG draft: [paper-draft.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/paper-draft.md:1)
- Master document: [master-report.md](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/master-report.md:1)
