# Meta-SWAG Appendix Draft

This appendix expands the first-draft main text for **Meta-SWAG: Bayesian Posterior Approximation over Markovian Learning Trajectories for Multi-Agent Policy Gradient and LLM Alignment**. It is written to support the current workshop-style draft, not to serve as a final camera-ready appendix.

## Appendix A. Formal Objects and Assumptions

### A.1 Core Objects

We use the following notation throughout.

- `\Theta \subseteq \mathbb{R}^d` is the parameter space of the trainable variables.
- `\theta_t` is the checkpoint at update time `t`.
- `\theta_{t+1} = \mathcal{T}(\theta_t,\xi_{t+1})` is the checkpoint update rule.
- `\mathcal{K}_T = \{t_1,\dots,t_K\}` is the retained post-burn-in index set.
- `\vartheta_k := \theta_{t_k}` is the `k`th retained checkpoint.
- `m_k` is the evidence score attached to `\vartheta_k`.
- `w_k = \psi(m_k) / \sum_j \psi(m_j)` is the normalized checkpoint weight.
- `\mathcal{J}_T \subseteq \{1,\dots,K\}` is the low-rank retention buffer.
- `q_T(\theta) = \mathcal{N}(\mu_T,\Sigma_T)` is the Meta-SWAG posterior.

In MARL, `\theta_t` denotes joint policy parameters. In LLM alignment, `\theta_t` denotes LoRA adapter parameters only; the base model remains frozen.

### A.2 Standing Assumptions

The current paper uses the following standing assumptions.

1. **(A1) Markov checkpoint process.** The post-burn-in checkpoint process is time-homogeneous and Markov on `\Theta`.
2. **(A2) Stable-region retention.** The retained checkpoints lie in a compact region `\mathcal{R} \subseteq \Theta` with finite second and fourth moments.
3. **(A3) Mixing.** The retained process is geometrically mixing on `\mathcal{R}`.
4. **(A4) Local linearization.** On `\mathcal{R}`, the update process admits a first-order linearization around a stable reference point or recurrent orbit, with square-integrable residual noise.
5. **(A5) Evidence regularity.** The score map and weighting rule produce strictly positive normalized weights.

These assumptions are intentionally broad enough to cover both the MARL and alignment views while still being strong enough for local moment arguments.

### A.3 Theorem-Specific Strengthening

- **Theorem 1** additionally assumes evidence-variance alignment: higher evidence implies lower checkpoint-level predictive variance on the retained window.
- **Theorem 2** uses the local linearization and mixing assumptions directly to justify covariance estimation in a stable region.
- **Theorem 3** introduces an explicit self-model `\tilde{\pi}_\theta`; this is an extra modeling object, not part of the basic Meta-SWAG algorithm.

## Appendix B. Posterior Construction Details

### B.1 Evidence Weighting

The draft main text now treats naive softmax weighting as a baseline rather than the default:

```math
w_t = \frac{\exp(\beta m_t)}{\sum_{s=1}^T \exp(\beta m_s)}.
```

Its main practical advantage is smooth weighting, but by itself it is vulnerable to Goodhart-style posterior collapse when `m_t` is a proxy objective. The current implementation therefore supports two Goodhart-resilient alternatives:

- **ESS-constrained softmax weighting**: choose `\beta` so that effective sample size stays above a fixed floor.
- **Thresholded satisficing weighting**: assign equal weight to checkpoints above a validation threshold.

Alternative weight definitions can be substituted later if a more principled marginal-likelihood estimator becomes available.

### B.2 Diagonal-Plus-Low-Rank Covariance

The covariance estimator is

```math
\Sigma_T = \frac{1}{2}\operatorname{diag}(v_T) + \frac{1}{2(|\mathcal{K}|-1)}D_TD_T^\top.
```

The diagonal term controls per-parameter uncertainty; the low-rank term preserves the dominant trajectory subspace. In the LLM setting the computation is performed only in LoRA space, which is critical both computationally and conceptually:

- computationally, because the adapter dimension is tractable;
- conceptually, because the alignment algorithms only modify that subspace.

### B.3 Posterior Prediction

Given samples `\theta^{(s)} \sim q_T`, posterior prediction uses

```math
p(y \mid x, \mathcal{D}) \approx \frac{1}{S}\sum_{s=1}^S p(y \mid x, \theta^{(s)}).
```

For MARL, the same averaging is performed over policy outputs or rollout-level statistics. For LLMs, the default deployment object is a Bayesian model average over LoRA-sampled policies rather than a single merged checkpoint.

## Appendix C. Expanded Theory Notes

### C.1 Theorem 1 Notes

The first theorem is now intentionally stated as an idealized local result. A cleaner proof path is:

1. Define the posterior predictive object and decompose its predictive variance into within-checkpoint and between-checkpoint components.
2. Condition on the retained checkpoint sigma-algebra.
3. Under evidence-variance alignment, compare the weighted within-checkpoint term to the uniformly weighted comparator via the AM-HM inequality.
4. Control the between-checkpoint term separately under the one-region assumption.

This is the mathematically honest version of the intended claim. Without evidence-variance alignment, the theorem should not promise HM/AM improvement.

### C.2 Theorem 2 Notes

The geometry theorem now rests on a local linearization plus mixing argument. A fuller derivation should:

1. define the local reference object, either a fixed point or a recurrent orbit;
2. write the centered dynamics as a linear term plus residual noise;
3. show that the retained empirical covariance converges to the covariance of the linearized process;
4. compare the leading eigenspaces of that covariance with the soft directions of the local geometry.

The theorem does **not** claim global posterior calibration. It claims local geometric faithfulness inside the retained region.

### C.3 Theorem 3 Notes

The self-knowledge theorem is best read as a conceptual limitation theorem, not as a direct guarantee for the implemented weighting rule. The appendix derivation should make that separation explicit.

Start from

```math
D_{KL}(\pi_\theta \,\|\, \hat{\pi}_{q_T})
=
D_{KL}(\pi_\theta \,\|\, \tilde{\pi}_\theta)
+ \mathbb{E}_{\pi_\theta}\left[\log \frac{\tilde{\pi}_\theta}{\hat{\pi}_{q_T}}\right].
```

Then:

1. identify the first term with `L_self(\theta)`;
2. bound the second term using the uniform log-density ratio constant `C`;
3. move from total variation to KL using Pinsker;
4. use Young's inequality to absorb the mixed term into a multiplicative constant on `L_self`;
5. append the finite-sample posterior approximation term `O(T^{-1})`.

The constant `1 + C^2/2` is intentionally simple and workshop-friendly. A tighter constant can be explored later without changing the meaning of the result.

## Appendix D. Experiment Protocols

### D.1 Matrix Games

Planned environments:

- Matching Pennies
- 3-action zero-sum variant
- Kim-style iterated IPD
- Kim-style iterated RPS

Planned comparisons:

- point estimate
- uniformly weighted SWAG over retained checkpoints
- naive softmax Meta-SWAG
- ESS-constrained Meta-SWAG
- thresholded Meta-SWAG

Primary metrics:

- effective predictive variance
- `HM/AM`-predicted versus empirical variance ratio
- effective sample size
- top-eigenvalue concentration of `\Sigma_T`
- basin-width proxy from convergence under perturbed initialization

Intended outputs:

- one two-panel figure in the main paper
- a small appendix table of seeds, checkpoint cadence, and retained window size

### D.2 LLM Alignment

Planned setup:

- base model: Llama-family checkpoint
- adaptation: LoRA only
- aligners: DPO primary, PPO and GRPO extensions
- data: Alpaca-style preference set or equivalent preference benchmark
- evaluation: proxy reward, gold reward model, best-of-`n` gap, posterior predictive variance

Planned comparisons:

- MAP aligned checkpoint
- standard softmax Meta-SWAG
- Goodhart-resilient Meta-SWAG
- baseline Bayesian method where available

The main draft should never collapse this section into a vague promise of future work. If the results are not finished, the paper should still report:

- the checkpoint collection rule;
- the evidence score definition;
- the evaluation metrics;
- the exact intended figure captions;
- which rows are still pending.

### D.3 Main-Text Figure Captions

**Figure 2.** Matrix-game validation of Meta-SWAG. Left: empirical predictive variance ratio compared against the `HM/AM` prediction. Right: posterior covariance spectrum compared against basin-width estimates.

**Figure 3.** LLM alignment validation. Top: proxy-gold reward gap versus best-of-`n` under DPO. Bottom: algorithm-agnostic behavior of Meta-SWAG across DPO, PPO, and GRPO. Unfinished entries marked `RESULT PENDING`.

## Appendix E. LoRA Tracking Implementation Notes

The implementation should track only the trainable LoRA tensors during alignment. A practical checkpointing recipe for the first experimental pass is:

1. save LoRA adapter weights every fixed number of optimizer steps after burn-in;
2. score each retained checkpoint on a held-out alignment split;
3. compute evidence weights from the held-out score;
4. accumulate the weighted mean and diagonal moments online;
5. retain only the latest `K` centered deviations for the low-rank factor;
6. sample LoRA adapters at evaluation time and merge them into the frozen base model.

This design is important for both memory and interpretation. Sampling full model weights would be expensive and would incorrectly imply posterior uncertainty over frozen parameters.

## Appendix F. Overflow Material for a Later Draft

The following items are intentionally deferred from the main text but can be promoted later if results are strong.

- a fuller derivation comparing Meta-SWAG to local Laplace approximations;
- additional plots for PPO and GRPO if the umbrella figure becomes too dense;
- ablations over the evidence-weight temperature `\beta`;
- sensitivity to checkpoint cadence and retained rank;
- richer posterior families such as MultiSWAG or mixtures over multiple basins.
