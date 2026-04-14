# Evidence-Weighted and Opponent-Shaped Policy Gradient with Composed Convergence Guarantees

Working first draft for the NExT-Game workshop paper.

Scope note: this draft is written from the April 13 plan, the dissertation convergence chapters, the March 2026 Omega technical report, Kim et al. (2021), and Foerster et al. (2018). The EW and LOLA sections are source-aligned with the technical report. The composition theorem in Section 5 is the new paper-specific claim and should be checked line-by-line against the final appendix proof before submission.

Companion note: for provenance, extended proof blueprints, experiment inventory, and writing crosswalks, see `MASTER_REPORT.md`.

## Abstract

Policy-gradient methods in general stochastic games now admit local convergence guarantees to stable Nash policies, but two practical weaknesses remain. First, agents often learn from observations of uneven quality, so equal-weight updates let the noisiest gradients dominate the joint dynamics. Second, even when a stable Nash policy exists, the basin of attraction may be small, making convergence sensitive to initialisation. We study two modifications that address these weaknesses without leaving the convergence regime of Giannou et al. (2022). Evidence-weighted policy gradient (EW-PG) rescales each agent's update by an inverse-variance evidence weight and reduces the effective gradient variance by a factor `HM(V) / AM(V) <= 1`. Opponent-shaped policy gradient (LOLA-PG) adds a one-step opponent-shaping term and, under an annealed shaping schedule, preserves local convergence while enlarging the stable basin when the shaping Hessian satisfies a spectral reinforcement condition. Our main contribution is a composition theorem: under the union of the EW and LOLA assumptions, EW-LOLA-PG inherits both the variance reduction and the basin enlargement. The result is supported by matrix-game and iterated-game experiments adapted from the Omega technical report and the Kim et al. (2021) benchmark family, and by a Lean 4 formalisation of the algebraic backbone of the method.

## 1. Introduction

Policy-gradient learning in multi-agent systems long lacked trajectory-level convergence guarantees outside narrow game classes. Giannou et al. (2022) changed that picture by proving local convergence to stable Nash policies in general stochastic games under a stable second-order condition. That theorem gives a clean baseline, but it does not by itself solve two problems that matter in practice. The first is heteroskedasticity across agents: gradient estimators can have sharply different variances because agents observe different histories and occupy different informational positions in the game. The second is geometric: local convergence only helps when initialisation lands in the attraction region of the target equilibrium, and that region may be small.

This paper studies two modifications that target those problems on orthogonal axes. Evidence-weighted policy gradient (EW-PG) scales each agent's update by a Keynesian evidence weight proportional to the inverse of its gradient variance. In the Omega technical report this yields a simple harmonic-mean versus arithmetic-mean variance improvement while preserving Giannou's convergence rate constant up to the same factor. Learning with Opponent-Learning Awareness (LOLA) takes a different route. Foerster et al. (2018) introduced a shaping term that anticipates how the current agent changes the opponent's next update. The original paper showed strong behavioural effects in iterated prisoner's dilemma and convergence to Nash play in repeated matching pennies, but it did not provide a general local convergence theorem for stochastic games. The technical report adds that missing layer by annealing the shaping schedule and proving both rate preservation and, under a spectral reinforcement condition, an enlarged stable basin.

The main claim of this paper is that these two improvements compose cleanly. EW rescales update magnitudes. LOLA changes the update direction by adding an opponent-shaped correction. Under the same local regularity conditions used in the separate analyses, these operations do not obstruct one another: the EW factor keeps the variance improvement, the annealed LOLA term keeps the bias under control, and the spectral basin improvement is inherited from the opponent-shaping Hessian. The resulting algorithm, EW-LOLA-PG, is a workshop-sized distillation of the broader Omega-gradient programme.

We make four contributions.

- We restate the Giannou et al. convergence setup in the minimal form needed for a workshop audience.
- We isolate the EW-PG result as a variance-reduction theorem with preserved local convergence.
- We isolate the LOLA-PG result as a rate-preserving opponent-shaping method with a basin-enlargement theorem.
- We prove a new composition theorem showing that EW-LOLA-PG inherits both improvements under the union of the separate assumptions.

Empirically, the paper focuses on the cleanest validation suite for those claims: matrix games with controlled noise heterogeneity, iterated prisoner's dilemma, iterated rock-paper-scissors, and N-player iterated RPS. The first set tests the harmonic-mean prediction. The second and third test shaping and composition on the Kim et al. (2021) benchmark family. The fourth checks whether the benefit survives beyond the two-player setting.

## 2. Preliminaries

We work with finite stochastic games in the tabular episodic setting. A game is given by

```math
G = (N, S, \{A_i\}_{i \in N}, \{R_i\}_{i \in N}, P, \rho),
```

where `N` is the set of agents, `S` is the state space, `A_i` is agent `i`'s action set, `R_i` is its reward function, `P` is the transition kernel, and `\rho` is the initial-state distribution. Each agent chooses a stationary Markov policy `\pi_i \in \Pi_i = \Delta(A_i)^{|S|}`, and the joint policy space is `\Pi = \prod_i \Pi_i`.

For each agent `i`, let `V_{i,\rho}(\pi)` denote the expected episodic return under joint policy `\pi`, and let

```math
v_i(\pi) = \nabla_{\pi_i} V_{i,\rho}(\pi)
```

be the policy gradient. Writing `v(\pi) = (v_i(\pi))_{i \in N}`, the projected policy-gradient update takes the form

```math
\pi_{n+1} = \operatorname{proj}_{\Pi}(\pi_n + \gamma_n \hat v_n),
```

where `\hat v_n` is a stochastic gradient estimator, such as REINFORCE, and `\gamma_n = \gamma / (n + m)^p`.

The relevant equilibrium notion is the stable Nash policy from Giannou et al. (2022). We use the second-order stationary refinement because it yields a quantitative drift inequality. A Nash policy `\pi^*` satisfies the stable second-order condition if there exists `\mu > 0` such that, for all `\pi` in a neighbourhood of `\pi^*`,

```math
\langle v(\pi), \pi - \pi^* \rangle \leq - \mu \| \pi - \pi^* \|^2.
```

This is the stable opponent-sensitive condition used to prove local convergence. Under the regularity assumptions in Giannou et al. (2022), if the initial policy lies in the relevant neighbourhood and the stochastic bias and variance exponents satisfy the Robbins-Monro style conditions of their theorem, then projected policy gradient converges to `\pi^*` with

```math
\mathbb{E}[\| \pi_n - \pi^* \|^2 \mid E] = O(C / n^q),
```

for an event `E` of high probability and exponent `q = \min(\ell_b, p - 2 \ell_\sigma)`.

We only need three ingredients from that framework.

- A local drift inequality near the target Nash policy.
- A bias bound for the gradient estimator compatible with the step-size schedule.
- A variance bound that lets the martingale term be controlled at rate `O(n^{-q})`.

Everything in the sequel is built to preserve those three ingredients while improving either variance or basin geometry.

## 3. Evidence-Weighted Policy Gradient

### 3.1 Motivation

In multi-agent learning the agents do not observe the game through equally informative histories. Some sit in stable information states with low-variance gradient estimates; others face noisier observations or more weakly identified best responses. Standard joint policy gradient ignores that heterogeneity. Every agent receives the same global step-size even when its estimator is much noisier than the rest. The result is a familiar failure mode: the shared update is bottlenecked by the noisiest component.

The Omega technical report proposes a simple correction. Let `V_{i,n} = \operatorname{Var}(\hat v_{i,n})` denote the variance of agent `i`'s gradient estimator at iteration `n`, and define

```math
w_{i,n} = \frac{V_{\min,n}}{V_{i,n}}, \qquad V_{\min,n} = \min_j V_{j,n}.
```

Agents with the lowest variance receive weight `1`, and noisier agents are downweighted proportionally. This is the evidence-weighting step. The label "evidence" is meant literally: a lower-variance estimator carries stronger evidence about the local policy improvement direction.

### 3.2 Update Rule and Convergence

The EW-PG update is

```math
\pi_{i,n+1} = \operatorname{proj}_{\Pi}(\pi_{i,n} + \gamma_n w_{i,n} \hat v_{i,n}).
```

The practical algorithm estimates `V_{i,n}` with an exponential moving average of recent gradient fluctuations. In the report, the variance tracker is

```math
\hat V_{i,n} \leftarrow \alpha \hat V_{i,n-1} + (1-\alpha)\|\hat v_{i,n} - \bar v_i\|^2,
```

with `\hat V_{\min,n} = \min_i \hat V_{i,n}` and `w_{i,n} = \hat V_{\min,n} / \hat V_{i,n}`.

The convergence statement is inherited directly from the report.

**Theorem 3.1 (Convergence of EW-PG).** Under the hypotheses of Giannou et al. (2022, Theorem 1), EW-PG converges to a stable Nash policy with

```math
\mathbb{E}[\| \pi_n - \pi^* \|^2 \mid E]
= O\!\left(\frac{\operatorname{HM}(V)}{\operatorname{AM}(V)} \cdot \frac{C}{n^q}\right).
```

The rate exponent stays the same. The gain appears in the leading constant, which shrinks by the harmonic-to-arithmetic mean ratio. The local convergence machinery remains intact.

### 3.3 Variance Improvement

The core algebraic fact is elementary and useful.

**Theorem 3.2 (AM-HM variance improvement).** Let `V_i = \operatorname{Var}(\hat v_i)` denote the per-agent gradient variances. Under EW-PG, the effective variance of the joint update satisfies

```math
\sigma_w^2
= \frac{\operatorname{HM}(V)}{\operatorname{AM}(V)} \sigma_{\mathrm{std}}^2
\leq \sigma_{\mathrm{std}}^2,
```

with equality if and only if all `V_i` are equal.

**Proof sketch.** The weighted variance sum is `\sum_i w_i^2 \sigma_i^2`. Substituting `w_i = V_{\min}/V_i` and `\sigma_i^2 \propto V_i` yields the harmonic-mean normalisation. The inequality is then exactly the AM-HM inequality.

This is the cleanest part of the paper because it connects a very practical choice to a sharp closed-form prediction. If all agents are equally informative, EW does nothing. If evidence is heterogeneous, EW removes variance without changing the target equilibrium.

## 4. Opponent-Shaping Policy Gradient

### 4.1 The LOLA Correction

Foerster et al. (2018) introduced Learning with Opponent-Learning Awareness as a way to account for the fact that the current agent changes the opponent's next update. In its simplest form, the method augments the policy gradient with a one-step opponent-shaping term,

```math
\mathrm{OS}_i(\pi)
= \sum_{j \ne i} \frac{\partial R_i}{\partial \theta_j'} \frac{\partial \theta_j'}{\partial \theta_i},
```

where `\theta_j' = \theta_j - \eta_j \nabla_{\theta_j} R_j` is agent `j`'s anticipated one-step update. The original paper showed that this term can induce cooperative behaviour in iterated prisoner's dilemma and yields Nash play in repeated matching pennies, but its analysis was not built around the stochastic-game convergence theorem of Giannou et al. (2022).

The technical report closes that gap by modifying the schedule. Constant shaping is behaviourally effective but theoretically awkward because it introduces a persistent bias term. The report therefore uses an annealed shaping coefficient

```math
\lambda_n = \lambda / (n + m)^r
```

with `r > 1 - p`. The resulting update is

```math
\pi_{i,n+1}
= \operatorname{proj}_{\Pi}\bigl(\pi_{i,n} + \gamma_n(\hat v_{i,n} + \lambda_n \mathrm{OS}_{i,n})\bigr).
```

### 4.2 Convergence with Annealed Shaping

The first theorem says that annealing absorbs the shaping term into the bias budget of the Giannou proof.

**Theorem 4.1 (Annealed LOLA preserves convergence).** Suppose the baseline policy-gradient assumptions of Giannou et al. (2022) hold, and let `\lambda_n = \lambda / (n + m)^r` with `r > 1 - p`. Then LOLA-PG converges to the same stable Nash policy at the same rate exponent `q` as standard policy gradient.

**Proof sketch.** The opponent-shaping term contributes an additional bias of order `O(\lambda_n G)`, where `G` bounds the local shaping correction. Because `\lambda_n` decays faster than the threshold implied by `p + \min(\ell_b, r) > 1`, the aggregated bias still satisfies the summability condition needed in the martingale argument.

This theorem is mainly about non-interference. It shows that one can keep the behaviourally useful opponent-shaping term without giving up the only general local convergence theorem currently available for stochastic games.

### 4.3 Basin Enlargement Under Spectral Reinforcement

The more distinctive result is geometric rather than asymptotic. Let `H = \nabla_\pi \mathrm{OS}(\pi^*)` denote the Jacobian of the opponent-shaping field at the target equilibrium, and let `S_H = (H + H^\top)/2` be its symmetric part.

**Theorem 4.2 (Basin enlargement under spectral reinforcement).** If `S_H` is negative semidefinite with largest eigenvalue `-\mu_H` for `\mu_H \ge 0`, then LOLA-PG improves the local drift parameter from `\mu` to

```math
\mu_{\mathrm{LOLA}} = \mu + \lambda \mu_H > \mu.
```

The conclusion is local but meaningful. A larger drift constant means a stronger inward pull around the stable Nash point, which translates into a larger attraction region and faster contraction inside that region. The condition is natural in zero-sum games, where antisymmetric interaction terms dominate, and in general-sum games with sufficiently negative cross-derivatives.

This is the theorem that most directly justifies bringing LOLA into the convergence story. The original paper established strong behavioural phenomena. The technical report turns that observation into a spectral stability statement.

## 5. Composition: EW-LOLA-PG

### 5.1 Combined Update

We now combine the two modifications:

```math
\pi_{i,n+1}
= \operatorname{proj}_{\Pi}\bigl(
\pi_{i,n}
  + \gamma_n w_{i,n} (\hat v_{i,n} + \lambda_n \mathrm{OS}_{i,n})
\bigr).
```

The design intuition is simple. EW rescales the update according to local evidence quality. LOLA alters the update by adding the first-order effect of anticipated opponent learning. One is multiplicative and variance-focused. The other is additive and geometry-focused.

### 5.2 Composition Theorem

The main theorem of this paper is that the two corrections remain orthogonal at the level of the convergence proof.

**Theorem 5.1 (EW-LOLA composition).** Assume:

- the local regularity, step-size, bias, and variance conditions of Giannou et al. (2022);
- the evidence-weight assumptions used for EW-PG;
- the annealed shaping schedule `\lambda_n = \lambda / (n + m)^r` with `r > 1 - p`;
- the spectral reinforcement condition of Theorem 4.2 near the target Nash policy.

Then EW-LOLA-PG converges locally to the same stable Nash policy with

```math
\mathbb{E}[\| \pi_n - \pi^* \|^2 \mid E]
= O\!\left(\frac{\operatorname{HM}(V)}{\operatorname{AM}(V)} \cdot \frac{C}{n^q}\right),
```

and local drift parameter

```math
\mu_{\mathrm{EW-LOLA}} = \mu + \lambda \mu_H.
```

In particular, the algorithm inherits both the EW variance reduction and the LOLA basin enlargement.

**Proof sketch.** The argument is modular.

1. Variance term. The evidence weight multiplies the whole update vector. The AM-HM calculation therefore applies to the combined estimator `\hat v_i + \lambda_n \mathrm{OS}_i` with the same algebraic structure as in EW-PG.
2. Bias term. The only new bias relative to EW-PG is the annealed opponent-shaping contribution. Since `w_{i,n} \le 1`, multiplying by the evidence weight cannot make that bias larger than in the pure LOLA analysis.
3. Local drift. The drift improvement comes from the symmetric part of the shaping Jacobian at `\pi^*`. Evidence weighting changes the step magnitude but not the sign structure of that Jacobian, so the spectral reinforcement term is preserved.

The theorem is deliberately framed as a union-of-assumptions result. EW contributes the variance improvement. LOLA contributes the basin enlargement. The claim of this paper is that both improvements survive when the two modifications are combined.

### 5.3 Contribution Positioning

The paper is intentionally narrow. Giannou et al. (2022) provide the convergence baseline. The dissertation supplies the measure-theoretic and stochastic-approximation scaffolding behind that baseline. The Omega technical report isolates the separate EW and LOLA improvements. The new step in this paper is the explicit composition theorem together with a workshop-sized empirical suite focused on that theorem.

## 6. Empirical Validation

The experiment section should stay tightly matched to the theorem structure. Four experiments are enough for the workshop version.

### 6.1 Variance Reduction in Matrix Games

This experiment validates Theorem 3.2. Use matching pennies, stag hunt, and iterated prisoner's dilemma with controlled per-agent noise levels. Compare standard PG, EW-PG, LOLA-PG, and EW-LOLA-PG. The key plot should place the empirical variance ratio against the theoretical prediction `HM(V) / AM(V)` as the heterogeneity level changes. The expected result is that EW-PG and EW-LOLA-PG track the prediction closely, while standard PG and LOLA-PG do not benefit from the variance correction.

Current placeholder output: a medium matrix sweep is complete and saved at [matrix_summary.png](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/medium_matrix/matrix_summary.png), with tables at [matrix_summary.csv](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/medium_matrix/matrix_summary.csv). In the current scaffold, `ew_lola` is best under heterogeneous noise on both matching pennies and rock-paper-scissors, while pure LOLA degrades sharply in the most asymmetric settings. This is the right slot for the final workshop figure once the current exploratory plot is replaced by a paper-ready one.

Placeholder caption: **Figure 2. Matrix-game variance and convergence proxy.** Left: noise heterogeneity versus observed variance proxy under `standard`, `ew`, `lola`, and `ew_lola`. Right: final distance-to-Nash proxy versus heterogeneity level. Under asymmetric noise, EW-LOLA combines the stability gains of evidence weighting with the shaping gains of LOLA.

### 6.2 Basin Enlargement in Iterated Prisoner's Dilemma

This experiment validates Theorem 4.2. Sweep the shaping strength `\lambda` and initialise agents near the empirical boundary of the standard-PG attraction region. Measure convergence success and convergence speed. The headline figure should be a heatmap over `(lambda, distance-from-basin-centre)`. The expected pattern is that annealed LOLA expands the region from which the target stable policy is reached.

Current placeholder status: this experiment is not yet complete in the new scaffold. The slot should be filled by a dedicated basin-mapping run rather than repurposing the generic iterated-game curves. The implementation target is a grid over initial conditions and a sweep over shaping strengths with a convergence-success criterion held fixed across methods.

Placeholder caption: **Figure 3. Basin enlargement under opponent shaping.** Heatmap of convergence success over shaping strength and initialization radius. Standard PG occupies the smallest attraction region. Annealed LOLA enlarges the convergent set, and EW-LOLA preserves that enlargement while remaining more stable under noisy gradients.

### 6.3 Composition in Iterated RPS

This experiment isolates Theorem 5.1 on a game family where exploitation and instability are both visible. Following the setup used around Experiments 10 and 12 in the technical report, evaluate standard PG, EW-PG, LOLA-PG, and EW-LOLA-PG on iterated RPS with biased opponent personas. The working hypothesis is that EW-LOLA-PG reaches the strongest payoff because EW stabilises the estimator while LOLA preserves the shaping signal needed to exploit bias.

Current placeholder status: the persona-conditioned experiment path is implemented, and the IPD persona sweep is complete at [kim_persona_summary.png](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/medium_kim_ipd/kim_persona_summary.png) with tables at [kim_persona_summary.csv](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/artifacts/medium_kim_ipd/kim_persona_summary.csv). The matching iterated-RPS persona campaign is still pending because the current LOLA correction uses nested finite differences through the stateful policy and needs optimisation before longer runs become practical. This subsection should keep its figure slot now and receive the final competitive-persona plot once that bottleneck is fixed.

Placeholder caption: **Figure 4. Composition in iterated RPS against Kim-style personas.** Mean learner return against biased rock, paper, and scissors opponents under `standard`, `ew`, `lola`, and `ew_lola`. The expected pattern is that EW-LOLA matches LOLA's exploitative advantage while reducing performance collapse in higher-noise settings.

### 6.4 Scaling to N Players

Adapt Kim et al. (2021) Question 6 style N-player iterated RPS benchmarks for `N = 2, 3, 4`. The aim is modest: show that the combined method still beats the plain baseline when the number of interacting learners grows. This gives a small but useful check that the composition survives beyond the two-player case.

Current placeholder status: no N-player run has been executed in the new EW-LOLA scaffold yet. This slot should remain reserved for the smallest scaling check needed by the workshop paper. If runtime becomes a problem, it is better to keep `N = 2, 3, 4` only and treat anything larger as follow-up work.

Placeholder caption: **Figure 5. Scaling to N players.** Mean return gap between each method and the standard baseline for `N = 2, 3, 4` iterated RPS. The target claim is modest: EW-LOLA should remain competitive as the number of learners increases.

### 6.5 Experimental Scope

The experiment suite is intentionally limited to the tasks that map most directly onto the theory. Matrix games test the AM-HM prediction. Iterated prisoner's dilemma and iterated RPS test shaping and composition under adaptive play. The N-player extension checks whether the combined method remains effective outside the two-player case. These environments already appear in the technical report and should be rerun for the workshop version with publication-quality figures and additional seeds. Any PettingZoo results should be presented as appendix-level extensions unless they are complete enough to be defended in the main text.

Current run log: the completed and partially completed experiments are tracked in [CURRENT_RESULTS.md](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/CURRENT_RESULTS.md). The working experiment code and artifact layout live under [experiments](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments).

### 6.6 Large-Scale Extension Placeholder

The workshop paper should remain centred on theory-facing validation, but it is worth reserving one paragraph and an appendix slot for a larger benchmark extension. The current staged roadmap is recorded in [LARGE_SCALE_EXPERIMENT_PLAN.md](/Users/meuge/coding/maynard/ICML%20Sprint/omega-ew-lola-paper/experiments/LARGE_SCALE_EXPERIMENT_PLAN.md). The intended sequence is:

- complete the theorem-facing matrix and persona-conditioned runs first;
- move to one medium benchmark family such as PettingZoo or MAgent2;
- choose one flagship large-scale suite, preferably Melting Pot for social generalisation or SMACv2 for cooperative micromanagement, rather than attempting both at once.

Placeholder caption: **Appendix Figure A1. Large-scale benchmark extension.** Aggregate scores on one medium-scale benchmark family and one flagship suite, reported with seed variance and instability statistics in addition to mean return.

## 7. Lean 4 Formalisation

One distinctive aspect of the Omega work is the partial formalisation in Lean 4. The technical report describes three files of roughly nine hundred lines total:

- `EvidenceWeightedPG.lean` for the energy inequality, AM-HM bound, and variance-ratio algebra;
- `OpponentShapingPG.lean` for bias absorption, annealing compatibility, and basin enlargement;
- `CooperativePG.lean` for cooperative extensions that are outside the scope of this paper.

For the workshop paper this section should stay short. The right claim is that the algebraic backbone has been formalised, while the full stochastic convergence statements still carry `sorry` annotations where current Mathlib support is thin. That is still worth reporting. It signals a level of proof hygiene unusual for a workshop paper without pretending that the entire stochastic analysis is already machine-checked.

## 8. Discussion and Limitations

The contribution is local in three senses.

First, the convergence guarantee is local because it inherits the Giannou theorem. EW-LOLA-PG does not solve global equilibrium selection. Second, the basin-enlargement theorem depends on a spectral reinforcement condition that need not hold in every general-sum game. Third, the experiments are still centred on matrix and iterated-game settings. They test the theory cleanly, but they do not yet show performance in deep MARL.

Those limitations are acceptable for this venue if they are stated directly. The paper works best as a compositional theory paper with a compact empirical validation suite.

There is also a useful relation to Kim et al. (2021). Meta-MAPG models own-learning and peer-learning gradients across a chain of policy updates. EW-LOLA-PG studies local convergence and estimator design. Kim's iterated-game benchmarks remain valuable because they stress the adaptation dynamics where opponent shaping matters.

## 9. Conclusion

Giannou et al. (2022) provided a local convergence theorem for policy gradient in general stochastic games. The Omega technical report added two orthogonal modifications to that picture: evidence weighting, which reduces variance, and annealed opponent shaping, which enlarges the stable basin under a spectral condition. This paper isolates those two pieces, proves that they compose without interference, and tests the resulting prediction on a compact set of matrix and iterated-game benchmarks. The resulting claim is simple: in local stochastic-game policy optimisation, variance control and opponent shaping can be combined in a mathematically disciplined way.

## References

- Agarwal, A., Kakade, S., Lee, J. D., and Mahajan, G. (2021). On the theory of policy gradient methods: optimality, approximation, and distribution shift.
- Foerster, J., Chen, R. Y., Al-Shedivat, M., Whiteson, S., Abbeel, P., and Mordatch, I. (2018). Learning with Opponent-Learning Awareness. Proceedings of AAMAS 2018.
- Giannou, A., Lotidis, K., Mertikopoulos, P., and Vlatakis-Gkaragkounis, E. V. (2022). On the Convergence of Policy Gradient Methods to Nash Equilibria in General Stochastic Games. NeurIPS.
- Kim, D.-K., Liu, M., Riemer, M., Sun, C., Abdulhai, M., Habibi, G., Lopez-Cot, S., Tesauro, G., and How, J. P. (2021). A Policy Gradient Algorithm for Learning to Learn in Multiagent Reinforcement Learning. ICML.
- Shcherbinin, E. (2026). Convergence of Policy Gradient Methods to Nash Equilibria in Stochastic Games. LSE dissertation.
- Shcherbinin, E. (2026). The Omega-Gradient: Evidence-Weighted Multi-Agent Policy Optimization with Opponent Shaping, Cooperative Communication, and the Blessing of Dimensionality. Technical report.
