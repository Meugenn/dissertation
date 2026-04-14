# EW-LOLA Master Report

Companion document to `paper-draft.md`.

Purpose: this is the unlimited-length working report for the EW-LOLA paper. It is meant to hold the full argument skeleton, provenance notes, proof blueprints, experiment inventory, reviewer-risk analysis, and section-level writing guidance that would be too long or too unstable for the workshop draft itself. Future agents should treat this as the long-form backbone and `paper-draft.md` as the compressed workshop-facing version.

## 1. How To Use This File

Use this report for four things.

- As the factual source map for where each theorem, experiment, and claim came from.
- As the longer narrative version of the paper, especially when writing appendices or a longer arXiv draft.
- As a decision log for what is in scope for the NExT-Game submission and what has been deliberately excluded.
- As a checklist of what still needs proof verification, reruns, or bibliographic cleanup before submission.

The intended workflow is:

1. Draft or revise the workshop paper in `paper-draft.md`.
2. Check the corresponding section of this report for provenance and omitted detail.
3. Pull only the amount of detail needed for the workshop version.
4. Keep any unstable, unresolved, or appendix-level material here rather than bloating the paper draft.

## 2. File And Source Inventory

Primary local sources used for this report:

- `resources/omega-pg-plan.docx`
- `resources/technical_report.pdf`
- `resources/dissertation bsc.pdf`
- `resources/dissertation extended.pdf`
- `meta-swag/papers/giannou22-policy-gradient-stochastic-games.pdf`
- `meta-swag/papers/kim21g-meta-mapg.pdf`
- `meta-swag/references.bib`
- `/Users/meuge/Downloads/tropes.md`

Primary external source checked for accuracy on LOLA:

- Foerster et al. (2018), "Learning with Opponent-Learning Awareness" via arXiv and the AAMAS proceedings PDF.

What each source is doing here:

- `omega-pg-plan.docx` gives the target workshop framing, scope cuts, timeline, and contribution hierarchy.
- `technical_report.pdf` supplies the EW and LOLA theorem statements, experiment templates, and Lean summary.
- `dissertation bsc.pdf` supplies the careful exposition of the Giannou convergence framework.
- `dissertation extended.pdf` supplies the broader Omega-context chapters, including a longer combined EW-LOLA chapter and more expansive experimental discussion.
- `giannou22-policy-gradient-stochastic-games.pdf` is the baseline theorem that the whole paper inherits.
- `kim21g-meta-mapg.pdf` gives the benchmark family, the peer-learning framing, and a useful point of contrast.
- `tropes.md` constrains style so the final paper reads like a human mathematics/ML draft rather than generic model prose.

## 3. Paper-Level Thesis

The workshop paper should make one claim and make it clearly:

Two orthogonal modifications to local policy-gradient learning in stochastic games compose without interfering with the baseline convergence guarantee. Evidence weighting lowers the variance prefactor. Annealed opponent shaping enlarges the basin of attraction under a spectral reinforcement condition. The combined update inherits both improvements.

That claim is stronger and cleaner than trying to sell the entire Omega programme in eight pages.

The concise workshop thesis is:

- Giannou et al. provide local convergence of policy gradient to stable Nash policies.
- The Omega work identifies two modifications that improve different parts of that picture.
- These modifications can be combined in one update rule without breaking the proof template.

Everything else is supporting material.

## 4. Scope Discipline

### 4.1 What This Paper Is

This paper is:

- a local-convergence paper for policy gradient in general stochastic games;
- a variance-reduction paper for heterogeneous multi-agent gradient estimates;
- an opponent-shaping stability paper under annealed schedules;
- a composition paper showing those two mechanisms coexist cleanly.

### 4.2 What This Paper Is Not

This paper is not:

- the full Omega-gradient paper;
- a cooperative-communication paper;
- a sparse-policy or blessing-of-dimensionality paper;
- a fixed-point equilibrium search paper;
- a Meta-MAPG paper;
- a deep MARL benchmark paper.

Those exclusions are important because otherwise the paper loses focus fast.

### 4.3 Why The Scope Cut Is Correct

The April 13 plan already makes the right editorial call. EW plus LOLA is the cleanest workshop-sized unit because:

- both modifications are already developed locally in the technical report;
- they share the Giannou convergence scaffold;
- they improve different terms of the Lyapunov recursion;
- their composition is the only truly new theorem-sized move required for the workshop;
- the experiments that support them already exist in template form.

By contrast:

- Coop-PG belongs with the separate cooperation-focused paper;
- sparse regularisation is its own contribution;
- FP-NE is a meta-algorithm sitting above the local gradient method;
- random restarts are a globalisation device with a different story.

## 5. Source-Derived Core Story

### 5.1 Giannou et al. (2022): The Baseline

Giannou et al. prove that projected policy gradient in general stochastic games converges locally to stable Nash policies under a second-order stability condition. The key ingredients are:

- a gradient-dominance property that links first-order stationarity to Nash;
- a stable or SOS-type local drift inequality around the target equilibrium;
- stochastic approximation control over bias and variance;
- a Lyapunov-style energy recursion with martingale terms.

The paper's workshop draft only needs a compressed version of this setup. The dissertation is useful here because it spells out the measure-theoretic foundations and makes the local stability claim legible without relying on appendices. The workshop draft should cite Giannou for the theorem and the dissertation for expository backup if needed, but it should avoid reproducing the full proof.

What must survive into the workshop draft:

- the stochastic game setting;
- the policy-gradient update;
- the stable Nash or SOS condition;
- the rate template `O(C / n^q)`.

What should stay out of the workshop draft:

- the full filtration and martingale machinery;
- the full extraction-of-subsequence argument;
- the restart extension.

### 5.2 EW-PG: What Is Proven In The Technical Report

The EW contribution in the technical report is conceptually simple and mathematically clean.

The mechanism:

- each agent has a gradient estimator variance `V_i`;
- define an evidence weight inversely proportional to that variance;
- rescale the local update by that weight.

The technical report's main EW results are:

1. An AM-HM variance improvement theorem.
2. A preserved local convergence theorem with the same rate exponent and a smaller constant.
3. A practical adaptive estimator for the weights using moving-window or EMA variance estimates.

The core intuition is that equal weighting is suboptimal when agents contribute gradients of very different quality. The evidence weight lets the lower-variance agents carry more of the update without changing the equilibrium being targeted.

The cleanest theorem statement for workshop use is:

```math
\sigma_w^2 = \frac{\mathrm{HM}(V)}{\mathrm{AM}(V)} \sigma_{\mathrm{std}}^2 \le \sigma_{\mathrm{std}}^2.
```

This theorem is attractive for reviewers because:

- it is sharp;
- it is easy to verify algebraically;
- it has a transparent equality condition;
- it maps directly to an empirical plot.

The longer report and dissertation materials also suggest a broader interpretation of evidence weighting as a recurring device across gradient quality, self-knowledge, and communication quality. That larger conceptual role should stay mostly out of the workshop draft. It can appear in one sentence in the discussion, but not as a major theme.

### 5.3 LOLA-PG: What Is Proven In The Technical Report

The LOLA contribution is more delicate because the method helps by changing the geometry of the dynamics but also risks introducing a persistent bias.

The technical report's move is to split the two regimes:

- constant shaping is useful to understand basin enlargement and local geometry;
- annealed shaping is what makes the convergence theorem compatible with Giannou's stochastic approximation conditions.

The main report-level results are:

1. A bias absorption lemma showing that the shaping term can be folded into the bias budget.
2. A schedule compatibility result requiring `r > 1 - p` when `\lambda_n = \lambda / (n+m)^r`.
3. A convergence theorem for annealed LOLA-PG that preserves the baseline rate exponent when the schedule is chosen appropriately.
4. A basin-enlargement theorem under a spectral reinforcement condition on the symmetric part of the shaping Jacobian.

The conceptual summary is:

- annealing addresses asymptotic correctness;
- spectral reinforcement explains local geometric improvement.

That split is important for the workshop draft. It prevents the paper from making a confused claim that one fixed scheduling choice simultaneously proves exact convergence and maximal shaping strength. The paper should instead say:

- for theory, anneal the shaping coefficient;
- for local geometry, analyse the shaped Jacobian at the equilibrium.

### 5.4 Combined EW-LOLA In The Extended Dissertation

The extended dissertation has an explicit section on combined EW-LOLA-PG. That matters because it means the workshop paper is not inventing the composition from nothing, but rather isolating and sharpening an already-present combined argument.

The extended dissertation frames the two improvements as acting on different terms of the Lyapunov recursion:

- EW reduces the variance constant;
- LOLA increases the contraction parameter.

This is the right organising sentence for the composition proof.

The dissertation also reports an empirical observation that the combined gains are approximately multiplicative in the relevant experiments. That is useful as empirical language, but the workshop paper should be careful not to oversell it as a universal theorem. The safe phrasing is:

- the mechanisms act on independent terms in the local recursion;
- in the tested games, the empirical improvements are close to multiplicative.

### 5.5 Kim et al. (2021): Why They Matter Here

Kim et al. derive the Meta-MAPG theorem, which decomposes the meta-gradient into:

- the current policy term;
- the own-learning term;
- the peer-learning term.

This is valuable for the EW-LOLA paper in three ways.

First, it gives a good benchmark family for iterated prisoner's dilemma, iterated RPS, and N-player RPS. Those tasks are already well-matched to learning-dynamics questions.

Second, it gives a principled contrast point. LOLA is a one-step shaping correction. Meta-MAPG differentiates through a chain of policy updates and explicitly models peer learning. The EW-LOLA paper should not pretend to subsume Meta-MAPG. Instead, it should say that the two sit at different levels:

- Meta-MAPG is a meta-learning treatment of non-stationarity across learning updates;
- EW-LOLA-PG is a local-convergence and estimator-design treatment of one-step gradient dynamics.

Third, Kim's experiments help us say why iterated games are a reasonable validation environment rather than a toy detour. They are exactly the settings where opponent adaptation matters.

### 5.6 Foerster et al. (2018): What Must Be Attributed Correctly

Foerster et al. introduced the opponent-shaping idea and showed:

- cooperation in iterated prisoner's dilemma between LOLA agents;
- Nash play in repeated matching pennies;
- practical policy-gradient computation of the shaping term;
- effectiveness in model-free settings and social dilemma environments.

What they did not provide is the stochastic-game local convergence theorem used here. That gap should be stated clearly and calmly. The EW-LOLA paper depends on that distinction.

The workshop paper should attribute to Foerster et al.:

- the core shaping update idea;
- the behavioural findings;
- the model-free gradient implementation.

It should attribute to the Omega work:

- annealed schedule compatibility with the Giannou theorem;
- the spectral basin-enlargement result in this stochastic-game local-convergence framing;
- the composition result with evidence weighting.

## 6. Theorem Inventory

This section lists theorems in the order they should be mentally organised while writing.

### 6.1 Baseline Theorem: Local PG Convergence

Source:

- Giannou et al. (2022)
- dissertation bsc.pdf

Role in paper:

- baseline theorem, cited not reproved.

Working statement:

- under SOS-type local stability and standard bias/variance assumptions, projected policy gradient converges locally to a stable Nash policy with rate `O(C / n^q)`.

Dependencies:

- stochastic game setup;
- gradient-dominance property;
- stochastic approximation conditions.

### 6.2 EW Theorem A: Variance Improvement

Source:

- technical_report.pdf Section 3
- dissertation extended.pdf Section 7.2.4

Role in paper:

- first headline theorem;
- can remain in main text with a one-paragraph proof or proof sketch.

Working statement:

- effective variance shrinks by `HM(V) / AM(V)`.

What must be checked before final submission:

- exact definition of `V_i` and whether the weight is written as `V_min / V_i` or normalised equivalently.
- consistency between the algebra in the technical report and the notation used in the workshop draft.

### 6.3 EW Theorem B: Preserved Convergence

Source:

- technical_report.pdf Section 3
- dissertation extended.pdf Section 7.2.5

Role in paper:

- support theorem following the variance theorem.

Working statement:

- local convergence rate exponent is preserved; leading constant improves by the same harmonic-to-arithmetic mean factor.

Potential pitfall:

- avoid implying that the factor is exact in every implementation with estimated weights. For finite-window estimated weights, the report only guarantees asymptotic recovery of the oracle effect.

### 6.4 LOLA Theorem A: Annealed Convergence

Source:

- technical_report.pdf Section 4
- dissertation extended.pdf Section 7.3.2

Role in paper:

- second headline theorem.

Working statement:

- with `\lambda_n = \lambda / (n+m)^r` and `r > 1-p`, LOLA-PG preserves local convergence to stable Nash;
- choosing `r = p` recovers the baseline rate exponent in the report's treatment.

Potential pitfall:

- keep the exact exponent statement aligned with the source. The report writes `q_LOLA = min(\ell_b, r, p - 2 \ell_\sigma)` before the schedule choice.

### 6.5 LOLA Theorem B: Basin Enlargement

Source:

- technical_report.pdf Section 4
- dissertation extended.pdf Section 7.3.3 and Chapter 14 basin experiments

Role in paper:

- geometric headline theorem.

Working statement:

- if the symmetric part of the opponent-shaping Jacobian is negative semidefinite with parameter `\mu_H`, then the effective drift constant becomes `\mu + \lambda \mu_H`.

Potential pitfall:

- keep clear whether `\lambda` in this expression refers to a local constant-strength analysis near the equilibrium, while the convergence theorem uses an annealed schedule. The workshop paper can handle this by saying that the spectral result is a local linearisation statement and the convergence result is an annealed asymptotic statement.

### 6.6 Composition Theorem

Source:

- implicit in the Omega combined update;
- explicit combined section in dissertation extended.pdf;
- isolated as new in `omega-pg-plan.docx`.

Role in paper:

- main novel theorem.

Working statement:

- under the union of EW and LOLA assumptions, EW-LOLA-PG preserves the EW variance improvement and the LOLA drift improvement.

What is genuinely new:

- making the composition explicit as the central theorem of a standalone paper;
- giving the proof in a modular, no-interference form;
- positioning the empirical suite around that combined claim rather than around the full Omega method.

What needs the most verification:

- the exact interplay of evidence weights with the shaped Jacobian in the local drift argument;
- the exact statement of the constant factor in the combined convergence rate;
- whether any extra boundedness assumption on the weighted shaping term is needed for a polished theorem statement.

## 7. Composition Proof Blueprint

This is the most important section of the master report because it contains the paper's risky step.

### 7.1 Combined Update

Write the combined update as

```math
\pi_{i,n+1}
= \operatorname{proj}_{\Pi}\bigl(
\pi_{i,n} + \gamma_n w_{i,n}(\hat v_{i,n} + \lambda_n \mathrm{OS}_{i,n})
\bigr).
```

Then mentally decompose the proof into the same three objects that appear in the Giannou recursion:

- drift;
- bias;
- noise variance.

### 7.2 Variance Channel

Target claim:

- EW still reduces the variance prefactor after LOLA is added.

Reason:

- the evidence weight acts multiplicatively on the full update vector;
- if the shaping term is treated as `\mathcal{F}_n`-measurable or otherwise bounded in the local analysis, the stochastic noise part still gets downweighted by the same `w_i`;
- the AM-HM improvement therefore survives on the stochastic part of the update.

Careful wording for the paper:

- "the same harmonic-mean improvement applies to the stochastic component of the combined update";
- avoid wording that suggests the deterministic shaping term itself has variance.

### 7.3 Bias Channel

Target claim:

- the annealed shaping contribution remains compatible with Giannou's bias condition under weighting.

Reason:

- the extra bias term from LOLA is of order `\lambda_n G`;
- if `0 < w_{i,n} <= 1`, multiplying by `w_{i,n}` cannot increase that term;
- thus the weighted combined method inherits the same summability condition as the unweighted annealed LOLA analysis.

Possible caveat:

- if the weights are estimated with noise, one may need to distinguish the oracle-weight statement from the practical-estimator statement. The workshop paper can stay with the oracle-weight theorem and mention the adaptive estimator as implementation detail or appendix remark.

### 7.4 Drift Channel

Target claim:

- the spectral reinforcement term from LOLA is preserved under evidence weighting.

This is the most delicate step.

One way to phrase it safely:

- the local drift improvement is a property of the shaped vector field;
- evidence weighting rescales each agent's step without changing the sign structure of the symmetric part responsible for contraction;
- under bounded positive weights, the local contraction argument continues to inherit the reinforcement term.

If a stricter proof is needed, one may want a theorem statement that keeps the drift claim in a slightly weaker form, for example:

- "the combined method preserves the enlarged local attraction region induced by the opponent-shaping field up to positive diagonal rescaling."

This is a useful fallback if the exact `\mu + \lambda \mu_H` statement becomes awkward under heterogeneous weights.

### 7.5 Strong Form And Weak Form

For writing purposes, define two versions of the composition theorem.

Strong form:

- exact EW constant factor plus exact LOLA drift parameter.

Weak form:

- EW variance reduction plus preservation of the LOLA-induced local basin enlargement under positive diagonal rescaling.

The current `paper-draft.md` uses the strong form. If appendix verification finds a snag, downgrade the drift part to the weak form rather than stretching the proof.

## 8. Experimental Master Plan

### 8.1 What The Workshop Paper Actually Needs

The paper does not need a giant benchmark section. It needs evidence for three specific statements:

1. EW matches the AM-HM variance prediction.
2. LOLA enlarges the basin in the games where spectral reinforcement is present.
3. EW plus LOLA together outperform either one alone in the combined setting.

Everything else is optional.

### 8.2 Experiment Inventory From Existing Material

From the technical report:

- direct EW variance measurement;
- convergence constant measurement;
- spectral reinforcement analysis;
- basin mapping and basin-size-vs-lambda;
- combined EW-LOLA comparison;
- iterated prisoner's dilemma adaptation;
- iterated RPS exploitation;
- strategy identification;
- scaling to N players.

From Kim et al.:

- IPD as a mixed-incentive adaptive game;
- RPS as a competitive adaptive game;
- N-player RPS as a scaling check.

### 8.3 Recommended Workshop Subset

Main text experiments:

1. Matrix-game variance ratio.
2. Basin mapping or lambda-sweep in a shaping-friendly game.
3. Iterated RPS composition comparison.
4. N-player RPS scaling.

Appendix candidates:

- strategy identification on IPD;
- PettingZoo extension;
- BR Jacobian spectral analysis;
- extra ablations on weight estimation and schedule choice.

### 8.4 Figure Plan

Figure 1:

- method overview, ideally one line showing standard PG, EW, LOLA, and EW-LOLA in a single schematic.

Figure 2:

- empirical variance ratio versus `HM / AM`.

Figure 3:

- basin heatmap or basin-size curve versus `lambda`.

Figure 4:

- iterated RPS payoff comparison across the four methods.

Figure 5:

- N-player scaling plot.

Appendix figures:

- strategy identification;
- spectral reinforcement diagnostics;
- schedule ablation;
- implementation details.

### 8.5 What Must Be Reported Honestly

Do not imply clean reruns if the workspace does not contain them. The master report should preserve the distinction between:

- existing technical-report evidence;
- benchmark templates that can be rerun;
- reruns that have actually been completed for the workshop paper.

The paper can still say that the experiments are adapted from the technical report and benchmark family, provided that the final submission reflects the true run status.

## 9. Related Work Positioning

### 9.1 Minimum Necessary Related Work

The workshop version only needs a disciplined set of references.

- Giannou et al. (2022): baseline convergence theorem.
- Foerster et al. (2018): LOLA update and behavioural findings.
- Kim et al. (2021): Meta-MAPG and iterated-game benchmark family.
- Possibly Letcher et al. (2019): stability-focused follow-up to LOLA if needed.

### 9.2 How To Position Against Meta-MAPG

Do not position EW-LOLA as "better than" Meta-MAPG in general. The cleaner position is:

- Meta-MAPG studies meta-gradients through learning chains;
- EW-LOLA studies local policy-gradient dynamics and convergence near stable Nash points;
- the two are complementary perspectives on multi-agent non-stationarity.

This avoids overclaiming and will read better to reviewers who know both literatures.

### 9.3 How To Position Against The Full Omega Programme

The full Omega programme is useful as background, but the workshop paper should refer to it only briefly:

- EW and LOLA are two components of a broader framework;
- the other components are treated elsewhere.

That sentence is enough.

## 10. Suggested Appendix Structure

If the workshop allows an appendix, the clean appendix order is:

1. Full preliminaries and notation.
2. Exact inherited Giannou theorem statement.
3. Proof of the EW variance theorem.
4. Proof of the EW convergence theorem.
5. Bias absorption and schedule compatibility lemmas for LOLA.
6. Proof of the LOLA convergence theorem.
7. Proof of the basin-enlargement theorem.
8. Proof of the composition theorem.
9. Adaptive weight estimation remark.
10. Extended experimental details.
11. Lean 4 status note.

This order mirrors the logical dependency chain.

## 11. Reviewer-Risk Analysis

### 11.1 Risk: Composition Theorem Feels Too Convenient

Reviewer concern:

- "You are claiming two independent benefits add together cleanly. Where exactly is the hard part?"

Response strategy:

- state clearly that the hard part is the no-interference argument in the local recursion;
- show the decomposition into variance, bias, and drift terms;
- make the proof modular rather than mystical.

### 11.2 Risk: LOLA Result Depends On Special Games

Reviewer concern:

- "Is the spectral reinforcement condition only true in cherry-picked zero-sum games?"

Response strategy:

- say directly that the condition is local and game-dependent;
- emphasise that zero-sum games are a natural positive case;
- present the condition as a sufficient condition, not a generic truth about all games.

### 11.3 Risk: Matrix Games Are Too Small

Reviewer concern:

- "This is all clean theory, but where is the evidence beyond toy games?"

Response strategy:

- remind the reader that the paper is validating theorem-level predictions;
- include an appendix-level PettingZoo extension if available;
- keep the main text experiments narrow and hypothesis-driven.

### 11.4 Risk: Lean Section Sounds Inflated

Reviewer concern:

- "Is this actually machine-checked or just partially sketched?"

Response strategy:

- be explicit that the algebraic backbone is formalised;
- say that some stochastic convergence statements still contain `sorry`;
- avoid phrases like "fully verified" unless every relevant theorem actually is.

## 12. Writing Crosswalk To The Paper Draft

This section links the short paper draft to the larger materials in this report.

### 12.1 Abstract

Paper draft goal:

- one paragraph that states the two weaknesses, two fixes, and one composition result.

Backstop in this report:

- Sections 3 to 7.

### 12.2 Introduction

Paper draft goal:

- establish Giannou as baseline;
- frame heteroskedasticity and basin size as the two remaining problems;
- state composition as the contribution.

Backstop in this report:

- Sections 3, 5, and 6.

### 12.3 Preliminaries

Paper draft goal:

- only enough of stochastic games, policy gradient, and SOS to make the theorem statements readable.

Backstop in this report:

- Section 5.1 and the dissertation-derived baseline discussion.

### 12.4 EW Section

Paper draft goal:

- define the weights;
- state variance theorem;
- state preserved convergence theorem;
- keep proof compact.

Backstop in this report:

- Sections 5.2 and 6.2 to 6.3.

### 12.5 LOLA Section

Paper draft goal:

- define opponent shaping;
- explain annealing;
- state convergence and basin theorems;
- distinguish asymptotic correctness from local geometry.

Backstop in this report:

- Sections 5.3 and 6.4 to 6.5.

### 12.6 Composition Section

Paper draft goal:

- present the combined update;
- state the theorem;
- give the modular proof sketch;
- say clearly that this is the paper's novel step.

Backstop in this report:

- Sections 6.6 and 7.

### 12.7 Experiments

Paper draft goal:

- align one experiment to each main theorem-level claim.

Backstop in this report:

- Section 8.

### 12.8 Lean, Limitations, Conclusion

Paper draft goal:

- keep each one short and concrete.

Backstop in this report:

- Sections 8.5, 9, and 11.

## 13. Open Tasks Before Submission

### 13.1 Theorem Verification

- Check the exact weight notation across the report, dissertation, and draft so the EW theorem is written consistently.
- Verify whether the combined theorem can safely keep the exact drift parameter `\mu + \lambda \mu_H` under heterogeneous weighting, or whether it should be weakened to a positive-diagonal-rescaling statement.
- Decide whether the adaptive evidence-weight estimator is in main text or appendix.

### 13.2 Bibliography

- Add missing citations if Letcher et al. (2019) or Sutton and Barto are referenced directly in the prose.
- Ensure the bibliography used in the final LaTeX file includes Foerster, Kim, Giannou, and any auxiliary prior-work citations added during polishing.

### 13.3 Experiments

- Confirm which technical-report experiments have already been rerun cleanly.
- Generate publication-quality figures for the selected subset.
- Increase seeds where possible for cleaner workshop plots.
- Decide whether any PettingZoo extension is mature enough for the appendix.

### 13.4 Lean

- Verify actual file names and current proof status before the final writeup.
- Keep claims about `sorry` annotations exact.

## 14. Recommended Next Passes

The highest-value next editing passes are:

1. Convert `paper-draft.md` into venue-ready LaTeX while preserving the current scope discipline.
2. Expand Section 5 of the paper draft into a more formal appendix proof based on the composition blueprint above.
3. Build a figure checklist with exact filenames and captions once the reruns are available.
4. Tighten the related work so every reference has a precise job.

## 15. Bottom Line

The EW-LOLA paper works if it stays narrow. The baseline is Giannou. The first inherited improvement is EW. The second inherited improvement is annealed LOLA plus spectral reinforcement. The new contribution is the explicit composition theorem and a validation suite organised around that theorem. This master report exists to keep that line clean while the shorter workshop draft stays readable.
