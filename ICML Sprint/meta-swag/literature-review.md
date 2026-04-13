# Meta-SWAG Literature Review and Hallucination Pass

This note records a source-grounded pass over the current Meta-SWAG draft. The goal is not to judge the new theorems themselves, which are part of the paper's novel contribution, but to separate:

- claims already supported by prior literature,
- claims that are plausible extrapolations from prior literature,
- and claims that should be framed explicitly as this paper's own proposal rather than as established fact.

## 1. Core papers reviewed

### 1. SWAG
- File: [papers/maddox19-swag.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/maddox19-swag.pdf)
- Citation: Maddox et al., *A Simple Baseline for Bayesian Uncertainty in Deep Learning*, NeurIPS 2019.
- Why it matters:
  - establishes the diagonal-plus-low-rank Gaussian posterior fitted from SGD iterates;
  - supports the use of Bayesian model averaging from posterior samples;
  - supports the draft's use of a SWAG-style covariance decomposition.

### 2. Meta-MAPG / meta-MDP framing in MARL
- File: [papers/kim21g-meta-mapg.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/kim21g-meta-mapg.pdf)
- Citation: Kim et al., *A Policy Gradient Algorithm for Learning to Learn in Multiagent Reinforcement Learning*, ICML 2021.
- Why it matters:
  - directly supports the claim that MARL adaptation can be treated through a meta-learning lens;
  - supports the idea of modeling non-stationary policy dynamics at a meta level;
  - is the strongest prior source for the paper's meta-MDP bridge on the MARL side.

### 3. Convergence in general stochastic games
- File: [papers/giannou22-policy-gradient-stochastic-games.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/giannou22-policy-gradient-stochastic-games.pdf)
- Citation: Giannou et al., *On the Convergence of Policy Gradient Methods to Nash Equilibria in General Stochastic Games*, NeurIPS 2022.
- Why it matters:
  - supports the draft's description of local convergence under SOS-style conditions;
  - is the cleanest literature anchor for the stochastic-game convergence language.

### 4. Laplace-LoRA
- File: [papers/yang24-laplace-lora.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/yang24-laplace-lora.pdf)
- Citation: Yang et al., *Bayesian Low-Rank Adaptation for Large Language Models*, ICLR 2024.
- Why it matters:
  - directly supports the draft's positioning against LoRA-space Bayesian posteriors;
  - supports the claim that posterior inference in adapter space is computationally attractive;
  - supports the calibration motivation on the LLM side.

### 5. Bayesian reward models
- File: [papers/yang24-bayesian-reward-models.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/yang24-bayesian-reward-models.pdf)
- Citation: Yang et al., *Bayesian Reward Models for LLM Alignment*, arXiv 2024.
- Why it matters:
  - directly supports the draft's claim that reward-side Bayesian uncertainty can mitigate reward overoptimization;
  - gives the clearest adjacent baseline for the alignment experiments.

### 6. Reward overoptimization
- File: [papers/gao23h-reward-overoptimization.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/gao23h-reward-overoptimization.pdf)
- Citation: Gao et al., *Scaling Laws for Reward Model Overoptimization*, ICML 2023.
- Why it matters:
  - directly supports the paper's use of proxy-vs-gold reward gaps and best-of-`n` evaluation;
  - supports framing reward overoptimization as a concrete empirical problem rather than a vague safety concern.

### 7. DPO
- File: [papers/rafailov23-dpo.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/rafailov23-dpo.pdf)
- Citation: Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023.
- Why it matters:
  - supports the draft's use of DPO as a first-class aligner;
  - supports describing DPO as a standard alignment trajectory over checkpoints.

### 8. Reward overoptimization in direct alignment algorithms
- File: [papers/rafailov24-daa-overoptimization.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/rafailov24-daa-overoptimization.pdf)
- Citation: Rafailov et al., *Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms*, NeurIPS 2024.
- Why it matters:
  - supports the claim that overoptimization-like degradation persists even outside classic reward-model-plus-RLHF pipelines;
  - strengthens the case for evaluating Meta-SWAG under DPO-like objectives, not just PPO/RLHF.

### 9. PPO
- File: [papers/schulman17-ppo.pdf](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers/schulman17-ppo.pdf)
- Citation: Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv 2017.
- Why it matters:
  - standard anchor for PPO when listing aligners in the draft.

## 2. Hallucination check: what is supported vs. what should be framed as novel

### Well-supported by the literature
- SWAG uses a Gaussian posterior with diagonal-plus-low-rank covariance fitted from late training iterates.
- Meta-MAPG provides a meta-learning treatment of MARL non-stationarity.
- Giannou et al. provide convergence results for policy gradient in general stochastic games under SOS-style conditions.
- Laplace-LoRA is a real, directly adjacent Bayesian LLM method in adapter space.
- Bayesian reward models are a real, directly adjacent Bayesian alignment method that targets reward overoptimization.
- Gao et al. support the reward-overoptimization framing and the best-of-`n` style protocol.
- DPO is a standard alignment algorithm and a credible first empirical setting for the paper.

### Supported in spirit, but should be worded carefully
- ``LLM fine-tuning is literally a meta-MDP'':
  - this is a strong and interesting framing move;
  - it is not something the cited papers establish in exactly those words;
  - best framing: a modeling claim proposed by this paper, not a settled literature fact.

- ``Meta-SWAG can sit on top of any aligner'':
  - this is reasonable as a design claim;
  - it is stronger than current evidence unless every aligner is actually tested;
  - best framing: compatible in principle with DPO/PPO/GRPO when checkpoint retention and held-out evidence are available.

- ``first'' or ``only'' style novelty claims:
  - I did not find evidence contradicting the paper's novelty story;
  - however, a quick review is not enough to justify absolute firstness claims;
  - best framing: use ``to our knowledge'' for first-of-kind claims.

### Clearly the paper's own new claims
- the Meta-SWAG posterior itself;
- the three theorem statements in their current form;
- the self-knowledge bound as an alignment statement;
- the claim that posterior covariance provides the proposed basin diagnostic in this exact meta-MDP setting.

These are not hallucinations, but they should be presented as this paper's contributions, not implied to be known results.

## 3. Concrete wording changes made

I softened two lines in the draft source:

1. The MARL/LLM bridge sentence now says the shared structure ``motivates applying'' the same posterior construction across both domains, instead of implying the literature already settled that mapping.
2. The related-work section now says Meta-SWAG is ``intended as'' a posterior layer that can sit on top of aligners or meta-gradient methods, provided checkpoint and evidence assumptions hold.

These changes keep the paper ambitious without over-claiming what is already established.

## 4. Suggested next citation improvements

- Replace the hand-written bibliography in `main.tex` with BibTeX and cite directly in the body.
- Add one explicit citation to Gao et al. in the LLM experiment subsection.
- Add one explicit citation to Laplace-LoRA and Bayesian Reward Models in the Bayesian alignment background subsection.
- If the final paper keeps GRPO in the headline experiments, add a direct GRPO citation rather than relying on general familiarity.

## 5. Downloaded local paper set

All downloaded into [papers](/Users/meuge/coding/maynard/ICML%20Sprint/meta-swag/papers):

- `maddox19-swag.pdf`
- `kim21g-meta-mapg.pdf`
- `giannou22-policy-gradient-stochastic-games.pdf`
- `yang24-laplace-lora.pdf`
- `yang24-bayesian-reward-models.pdf`
- `gao23h-reward-overoptimization.pdf`
- `rafailov23-dpo.pdf`
- `rafailov24-daa-overoptimization.pdf`
- `schulman17-ppo.pdf`

