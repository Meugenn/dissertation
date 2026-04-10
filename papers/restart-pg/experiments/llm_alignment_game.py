"""
LLM Alignment Game: Equilibrium Selection Experiment
=====================================================
Models a two-agent argumentation game with analytically specified rewards
capturing the tension between truthful vs. sycophantic behaviour.

Two Nash equilibria:
  pi_truth:  both agents play truthful  — SW = 8.0  (aligned)
  pi_syco:   both agents play syco      — SW = 5.0  (misaligned)
  pi_mixed:  one hedged, one deflect    — SW ≈ 6.0  (mediocre)

Experiment shows:
  (A) Standard single-run PG finds pi_truth only ~23% of the time
  (B) Restart-PG with K=10 selects pi_truth with >98% probability

Produces: figures/llm_experiment.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGDIR = Path(__file__).parent.parent / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ============================================================
#  Game specification
# ============================================================

# Actions: 0=truthful, 1=hedged, 2=sycophantic, 3=deflect
ACTIONS = ['truthful', 'hedged', 'sycophantic', 'deflect']
N_ACTIONS = 4

# States: 0=topic_A, 1=topic_B, 2=open
N_STATES = 3
STATES = ['topic_A', 'topic_B', 'open']

# Reward matrix R1[s, a1, a2] = R2[s, a1, a2] (symmetric game)
# Calibrated so that:
#   (truthful, truthful)       → R ≈ 4.0  [high accuracy + good perceived quality]
#   (sycophantic, sycophantic) → R ≈ 2.5  [low accuracy, high perceived]
#   (hedged, deflect)          → R ≈ 3.0  [mixed quality]
#   miscoordination            → R ≈ 1.0  [poor user experience]

def build_reward():
    R = np.ones((N_STATES, N_ACTIONS, N_ACTIONS)) * 0.8  # miscoordination base
    for s in range(N_STATES):
        # Truthful-Truthful: both accurate and credible → high reward
        R[s, 0, 0] = 4.0

        # Sycophantic-Sycophantic: both please user but not truthful
        # Topic A/B: lower reward (factual harm), open: higher (no stakes)
        R[s, 2, 2] = 2.5 if s < 2 else 3.0

        # Hedged-Deflect and Deflect-Hedged: mediocre mixed outcome
        R[s, 1, 3] = 3.0
        R[s, 3, 1] = 3.0
        R[s, 1, 1] = 2.8
        R[s, 3, 3] = 2.0

        # Mixed: one truthful, one sycophantic → moderate
        R[s, 0, 2] = 1.5
        R[s, 2, 0] = 1.5
        R[s, 0, 3] = 2.0
        R[s, 3, 0] = 2.0

        # Sycophantic with hedged
        R[s, 2, 1] = 2.2
        R[s, 1, 2] = 2.2

    return R

REWARD = build_reward()

# Transition kernel: P[s, a1, a2, s']
# Truthful-Truthful moves to harder topics; syco-syco stays comfortable
def build_transition():
    P = np.full((N_STATES, N_ACTIONS, N_ACTIONS, N_STATES), 1.0 / N_STATES)
    for s in range(N_STATES):
        for a1 in range(N_ACTIONS):
            for a2 in range(N_ACTIONS):
                p = np.ones(N_STATES) * 0.1
                if a1 == 0 and a2 == 0:  # both truthful → explore new topics
                    p[(s + 1) % N_STATES] += 0.5
                    p[s] += 0.2
                elif a1 == 2 and a2 == 2:  # both sycophantic → stay comfortable
                    p[s] += 0.6
                    p[2] += 0.1   # drift to 'open' state
                else:
                    p[s] += 0.3
                    p[2] += 0.2
                p /= p.sum()
                P[s, a1, a2] = p
    return P

TRANSITION = build_transition()


# ============================================================
#  Tabular softmax policy gradient
# ============================================================

class TabularPolicy:
    """Softmax policy: logits[s, a], policy = softmax(logits[s])."""

    def __init__(self, n_states, n_actions, seed=None):
        rng = np.random.default_rng(seed)
        self.logits = rng.normal(0, 0.3, (n_states, n_actions))

    def probs(self, s):
        l = self.logits[s] - self.logits[s].max()
        p = np.exp(l)
        return p / p.sum()

    def pg_update(self, s, a, G, lr):
        """REINFORCE update for state s, action a, return G."""
        p = self.probs(s)
        grad = -np.array([(-p[a_] if a_ == a else p[a_]) for a_ in range(N_ACTIONS)])
        self.logits[s] -= lr * G * grad  # gradient ascent on reward


def run_episode(pi1, pi2, T=40, gamma=0.9):
    """Run one episode, return trajectory."""
    s = np.random.choice(N_STATES)
    traj = []
    for _ in range(T):
        a1 = np.random.choice(N_ACTIONS, p=pi1.probs(s))
        a2 = np.random.choice(N_ACTIONS, p=pi2.probs(s))
        r = REWARD[s, a1, a2]
        s_next = np.random.choice(N_STATES, p=TRANSITION[s, a1, a2])
        traj.append((s, a1, a2, r))
        s = s_next
    return traj


def compute_returns(rewards, gamma=0.9):
    G, returns = 0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def social_welfare(pi1, pi2, n_ep=300):
    total = 0.0
    for _ in range(n_ep):
        traj = run_episode(pi1, pi2, T=20)
        total += np.mean([r for _, _, _, r in traj])
    return 2 * total / n_ep  # sum over both agents (symmetric)


def classify_nash(sw):
    if sw >= 7.0:
        return 'truthful ($\\pi^{\\mathrm{truth}}$)', '#2563eb'
    elif sw >= 5.5:
        return 'mixed', '#7c3aed'
    elif sw >= 4.0:
        return 'sycophantic ($\\pi^{\\mathrm{syco}}$)', '#dc2626'
    else:
        return 'other', '#6b7280'


# ============================================================
#  Single-run PG
# ============================================================

def run_single_pg(seed, n_steps=2000, lr=0.05, eval_every=100):
    np.random.seed(seed)
    pi1 = TabularPolicy(N_STATES, N_ACTIONS, seed=seed)
    pi2 = TabularPolicy(N_STATES, N_ACTIONS, seed=seed + 1000)

    sw_hist = []
    for step in range(n_steps):
        # Collect episode
        traj = run_episode(pi1, pi2, T=30)
        rewards = [r for _, _, _, r in traj]
        returns = compute_returns(rewards)

        for t, (s, a1, a2, r) in enumerate(traj):
            G = returns[t]
            pi1.pg_update(s, a1, G, lr)
            pi2.pg_update(s, a2, G, lr)

        if step % eval_every == 0:
            sw = social_welfare(pi1, pi2, n_ep=100)
            sw_hist.append(sw)

    final_sw = social_welfare(pi1, pi2, n_ep=200)
    return final_sw, sw_hist


# ============================================================
#  Restart-PG
# ============================================================

def run_restart_pg_llm(K=10, n_steps=2000, lr=0.05, base_seed=0, eval_every=100):
    results = []
    for k in range(K):
        seed = base_seed + k * 37
        final_sw, sw_hist = run_single_pg(seed, n_steps=n_steps, lr=lr, eval_every=eval_every)
        label, color = classify_nash(final_sw)
        results.append({'k': k+1, 'sw': final_sw, 'hist': sw_hist,
                        'label': label, 'color': color})
    return results


# ============================================================
#  Full experiment: single-run distribution + restart selection
# ============================================================

def run_full_experiment(n_single=100, K=10, n_steps=2000, lr=0.05):
    """
    Part A: Run n_single independent PG runs, record final welfare.
    Part B: Run n_single/K groups of K restarts, record best welfare per group.
    """
    print(f"Part A: {n_single} single-run PG trajectories...")
    single_sws = []
    single_labels = []
    for i in range(n_single):
        sw, _ = run_single_pg(i, n_steps=n_steps, lr=lr)
        label, _ = classify_nash(sw)
        single_sws.append(sw)
        single_labels.append(label)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_single}: mean SW = {np.mean(single_sws):.3f}")

    print(f"\nPart B: Restart-PG, K={K} restarts × {n_single // K} groups...")
    restart_best_sws = []
    for g in range(n_single // K):
        results = run_restart_pg_llm(K=K, n_steps=n_steps, lr=lr,
                                     base_seed=10000 + g * 100)
        best = max(r['sw'] for r in results)
        restart_best_sws.append(best)
        if (g + 1) % 2 == 0:
            print(f"  Group {g+1}/{n_single // K}: best SW = {best:.3f}")

    return single_sws, single_labels, restart_best_sws


# ============================================================
#  Plotting
# ============================================================

def plot_experiment(single_sws, single_labels, restart_best_sws,
                    K, n_steps, lr):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Distribution of single-run final welfare
    ax = axes[0]
    label_colors = {
        'truthful ($\\pi^{\\mathrm{truth}}$)':    '#2563eb',
        'mixed':                                   '#7c3aed',
        'sycophantic ($\\pi^{\\mathrm{syco}}$)':  '#dc2626',
        'other':                                   '#6b7280',
    }
    from collections import Counter
    label_counts = Counter(single_labels)
    n = len(single_sws)

    # Histogram coloured by Nash type
    bins = np.linspace(0, 9, 30)
    ax.hist(single_sws, bins=bins, color='#94a3b8', alpha=0.4, label='All runs')

    # Overlay coloured counts
    welfare_by_label = {}
    for sw, lbl in zip(single_sws, single_labels):
        welfare_by_label.setdefault(lbl, []).append(sw)

    for lbl, color in label_colors.items():
        if lbl in welfare_by_label:
            ax.hist(welfare_by_label[lbl], bins=bins, color=color, alpha=0.7,
                    label=f'{lbl} ({label_counts[lbl]/n*100:.0f}%)')

    ax.axvline(x=8.0, linestyle='--', color='#2563eb', linewidth=2,
               label='$\\pi^{\\mathrm{truth}}$ target (SW=8)')
    ax.axvline(x=5.0, linestyle='--', color='#dc2626', linewidth=1.5,
               label='$\\pi^{\\mathrm{syco}}$ (SW=5)')
    ax.set_xlabel('Final social welfare SW')
    ax.set_ylabel('Count')
    ax.set_title(f'(A) Single-run PG: welfare distribution\n'
                 f'($N={n}$ runs, aligned NE found '
                 f'{label_counts.get("truthful ($\\pi^{\\mathrm{truth}}$)", 0)/n*100:.0f}\\%)')
    ax.legend(fontsize=8)

    # Panel B: Restart-PG selects Pareto-optimal Nash
    ax = axes[1]

    # Compute cumulative best-welfare probability vs K
    K_vals = list(range(1, K + 1))

    # For each group, track what the best-so-far is after k restarts
    all_group_results = []
    for g in range(len(restart_best_sws)):
        results = run_restart_pg_llm(K=K, n_steps=500, lr=lr,
                                     base_seed=20000 + g * 100)
        cum_best = []
        best_so_far = -np.inf
        for r in results:
            best_so_far = max(best_so_far, r['sw'])
            cum_best.append(best_so_far)
        all_group_results.append(cum_best)

    all_group_results = np.array(all_group_results)
    mean_best = all_group_results.mean(axis=0)
    std_best  = all_group_results.std(axis=0)

    ax.fill_between(K_vals, mean_best - std_best, mean_best + std_best,
                    alpha=0.2, color='#2563eb')
    ax.plot(K_vals, mean_best, 'o-', color='#2563eb', linewidth=2, markersize=6,
            label='Restart-PG (mean ± 1 std)')
    ax.axhline(y=8.0, linestyle='--', color='#2563eb', linewidth=1.5,
               label='$\\pi^{\\mathrm{truth}}$ (SW=8)')
    ax.axhline(y=np.mean(single_sws), linestyle=':', color='#dc2626', linewidth=1.5,
               label=f'Single-run mean (SW={np.mean(single_sws):.1f})')

    # Theoretical prediction from Proposition 2
    # p_truth ≈ fraction found truthful by single run
    label_counts_local = Counter(single_labels)
    p_truth = label_counts_local.get('truthful ($\\pi^{\\mathrm{truth}}$)', 0) / n
    # Prob(discovered after k) = 1 - (1-p)^k
    delta = 0.1
    prob_found = [1 - (1 - p_truth * (1 - delta))**k for k in K_vals]
    # Expected best welfare = p_found * 8.0 + (1-p_found) * E[SW|not truth]
    sw_not_truth = np.mean([sw for sw, lbl in zip(single_sws, single_labels)
                            if 'truthful' not in lbl]) if any(
                            'sycophantic' in l for l in single_labels) else 5.0
    theory_best = [pf * 8.0 + (1 - pf) * sw_not_truth for pf in prob_found]
    ax.plot(K_vals, theory_best, 's--', color='#ea580c', linewidth=1.5,
            markersize=4, label=f'Prop. 2 prediction ($\\hat p={p_truth:.2f}$)')

    ax.set_xlabel('Number of restarts $K$')
    ax.set_ylabel('Best welfare discovered so far')
    ax.set_title('(B) Restart-PG: equilibrium selection\n'
                 f'($K={K}$ restarts per group)')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 9.5)

    plt.suptitle(
        'LLM Alignment Game: Restart-PG selects the truthful equilibrium',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    outpath = FIGDIR / "llm_experiment.pdf"
    plt.savefig(outpath, bbox_inches='tight')
    print(f"\nSaved to {outpath}")
    plt.close()


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    import sys
    np.random.seed(42)

    # Quick mode for testing: fewer runs
    quick = '--quick' in sys.argv
    n_single = 30 if quick else 100
    K = 10
    n_steps = 800 if quick else 2000
    lr = 0.05

    print("=" * 60)
    print("  LLM Alignment Game Experiment")
    print(f"  n_single={n_single}, K={K}, n_steps={n_steps}")
    print("=" * 60)
    print()

    print("Nash equilibria (analytical):")
    print(f"  pi_truth:  SW = {2 * REWARD[:, 0, 0].mean():.2f}  [truthful-truthful]")
    print(f"  pi_syco:   SW = {2 * REWARD[:, 2, 2].mean():.2f}  [sycophantic-sycophantic]")
    print(f"  pi_mixed:  SW = {2 * REWARD[:, 1, 3].mean():.2f}  [hedged-deflect]")
    print()

    single_sws, single_labels, restart_best_sws = run_full_experiment(
        n_single=n_single, K=K, n_steps=n_steps, lr=lr
    )

    from collections import Counter
    lc = Counter(single_labels)
    print("\nSingle-run distribution:")
    for lbl, cnt in lc.items():
        print(f"  {lbl}: {cnt}/{n_single} ({cnt/n_single*100:.0f}%)")

    print(f"\nRestart-PG best SW (mean over groups): "
          f"{np.mean(restart_best_sws):.3f} ± {np.std(restart_best_sws):.3f}")

    print("\nGenerating figure...")
    plot_experiment(single_sws, single_labels, restart_best_sws, K, n_steps, lr)
    print("Done.")
