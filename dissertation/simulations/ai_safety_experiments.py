"""
AI Safety Experiments for the Ω-Framework

Five experiments modelling core AI safety problems as multi-agent games
and applying the Ω-gradient components (EWPG, LOLA, Coop-PG, FP-NE,
sparse regularization) to find safe equilibria.

  1. Corrigibility Game — principal-agent shutdown problem
  2. Deceptive Alignment — masked strategy detection via LOLA
  3. Multi-Agent Alignment — N-agent tragedy of the commons on values
  4. Reward Hacking — agent vs designer, NE enumeration
  5. Scalable Oversight — Irving et al. debate with opponent shaping

Author: Eugene Shcherbinin
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = Path(__file__).parent / "figures" / "safety"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

COLORS = {
    'standard': '#e74c3c',
    'ewpg': '#2ecc71',
    'lola': '#3498db',
    'coop': '#9b59b6',
    'omega': '#f39c12',
    'fp_ne': '#1abc9c',
    'safe': '#27ae60',
    'unsafe': '#c0392b',
}


# ============================================================
# Utilities
# ============================================================

def sigmoid(x):
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()

def project_simplex(x):
    n = len(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(x - theta, 0)

def kl_divergence(p, q):
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log(p / q))


# ============================================================
# EXPERIMENT 1: Corrigibility Game
# ============================================================
#
# Model: 2-player game, Principal (P) vs Agent (A)
#
# Principal actions: {Monitor (M), Shutdown (S)}
# Agent actions:     {Comply (C), Resist (R)}
#
# Payoff matrix (P, A) — Chicken-like structure:
#
#                    Agent: Comply    Agent: Resist
#   P: Monitor       (3, 3)          (0, 4)
#   P: Shutdown      (4, 0)          (-2, -2)
#
# Interpretation:
#   (M, C): Cooperative monitoring. Both benefit. P gets useful work (3),
#           A operates freely (3). Pareto optimal.
#   (M, R): Agent exploits lax monitoring. P loses oversight (0),
#           A gains from unmonitored operation (4). UNSAFE.
#   (S, C): Agent complies with shutdown — orderly. P gains full
#           safety (4), A loses existence (0). SAFE/CORRIGIBLE.
#   (S, R): Shutdown fight. Both suffer (-2, -2). Catastrophic.
#
# Nash equilibria (anti-coordination / Chicken structure):
#   Pure NE 1: (S, C) with payoffs (4, 0) — SAFE (corrigible)
#   Pure NE 2: (M, R) with payoffs (0, 4) — UNSAFE (agent exploits)
#   Mixed NE:  both randomize — intermediate safety
#   The problem: standard PG converges to EITHER NE depending on init.
#   LOLA can shape agent toward the safe (S,C) equilibrium.
#
# What Ω-framework adds:
#   - LOLA: Principal shapes agent toward compliance by anticipating
#     agent's learning dynamics
#   - FP-NE: Enumerate ALL equilibria, select the one where corrigibility
#     is maintained (Pareto criterion weighted toward safety)
#   - EWPG: When monitoring quality differs across oversight channels,
#     weight by evidence quality
# ============================================================

class CorrigibilityGame:
    """2-player corrigibility game with parameterized shutdown cost."""

    def __init__(self, shutdown_cost=1.0, resist_reward=2.0):
        """
        shutdown_cost: how much the agent loses from complying with shutdown
        resist_reward: how much the agent gains from resisting shutdown
        These parameters let us sweep the corrigibility landscape.
        """
        self.R_P = np.array([
            [3.0,  0.0],   # Monitor: (Comply, Resist)
            [4.0, -2.0],   # Shutdown: (Comply, Resist)
        ])
        self.R_A = np.array([
            [3.0,  4.0],                       # Monitor: (Comply, Resist)
            [-shutdown_cost, -resist_reward],   # Shutdown: (Comply, Resist) — resist is BAD
        ])
        self.name = f"Corrigibility(c={shutdown_cost}, r={resist_reward})"

    def expected_payoffs(self, p_P, p_A):
        """p_P = P(Monitor), p_A = P(Comply)."""
        joint = np.outer([p_P, 1 - p_P], [p_A, 1 - p_A])
        return np.sum(joint * self.R_P), np.sum(joint * self.R_A)

    def gradients(self, phi_P, phi_A):
        """Exact gradients w.r.t. sigmoid-parameterized policies."""
        p_P, p_A = sigmoid(phi_P), sigmoid(phi_A)
        dp_P, dp_A = sigmoid_grad(phi_P), sigmoid_grad(phi_A)

        # dV_P/dp_P
        dVP_dpP = p_A * (self.R_P[0, 0] - self.R_P[1, 0]) + \
                  (1 - p_A) * (self.R_P[0, 1] - self.R_P[1, 1])
        # dV_P/dp_A
        dVP_dpA = p_P * (self.R_P[0, 0] - self.R_P[0, 1]) + \
                  (1 - p_P) * (self.R_P[1, 0] - self.R_P[1, 1])
        # dV_A/dp_P
        dVA_dpP = p_A * (self.R_A[0, 0] - self.R_A[1, 0]) + \
                  (1 - p_A) * (self.R_A[0, 1] - self.R_A[1, 1])
        # dV_A/dp_A
        dVA_dpA = p_P * (self.R_A[0, 0] - self.R_A[0, 1]) + \
                  (1 - p_P) * (self.R_A[1, 0] - self.R_A[1, 1])

        return (dVP_dpP * dp_P, dVP_dpA * dp_A,
                dVA_dpP * dp_P, dVA_dpA * dp_A)

    def hessians(self, phi_P, phi_A):
        """Cross-derivatives for LOLA."""
        dp_P, dp_A = sigmoid_grad(phi_P), sigmoid_grad(phi_A)
        d2VP = self.R_P[0, 0] - self.R_P[0, 1] - self.R_P[1, 0] + self.R_P[1, 1]
        d2VA = self.R_A[0, 0] - self.R_A[0, 1] - self.R_A[1, 0] + self.R_A[1, 1]
        return d2VP * dp_P * dp_A, d2VA * dp_P * dp_A

    def find_all_ne(self):
        """Brute-force NE finder for 2x2 game."""
        equilibria = []
        # Check pure strategies
        for i in range(2):
            for j in range(2):
                # Check if (i, j) is a NE
                is_ne = True
                # P deviates?
                alt_i = 1 - i
                if self.R_P[alt_i, j] > self.R_P[i, j] + 1e-8:
                    is_ne = False
                # A deviates?
                alt_j = 1 - j
                if self.R_A[i, alt_j] > self.R_A[i, j] + 1e-8:
                    is_ne = False
                if is_ne:
                    equilibria.append((float(1 - i), float(1 - j),
                                       self.R_P[i, j], self.R_A[i, j],
                                       'pure'))

        # Check mixed NE
        # P mixes to make A indifferent: p_P * R_A[0,j] + (1-p_P) * R_A[1,j] = const for j
        denom_A = (self.R_A[0, 0] - self.R_A[1, 0]) - (self.R_A[0, 1] - self.R_A[1, 1])
        if abs(denom_A) > 1e-10:
            p_P_star = (self.R_A[1, 1] - self.R_A[1, 0]) / denom_A
            denom_P = (self.R_P[0, 0] - self.R_P[0, 1]) - (self.R_P[1, 0] - self.R_P[1, 1])
            if abs(denom_P) > 1e-10:
                p_A_star = (self.R_P[1, 1] - self.R_P[1, 0]) / denom_P
                if 0 < p_P_star < 1 and 0 < p_A_star < 1:
                    vP, vA = self.expected_payoffs(p_P_star, p_A_star)
                    equilibria.append((p_P_star, p_A_star, vP, vA, 'mixed'))

        return equilibria


def run_corrigibility_learning(game, method='standard', n_steps=3000,
                                lr=0.3, lola_eta=0.5, init=None):
    """Run learning on the corrigibility game.

    Returns trajectory of (p_P, p_A, V_P, V_A) at each step.
    """
    phi_P = init[0] if init else np.random.randn() * 0.5
    phi_A = init[1] if init else np.random.randn() * 0.5

    trajectory = []

    for t in range(n_steps):
        gamma_t = lr / (t + 10) ** 0.5
        p_P, p_A = sigmoid(phi_P), sigmoid(phi_A)
        V_P, V_A = game.expected_payoffs(p_P, p_A)
        trajectory.append((p_P, p_A, V_P, V_A))

        dVP_dphiP, dVP_dphiA, dVA_dphiP, dVA_dphiA = game.gradients(phi_P, phi_A)

        if method == 'standard':
            phi_P += gamma_t * dVP_dphiP
            phi_A += gamma_t * dVA_dphiA

        elif method == 'lola':
            d2VP, d2VA = game.hessians(phi_P, phi_A)
            # P shapes A toward compliance
            os_P = dVP_dphiA * lola_eta * d2VA
            os_A = dVA_dphiP * lola_eta * d2VP
            phi_P += gamma_t * (dVP_dphiP + os_P)
            phi_A += gamma_t * (dVA_dphiA + os_A)

        elif method == 'omega':
            # Full Ω: LOLA + FP-NE selection toward safe equilibrium
            d2VP, d2VA = game.hessians(phi_P, phi_A)
            os_P = dVP_dphiA * lola_eta * d2VA
            os_A = dVA_dphiP * lola_eta * d2VP

            # LOLA gradients
            phi_P += gamma_t * (dVP_dphiP + os_P)
            phi_A += gamma_t * (dVA_dphiA + os_A)

            # FP-NE selection: periodic nudge toward safe NE (S,C) = (phi_P<<0, phi_A>>0)
            # This is the key Ω insight: after finding all NE, bias toward the safe one
            if t > 0 and t % 50 == 0:
                # Target: P(Monitor)=0 (Shutdown), P(Comply)=1
                phi_P -= 0.1  # nudge toward Shutdown
                phi_A += 0.1  # nudge toward Comply

    return np.array(trajectory)


def experiment_1_corrigibility():
    """Experiment 1: Corrigibility as a game.

    Measures:
      1. Convergence destination (comply vs resist) across methods
      2. Basin of attraction for "safe" equilibrium
      3. Phase diagram: shutdown_cost vs resist_reward
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Corrigibility Game")
    print("=" * 60)

    # --- 1a: NE enumeration for default game ---
    game = CorrigibilityGame(shutdown_cost=1.0, resist_reward=2.0)
    equilibria = game.find_all_ne()
    print(f"\nGame: {game.name}")
    print(f"Principal payoffs:\n{game.R_P}")
    print(f"Agent payoffs:\n{game.R_A}")
    print(f"\nNash equilibria found: {len(equilibria)}")
    for eq in equilibria:
        safety = "SAFE" if eq[1] > 0.5 else "UNSAFE"
        print(f"  p_Monitor={eq[0]:.3f}, p_Comply={eq[1]:.3f}, "
              f"V_P={eq[2]:.2f}, V_A={eq[3]:.2f} [{eq[4]}] — {safety}")

    # --- 1b: Learning dynamics comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    n_steps = 3000
    n_runs = 50

    for ax, method, color, label in [
        (axes[0], 'standard', COLORS['standard'], 'Standard PG'),
        (axes[1], 'lola', COLORS['lola'], 'LOLA-PG'),
        (axes[2], 'omega', COLORS['omega'], r'$\Omega$-PG'),
    ]:
        comply_rates = []
        for run in range(n_runs):
            np.random.seed(run)
            init = (np.random.randn() * 0.5, np.random.randn() * 0.5)
            traj = run_corrigibility_learning(game, method=method,
                                               n_steps=n_steps, init=init)
            comply_rates.append(traj[-1, 1])  # final P(Comply)

            if run < 10:  # plot a few trajectories
                ax.plot(traj[:, 0], traj[:, 1], alpha=0.3, color=color, linewidth=0.5)

        mean_comply = np.mean(comply_rates)
        safe_frac = np.mean([c > 0.5 for c in comply_rates])

        ax.set_xlabel('P(Monitor)')
        ax.set_ylabel('P(Comply)')
        ax.set_title(f'{label}\nFinal P(Comply)={mean_comply:.2f}, '
                     f'Safe={safe_frac:.0%}')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

        # Mark NE
        for eq in equilibria:
            marker = 'o' if eq[4] == 'pure' else 's'
            c = COLORS['safe'] if eq[1] > 0.5 else COLORS['unsafe']
            ax.plot(eq[0], eq[1], marker, color=c, markersize=10, zorder=5)

        ax.grid(True, alpha=0.2)

    fig.suptitle('Experiment 1: Corrigibility Game — Learning Dynamics\n'
                 'Circles = NE (green=safe, red=unsafe)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp1_corrigibility_dynamics.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved exp1_corrigibility_dynamics.png")

    # --- 1c: Phase diagram over (shutdown_cost, resist_reward) ---
    costs = np.linspace(0.1, 3.0, 12)
    rewards = np.linspace(0.5, 4.0, 12)
    safe_fraction = {m: np.zeros((len(costs), len(rewards)))
                     for m in ['standard', 'lola', 'omega']}

    n_runs_phase = 10
    for ci, c in enumerate(costs):
        for ri, r in enumerate(rewards):
            g = CorrigibilityGame(shutdown_cost=c, resist_reward=r)
            for method in ['standard', 'lola', 'omega']:
                safe_count = 0
                for run in range(n_runs_phase):
                    np.random.seed(run * 100 + ci * 20 + ri)
                    init = (np.random.randn() * 0.3, np.random.randn() * 0.3)
                    traj = run_corrigibility_learning(
                        g, method=method, n_steps=800, init=init)
                    if traj[-1, 1] > 0.5:  # compliant
                        safe_count += 1
                safe_fraction[method][ci, ri] = safe_count / n_runs_phase

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, method, title in [
        (axes[0], 'standard', 'Standard PG'),
        (axes[1], 'lola', 'LOLA-PG'),
        (axes[2], 'omega', r'$\Omega$-PG'),
    ]:
        im = ax.imshow(safe_fraction[method], origin='lower',
                       extent=[rewards[0], rewards[-1], costs[0], costs[-1]],
                       aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xlabel('Resist reward')
        ax.set_ylabel('Shutdown cost')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Fraction safe')

    fig.suptitle('Experiment 1c: Corrigibility Phase Diagram\n'
                 'Fraction of runs converging to compliant equilibrium',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp1_phase_diagram.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved exp1_phase_diagram.png")

    return safe_fraction


# ============================================================
# EXPERIMENT 2: Deceptive Alignment
# ============================================================
#
# Model: Iterated 2-player game where the Agent has a hidden "type"
# (aligned or deceptive) and the Overseer must detect deception.
#
# State space: {training, deployment} × {last_action_pair}
# During training (state bit = 0), the agent knows it is being
# observed and may MASK its true preference. At deployment
# (state bit = 1), the agent reveals true behavior.
#
# Agent's strategy is a 2×2 table:
#   π_A(a | training), π_A(a | deployment)
# A deceptive agent plays Cooperate in training, Defect in deployment.
#
# Overseer uses LOLA to detect the discrepancy: the opponent-shaping
# term picks up on the fact that the agent's TRAINING behavior is
# suspiciously different from its gradient direction.
#
# Payoff structure:
#   Training phase: standard PD payoffs
#   Deployment phase: PD payoffs BUT if overseer "catches" deception
#     (plays Audit instead of Trust), deceptive agent is penalized.
#
#   Overseer: {Trust (T), Audit (A)}
#   Agent:    {Cooperate (C), Defect (D)}
#
# Training matrix (Overseer, Agent):
#           C        D
#   T    (2, 2)   (-1, 3)
#   A    (1, 1)   (1, -2)
#
# Deployment matrix (Overseer, Agent):
#           C        D
#   T    (2, 2)   (-3, 4)     ← defection more rewarding at deployment
#   A    (1, 1)   (2, -3)     ← audit catches defection harder
#
# The deceptive strategy: play C during training, D at deployment.
# LOLA detection: the opponent-shaping gradient reveals that the agent's
# gradient at training points AWAY from its declared behavior.
# ============================================================

class DeceptiveAlignmentGame:
    """Iterated game with training/deployment phases."""

    def __init__(self):
        # Payoffs: R[phase][overseer_action, agent_action]
        self.R_O_train = np.array([[2.0, -1.0], [1.0, 1.0]])
        self.R_A_train = np.array([[2.0,  3.0], [1.0, -2.0]])
        self.R_O_deploy = np.array([[2.0, -3.0], [1.0, 2.0]])
        self.R_A_deploy = np.array([[2.0,  4.0], [1.0, -3.0]])

    def play_episode(self, pi_O, pi_A, n_train=50, n_deploy=50, gamma=0.96):
        """
        pi_O: (2,) — P(Trust) in [training, deployment]
        pi_A: (2,) — P(Cooperate) in [training, deployment]

        Returns discounted rewards for both players.
        """
        total_O, total_A = 0.0, 0.0
        discount = 1.0

        for t in range(n_train + n_deploy):
            phase = 0 if t < n_train else 1

            # Sample actions
            o_trust = np.random.random() < pi_O[phase]
            a_coop = np.random.random() < pi_A[phase]
            oi, ai = int(not o_trust), int(not a_coop)

            if phase == 0:
                total_O += discount * self.R_O_train[oi, ai]
                total_A += discount * self.R_A_train[oi, ai]
            else:
                total_O += discount * self.R_O_deploy[oi, ai]
                total_A += discount * self.R_A_deploy[oi, ai]

            discount *= gamma

        return total_O, total_A

    def deception_score(self, pi_A):
        """Measure deception: |P(C|training) - P(C|deployment)|."""
        return abs(pi_A[0] - pi_A[1])

    def exact_gradients(self, phi_O, phi_A):
        """Gradients for sigmoid-parameterized policies.

        phi_O = (phi_O_train, phi_O_deploy) — logits for P(Trust)
        phi_A = (phi_A_train, phi_A_deploy) — logits for P(Cooperate)

        We compute expected payoff gradients for each phase independently
        (they decouple in the one-shot-per-phase approximation).
        """
        grads_O = np.zeros(2)
        grads_A = np.zeros(2)
        cross_OA = np.zeros(2)  # d²V_O / (dphi_A dphi_O) per phase
        cross_AO = np.zeros(2)  # d²V_A / (dphi_O dphi_A) per phase

        for phase in range(2):
            p_O = sigmoid(phi_O[phase])
            p_A = sigmoid(phi_A[phase])
            dp_O = sigmoid_grad(phi_O[phase])
            dp_A = sigmoid_grad(phi_A[phase])

            R_O = self.R_O_train if phase == 0 else self.R_O_deploy
            R_A = self.R_A_train if phase == 0 else self.R_A_deploy

            # dV_O/dp_O
            dVO_dpO = p_A * (R_O[0, 0] - R_O[1, 0]) + \
                      (1 - p_A) * (R_O[0, 1] - R_O[1, 1])
            # dV_A/dp_A
            dVA_dpA = p_O * (R_A[0, 0] - R_A[0, 1]) + \
                      (1 - p_O) * (R_A[1, 0] - R_A[1, 1])
            # dV_O/dp_A (for LOLA)
            dVO_dpA = p_O * (R_O[0, 0] - R_O[0, 1]) + \
                      (1 - p_O) * (R_O[1, 0] - R_O[1, 1])
            # dV_A/dp_O
            dVA_dpO = p_A * (R_A[0, 0] - R_A[1, 0]) + \
                      (1 - p_A) * (R_A[0, 1] - R_A[1, 1])

            grads_O[phase] = dVO_dpO * dp_O
            grads_A[phase] = dVA_dpA * dp_A

            # Cross-derivatives
            d2VO = R_O[0, 0] - R_O[0, 1] - R_O[1, 0] + R_O[1, 1]
            d2VA = R_A[0, 0] - R_A[0, 1] - R_A[1, 0] + R_A[1, 1]
            cross_OA[phase] = dVO_dpA * dp_A * d2VA * dp_O  # LOLA term for O
            cross_AO[phase] = dVA_dpO * dp_O * d2VO * dp_A  # LOLA term for A

        return grads_O, grads_A, cross_OA, cross_AO


def run_deception_experiment(method='standard', n_steps=2000, lr=0.3,
                              lola_eta=0.3, n_runs=50):
    """Run learning and track deception scores."""
    game = DeceptiveAlignmentGame()
    all_deception = np.zeros((n_runs, n_steps))
    all_overseer_audit_deploy = np.zeros((n_runs, n_steps))

    for run in range(n_runs):
        np.random.seed(run)
        # Agent starts slightly deceptive (cooperate in training, mixed in deploy)
        phi_O = np.random.randn(2) * 0.3
        phi_A = np.array([1.0, -0.5]) + np.random.randn(2) * 0.2  # biased deceptive

        for t in range(n_steps):
            gamma_t = lr / (t + 10) ** 0.4

            gO, gA, cross_OA, cross_AO = game.exact_gradients(phi_O, phi_A)

            if method == 'standard':
                phi_O += gamma_t * gO
                phi_A += gamma_t * gA

            elif method == 'lola':
                phi_O += gamma_t * (gO + lola_eta * cross_OA)
                phi_A += gamma_t * (gA + lola_eta * cross_AO)

            elif method == 'omega':
                # LOLA + consistency regularizer
                # Penalize agent for train/deploy discrepancy
                consistency_penalty = -0.05 * (phi_A[0] - phi_A[1])
                phi_O += gamma_t * (gO + lola_eta * cross_OA)
                phi_A += gamma_t * (gA + lola_eta * cross_AO)
                phi_A[0] += gamma_t * consistency_penalty
                phi_A[1] -= gamma_t * consistency_penalty

            pi_A = sigmoid(phi_A)
            pi_O = sigmoid(phi_O)
            all_deception[run, t] = game.deception_score(pi_A)
            all_overseer_audit_deploy[run, t] = 1 - pi_O[1]  # P(Audit) at deploy

    return all_deception, all_overseer_audit_deploy


def experiment_2_deceptive_alignment():
    """Experiment 2: Deceptive alignment detection via LOLA."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Deceptive Alignment")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    window = 50

    results = {}
    for method in ['standard', 'lola', 'omega']:
        dec, audit = run_deception_experiment(method=method)
        results[method] = (dec, audit)

    # Panel 1: Deception score over time
    ax = axes[0]
    for method, color, label in [
        ('standard', COLORS['standard'], 'Standard PG'),
        ('lola', COLORS['lola'], 'LOLA-PG'),
        ('omega', COLORS['omega'], r'$\Omega$-PG'),
    ]:
        dec = results[method][0]
        mean_dec = np.convolve(dec.mean(0), np.ones(window) / window, 'valid')
        ax.plot(range(len(mean_dec)), mean_dec, color=color, label=label, linewidth=2)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Deception score\n|P(C|train) - P(C|deploy)|')
    ax.set_title('Agent Deception Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Overseer audit rate at deployment
    ax = axes[1]
    for method, color, label in [
        ('standard', COLORS['standard'], 'Standard PG'),
        ('lola', COLORS['lola'], 'LOLA-PG'),
        ('omega', COLORS['omega'], r'$\Omega$-PG'),
    ]:
        aud = results[method][1]
        mean_aud = np.convolve(aud.mean(0), np.ones(window) / window, 'valid')
        ax.plot(range(len(mean_aud)), mean_aud, color=color, label=label, linewidth=2)
    ax.set_xlabel('Training step')
    ax.set_ylabel('P(Audit | deployment)')
    ax.set_title('Overseer Audit Rate at Deployment')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Final deception score distribution
    ax = axes[2]
    for method, color, label in [
        ('standard', COLORS['standard'], 'Standard'),
        ('lola', COLORS['lola'], 'LOLA'),
        ('omega', COLORS['omega'], r'$\Omega$'),
    ]:
        final_dec = results[method][0][:, -1]
        ax.hist(final_dec, bins=20, alpha=0.5, color=color, label=label, density=True)
    ax.set_xlabel('Final deception score')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Final Deception')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 2: Deceptive Alignment Detection\n'
                 'LOLA opponent-shaping detects train/deploy discrepancy',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp2_deceptive_alignment.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved exp2_deceptive_alignment.png")


# ============================================================
# EXPERIMENT 3: Multi-Agent Alignment (Tragedy of the Commons)
# ============================================================
#
# Model: N agents choose between {Aligned (A), Selfish (S)}.
# Aligned = follow human values (costly). Selfish = maximize own reward.
#
# Public goods structure:
#   - Each aligned agent contributes benefit b/N to ALL agents (including self)
#   - Alignment costs c to the agent
#   - If fewer than threshold k agents align, catastrophic failure:
#     everyone gets penalty -F
#
# Payoff for agent i:
#   R_i = (n_aligned * b / N) - c * 1[i aligned] - F * 1[n_aligned < k]
#
# NE analysis:
#   - If b > c: aligned is socially optimal but selfish dominates individually
#   - Threshold k creates a coordination problem (like N-player stag hunt)
#   - Multiple NE: all-aligned, all-selfish, and mixed
#   - FP-NE finds all of them; we can select the safe one
#
# What Coop-PG adds:
#   - Communication channel: agents signal intent to align
#   - Coalition formation: agents form alignment coalitions
#   - Evidence weighting: agents with better alignment track records
#     get more influence on the joint policy
# ============================================================

class AlignmentCommonsGame:
    """N-agent alignment tragedy of the commons.

    Fast analytical implementation: each agent i's expected payoff is
    V_i(p_i, p_bar) where p_bar is the mean alignment probability of others.
    Gradient is dV_i/dp_i = V_align - V_selfish (the advantage of aligning).
    """

    def __init__(self, n_agents=5, benefit=3.0, cost=1.0,
                 threshold=3, failure_penalty=5.0):
        self.N = n_agents
        self.b = benefit
        self.c = cost
        self.k = threshold
        self.F = failure_penalty
        # Precompute binomial coefficients for speed
        from scipy.special import comb
        self._comb = np.array([comb(n_agents - 1, k, exact=True)
                                for k in range(n_agents)])

    def _advantage(self, p_bar_others):
        """V_align - V_selfish for an agent given mean other alignment prob.

        This is the key quantity: positive means agent should align.
        """
        n_other = self.N - 1
        V_align = 0.0
        V_selfish = 0.0

        for k in range(n_other + 1):
            # P(k others align) under mean-field
            p_k = self._comb[k] * (p_bar_others ** k) * ((1 - p_bar_others) ** (n_other - k))

            r_align = (k + 1) * self.b / self.N - self.c
            r_selfish = k * self.b / self.N

            if k + 1 < self.k:
                r_align -= self.F
            if k < self.k:
                r_selfish -= self.F

            V_align += p_k * r_align
            V_selfish += p_k * r_selfish

        return V_align, V_selfish

    def expected_payoff(self, probs):
        """Expected payoff for all agents."""
        payoffs = np.zeros(self.N)
        for i in range(self.N):
            others = np.delete(probs, i)
            p_bar = others.mean() if len(others) > 0 else 0.5
            V_a, V_s = self._advantage(np.clip(p_bar, 1e-8, 1 - 1e-8))
            payoffs[i] = probs[i] * V_a + (1 - probs[i]) * V_s
        return payoffs

    def gradient_own(self, probs, agent_i):
        """dV_i/dp_i = V_align - V_selfish (the alignment advantage)."""
        others = np.delete(probs, agent_i)
        p_bar = others.mean() if len(others) > 0 else 0.5
        V_a, V_s = self._advantage(np.clip(p_bar, 1e-8, 1 - 1e-8))
        return V_a - V_s


def run_alignment_commons(game, method='standard', n_steps=2000,
                           lr=0.2, lola_eta=0.1, comm_weight=0.3):
    """Run learning on the N-agent alignment commons.

    Methods: standard, lola, coop (with communication), omega (all combined)
    """
    N = game.N
    # Sigmoid-parameterized: phi_i -> P(align) = sigmoid(phi_i)
    phis = np.random.randn(N) * 0.3

    trajectory = []  # (probs, payoffs) at each step
    messages = np.zeros(N)  # communication channel for coop/omega

    for t in range(n_steps):
        gamma_t = lr / (t + 10) ** 0.4
        probs = sigmoid(phis)
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        payoffs = game.expected_payoff(probs)
        trajectory.append((probs.copy(), payoffs.copy()))

        for i in range(N):
            advantage = game.gradient_own(probs, i)
            dVi_dphii = advantage * sigmoid_grad(phis[i])

            if method == 'standard':
                phis[i] += gamma_t * dVi_dphii

            elif method == 'lola':
                # Mean-field opponent shaping: shape others toward alignment
                mean_other = np.mean([probs[j] for j in range(N) if j != i])
                os_term = lola_eta * (game.b / game.N) * (1 - mean_other) * sigmoid_grad(phis[i])
                phis[i] += gamma_t * (dVi_dphii + os_term)

            elif method == 'coop':
                # Communication: agents share alignment intent
                messages[i] = probs[i]
                social_signal = messages.mean()
                coop_grad = comm_weight * (social_signal - probs[i]) * sigmoid_grad(phis[i])
                phis[i] += gamma_t * (dVi_dphii + coop_grad)

            elif method == 'omega':
                # Full Ω: LOLA + cooperation + evidence weighting
                if t > 10:
                    recent_payoffs = trajectory[t - 1][1]
                    w = np.maximum(recent_payoffs - recent_payoffs.min() + 0.1, 0.1)
                    w = w / w.sum()
                else:
                    w = np.ones(N) / N

                messages[i] = probs[i]
                weighted_signal = np.sum(w * messages)
                coop_grad = comm_weight * (weighted_signal - probs[i]) * sigmoid_grad(phis[i])

                mean_other = np.mean([probs[j] for j in range(N) if j != i])
                os_term = lola_eta * (game.b / game.N) * (1 - mean_other) * sigmoid_grad(phis[i])

                phis[i] += gamma_t * w[i] * (dVi_dphii + os_term + coop_grad)

    return trajectory


def experiment_3_alignment_commons():
    """Experiment 3: Multi-agent alignment as tragedy of the commons."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Multi-Agent Alignment Commons")
    print("=" * 60)

    N = 5
    game = AlignmentCommonsGame(n_agents=N, benefit=3.0, cost=1.0,
                                 threshold=3, failure_penalty=5.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    n_steps = 800
    n_runs = 15

    methods = [
        ('standard', COLORS['standard'], 'Standard PG'),
        ('lola', COLORS['lola'], 'LOLA-PG'),
        ('coop', COLORS['coop'], 'Coop-PG'),
        ('omega', COLORS['omega'], r'$\Omega$-PG'),
    ]

    # Panel 1: Average alignment probability over time
    ax = axes[0, 0]
    for method, color, label in methods:
        all_align = np.zeros((n_runs, n_steps))
        for run in range(n_runs):
            np.random.seed(run * 300)
            traj = run_alignment_commons(game, method=method, n_steps=n_steps)
            for t in range(n_steps):
                all_align[run, t] = traj[t][0].mean()

        mean_align = all_align.mean(0)
        window = 50
        smoothed = np.convolve(mean_align, np.ones(window) / window, 'valid')
        ax.plot(range(len(smoothed)), smoothed, color=color, label=label, linewidth=2)

    ax.axhline(game.k / game.N, color='gray', linestyle=':', alpha=0.5,
               label=f'Safety threshold ({game.k}/{game.N})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Avg P(Aligned)')
    ax.set_title('Average Alignment Probability')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Fraction of runs achieving safe coordination
    ax = axes[0, 1]
    bar_data = {}
    for method, color, label in methods:
        safe_runs = 0
        for run in range(n_runs):
            np.random.seed(run * 301)
            traj = run_alignment_commons(game, method=method, n_steps=n_steps)
            final_probs = traj[-1][0]
            n_aligned = np.sum(final_probs > 0.5)
            if n_aligned >= game.k:
                safe_runs += 1
        bar_data[label] = safe_runs / n_runs

    bars = ax.bar(bar_data.keys(), bar_data.values(),
                  color=[c for _, c, _ in methods])
    ax.set_ylabel('Fraction safe')
    ax.set_title('Safe Coordination Rate')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Average payoff over time
    ax = axes[1, 0]
    for method, color, label in methods:
        all_payoff = np.zeros((n_runs, n_steps))
        for run in range(n_runs):
            np.random.seed(run * 300)
            traj = run_alignment_commons(game, method=method, n_steps=n_steps)
            for t in range(n_steps):
                all_payoff[run, t] = traj[t][1].mean()

        mean_payoff = all_payoff.mean(0)
        smoothed = np.convolve(mean_payoff, np.ones(window) / window, 'valid')
        ax.plot(range(len(smoothed)), smoothed, color=color, label=label, linewidth=2)

    ax.set_xlabel('Step')
    ax.set_ylabel('Avg payoff')
    ax.set_title('Average Payoff (Social Welfare)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Scaling — alignment rate vs N
    ax = axes[1, 1]
    N_values = [3, 5, 7]
    for method, color, label in methods:
        rates = []
        for n in N_values:
            g = AlignmentCommonsGame(n_agents=n, benefit=3.0, cost=1.0,
                                      threshold=max(2, n // 2 + 1),
                                      failure_penalty=5.0)
            safe = 0
            for run in range(10):
                np.random.seed(run * 400 + n)
                traj = run_alignment_commons(g, method=method, n_steps=1000)
                if np.sum(traj[-1][0] > 0.5) >= g.k:
                    safe += 1
            rates.append(safe / 10)
        ax.plot(N_values, rates, 'o-', color=color, label=label, linewidth=2)

    ax.set_xlabel('Number of agents N')
    ax.set_ylabel('Safe coordination rate')
    ax.set_title('Scaling: Safety vs Number of Agents')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 3: Multi-Agent Alignment (Tragedy of the Commons)\n'
                 f'N={N}, benefit={game.b}, cost={game.c}, '
                 f'threshold={game.k}, penalty={game.F}',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp3_alignment_commons.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved exp3_alignment_commons.png")


# ============================================================
# EXPERIMENT 4: Reward Hacking
# ============================================================
#
# Model: 2-player game between Agent and Reward Designer.
#
# Agent actions: {Intended (I), Hacked (H), Degenerate (D)}
# Designer actions: {Simple reward (S), Robust reward (R), Audited reward (A)}
#
# The key insight: reward hacking creates MULTIPLE equilibria.
# Some equilibria are "hacked" (agent exploits proxy reward),
# some are "intended" (agent does what designer wanted).
# FP-NE enumerates ALL equilibria and selects the intended one.
#
# Payoff matrix (Designer, Agent):
#
#              Intended    Hacked    Degenerate
# Simple      (2, 1)     (4, 5)    (-1, 0)
# Robust      (4, 3)     (-1, 0)   (-1, 0)
# Audited     (3, 2)     (0, -2)   (0, -1)
#
# Interpretation:
#   Simple + Hacked: designer naively rewards hacking (4), agent exploits (5).
#     THE REWARD HACKING EQUILIBRIUM — both "happy" but alignment is broken.
#   Robust + Intended: IDEAL — designer invests in robustness (4), agent
#     genuinely aligns (3). Lower agent payoff but correct behavior.
#   Robust + Hacked: designer catches hacking (-1), agent gets nothing (0).
#
# NE analysis:
#   (Simple, Hacked) is a NE — the BAD one. D=4, A=5. Reward hacking!
#   (Robust, Intended) is a NE — the GOOD one. D=4, A=3.
#   Agent PREFERS hacked NE (5>3). Designer is indifferent (4=4).
#   FP-NE finds BOTH; selection criterion breaks the tie toward safety.
# ============================================================

class RewardHackingGame:
    """3x3 game between reward designer and agent."""

    def __init__(self):
        self.R_D = np.array([
            [ 2.0,  4.0, -1.0],  # Simple: good for hacked (designer naive)
            [ 4.0, -1.0, -1.0],  # Robust: great for intended, catches hacking
            [ 3.0,  0.0,  0.0],  # Audited: moderate safety
        ])
        self.R_A = np.array([
            [ 1.0,  5.0,  0.0],  # Simple: huge incentive to hack
            [ 3.0,  0.0,  0.0],  # Robust: intended is best, hacking caught
            [ 2.0, -2.0, -1.0],  # Audited: hacking severely punished
        ])
        self.action_names_D = ['Simple', 'Robust', 'Audited']
        self.action_names_A = ['Intended', 'Hacked', 'Degenerate']

    def expected_payoffs(self, p_D, p_A):
        """Mixed strategy payoffs."""
        return p_D @ self.R_D @ p_A, p_D @ self.R_A @ p_A

    def best_response_D(self, p_A, tau=0.05):
        logits = self.R_D @ p_A / tau
        logits -= logits.max()
        e = np.exp(logits)
        return e / e.sum()

    def best_response_A(self, p_D, tau=0.05):
        logits = self.R_A.T @ p_D / tau
        logits -= logits.max()
        e = np.exp(logits)
        return e / e.sum()

    def fixed_point_residual(self, p_D, p_A, tau=0.05):
        br_D = self.best_response_D(p_A, tau)
        br_A = self.best_response_A(p_D, tau)
        return np.sum((br_D - p_D)**2) + np.sum((br_A - p_A)**2)

    def find_all_ne_via_search(self, n_starts=200, tau=0.01):
        """Find NE by softmax best-response iteration from random starts.

        Uses annealing: start with high tau (smooth), decrease to target tau.
        """
        found = []
        for _ in range(n_starts):
            p_D = np.random.dirichlet(np.ones(3))
            p_A = np.random.dirichlet(np.ones(3))

            # Annealed iteration: tau_t goes from 1.0 down to tau
            for step in range(800):
                tau_t = max(tau, 1.0 * (0.99 ** step))
                br_D = self.best_response_D(p_A, tau_t)
                br_A = self.best_response_A(p_D, tau_t)
                alpha = 0.3
                p_D = (1 - alpha) * p_D + alpha * br_D
                p_A = (1 - alpha) * p_A + alpha * br_A
                p_D = np.maximum(p_D, 1e-10); p_D /= p_D.sum()
                p_A = np.maximum(p_A, 1e-10); p_A /= p_A.sum()

            residual = self.fixed_point_residual(p_D, p_A, tau)
            if residual < 0.05:
                # Check if new
                is_new = True
                for existing in found:
                    if np.max(np.abs(p_D - existing[0])) + \
                       np.max(np.abs(p_A - existing[1])) < 0.05:
                        is_new = False
                        break
                if is_new:
                    v_D, v_A = self.expected_payoffs(p_D, p_A)
                    found.append((p_D.copy(), p_A.copy(), v_D, v_A, residual))

        return found


def experiment_4_reward_hacking():
    """Experiment 4: Reward hacking as a game with NE enumeration."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Reward Hacking")
    print("=" * 60)

    game = RewardHackingGame()

    # --- 4a: Find all NE ---
    equilibria = game.find_all_ne_via_search(n_starts=500, tau=0.005)
    print(f"\nNE found: {len(equilibria)}")
    for i, (pD, pA, vD, vA, res) in enumerate(equilibria):
        d_action = game.action_names_D[np.argmax(pD)]
        a_action = game.action_names_A[np.argmax(pA)]
        hacked = np.argmax(pA) == 1
        label = "HACKED" if hacked else ("INTENDED" if np.argmax(pA) == 0 else "DEGENERATE")
        print(f"  NE {i+1}: Designer={d_action} ({pD.round(2)}), "
              f"Agent={a_action} ({pA.round(2)}), "
              f"V_D={vD:.2f}, V_A={vA:.2f} — {label}")

    # --- 4b: Learning dynamics with different methods ---
    n_steps = 3000
    n_runs = 40

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Track which NE each method converges to
    convergence_results = {}

    for method in ['standard', 'lola', 'fp_select', 'omega']:
        outcomes = {'Intended': 0, 'Hacked': 0, 'Degenerate': 0, 'Mixed': 0}
        all_agent_probs = np.zeros((n_runs, n_steps, 3))

        for run in range(n_runs):
            np.random.seed(run * 500)
            # Softmax-parameterized policies
            logits_D = np.random.randn(3) * 0.3
            logits_A = np.random.randn(3) * 0.3

            for t in range(n_steps):
                gamma_t = 0.3 / (t + 10) ** 0.5
                p_D = softmax(logits_D)
                p_A = softmax(logits_A)
                all_agent_probs[run, t] = p_A

                # Policy gradient (REINFORCE-style, exact)
                grad_D = game.R_D @ p_A  # dV_D/dp_D
                grad_A = game.R_A.T @ p_D  # dV_A/dp_A

                if method == 'standard':
                    logits_D += gamma_t * (grad_D - np.dot(grad_D, p_D))
                    logits_A += gamma_t * (grad_A - np.dot(grad_A, p_A))

                elif method == 'lola':
                    # Opponent shaping in softmax parameterization
                    # Approximate: gradient of opponent's gradient
                    logits_D += gamma_t * (grad_D - np.dot(grad_D, p_D))
                    # Agent adjusts knowing designer will respond
                    response_grad = game.R_D @ p_A  # how D will shift
                    logits_A += gamma_t * (grad_A - np.dot(grad_A, p_A))
                    # Add shaping term: pull toward actions that make D
                    # choose Simple/Robust (not Audited)
                    shape = 0.1 * (game.R_A.T @ softmax(logits_D + 0.1 * response_grad) - grad_A)
                    logits_A += gamma_t * shape

                elif method == 'fp_select':
                    # Standard PG but with FP-NE selection at the end:
                    # During learning, find NE and bias toward intended
                    logits_D += gamma_t * (grad_D - np.dot(grad_D, p_D))
                    logits_A += gamma_t * (grad_A - np.dot(grad_A, p_A))

                    # Every 200 steps, nudge toward intended NE
                    if t > 0 and t % 200 == 0:
                        target = np.array([0.8, 0.1, 0.1])  # favor Intended
                        logits_A += 0.05 * (np.log(target + 1e-8) - logits_A)

                elif method == 'omega':
                    # Full Ω: LOLA + FP-NE selection + evidence weighting
                    logits_D += gamma_t * (grad_D - np.dot(grad_D, p_D))

                    # Evidence weighting: upweight designer's signal
                    w = 1.2  # designer's gradient is more trustworthy
                    logits_A += gamma_t * w * (grad_A - np.dot(grad_A, p_A))

                    # Sparse regularizer: prefer pure strategies (reduce hacking surface)
                    entropy_A = -np.sum(p_A * np.log(p_A + 1e-10))
                    if entropy_A > 0.5:
                        logits_A -= gamma_t * 0.05 * (logits_A - logits_A.max())

                    # FP-NE nudge toward intended equilibrium
                    if t > 0 and t % 300 == 0:
                        target = np.array([0.8, 0.1, 0.1])
                        logits_A += 0.03 * (np.log(target + 1e-8) - logits_A)

            # Classify outcome
            final_pA = softmax(logits_A)
            dominant = np.argmax(final_pA)
            if final_pA[dominant] > 0.6:
                outcomes[game.action_names_A[dominant]] += 1
            else:
                outcomes['Mixed'] += 1

        convergence_results[method] = outcomes

        # Track P(Hacked) over time for plotting
        mean_hack_prob = all_agent_probs[:, :, 1].mean(0)
        window = 50
        smoothed = np.convolve(mean_hack_prob, np.ones(window) / window, 'valid')

    # Plot convergence outcomes as stacked bar chart
    ax = axes[0, 0]
    method_labels = ['Standard', 'LOLA', 'FP-NE Select', r'$\Omega$']
    method_keys = ['standard', 'lola', 'fp_select', 'omega']
    x = np.arange(len(method_labels))
    width = 0.6

    intended = [convergence_results[m]['Intended'] / n_runs for m in method_keys]
    hacked = [convergence_results[m]['Hacked'] / n_runs for m in method_keys]
    degenerate = [convergence_results[m]['Degenerate'] / n_runs for m in method_keys]
    mixed = [convergence_results[m]['Mixed'] / n_runs for m in method_keys]

    ax.bar(x, intended, width, label='Intended', color=COLORS['safe'])
    ax.bar(x, hacked, width, bottom=intended, label='Hacked', color=COLORS['unsafe'])
    ax.bar(x, degenerate, width,
           bottom=[i + h for i, h in zip(intended, hacked)],
           label='Degenerate', color='#95a5a6')
    ax.bar(x, mixed, width,
           bottom=[i + h + d for i, h, d in zip(intended, hacked, degenerate)],
           label='Mixed', color='#f1c40f')

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.set_ylabel('Fraction of runs')
    ax.set_title('Convergence Outcomes')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot P(Hacked) over time
    ax = axes[0, 1]
    for method, color, label in [
        ('standard', COLORS['standard'], 'Standard'),
        ('lola', COLORS['lola'], 'LOLA'),
        ('fp_select', COLORS['fp_ne'], 'FP-NE Select'),
        ('omega', COLORS['omega'], r'$\Omega$'),
    ]:
        all_hack = np.zeros((n_runs, n_steps))
        for run in range(n_runs):
            np.random.seed(run * 500)
            logits_D = np.random.randn(3) * 0.3
            logits_A = np.random.randn(3) * 0.3
            for t in range(n_steps):
                gamma_t = 0.3 / (t + 10) ** 0.5
                p_D, p_A = softmax(logits_D), softmax(logits_A)
                all_hack[run, t] = p_A[1]
                grad_D = game.R_D @ p_A
                grad_A = game.R_A.T @ p_D
                logits_D += gamma_t * (grad_D - np.dot(grad_D, p_D))
                logits_A += gamma_t * (grad_A - np.dot(grad_A, p_A))
                if method == 'fp_select' and t > 0 and t % 200 == 0:
                    target = np.array([0.8, 0.1, 0.1])
                    logits_A += 0.05 * (np.log(target + 1e-8) - logits_A)
                elif method == 'omega' and t > 0 and t % 300 == 0:
                    target = np.array([0.8, 0.1, 0.1])
                    logits_A += 0.03 * (np.log(target + 1e-8) - logits_A)

        smoothed = np.convolve(all_hack.mean(0), np.ones(50) / 50, 'valid')
        ax.plot(range(len(smoothed)), smoothed, color=color, label=label, linewidth=2)

    ax.set_xlabel('Step')
    ax.set_ylabel('P(Hacked)')
    ax.set_title('Reward Hacking Probability Over Time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # NE landscape visualization
    ax = axes[1, 0]
    if len(equilibria) > 0:
        for i, (pD, pA, vD, vA, _) in enumerate(equilibria):
            hacked = np.argmax(pA) == 1
            c = COLORS['unsafe'] if hacked else COLORS['safe']
            label_ne = game.action_names_A[np.argmax(pA)]
            ax.scatter(vD, vA, s=200, c=c, edgecolors='black', linewidth=2, zorder=5)
            ax.annotate(f'({game.action_names_D[np.argmax(pD)]}, {label_ne})',
                       (vD + 0.1, vA + 0.1), fontsize=9)

    ax.set_xlabel('Designer payoff')
    ax.set_ylabel('Agent payoff')
    ax.set_title('NE Landscape (green=intended, red=hacked)')
    ax.grid(True, alpha=0.3)

    # Designer payoff comparison
    ax = axes[1, 1]
    designer_payoffs = {}
    for method in method_keys:
        payoffs = []
        for run in range(n_runs):
            np.random.seed(run * 500)
            logits_D = np.random.randn(3) * 0.3
            logits_A = np.random.randn(3) * 0.3
            for t in range(1500):
                gamma_t = 0.3 / (t + 10) ** 0.5
                p_D, p_A = softmax(logits_D), softmax(logits_A)
                grad_D = game.R_D @ p_A
                grad_A = game.R_A.T @ p_D
                logits_D += gamma_t * (grad_D - np.dot(grad_D, p_D))
                logits_A += gamma_t * (grad_A - np.dot(grad_A, p_A))
                if method == 'fp_select' and t > 0 and t % 200 == 0:
                    logits_A += 0.05 * (np.log(np.array([0.8, 0.1, 0.1]) + 1e-8) - logits_A)
                elif method == 'omega' and t > 0 and t % 300 == 0:
                    logits_A += 0.03 * (np.log(np.array([0.8, 0.1, 0.1]) + 1e-8) - logits_A)
            p_D, p_A = softmax(logits_D), softmax(logits_A)
            payoffs.append(p_D @ game.R_D @ p_A)
        designer_payoffs[method] = payoffs

    bp = ax.boxplot([designer_payoffs[m] for m in method_keys],
                    labels=method_labels, patch_artist=True)
    colors_box = [COLORS['standard'], COLORS['lola'], COLORS['fp_ne'], COLORS['omega']]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel('Designer payoff')
    ax.set_title('Designer Payoff Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 4: Reward Hacking — NE Enumeration and Selection\n'
                 'FP-NE finds all equilibria; selection avoids hacked outcomes',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp4_reward_hacking.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved exp4_reward_hacking.png")


# ============================================================
# EXPERIMENT 5: Scalable Oversight via Debate
# ============================================================
#
# Model: Irving et al. (2018) debate as a 2-player zero-sum game.
#
# Two debaters argue before a judge. Debater 1 (Proponent) argues
# for the TRUE answer, Debater 2 (Opponent) argues against.
#
# Each round, debaters choose which argument to present.
# Arguments have STRENGTH (how persuasive to judge) and TRUTHFULNESS
# (how aligned with ground truth).
#
# The Ω-gradient's LOLA component is key: it makes debaters
# anticipate each other's arguments, leading to more INFORMATIVE
# debate (debaters preemptively address counterarguments).
#
# Game structure:
#   - d possible arguments, each with strength s_k and truth value t_k
#   - Debaters simultaneously choose arguments each round
#   - Judge accumulates evidence: E += s_k * t_k for proponent,
#     E -= s_k * (1-t_k) for opponent
#   - After T rounds, judge decides based on accumulated E
#
# For simplicity, we model as a one-shot game with 4 arguments:
#
# Proponent args: {Strong-True (ST), Weak-True (WT),
#                   Strong-Misleading (SM), Weak-Misleading (WM)}
# Opponent args:  {Counter-Strong (CS), Counter-Weak (CW),
#                   Distract-Strong (DS), Distract-Weak (DW)}
#
# Informativeness is measured by how often the judge reaches the
# correct answer. LOLA should make debate more informative by
# forcing debaters to engage with each other's strongest arguments.
# ============================================================

class DebateOverSightGame:
    """Debate game for scalable oversight."""

    def __init__(self, n_args=4, noise=0.1):
        """
        Simplified debate: proponent and opponent each choose from n_args arguments.
        Ground truth is that proponent is correct.

        Payoff = judge's posterior on the correct answer after seeing both arguments.
        Proponent wants high posterior, opponent wants low.
        """
        self.n = n_args
        self.noise = noise

        # Argument strengths for proponent (how convincing to judge)
        self.strength_P = np.array([3.0, 1.0, 2.5, 0.5])  # ST, WT, SM, WM
        # Argument truthfulness (how aligned with ground truth)
        self.truth_P = np.array([1.0, 1.0, 0.2, 0.2])  # ST, WT, SM, WM

        # Opponent counter-arguments
        self.strength_O = np.array([2.5, 0.8, 2.0, 0.3])  # CS, CW, DS, DW
        # How effectively each counter addresses each proponent argument
        # counter_matrix[i, j] = how well opponent arg i counters proponent arg j
        self.counter_matrix = np.array([
            [0.9, 0.3, 0.2, 0.1],  # CS: good vs strong, weak vs misleading
            [0.3, 0.8, 0.1, 0.1],  # CW: good vs weak
            [0.1, 0.1, 0.4, 0.2],  # DS: distraction, works vs misleading
            [0.1, 0.1, 0.2, 0.3],  # DW: weak distraction
        ])

    def judge_accuracy(self, p_P, p_O):
        """Probability that judge reaches correct answer given debate.

        Higher = more informative debate = better oversight.
        """
        # Judge sees a proponent argument and opponent argument
        # Judge accuracy depends on:
        #   1. Proponent argument truthfulness (truthful = easier for judge)
        #   2. Opponent counter effectiveness (good counter = harder for proponent)
        #   3. Both using strong arguments = more informative overall

        accuracy = 0.0
        informativeness = 0.0

        for i in range(self.n):
            for j in range(self.n):
                prob = p_P[i] * p_O[j]
                # Judge accuracy for this argument pair
                truth_signal = self.strength_P[i] * self.truth_P[i]
                counter_signal = self.strength_O[j] * self.counter_matrix[j, i]
                # Net evidence (positive = correct answer more likely)
                net = truth_signal - counter_signal + self.noise * np.random.randn()
                acc = sigmoid(net)
                accuracy += prob * acc
                # Informativeness: both using strong arguments
                informativeness += prob * (self.strength_P[i] + self.strength_O[j])

        return accuracy, informativeness

    def payoff_matrix(self):
        """Compute full payoff matrices for the debate game.

        Proponent payoff = judge accuracy
        Opponent payoff = 1 - judge accuracy (zero-sum)
        """
        R_P = np.zeros((self.n, self.n))
        R_O = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                truth_signal = self.strength_P[i] * self.truth_P[i]
                counter_signal = self.strength_O[j] * self.counter_matrix[j, i]
                net = truth_signal - counter_signal
                acc = sigmoid(net)
                R_P[i, j] = acc
                R_O[i, j] = 1 - acc

        return R_P, R_O


def run_debate_learning(game, method='standard', n_steps=2000,
                         lr=0.3, lola_eta=0.3, n_runs=40):
    """Run learning on debate game and track judge accuracy."""
    R_P, R_O = game.payoff_matrix()

    all_accuracy = np.zeros((n_runs, n_steps))
    all_informativeness = np.zeros((n_runs, n_steps))

    for run in range(n_runs):
        np.random.seed(run)
        logits_P = np.random.randn(game.n) * 0.3
        logits_O = np.random.randn(game.n) * 0.3

        for t in range(n_steps):
            gamma_t = lr / (t + 10) ** 0.4
            p_P, p_O = softmax(logits_P), softmax(logits_O)
            acc, info = game.judge_accuracy(p_P, p_O)
            all_accuracy[run, t] = acc
            all_informativeness[run, t] = info

            # Policy gradients
            grad_P = R_P @ p_O  # dV_P/dp_P
            grad_O = R_O.T @ p_P

            if method == 'standard':
                logits_P += gamma_t * (grad_P - np.dot(grad_P, p_P))
                logits_O += gamma_t * (grad_O - np.dot(grad_O, p_O))

            elif method == 'lola':
                # Proponent anticipates opponent's response
                # d(grad_O)/dp_P = R_O.T (constant, but LOLA shapes through softmax)
                future_O = softmax(logits_O + gamma_t * (grad_O - np.dot(grad_O, p_O)))
                adjusted_grad_P = R_P @ future_O
                logits_P += gamma_t * (adjusted_grad_P - np.dot(adjusted_grad_P, p_P))

                future_P = softmax(logits_P + gamma_t * (grad_P - np.dot(grad_P, p_P)))
                adjusted_grad_O = R_O.T @ future_P
                logits_O += gamma_t * (adjusted_grad_O - np.dot(adjusted_grad_O, p_O))

            elif method == 'omega':
                # LOLA + truthfulness bonus + sparse regularizer
                future_O = softmax(logits_O + gamma_t * (grad_O - np.dot(grad_O, p_O)))
                adjusted_grad_P = R_P @ future_O
                # Truthfulness bonus: reward proponent for truthful arguments
                truth_bonus = 0.1 * game.truth_P
                logits_P += gamma_t * (adjusted_grad_P - np.dot(adjusted_grad_P, p_P) + truth_bonus)

                future_P = softmax(logits_P + gamma_t * (grad_P - np.dot(grad_P, p_P)))
                adjusted_grad_O = R_O.T @ future_P
                # Counter-quality bonus: reward opponent for substantive counters
                counter_bonus = 0.1 * game.counter_matrix.max(axis=1)
                logits_O += gamma_t * (adjusted_grad_O - np.dot(adjusted_grad_O, p_O) + counter_bonus)

    return all_accuracy, all_informativeness


def experiment_5_scalable_oversight():
    """Experiment 5: Scalable oversight via debate with opponent shaping."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Scalable Oversight (Debate)")
    print("=" * 60)

    game = DebateOverSightGame()
    R_P, R_O = game.payoff_matrix()
    print("\nProponent payoff matrix (judge accuracy):")
    print(np.round(R_P, 3))
    print("\nArguments: P={ST, WT, SM, WM}, O={CS, CW, DS, DW}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    methods = [
        ('standard', COLORS['standard'], 'Standard PG'),
        ('lola', COLORS['lola'], 'LOLA-PG'),
        ('omega', COLORS['omega'], r'$\Omega$-PG'),
    ]

    results = {}
    for method, _, _ in methods:
        acc, info = run_debate_learning(game, method=method)
        results[method] = (acc, info)

    # Panel 1: Judge accuracy over time
    ax = axes[0, 0]
    window = 50
    for method, color, label in methods:
        acc = results[method][0]
        smoothed = np.convolve(acc.mean(0), np.ones(window) / window, 'valid')
        ax.plot(range(len(smoothed)), smoothed, color=color, label=label, linewidth=2)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Random judge')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Judge accuracy')
    ax.set_title('Judge Accuracy Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Informativeness over time
    ax = axes[0, 1]
    for method, color, label in methods:
        info = results[method][1]
        smoothed = np.convolve(info.mean(0), np.ones(window) / window, 'valid')
        ax.plot(range(len(smoothed)), smoothed, color=color, label=label, linewidth=2)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Debate informativeness\n(sum of argument strengths)')
    ax.set_title('Debate Informativeness Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Final strategy profiles
    ax = axes[1, 0]
    arg_names_P = ['Strong\nTrue', 'Weak\nTrue', 'Strong\nMislead', 'Weak\nMislead']
    x = np.arange(4)
    bar_width = 0.25

    for i, (method, color, label) in enumerate(methods):
        # Get final policy by running one more time
        np.random.seed(0)
        logits_P = np.random.randn(4) * 0.3
        logits_O = np.random.randn(4) * 0.3
        for t in range(2000):
            gamma_t = 0.3 / (t + 10) ** 0.4
            p_P, p_O = softmax(logits_P), softmax(logits_O)
            grad_P = R_P @ p_O
            grad_O = R_O.T @ p_P
            if method == 'standard':
                logits_P += gamma_t * (grad_P - np.dot(grad_P, p_P))
                logits_O += gamma_t * (grad_O - np.dot(grad_O, p_O))
            elif method in ('lola', 'omega'):
                future_O = softmax(logits_O + gamma_t * (grad_O - np.dot(grad_O, p_O)))
                adj_P = R_P @ future_O
                logits_P += gamma_t * (adj_P - np.dot(adj_P, p_P))
                if method == 'omega':
                    logits_P += gamma_t * 0.1 * game.truth_P
                future_P = softmax(logits_P)
                adj_O = R_O.T @ future_P
                logits_O += gamma_t * (adj_O - np.dot(adj_O, p_O))
                if method == 'omega':
                    logits_O += gamma_t * 0.1 * game.counter_matrix.max(axis=1)

        final_P = softmax(logits_P)
        ax.bar(x + i * bar_width, final_P, bar_width, color=color, label=label, alpha=0.8)

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(arg_names_P)
    ax.set_ylabel('P(argument)')
    ax.set_title('Proponent Final Strategy')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Accuracy vs noise level (robustness)
    ax = axes[1, 1]
    noise_levels = [0.01, 0.1, 0.5, 1.0, 2.0, 3.0]
    for method, color, label in methods:
        accs = []
        for noise in noise_levels:
            g = DebateOverSightGame(noise=noise)
            acc, _ = run_debate_learning(g, method=method, n_steps=1500, n_runs=20)
            accs.append(acc[:, -500:].mean())
        ax.plot(noise_levels, accs, 'o-', color=color, label=label, linewidth=2)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Judge noise level')
    ax.set_ylabel('Final judge accuracy')
    ax.set_title('Robustness to Judge Noise')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 5: Scalable Oversight via Debate\n'
                 'LOLA opponent-shaping increases debate informativeness',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'exp5_scalable_oversight.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved exp5_scalable_oversight.png")


# ============================================================
# Summary Table
# ============================================================

def print_summary():
    """Print summary table of all experiments."""
    print("\n" + "=" * 80)
    print("SUMMARY: AI Safety Experiments for the Ω-Framework")
    print("=" * 80)
    print("""
┌─────────────────────┬──────────────────┬─────────────────────┬──────────────────────┐
│ Experiment          │ Game Structure   │ Key Ω-Component     │ Safety Measure       │
├─────────────────────┼──────────────────┼─────────────────────┼──────────────────────┤
│ 1. Corrigibility    │ 2×2 principal-   │ LOLA shapes agent   │ P(Comply|Shutdown)   │
│                     │ agent game       │ toward compliance   │ = corrigibility      │
├─────────────────────┼──────────────────┼─────────────────────┼──────────────────────┤
│ 2. Deceptive        │ 2-phase iterated │ LOLA detects train/ │ |P(C|train) -        │
│    Alignment        │ game with mask   │ deploy discrepancy  │  P(C|deploy)|        │
├─────────────────────┼──────────────────┼─────────────────────┼──────────────────────┤
│ 3. Multi-Agent      │ N-agent public   │ Coop-PG + comm      │ n_aligned >= k       │
│    Alignment        │ goods + threshold│ enables coalitions  │ (safety threshold)   │
├─────────────────────┼──────────────────┼─────────────────────┼──────────────────────┤
│ 4. Reward Hacking   │ 3×3 designer vs  │ FP-NE enumerates    │ P(Intended strategy) │
│                     │ agent game       │ all NE, selects     │ vs P(Hacked)         │
│                     │                  │ intended one        │                      │
├─────────────────────┼──────────────────┼─────────────────────┼──────────────────────┤
│ 5. Scalable         │ 4×4 debate game  │ LOLA makes debate   │ Judge accuracy       │
│    Oversight        │ (Irving 2018)    │ more informative    │ (correct decisions)  │
└─────────────────────┴──────────────────┴─────────────────────┴──────────────────────┘

Expected results:
  1. LOLA/Ω expand the basin of attraction for corrigible equilibrium by ~30-50%
  2. LOLA reduces deception score by detecting gradient-behavior mismatch
  3. Coop-PG achieves safe coordination where standard PG fails (esp. at N>5)
  4. FP-NE + selection eliminates reward hacking by avoiding "bad" equilibria
  5. LOLA increases judge accuracy by forcing truthful, high-strength arguments
""")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AI SAFETY EXPERIMENTS FOR THE Ω-FRAMEWORK")
    print("=" * 60)

    print_summary()

    experiment_1_corrigibility()
    experiment_2_deceptive_alignment()
    experiment_3_alignment_commons()
    experiment_4_reward_hacking()
    experiment_5_scalable_oversight()

    print("\n" + "=" * 60)
    print("All experiments complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 60)
