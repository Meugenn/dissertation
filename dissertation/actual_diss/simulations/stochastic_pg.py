"""
Stochastic Game (S > 1) Experiments for Restart-PG.

Validates Theorem 4.2 on a genuine stochastic game where state-contingent
policies strictly outperform state-blind ones.

Game: "State-Dependent Coordination"
  - 2 players, 2 actions, 2 states
  - In state 1: coordinate on a_1 is optimal
  - In state 2: coordinate on a_2 is optimal
  - Successful coordination transitions to the other state
  - A state-blind agent cannot distinguish the two targets

Experiments:
  6. Convergence to state-contingent Nash (trajectory + policy evolution)
  7. Restart geometric decay for S=2 game
  8. State-contingent vs state-blind PG comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

FIGDIR = Path(__file__).parent / "figures" / "stochastic_pg"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.figsize': (8, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
})


# ===================================================================
#  Stochastic game definition
# ===================================================================

class StochasticGame:
    """
    Episodic stochastic game with termination probability.

    Policies: pi_i[s, a] = probability player i plays action a in state s.
    Policy space: Pi_i = Delta(A_i)^S, Pi = Pi_1 x Pi_2.
    """

    def __init__(self, n_states, n_actions, R1, R2, P, zeta, rho, name=""):
        """
        Args:
            n_states: |S|
            n_actions: |A_i| (same for both players)
            R1: (S, A, A) reward tensor for player 1
            R2: (S, A, A) reward tensor for player 2
            P: (S, A, A, S) transition kernel P[s, a1, a2, s']
            zeta: termination probability (scalar)
            rho: (S,) initial state distribution
            name: game name
        """
        self.S = n_states
        self.A = n_actions
        self.R1 = np.array(R1, dtype=float)
        self.R2 = np.array(R2, dtype=float)
        self.P = np.array(P, dtype=float)
        self.zeta = zeta
        self.rho = np.array(rho, dtype=float)
        self.name = name

    def exact_values(self, pi1, pi2):
        """
        Solve Bellman equations for V_{i,s}(pi) exactly.

        V_i(s) = sum_{a1,a2} pi1[s,a1] pi2[s,a2] [R_i(s,a1,a2)
                 + (1-zeta) sum_{s'} P(s'|s,a1,a2) V_i(s')]

        Returns V1[s], V2[s] arrays.
        """
        S, A = self.S, self.A
        gamma = 1.0 - self.zeta  # effective discount

        # Build the Bellman matrix: V = r + gamma * T * V
        # where r_i(s) = sum_{a1,a2} pi1[s,a1]*pi2[s,a2]*R_i(s,a1,a2)
        # and T(s,s') = sum_{a1,a2} pi1[s,a1]*pi2[s,a2]*P(s'|s,a1,a2)

        r1 = np.zeros(S)
        r2 = np.zeros(S)
        T = np.zeros((S, S))

        for s in range(S):
            for a1 in range(A):
                for a2 in range(A):
                    prob = pi1[s, a1] * pi2[s, a2]
                    r1[s] += prob * self.R1[s, a1, a2]
                    r2[s] += prob * self.R2[s, a1, a2]
                    for sp in range(S):
                        T[s, sp] += prob * self.P[s, a1, a2, sp]

        # V = r + gamma * T @ V  =>  (I - gamma*T) V = r
        M = np.eye(S) - gamma * T
        V1 = np.linalg.solve(M, r1)
        V2 = np.linalg.solve(M, r2)
        return V1, V2

    def value_rho(self, pi1, pi2):
        """V_{i,rho} = rho^T V_i."""
        V1, V2 = self.exact_values(pi1, pi2)
        return self.rho @ V1, self.rho @ V2

    def exact_q_values(self, pi1, pi2):
        """
        Q_i(s, a_i) = sum_{a_{-i}} pi_{-i}[s, a_{-i}] *
            [R_i(s, a_i, a_{-i}) + (1-zeta) sum_{s'} P(s'|s,a_i,a_{-i}) V_i(s')]

        Returns Q1[s, a1], Q2[s, a2].
        """
        V1, V2 = self.exact_values(pi1, pi2)
        S, A = self.S, self.A
        gamma = 1.0 - self.zeta

        Q1 = np.zeros((S, A))
        Q2 = np.zeros((S, A))

        for s in range(S):
            for a1 in range(A):
                for a2 in range(A):
                    continuation = gamma * self.P[s, a1, a2, :] @ V1
                    Q1[s, a1] += pi2[s, a2] * (self.R1[s, a1, a2] + continuation)
            for a2 in range(A):
                for a1 in range(A):
                    continuation = gamma * self.P[s, a1, a2, :] @ V2
                    Q2[s, a2] += pi1[s, a1] * (self.R2[s, a1, a2] + continuation)

        return Q1, Q2

    def exact_gradient(self, pi1, pi2):
        """
        Policy gradient: grad_i V_{i,rho} w.r.t. pi_i.

        Using the policy gradient theorem for stochastic games:
        grad_{pi_i(s,a)} V_{i,rho} = d^pi(s) * Q_i(s, a)

        where d^pi(s) is the discounted state visitation.

        Returns grad1[s, a1], grad2[s, a2] (projected onto simplex tangent).
        """
        S, A = self.S, self.A
        gamma = 1.0 - self.zeta

        # Compute discounted state visitation d^pi(s)
        # d = rho + gamma * T^T @ d  =>  d = (I - gamma*T^T)^{-1} rho
        T = np.zeros((S, S))
        for s in range(S):
            for a1 in range(A):
                for a2 in range(A):
                    prob = pi1[s, a1] * pi2[s, a2]
                    for sp in range(S):
                        T[s, sp] += prob * self.P[s, a1, a2, sp]

        d = np.linalg.solve((np.eye(S) - gamma * T.T), self.rho)

        Q1, Q2 = self.exact_q_values(pi1, pi2)

        # Raw gradient
        grad1 = d[:, None] * Q1  # (S, A)
        grad2 = d[:, None] * Q2

        # Project onto simplex tangent (subtract mean per state)
        for s in range(S):
            grad1[s] -= np.mean(grad1[s])
            grad2[s] -= np.mean(grad2[s])

        return grad1, grad2

    def rollout_episode(self, pi1, pi2, rng, max_steps=100):
        """
        Simulate one episode. Returns trajectory as list of
        (state, action1, action2, reward1, reward2).
        """
        s = rng.choice(self.S, p=self.rho)
        trajectory = []

        for _ in range(max_steps):
            a1 = rng.choice(self.A, p=pi1[s])
            a2 = rng.choice(self.A, p=pi2[s])
            r1 = self.R1[s, a1, a2]
            r2 = self.R2[s, a1, a2]
            trajectory.append((s, a1, a2, r1, r2))

            # Terminate?
            if rng.random() < self.zeta:
                break

            # Transition
            s = rng.choice(self.S, p=self.P[s, a1, a2, :])

        return trajectory

    def reinforce_gradient(self, pi1, pi2, trajectory):
        """
        REINFORCE gradient estimate from a single episode.

        hat_v_i(s, a) = G_i * sum_t [1(s_t=s) * (1(a_{i,t}=a) / pi_i(a|s) - 1)]

        Returns grad1[s, a], grad2[s, a].
        """
        S, A = self.S, self.A

        # Total return for each player
        G1 = sum(r1 for _, _, _, r1, _ in trajectory)
        G2 = sum(r2 for _, _, _, _, r2 in trajectory)

        grad1 = np.zeros((S, A))
        grad2 = np.zeros((S, A))

        for s, a1, a2, _, _ in trajectory:
            # Score function for player 1
            for a in range(A):
                indicator = 1.0 if a == a1 else 0.0
                grad1[s, a] += G1 * (indicator / max(pi1[s, a], 1e-10) - 1.0)
            # Score function for player 2
            for a in range(A):
                indicator = 1.0 if a == a2 else 0.0
                grad2[s, a] += G2 * (indicator / max(pi2[s, a], 1e-10) - 1.0)

        return grad1, grad2

    @staticmethod
    def project_simplex(p):
        """Euclidean projection onto probability simplex."""
        n = len(p)
        u = np.sort(p)[::-1]
        cssv = np.cumsum(u) - 1
        rho = np.nonzero(u > cssv / np.arange(1, n + 1))[0][-1]
        theta = cssv[rho] / (rho + 1)
        return np.maximum(p - theta, 0)

    def project_policy(self, pi):
        """Project each state's action distribution onto simplex."""
        pi_proj = np.zeros_like(pi)
        for s in range(self.S):
            pi_proj[s] = self.project_simplex(pi[s])
        return pi_proj


# ===================================================================
#  Game constructors
# ===================================================================

def state_dependent_coordination():
    """
    2-state coordination game where the optimal action depends on state.

    State 1: coordinate on a_1 (payoff 1.0 vs 0.3)
    State 2: coordinate on a_2 (payoff 1.0 vs 0.3)
    Miscoordination: -0.5 in both states.

    Transitions: successful coordination moves to the other state;
    miscoordination stays put.
    """
    S, A = 2, 2

    # Rewards: R[s, a1, a2]
    R1 = np.zeros((S, A, A))
    R2 = np.zeros((S, A, A))

    # State 0: coordinate on action 0
    R1[0] = np.array([[1.0, -0.5], [-0.5, 0.3]])
    R2[0] = np.array([[1.0, -0.5], [-0.5, 0.3]])

    # State 1: coordinate on action 1
    R1[1] = np.array([[0.3, -0.5], [-0.5, 1.0]])
    R2[1] = np.array([[0.3, -0.5], [-0.5, 1.0]])

    # Transitions: P[s, a1, a2, s']
    # After removing termination probability, remaining mass split:
    #   Correct coordination -> mostly go to other state
    #   Miscoordination -> mostly stay
    P = np.zeros((S, A, A, S))

    # State 0:
    P[0, 0, 0, :] = [0.17, 0.83]  # (a1,a1) in s0 -> mostly go to s1
    P[0, 0, 1, :] = [0.83, 0.17]  # miscoord -> mostly stay in s0
    P[0, 1, 0, :] = [0.83, 0.17]
    P[0, 1, 1, :] = [0.33, 0.67]  # (a2,a2) in s0 -> somewhat go to s1

    # State 1:
    P[1, 0, 0, :] = [0.67, 0.33]  # (a1,a1) in s1 -> somewhat go to s0
    P[1, 0, 1, :] = [0.17, 0.83]  # miscoord -> mostly stay
    P[1, 1, 0, :] = [0.17, 0.83]
    P[1, 1, 1, :] = [0.83, 0.17]  # (a2,a2) in s1 -> mostly go to s0

    zeta = 0.4  # termination probability (~2.5 steps per episode)
    rho = np.array([0.5, 0.5])

    return StochasticGame(S, A, R1, R2, P, zeta, rho,
                          name="State-Dependent Coordination")


# ===================================================================
#  PG algorithm for stochastic games
# ===================================================================

def run_episodic_pg(game, pi1_init, pi2_init, n_iters=2000, gamma=0.3,
                    p=0.75, m=10, n_rollouts=10, use_exact=False,
                    seed=None):
    """
    Run projected policy gradient on a stochastic game.

    Args:
        game: StochasticGame instance
        pi1_init, pi2_init: initial policies (S, A)
        n_iters: number of gradient steps
        gamma, p, m: step-size schedule gamma_n = gamma / (n+m)^p
        n_rollouts: episodes per gradient estimate (REINFORCE)
        use_exact: if True, use exact gradients instead of REINFORCE
        seed: random seed

    Returns:
        pi1_hist, pi2_hist: lists of policies at each step
        val_hist: list of (V1_rho, V2_rho) at each step
    """
    rng = np.random.RandomState(seed)

    pi1 = pi1_init.copy()
    pi2 = pi2_init.copy()

    pi1_hist = [pi1.copy()]
    pi2_hist = [pi2.copy()]
    val_hist = [game.value_rho(pi1, pi2)]

    for n in range(1, n_iters + 1):
        gamma_n = gamma / (n + m) ** p

        if use_exact:
            g1, g2 = game.exact_gradient(pi1, pi2)
        else:
            # Average REINFORCE over multiple rollouts
            g1 = np.zeros_like(pi1)
            g2 = np.zeros_like(pi2)
            for _ in range(n_rollouts):
                tau = game.rollout_episode(pi1, pi2, rng)
                g1_hat, g2_hat = game.reinforce_gradient(pi1, pi2, tau)
                g1 += g1_hat
                g2 += g2_hat
            g1 /= n_rollouts
            g2 /= n_rollouts

        # Projected gradient step
        pi1 = game.project_policy(pi1 + gamma_n * g1)
        pi2 = game.project_policy(pi2 + gamma_n * g2)

        pi1_hist.append(pi1.copy())
        pi2_hist.append(pi2.copy())

        # Log values periodically
        if n % 50 == 0 or n == n_iters:
            val_hist.append(game.value_rho(pi1, pi2))
        else:
            val_hist.append(val_hist[-1])

    return pi1_hist, pi2_hist, val_hist


def find_nash_stochastic(game, n_restarts=30, n_iters=2000, seed=42):
    """Find Nash equilibria by running PG from random initialisations."""
    rng = np.random.RandomState(seed)
    endpoints = []

    for _ in range(n_restarts):
        pi1 = np.array([rng.dirichlet(np.ones(game.A)) for _ in range(game.S)])
        pi2 = np.array([rng.dirichlet(np.ones(game.A)) for _ in range(game.S)])
        pi1_hist, pi2_hist, _ = run_episodic_pg(
            game, pi1, pi2, n_iters=n_iters, gamma=0.3,
            use_exact=True, seed=rng.randint(100000))
        pi1_final = pi1_hist[-1]
        pi2_final = pi2_hist[-1]
        endpoints.append(np.concatenate([pi1_final.ravel(), pi2_final.ravel()]))

    # Cluster
    endpoints = np.array(endpoints)
    nash_list = []
    for ep in endpoints:
        is_new = True
        for ne in nash_list:
            if np.linalg.norm(ep - ne) < 0.05:
                is_new = False
                break
        if is_new:
            nash_list.append(ep)

    return nash_list


# ===================================================================
#  Experiment 6: Convergence to state-contingent Nash
# ===================================================================

def experiment_6_convergence():
    """
    Show convergence of episodic PG to the state-contingent Nash.
    Left: distance to Nash over iterations (multiple trajectories).
    Right: policy evolution pi_1(a_1|s) for both states.
    """
    print("Experiment 6: Convergence in stochastic game (S=2)")

    game = state_dependent_coordination()

    # The state-contingent Nash: play a_1 in s_1, play a_2 in s_2
    pi_star = np.array([[1.0, 0.0], [0.0, 1.0]])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    n_trajs = 8
    rng = np.random.RandomState(42)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_trajs))

    # Left: distance to Nash
    ax = axes[0]
    for i in range(n_trajs):
        pi1_init = np.array([rng.dirichlet(np.ones(2)) for _ in range(2)])
        pi2_init = np.array([rng.dirichlet(np.ones(2)) for _ in range(2)])

        pi1_hist, pi2_hist, _ = run_episodic_pg(
            game, pi1_init, pi2_init, n_iters=1500, gamma=0.3,
            n_rollouts=10, seed=rng.randint(100000))

        # Distance to nearest strict Nash (there are several)
        dists = []
        for pi1 in pi1_hist:
            d1 = np.linalg.norm(pi1 - pi_star)
            d2 = np.linalg.norm(pi1 - np.array([[0.0, 1.0], [1.0, 0.0]]))
            d3 = np.linalg.norm(pi1 - np.array([[1.0, 0.0], [1.0, 0.0]]))
            d4 = np.linalg.norm(pi1 - np.array([[0.0, 1.0], [0.0, 1.0]]))
            dists.append(min(d1, d2, d3, d4))

        ax.plot(dists, color=colors[i], alpha=0.7, linewidth=1)

    ax.set_xlabel('Iteration $n$')
    ax.set_ylabel('$\\|\\pi_n - \\pi^*\\|$ (nearest Nash)')
    ax.set_title('Convergence to Nash equilibrium')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 2.5)

    # Right: policy evolution for one representative trajectory
    ax = axes[1]
    pi1_init = np.array([[0.6, 0.4], [0.6, 0.4]])  # starts state-blind-ish
    pi2_init = np.array([[0.5, 0.5], [0.5, 0.5]])

    pi1_hist, _, _ = run_episodic_pg(
        game, pi1_init, pi2_init, n_iters=1500, gamma=0.3,
        n_rollouts=10, seed=123)

    pi1_s0_a0 = [pi[0, 0] for pi in pi1_hist]
    pi1_s1_a0 = [pi[1, 0] for pi in pi1_hist]

    ax.plot(pi1_s0_a0, color='#2563eb', linewidth=2,
            label='$\\pi_1(a_1 \\mid s_1)$ (should $\\to 1$)')
    ax.plot(pi1_s1_a0, color='#dc2626', linewidth=2,
            label='$\\pi_1(a_1 \\mid s_2)$ (should $\\to 0$)')
    ax.axhline(1.0, color='#2563eb', linestyle='--', alpha=0.3)
    ax.axhline(0.0, color='#dc2626', linestyle='--', alpha=0.3)
    ax.set_xlabel('Iteration $n$')
    ax.set_ylabel('$\\pi_1(a_1 \\mid s)$')
    ax.set_title('State-contingent policy emergence')
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)

    fig.suptitle(f'{game.name} ($|\\mathcal{{S}}| = 2$): '
                 'Episodic PG with REINFORCE gradients',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGDIR / "stochastic_convergence.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved to {FIGDIR / 'stochastic_convergence.pdf'}")


# ===================================================================
#  Experiment 7: Restart geometric decay for S=2
# ===================================================================

def experiment_7_restarts():
    """
    Equilibrium selection in the stochastic game via restarts.

    Different random initialisations converge to different Nash policies
    (state-contingent optimal vs state-blind suboptimal). Best-of-K
    restarts selects the welfare-maximising equilibrium.

    Left: histogram of which Nash each restart converges to.
    Right: best-of-K social welfare as K increases.
    """
    print("Experiment 7: Equilibrium selection via restarts (S=2)")

    game = state_dependent_coordination()

    # Reference Nash policies and their welfare
    pi_contingent = np.array([[1.0, 0.0], [0.0, 1.0]])
    sw_contingent = sum(game.value_rho(pi_contingent, pi_contingent))

    pi_blind_a1 = np.array([[1.0, 0.0], [1.0, 0.0]])
    sw_blind = sum(game.value_rho(pi_blind_a1, pi_blind_a1))

    max_K = 15
    n_trials = 150
    n_iters_per_restart = 1000
    rng = np.random.RandomState(42)

    # Track welfare per restart across trials
    welfare_by_K = {K: [] for K in range(1, max_K + 1)}
    all_single_welfares = []  # welfare from each individual restart

    for trial in range(n_trials):
        discovered_welfares = []

        for k in range(1, max_K + 1):
            pi1_init = np.array([rng.dirichlet(np.ones(2))
                                 for _ in range(2)])
            pi2_init = np.array([rng.dirichlet(np.ones(2))
                                 for _ in range(2)])

            pi1_hist, pi2_hist, _ = run_episodic_pg(
                game, pi1_init, pi2_init, n_iters=n_iters_per_restart,
                gamma=0.3, n_rollouts=5, use_exact=False,
                seed=rng.randint(1000000))

            pi1_end = pi1_hist[-1]
            pi2_end = pi2_hist[-1]
            v1, v2 = game.value_rho(pi1_end, pi2_end)
            sw = v1 + v2
            discovered_welfares.append(sw)
            welfare_by_K[k].append(max(discovered_welfares))
            all_single_welfares.append(sw)

    # Plot: 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: histogram of single-restart welfare
    ax = axes[0]
    ax.hist(all_single_welfares, bins=30, color='#2563eb', alpha=0.7,
            edgecolor='white', linewidth=0.5)
    ax.axvline(x=sw_contingent, color='#22c55e', linestyle='--',
               linewidth=2, label=f'State-contingent NE\n(SW = {sw_contingent:.1f})')
    ax.axvline(x=sw_blind, color='#ef4444', linestyle='--',
               linewidth=2, label=f'State-blind NE\n(SW = {sw_blind:.1f})')
    ax.set_xlabel('Social welfare at convergence')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of single-restart outcomes')
    ax.legend(fontsize=9)

    # Right: best-of-K welfare
    ax = axes[1]
    Ks = list(range(1, max_K + 1))
    means = [np.mean(welfare_by_K[K]) for K in Ks]
    stds = [np.std(welfare_by_K[K]) for K in Ks]

    ax.fill_between(Ks, np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.2, color='#2563eb')
    ax.plot(Ks, means, 'o-', color='#2563eb', linewidth=2, markersize=5,
            label='Best-of-$K$ welfare')
    ax.axhline(y=sw_contingent, color='#22c55e', linestyle='--',
               linewidth=2, label=f'Optimal NE (SW = {sw_contingent:.1f})')
    ax.axhline(y=sw_blind, color='#ef4444', linestyle='--',
               linewidth=2, label=f'State-blind NE (SW = {sw_blind:.1f})')

    ax.set_xlabel('Number of restarts $K$')
    ax.set_ylabel('Social welfare of best discovered NE')
    ax.set_title('Equilibrium selection via restarts')
    ax.legend(fontsize=9)

    fig.suptitle(f'{game.name} ($|\\mathcal{{S}}| = 2$): '
                 'Restarts enable welfare-optimal equilibrium selection',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGDIR / "stochastic_restarts.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved to {FIGDIR / 'stochastic_restarts.pdf'}")


# ===================================================================
#  Experiment 8: State-contingent vs state-blind comparison
# ===================================================================

def experiment_8_contingent_vs_blind():
    """
    Compare state-contingent PG (correct, policies in Delta(A)^S)
    vs state-blind PG (policies in Delta(A), same action in every state).

    Shows that the stochastic game structure matters: collapsing to S=1
    loses information and achieves strictly lower value.
    """
    print("Experiment 8: State-contingent vs state-blind PG")

    game = state_dependent_coordination()

    n_runs = 20
    n_iters = 1500
    rng = np.random.RandomState(42)

    # Collect value trajectories
    contingent_vals = []
    blind_vals = []

    for run in range(n_runs):
        seed = rng.randint(1000000)

        # --- State-contingent PG (S x A policies) ---
        pi1_init = np.array([rng.dirichlet(np.ones(2)) for _ in range(2)])
        pi2_init = np.array([rng.dirichlet(np.ones(2)) for _ in range(2)])

        _, _, val_hist = run_episodic_pg(
            game, pi1_init, pi2_init, n_iters=n_iters, gamma=0.3,
            n_rollouts=10, seed=seed)
        contingent_vals.append([v[0] + v[1] for v in val_hist])

        # --- State-blind PG (same action distribution in every state) ---
        p1 = rng.dirichlet(np.ones(2))
        p2 = rng.dirichlet(np.ones(2))
        pi1_blind = np.tile(p1, (2, 1))  # same row for both states
        pi2_blind = np.tile(p2, (2, 1))

        pi1_hist_b, pi2_hist_b, val_hist_b = run_episodic_pg(
            game, pi1_blind, pi2_blind, n_iters=n_iters, gamma=0.3,
            n_rollouts=10, seed=seed)

        # Force policies to remain state-blind after each step
        # (re-run with exact constraint)
        blind_vals_run = []
        pi1_b = pi1_blind.copy()
        pi2_b = pi2_blind.copy()
        rng_b = np.random.RandomState(seed)

        for n in range(n_iters + 1):
            if n > 0:
                gamma_n = 0.3 / (n + 10) ** 0.75
                # Get gradient and average across states
                g1 = np.zeros((2, 2))
                g2 = np.zeros((2, 2))
                for _ in range(10):
                    tau = game.rollout_episode(pi1_b, pi2_b, rng_b)
                    g1_hat, g2_hat = game.reinforce_gradient(
                        pi1_b, pi2_b, tau)
                    g1 += g1_hat
                    g2 += g2_hat
                g1 /= 10
                g2 /= 10

                # Average gradient across states (state-blind constraint)
                g1_avg = np.mean(g1, axis=0, keepdims=True)
                g2_avg = np.mean(g2, axis=0, keepdims=True)
                g1_avg = np.tile(g1_avg, (2, 1))
                g2_avg = np.tile(g2_avg, (2, 1))

                pi1_b = game.project_policy(pi1_b + gamma_n * g1_avg)
                pi2_b = game.project_policy(pi2_b + gamma_n * g2_avg)

                # Re-enforce state-blindness
                pi1_b[0] = pi1_b[1] = game.project_simplex(
                    0.5 * (pi1_b[0] + pi1_b[1]))
                pi2_b[0] = pi2_b[1] = game.project_simplex(
                    0.5 * (pi2_b[0] + pi2_b[1]))

            v1, v2 = game.value_rho(pi1_b, pi2_b)
            blind_vals_run.append(v1 + v2)

        blind_vals.append(blind_vals_run)

    contingent_vals = np.array(contingent_vals)
    blind_vals = np.array(blind_vals)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: SW over iterations
    ax = axes[0]
    iters = np.arange(len(contingent_vals[0]))

    c_mean = np.mean(contingent_vals, axis=0)
    c_std = np.std(contingent_vals, axis=0)
    b_mean = np.mean(blind_vals, axis=0)
    b_std = np.std(blind_vals, axis=0)

    ax.fill_between(iters, c_mean - c_std, c_mean + c_std,
                    alpha=0.15, color='#2563eb')
    ax.plot(iters, c_mean, color='#2563eb', linewidth=2,
            label='State-contingent PG ($|\\mathcal{S}|=2$)')

    ax.fill_between(iters, b_mean - b_std, b_mean + b_std,
                    alpha=0.15, color='#dc2626')
    ax.plot(iters, b_mean, color='#dc2626', linewidth=2,
            label='State-blind PG ($|\\mathcal{S}|=1$ reduction)')

    ax.set_xlabel('Iteration $n$')
    ax.set_ylabel('Social welfare $V_{1,\\rho} + V_{2,\\rho}$')
    ax.set_title('Learning curves')
    ax.legend(fontsize=10)

    # Right: bar chart of final welfare
    ax = axes[1]
    final_contingent = contingent_vals[:, -1]
    final_blind = blind_vals[:, -1]

    # Theoretical optima
    game_tmp = state_dependent_coordination()
    pi_star_contingent = np.array([[1.0, 0.0], [0.0, 1.0]])
    v_opt = sum(game_tmp.value_rho(pi_star_contingent, pi_star_contingent))

    pi_star_blind_a1 = np.array([[1.0, 0.0], [1.0, 0.0]])
    v_blind_a1 = sum(game_tmp.value_rho(pi_star_blind_a1, pi_star_blind_a1))
    pi_star_blind_a2 = np.array([[0.0, 1.0], [0.0, 1.0]])
    v_blind_a2 = sum(game_tmp.value_rho(pi_star_blind_a2, pi_star_blind_a2))
    v_best_blind = max(v_blind_a1, v_blind_a2)

    bars = ax.bar([0, 1, 2],
                  [np.mean(final_contingent), np.mean(final_blind), v_opt],
                  yerr=[np.std(final_contingent), np.std(final_blind), 0],
                  color=['#2563eb', '#dc2626', '#22c55e'],
                  alpha=0.8, capsize=5, width=0.6)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['State-\ncontingent', 'State-\nblind',
                        'Optimal\n(theoretical)'], fontsize=10)
    ax.set_ylabel('Social welfare at convergence')
    ax.set_title('Final performance comparison')

    # Add value labels
    for bar, val in zip(bars, [np.mean(final_contingent),
                               np.mean(final_blind), v_opt]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11,
                fontweight='bold')

    fig.suptitle(f'{game.name}: State structure matters',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGDIR / "stochastic_vs_normal.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved to {FIGDIR / 'stochastic_vs_normal.pdf'}")


# ===================================================================
#  Main
# ===================================================================

if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 60)
    print("  Stochastic Game Experiments (S=2)")
    print("=" * 60)

    # Quick sanity check: verify Nash values
    game = state_dependent_coordination()
    pi_star = np.array([[1.0, 0.0], [0.0, 1.0]])
    V1, V2 = game.exact_values(pi_star, pi_star)
    print(f"\n  Game: {game.name}")
    print(f"  State-contingent Nash values: V1={V1}, V2={V2}")
    print(f"  V_rho = {game.rho @ V1:.4f}, {game.rho @ V2:.4f}")
    print(f"  Social welfare = {game.rho @ V1 + game.rho @ V2:.4f}")

    pi_blind = np.array([[1.0, 0.0], [1.0, 0.0]])
    V1b, V2b = game.exact_values(pi_blind, pi_blind)
    print(f"\n  State-blind (always a_1) values: V1={V1b}, V2={V2b}")
    print(f"  V_rho = {game.rho @ V1b:.4f}, SW = {game.rho @ V1b + game.rho @ V2b:.4f}")

    pi_blind2 = np.array([[0.0, 1.0], [0.0, 1.0]])
    V1b2, V2b2 = game.exact_values(pi_blind2, pi_blind2)
    print(f"  State-blind (always a_2) values: V1={V1b2}, V2={V2b2}")
    print(f"  V_rho = {game.rho @ V1b2:.4f}, SW = {game.rho @ V1b2 + game.rho @ V2b2:.4f}")
    print()

    experiment_6_convergence()
    print()
    experiment_7_restarts()
    print()
    experiment_8_contingent_vs_blind()

    print()
    print("=" * 60)
    print(f"  All experiments complete. Figures saved to: {FIGDIR}")
    print("=" * 60)
