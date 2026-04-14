"""
Restart-PG: Empirical validation of Theorem 4.1
(Global convergence via random restarts)

Tests:
1. Convergence probability vs number of restarts (geometric decay of failure)
2. Basin of attraction visualisation (2D simplex)
3. Equilibrium selection across multiple Nash policies
4. Scaling with game dimension (blessing of dimensionality check)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

FIGDIR = Path(__file__).parent / "figures" / "restart_pg"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.figsize': (8, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
})


# ===================================================================
#  Game definitions
# ===================================================================

class MatrixGame:
    """N-action 2-player normal-form game with softmax policies."""

    def __init__(self, R1, R2, name="Game"):
        self.R1 = np.array(R1, dtype=float)
        self.R2 = np.array(R2, dtype=float)
        self.n1, self.n2 = self.R1.shape
        self.name = name

    def value(self, p1, p2):
        """Expected payoffs for mixed strategies p1, p2."""
        v1 = p1 @ self.R1 @ p2
        v2 = p1 @ self.R2 @ p2
        return v1, v2

    def pg_gradient(self, p1, p2):
        """Policy gradient for both players (direct parameterisation)."""
        # grad_i V_i = R_i @ p_{-i} (projected onto tangent of simplex)
        g1 = self.R1 @ p2
        g1 = g1 - np.mean(g1)  # project onto simplex tangent
        g2 = self.R2.T @ p1
        g2 = g2 - np.mean(g2)
        return g1, g2

    def project_simplex(self, p):
        """Euclidean projection onto the probability simplex."""
        n = len(p)
        u = np.sort(p)[::-1]
        cssv = np.cumsum(u) - 1
        rho = np.nonzero(u > cssv / np.arange(1, n + 1))[0][-1]
        theta = cssv[rho] / (rho + 1)
        return np.maximum(p - theta, 0)


def matching_pennies():
    return MatrixGame([[1, -1], [-1, 1]], [[-1, 1], [1, -1]], "Matching Pennies")

def coordination():
    return MatrixGame([[2, 0], [0, 1]], [[2, 0], [0, 1]], "Coordination")

def stag_hunt():
    return MatrixGame([[4, 0], [3, 2]], [[4, 3], [0, 2]], "Stag Hunt")

def battle_of_sexes():
    return MatrixGame([[3, 0], [0, 2]], [[2, 0], [0, 3]], "Battle of the Sexes")

def prisoners_dilemma():
    return MatrixGame([[3, 0], [5, 1]], [[3, 5], [0, 1]], "Prisoner's Dilemma")


# ===================================================================
#  PG algorithm (single run)
# ===================================================================

def run_pg(game, p1_init, p2_init, n_episodes=2000, gamma=0.5, p=0.75,
           m=10, noise_scale=0.0, seed=None):
    """
    Run policy gradient with step-size gamma_n = gamma / (n + m)^p.
    Returns trajectory and whether it converged to a Nash.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    p1 = p1_init.copy()
    p2 = p2_init.copy()
    hist_p1 = [p1.copy()]
    hist_p2 = [p2.copy()]

    for n in range(1, n_episodes + 1):
        gamma_n = gamma / (n + m) ** p
        g1, g2 = game.pg_gradient(p1, p2)

        # Add REINFORCE-style noise
        if noise_scale > 0:
            g1 = g1 + rng.randn(len(g1)) * noise_scale / np.sqrt(n)
            g2 = g2 + rng.randn(len(g2)) * noise_scale / np.sqrt(n)

        p1 = game.project_simplex(p1 + gamma_n * g1)
        p2 = game.project_simplex(p2 + gamma_n * g2)

        hist_p1.append(p1.copy())
        hist_p2.append(p2.copy())

    return np.array(hist_p1), np.array(hist_p2)


def find_nash_policies(game, n_restarts=50, n_episodes=2000):
    """Find Nash equilibria by running many PG trajectories and clustering endpoints."""
    endpoints = []
    for _ in range(n_restarts):
        p1_init = np.random.dirichlet(np.ones(game.n1))
        p2_init = np.random.dirichlet(np.ones(game.n2))
        h1, h2 = run_pg(game, p1_init, p2_init, n_episodes=n_episodes,
                        gamma=0.3, noise_scale=0.1)
        endpoints.append(np.concatenate([h1[-1], h2[-1]]))

    # Cluster endpoints
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
#  Experiment 1: Geometric convergence of restart probability
# ===================================================================

def experiment_1_geometric_convergence():
    """
    Validate Theorem 4.1: P(K* > k) <= (1 - p(1-delta))^k.
    Run many restart schedules, measure empirical failure probability.
    """
    print("Experiment 1: Geometric convergence of restart failure probability")

    games = [coordination(), stag_hunt(), battle_of_sexes()]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, game in zip(axes, games):
        max_restarts = 20
        n_trials = 200
        n_episodes_per_restart = 800

        # For each trial, run restarts until convergence
        first_success = []
        nash_policies = find_nash_policies(game)
        if not nash_policies:
            ax.set_title(f"{game.name}\n(no stable NE found)")
            continue

        for trial in range(n_trials):
            found = False
            for k in range(1, max_restarts + 1):
                p1_init = np.random.dirichlet(np.ones(game.n1))
                p2_init = np.random.dirichlet(np.ones(game.n2))
                h1, h2 = run_pg(game, p1_init, p2_init,
                                n_episodes=n_episodes_per_restart,
                                gamma=0.3, noise_scale=0.1,
                                seed=trial * 1000 + k)
                endpoint = np.concatenate([h1[-1], h2[-1]])

                # Check if converged to any known Nash
                for ne in nash_policies:
                    if np.linalg.norm(endpoint - ne) < 0.05:
                        first_success.append(k)
                        found = True
                        break
                if found:
                    break
            if not found:
                first_success.append(max_restarts + 1)

        first_success = np.array(first_success)

        # Empirical P(K* > k)
        ks = np.arange(1, max_restarts + 1)
        empirical_fail = np.array([np.mean(first_success > k) for k in ks])

        # Fit geometric parameter
        successes_in_1 = np.mean(first_success == 1)
        p_hat = successes_in_1
        theoretical_fail = (1 - p_hat) ** ks

        ax.semilogy(ks, empirical_fail + 1e-10, 'o-', color='#2563eb',
                    markersize=4, label='Empirical $\\mathbb{P}(K^* > k)$')
        ax.semilogy(ks, theoretical_fail + 1e-10, '--', color='#dc2626',
                    linewidth=2, label=f'Geometric$(\\hat p = {p_hat:.2f})$')
        ax.set_xlabel('Number of restarts $k$')
        ax.set_ylabel('$\\mathbb{P}(K^* > k)$')
        ax.set_title(f'{game.name}')
        ax.legend(fontsize=9)
        ax.set_ylim(1e-4, 1.5)

    fig.suptitle('Theorem 4.1: Geometric decay of restart failure probability',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGDIR / "geometric_convergence.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved to {FIGDIR / 'geometric_convergence.pdf'}")


# ===================================================================
#  Experiment 2: Basin of attraction visualisation
# ===================================================================

def experiment_2_basins():
    """
    Visualise basins of attraction on the 2D simplex (2-action games).
    Color each initialisation by which Nash it converges to.
    """
    print("Experiment 2: Basin of attraction visualisation")

    games = [coordination(), stag_hunt(), battle_of_sexes()]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    resolution = 35
    for ax, game in zip(axes, games):
        nash_policies = find_nash_policies(game)
        n_nash = len(nash_policies)

        # Grid over player 1's strategy (p1[0]) and player 2's strategy (p2[0])
        x = np.linspace(0.01, 0.99, resolution)
        y = np.linspace(0.01, 0.99, resolution)
        basin_map = np.zeros((resolution, resolution))

        for i, p1_0 in enumerate(x):
            for j, p2_0 in enumerate(y):
                p1_init = np.array([p1_0, 1 - p1_0])
                p2_init = np.array([p2_0, 1 - p2_0])
                h1, h2 = run_pg(game, p1_init, p2_init,
                                n_episodes=2000, gamma=0.3,
                                noise_scale=0.0)
                endpoint = np.concatenate([h1[-1], h2[-1]])

                # Classify
                best_dist = np.inf
                best_idx = -1
                for idx, ne in enumerate(nash_policies):
                    d = np.linalg.norm(endpoint - ne)
                    if d < best_dist:
                        best_dist = d
                        best_idx = idx
                basin_map[j, i] = best_idx if best_dist < 0.1 else -1

        colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6']
        cmap = ListedColormap(colors[:max(n_nash, 1)] + ['#d1d5db'])

        im = ax.imshow(basin_map, origin='lower', extent=[0, 1, 0, 1],
                       cmap=cmap, vmin=-1, vmax=n_nash - 1, aspect='equal')

        # Mark Nash equilibria
        for idx, ne in enumerate(nash_policies):
            ax.plot(ne[0], ne[2], '*', color='white', markersize=15,
                    markeredgecolor='black', markeredgewidth=1.5)

        ax.set_xlabel('Player 1: $\\pi_1(a_1)$')
        ax.set_ylabel('Player 2: $\\pi_2(a_1)$')
        ax.set_title(f'{game.name} ({n_nash} NE)')

    fig.suptitle('Basins of attraction under deterministic PG',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGDIR / "basins_of_attraction.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved to {FIGDIR / 'basins_of_attraction.pdf'}")


# ===================================================================
#  Experiment 3: Equilibrium selection via restarts
# ===================================================================

def experiment_3_equilibrium_selection():
    """
    Run K restarts, discover Nash policies, select best by social welfare.
    Compare welfare of restart-selected NE vs random convergence.
    """
    print("Experiment 3: Equilibrium selection via restarts")

    game = stag_hunt()
    n_trials = 100
    max_K = 15
    n_episodes = 1000

    nash_policies = find_nash_policies(game)
    print(f"  Found {len(nash_policies)} Nash policies for {game.name}")

    # Compute welfare of each Nash
    nash_welfares = []
    for ne in nash_policies:
        p1, p2 = ne[:game.n1], ne[game.n1:]
        v1, v2 = game.value(p1, p2)
        nash_welfares.append(v1 + v2)
    best_welfare = max(nash_welfares) if nash_welfares else 0

    # For each K, run K restarts and record best discovered welfare
    welfare_by_K = {K: [] for K in range(1, max_K + 1)}

    for trial in range(n_trials):
        discovered = []
        for k in range(1, max_K + 1):
            p1_init = np.random.dirichlet(np.ones(game.n1))
            p2_init = np.random.dirichlet(np.ones(game.n2))
            h1, h2 = run_pg(game, p1_init, p2_init,
                            n_episodes=n_episodes, gamma=0.3,
                            noise_scale=0.1)
            p1_end, p2_end = h1[-1], h2[-1]
            v1, v2 = game.value(p1_end, p2_end)
            discovered.append(v1 + v2)
            welfare_by_K[k].append(max(discovered))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    Ks = list(range(1, max_K + 1))
    means = [np.mean(welfare_by_K[K]) for K in Ks]
    stds = [np.std(welfare_by_K[K]) for K in Ks]

    ax.fill_between(Ks, np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.2, color='#2563eb')
    ax.plot(Ks, means, 'o-', color='#2563eb', linewidth=2,
            label='Restart-PG (best of $K$)')
    ax.axhline(y=best_welfare, color='#22c55e', linestyle='--',
               linewidth=2, label=f'Best NE (SW = {best_welfare:.1f})')
    if len(nash_welfares) > 1:
        worst_welfare = min(nash_welfares)
        ax.axhline(y=worst_welfare, color='#ef4444', linestyle='--',
                   linewidth=2, label=f'Worst NE (SW = {worst_welfare:.1f})')

    ax.set_xlabel('Number of restarts $K$')
    ax.set_ylabel('Social welfare of best discovered NE')
    ax.set_title(f'{game.name}: Equilibrium selection via restarts')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / "equilibrium_selection.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved to {FIGDIR / 'equilibrium_selection.pdf'}")


# ===================================================================
#  Experiment 4: Scaling with dimension (blessing check)
# ===================================================================

def experiment_4_dimension_scaling():
    """
    Test restart complexity as game dimension grows.
    Generate random d-action games, measure restarts needed.
    Check if scaling is sub-linear (blessing) or linear/exponential (curse).
    """
    print("Experiment 4: Restart complexity vs game dimension")

    dims = [2, 3, 4, 5, 7]
    n_games = 15
    n_episodes = 1500
    max_restarts = 50

    median_restarts = []
    q25_restarts = []
    q75_restarts = []

    for d in dims:
        restarts_for_dim = []
        for _ in range(n_games):
            # Random game with sparse Nash (diagonal-dominant payoffs)
            R1 = np.random.randn(d, d) * 0.3
            R2 = np.random.randn(d, d) * 0.3
            # Add diagonal dominance to ensure pure Nash exists
            R1 += np.eye(d) * 2
            R2 += np.eye(d) * 2
            game = MatrixGame(R1, R2, f"Random-{d}")

            # Find a Nash by long run
            ne_candidates = []
            for trial in range(20):
                p1_init = np.random.dirichlet(np.ones(d))
                p2_init = np.random.dirichlet(np.ones(d))
                h1, h2 = run_pg(game, p1_init, p2_init,
                                n_episodes=n_episodes, gamma=0.2)
                ne_candidates.append(np.concatenate([h1[-1], h2[-1]]))

            # Cluster to find distinct Nash
            nash_list = []
            for ep in ne_candidates:
                is_new = True
                for ne in nash_list:
                    if np.linalg.norm(ep - ne) < 0.1:
                        is_new = False
                        break
                if is_new:
                    nash_list.append(ep)

            if not nash_list:
                continue

            # Now measure restarts needed to find ANY Nash
            for attempt in range(5):
                for k in range(1, max_restarts + 1):
                    p1_init = np.random.dirichlet(np.ones(d))
                    p2_init = np.random.dirichlet(np.ones(d))
                    h1, h2 = run_pg(game, p1_init, p2_init,
                                    n_episodes=n_episodes, gamma=0.2,
                                    noise_scale=0.1)
                    endpoint = np.concatenate([h1[-1], h2[-1]])
                    found = any(np.linalg.norm(endpoint - ne) < 0.1
                                for ne in nash_list)
                    if found:
                        restarts_for_dim.append(k)
                        break
                else:
                    restarts_for_dim.append(max_restarts)

        if restarts_for_dim:
            median_restarts.append(np.median(restarts_for_dim))
            q25_restarts.append(np.percentile(restarts_for_dim, 25))
            q75_restarts.append(np.percentile(restarts_for_dim, 75))
        else:
            median_restarts.append(np.nan)
            q25_restarts.append(np.nan)
            q75_restarts.append(np.nan)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(dims, q25_restarts, q75_restarts,
                    alpha=0.2, color='#2563eb')
    ax.plot(dims, median_restarts, 'o-', color='#2563eb', linewidth=2,
            markersize=8, label='Median restarts to find NE')

    # Fit log scaling
    valid = ~np.isnan(median_restarts)
    if np.sum(valid) >= 2:
        from numpy.polynomial import polynomial as P
        log_dims = np.log(np.array(dims)[valid])
        log_restarts = np.log(np.array(median_restarts)[valid] + 1)
        coeffs = np.polyfit(log_dims, log_restarts, 1)
        fit_label = f'Power fit: $K^* \\propto d^{{{coeffs[0]:.2f}}}$'
        d_fit = np.linspace(min(dims), max(dims), 100)
        ax.plot(d_fit, np.exp(np.polyval(coeffs, np.log(d_fit))),
                '--', color='#dc2626', linewidth=2, label=fit_label)

    ax.set_xlabel('Action space dimension $d = |\\mathcal{A}_i|$')
    ax.set_ylabel('Restarts needed $K^*$')
    ax.set_title('Restart complexity vs game dimension\n'
                 '(diagonal-dominant games with sparse Nash)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / "dimension_scaling.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved to {FIGDIR / 'dimension_scaling.pdf'}")


# ===================================================================
#  Experiment 5: Noisy PG (REINFORCE) restarts
# ===================================================================

def experiment_5_stochastic_restarts():
    """
    Same as Exp 1 but with stochastic gradients (nonzero noise).
    Validates that the theorem holds with REINFORCE-style noise.
    """
    print("Experiment 5: Stochastic PG restarts")

    game = coordination()
    noise_levels = [0.0, 0.5, 1.0, 2.0]
    max_restarts = 15
    n_trials = 150
    n_episodes = 1000

    nash_policies = find_nash_policies(game)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2563eb', '#7c3aed', '#dc2626', '#ea580c']

    for noise, color in zip(noise_levels, colors):
        first_success = []
        for trial in range(n_trials):
            for k in range(1, max_restarts + 1):
                p1_init = np.random.dirichlet(np.ones(game.n1))
                p2_init = np.random.dirichlet(np.ones(game.n2))
                h1, h2 = run_pg(game, p1_init, p2_init,
                                n_episodes=n_episodes, gamma=0.3,
                                noise_scale=noise)
                endpoint = np.concatenate([h1[-1], h2[-1]])
                found = any(np.linalg.norm(endpoint - ne) < 0.08
                            for ne in nash_policies)
                if found:
                    first_success.append(k)
                    break
            else:
                first_success.append(max_restarts + 1)

        first_success = np.array(first_success)
        ks = np.arange(1, max_restarts + 1)
        fail_prob = [np.mean(first_success > k) for k in ks]
        label = f'$\\sigma = {noise}$' if noise > 0 else 'Exact gradients'
        ax.semilogy(ks, np.array(fail_prob) + 1e-10, 'o-', color=color,
                    markersize=4, label=label)

    ax.set_xlabel('Number of restarts $k$')
    ax.set_ylabel('$\\mathbb{P}(K^* > k)$')
    ax.set_title(f'{game.name}: Restart failure under varying noise levels')
    ax.legend()
    ax.set_ylim(1e-4, 1.5)
    plt.tight_layout()
    plt.savefig(FIGDIR / "stochastic_restarts.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved to {FIGDIR / 'stochastic_restarts.pdf'}")


# ===================================================================
#  Main
# ===================================================================

if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 60)
    print("  Restart-PG: Empirical Validation of Theorem 4.1")
    print("=" * 60)
    print()

    experiment_1_geometric_convergence()
    print()
    experiment_2_basins()
    print()
    experiment_3_equilibrium_selection()
    print()
    experiment_4_dimension_scaling()
    print()
    experiment_5_stochastic_restarts()

    print()
    print("=" * 60)
    print("  All experiments complete. Figures saved to:")
    print(f"  {FIGDIR}")
    print("=" * 60)
