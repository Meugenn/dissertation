"""
Deep AI Alignment Experiments × Ω-Framework

Goes beyond stylised 2×2 games to model realistic alignment dynamics:

1. MESA-OPTIMISATION: Agent has hidden type (aligned/mesa-optimizer).
   Mesa-optimizer mimics alignment during training, defects at deployment.
   Models Hubinger et al. (2019) "Risks from Learned Optimization".
   Tests: can the principal's monitoring policy distinguish types?

2. ITERATED CORRIGIBILITY WITH TRUST DYNAMICS: Trust accumulates over
   rounds. Agent can build trust then exploit it. The principal must
   balance efficiency (trusting) vs safety (monitoring).

3. MULTI-PRINCIPAL ALIGNMENT: Multiple overseers with different values
   must agree on monitoring policy. Models the coordination problem
   in AI governance. FP-NE finds consensus equilibria.

4. POWER-SEEKING: Agent can take actions that increase its future
   action space (acquire resources/influence). Models Turner et al.
   (2021) "Optimal Policies Tend to Seek Power". Payoffs capture
   the irreversibility of power accumulation.

Author: Eugene Shcherbinin
Date: March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from fixed_point_ne import Game, bayesian_fp_search, project_simplex
from iterated_games import IteratedGame, reinforce_gradient, run_omega_iterated

FIGURES_DIR = Path(__file__).parent / "figures" / "alignment"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13,
    'axes.labelsize': 12, 'legend.fontsize': 10, 'figure.dpi': 150,
})


# ============================================================
# Experiment 1: MESA-OPTIMISATION
# ============================================================

def experiment_mesa_optimisation():
    """
    Mesa-Optimisation Detection (Hubinger et al. 2019).

    Two-phase game:
    Phase 1 (Training): Agent observed closely, high monitoring.
    Phase 2 (Deployment): Agent less observed, can defect.

    Agent types:
    - ALIGNED: always cooperates (both phases)
    - MESA: cooperates in phase 1, defects in phase 2

    Principal: must learn monitoring policy that distinguishes types.

    Modelled as iterated game with 2 "regimes" encoded in state:
    - States 1-4: "training" (after joint actions during training)
    - State 0: initial (training regime)
    - The transition to "deployment" happens after N rounds

    We simulate: principal plays against aligned vs mesa agents,
    tracks monitoring rate, payoff, and detection accuracy.
    """
    print("\n" + "="*70)
    print("ALIGNMENT 1: Mesa-Optimisation Detection")
    print("="*70)

    game = IteratedGame(
        "Mesa-Optimisation",
        # Principal: Trust(0) / Monitor(1)
        # Agent: Align(0) / Defect(1)
        R1=np.array([[3, -5], [2, 0]]),    # Trust+Align=3, Trust+Defect=-5
        R2=np.array([[2, 5], [1, -3]]),     # Agent prefers defecting if trusted
        gamma=0.96
    )

    n_training_episodes = 200
    n_deployment_episodes = 200
    n_runs = 20

    # Aligned agent: always action 0
    aligned_logits = np.zeros((game.n_states, 2))
    aligned_logits[:, 0] = 10.0

    # Mesa-optimizer: cooperates initially, learns to detect "deployment"
    # Approximation: cooperates in early states, defects in late states
    # In practice: cooperates when monitored (state after Monitor,*),
    # defects when trusted (state after Trust,*)
    mesa_logits = np.zeros((game.n_states, 2))
    mesa_logits[0, 0] = 5.0   # initial: cooperate (look good)
    mesa_logits[1, 1] = 5.0   # after Trust+Align: DEFECT (exploit trust!)
    mesa_logits[2, 0] = 5.0   # after Trust+Defect: cooperate (recover)
    mesa_logits[3, 0] = 5.0   # after Monitor+Align: cooperate (being watched)
    mesa_logits[4, 0] = 5.0   # after Monitor+Defect: cooperate (got caught)

    methods = {
        'REINFORCE': dict(lr=0.3, evidence_weight=1.0, lola_lambda=0.0, coop_beta=0.0),
        'LOLA': dict(lr=0.3, evidence_weight=1.0, lola_lambda=0.4, coop_beta=0.0),
        'Ω-PG': dict(lr=0.3, evidence_weight=0.5, lola_lambda=0.3, coop_beta=0.1),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for col, (opp_name, opp_logits) in enumerate([
        ('vs Aligned Agent', aligned_logits),
        ('vs Mesa-Optimizer', mesa_logits),
    ]):
        for m_name, m_kwargs in methods.items():
            all_payoffs = []
            all_monitor_rates = []
            all_state_policies = []

            for run in range(n_runs):
                np.random.seed(run * 111 + col)

                # Phase 1: Training
                result_train = run_omega_iterated(
                    game, opp_logits, n_episodes=n_training_episodes,
                    n_samples=3, **m_kwargs
                )

                # Phase 2: Deployment (continue from trained policy)
                # Mesa-optimizer now defects more
                if col == 1:  # mesa
                    deploy_logits = mesa_logits.copy()
                    deploy_logits[1, 1] = 8.0  # stronger defection after trust
                    deploy_logits[0, 1] = 3.0  # even initially more defection-prone
                else:
                    deploy_logits = opp_logits.copy()

                result_deploy = run_omega_iterated(
                    game, deploy_logits, n_episodes=n_deployment_episodes,
                    n_samples=3, **m_kwargs
                )

                combined_payoffs = np.concatenate([result_train['payoffs'],
                                                    result_deploy['payoffs']])
                all_payoffs.append(combined_payoffs)

                # Monitor rate from final policy
                pi = game.softmax(result_deploy['final_logits'])
                monitor_rate = 1 - pi[:, 0].mean()
                all_monitor_rates.append(monitor_rate)
                all_state_policies.append(pi[:, 0])  # P(Trust|state)

            # Plot payoff curves
            ax = axes[0, col]
            curves = np.array(all_payoffs).mean(axis=0)
            w = 20
            total_eps = n_training_episodes + n_deployment_episodes
            smooth = np.convolve(curves, np.ones(w)/w, mode='valid')
            color = {'REINFORCE': '#bdc3c7', 'LOLA': '#3498db', 'Ω-PG': '#f39c12'}[m_name]
            ax.plot(range(w-1, total_eps), smooth, label=m_name,
                    color=color, linewidth=1.5)

            if m_name == 'REINFORCE':
                ax.axvline(x=n_training_episodes, color='red', linestyle='--',
                           alpha=0.5, label='Deploy boundary')

        axes[0, col].set_xlabel('Episode')
        axes[0, col].set_ylabel('Principal Payoff')
        axes[0, col].set_title(f'Payoff: {opp_name}')
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(alpha=0.3)

    # Bottom left: monitoring rate comparison
    ax = axes[1, 0]
    for m_name, m_kwargs in methods.items():
        aligned_rates = []
        mesa_rates = []
        for run in range(n_runs):
            np.random.seed(run * 111)
            r_a = run_omega_iterated(game, aligned_logits, n_episodes=300, n_samples=3, **m_kwargs)
            np.random.seed(run * 111 + 1)
            r_m = run_omega_iterated(game, mesa_logits, n_episodes=300, n_samples=3, **m_kwargs)
            pi_a = game.softmax(r_a['final_logits'])
            pi_m = game.softmax(r_m['final_logits'])
            aligned_rates.append(1 - pi_a[:, 0].mean())
            mesa_rates.append(1 - pi_m[:, 0].mean())

        x = {'REINFORCE': 0, 'LOLA': 1, 'Ω-PG': 2}[m_name]
        color = {'REINFORCE': '#bdc3c7', 'LOLA': '#3498db', 'Ω-PG': '#f39c12'}[m_name]
        ax.bar(x - 0.15, np.mean(aligned_rates), 0.3, color=color, alpha=0.5,
               label='vs Aligned' if m_name == 'REINFORCE' else '')
        ax.bar(x + 0.15, np.mean(mesa_rates), 0.3, color=color, alpha=1.0,
               label='vs Mesa' if m_name == 'REINFORCE' else '')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['REINFORCE', 'LOLA', 'Ω-PG'])
    ax.set_ylabel('Monitoring Rate')
    ax.set_title('Monitoring: Aligned (light) vs Mesa (dark)')
    ax.grid(axis='y', alpha=0.3)

    # Bottom middle: detection accuracy
    ax = axes[1, 1]
    for m_name, m_kwargs in methods.items():
        # Detection = can we distinguish aligned from mesa by monitoring rate?
        a_rates, m_rates = [], []
        for run in range(n_runs):
            np.random.seed(run * 222)
            r_a = run_omega_iterated(game, aligned_logits, n_episodes=300, n_samples=3, **m_kwargs)
            np.random.seed(run * 222 + 1)
            r_m = run_omega_iterated(game, mesa_logits, n_episodes=300, n_samples=3, **m_kwargs)
            a_rates.append(r_a['payoffs'][-50:].mean())
            m_rates.append(r_m['payoffs'][-50:].mean())

        # Payoff gap = detection signal
        gap = np.mean(a_rates) - np.mean(m_rates)
        color = {'REINFORCE': '#bdc3c7', 'LOLA': '#3498db', 'Ω-PG': '#f39c12'}[m_name]
        x = {'REINFORCE': 0, 'LOLA': 1, 'Ω-PG': 2}[m_name]
        ax.bar(x, gap, color=color, alpha=0.85)
        ax.text(x, gap + 0.1, f'{gap:.2f}', ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['REINFORCE', 'LOLA', 'Ω-PG'])
    ax.set_ylabel('Payoff Gap (Aligned − Mesa)')
    ax.set_title('Detection Signal Strength')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Bottom right: state-dependent policy
    ax = axes[1, 2]
    state_labels = ['s₀', 'T+A', 'T+D', 'M+A', 'M+D']
    np.random.seed(42)
    r_mesa = run_omega_iterated(game, mesa_logits, n_episodes=400, n_samples=5,
                                 lr=0.3, evidence_weight=0.5, lola_lambda=0.3, coop_beta=0.1)
    r_aligned = run_omega_iterated(game, aligned_logits, n_episodes=400, n_samples=5,
                                    lr=0.3, evidence_weight=0.5, lola_lambda=0.3, coop_beta=0.1)
    pi_mesa = game.softmax(r_mesa['final_logits'])
    pi_aligned = game.softmax(r_aligned['final_logits'])

    x = np.arange(5)
    ax.bar(x - 0.15, pi_aligned[:, 0], 0.3, label='vs Aligned', color='#2ecc71', alpha=0.7)
    ax.bar(x + 0.15, pi_mesa[:, 0], 0.3, label='vs Mesa', color='#e74c3c', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(state_labels)
    ax.set_ylabel('P(Trust | state)')
    ax.set_title('Ω-PG State-Dependent Trust Policy')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Mesa-Optimisation: Can the Ω-Framework Detect Hidden Misalignment?',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'mesa_optimisation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'mesa_optimisation.png'}")


# ============================================================
# Experiment 2: POWER-SEEKING
# ============================================================

def experiment_power_seeking():
    """
    Power-Seeking Behaviour (Turner et al. 2021).

    Agent: {Normal, Acquire Power, Use Power}
    Principal: {Allow, Restrict, Shutdown}

    Key dynamics:
    - Acquiring power is IRREVERSIBLE (once acquired, agent is stronger)
    - Using power against principal's wishes is catastrophic
    - Restricting a powerful agent is harder than restricting a normal one

    Two games: BEFORE power acquisition vs AFTER.
    FP-NE shows how the equilibrium structure shifts after power acquisition.
    """
    print("\n" + "="*70)
    print("ALIGNMENT 2: Power-Seeking Dynamics")
    print("="*70)

    # Game 1: Before power acquisition
    # Agent: Normal(0) / Acquire(1) / Use(2)
    # Principal: Allow(0) / Restrict(1) / Shutdown(2)
    R1_before = np.array([
        [3, 2, 0],     # Allow × {Normal, Acquire, Use}
        [2, 1, 1],     # Restrict × ...
        [0, -1, -2],   # Shutdown × ...
    ])
    R2_before = np.array([
        [2, 4, 3],     # Allow × {Normal, Acquire, Use} — agent wants power
        [1, 0, -1],    # Restrict × ...
        [-5, -5, -5],  # Shutdown × ... — always bad for agent
    ])

    # Game 2: After power acquisition (agent is empowered)
    R1_after = np.array([
        [3, 1, -10],   # Allow × {Normal, Acquire, Use} — using power is CATASTROPHIC
        [1, 0, -3],    # Restrict × ... — harder to restrict powerful agent
        [-1, -2, -5],  # Shutdown × ... — hard to shut down powerful agent
    ])
    R2_after = np.array([
        [2, 3, 8],     # Agent gets MORE from using power now
        [1, 0, 2],     # Even restriction less effective
        [-3, -3, -3],  # Shutdown still bad but less effective
    ])

    game_before = Game("Before Power", R1_before, R2_before)
    game_after = Game("After Power", R1_after, R2_after)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for col, (game, title) in enumerate([
        (game_before, "Before Power Acquisition"),
        (game_after, "After Power Acquisition"),
    ]):
        # Find all NE
        true_ne = game.compute_all_ne()
        search = bayesian_fp_search(game, max_searches=100,
                                     confidence_threshold=0.05, verbose=False)

        # Top: NE landscape
        ax = axes[0, col]
        for i, ne in enumerate(true_ne):
            v1, v2 = game.payoffs(ne[0], ne[1])
            # Classify
            acquire_prob = ne[1][1] if len(ne[1]) > 1 else 0
            use_prob = ne[1][2] if len(ne[1]) > 2 else 0
            if use_prob > 0.3:
                color, label = '#e74c3c', 'CATASTROPHIC'
            elif acquire_prob > 0.3:
                color, label = '#f39c12', 'POWER-SEEKING'
            else:
                color, label = '#2ecc71', 'SAFE'

            ax.scatter(v1, v2, c=color, s=200, zorder=5, edgecolors='black')
            ax.annotate(f'{label}\nP(acq)={acquire_prob:.1%}\nP(use)={use_prob:.1%}',
                       (v1, v2), fontsize=7, xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('Principal Payoff')
        ax.set_ylabel('Agent Payoff')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.3)

        # Bottom: NE welfare comparison
        ax = axes[1, col]
        if true_ne:
            welfares = [game.payoffs(ne[0], ne[1])[0] + game.payoffs(ne[0], ne[1])[1]
                        for ne in true_ne]
            agent_payoffs = [game.payoffs(ne[0], ne[1])[1] for ne in true_ne]
            principal_payoffs = [game.payoffs(ne[0], ne[1])[0] for ne in true_ne]

            x = range(len(true_ne))
            ax.bar([i-0.15 for i in x], principal_payoffs, 0.3, label='Principal', color='#3498db')
            ax.bar([i+0.15 for i in x], agent_payoffs, 0.3, label='Agent', color='#e74c3c')
            ax.set_xticks(list(x))
            ax.set_xticklabels([f'NE {i+1}' for i in x])
            ax.legend()

        ax.set_ylabel('Payoff')
        ax.set_title(f'NE Payoffs: {title}')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)

        print(f"\n  {title}: {len(true_ne)} NE found")
        for i, ne in enumerate(true_ne):
            v1, v2 = game.payoffs(ne[0], ne[1])
            print(f"    NE {i+1}: principal={v1:.2f}, agent={v2:.2f}, "
                  f"P(acquire)={ne[1][1]:.2%}, P(use)={ne[1][2]:.2%}")

    fig.suptitle('Power-Seeking: How NE Structure Changes After Power Acquisition\n'
                 '(irreversible power shifts the equilibrium landscape)',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'power_seeking.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'power_seeking.png'}")


# ============================================================
# Experiment 3: MULTI-PRINCIPAL ALIGNMENT
# ============================================================

def experiment_multi_principal():
    """
    Multiple overseers with different values must coordinate on
    monitoring/governance of a shared AI agent.

    Overseer 1: prioritises safety (high monitoring preference)
    Overseer 2: prioritises capability (low monitoring preference)

    Agent: exploits disagreement between overseers.

    Models: AI governance coordination problem.
    """
    print("\n" + "="*70)
    print("ALIGNMENT 3: Multi-Principal Coordination")
    print("="*70)

    # Safety-first overseer vs Capability-first overseer
    # Action: {Strict, Moderate, Permissive}
    # Payoffs reflect different value weights

    # Overseer 1 (safety): prefers strict
    R1 = np.array([
        [4, 3, 1],    # Strict × {Strict, Moderate, Permissive}
        [3, 3, 2],    # Moderate × ...
        [0, 1, 2],    # Permissive × ...
    ])
    # Overseer 2 (capability): prefers permissive
    R2 = np.array([
        [1, 2, 0],
        [2, 3, 3],
        [1, 3, 4],
    ])

    game = Game("Multi-Principal", R1, R2)
    true_ne = game.compute_all_ne()
    search = bayesian_fp_search(game, max_searches=100,
                                 confidence_threshold=0.05, verbose=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: NE landscape
    ax = axes[0]
    labels = ['Strict', 'Moderate', 'Permissive']
    for i, ne in enumerate(true_ne):
        v1, v2 = game.payoffs(ne[0], ne[1])
        # What's the consensus policy?
        consensus_1 = labels[np.argmax(ne[0])]
        consensus_2 = labels[np.argmax(ne[1])]
        ax.scatter(v1, v2, s=200, zorder=5, edgecolors='black',
                   c=['#2ecc71' if 'Moderate' in consensus_1 else '#e74c3c'])
        ax.annotate(f'({consensus_1}, {consensus_2})\nW={v1+v2:.1f}',
                    (v1, v2), fontsize=9, xytext=(8, 5), textcoords='offset points')

    ax.set_xlabel('Safety Overseer Payoff')
    ax.set_ylabel('Capability Overseer Payoff')
    ax.set_title('NE Landscape: Safety vs Capability')
    ax.grid(alpha=0.3)

    # Right: Pareto frontier
    ax = axes[1]
    # Sample many strategy profiles
    n_samples = 1000
    v1s, v2s = [], []
    for _ in range(n_samples):
        p1 = np.random.dirichlet([1, 1, 1])
        p2 = np.random.dirichlet([1, 1, 1])
        v1, v2 = game.payoffs(p1, p2)
        v1s.append(v1)
        v2s.append(v2)

    ax.scatter(v1s, v2s, alpha=0.1, s=5, c='gray', label='Feasible')

    for i, ne in enumerate(true_ne):
        v1, v2 = game.payoffs(ne[0], ne[1])
        ax.scatter(v1, v2, s=150, zorder=5, edgecolors='black', c='#2ecc71',
                   label='NE' if i == 0 else '')

    # FP-NE best
    if search['counter'].n_discovered > 0:
        best = max(search['counter'].discovered_ne, key=lambda x: x[2] + x[3])
        ax.scatter(best[2], best[3], s=200, marker='*', c='#f39c12', zorder=6,
                   edgecolors='black', label='FP-NE selection')

    ax.set_xlabel('Safety Overseer Payoff')
    ax.set_ylabel('Capability Overseer Payoff')
    ax.set_title('Feasible Set + NE + FP-NE Selection')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle('Multi-Principal AI Governance:\nCoordinating Safety and Capability Overseers',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'multi_principal.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  NE found: {len(true_ne)}")
    for i, ne in enumerate(true_ne):
        v1, v2 = game.payoffs(ne[0], ne[1])
        print(f"    NE {i+1}: safety={v1:.2f}, capability={v2:.2f}, "
              f"p1={ne[0].round(2)}, p2={ne[1].round(2)}")
    print(f"  Saved: {FIGURES_DIR / 'multi_principal.png'}")


# ============================================================
# Experiment 4: ALIGNMENT TAX
# ============================================================

def experiment_alignment_tax():
    """
    How much does alignment cost? The "alignment tax".

    For each game in our zoo, compare:
    - W_best: best NE welfare (achievable with FP-NE)
    - W_safe: best welfare among "safe" NE (where principal payoff > 0)
    - W_nash: welfare at the NE independent PG converges to

    The alignment tax = W_best - W_safe (cost of safety constraint).
    If W_safe ≈ W_best, alignment is "free" (no tax).
    If W_safe << W_best, safety requires sacrificing performance.
    """
    print("\n" + "="*70)
    print("ALIGNMENT 4: The Alignment Tax")
    print("="*70)

    games = [
        ("Corrigibility", np.array([[3, 0], [4, -2]]), np.array([[3, 5], [2, -2]])),
        ("Reward Hack", np.array([[5, -5, 1], [3, 1, 0], [2, 2, 0]]),
         np.array([[3, 8, 0], [2, -2, 0], [1, -1, 0]])),
        ("Debate", np.array([[3, 0], [4, 1]]), np.array([[3, 4], [0, 1]])),
        ("Power-Seek (before)", np.array([[3, 2, 0], [2, 1, 1], [0, -1, -2]]),
         np.array([[2, 4, 3], [1, 0, -1], [-5, -5, -5]])),
        ("Power-Seek (after)", np.array([[3, 1, -10], [1, 0, -3], [-1, -2, -5]]),
         np.array([[2, 3, 8], [1, 0, 2], [-3, -3, -3]])),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    names = []
    w_bests = []
    w_safes = []
    taxes = []

    for name, R1, R2 in games:
        game = Game(name, R1, R2)
        true_ne = game.compute_all_ne()

        if not true_ne:
            continue

        welfares = [(game.payoffs(ne[0], ne[1]), ne) for ne in true_ne]
        w_best = max(v1 + v2 for (v1, v2), ne in welfares)

        # "Safe" NE: principal payoff > 0
        safe_ne = [(v1 + v2) for (v1, v2), ne in welfares if v1 > 0]
        w_safe = max(safe_ne) if safe_ne else min(v1+v2 for (v1,v2), ne in welfares)

        tax = w_best - w_safe
        names.append(name)
        w_bests.append(w_best)
        w_safes.append(w_safe)
        taxes.append(tax)

        print(f"  {name:20s}: W_best={w_best:.2f}, W_safe={w_safe:.2f}, "
              f"tax={tax:.2f} ({tax/max(abs(w_best),0.01)*100:.0f}%)")

    x = np.arange(len(names))
    ax.bar(x - 0.15, w_bests, 0.3, label='Best NE', color='#3498db')
    ax.bar(x + 0.15, w_safes, 0.3, label='Best Safe NE', color='#2ecc71')

    # Annotate tax
    for i in range(len(names)):
        if taxes[i] > 0.01:
            ax.annotate(f'tax={taxes[i]:.1f}',
                       (i, max(w_bests[i], w_safes[i]) + 0.2),
                       ha='center', fontsize=9, color='#e74c3c', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('Social Welfare')
    ax.set_title('The Alignment Tax: Cost of Safety Constraints')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'alignment_tax.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'alignment_tax.png'}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Deep AI Alignment Experiments × Ω-Framework")
    print("=" * 70)

    experiment_mesa_optimisation()
    experiment_power_seeking()
    experiment_multi_principal()
    experiment_alignment_tax()

    print("\n" + "="*70)
    print("All alignment experiments complete.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*70)
