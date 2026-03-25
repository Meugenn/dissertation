"""Run the Meta-MAPG vs Independent PG vs Meta-PG ablation experiment.

Measures:
1. Average discounted return per agent across meta-steps
2. Cascade depth and severity
3. Exit frequency (how often agents choose exit_bloc)
4. Term 3 magnitude ratio (peer-learning contribution)
5. Wall-clock time

Usage:
    python -m west_africa.bridge.run_experiments
"""

from __future__ import annotations

import copy
import json
import time
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from west_africa.core.graph import WestAfricaGraph
from west_africa.bridge.marl_network_env import TradeNetworkGame, TradeAgent, TradeAction
from west_africa.bridge.meta_mapg import MetaMAPGTrainer, GradientTerms
from west_africa.bridge.cascade_damping import CascadeDampingAnalyser


def run_training_experiment(
    game: TradeNetworkGame,
    n_meta_steps: int = 40,
    inner_steps: int = 3,
    n_trajectories: int = 5,
    eval_episodes: int = 15,
) -> dict:
    """Run full three-way ablation with evaluation."""

    configs = {
        "independent": (False, False),
        "meta_pg": (True, False),
        "meta_mapg": (True, True),
    }

    results = {}

    for name, (t2, t3) in configs.items():
        print(f"\n{'='*70}")
        print(f"  TRAINING: {name.upper()} (Term2={t2}, Term3={t3})")
        print(f"{'='*70}")

        agents = copy.deepcopy(game.agents)
        trainer = MetaMAPGTrainer(
            game=game,
            inner_steps=inner_steps,
            n_trajectories=n_trajectories,
            include_term2=t2,
            include_term3=t3,
        )

        # Train
        t_start = time.time()
        history = trainer.train(agents, n_meta_steps=n_meta_steps)
        train_time = time.time() - t_start

        # Extract training metrics
        returns_per_step = []
        t3_ratios_per_step = []
        for step_terms in history:
            step_returns = [t.meta_return for t in step_terms.values()]
            returns_per_step.append(float(np.mean(step_returns)))
            if t3:
                t3_rats = [t.term3_magnitude_ratio for t in step_terms.values()]
                t3_ratios_per_step.append(float(np.mean(t3_rats)))

        # Evaluate trained agents
        print(f"\n  Evaluating {name} over {eval_episodes} episodes...")
        eval_results = evaluate_agents(game, agents, eval_episodes)

        results[name] = {
            "train_time_s": round(train_time, 2),
            "returns_per_step": returns_per_step,
            "final_avg_return": round(returns_per_step[-1], 6) if returns_per_step else 0.0,
            "first_avg_return": round(returns_per_step[0], 6) if returns_per_step else 0.0,
            "improvement": round(
                (returns_per_step[-1] - returns_per_step[0]), 6
            ) if len(returns_per_step) >= 2 else 0.0,
            "t3_ratios": t3_ratios_per_step,
            "avg_t3_ratio": round(float(np.mean(t3_ratios_per_step)), 6) if t3_ratios_per_step else 0.0,
            **eval_results,
        }

    return results


def evaluate_agents(
    game: TradeNetworkGame,
    agents: dict[str, TradeAgent],
    n_episodes: int = 15,
) -> dict:
    """Evaluate trained agents: returns, cascade metrics, action distribution."""

    episode_returns = []
    total_cascade_depth = 0
    total_cascade_severity = 0.0
    total_exits = 0
    total_steps = 0
    action_counts = {a.value: 0 for a in TradeAction}

    from west_africa.signals.cascade import EconomicCascadeSimulator

    for ep in range(n_episodes):
        tau = game.collect_trajectory(agents)
        ep_return = np.mean([
            tau.discounted_return(aid, game.gamma)
            for aid in agents
        ])
        episode_returns.append(float(ep_return))

        # Count actions and cascades
        for t in range(tau.horizon):
            for aid, action in tau.actions[t].items():
                action_counts[action.value] += 1
                total_steps += 1
                if action == TradeAction.EXIT_BLOC:
                    total_exits += 1
                    agent = game.agents[aid]
                    if agent.city_ids and game.current_graph:
                        sim = EconomicCascadeSimulator(game.current_graph)
                        result = sim.simulate_exit(agent.city_ids[0])
                        total_cascade_depth += len(result.trade_disrupted_nodes) + len(result.isolated_nodes)
                        total_cascade_severity += result.severity

    return {
        "eval_mean_return": round(float(np.mean(episode_returns)), 6),
        "eval_std_return": round(float(np.std(episode_returns)), 6),
        "eval_min_return": round(float(np.min(episode_returns)), 6),
        "eval_max_return": round(float(np.max(episode_returns)), 6),
        "total_exits": total_exits,
        "exit_rate": round(total_exits / max(total_steps, 1), 6),
        "total_cascade_depth": total_cascade_depth,
        "avg_cascade_depth_per_exit": round(
            total_cascade_depth / max(total_exits, 1), 4
        ),
        "total_cascade_severity": round(total_cascade_severity, 6),
        "avg_cascade_severity_per_exit": round(
            total_cascade_severity / max(total_exits, 1), 6
        ),
        "action_distribution": {
            k: round(v / max(total_steps, 1), 4)
            for k, v in action_counts.items()
        },
    }


def run_spectral_analysis(game: TradeNetworkGame) -> dict:
    """Run spectral and structural analysis."""
    game.reset()
    analyser = CascadeDampingAnalyser(game)

    spectral = analyser.spectral_analysis()
    artic = analyser.articulation_point_analysis()
    bound = analyser.theoretical_damping_bound(alpha=0.01, L=3)

    return {
        "spectral": spectral,
        "articulation": {
            "n_articulation_points": artic["n_articulation_points"],
            "n_bridges": artic["n_bridges"],
            "critical_agents": artic["critical_agents"],
        },
        "theoretical_damping_bound": round(bound, 6),
    }


def print_comparison(results: dict) -> None:
    """Print a formatted comparison table."""

    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    # Training performance
    print("\n--- Training Performance ---")
    print(f"{'Metric':<35} {'Independent':>14} {'Meta-PG':>14} {'Meta-MAPG':>14}")
    print("-" * 77)

    metrics = [
        ("First avg return", "first_avg_return"),
        ("Final avg return", "final_avg_return"),
        ("Improvement", "improvement"),
        ("Train time (s)", "train_time_s"),
    ]
    for label, key in metrics:
        vals = [str(results[m].get(key, "N/A")) for m in ["independent", "meta_pg", "meta_mapg"]]
        print(f"{label:<35} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")

    # Evaluation performance
    print("\n--- Evaluation Performance ---")
    print(f"{'Metric':<35} {'Independent':>14} {'Meta-PG':>14} {'Meta-MAPG':>14}")
    print("-" * 77)

    eval_metrics = [
        ("Eval mean return", "eval_mean_return"),
        ("Eval std return", "eval_std_return"),
        ("Exit rate", "exit_rate"),
        ("Total exits", "total_exits"),
        ("Avg cascade depth / exit", "avg_cascade_depth_per_exit"),
        ("Avg cascade severity / exit", "avg_cascade_severity_per_exit"),
    ]
    for label, key in eval_metrics:
        vals = [str(results[m].get(key, "N/A")) for m in ["independent", "meta_pg", "meta_mapg"]]
        print(f"{label:<35} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")

    # Term 3 analysis
    if results["meta_mapg"].get("avg_t3_ratio", 0) > 0:
        print(f"\n--- Term 3 Analysis ---")
        print(f"Avg Term 3 magnitude ratio (Meta-MAPG): {results['meta_mapg']['avg_t3_ratio']:.6f}")

    # Action distributions
    print("\n--- Action Distributions ---")
    for method in ["independent", "meta_pg", "meta_mapg"]:
        dist = results[method].get("action_distribution", {})
        print(f"\n  {method}:")
        for action, freq in sorted(dist.items(), key=lambda x: -x[1]):
            bar = "█" * int(freq * 50)
            print(f"    {action:<20} {freq:.4f} {bar}")

    # Performance improvement
    print("\n--- Performance Improvement Over Independent PG ---")
    ind_ret = results["independent"]["eval_mean_return"]
    for method in ["meta_pg", "meta_mapg"]:
        m_ret = results[method]["eval_mean_return"]
        if abs(ind_ret) > 1e-10:
            pct = (m_ret - ind_ret) / abs(ind_ret) * 100
        else:
            pct = 0.0 if abs(m_ret - ind_ret) < 1e-10 else float("inf")
        print(f"  {method}: {pct:+.2f}% (return: {ind_ret:.6f} → {m_ret:.6f})")

    # Cascade damping
    ind_depth = results["independent"]["avg_cascade_depth_per_exit"]
    mapg_depth = results["meta_mapg"]["avg_cascade_depth_per_exit"]
    if ind_depth > 0:
        damping = mapg_depth / ind_depth
        print(f"\n--- Cascade Damping ---")
        print(f"  Damping ratio (Meta-MAPG / Independent): {damping:.4f}")
        print(f"  Independent avg cascade depth: {ind_depth:.4f}")
        print(f"  Meta-MAPG avg cascade depth:   {mapg_depth:.4f}")


def run_multi_seed(
    game: TradeNetworkGame,
    n_seeds: int = 3,
    n_meta_steps: int = 40,
    inner_steps: int = 3,
    n_trajectories: int = 5,
    eval_episodes: int = 15,
) -> dict:
    """Run experiment across multiple random seeds for statistical reliability."""
    all_results = {m: [] for m in ["independent", "meta_pg", "meta_mapg"]}

    for seed in range(n_seeds):
        print(f"\n{'#'*70}")
        print(f"  SEED {seed}")
        print(f"{'#'*70}")
        np.random.seed(seed * 42 + 7)

        results = run_training_experiment(
            game,
            n_meta_steps=n_meta_steps,
            inner_steps=inner_steps,
            n_trajectories=n_trajectories,
            eval_episodes=eval_episodes,
        )
        for method, data in results.items():
            all_results[method].append(data)

    # Aggregate across seeds
    aggregated = {}
    for method, runs in all_results.items():
        agg = {}
        for key in [
            "eval_mean_return", "exit_rate", "total_exits",
            "avg_cascade_depth_per_exit", "avg_cascade_severity_per_exit",
            "total_cascade_depth", "total_cascade_severity",
        ]:
            vals = [r[key] for r in runs]
            agg[f"{key}_mean"] = round(float(np.mean(vals)), 6)
            agg[f"{key}_std"] = round(float(np.std(vals)), 6)
        agg["n_seeds"] = n_seeds
        agg["runs"] = runs
        aggregated[method] = agg

    return aggregated


def print_multi_seed_comparison(agg: dict) -> None:
    """Print aggregated multi-seed results."""
    print("\n" + "=" * 80)
    print("  MULTI-SEED AGGREGATED RESULTS")
    print("=" * 80)

    metrics = [
        ("Eval return (mean±std)", "eval_mean_return"),
        ("Exit rate", "exit_rate"),
        ("Cascade depth/exit", "avg_cascade_depth_per_exit"),
        ("Cascade severity/exit", "avg_cascade_severity_per_exit"),
    ]

    print(f"\n{'Metric':<30} {'Independent':>18} {'Meta-PG':>18} {'Meta-MAPG':>18}")
    print("-" * 84)

    for label, key in metrics:
        vals = []
        for m in ["independent", "meta_pg", "meta_mapg"]:
            mean = agg[m][f"{key}_mean"]
            std = agg[m][f"{key}_std"]
            vals.append(f"{mean:.4f}±{std:.4f}")
        print(f"{label:<30} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

    # Compute improvement percentages
    print("\n--- Performance vs Independent PG (mean across seeds) ---")
    ind_ret = agg["independent"]["eval_mean_return_mean"]
    for method in ["meta_pg", "meta_mapg"]:
        m_ret = agg[method]["eval_mean_return_mean"]
        if abs(ind_ret) > 1e-10:
            pct = (m_ret - ind_ret) / abs(ind_ret) * 100
        else:
            pct = 0.0
        print(f"  {method}: {pct:+.2f}% return improvement")

    print("\n--- Cascade Damping (mean across seeds) ---")
    ind_depth = agg["independent"]["avg_cascade_depth_per_exit_mean"]
    mapg_depth = agg["meta_mapg"]["avg_cascade_depth_per_exit_mean"]
    mpg_depth = agg["meta_pg"]["avg_cascade_depth_per_exit_mean"]
    if ind_depth > 0:
        print(f"  Meta-PG damping ratio:   {mpg_depth/ind_depth:.4f} ({(1-mpg_depth/ind_depth)*100:+.1f}%)")
        print(f"  Meta-MAPG damping ratio: {mapg_depth/ind_depth:.4f} ({(1-mapg_depth/ind_depth)*100:+.1f}%)")

    ind_sev = agg["independent"]["avg_cascade_severity_per_exit_mean"]
    mapg_sev = agg["meta_mapg"]["avg_cascade_severity_per_exit_mean"]
    if ind_sev > 0:
        print(f"  Meta-MAPG severity ratio: {mapg_sev/ind_sev:.4f} ({(1-mapg_sev/ind_sev)*100:+.1f}%)")


def main():
    print("Loading West Africa trade network...")
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    g = WestAfricaGraph.from_seed_data(data_dir)
    print(f"  {g.node_count} cities, {g.edge_count} edges, {len(g.get_ftz_targets())} FTZ targets")

    game = TradeNetworkGame(graph=g, horizon=6)
    print(f"  {game.n_agents} agents (countries)")

    # Spectral analysis
    print("\n--- Spectral & Structural Analysis ---")
    spectral = run_spectral_analysis(game)
    print(f"  Spectral radius: {spectral['spectral']['spectral_radius']:.4f}")
    print(f"  Supercritical: {spectral['spectral']['supercritical']}")
    print(f"  Articulation points: {spectral['articulation']['n_articulation_points']}")
    print(f"  Bridges: {spectral['articulation']['n_bridges']}")
    print(f"  Critical agents: {spectral['articulation']['critical_agents']}")
    print(f"  Theoretical damping bound (α=0.01, L=3): {spectral['theoretical_damping_bound']:.6f}")

    # Multi-seed experiment
    print("\nStarting multi-seed ablation experiment...")
    agg = run_multi_seed(
        game,
        n_seeds=3,
        n_meta_steps=40,
        inner_steps=3,
        n_trajectories=5,
        eval_episodes=15,
    )

    print_multi_seed_comparison(agg)

    # Single detailed comparison for the last seed
    print("\n\nDetailed single-seed results (last seed):")
    last_seed_results = {
        method: agg[method]["runs"][-1]
        for method in ["independent", "meta_pg", "meta_mapg"]
    }
    print_comparison(last_seed_results)

    # Save results
    output_path = pathlib.Path(__file__).resolve().parent / "experiment_results.json"
    serialisable = {}
    for method, data in agg.items():
        serialisable[method] = {
            k: v for k, v in data.items()
            if k != "runs"
        }
    serialisable["spectral_analysis"] = spectral

    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
