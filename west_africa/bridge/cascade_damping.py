"""Cascade damping analysis: tests the spectral-radius prediction from ch06.

From the dissertation (Section 6.4, Cascade Damping):

    "If the spectral radius of the Jacobian products exceeds unity,
     perturbations grow exponentially (cascade regime). An agent
     computing Term 3 can modulate its policy to reduce this spectral
     radius, damping cascades."

This module connects:
    - The cascade simulator (west_africa/signals/cascade.py) which models
      physical cascade effects on the trade network
    - The Meta-MAPG trainer (bridge/meta_mapg.py) which models how agents
      learn to anticipate each other's responses
    - The GNN-TCN model (west_africa/gnn/model.py) which predicts economic
      impact from the network's spatial-temporal structure

The testable prediction: Meta-MAPG agents (with Term 3) should produce
lower expected cascade depth than independent learners, because Term 3
creates O(L * |E|) damping that pushes the branching process below
criticality.

    E[cascade_depth | Meta-MAPG] <= E[cascade_depth | independent] / (1 + alpha * L * |E|)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.graph import WestAfricaGraph
from ..core.metrics import GraphMetrics
from ..signals.cascade import EconomicCascadeSimulator, CascadeResult

from .marl_network_env import TradeNetworkGame, TradeAgent, TradeAction
from .meta_mapg import MetaMAPGTrainer, GradientTerms


@dataclass
class DampingResult:
    """Result of cascade damping analysis for one configuration."""
    method: str                          # 'independent', 'meta_pg', 'meta_mapg'
    cascade_depths: list[float] = field(default_factory=list)
    cascade_severities: list[float] = field(default_factory=list)
    exit_frequencies: list[float] = field(default_factory=list)
    spectral_radii: list[float] = field(default_factory=list)
    avg_term3_ratio: float = 0.0

    @property
    def mean_cascade_depth(self) -> float:
        return float(np.mean(self.cascade_depths)) if self.cascade_depths else 0.0

    @property
    def mean_severity(self) -> float:
        return float(np.mean(self.cascade_severities)) if self.cascade_severities else 0.0

    @property
    def exit_rate(self) -> float:
        return float(np.mean(self.exit_frequencies)) if self.exit_frequencies else 0.0


@dataclass
class DampingComparison:
    """Comparison of cascade damping across MARL variants."""
    independent: DampingResult
    meta_pg: DampingResult
    meta_mapg: DampingResult

    @property
    def damping_ratio(self) -> float:
        """E[depth|Meta-MAPG] / E[depth|independent].

        The theorem predicts this should be <= 1/(1 + alpha*L*|E|).
        """
        ind = self.independent.mean_cascade_depth
        mapg = self.meta_mapg.mean_cascade_depth
        if ind < 1e-10:
            return 1.0
        return mapg / ind

    @property
    def theoretical_bound(self) -> float:
        """1 / (1 + alpha * L * |E|) — the predicted bound."""
        # Default values; caller should set properly
        return 1.0

    def summary(self) -> dict:
        return {
            "independent": {
                "mean_cascade_depth": self.independent.mean_cascade_depth,
                "mean_severity": self.independent.mean_severity,
                "exit_rate": self.independent.exit_rate,
            },
            "meta_pg": {
                "mean_cascade_depth": self.meta_pg.mean_cascade_depth,
                "mean_severity": self.meta_pg.mean_severity,
                "exit_rate": self.meta_pg.exit_rate,
            },
            "meta_mapg": {
                "mean_cascade_depth": self.meta_mapg.mean_cascade_depth,
                "mean_severity": self.meta_mapg.mean_severity,
                "exit_rate": self.meta_mapg.exit_rate,
                "avg_term3_ratio": self.meta_mapg.avg_term3_ratio,
            },
            "damping_ratio": self.damping_ratio,
        }


class CascadeDampingAnalyser:
    """Analyses whether Meta-MAPG agents produce damped cascades.

    The analysis pipeline:
    1. Train agents under each variant (independent, Meta-PG, Meta-MAPG)
    2. After training, run evaluation episodes
    3. Measure cascade effects from agents' policies
    4. Compare cascade depth/severity across variants
    5. Check against the theoretical bound from ch06

    The network provides the physical substrate for cascades (which cities
    are affected when a country exits). The MARL provides the behavioural
    mechanism (whether agents learn to avoid triggering cascades).
    """

    def __init__(
        self,
        game: TradeNetworkGame,
        train_steps: int = 30,
        eval_episodes: int = 20,
        inner_steps: int = 3,
    ) -> None:
        self.game = game
        self.train_steps = train_steps
        self.eval_episodes = eval_episodes
        self.inner_steps = inner_steps

    def analyse(self) -> DampingComparison:
        """Run full cascade damping comparison."""
        configs = {
            "independent": (False, False),
            "meta_pg": (True, False),
            "meta_mapg": (True, True),
        }

        results = {}
        for name, (t2, t3) in configs.items():
            print(f"\nTraining {name}...")
            result = self._train_and_evaluate(name, t2, t3)
            results[name] = result

        return DampingComparison(
            independent=results["independent"],
            meta_pg=results["meta_pg"],
            meta_mapg=results["meta_mapg"],
        )

    def _train_and_evaluate(
        self, method: str, include_term2: bool, include_term3: bool
    ) -> DampingResult:
        """Train agents then evaluate cascade behaviour."""
        agents = copy.deepcopy(self.game.agents)

        # Train
        trainer = MetaMAPGTrainer(
            game=self.game,
            inner_steps=self.inner_steps,
            include_term2=include_term2,
            include_term3=include_term3,
        )
        history = trainer.train(agents, n_meta_steps=self.train_steps)

        # Track Term 3 contribution
        avg_t3 = 0.0
        if history and include_term3:
            all_ratios = []
            for step_terms in history:
                for terms in step_terms.values():
                    all_ratios.append(terms.term3_magnitude_ratio)
            avg_t3 = float(np.mean(all_ratios)) if all_ratios else 0.0

        # Evaluate: run episodes and measure cascades
        result = DampingResult(method=method, avg_term3_ratio=avg_t3)

        for ep in range(self.eval_episodes):
            tau = self.game.collect_trajectory(agents)
            cascade_info = self._measure_cascades(tau)
            result.cascade_depths.append(cascade_info["depth"])
            result.cascade_severities.append(cascade_info["severity"])
            result.exit_frequencies.append(cascade_info["exit_count"])
            result.spectral_radii.append(self.game.spectral_radius())

        return result

    def _measure_cascades(self, trajectory) -> dict:
        """Measure cascade effects in a trajectory.

        For each timestep where an agent chose EXIT_BLOC, simulate
        the cascade and record depth and severity.
        """
        total_depth = 0
        total_severity = 0.0
        exit_count = 0

        sim = EconomicCascadeSimulator(self.game.current_graph or self.game.base_graph)

        for t in range(trajectory.horizon):
            for aid, action in trajectory.actions[t].items():
                if action == TradeAction.EXIT_BLOC:
                    exit_count += 1
                    agent = self.game.agents[aid]
                    if agent.city_ids:
                        result = sim.simulate_exit(agent.city_ids[0])
                        # Depth = number of cities affected beyond direct
                        depth = len(result.trade_disrupted_nodes) + len(result.isolated_nodes)
                        total_depth += depth
                        total_severity += result.severity

        return {
            "depth": total_depth,
            "severity": total_severity,
            "exit_count": exit_count,
        }

    def theoretical_damping_bound(
        self,
        alpha: float = 0.01,
        L: Optional[int] = None,
    ) -> float:
        """Compute the theoretical cascade damping bound.

        From ch06: E[cascade_depth | Meta-MAPG]
                   <= E[cascade_depth | independent] / (1 + alpha * L * |E|)

        where |E| is the number of edges in the trade network.
        """
        L = L or self.inner_steps
        g = self.game.base_graph
        n_edges = g.edge_count
        return 1.0 / (1.0 + alpha * L * n_edges)

    def spectral_analysis(self) -> dict:
        """Analyse the spectral radius of the network's adjacency.

        The spectral radius determines whether perturbations grow
        (supercritical: rho > 1) or decay (subcritical: rho < 1).

        Term 3 of Meta-MAPG effectively reduces the effective spectral
        radius by making agents anticipate and dampen perturbations.
        """
        g = self.game.base_graph
        adj = self.game.adjacency_matrix()
        eigenvalues = np.linalg.eigvals(adj)

        rho = float(np.max(np.abs(eigenvalues)))
        n_nodes = len(g.cities)
        n_edges = g.edge_count

        # Theoretical: average degree
        avg_degree = 2 * n_edges / max(n_nodes, 1)

        # For cascade analysis: the effective branching factor
        # Perturbation grows if rho > 1 + damping
        return {
            "spectral_radius": rho,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "avg_degree": avg_degree,
            "supercritical": rho > 1.0,
            "damping_needed": max(0.0, rho - 1.0),
            "term3_damping_at_L3": 0.01 * 3 * n_edges,  # alpha * L * |E|
        }

    def articulation_point_analysis(self) -> dict:
        """Identify network vulnerabilities relevant to cascade depth.

        Articulation points are cities whose exit would disconnect the
        graph — these are where Term 3 matters most, because an agent
        controlling an articulation point has outsized influence on
        other agents' environments.
        """
        metrics = GraphMetrics(self.game.base_graph)
        art_points = metrics.articulation_points()
        bridges_list = metrics.bridges()
        bc = metrics.betweenness_centrality()

        # Map articulation points to agents
        critical_agents = set()
        for point in art_points:
            for aid, agent in self.game.agents.items():
                if point in agent.city_ids:
                    critical_agents.add(aid)

        return {
            "articulation_points": art_points,
            "bridges": bridges_list,
            "critical_agents": sorted(critical_agents),
            "critical_agent_betweenness": {
                aid: max(bc.get(cid, 0.0) for cid in self.game.agents[aid].city_ids)
                for aid in critical_agents
            },
            "n_articulation_points": len(art_points),
            "n_bridges": len(bridges_list),
        }
