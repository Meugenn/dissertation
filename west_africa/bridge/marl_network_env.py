"""Stochastic game environment wrapping the West Africa trade network.

Maps the dissertation's formal definition (Definition 5.1) onto the network:

    M_n = <I, S, A, P, R, gamma>

where:
    I = {agent_1, ..., agent_n}  -- trade policy agents per country/city
    S = network state (node features + edge weights + bloc memberships)
    A^i = {adjust_tariff, invest_infrastructure, form_agreement, exit_bloc, ...}
    P(s' | s, a) = transition driven by cascade dynamics + economic state updates
    R^i = trade volume growth + FTZ impact score for agent i's cities
    gamma = quarterly discount factor

The non-stationarity problem from ch05 is concrete here: when Nigeria
adjusts tariffs, Ghana's optimal policy changes because the trade graph
changes. Independent learners miss this coupling. Meta-MAPG captures it
through Term 3 (peer learning anticipation).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import networkx as nx

from ..core.graph import WestAfricaGraph
from ..core.types import BlocMembership, ConnectionType, EconomicState
from ..core.metrics import GraphMetrics
from ..signals.cascade import EconomicCascadeSimulator, CascadeResult


class TradeAction(str, Enum):
    """Actions available to each trade policy agent."""
    LOWER_TARIFF = "lower_tariff"           # Reduce tariffs on intra-bloc trade
    RAISE_TARIFF = "raise_tariff"           # Increase tariffs (protectionist)
    INVEST_INFRASTRUCTURE = "invest_infra"  # Improve transport/logistics edges
    FORM_BILATERAL = "form_bilateral"       # New bilateral trade agreement
    EXIT_BLOC = "exit_bloc"                 # Leave ECOWAS/FTZ
    REJOIN_BLOC = "rejoin_bloc"             # Re-enter ECOWAS/FTZ
    NO_ACTION = "no_action"                 # Maintain current policy


@dataclass
class AgentObservation:
    """What agent i observes at time t.

    Partial observability: agent sees own city features + direct
    neighbours, but not the full graph. This maps to the POMDP
    extension discussed in ch04.
    """
    agent_id: str
    own_features: np.ndarray          # (n_features,)
    neighbour_features: np.ndarray    # (n_neighbours, n_features)
    neighbour_ids: list[str]
    trade_volumes: np.ndarray         # trade volume on each edge
    tariff_rates: np.ndarray          # tariffs on each edge
    bloc_status: str                  # current bloc membership
    global_metrics: dict              # aggregate stats visible to all


@dataclass
class TradeAgent:
    """A trade policy agent controlling one country's nodes.

    The policy is parameterised as a simple softmax over actions,
    conditioned on the observation. phi^i in the theorem notation.
    """
    agent_id: str
    country_iso3: str
    city_ids: list[str]
    # Policy parameters: (n_obs_features, n_actions)
    phi: np.ndarray = field(default=None)
    # Learning rate (alpha^i in eq. 5.2)
    lr: float = 0.05

    def __post_init__(self):
        if self.phi is None:
            # Initialise policy parameters
            n_obs = 16  # observation feature dimension
            n_act = len(TradeAction)
            self.phi = np.random.randn(n_obs, n_act) * 0.01

    def policy(self, obs: AgentObservation) -> np.ndarray:
        """Softmax policy: pi^i(a | s, phi^i).

        Returns probability distribution over TradeAction values.
        """
        features = self._encode_obs(obs)
        logits = features @ self.phi
        # Numerically stable softmax
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    def sample_action(self, obs: AgentObservation) -> TradeAction:
        """Sample action from policy."""
        probs = self.policy(obs)
        actions = list(TradeAction)
        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]

    def log_prob(self, obs: AgentObservation, action: TradeAction) -> float:
        """log pi^i(a^i | s, phi^i) — needed for REINFORCE gradient."""
        probs = self.policy(obs)
        idx = list(TradeAction).index(action)
        return float(np.log(probs[idx] + 1e-10))

    def _encode_obs(self, obs: AgentObservation) -> np.ndarray:
        """Encode observation into fixed-size feature vector."""
        own = obs.own_features[:8] if len(obs.own_features) >= 8 else np.zeros(8)
        # Aggregate neighbour info
        if len(obs.neighbour_features) > 0:
            neigh_mean = obs.neighbour_features.mean(axis=0)[:4]
        else:
            neigh_mean = np.zeros(4)
        # Trade stats
        trade_stats = np.array([
            obs.trade_volumes.sum() if len(obs.trade_volumes) > 0 else 0.0,
            obs.trade_volumes.mean() if len(obs.trade_volumes) > 0 else 0.0,
            obs.tariff_rates.mean() if len(obs.tariff_rates) > 0 else 0.0,
            1.0 if obs.bloc_status in ("ECOWAS", "UEMOA") else 0.0,
        ])
        return np.concatenate([own, neigh_mean, trade_stats])


@dataclass
class Trajectory:
    """A single trajectory tau = (s_0, a_0, r_0, s_1, ..., s_H).

    Stores per-agent observations, actions, and rewards for one episode.
    This is tau_ell in the policy chain notation (eq. 5.1).
    """
    observations: list[dict[str, AgentObservation]]  # t -> agent_id -> obs
    actions: list[dict[str, TradeAction]]             # t -> agent_id -> action
    rewards: list[dict[str, float]]                   # t -> agent_id -> reward
    log_probs: list[dict[str, float]]                 # t -> agent_id -> log_prob
    horizon: int = 0

    def discounted_return(self, agent_id: str, gamma: float = 0.99) -> float:
        """G^i(tau) = sum_t gamma^t r_t^i."""
        g = 0.0
        for t in range(self.horizon):
            g += (gamma ** t) * self.rewards[t].get(agent_id, 0.0)
        return g

    def total_log_prob(self, agent_id: str) -> float:
        """log pi^i(tau | phi^i) = sum_t log pi^i(a_t^i | s_t, phi^i)."""
        return sum(lp.get(agent_id, 0.0) for lp in self.log_probs)


class TradeNetworkGame:
    """Stochastic game on the West Africa trade network.

    This is M_n from Definition 5.1, instantiated on the real network.
    The game captures the non-stationarity problem: each agent's
    environment includes all other agents' policies (since tariff
    changes, bloc exits, etc. affect the shared trade graph).
    """

    def __init__(
        self,
        graph: Optional[WestAfricaGraph] = None,
        horizon: int = 8,          # 8 quarters = 2 years per episode
        gamma: float = 0.99,
        tariff_delta: float = 0.02,
        infra_boost: float = 0.1,
    ) -> None:
        self.base_graph = graph or WestAfricaGraph.from_seed_data()
        self.horizon = horizon
        self.gamma = gamma
        self.tariff_delta = tariff_delta
        self.infra_boost = infra_boost

        # Create agents: one per country with ECOWAS/UEMOA/SUSPENDED cities
        self.agents: dict[str, TradeAgent] = {}
        self._build_agents()

        # Simulation state (reset per episode)
        self.current_graph: Optional[WestAfricaGraph] = None
        self.cascade_sim: Optional[EconomicCascadeSimulator] = None
        self.step_count = 0

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    @property
    def agent_ids(self) -> list[str]:
        return list(self.agents.keys())

    def _build_agents(self) -> None:
        """Create one agent per country represented in the graph."""
        countries: dict[str, list[str]] = {}
        for city in self.base_graph.cities.values():
            iso = city.country_iso3
            if iso not in countries:
                countries[iso] = []
            countries[iso].append(city.id)

        for iso, city_ids in countries.items():
            self.agents[iso] = TradeAgent(
                agent_id=iso,
                country_iso3=iso,
                city_ids=city_ids,
            )

    def reset(self) -> dict[str, AgentObservation]:
        """Reset environment for new episode. Returns initial observations."""
        self.current_graph = copy.deepcopy(self.base_graph)
        self.cascade_sim = EconomicCascadeSimulator(self.current_graph)
        self.step_count = 0
        return self._get_observations()

    def step(
        self, joint_action: dict[str, TradeAction]
    ) -> tuple[dict[str, AgentObservation], dict[str, float], bool]:
        """Execute joint action, return (observations, rewards, done).

        This is the transition P(s' | s, a) where a = (a^1, ..., a^n).
        The coupling is explicit: Nigeria's tariff change affects the
        edges that Ghana's reward depends on.
        """
        rewards = {}

        # Apply each agent's action to the shared graph
        for agent_id, action in joint_action.items():
            agent = self.agents[agent_id]
            r = self._apply_action(agent, action)
            rewards[agent_id] = r

        # Add cascade effects: actions may trigger secondary disruptions
        cascade_penalty = self._compute_cascade_effects(joint_action)
        for agent_id in rewards:
            rewards[agent_id] += cascade_penalty.get(agent_id, 0.0)

        self.step_count += 1
        done = self.step_count >= self.horizon
        obs = self._get_observations()

        return obs, rewards, done

    def collect_trajectory(self, agents: dict[str, TradeAgent]) -> Trajectory:
        """Roll out one full episode using the given agent policies.

        This produces tau_ell in the policy chain (eq. 5.1).
        """
        obs_history = []
        act_history = []
        rew_history = []
        lp_history = []

        obs = self.reset()

        for t in range(self.horizon):
            # Each agent samples from its policy
            actions = {}
            log_probs = {}
            for aid, agent in agents.items():
                a = agent.sample_action(obs[aid])
                lp = agent.log_prob(obs[aid], a)
                actions[aid] = a
                log_probs[aid] = lp

            obs_history.append(obs)
            act_history.append(actions)
            lp_history.append(log_probs)

            obs, rewards, done = self.step(actions)
            rew_history.append(rewards)

            if done:
                break

        return Trajectory(
            observations=obs_history,
            actions=act_history,
            rewards=rew_history,
            log_probs=lp_history,
            horizon=len(rew_history),
        )

    # ------------------------------------------------------------------
    # Internal: action effects on the network
    # ------------------------------------------------------------------

    def _apply_action(self, agent: TradeAgent, action: TradeAction) -> float:
        """Apply agent's action to the graph, return immediate reward."""
        g = self.current_graph
        reward = 0.0

        if action == TradeAction.LOWER_TARIFF:
            for cid in agent.city_ids:
                for u, v, key, data in list(g.G.edges(cid, keys=True, data=True)):
                    if data.get("edge_type") == ConnectionType.TRADE.value:
                        old_tariff = data.get("tariff_rate", 0.0)
                        new_tariff = max(0.0, old_tariff - self.tariff_delta)
                        g.G[u][v][key]["tariff_rate"] = new_tariff
                        # Lower tariffs boost trade volume
                        vol = data.get("volume", 0.0)
                        boost = vol * self.tariff_delta * 0.5
                        g.G[u][v][key]["volume"] = vol + boost
                        reward += boost * 0.01  # Scaled reward

        elif action == TradeAction.RAISE_TARIFF:
            for cid in agent.city_ids:
                for u, v, key, data in list(g.G.edges(cid, keys=True, data=True)):
                    if data.get("edge_type") == ConnectionType.TRADE.value:
                        old_tariff = data.get("tariff_rate", 0.0)
                        new_tariff = min(1.0, old_tariff + self.tariff_delta)
                        g.G[u][v][key]["tariff_rate"] = new_tariff
                        # Protectionism: short-term revenue, long-term cost
                        reward += self.tariff_delta * 0.3

        elif action == TradeAction.INVEST_INFRASTRUCTURE:
            for cid in agent.city_ids:
                for u, v, key, data in list(g.G.edges(cid, keys=True, data=True)):
                    if data.get("edge_type") == ConnectionType.INFRASTRUCTURE.value:
                        old_w = data.get("weight", 1.0)
                        g.G[u][v][key]["weight"] = max(0.1, old_w - self.infra_boost)
                        reward += self.infra_boost * 0.5
            reward -= 0.2  # Investment cost

        elif action == TradeAction.EXIT_BLOC:
            # Triggers cascade effects — handled in _compute_cascade_effects
            for cid in agent.city_ids:
                if cid in g.G.nodes:
                    g.G.nodes[cid]["bloc"] = BlocMembership.SUSPENDED.value
            reward -= 0.5  # Immediate disruption cost

        elif action == TradeAction.REJOIN_BLOC:
            for cid in agent.city_ids:
                if cid in g.G.nodes:
                    g.G.nodes[cid]["bloc"] = BlocMembership.ECOWAS.value
            reward += 0.3  # Restored trade benefit

        elif action == TradeAction.FORM_BILATERAL:
            reward += 0.1  # Small benefit from new agreement

        # NO_ACTION: reward = 0
        return reward

    def _compute_cascade_effects(
        self, joint_action: dict[str, TradeAction]
    ) -> dict[str, float]:
        """Compute cascade penalties from disruptive actions.

        This is where network structure creates the multi-agent coupling:
        agent i's exit cascades to affect agent j's trade volumes.
        Term 3 of Meta-MAPG captures exactly this: agent i should
        anticipate how its action changes agent j's learning environment.
        """
        penalties: dict[str, float] = {aid: 0.0 for aid in self.agents}

        exit_nodes = []
        for aid, action in joint_action.items():
            if action == TradeAction.EXIT_BLOC:
                agent = self.agents[aid]
                exit_nodes.extend(agent.city_ids)

        if not exit_nodes:
            return penalties

        # Run cascade simulation
        if len(exit_nodes) == 1:
            result = self.cascade_sim.simulate_exit(exit_nodes[0])
        else:
            result = self.cascade_sim.simulate_multi_exit(exit_nodes)

        # Distribute cascade penalties based on trade disruption
        for disrupted_city in result.trade_disrupted_nodes:
            for aid, agent in self.agents.items():
                if disrupted_city in agent.city_ids:
                    penalties[aid] -= result.severity * 2.0

        for isolated_city in result.isolated_nodes:
            for aid, agent in self.agents.items():
                if isolated_city in agent.city_ids:
                    penalties[aid] -= result.severity * 5.0

        return penalties

    def _get_observations(self) -> dict[str, AgentObservation]:
        """Build partial observations for each agent."""
        g = self.current_graph
        obs = {}

        # Global metrics visible to all
        metrics = GraphMetrics(g)
        bc = metrics.betweenness_centrality()
        global_metrics = {
            "n_components": metrics.component_count(),
            "mean_betweenness": np.mean(list(bc.values())) if bc else 0.0,
        }

        for aid, agent in self.agents.items():
            # Own city features
            own_feats = []
            for cid in agent.city_ids:
                node = g.G.nodes.get(cid, {})
                own_feats.append([
                    node.get("population", 0) / 1e7,
                    node.get("gdp_per_capita", 0) / 5000,
                    node.get("trade_openness", 0),
                    bc.get(cid, 0.0),
                    1.0 if node.get("is_port", False) else 0.0,
                    1.0 if node.get("is_ftz_target", False) else 0.0,
                    node.get("ease_of_business", 0) / 100,
                    1.0 if node.get("cfa_zone", False) else 0.0,
                ])
            own_features = np.mean(own_feats, axis=0) if own_feats else np.zeros(8)

            # Neighbour features
            neighbour_ids = set()
            trade_vols = []
            tariffs = []
            for cid in agent.city_ids:
                for _, nbr, data in g.G.edges(cid, data=True):
                    if nbr not in set(agent.city_ids):
                        neighbour_ids.add(nbr)
                        if data.get("edge_type") == ConnectionType.TRADE.value:
                            trade_vols.append(data.get("volume", 0.0))
                            tariffs.append(data.get("tariff_rate", 0.0))

            nbr_feats = []
            for nid in neighbour_ids:
                node = g.G.nodes.get(nid, {})
                nbr_feats.append([
                    node.get("population", 0) / 1e7,
                    node.get("gdp_per_capita", 0) / 5000,
                    node.get("trade_openness", 0),
                    bc.get(nid, 0.0),
                ])

            bloc_status = "EXTERNAL"
            if agent.city_ids:
                bloc_status = g.G.nodes.get(agent.city_ids[0], {}).get(
                    "bloc", "EXTERNAL"
                )

            obs[aid] = AgentObservation(
                agent_id=aid,
                own_features=own_features,
                neighbour_features=np.array(nbr_feats) if nbr_feats else np.zeros((0, 4)),
                neighbour_ids=sorted(neighbour_ids),
                trade_volumes=np.array(trade_vols),
                tariff_rates=np.array(tariffs),
                bloc_status=bloc_status,
                global_metrics=global_metrics,
            )

        return obs

    def adjacency_matrix(self) -> np.ndarray:
        """Binary adjacency matrix of the current graph (for GNN input)."""
        g = self.current_graph or self.base_graph
        nodes = sorted(g.G.nodes())
        n = len(nodes)
        idx = {nid: i for i, nid in enumerate(nodes)}
        adj = np.zeros((n, n))
        for u, v in g.G.edges():
            adj[idx[u]][idx[v]] = 1.0
            adj[idx[v]][idx[u]] = 1.0
        np.fill_diagonal(adj, 1.0)  # Self-loops for GAT
        return adj

    def spectral_radius(self) -> float:
        """Spectral radius of the adjacency — relates to cascade threshold.

        From ch06 cascade damping: if spectral radius of the Jacobian
        product exceeds 1, perturbations grow exponentially.
        The adjacency spectral radius is a proxy for this.
        """
        adj = self.adjacency_matrix()
        eigenvalues = np.linalg.eigvals(adj)
        return float(np.max(np.abs(eigenvalues)))
