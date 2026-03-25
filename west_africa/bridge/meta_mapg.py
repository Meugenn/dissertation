"""Meta-MAPG trainer: implements the three-term gradient from Theorem 6.1.

The gradient of the meta-value function decomposes as:

    nabla_{phi_0^i} V^i = E[ G^i * (Term1 + Term2 + Term3) ]

where:
    Term 1: nabla_{phi_0^i} log pi^i(tau_0 | phi_0^i)
            = standard REINFORCE (current policy gradient)

    Term 2: sum_{l'=0}^{L} nabla_{phi_0^i} log pi^i(tau_{l'+1} | phi_{l'+1}^i)
            = own future learning anticipation (Meta-PG contribution)

    Term 3: sum_{l'=0}^{L} nabla_{phi_0^i} log pi^{-i}(tau_{l'+1} | phi_{l'+1}^{-i})
            = peer learning anticipation (how my action shapes others' learning)

Setting Term 3 = 0 recovers Meta-PG.
Setting Terms 2,3 = 0 recovers independent PG.
LOLA approximates Terms 1+3 for L=1 via Taylor expansion.

On the West Africa network:
    - Term 1: "lower my tariff to increase trade now"
    - Term 2: "lower my tariff now so future data shows higher trade,
               so my future policy learns to cooperate more"
    - Term 3: "lower my tariff now so Ghana observes higher trade volumes,
               so Ghana's future policy lowers their tariffs too, improving
               my environment"

This is the coupling the cascade simulator already models physically.
Meta-MAPG provides the learning-theoretic counterpart.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .marl_network_env import (
    TradeNetworkGame,
    TradeAgent,
    TradeAction,
    Trajectory,
)


@dataclass
class GradientTerms:
    """Decomposed gradient for one agent at one meta-step.

    Tracks Terms 1, 2, 3 separately for analysis and ablation.
    """
    agent_id: str
    term1: np.ndarray          # Direct policy gradient
    term2: np.ndarray          # Own future learning
    term3: np.ndarray          # Peer learning anticipation
    total: np.ndarray = field(default=None)
    meta_return: float = 0.0   # G^i evaluated at final policy

    def __post_init__(self):
        if self.total is None:
            self.total = self.term1 + self.term2 + self.term3

    @property
    def term3_magnitude_ratio(self) -> float:
        """||Term 3|| / ||Total|| — measures peer-learning contribution."""
        total_norm = np.linalg.norm(self.total)
        if total_norm < 1e-10:
            return 0.0
        return float(np.linalg.norm(self.term3) / total_norm)


class MetaMAPGTrainer:
    """Implements Meta-MAPG training on the trade network game.

    The key difference from independent PG: this trainer computes
    how phi_0^i affects phi_{l+1}^{-i} through the shared trajectories
    (the chain factorisation in the proof, Step 4).

    On the network, this means: Nigeria's initial tariff policy
    affects the trajectory data Ghana collects, which affects Ghana's
    gradient update, which affects Nigeria's future environment.

    Parameters:
        game: The stochastic game environment.
        inner_steps: L in the policy chain (number of inner-loop updates).
        n_trajectories: Trajectories per inner step for gradient estimation.
        meta_lr: Learning rate for the outer (meta) update.
        include_term2: Whether to include own-learning term (True=Meta-PG).
        include_term3: Whether to include peer-learning term (True=full Meta-MAPG).
    """

    def __init__(
        self,
        game: TradeNetworkGame,
        inner_steps: int = 3,
        n_trajectories: int = 8,
        meta_lr: float = 0.02,
        include_term2: bool = True,
        include_term3: bool = True,
    ) -> None:
        self.game = game
        self.inner_steps = inner_steps
        self.n_trajectories = n_trajectories
        self.meta_lr = meta_lr
        self.include_term2 = include_term2
        self.include_term3 = include_term3

    def meta_update(
        self, agents: dict[str, TradeAgent]
    ) -> dict[str, GradientTerms]:
        """One outer-loop meta-update for all agents.

        Implements the full Meta-MAPG gradient (eq. 6.1):
        1. Save initial parameters phi_0
        2. Run L inner-loop steps, collecting trajectories
        3. Compute three-term gradient using the trajectory chain
        4. Update phi_0 using the meta-gradient

        Returns per-agent GradientTerms for analysis.
        """
        # Save initial parameters (phi_0 for all agents)
        phi_0 = {aid: agent.phi.copy() for aid, agent in agents.items()}

        # Collect trajectory chain: tau_0, tau_1, ..., tau_L
        trajectory_chain: list[list[Trajectory]] = []
        policy_chain: list[dict[str, np.ndarray]] = [
            {aid: a.phi.copy() for aid, a in agents.items()}
        ]

        for ell in range(self.inner_steps + 1):
            # Collect n_trajectories at current joint policy
            trajectories = []
            for _ in range(self.n_trajectories):
                tau = self.game.collect_trajectory(agents)
                trajectories.append(tau)
            trajectory_chain.append(trajectories)

            # Inner-loop update (eq. 5.2): phi_{l+1} = phi_l + alpha * grad V
            if ell < self.inner_steps:
                for aid, agent in agents.items():
                    grad = self._reinforce_gradient(agent, trajectories)
                    agent.phi = agent.phi + agent.lr * grad

                policy_chain.append(
                    {aid: a.phi.copy() for aid, a in agents.items()}
                )

        # Compute meta-gradient for each agent
        gradient_terms = {}
        for aid, agent in agents.items():
            terms = self._compute_meta_gradient(
                aid, agents, phi_0, policy_chain, trajectory_chain
            )
            gradient_terms[aid] = terms

        # Restore initial params and apply meta-update
        for aid, agent in agents.items():
            agent.phi = phi_0[aid] + self.meta_lr * gradient_terms[aid].total

        return gradient_terms

    def train(
        self,
        agents: dict[str, TradeAgent],
        n_meta_steps: int = 50,
    ) -> list[dict[str, GradientTerms]]:
        """Run multiple meta-updates, tracking gradient decomposition.

        Returns history of GradientTerms per meta-step for analysis.
        """
        history = []
        for step in range(n_meta_steps):
            terms = self.meta_update(agents)
            history.append(terms)

            if step % 10 == 0:
                avg_return = np.mean([
                    t.meta_return for t in terms.values()
                ])
                avg_t3_ratio = np.mean([
                    t.term3_magnitude_ratio for t in terms.values()
                ])
                print(
                    f"Meta-step {step:3d} | "
                    f"avg return: {avg_return:+.4f} | "
                    f"avg Term3 ratio: {avg_t3_ratio:.4f}"
                )

        return history

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def _reinforce_gradient(
        self, agent: TradeAgent, trajectories: list[Trajectory]
    ) -> np.ndarray:
        """Standard REINFORCE gradient estimate for inner-loop update.

        This is the gradient used in eq. 5.2 (inner-loop update rule).
        """
        grad = np.zeros_like(agent.phi)
        for tau in trajectories:
            G_i = tau.discounted_return(agent.agent_id, self.game.gamma)
            # Score function: nabla log pi(tau | phi)
            score = self._score_function(agent, tau)
            grad += G_i * score
        grad /= max(len(trajectories), 1)
        return grad

    def _compute_meta_gradient(
        self,
        agent_id: str,
        agents: dict[str, TradeAgent],
        phi_0: dict[str, np.ndarray],
        policy_chain: list[dict[str, np.ndarray]],
        trajectory_chain: list[list[Trajectory]],
    ) -> GradientTerms:
        """Compute the three-term meta-gradient for agent i.

        Implements eq. 6.1 using the collected trajectory chain.

        The key insight: Term 3 requires computing
            nabla_{phi_0^i} log pi^{-i}(tau_{l'+1} | phi_{l'+1}^{-i})
        which captures how agent i's initial policy affects other agents'
        future policies through the shared trajectory data.
        """
        shape = phi_0[agent_id].shape

        # Term 1: nabla_{phi_0^i} log pi^i(tau_0 | phi_0^i)
        # Standard REINFORCE at step 0
        term1 = np.zeros(shape)
        final_trajs = trajectory_chain[-1]
        for tau in final_trajs:
            G_i = tau.discounted_return(agent_id, self.game.gamma)
            # Score function at initial policy
            agent_copy = TradeAgent(
                agent_id=agent_id,
                country_iso3=agents[agent_id].country_iso3,
                city_ids=agents[agent_id].city_ids,
                phi=phi_0[agent_id].copy(),
            )
            score = self._score_function(agent_copy, trajectory_chain[0][0])
            term1 += G_i * score
        term1 /= max(len(final_trajs), 1)

        # Term 2: own future learning anticipation
        term2 = np.zeros(shape)
        if self.include_term2:
            for ell_prime in range(self.inner_steps):
                # How phi_0^i affects phi_{l'+1}^i through the update chain
                # Approximation: finite-difference Jacobian
                jac = self._own_learning_jacobian(
                    agent_id, agents, phi_0, policy_chain, ell_prime
                )
                # Score at step l'+1
                for tau in trajectory_chain[ell_prime + 1]:
                    G_i = tau.discounted_return(agent_id, self.game.gamma)
                    agent_copy = TradeAgent(
                        agent_id=agent_id,
                        country_iso3=agents[agent_id].country_iso3,
                        city_ids=agents[agent_id].city_ids,
                        phi=policy_chain[ell_prime + 1][agent_id].copy(),
                    )
                    score = self._score_function(agent_copy, tau)
                    # Chain rule: nabla_{phi_0} log pi(tau | phi_{l+1})
                    #           = (d phi_{l+1} / d phi_0) . nabla_{phi_{l+1}} log pi
                    term2 += G_i * (jac @ score.reshape(-1)).reshape(shape)
                term2 /= max(len(trajectory_chain[ell_prime + 1]), 1)

        # Term 3: peer learning anticipation
        term3 = np.zeros(shape)
        if self.include_term3:
            for ell_prime in range(self.inner_steps):
                for peer_id in agents:
                    if peer_id == agent_id:
                        continue
                    # How phi_0^i affects phi_{l'+1}^{-i} through shared trajectories
                    cross_jac = self._peer_learning_jacobian(
                        agent_id, peer_id, agents, phi_0, policy_chain, ell_prime
                    )
                    for tau in trajectory_chain[ell_prime + 1]:
                        G_i = tau.discounted_return(agent_id, self.game.gamma)
                        peer_copy = TradeAgent(
                            agent_id=peer_id,
                            country_iso3=agents[peer_id].country_iso3,
                            city_ids=agents[peer_id].city_ids,
                            phi=policy_chain[ell_prime + 1][peer_id].copy(),
                        )
                        peer_score = self._score_function(peer_copy, tau)
                        # Cross-agent chain rule
                        contrib = G_i * (cross_jac @ peer_score.reshape(-1))
                        # Project back to agent i's parameter space
                        term3 += contrib[:np.prod(shape)].reshape(shape)
                    term3 /= max(len(trajectory_chain[ell_prime + 1]), 1)

        meta_return = np.mean([
            tau.discounted_return(agent_id, self.game.gamma)
            for tau in final_trajs
        ])

        return GradientTerms(
            agent_id=agent_id,
            term1=term1,
            term2=term2,
            term3=term3,
            meta_return=meta_return,
        )

    def _score_function(
        self, agent: TradeAgent, trajectory: Trajectory
    ) -> np.ndarray:
        """Compute nabla_{phi} log pi^i(tau | phi^i).

        Uses the log-derivative trick: for softmax policy,
        nabla_phi log pi(a|s,phi) = phi_encode(s) (e_a - pi(.|s,phi))
        where e_a is the one-hot action vector.
        """
        grad = np.zeros_like(agent.phi)
        for t in range(trajectory.horizon):
            obs = trajectory.observations[t].get(agent.agent_id)
            act = trajectory.actions[t].get(agent.agent_id)
            if obs is None or act is None:
                continue

            features = agent._encode_obs(obs)
            probs = agent.policy(obs)
            act_idx = list(TradeAction).index(act)

            # Softmax gradient: features^T (one_hot - probs)
            one_hot = np.zeros(len(TradeAction))
            one_hot[act_idx] = 1.0
            grad += np.outer(features, one_hot - probs)

        return grad

    def _own_learning_jacobian(
        self,
        agent_id: str,
        agents: dict[str, TradeAgent],
        phi_0: dict[str, np.ndarray],
        policy_chain: list[dict[str, np.ndarray]],
        ell: int,
    ) -> np.ndarray:
        """Approximate d phi_{l+1}^i / d phi_0^i via finite differences.

        This Jacobian captures how agent i's initial parameters
        affect its own parameters after l+1 inner-loop updates.
        """
        d = np.prod(phi_0[agent_id].shape)
        jac = np.eye(d)  # Identity at ell=0

        # For each inner step, the Jacobian compounds:
        # d phi_{l+1} / d phi_0 = prod_{k=0}^{l} (I + alpha * d^2 V / d phi^2)
        # We approximate this as identity plus accumulated learning
        alpha = agents[agent_id].lr
        for k in range(ell + 1):
            # Approximate Hessian contribution as small perturbation
            phi_k = policy_chain[k][agent_id]
            phi_k1 = policy_chain[min(k + 1, len(policy_chain) - 1)][agent_id]
            delta = (phi_k1 - phi_k).reshape(-1)
            if np.linalg.norm(delta) > 1e-10:
                # Rank-1 update approximation of the Hessian effect
                delta_norm = delta / np.linalg.norm(delta)
                jac = jac + alpha * np.outer(delta_norm, delta_norm)

        return jac

    def _peer_learning_jacobian(
        self,
        agent_id: str,
        peer_id: str,
        agents: dict[str, TradeAgent],
        phi_0: dict[str, np.ndarray],
        policy_chain: list[dict[str, np.ndarray]],
        ell: int,
    ) -> np.ndarray:
        """Approximate d phi_{l+1}^{-i} / d phi_0^i via finite differences.

        THIS IS THE KEY TERM THAT DISTINGUISHES META-MAPG.

        On the network: how does Nigeria's initial tariff policy affect
        Ghana's policy after l+1 inner-loop updates? The mechanism:

        1. Nigeria's phi_0^i determines Nigeria's actions in tau_0
        2. Nigeria's actions change the trade graph (lower tariffs →
           higher trade volumes on Nigeria-Ghana edges)
        3. Ghana observes higher trade volumes in tau_0
        4. Ghana's inner-loop gradient uses tau_0, so Ghana's phi_1^{-i}
           changes in response to Nigeria's phi_0^i

        This is the shared-trajectory coupling from Step 4 of the proof.
        """
        d_i = np.prod(phi_0[agent_id].shape)
        d_j = np.prod(phi_0[peer_id].shape)

        # Cross-agent Jacobian: starts at zero (no direct coupling)
        # and accumulates through shared trajectory effects
        cross_jac = np.zeros((d_j, d_i))

        for k in range(ell + 1):
            # The coupling comes through the trajectory:
            # phi_0^i -> tau_k (via pi^i's effect on joint trajectory)
            #         -> grad^{-i}(tau_k) (peer's gradient uses this trajectory)
            #         -> phi_{k+1}^{-i}
            #
            # Approximate: perturbation in phi_0^i changes the trajectory
            # distribution, which changes the peer's gradient.

            # Estimate coupling strength from policy chain
            phi_i_k = policy_chain[k][agent_id]
            phi_j_k = policy_chain[k][peer_id]
            phi_j_k1 = policy_chain[min(k + 1, len(policy_chain) - 1)][peer_id]

            peer_delta = (phi_j_k1 - phi_j_k).reshape(-1)

            # The coupling is proportional to:
            # - agent i's influence on shared state (betweenness centrality)
            # - the learning rate
            # - the trajectory overlap (shared edges in the network)
            coupling = self._network_coupling_strength(agent_id, peer_id)
            alpha_j = agents[peer_id].lr

            if np.linalg.norm(peer_delta) > 1e-10:
                peer_dir = peer_delta / np.linalg.norm(peer_delta)
                # Agent i's influence direction on shared trajectories
                phi_i_delta = (
                    policy_chain[min(k + 1, len(policy_chain) - 1)][agent_id]
                    - phi_i_k
                ).reshape(-1)
                if np.linalg.norm(phi_i_delta) > 1e-10:
                    i_dir = phi_i_delta / np.linalg.norm(phi_i_delta)
                else:
                    i_dir = np.zeros(d_i)

                # Rank-1 coupling: peer's learning direction × agent i's influence
                cross_jac += coupling * alpha_j * np.outer(peer_dir, i_dir[:d_i])

        return cross_jac

    def _network_coupling_strength(
        self, agent_id: str, peer_id: str
    ) -> float:
        """Estimate coupling strength between two agents via network structure.

        The coupling is mediated by shared edges in the trade graph:
        more trade between Nigeria and Ghana → stronger coupling →
        larger Term 3 contribution.

        Normalised by agent i's OWN total trade (not network total),
        giving the fraction of agent i's trade that flows to/from peer j.
        This is the local perspective: "how much does peer j matter to me?"

        This is where the network structure enters the MARL gradient:
        the topology determines which agents affect each other's learning.
        """
        g = self.game.current_graph or self.game.base_graph
        agent_cities = set(self.game.agents[agent_id].city_ids)
        peer_cities = set(self.game.agents[peer_id].city_ids)

        # Count shared trade edges and total volume
        shared_volume = 0.0
        shared_edges = 0
        for u, v, data in g.G.edges(data=True):
            if data.get("edge_type") == "TRADE":
                if (u in agent_cities and v in peer_cities) or \
                   (v in agent_cities and u in peer_cities):
                    shared_volume += data.get("volume", 0.0)
                    shared_edges += 1

        # Normalise by agent i's total trade volume (local perspective)
        agent_volume = 0.0
        for u, v, data in g.G.edges(data=True):
            if data.get("edge_type") == "TRADE":
                if u in agent_cities or v in agent_cities:
                    agent_volume += data.get("volume", 0.0)

        if agent_volume < 1e-10:
            # Fallback: use edge connectivity (1 if any shared edge, 0 otherwise)
            return 1.0 if shared_edges > 0 else 0.0

        # Coupling = fraction of agent i's trade going to peer j, bounded [0, 1]
        return min(shared_volume / agent_volume, 1.0)


def run_ablation(
    game: TradeNetworkGame,
    n_meta_steps: int = 30,
    inner_steps: int = 3,
) -> dict[str, list[dict[str, GradientTerms]]]:
    """Run the three ablation variants for comparison.

    Returns histories for:
        - 'independent': Term 1 only (standard REINFORCE)
        - 'meta_pg':     Terms 1+2 (Meta-PG, no peer modelling)
        - 'meta_mapg':   Terms 1+2+3 (full Meta-MAPG)

    This directly tests the prediction from ch06: Meta-MAPG should
    outperform both baselines on the network game because it captures
    the cross-agent coupling that cascade dynamics create.
    """
    results = {}

    configs = {
        "independent": (False, False),
        "meta_pg": (True, False),
        "meta_mapg": (True, True),
    }

    for name, (t2, t3) in configs.items():
        print(f"\n{'='*60}")
        print(f"Running: {name} (Term2={t2}, Term3={t3})")
        print(f"{'='*60}")

        # Fresh agents for each run
        agents = copy.deepcopy(game.agents)
        trainer = MetaMAPGTrainer(
            game=game,
            inner_steps=inner_steps,
            include_term2=t2,
            include_term3=t3,
        )
        history = trainer.train(agents, n_meta_steps=n_meta_steps)
        results[name] = history

    return results
