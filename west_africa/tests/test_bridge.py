"""Tests for the MARL-network bridge module.

Verifies that the Meta-MAPG framework correctly interfaces with the
West Africa trade network infrastructure.
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from west_africa.core.graph import WestAfricaGraph
from west_africa.bridge.marl_network_env import (
    TradeNetworkGame,
    TradeAgent,
    TradeAction,
    Trajectory,
)
from west_africa.bridge.meta_mapg import MetaMAPGTrainer, GradientTerms
from west_africa.bridge.cascade_damping import CascadeDampingAnalyser


def _build_graph() -> WestAfricaGraph:
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    return WestAfricaGraph.from_seed_data(data_dir)


class TestTradeNetworkGame:
    def test_game_init(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g)
        assert game.n_agents > 0
        assert game.horizon == 8

    def test_agents_cover_all_countries(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g)
        all_cities = set()
        for agent in game.agents.values():
            all_cities.update(agent.city_ids)
        # Every city in the graph should be assigned to an agent
        assert all_cities == set(g.cities.keys())

    def test_reset_returns_observations(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g)
        obs = game.reset()
        assert len(obs) == game.n_agents
        for aid, o in obs.items():
            assert o.agent_id == aid
            assert len(o.own_features) == 8

    def test_step_returns_rewards(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g)
        game.reset()
        joint_action = {aid: TradeAction.NO_ACTION for aid in game.agent_ids}
        obs, rewards, done = game.step(joint_action)
        assert len(rewards) == game.n_agents
        assert not done  # First step shouldn't be terminal

    def test_trajectory_collection(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g, horizon=4)
        tau = game.collect_trajectory(game.agents)
        assert tau.horizon == 4
        for aid in game.agent_ids:
            ret = tau.discounted_return(aid)
            assert isinstance(ret, float)

    def test_cascade_from_exit(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g)
        game.reset()
        # Nigeria exits — should trigger significant cascade
        joint_action = {aid: TradeAction.NO_ACTION for aid in game.agent_ids}
        joint_action["NGA"] = TradeAction.EXIT_BLOC
        obs, rewards, done = game.step(joint_action)
        # Nigeria should face a penalty
        assert rewards["NGA"] < 0
        # Neighbours should also be affected
        affected = [aid for aid, r in rewards.items() if r < 0 and aid != "NGA"]
        assert len(affected) > 0, "Cascade should affect neighbouring countries"

    def test_adjacency_matrix(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g)
        game.reset()
        adj = game.adjacency_matrix()
        n = len(g.cities)
        assert adj.shape == (n, n)
        # Self-loops
        assert all(adj[i, i] == 1.0 for i in range(n))
        # Symmetric
        assert np.allclose(adj, adj.T)

    def test_spectral_radius(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g)
        game.reset()
        rho = game.spectral_radius()
        assert rho > 1.0  # Dense graph should be supercritical


class TestTradeAgent:
    def test_policy_sums_to_one(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g)
        obs = game.reset()
        for aid, agent in game.agents.items():
            probs = agent.policy(obs[aid])
            assert abs(probs.sum() - 1.0) < 1e-6

    def test_log_prob_finite(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g)
        obs = game.reset()
        for aid, agent in game.agents.items():
            action = agent.sample_action(obs[aid])
            lp = agent.log_prob(obs[aid], action)
            assert np.isfinite(lp)


class TestMetaMAPGTrainer:
    def test_gradient_terms_computed(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g, horizon=3)
        agents = game.agents

        trainer = MetaMAPGTrainer(
            game=game,
            inner_steps=1,
            n_trajectories=2,
            include_term2=True,
            include_term3=True,
        )
        terms = trainer.meta_update(agents)
        assert len(terms) == game.n_agents
        for aid, gt in terms.items():
            assert gt.agent_id == aid
            assert gt.term1.shape == agents[aid].phi.shape
            assert gt.term2.shape == agents[aid].phi.shape
            assert gt.term3.shape == agents[aid].phi.shape
            assert np.isfinite(gt.meta_return)

    def test_independent_has_zero_term3(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g, horizon=3)
        agents = game.agents

        trainer = MetaMAPGTrainer(
            game=game,
            inner_steps=1,
            n_trajectories=2,
            include_term2=False,
            include_term3=False,
        )
        terms = trainer.meta_update(agents)
        for gt in terms.values():
            assert np.allclose(gt.term2, 0.0)
            assert np.allclose(gt.term3, 0.0)

    def test_term3_nonzero_for_coupled_agents(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g, horizon=3)
        agents = game.agents

        trainer = MetaMAPGTrainer(
            game=game,
            inner_steps=2,
            n_trajectories=3,
            include_term2=True,
            include_term3=True,
        )
        terms = trainer.meta_update(agents)
        # At least one agent should have nonzero Term 3
        has_nonzero_t3 = any(
            np.linalg.norm(gt.term3) > 1e-12
            for gt in terms.values()
        )
        assert has_nonzero_t3, "Term 3 should be nonzero for coupled agents"


class TestCascadeDamping:
    def test_spectral_analysis(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g)
        game.reset()
        analyser = CascadeDampingAnalyser(game, train_steps=1, eval_episodes=1)
        result = analyser.spectral_analysis()
        assert result["spectral_radius"] > 0
        assert result["n_nodes"] == 45
        assert result["n_edges"] > 0

    def test_articulation_point_analysis(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g)
        analyser = CascadeDampingAnalyser(game)
        result = analyser.articulation_point_analysis()
        assert isinstance(result["articulation_points"], list)
        assert isinstance(result["critical_agents"], list)

    def test_theoretical_bound(self):
        g = _build_graph()
        game = TradeNetworkGame(graph=g)
        analyser = CascadeDampingAnalyser(game, inner_steps=3)
        bound = analyser.theoretical_damping_bound(alpha=0.01)
        # Bound should be < 1 (damping present)
        assert 0.0 < bound < 1.0
        # With more edges, bound should be tighter
        n_edges = game.base_graph.edge_count
        expected = 1.0 / (1.0 + 0.01 * 3 * n_edges)
        assert abs(bound - expected) < 1e-10
