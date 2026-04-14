from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from matching_gfm.compact_model import CompactModelConfig, fit_compact_graph_matcher  # noqa: E402
from matching_gfm.matching import blocking_pairs, gale_shapley_from_scores, sinkhorn  # noqa: E402
from matching_gfm.synthetic_market import generate_synthetic_market  # noqa: E402


class MatchingGFMTests(unittest.TestCase):
    def test_sinkhorn_is_nearly_doubly_stochastic(self) -> None:
        scores = np.array([[2.0, 0.0, 1.0], [1.0, 3.0, 0.0], [0.5, 1.5, 2.0]], dtype=np.float64)
        matrix = sinkhorn(scores, temperature=0.4, iterations=80)
        self.assertTrue(np.allclose(matrix.sum(axis=1), np.ones(3), atol=1e-4))
        self.assertTrue(np.allclose(matrix.sum(axis=0), np.ones(3), atol=1e-4))

    def test_gale_shapley_is_stable_on_simple_instance(self) -> None:
        buyer_scores = np.array([[3.0, 2.0], [2.0, 3.0]], dtype=np.float64)
        seller_scores = np.array([[3.0, 2.0], [2.0, 3.0]], dtype=np.float64)
        matching = gale_shapley_from_scores(buyer_scores, seller_scores)
        self.assertTrue(np.array_equal(matching, np.array([0, 1])))
        self.assertEqual(blocking_pairs(matching, buyer_scores, seller_scores), [])

    def test_synthetic_market_has_nonempty_train_and_eval_edges(self) -> None:
        market = generate_synthetic_market(num_buyers=6, num_sellers=6, num_steps=20, interactions_per_step=4, seed=7)
        self.assertEqual(market.true_buyer_utilities.shape, (6, 6))
        self.assertEqual(market.true_seller_utilities.shape, (6, 6))
        self.assertGreater(len(market.train_edges), 0)
        self.assertGreater(len(market.eval_edges), 0)
        self.assertEqual(market.train_pair_tensor().shape, (6, 6, 3))

    def test_compact_model_smoke_fit(self) -> None:
        market = generate_synthetic_market(num_buyers=5, num_sellers=5, num_steps=18, interactions_per_step=3, seed=3)
        result = fit_compact_graph_matcher(
            market,
            config=CompactModelConfig(hidden_dim=2, maxiter=5, random_seed=3),
        )
        self.assertEqual(result.buyer_scores.shape, (5, 5))
        self.assertEqual(result.seller_scores.shape, (5, 5))
        self.assertEqual(result.soft_matching.shape, (5, 5))
        self.assertTrue(np.isfinite(result.objective))


if __name__ == "__main__":
    unittest.main()
