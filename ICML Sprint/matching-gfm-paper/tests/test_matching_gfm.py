from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from matching_gfm.compact_model import CompactModelConfig, fit_compact_graph_matcher  # noqa: E402
from matching_gfm.hm_local import load_hm_local_market  # noqa: E402
from matching_gfm.matching import blocking_pairs, gale_shapley_from_scores, sinkhorn  # noqa: E402
from matching_gfm.real_market import ObservedMarket  # noqa: E402
from matching_gfm.synthetic_market import generate_synthetic_market  # noqa: E402
from matching_gfm.synthetic_market import TemporalEdge  # noqa: E402


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

    def test_compact_model_supports_asymmetric_feature_dims(self) -> None:
        market = ObservedMarket(
            buyer_ids=("b0", "b1", "b2"),
            seller_ids=("s0", "s1", "s2"),
            buyer_features=np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float64),
            seller_features=np.array([[1.0, 0.0, 0.0], [0.4, 0.5, 0.1], [0.0, 1.0, 0.0]], dtype=np.float64),
            train_edges=(
                TemporalEdge(0, 0, 0, 1.0),
                TemporalEdge(1, 1, 0, 2.0),
                TemporalEdge(2, 2, 1, 3.0),
            ),
            eval_edges=(TemporalEdge(0, 1, 0, 4.0),),
            edge_type_names=("buy", "sell"),
            edge_type_weights=np.array([1.0, 1.0], dtype=np.float64),
            train_cutoff=3.0,
            total_horizon=4.0,
            source_name="test",
        )
        result = fit_compact_graph_matcher(market, config=CompactModelConfig(hidden_dim=2, maxiter=5, random_seed=2))
        self.assertEqual(result.buyer_scores.shape, (3, 3))
        self.assertEqual(result.seller_scores.shape, (3, 3))
        self.assertTrue(np.isfinite(result.fit_loss))

    def test_hm_local_loader_builds_observed_market(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            pd.DataFrame(
                {
                    "customer_id": ["c1", "c2", "c3"],
                    "FN": [1, 0, 1],
                    "Active": [1, 1, 0],
                    "club_member_status": ["ACTIVE", "ACTIVE", "PRE-CREATE"],
                    "fashion_news_frequency": ["Regularly", "NONE", "Monthly"],
                    "age": [25, 35, 29],
                }
            ).to_csv(data_dir / "customers.csv", index=False)
            pd.DataFrame(
                {
                    "article_id": ["a1", "a2", "a3"],
                    "product_type_no": [1, 2, 3],
                    "graphical_appearance_no": [10, 11, 12],
                    "colour_group_code": [100, 110, 120],
                    "perceived_colour_value_id": [5, 6, 7],
                    "department_no": [20, 20, 30],
                    "section_no": [1, 2, 3],
                    "garment_group_no": [1000, 1001, 1002],
                    "index_group_name": ["Ladieswear", "Menswear", "Ladieswear"],
                    "product_group_name": ["Garment Upper body", "Shoes", "Accessories"],
                }
            ).to_csv(data_dir / "articles.csv", index=False)
            pd.DataFrame(
                {
                    "t_dat": [
                        "2020-09-20",
                        "2020-09-21",
                        "2020-09-22",
                        "2020-09-23",
                        "2020-09-24",
                        "2020-09-25",
                    ],
                    "customer_id": ["c1", "c1", "c2", "c2", "c3", "c3"],
                    "article_id": ["a1", "a2", "a1", "a2", "a1", "a3"],
                    "price": [0.1, 0.2, 0.12, 0.21, 0.11, 0.31],
                    "sales_channel_id": [1, 2, 1, 2, 1, 2],
                }
            ).to_csv(data_dir / "transactions_train.csv", index=False)

            market = load_hm_local_market(data_dir, max_rows=None, min_customer_transactions=1, min_article_transactions=1)
            self.assertEqual(market.source_name, "hm_local")
            self.assertEqual(market.num_buyers, 3)
            self.assertEqual(market.num_sellers, 3)
            self.assertGreater(len(market.train_edges), 0)
            self.assertGreater(len(market.eval_edges), 0)
            self.assertEqual(market.edge_type_names, ("channel_1", "channel_2"))


if __name__ == "__main__":
    unittest.main()
