from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .synthetic_market import SyntheticMarket


def _normalize(matrix: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(matrix))
    if scale <= 1e-12:
        return matrix.copy()
    return matrix / scale


def _pair_feature_block(buyer_vec: np.ndarray, seller_vec: np.ndarray) -> np.ndarray:
    shared_dim = min(buyer_vec.size, seller_vec.size)
    interaction = buyer_vec[:shared_dim] * seller_vec[:shared_dim]
    summary = np.array(
        [
            float(np.linalg.norm(buyer_vec)),
            float(np.linalg.norm(seller_vec)),
            float(np.dot(buyer_vec[:shared_dim], seller_vec[:shared_dim])) if shared_dim > 0 else 0.0,
        ],
        dtype=np.float64,
    )
    return np.concatenate([buyer_vec, seller_vec, interaction, summary])


def _build_pair_features(market: SyntheticMarket, half_life: float = 10.0) -> np.ndarray:
    tensor = market.train_pair_tensor(half_life=half_life)
    pair_strength = np.tensordot(tensor, market.edge_type_weights, axes=([2], [0]))
    buyer_totals = tensor.sum(axis=1)
    seller_totals = tensor.sum(axis=0)
    buyer_thickness = (pair_strength > 0).sum(axis=1, keepdims=True).astype(np.float64)
    seller_thickness = (pair_strength > 0).sum(axis=0, keepdims=True).T.astype(np.float64)

    rows: list[np.ndarray] = []
    for buyer in range(market.num_buyers):
        for seller in range(market.num_sellers):
            rows.append(
                np.concatenate(
                    [
                        _pair_feature_block(market.buyer_features[buyer], market.seller_features[seller]),
                        tensor[buyer, seller],
                        buyer_totals[buyer],
                        seller_totals[seller],
                        buyer_thickness[buyer],
                        seller_thickness[seller],
                    ]
                )
            )
    return np.asarray(rows, dtype=np.float64)


@dataclass(frozen=True)
class PointwiseBaselineResult:
    buyer_scores: np.ndarray
    seller_scores: np.ndarray
    train_mse: float


def fit_pointwise_gbm(
    market: SyntheticMarket,
    half_life: float = 10.0,
    random_state: int = 0,
) -> PointwiseBaselineResult:
    features = _build_pair_features(market, half_life=half_life)
    target = market.train_pair_matrix(half_life=half_life).reshape(-1)
    regressor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=4,
        max_iter=250,
        l2_regularization=0.02,
        random_state=random_state,
    )
    regressor.fit(features, target)
    predictions = regressor.predict(features).reshape(market.num_buyers, market.num_sellers)
    train_mse = float(np.mean((predictions.reshape(-1) - target) ** 2))
    buyer_scores = _normalize(predictions)
    seller_scores = buyer_scores.T.copy()
    return PointwiseBaselineResult(buyer_scores=buyer_scores, seller_scores=seller_scores, train_mse=train_mse)
