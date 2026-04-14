from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TemporalEdge:
    buyer: int
    seller: int
    edge_type: int
    time: float


@dataclass(frozen=True)
class SyntheticMarket:
    buyer_features: np.ndarray
    seller_features: np.ndarray
    true_buyer_utilities: np.ndarray
    true_seller_utilities: np.ndarray
    train_edges: tuple[TemporalEdge, ...]
    eval_edges: tuple[TemporalEdge, ...]
    edge_type_names: tuple[str, ...]
    edge_type_weights: np.ndarray
    train_cutoff: float
    total_horizon: float

    @property
    def num_buyers(self) -> int:
        return int(self.buyer_features.shape[0])

    @property
    def num_sellers(self) -> int:
        return int(self.seller_features.shape[0])

    @property
    def num_edge_types(self) -> int:
        return len(self.edge_type_names)

    def pair_tensor(
        self,
        edges: tuple[TemporalEdge, ...],
        reference_time: float | None = None,
        half_life: float = 10.0,
    ) -> np.ndarray:
        reference = self.total_horizon if reference_time is None else reference_time
        tensor = np.zeros((self.num_buyers, self.num_sellers, self.num_edge_types), dtype=np.float64)
        for edge in edges:
            age = max(reference - edge.time, 0.0)
            decay = 0.5 ** (age / max(half_life, 1e-6))
            tensor[edge.buyer, edge.seller, edge.edge_type] += decay
        return tensor

    def weighted_pair_matrix(
        self,
        edges: tuple[TemporalEdge, ...],
        reference_time: float | None = None,
        half_life: float = 10.0,
    ) -> np.ndarray:
        tensor = self.pair_tensor(edges=edges, reference_time=reference_time, half_life=half_life)
        return np.tensordot(tensor, self.edge_type_weights, axes=([2], [0]))

    def train_pair_tensor(self, half_life: float = 10.0) -> np.ndarray:
        return self.pair_tensor(self.train_edges, reference_time=self.train_cutoff, half_life=half_life)

    def eval_pair_tensor(self, half_life: float = 10.0) -> np.ndarray:
        return self.pair_tensor(self.eval_edges, reference_time=self.total_horizon, half_life=half_life)

    def train_pair_matrix(self, half_life: float = 10.0) -> np.ndarray:
        return self.weighted_pair_matrix(self.train_edges, reference_time=self.train_cutoff, half_life=half_life)

    def eval_pair_matrix(self, half_life: float = 10.0) -> np.ndarray:
        return self.weighted_pair_matrix(self.eval_edges, reference_time=self.total_horizon, half_life=half_life)


def _softmax_sample(logits: np.ndarray, rng: np.random.Generator) -> int:
    centered = logits - np.max(logits)
    probs = np.exp(centered)
    probs /= np.clip(np.sum(probs), 1e-12, None)
    return int(rng.choice(logits.size, p=probs))


def _row_standardize(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    scale = matrix.std(axis=1, keepdims=True) + 1e-6
    return centered / scale


def generate_synthetic_market(
    num_buyers: int = 12,
    num_sellers: int = 12,
    feature_dim: int = 4,
    latent_dim: int = 3,
    num_steps: int = 48,
    interactions_per_step: int = 6,
    train_fraction: float = 0.7,
    choice_temperature: float = 0.55,
    edge_noise: float = 0.2,
    seed: int = 0,
) -> SyntheticMarket:
    rng = np.random.default_rng(seed)
    buyer_features = rng.normal(size=(num_buyers, feature_dim))
    seller_features = rng.normal(size=(num_sellers, feature_dim))

    buyer_pref_proj = rng.normal(scale=0.9, size=(feature_dim, latent_dim))
    buyer_attr_proj = rng.normal(scale=0.9, size=(feature_dim, latent_dim))
    seller_pref_proj = rng.normal(scale=0.9, size=(feature_dim, latent_dim))
    seller_attr_proj = rng.normal(scale=0.9, size=(feature_dim, latent_dim))

    buyer_pref = buyer_features @ buyer_pref_proj
    buyer_attr = buyer_features @ buyer_attr_proj
    seller_pref = seller_features @ seller_pref_proj
    seller_attr = seller_features @ seller_attr_proj

    buyer_bias = buyer_features[:, :1] @ np.ones((1, num_sellers))
    seller_bias = seller_features[:, :1] @ np.ones((1, num_buyers))

    true_buyer_utilities = buyer_pref @ seller_attr.T + 0.15 * buyer_bias
    true_seller_utilities = seller_pref @ buyer_attr.T + 0.15 * seller_bias

    true_buyer_utilities = _row_standardize(true_buyer_utilities)
    true_seller_utilities = _row_standardize(true_seller_utilities)

    edge_type_names = ("view", "message", "order")
    edge_type_weights = np.array([0.25, 0.55, 1.0], dtype=np.float64)

    train_cutoff = float(math.floor(num_steps * train_fraction))
    train_edges: list[TemporalEdge] = []
    eval_edges: list[TemporalEdge] = []
    seller_exposure = np.zeros(num_sellers, dtype=np.float64)

    for step in range(num_steps):
        seasonal = 0.2 * math.sin(2.0 * math.pi * step / max(num_steps, 1))
        active_buyers = rng.choice(num_buyers, size=min(num_buyers, interactions_per_step), replace=False)
        for buyer in active_buyers:
            logits = true_buyer_utilities[buyer] / max(choice_temperature, 1e-3)
            logits = logits + 0.12 * seller_exposure + seasonal
            seller = _softmax_sample(logits, rng)
            joint_score = 0.6 * true_buyer_utilities[buyer, seller] + 0.4 * true_seller_utilities[seller, buyer]
            view_edge = TemporalEdge(buyer=buyer, seller=seller, edge_type=0, time=float(step))
            target = train_edges if step < train_cutoff else eval_edges
            target.append(view_edge)

            message_prob = 1.0 / (1.0 + np.exp(-(joint_score - 0.05 + rng.normal(scale=edge_noise))))
            order_prob = 1.0 / (1.0 + np.exp(-(joint_score + 0.15 + rng.normal(scale=edge_noise))))
            if rng.random() < message_prob:
                target.append(TemporalEdge(buyer=buyer, seller=seller, edge_type=1, time=float(step) + 0.05))
            if rng.random() < order_prob:
                target.append(TemporalEdge(buyer=buyer, seller=seller, edge_type=2, time=float(step) + 0.1))
                seller_exposure[seller] += 1.0

    return SyntheticMarket(
        buyer_features=buyer_features.astype(np.float64),
        seller_features=seller_features.astype(np.float64),
        true_buyer_utilities=true_buyer_utilities.astype(np.float64),
        true_seller_utilities=true_seller_utilities.astype(np.float64),
        train_edges=tuple(train_edges),
        eval_edges=tuple(eval_edges),
        edge_type_names=edge_type_names,
        edge_type_weights=edge_type_weights,
        train_cutoff=train_cutoff,
        total_horizon=float(num_steps),
    )
