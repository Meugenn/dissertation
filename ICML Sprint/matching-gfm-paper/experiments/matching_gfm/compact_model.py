from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .matching import sinkhorn
from .synthetic_market import SyntheticMarket


def _normalize(matrix: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(matrix))
    if scale <= 1e-12:
        return matrix.copy()
    return matrix / scale


@dataclass(frozen=True)
class CompactModelConfig:
    hidden_dim: int = 3
    half_life: float = 10.0
    sinkhorn_temperature: float = 0.35
    sinkhorn_iterations: int = 50
    stability_weight: float = 0.45
    welfare_weight: float = 0.1
    regularization: float = 1e-3
    maxiter: int = 50
    random_seed: int = 0


@dataclass(frozen=True)
class CompactModelResult:
    buyer_scores: np.ndarray
    seller_scores: np.ndarray
    soft_matching: np.ndarray
    fit_loss: float
    stability_loss: float
    welfare_term: float
    objective: float
    success: bool
    iterations: int


def _prepare_market(market: SyntheticMarket, half_life: float) -> dict[str, np.ndarray]:
    history = market.train_pair_tensor(half_life=half_life)
    target = market.train_pair_matrix(half_life=half_life)
    target = _normalize(target)
    buyer_degree = history.sum(axis=(1, 2), keepdims=False)[:, None]
    seller_degree = history.sum(axis=(0, 2), keepdims=False)[:, None]
    buyer_messages = np.stack([history[:, :, edge_type] @ market.seller_features for edge_type in range(market.num_edge_types)], axis=0)
    seller_messages = np.stack([history[:, :, edge_type].T @ market.buyer_features for edge_type in range(market.num_edge_types)], axis=0)
    buyer_messages /= np.clip(buyer_degree[None, :, :], 1.0, None)
    seller_messages /= np.clip(seller_degree[None, :, :], 1.0, None)
    return {
        "history": history,
        "target": target,
        "buyer_messages": buyer_messages,
        "seller_messages": seller_messages,
    }


def _unpack(
    theta: np.ndarray,
    buyer_dim: int,
    seller_dim: int,
    hidden_dim: int,
    num_edge_types: int,
) -> dict[str, np.ndarray]:
    cursor = 0

    def take(size: int) -> np.ndarray:
        nonlocal cursor
        values = theta[cursor : cursor + size]
        cursor += size
        return values

    w_b_attr = take(buyer_dim * hidden_dim).reshape(buyer_dim, hidden_dim)
    w_s_attr = take(seller_dim * hidden_dim).reshape(seller_dim, hidden_dim)
    w_b_ctx = take(seller_dim * hidden_dim).reshape(seller_dim, hidden_dim)
    w_s_ctx = take(buyer_dim * hidden_dim).reshape(buyer_dim, hidden_dim)
    alpha_b = take(num_edge_types)
    alpha_s = take(num_edge_types)
    history_b = take(num_edge_types)
    history_s = take(num_edge_types)
    scale_b = take(hidden_dim)
    scale_s = take(hidden_dim)
    return {
        "w_b_attr": w_b_attr,
        "w_s_attr": w_s_attr,
        "w_b_ctx": w_b_ctx,
        "w_s_ctx": w_s_ctx,
        "alpha_b": alpha_b,
        "alpha_s": alpha_s,
        "history_b": history_b,
        "history_s": history_s,
        "scale_b": scale_b,
        "scale_s": scale_s,
    }


def _score_matrices(
    theta: np.ndarray,
    market: SyntheticMarket,
    prepared: dict[str, np.ndarray],
    config: CompactModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = _unpack(
        theta=theta,
        buyer_dim=market.buyer_features.shape[1],
        seller_dim=market.seller_features.shape[1],
        hidden_dim=config.hidden_dim,
        num_edge_types=market.num_edge_types,
    )
    buyer_context = np.tensordot(params["alpha_b"], prepared["buyer_messages"], axes=([0], [0]))
    seller_context = np.tensordot(params["alpha_s"], prepared["seller_messages"], axes=([0], [0]))

    buyer_embedding = np.tanh(market.buyer_features @ params["w_b_attr"] + buyer_context @ params["w_b_ctx"])
    seller_embedding = np.tanh(market.seller_features @ params["w_s_attr"] + seller_context @ params["w_s_ctx"])

    buyer_history = np.tensordot(prepared["history"], params["history_b"], axes=([2], [0]))
    seller_history = np.tensordot(prepared["history"], params["history_s"], axes=([2], [0])).T

    buyer_scores = (buyer_embedding * params["scale_b"]) @ seller_embedding.T + buyer_history
    seller_scores = (seller_embedding * params["scale_s"]) @ buyer_embedding.T + seller_history
    combined_scores = 0.5 * (buyer_scores + seller_scores.T)
    return buyer_scores, seller_scores, combined_scores


def fit_compact_graph_matcher(
    market: SyntheticMarket,
    config: CompactModelConfig | None = None,
) -> CompactModelResult:
    cfg = CompactModelConfig() if config is None else config
    prepared = _prepare_market(market, half_life=cfg.half_life)
    buyer_dim = market.buyer_features.shape[1]
    seller_dim = market.seller_features.shape[1]
    theta_size = (
        2 * buyer_dim * cfg.hidden_dim
        + 2 * seller_dim * cfg.hidden_dim
        + 4 * market.num_edge_types
        + 2 * cfg.hidden_dim
    )
    rng = np.random.default_rng(cfg.random_seed)
    theta0 = rng.normal(scale=0.15, size=theta_size)

    def objective(theta: np.ndarray) -> float:
        buyer_scores, seller_scores, combined_scores = _score_matrices(theta, market, prepared, cfg)
        target = prepared["target"]
        fit_loss = np.mean((buyer_scores - target) ** 2) + np.mean((seller_scores.T - target) ** 2)
        soft_matching = sinkhorn(
            combined_scores,
            temperature=cfg.sinkhorn_temperature,
            iterations=cfg.sinkhorn_iterations,
        )
        matched_buyer_values = np.sum(soft_matching * buyer_scores, axis=1, keepdims=True)
        matched_seller_values = np.sum(soft_matching * seller_scores.T, axis=0, keepdims=True)
        blocking = np.maximum(buyer_scores - matched_buyer_values, 0.0) * np.maximum(
            seller_scores.T - matched_seller_values,
            0.0,
        )
        stability_loss = float(np.mean(blocking))
        welfare_term = -float(np.mean(soft_matching * (buyer_scores + seller_scores.T)))
        ridge = cfg.regularization * float(np.mean(theta**2))
        return fit_loss + cfg.stability_weight * stability_loss + cfg.welfare_weight * welfare_term + ridge

    result = minimize(objective, theta0, method="L-BFGS-B", options={"maxiter": cfg.maxiter})
    buyer_scores, seller_scores, combined_scores = _score_matrices(result.x, market, prepared, cfg)
    soft_matching = sinkhorn(combined_scores, temperature=cfg.sinkhorn_temperature, iterations=cfg.sinkhorn_iterations)
    matched_buyer_values = np.sum(soft_matching * buyer_scores, axis=1, keepdims=True)
    matched_seller_values = np.sum(soft_matching * seller_scores.T, axis=0, keepdims=True)
    blocking = np.maximum(buyer_scores - matched_buyer_values, 0.0) * np.maximum(
        seller_scores.T - matched_seller_values,
        0.0,
    )
    fit_loss = float(np.mean((buyer_scores - prepared["target"]) ** 2) + np.mean((seller_scores.T - prepared["target"]) ** 2))
    stability_loss = float(np.mean(blocking))
    welfare_term = -float(np.mean(soft_matching * (buyer_scores + seller_scores.T)))
    return CompactModelResult(
        buyer_scores=_normalize(buyer_scores),
        seller_scores=_normalize(seller_scores),
        soft_matching=soft_matching,
        fit_loss=fit_loss,
        stability_loss=stability_loss,
        welfare_term=welfare_term,
        objective=float(result.fun),
        success=bool(result.success),
        iterations=int(result.nit),
    )
