from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matching import blocking_pairs, gale_shapley_from_scores
from .synthetic_market import SyntheticMarket


def _ranking_positions(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    positions = np.empty_like(order)
    positions[order] = np.arange(order.size)
    return positions


def kendall_tau_row(pred_scores: np.ndarray, true_scores: np.ndarray) -> float:
    pred_ranks = _ranking_positions(pred_scores)
    true_order = np.argsort(-true_scores)
    inversions = 0
    total = 0
    for i in range(true_order.size):
        for j in range(i + 1, true_order.size):
            total += 1
            left = true_order[i]
            right = true_order[j]
            if pred_ranks[left] > pred_ranks[right]:
                inversions += 1
    if total == 0:
        return 1.0
    return 1.0 - (2.0 * inversions / total)


def mean_kendall_tau(pred_scores: np.ndarray, true_scores: np.ndarray) -> float:
    return float(np.mean([kendall_tau_row(pred_scores[i], true_scores[i]) for i in range(pred_scores.shape[0])]))


def binary_relevance_from_scores(scores: np.ndarray, top_k: int = 3) -> np.ndarray:
    relevance = np.zeros_like(scores, dtype=np.float64)
    k = min(top_k, scores.shape[1])
    top_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    row_indices = np.arange(scores.shape[0])[:, None]
    relevance[row_indices, top_indices] = 1.0
    return relevance


def _dcg(relevance: np.ndarray) -> float:
    gains = (2.0**relevance - 1.0) / np.log2(np.arange(relevance.size, dtype=np.float64) + 2.0)
    return float(np.sum(gains))


def ranking_metrics(pred_scores: np.ndarray, relevance: np.ndarray, cutoff: int = 5) -> dict[str, float]:
    num_rows = pred_scores.shape[0]
    rr_values: list[float] = []
    hits_values: list[float] = []
    ndcg_values: list[float] = []
    for row in range(num_rows):
        if np.sum(relevance[row]) <= 0:
            continue
        order = np.argsort(-pred_scores[row])
        ranked_relevance = relevance[row, order]
        hit_positions = np.flatnonzero(ranked_relevance > 0)
        rr_values.append(0.0 if hit_positions.size == 0 else 1.0 / float(hit_positions[0] + 1))
        hits_values.append(float(np.any(ranked_relevance[:cutoff] > 0)))
        actual = ranked_relevance[:cutoff]
        ideal = np.sort(relevance[row])[::-1][:cutoff]
        denom = _dcg(ideal)
        ndcg_values.append(0.0 if denom <= 0 else _dcg(actual) / denom)
    if not rr_values:
        return {"mrr": np.nan, f"hits_at_{cutoff}": np.nan, f"ndcg_at_{cutoff}": np.nan}
    return {
        "mrr": float(np.mean(rr_values)),
        f"hits_at_{cutoff}": float(np.mean(hits_values)),
        f"ndcg_at_{cutoff}": float(np.mean(ndcg_values)),
    }


def matching_welfare(matching: np.ndarray, buyer_utils: np.ndarray, seller_utils: np.ndarray) -> float:
    welfare = 0.0
    for buyer, seller in enumerate(matching):
        if seller >= 0:
            welfare += buyer_utils[buyer, seller] + seller_utils[seller, buyer]
    return float(welfare)


@dataclass(frozen=True)
class EvaluationResult:
    summary: dict[str, float | int | str]
    matching: np.ndarray


def evaluate_model(
    model_name: str,
    buyer_scores: np.ndarray,
    seller_scores: np.ndarray,
    market: SyntheticMarket,
    latent_top_k: int = 1,
    eval_cutoff: int = 5,
) -> EvaluationResult:
    matching = gale_shapley_from_scores(buyer_scores, seller_scores)
    blocks = blocking_pairs(matching, market.true_buyer_utilities, market.true_seller_utilities)
    stability_ratio = 1.0 - (len(blocks) / float(market.num_buyers * market.num_sellers))

    latent_relevance = binary_relevance_from_scores(market.true_buyer_utilities, top_k=latent_top_k)
    future_relevance = (market.eval_pair_matrix() > 0).astype(np.float64)

    buyer_metrics = ranking_metrics(buyer_scores, latent_relevance, cutoff=eval_cutoff)
    future_metrics = ranking_metrics(buyer_scores, future_relevance, cutoff=eval_cutoff)

    summary: dict[str, float | int | str] = {
        "model": model_name,
        "buyer_tau": mean_kendall_tau(buyer_scores, market.true_buyer_utilities),
        "seller_tau": mean_kendall_tau(seller_scores, market.true_seller_utilities),
        "stability_ratio_true": stability_ratio,
        "blocking_pairs_true": len(blocks),
        "social_welfare_true": matching_welfare(matching, market.true_buyer_utilities, market.true_seller_utilities),
        **buyer_metrics,
        **{f"future_{key}": value for key, value in future_metrics.items()},
    }
    return EvaluationResult(summary=summary, matching=matching)
