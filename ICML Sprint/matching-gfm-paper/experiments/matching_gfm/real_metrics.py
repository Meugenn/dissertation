from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matching import blocking_pairs, gale_shapley_from_scores
from .metrics import ranking_metrics
from .real_market import ObservedMarket


@dataclass(frozen=True)
class RealEvaluationResult:
    summary: dict[str, float | int | str]
    matching: np.ndarray


def evaluate_real_model(
    model_name: str,
    buyer_scores: np.ndarray,
    seller_scores: np.ndarray,
    market: ObservedMarket,
    eval_cutoff: int = 5,
) -> RealEvaluationResult:
    future_relevance = (market.eval_pair_matrix() > 0).astype(np.float64)
    train_relevance = (market.train_pair_matrix() > 0).astype(np.float64)
    future_metrics = ranking_metrics(buyer_scores, future_relevance, cutoff=eval_cutoff)
    train_metrics = ranking_metrics(buyer_scores, train_relevance, cutoff=eval_cutoff)

    matching = gale_shapley_from_scores(buyer_scores, seller_scores)
    predicted_blocks = blocking_pairs(matching, buyer_scores, seller_scores)
    matched_future_hits = [
        float(future_relevance[buyer, seller] > 0.0)
        for buyer, seller in enumerate(matching)
        if seller >= 0
    ]

    summary: dict[str, float | int | str] = {
        "model": model_name,
        "future_mrr": future_metrics["mrr"],
        "future_hits_at_5": future_metrics[f"hits_at_{eval_cutoff}"],
        "future_ndcg_at_5": future_metrics[f"ndcg_at_{eval_cutoff}"],
        "train_mrr": train_metrics["mrr"],
        "train_hits_at_5": train_metrics[f"hits_at_{eval_cutoff}"],
        "predicted_stability_ratio": 1.0 - (len(predicted_blocks) / float(market.num_buyers * market.num_sellers)),
        "predicted_blocking_pairs": len(predicted_blocks),
        "matching_future_hit_rate": float(np.mean(matched_future_hits)) if matched_future_hits else np.nan,
        "matched_mean_score": float(np.mean([buyer_scores[buyer, seller] for buyer, seller in enumerate(matching) if seller >= 0])),
        "buyers": market.num_buyers,
        "sellers": market.num_sellers,
        "train_edges": len(market.train_edges),
        "eval_edges": len(market.eval_edges),
    }
    return RealEvaluationResult(summary=summary, matching=matching)
