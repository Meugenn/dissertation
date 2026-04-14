from __future__ import annotations

import numpy as np


def sinkhorn(scores: np.ndarray, temperature: float = 0.4, iterations: int = 60) -> np.ndarray:
    logits = (scores - np.max(scores)) / max(temperature, 1e-6)
    matrix = np.exp(np.clip(logits, -50.0, 50.0))
    matrix /= np.clip(np.sum(matrix), 1e-12, None)
    for _ in range(iterations):
        matrix /= np.clip(matrix.sum(axis=1, keepdims=True), 1e-12, None)
        matrix /= np.clip(matrix.sum(axis=0, keepdims=True), 1e-12, None)
    return matrix


def _preference_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, axis=1)


def gale_shapley_from_scores(buyer_scores: np.ndarray, seller_scores: np.ndarray) -> np.ndarray:
    buyer_prefs = _preference_order(buyer_scores)
    seller_prefs = _preference_order(seller_scores)
    num_buyers, num_sellers = buyer_scores.shape
    buyer_matches = np.full(num_buyers, -1, dtype=int)
    seller_matches = np.full(num_sellers, -1, dtype=int)
    next_choice = np.zeros(num_buyers, dtype=int)
    seller_rank = np.empty((num_sellers, num_buyers), dtype=int)

    for seller in range(num_sellers):
        seller_rank[seller, seller_prefs[seller]] = np.arange(num_buyers, dtype=int)

    free_buyers = list(range(num_buyers))
    while free_buyers:
        buyer = free_buyers.pop()
        if next_choice[buyer] >= num_sellers:
            continue
        seller = int(buyer_prefs[buyer, next_choice[buyer]])
        next_choice[buyer] += 1
        current = seller_matches[seller]
        if current == -1:
            seller_matches[seller] = buyer
            buyer_matches[buyer] = seller
            continue
        if seller_rank[seller, buyer] < seller_rank[seller, current]:
            seller_matches[seller] = buyer
            buyer_matches[buyer] = seller
            buyer_matches[current] = -1
            free_buyers.append(current)
        else:
            free_buyers.append(buyer)
    return buyer_matches


def inverse_matching(matching: np.ndarray, num_sellers: int) -> np.ndarray:
    inverse = np.full(num_sellers, -1, dtype=int)
    for buyer, seller in enumerate(matching):
        if seller >= 0:
            inverse[seller] = buyer
    return inverse


def blocking_pairs(matching: np.ndarray, buyer_utils: np.ndarray, seller_utils: np.ndarray) -> list[tuple[int, int]]:
    num_buyers, num_sellers = buyer_utils.shape
    seller_match = inverse_matching(matching, num_sellers=num_sellers)
    pairs: list[tuple[int, int]] = []
    for buyer in range(num_buyers):
        current_seller = matching[buyer]
        current_buyer_value = buyer_utils[buyer, current_seller] if current_seller >= 0 else -np.inf
        for seller in range(num_sellers):
            current_buyer = seller_match[seller]
            current_seller_value = seller_utils[seller, current_buyer] if current_buyer >= 0 else -np.inf
            buyer_improves = buyer_utils[buyer, seller] > current_buyer_value + 1e-12
            seller_improves = seller_utils[seller, buyer] > current_seller_value + 1e-12
            if buyer_improves and seller_improves:
                pairs.append((buyer, seller))
    return pairs
