from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MetaSwagPosterior:
    mean: np.ndarray
    covariance: np.ndarray
    weights: np.ndarray
    diagonal: np.ndarray
    deviations: np.ndarray
    beta: float | None
    weighting_scheme: str
    effective_sample_size: float
    threshold: float | None

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(self.mean, self.covariance, size=num_samples)


def effective_sample_size(weights: np.ndarray) -> float:
    numerator = float(np.sum(weights) ** 2)
    denominator = float(np.sum(weights**2))
    return numerator / denominator if denominator > 0 else 0.0


def softmax_weights(scores: np.ndarray, beta: float) -> np.ndarray:
    centered = scores - np.max(scores)
    logits = beta * centered
    exps = np.exp(logits)
    return exps / exps.sum()


def threshold_weights(scores: np.ndarray, threshold: float) -> np.ndarray:
    mask = scores >= threshold
    if not np.any(mask):
        mask[np.argmax(scores)] = True
    weights = mask.astype(float)
    return weights / weights.sum()


def find_beta_for_target_ess(
    scores: np.ndarray,
    target_ess: float,
    beta_min: float = 0.0,
    beta_max: float = 100.0,
    steps: int = 60,
) -> float:
    target_ess = float(np.clip(target_ess, 1.0, len(scores)))
    lo, hi = beta_min, beta_max
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        ess = effective_sample_size(softmax_weights(scores, mid))
        if ess < target_ess:
            hi = mid
        else:
            lo = mid
    return lo


def resolve_weights(
    scores: np.ndarray,
    weighting_scheme: str,
    beta: float,
    target_ess_fraction: float,
    threshold_quantile: float,
) -> tuple[np.ndarray, float | None, float, float | None]:
    scheme = weighting_scheme.lower()
    if scheme == "softmax":
        weights = softmax_weights(scores, beta)
        resolved_beta = beta
        threshold = None
    elif scheme == "ess":
        target_ess = max(1.0, target_ess_fraction * len(scores))
        resolved_beta = find_beta_for_target_ess(scores, target_ess)
        weights = softmax_weights(scores, resolved_beta)
        threshold = None
    elif scheme == "threshold":
        threshold = float(np.quantile(scores, threshold_quantile))
        weights = threshold_weights(scores, threshold)
        resolved_beta = None
    else:
        raise ValueError(f"Unknown weighting scheme: {weighting_scheme}")

    return weights, resolved_beta, effective_sample_size(weights), threshold


def fit_meta_swag(
    checkpoints: np.ndarray,
    evidence_scores: np.ndarray,
    beta: float = 3.0,
    rank: int = 20,
    weighting_scheme: str = "softmax",
    target_ess_fraction: float = 0.5,
    threshold_quantile: float = 0.75,
) -> MetaSwagPosterior:
    weights, resolved_beta, ess, threshold = resolve_weights(
        evidence_scores,
        weighting_scheme=weighting_scheme,
        beta=beta,
        target_ess_fraction=target_ess_fraction,
        threshold_quantile=threshold_quantile,
    )
    mean = np.sum(checkpoints * weights[:, None], axis=0)
    centered = checkpoints - mean
    diagonal = np.sum(weights[:, None] * centered * centered, axis=0)

    retain = min(rank, checkpoints.shape[0])
    retained = centered[-retain:]
    retained_weights = np.sqrt(weights[-retain:])[:, None]
    deviations = retained * retained_weights
    if retain > 1:
        low_rank = deviations.T @ deviations / max(retain - 1, 1)
    else:
        low_rank = np.zeros((checkpoints.shape[1], checkpoints.shape[1]), dtype=float)

    covariance = 0.5 * np.diag(diagonal) + 0.5 * low_rank
    covariance = covariance + 1e-6 * np.eye(covariance.shape[0])
    return MetaSwagPosterior(
        mean,
        covariance,
        weights,
        diagonal,
        deviations,
        resolved_beta,
        weighting_scheme,
        ess,
        threshold,
    )
