from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / exps.sum()


def joint_policy(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = logits.size // 2
    return softmax(logits[:half]), softmax(logits[half:])


def expected_payoffs(
    logits: np.ndarray,
    payoff_p1: np.ndarray,
    payoff_p2: np.ndarray,
) -> tuple[float, float]:
    p1, p2 = joint_policy(logits)
    joint = np.outer(p1, p2)
    return float(np.sum(joint * payoff_p1)), float(np.sum(joint * payoff_p2))


def distance_to_target_policy(
    logits: np.ndarray,
    target_p1: np.ndarray,
    target_p2: np.ndarray,
) -> float:
    p1, p2 = joint_policy(logits)
    return float(np.linalg.norm(p1 - target_p1) + np.linalg.norm(p2 - target_p2))


def finite_difference_gradient(
    objective_fn,
    theta: np.ndarray,
    epsilon: float = 1e-4,
) -> np.ndarray:
    grad = np.zeros_like(theta, dtype=float)
    for idx in range(theta.size):
        basis = np.zeros_like(theta)
        basis[idx] = epsilon
        grad[idx] = (objective_fn(theta + basis) - objective_fn(theta - basis)) / (2.0 * epsilon)
    return grad


def finite_difference_hessian(
    objective_fn,
    theta: np.ndarray,
    epsilon: float = 1e-3,
) -> np.ndarray:
    dim = theta.size
    hessian = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        ei = np.zeros(dim)
        ei[i] = epsilon
        for j in range(dim):
            ej = np.zeros(dim)
            ej[j] = epsilon
            hessian[i, j] = (
                objective_fn(theta + ei + ej)
                - objective_fn(theta + ei - ej)
                - objective_fn(theta - ei + ej)
                + objective_fn(theta - ei - ej)
            ) / (4.0 * epsilon * epsilon)
    return hessian
