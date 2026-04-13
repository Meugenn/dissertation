from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .games import MatrixGame
from .policies import distance_to_target_policy, expected_payoffs, finite_difference_gradient


@dataclass
class Trajectory:
    checkpoints: np.ndarray
    evidence_scores: np.ndarray
    noise_variance_pair: np.ndarray
    final_theta: np.ndarray


def meta_objective(theta: np.ndarray, game: MatrixGame) -> float:
    if game.nash_policy_p1 is not None and game.nash_policy_p2 is not None:
        return -distance_to_target_policy(theta, game.nash_policy_p1, game.nash_policy_p2)
    r1, r2 = expected_payoffs(theta, game.payoff_p1, game.payoff_p2)
    return 0.5 * (r1 + r2)


def simulate_trajectory(
    game: MatrixGame,
    rng: np.random.Generator,
    steps: int = 250,
    burn_in: int = 100,
    lr: float = 0.35,
    noise_variance_pair: tuple[float, float] = (1.0, 1.0),
) -> Trajectory:
    theta = rng.normal(scale=0.3, size=2 * game.num_actions)
    checkpoints: list[np.ndarray] = []
    evidence_scores: list[float] = []
    def objective(local_theta: np.ndarray) -> float:
        return meta_objective(local_theta, game)

    for step in range(steps):
        grad = finite_difference_gradient(objective, theta)
        half = theta.size // 2
        p1_std = np.sqrt(noise_variance_pair[0])
        p2_std = np.sqrt(noise_variance_pair[1])
        per_coordinate_std = np.concatenate(
            [
                np.full(half, p1_std, dtype=float),
                np.full(half, p2_std, dtype=float),
            ]
        )
        noise = rng.normal(scale=per_coordinate_std, size=theta.shape)
        theta = theta + lr * (grad + noise)

        if step >= burn_in:
            score = objective(theta)
            checkpoints.append(theta.copy())
            evidence_scores.append(score)

    return Trajectory(
        checkpoints=np.asarray(checkpoints),
        evidence_scores=np.asarray(evidence_scores),
        noise_variance_pair=np.asarray(noise_variance_pair, dtype=float),
        final_theta=theta.copy(),
    )
