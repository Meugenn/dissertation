from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .games import MatrixGame
from .policies import distance_to_target_policy, expected_payoffs, finite_difference_hessian
from .posterior import MetaSwagPosterior


@dataclass
class ExperimentMetrics:
    game: str
    hm_am_ratio: float
    point_predictive_variance: float
    posterior_predictive_variance: float
    variance_ratio: float
    basin_proxy: float
    posterior_top_eigenvalue: float


def harmonic_mean(values: np.ndarray) -> float:
    return float(len(values) / np.sum(1.0 / values))


def arithmetic_mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def predictive_value(theta: np.ndarray, game: MatrixGame) -> float:
    if game.nash_policy_p1 is not None and game.nash_policy_p2 is not None:
        return -distance_to_target_policy(theta, game.nash_policy_p1, game.nash_policy_p2)
    r1, r2 = expected_payoffs(theta, game.payoff_p1, game.payoff_p2)
    return 0.5 * (r1 + r2)


def evaluate_metrics(
    game: MatrixGame,
    posterior: MetaSwagPosterior,
    checkpoints: np.ndarray,
    noise_variance_pair: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> ExperimentMetrics:
    samples = posterior.sample(num_samples, rng)
    sample_values = np.array([predictive_value(theta, game) for theta in samples])
    point_value = np.array([predictive_value(theta, game) for theta in checkpoints[-num_samples:]])

    def objective(theta: np.ndarray) -> float:
        return predictive_value(theta, game)

    hessian = finite_difference_hessian(objective, posterior.mean)
    eigvals = np.linalg.eigvalsh(-hessian)
    positive = eigvals[eigvals > 1e-6]
    basin_proxy = float(np.mean(1.0 / positive)) if positive.size else 0.0
    posterior_eigs = np.linalg.eigvalsh(posterior.covariance)

    hm_am_ratio = harmonic_mean(noise_variance_pair) / arithmetic_mean(noise_variance_pair)
    posterior_var = float(np.var(sample_values))
    point_var = float(np.var(point_value))
    variance_ratio = posterior_var / point_var if point_var > 0 else 0.0

    return ExperimentMetrics(
        game=game.name,
        hm_am_ratio=hm_am_ratio,
        point_predictive_variance=point_var,
        posterior_predictive_variance=posterior_var,
        variance_ratio=variance_ratio,
        basin_proxy=basin_proxy,
        posterior_top_eigenvalue=float(posterior_eigs[-1]),
    )
