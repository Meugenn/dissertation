from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policies import finite_difference_gradient, softmax


@dataclass(frozen=True)
class IteratedGameSpec:
    name: str
    num_actions: int
    payoff_p1: np.ndarray

    @property
    def num_states(self) -> int:
        return 1 + self.num_actions**2


def ipd_spec() -> IteratedGameSpec:
    return IteratedGameSpec(
        name="ipd",
        num_actions=2,
        payoff_p1=np.array([-0.5, 1.5, -1.5, 0.5], dtype=float),
    )


def rps_spec() -> IteratedGameSpec:
    return IteratedGameSpec(
        name="rps",
        num_actions=3,
        payoff_p1=np.array([0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0], dtype=float),
    )


def _joint_index(a_self: int, a_opp: int, num_actions: int) -> int:
    return num_actions * a_self + a_opp


def logits_to_policy(theta: np.ndarray, num_states: int, num_actions: int) -> np.ndarray:
    reshaped = theta.reshape(num_states, num_actions)
    return np.stack([softmax(row) for row in reshaped], axis=0)


def build_markov_kernel(
    learner_policy: np.ndarray,
    opponent_policy: np.ndarray,
    spec: IteratedGameSpec,
) -> tuple[np.ndarray, np.ndarray]:
    num_states = spec.num_states
    num_actions = spec.num_actions
    transition = np.zeros((num_states, num_states), dtype=float)
    rewards = np.zeros(num_states, dtype=float)

    for state in range(num_states):
        for a_self in range(num_actions):
            for a_opp in range(num_actions):
                prob = learner_policy[state, a_self] * opponent_policy[state, a_opp]
                next_state = 1 + _joint_index(a_self, a_opp, num_actions)
                transition[state, next_state] += prob
                rewards[state] += prob * spec.payoff_p1[_joint_index(a_self, a_opp, num_actions)]

    return transition, rewards


def discounted_return(
    theta: np.ndarray,
    opponent_policy: np.ndarray,
    spec: IteratedGameSpec,
    gamma: float,
) -> float:
    learner_policy = logits_to_policy(theta, spec.num_states, spec.num_actions)
    transition, rewards = build_markov_kernel(learner_policy, opponent_policy, spec)
    identity = np.eye(spec.num_states)
    value = np.linalg.solve(identity - gamma * transition, rewards)
    return float(value[0])


def discounted_state_visitation(
    theta: np.ndarray,
    opponent_policy: np.ndarray,
    spec: IteratedGameSpec,
    gamma: float,
) -> np.ndarray:
    learner_policy = logits_to_policy(theta, spec.num_states, spec.num_actions)
    transition, _ = build_markov_kernel(learner_policy, opponent_policy, spec)
    start = np.zeros(spec.num_states, dtype=float)
    start[0] = 1.0
    identity = np.eye(spec.num_states)
    visitation = (1.0 - gamma) * start @ np.linalg.inv(identity - gamma * transition)
    return visitation / np.clip(visitation.sum(), 1e-12, None)


def cooperation_rate(
    theta: np.ndarray,
    opponent_policy: np.ndarray,
    spec: IteratedGameSpec,
    gamma: float,
) -> float:
    if spec.name != "ipd":
        raise ValueError("Cooperation rate is only defined for IPD.")
    learner_policy = logits_to_policy(theta, spec.num_states, spec.num_actions)
    visitation = discounted_state_visitation(theta, opponent_policy, spec, gamma)
    return float(np.sum(visitation * learner_policy[:, 0]))


@dataclass
class IteratedTrajectory:
    checkpoints: np.ndarray
    evidence_scores: np.ndarray
    final_theta: np.ndarray
    objective_trace: np.ndarray


def simulate_iterated_adaptation(
    opponent_policy: np.ndarray,
    spec: IteratedGameSpec,
    rng: np.random.Generator,
    steps: int = 80,
    burn_in: int = 20,
    lr: float = 0.4,
    grad_noise_std: float = 0.15,
    gamma: float = 0.96,
) -> IteratedTrajectory:
    theta = rng.normal(scale=0.05, size=spec.num_states * spec.num_actions)
    checkpoints: list[np.ndarray] = []
    evidence_scores: list[float] = []
    objective_trace: list[float] = []

    def objective(local_theta: np.ndarray) -> float:
        return discounted_return(local_theta, opponent_policy, spec, gamma)

    for step in range(steps):
        grad = finite_difference_gradient(objective, theta)
        noise = rng.normal(scale=grad_noise_std, size=theta.shape)
        theta = theta + lr * (grad + noise)
        value = objective(theta)
        objective_trace.append(value)
        if step >= burn_in:
            checkpoints.append(theta.copy())
            evidence_scores.append(value)

    return IteratedTrajectory(
        checkpoints=np.asarray(checkpoints),
        evidence_scores=np.asarray(evidence_scores),
        final_theta=theta.copy(),
        objective_trace=np.asarray(objective_trace),
    )
