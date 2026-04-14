from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.clip(np.sum(exps), 1e-12, None)


def finite_difference_gradient(objective_fn, theta: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    grad = np.zeros_like(theta, dtype=float)
    for idx in range(theta.size):
        basis = np.zeros_like(theta)
        basis[idx] = epsilon
        grad[idx] = (objective_fn(theta + basis) - objective_fn(theta - basis)) / (2.0 * epsilon)
    return grad


@dataclass(frozen=True)
class MatrixGame:
    name: str
    payoff_p1: np.ndarray
    payoff_p2: np.ndarray
    nash_policy_p1: np.ndarray | None = None
    nash_policy_p2: np.ndarray | None = None

    @property
    def num_actions(self) -> int:
        return int(self.payoff_p1.shape[0])


def matching_pennies() -> MatrixGame:
    payoff = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    uniform = np.full(2, 0.5, dtype=float)
    return MatrixGame("matching_pennies", payoff, -payoff, uniform, uniform)


def stag_hunt() -> MatrixGame:
    payoff = np.array([[4.0, 1.0], [3.0, 2.0]], dtype=float)
    return MatrixGame("stag_hunt", payoff, payoff.copy())


def rock_paper_scissors() -> MatrixGame:
    payoff = np.array(
        [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]],
        dtype=float,
    )
    uniform = np.full(3, 1.0 / 3.0, dtype=float)
    return MatrixGame("rock_paper_scissors", payoff, -payoff, uniform, uniform)


def prisoners_dilemma() -> MatrixGame:
    payoff_p1 = np.array([[3.0, 0.0], [5.0, 1.0]], dtype=float)
    payoff_p2 = np.array([[3.0, 5.0], [0.0, 1.0]], dtype=float)
    return MatrixGame("prisoners_dilemma", payoff_p1, payoff_p2)


def default_matrix_games() -> list[MatrixGame]:
    return [matching_pennies(), rock_paper_scissors(), stag_hunt(), prisoners_dilemma()]


def split_joint_logits(theta: np.ndarray, num_actions: int) -> tuple[np.ndarray, np.ndarray]:
    return theta[:num_actions], theta[num_actions:]


def matrix_policies(theta: np.ndarray, game: MatrixGame) -> tuple[np.ndarray, np.ndarray]:
    logits_p1, logits_p2 = split_joint_logits(theta, game.num_actions)
    return softmax(logits_p1), softmax(logits_p2)


def matrix_payoffs(theta: np.ndarray, game: MatrixGame) -> tuple[float, float]:
    p1, p2 = matrix_policies(theta, game)
    joint = np.outer(p1, p2)
    return float(np.sum(joint * game.payoff_p1)), float(np.sum(joint * game.payoff_p2))


def matrix_distance_to_nash(theta: np.ndarray, game: MatrixGame) -> float:
    if game.nash_policy_p1 is None or game.nash_policy_p2 is None:
        return np.nan
    p1, p2 = matrix_policies(theta, game)
    return float(np.linalg.norm(p1 - game.nash_policy_p1) + np.linalg.norm(p2 - game.nash_policy_p2))


@dataclass(frozen=True)
class IteratedGameSpec:
    name: str
    stage_payoff_p1: np.ndarray
    stage_payoff_p2: np.ndarray

    @property
    def num_actions(self) -> int:
        return int(self.stage_payoff_p1.shape[0])

    @property
    def num_states(self) -> int:
        return 1 + self.num_actions**2


def ipd_spec() -> IteratedGameSpec:
    return IteratedGameSpec(
        name="ipd",
        stage_payoff_p1=np.array([[3.0, 0.0], [5.0, 1.0]], dtype=float),
        stage_payoff_p2=np.array([[3.0, 5.0], [0.0, 1.0]], dtype=float),
    )


def iterated_rps_spec() -> IteratedGameSpec:
    payoff = np.array(
        [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]],
        dtype=float,
    )
    return IteratedGameSpec("iterated_rps", payoff, -payoff)


def joint_index(a_self: int, a_opp: int, num_actions: int) -> int:
    return num_actions * a_self + a_opp


def stateful_policy(theta: np.ndarray, num_states: int, num_actions: int) -> np.ndarray:
    reshaped = theta.reshape(num_states, num_actions)
    return np.stack([softmax(row) for row in reshaped], axis=0)


def build_iterated_kernel(
    theta_joint: np.ndarray,
    spec: IteratedGameSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_player_dim = spec.num_states * spec.num_actions
    theta_p1 = theta_joint[:per_player_dim]
    theta_p2 = theta_joint[per_player_dim:]
    policy_p1 = stateful_policy(theta_p1, spec.num_states, spec.num_actions)
    policy_p2 = stateful_policy(theta_p2, spec.num_states, spec.num_actions)
    transition = np.zeros((spec.num_states, spec.num_states), dtype=float)
    rewards_p1 = np.zeros(spec.num_states, dtype=float)
    rewards_p2 = np.zeros(spec.num_states, dtype=float)
    for state in range(spec.num_states):
        for a1 in range(spec.num_actions):
            for a2 in range(spec.num_actions):
                prob = policy_p1[state, a1] * policy_p2[state, a2]
                next_state = 1 + joint_index(a1, a2, spec.num_actions)
                transition[state, next_state] += prob
                rewards_p1[state] += prob * spec.stage_payoff_p1[a1, a2]
                rewards_p2[state] += prob * spec.stage_payoff_p2[a1, a2]
    return transition, rewards_p1, rewards_p2


def iterated_returns(theta_joint: np.ndarray, spec: IteratedGameSpec, gamma: float) -> tuple[float, float]:
    transition, rewards_p1, rewards_p2 = build_iterated_kernel(theta_joint, spec)
    identity = np.eye(spec.num_states)
    value_p1 = np.linalg.solve(identity - gamma * transition, rewards_p1)
    value_p2 = np.linalg.solve(identity - gamma * transition, rewards_p2)
    return float(value_p1[0]), float(value_p2[0])


def objective_player(theta_joint: np.ndarray, env, player_idx: int, gamma: float = 0.96) -> float:
    if isinstance(env, MatrixGame):
        r1, r2 = matrix_payoffs(theta_joint, env)
    else:
        r1, r2 = iterated_returns(theta_joint, env, gamma)
    return r1 if player_idx == 0 else r2


def player_slice(player_idx: int, per_player_dim: int) -> slice:
    start = player_idx * per_player_dim
    return slice(start, start + per_player_dim)


def player_gradient(theta_joint: np.ndarray, env, player_idx: int, gamma: float = 0.96) -> np.ndarray:
    per_player_dim = theta_joint.size // 2
    sl = player_slice(player_idx, per_player_dim)

    def player_objective(local_params: np.ndarray) -> float:
        theta = theta_joint.copy()
        theta[sl] = local_params
        return objective_player(theta, env, player_idx, gamma=gamma)

    return finite_difference_gradient(player_objective, theta_joint[sl])


def lola_correction(
    theta_joint: np.ndarray,
    env,
    player_idx: int,
    gamma: float = 0.96,
    opponent_lr: float = 0.5,
) -> np.ndarray:
    per_player_dim = theta_joint.size // 2
    self_sl = player_slice(player_idx, per_player_dim)
    opp_idx = 1 - player_idx
    opp_sl = player_slice(opp_idx, per_player_dim)
    base_grad = player_gradient(theta_joint, env, player_idx, gamma=gamma)

    def lola_objective(local_params: np.ndarray) -> float:
        theta = theta_joint.copy()
        theta[self_sl] = local_params
        opp_grad = player_gradient(theta, env, opp_idx, gamma=gamma)
        theta[opp_sl] = theta[opp_sl] + opponent_lr * opp_grad
        return objective_player(theta, env, player_idx, gamma=gamma)

    total_grad = finite_difference_gradient(lola_objective, theta_joint[self_sl])
    return total_grad - base_grad


@dataclass
class RolloutResult:
    trace_rows: list[dict[str, float | int | str]]
    summary_row: dict[str, float | int | str]


def run_two_player_rollout(
    env,
    method: str,
    rng: np.random.Generator,
    steps: int,
    lr: float,
    noise_stds: tuple[float, float],
    gamma: float = 0.96,
    lambda_lola: float = 0.6,
    lambda_power: float = 0.75,
    lambda_offset: float = 5.0,
    opponent_lr: float = 0.5,
    init_scale: float = 0.1,
    theta_init: np.ndarray | None = None,
    method_p1: str | None = None,
    method_p2: str | None = None,
) -> RolloutResult:
    if isinstance(env, MatrixGame):
        per_player_dim = env.num_actions
    else:
        per_player_dim = env.num_states * env.num_actions
    theta = theta_init.copy() if theta_init is not None else rng.normal(scale=init_scale, size=2 * per_player_dim)
    variances = np.square(np.asarray(noise_stds, dtype=float))
    min_variance = float(np.min(variances))
    weights = min_variance / np.clip(variances, 1e-12, None)
    method_p1 = method if method_p1 is None else method_p1
    method_p2 = method if method_p2 is None else method_p2
    methods = [method_p1, method_p2]
    trace_rows: list[dict[str, float | int | str]] = []
    grad_norm_acc = np.zeros(2, dtype=float)
    noise_norm_acc = np.zeros(2, dtype=float)

    for step in range(steps):
        updates: list[np.ndarray] = []
        lambda_step = lambda_lola / ((step + lambda_offset) ** lambda_power)
        for player_idx in range(2):
            player_method = methods[player_idx]
            base_grad = player_gradient(theta, env, player_idx, gamma=gamma)
            noise = rng.normal(scale=noise_stds[player_idx], size=base_grad.shape)
            noisy_grad = base_grad + noise
            correction = np.zeros_like(base_grad)
            if player_method in {"lola", "ew_lola"}:
                correction = lola_correction(
                    theta_joint=theta,
                    env=env,
                    player_idx=player_idx,
                    gamma=gamma,
                    opponent_lr=opponent_lr,
                )
            update = noisy_grad + lambda_step * correction
            if player_method in {"ew", "ew_lola"}:
                update = weights[player_idx] * update
            updates.append(update)
            grad_norm_acc[player_idx] += float(np.linalg.norm(base_grad))
            noise_norm_acc[player_idx] += float(np.linalg.norm(noise))

        theta[:per_player_dim] += lr * updates[0]
        theta[per_player_dim:] += lr * updates[1]

        if isinstance(env, MatrixGame):
            r1, r2 = matrix_payoffs(theta, env)
            distance = matrix_distance_to_nash(theta, env)
        else:
            r1, r2 = iterated_returns(theta, env, gamma)
            distance = np.nan

        trace_rows.append(
            {
                "step": step,
                "method": method,
                "method_p1": method_p1,
                "method_p2": method_p2,
                "reward_p1": r1,
                "reward_p2": r2,
                "lambda_step": lambda_step,
                "distance_to_nash": distance,
                "avg_grad_norm_p1": grad_norm_acc[0] / (step + 1),
                "avg_grad_norm_p2": grad_norm_acc[1] / (step + 1),
                "avg_noise_norm_p1": noise_norm_acc[0] / (step + 1),
                "avg_noise_norm_p2": noise_norm_acc[1] / (step + 1),
            }
        )

    final = trace_rows[-1]
    summary_row: dict[str, float | int | str] = {
        "method": method,
        "method_p1": method_p1,
        "method_p2": method_p2,
        "final_reward_p1": float(final["reward_p1"]),
        "final_reward_p2": float(final["reward_p2"]),
        "mean_reward_p1": float(np.mean([row["reward_p1"] for row in trace_rows])),
        "mean_reward_p2": float(np.mean([row["reward_p2"] for row in trace_rows])),
        "final_distance_to_nash": float(final["distance_to_nash"]),
        "avg_noise_norm_p1": float(final["avg_noise_norm_p1"]),
        "avg_noise_norm_p2": float(final["avg_noise_norm_p2"]),
        "weight_p1": float(weights[0] if method in {"ew", "ew_lola"} else 1.0),
        "weight_p2": float(weights[1] if method in {"ew", "ew_lola"} else 1.0),
    }
    return RolloutResult(trace_rows=trace_rows, summary_row=summary_row)
