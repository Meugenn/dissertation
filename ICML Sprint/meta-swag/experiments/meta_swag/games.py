from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def prisoners_dilemma() -> MatrixGame:
    payoff_p1 = np.array([[3.0, 0.0], [5.0, 1.0]], dtype=float)
    payoff_p2 = np.array([[3.0, 5.0], [0.0, 1.0]], dtype=float)
    return MatrixGame("prisoners_dilemma", payoff_p1, payoff_p2)


def rock_paper_scissors() -> MatrixGame:
    payoff = np.array(
        [
            [0.0, -1.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    uniform = np.full(3, 1.0 / 3.0, dtype=float)
    return MatrixGame("rock_paper_scissors", payoff, -payoff, uniform, uniform)


def default_games() -> list[MatrixGame]:
    return [matching_pennies(), rock_paper_scissors(), stag_hunt(), prisoners_dilemma()]
