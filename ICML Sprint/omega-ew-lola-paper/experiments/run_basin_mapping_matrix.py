from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ew_lola_core import (
    lola_correction,
    matching_pennies,
    matrix_distance_to_nash,
    matrix_payoffs,
    player_gradient,
)


def symmetric_theta(x: float, y: float) -> np.ndarray:
    return np.array([x, -x, y, -y], dtype=float)


def deterministic_step(
    theta: np.ndarray,
    method: str,
    lr: float,
    lambda_lola: float,
    opponent_lr: float,
) -> np.ndarray:
    updated = theta.copy()
    updates: list[np.ndarray] = []
    game = matching_pennies()
    for player_idx in range(2):
        grad = player_gradient(updated, game, player_idx)
        correction = np.zeros_like(grad)
        if method == "lola":
            correction = lola_correction(
                theta_joint=updated,
                env=game,
                player_idx=player_idx,
                opponent_lr=opponent_lr,
            )
        updates.append(grad + lambda_lola * correction)

    updated[:2] += lr * updates[0]
    updated[2:] += lr * updates[1]
    return updated


def run_trajectory(
    theta_init: np.ndarray,
    method: str,
    lr: float,
    lambda_lola: float,
    opponent_lr: float,
    steps: int,
) -> np.ndarray:
    theta = theta_init.copy()
    for _ in range(steps):
        theta = deterministic_step(
            theta=theta,
            method=method,
            lr=lr,
            lambda_lola=lambda_lola,
            opponent_lr=opponent_lr,
        )
    return theta


def make_plot(
    basin_df: pd.DataFrame,
    output_path: Path,
    methods: list[str],
    lambdas: list[float],
) -> None:
    fig, axes = plt.subplots(
        nrows=len(methods),
        ncols=len(lambdas),
        figsize=(3.2 * len(lambdas), 3.2 * len(methods)),
        squeeze=False,
    )

    x_values = np.sort(basin_df["x"].unique())
    y_values = np.sort(basin_df["y"].unique())
    extent = [x_values.min(), x_values.max(), y_values.min(), y_values.max()]

    for row_idx, method in enumerate(methods):
        for col_idx, lambda_lola in enumerate(lambdas):
            ax = axes[row_idx][col_idx]
            subset = basin_df[
                (basin_df["method"] == method)
                & (np.isclose(basin_df["lambda_lola"], lambda_lola))
            ].copy()
            pivot = subset.pivot(index="y", columns="x", values="converged").sort_index()
            image = ax.imshow(
                pivot.to_numpy(),
                origin="lower",
                extent=extent,
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
                aspect="auto",
            )
            success_rate = subset["converged"].mean()
            ax.set_title(f"{method}, $\\lambda={lambda_lola:.2f}$\nrate={success_rate:.2f}")
            ax.set_xlabel("player 1 logit skew")
            ax.set_ylabel("player 2 logit skew")

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, label="converged")
    fig.suptitle("Matching-pennies basin map near the mixed Nash policy", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map local convergence basins for matching pennies.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.35)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    parser.add_argument("--methods", nargs="+", default=["standard", "lola"])
    parser.add_argument("--tol", type=float, default=0.15)
    parser.add_argument("--opponent-lr", type=float, default=0.5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    xs = np.linspace(-args.radius, args.radius, args.grid_size)
    ys = np.linspace(-args.radius, args.radius, args.grid_size)
    game = matching_pennies()

    basin_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for method in args.methods:
        for lambda_lola in args.lambdas:
            if method == "standard" and lambda_lola != 0.0:
                continue
            successes = []
            final_distances = []
            for x in xs:
                for y in ys:
                    theta_init = symmetric_theta(x, y)
                    theta_final = run_trajectory(
                        theta_init=theta_init,
                        method=method,
                        lr=args.lr,
                        lambda_lola=lambda_lola,
                        opponent_lr=args.opponent_lr,
                        steps=args.steps,
                    )
                    final_distance = matrix_distance_to_nash(theta_final, game)
                    reward_p1, reward_p2 = matrix_payoffs(theta_final, game)
                    converged = float(final_distance <= args.tol)
                    successes.append(converged)
                    final_distances.append(final_distance)
                    basin_rows.append(
                        {
                            "method": method,
                            "lambda_lola": lambda_lola,
                            "x": float(x),
                            "y": float(y),
                            "converged": converged,
                            "final_distance_to_nash": final_distance,
                            "final_reward_p1": reward_p1,
                            "final_reward_p2": reward_p2,
                        }
                    )

            summary_rows.append(
                {
                    "method": method,
                    "lambda_lola": lambda_lola,
                    "success_rate": float(np.mean(successes)),
                    "mean_final_distance_to_nash": float(np.mean(final_distances)),
                    "median_final_distance_to_nash": float(np.median(final_distances)),
                    "grid_size": args.grid_size,
                    "steps": args.steps,
                    "lr": args.lr,
                    "tol": args.tol,
                }
            )

    basin_df = pd.DataFrame(basin_rows)
    summary_df = pd.DataFrame(summary_rows)
    basin_df.to_csv(args.output_dir / "basin_map.csv", index=False)
    summary_df.to_csv(args.output_dir / "basin_summary.csv", index=False)
    make_plot(
        basin_df=basin_df,
        output_path=args.output_dir / "basin_map.png",
        methods=args.methods,
        lambdas=args.lambdas,
    )


if __name__ == "__main__":
    main()
