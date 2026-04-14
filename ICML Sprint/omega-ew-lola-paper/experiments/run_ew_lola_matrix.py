from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ew_lola_core import default_matrix_games, run_two_player_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EW/LOLA matrix-game experiments.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/matrix"))
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--noise-pairs", nargs="+", default=["1,1", "4,1", "16,1"])
    parser.add_argument("--lambda-lola", type=float, default=0.8)
    parser.add_argument("--lambda-power", type=float, default=0.75)
    parser.add_argument("--lambda-offset", type=float, default=5.0)
    parser.add_argument("--opponent-lr", type=float, default=0.5)
    return parser.parse_args()


def parse_noise_pair(spec: str) -> tuple[float, float]:
    left, right = spec.split(",")
    return float(left), float(right)


def make_plot(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    grouped = (
        df.groupby(["game", "noise_pair", "method"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["game", "noise_pair", "method"])
    )
    color_map = {
        "standard": "tab:blue",
        "ew": "tab:green",
        "lola": "tab:orange",
        "ew_lola": "tab:red",
    }
    for method, sub in grouped.groupby("method"):
        axes[0].scatter(
            sub["noise_ratio"],
            sub["avg_noise_norm_p1"] / np.clip(sub["avg_noise_norm_p2"], 1e-12, None),
            label=method,
            color=color_map[method],
        )
        axes[1].scatter(
            sub["noise_ratio"],
            sub["final_distance_to_nash"],
            label=method,
            color=color_map[method],
        )
    axes[0].set_xlabel("Configured variance ratio")
    axes[0].set_ylabel("Observed noise-norm ratio p1/p2")
    axes[0].set_title("Noise heterogeneity")
    axes[1].set_xlabel("Configured variance ratio")
    axes[1].set_ylabel("Final distance to Nash")
    axes[1].set_title("Convergence proxy")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "matrix_summary.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, float | int | str]] = []
    trace_rows: list[dict[str, float | int | str]] = []
    methods = ["standard", "ew", "lola", "ew_lola"]

    for game in default_matrix_games():
        if game.nash_policy_p1 is None or game.nash_policy_p2 is None:
            continue
        for noise_spec in args.noise_pairs:
            noise_pair = parse_noise_pair(noise_spec)
            noise_ratio = (noise_pair[0] ** 2) / max(noise_pair[1] ** 2, 1e-12)
            for seed in range(args.seeds):
                seed_rng = np.random.default_rng(seed)
                theta_init = seed_rng.normal(scale=0.1, size=2 * game.num_actions)
                for method in methods:
                    rng = np.random.default_rng(seed + 10_000 * (methods.index(method) + 1))
                    rollout = run_two_player_rollout(
                        env=game,
                        method=method,
                        rng=rng,
                        steps=args.steps,
                        lr=args.lr,
                        noise_stds=noise_pair,
                        lambda_lola=args.lambda_lola,
                        lambda_power=args.lambda_power,
                        lambda_offset=args.lambda_offset,
                        opponent_lr=args.opponent_lr,
                        theta_init=theta_init,
                    )
                    summary_rows.append(
                        {
                            "game": game.name,
                            "noise_pair": noise_spec,
                            "noise_ratio": noise_ratio,
                            "seed": seed,
                            **rollout.summary_row,
                        }
                    )
                    for row in rollout.trace_rows:
                        trace_rows.append(
                            {
                                "game": game.name,
                                "noise_pair": noise_spec,
                                "noise_ratio": noise_ratio,
                                "seed": seed,
                                **row,
                            }
                        )

    summary_df = pd.DataFrame(summary_rows)
    trace_df = pd.DataFrame(trace_rows)
    summary_df.to_csv(args.output_dir / "matrix_summary.csv", index=False)
    trace_df.to_csv(args.output_dir / "matrix_trace.csv", index=False)
    make_plot(summary_df, args.output_dir)
    grouped = (
        summary_df.groupby(["game", "noise_pair", "method"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["game", "noise_pair", "method"])
    )
    print(grouped.to_string(index=False))
    print(f"\nSaved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
