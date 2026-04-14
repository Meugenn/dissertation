from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ew_lola_core import ipd_spec, iterated_rps_spec, run_two_player_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EW/LOLA iterated-game experiments.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/iterated"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument("--lambda-lola", type=float, default=0.5)
    parser.add_argument("--lambda-power", type=float, default=0.75)
    parser.add_argument("--lambda-offset", type=float, default=5.0)
    parser.add_argument("--opponent-lr", type=float, default=0.35)
    parser.add_argument("--noise-pairs", nargs="+", default=["0.2,0.2", "0.5,0.2"])
    return parser.parse_args()


def parse_noise_pair(spec: str) -> tuple[float, float]:
    left, right = spec.split(",")
    return float(left), float(right)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, float | int | str]] = []
    trace_rows: list[dict[str, float | int | str]] = []
    methods = ["standard", "ew", "lola", "ew_lola"]
    envs = [ipd_spec(), iterated_rps_spec()]

    for env in envs:
        per_player_dim = env.num_states * env.num_actions
        for noise_spec in args.noise_pairs:
            noise_pair = parse_noise_pair(noise_spec)
            for seed in range(args.seeds):
                seed_rng = np.random.default_rng(seed)
                theta_init = seed_rng.normal(scale=0.05, size=2 * per_player_dim)
                for method in methods:
                    rng = np.random.default_rng(seed + 20_000 * (methods.index(method) + 1))
                    rollout = run_two_player_rollout(
                        env=env,
                        method=method,
                        rng=rng,
                        steps=args.steps,
                        lr=args.lr,
                        noise_stds=noise_pair,
                        gamma=args.gamma,
                        lambda_lola=args.lambda_lola,
                        lambda_power=args.lambda_power,
                        lambda_offset=args.lambda_offset,
                        opponent_lr=args.opponent_lr,
                        init_scale=0.05,
                        theta_init=theta_init,
                    )
                    summary_rows.append(
                        {
                            "env_name": env.name,
                            "noise_pair": noise_spec,
                            "seed": seed,
                            **rollout.summary_row,
                        }
                    )
                    for row in rollout.trace_rows:
                        trace_rows.append(
                            {
                                "env_name": env.name,
                                "noise_pair": noise_spec,
                                "seed": seed,
                                **row,
                            }
                        )

    summary_df = pd.DataFrame(summary_rows)
    trace_df = pd.DataFrame(trace_rows)
    summary_df.to_csv(args.output_dir / "iterated_summary.csv", index=False)
    trace_df.to_csv(args.output_dir / "iterated_trace.csv", index=False)
    grouped = (
        summary_df.groupby(["env_name", "noise_pair", "method"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["env_name", "noise_pair", "method"])
    )
    print(grouped.to_string(index=False))
    print(f"\nSaved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
