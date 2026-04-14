from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ew_lola_core import ipd_spec, iterated_rps_spec, run_two_player_rollout

META_SWAG_EXPERIMENTS = Path("/Users/meuge/coding/maynard/ICML Sprint/meta-swag/experiments")
if str(META_SWAG_EXPERIMENTS) not in sys.path:
    sys.path.append(str(META_SWAG_EXPERIMENTS))

from meta_swag.kim_reference import load_ipd_personas, load_rps_personas  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EW/LOLA experiments against Kim personas.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/kim_personas"))
    parser.add_argument("--personas-per-group", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.18)
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument("--lambda-lola", type=float, default=0.5)
    parser.add_argument("--lambda-power", type=float, default=0.75)
    parser.add_argument("--lambda-offset", type=float, default=5.0)
    parser.add_argument("--opponent-lr", type=float, default=0.25)
    parser.add_argument("--noise-pairs", nargs="+", default=["0.2,0.2", "0.5,0.2"])
    parser.add_argument(
        "--envs",
        nargs="+",
        choices=["ipd", "iterated_rps"],
        default=["ipd", "iterated_rps"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["standard", "ew", "lola", "ew_lola"],
        default=["standard", "ew", "lola", "ew_lola"],
    )
    parser.add_argument("--save-every", type=int, default=8)
    return parser.parse_args()


def parse_noise_pair(spec: str) -> tuple[float, float]:
    left, right = spec.split(",")
    return float(left), float(right)


def select_personas(personas: list[np.ndarray], limit: int) -> list[np.ndarray]:
    return personas[: min(limit, len(personas))]


def persona_tasks(limit: int, envs: set[str]) -> list[tuple[str, str, np.ndarray]]:
    tasks: list[tuple[str, str, np.ndarray]] = []
    if "ipd" in envs:
        for group, bundle in load_ipd_personas(split="test").items():
            for idx, persona in enumerate(select_personas(bundle.personas, limit)):
                tasks.append((f"ipd:{group}:{idx}", "ipd", np.asarray(persona, dtype=float)))
    if "iterated_rps" in envs:
        for group, bundle in load_rps_personas(split="test").items():
            for idx, persona in enumerate(select_personas(bundle.personas, limit)):
                tasks.append((f"rps:{group}:{idx}", "iterated_rps", np.asarray(persona, dtype=float)))
    return tasks


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, float | int | str]] = []
    trace_rows: list[dict[str, float | int | str]] = []
    methods = args.methods
    completed_rollouts = 0

    try:
        for persona_id, env_name, opponent_persona in persona_tasks(args.personas_per_group, set(args.envs)):
            env = ipd_spec() if env_name == "ipd" else iterated_rps_spec()
            per_player_dim = env.num_states * env.num_actions
            if opponent_persona.shape != (env.num_states, env.num_actions):
                raise ValueError(
                    f"Persona shape mismatch for {persona_id}: expected {(env.num_states, env.num_actions)}, got {opponent_persona.shape}"
                )
            opponent_theta = opponent_persona.reshape(-1)

            for noise_spec in args.noise_pairs:
                noise_pair = parse_noise_pair(noise_spec)
                for seed in range(args.seeds):
                    init_rng = np.random.default_rng(seed)
                    learner_theta = init_rng.normal(scale=0.05, size=per_player_dim)
                    theta_init = np.concatenate([learner_theta, opponent_theta], axis=0)

                    for method in methods:
                        rng = np.random.default_rng(seed + 30_000 * (methods.index(method) + 1))
                        rollout = run_two_player_rollout(
                            env=env,
                            method=method,
                            method_p1=method,
                            method_p2="standard",
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
                                "persona_id": persona_id,
                                "env_name": env_name,
                                "noise_pair": noise_spec,
                                "seed": seed,
                                **rollout.summary_row,
                            }
                        )
                        for row in rollout.trace_rows:
                            trace_rows.append(
                                {
                                    "persona_id": persona_id,
                                    "env_name": env_name,
                                    "noise_pair": noise_spec,
                                    "seed": seed,
                                    **row,
                                }
                            )
                        completed_rollouts += 1
                        if completed_rollouts % max(args.save_every, 1) == 0:
                            flush_partial(summary_rows, trace_rows, args.output_dir)
    finally:
        flush_partial(summary_rows, trace_rows, args.output_dir)

    summary_df = pd.DataFrame(summary_rows)
    trace_df = pd.DataFrame(trace_rows)
    summary_df.to_csv(args.output_dir / "kim_persona_summary.csv", index=False)
    trace_df.to_csv(args.output_dir / "kim_persona_trace.csv", index=False)
    grouped = (
        summary_df.groupby(["env_name", "method", "noise_pair"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["env_name", "method", "noise_pair"])
    )
    make_plot(grouped, args.output_dir)
    print(grouped.to_string(index=False))
    print(f"\nSaved artifacts to {args.output_dir}")


def flush_partial(
    summary_rows: list[dict[str, float | int | str]],
    trace_rows: list[dict[str, float | int | str]],
    output_dir: Path,
) -> None:
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(output_dir / "kim_persona_summary.partial.csv", index=False)
    if trace_rows:
        pd.DataFrame(trace_rows).to_csv(output_dir / "kim_persona_trace.partial.csv", index=False)


def make_plot(grouped: pd.DataFrame, output_dir: Path) -> None:
    envs = list(grouped["env_name"].unique())
    methods = ["standard", "ew", "lola", "ew_lola"]
    color_map = {
        "standard": "tab:blue",
        "ew": "tab:green",
        "lola": "tab:orange",
        "ew_lola": "tab:red",
    }
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 4), squeeze=False)
    for idx, env_name in enumerate(envs):
        ax = axes[0, idx]
        sub = grouped[grouped["env_name"] == env_name]
        noise_pairs = list(sub["noise_pair"].unique())
        x = np.arange(len(noise_pairs))
        width = 0.18
        for method_idx, method in enumerate(methods):
            method_sub = sub[sub["method"] == method].set_index("noise_pair").reindex(noise_pairs)
            ax.bar(
                x + (method_idx - 1.5) * width,
                method_sub["mean_reward_p1"],
                width=width,
                label=method,
                color=color_map[method],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(noise_pairs)
        ax.set_title(env_name)
        ax.set_xlabel("Noise pair")
        ax.set_ylabel("Mean learner return")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "kim_persona_summary.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
