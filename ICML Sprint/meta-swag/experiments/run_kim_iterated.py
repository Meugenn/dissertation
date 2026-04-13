from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from meta_swag.iterated_games import (
    cooperation_rate,
    discounted_return,
    ipd_spec,
    rps_spec,
    simulate_iterated_adaptation,
)
from meta_swag.kim_reference import (
    PersonaBundle,
    load_ipd_personas,
    load_rps_personas,
    policy_from_persona_logits,
)
from meta_swag.posterior import fit_meta_swag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kim-style iterated Meta-SWAG experiments.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--personas-per-group", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--burn-in", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument("--grad-noise-std", type=float, default=0.15)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--weighting-schemes",
        nargs="+",
        default=["softmax", "ess", "threshold"],
        choices=["softmax", "ess", "threshold"],
    )
    parser.add_argument("--target-ess-fraction", type=float, default=0.5)
    parser.add_argument("--threshold-quantile", type=float, default=0.75)
    return parser.parse_args()


def select_personas(bundle: PersonaBundle, limit: int) -> list[np.ndarray]:
    return bundle.personas[: min(limit, len(bundle.personas))]


def load_persona_tasks(limit: int) -> list[tuple[str, str, np.ndarray]]:
    tasks: list[tuple[str, str, np.ndarray]] = []
    for group, bundle in load_ipd_personas(split="test").items():
        for idx, persona in enumerate(select_personas(bundle, limit)):
            tasks.append((f"ipd:{group}:{idx}", "ipd", policy_from_persona_logits(persona)))
    for group, bundle in load_rps_personas(split="test").items():
        for idx, persona in enumerate(select_personas(bundle, limit)):
            tasks.append((f"rps:{group}:{idx}", "rps", policy_from_persona_logits(persona)))
    return tasks


def evaluate_map_baseline(
    checkpoints: np.ndarray,
    evidence_scores: np.ndarray,
    opponent_policy: np.ndarray,
    env_name: str,
    gamma: float,
) -> dict[str, float | str | None]:
    spec = ipd_spec() if env_name == "ipd" else rps_spec()
    best_idx = int(np.argmax(evidence_scores))
    best_theta = checkpoints[best_idx]
    row: dict[str, float | str | None] = {
        "weighting_scheme": "map",
        "objective_mean": discounted_return(best_theta, opponent_policy, spec, gamma),
        "objective_std": 0.0,
        "effective_sample_size": 1.0,
        "resolved_beta": None,
        "threshold": None,
        "top_posterior_eigenvalue": 0.0,
        "cooperation_rate": None,
    }
    if env_name == "ipd":
        row["cooperation_rate"] = cooperation_rate(best_theta, opponent_policy, spec, gamma)
    return row


def posterior_snapshot_rows(
    checkpoints: np.ndarray,
    scores: np.ndarray,
    args: argparse.Namespace,
    env_name: str,
    persona_id: str,
    seed: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    snapshot_indices = np.linspace(5, len(checkpoints), num=5, dtype=int)
    snapshot_indices = np.unique(np.clip(snapshot_indices, 2, len(checkpoints)))
    for snapshot_id, end_idx in enumerate(snapshot_indices, start=1):
        posterior = fit_meta_swag(
            checkpoints[:end_idx],
            scores[:end_idx],
            beta=args.beta,
            rank=min(args.rank, end_idx),
            weighting_scheme="ess",
            target_ess_fraction=args.target_ess_fraction,
            threshold_quantile=args.threshold_quantile,
        )
        eigenvalues = np.linalg.eigvalsh(posterior.deviations.T @ posterior.deviations)
        top_five = np.sort(eigenvalues)[-5:][::-1]
        for rank_idx, value in enumerate(top_five, start=1):
            rows.append(
                {
                    "env_name": env_name,
                    "persona_id": persona_id,
                    "seed": seed,
                    "snapshot_id": snapshot_id,
                    "retained_checkpoints": int(end_idx),
                    "eigen_rank": rank_idx,
                    "eigenvalue": float(value),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, float | str | None | int]] = []
    eigen_rows: list[dict[str, float | str | int]] = []

    for persona_id, env_name, opponent_policy in load_persona_tasks(args.personas_per_group):
        spec = ipd_spec() if env_name == "ipd" else rps_spec()
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            trajectory = simulate_iterated_adaptation(
                opponent_policy=opponent_policy,
                spec=spec,
                rng=rng,
                steps=args.steps,
                burn_in=args.burn_in,
                lr=args.lr,
                grad_noise_std=args.grad_noise_std,
                gamma=args.gamma,
            )

            base_row = {
                "env_name": env_name,
                "persona_id": persona_id,
                "seed": seed,
                "final_objective": float(trajectory.objective_trace[-1]),
                "best_checkpoint_objective": float(np.max(trajectory.evidence_scores)),
            }
            metric_rows.append(base_row | evaluate_map_baseline(
                trajectory.checkpoints,
                trajectory.evidence_scores,
                opponent_policy,
                env_name,
                args.gamma,
            ))

            for weighting_scheme in args.weighting_schemes:
                posterior = fit_meta_swag(
                    trajectory.checkpoints,
                    trajectory.evidence_scores,
                    beta=args.beta,
                    rank=args.rank,
                    weighting_scheme=weighting_scheme,
                    target_ess_fraction=args.target_ess_fraction,
                    threshold_quantile=args.threshold_quantile,
                )
                samples = posterior.sample(num_samples=min(12, len(trajectory.checkpoints)), rng=rng)
                sample_returns = np.array(
                    [discounted_return(sample, opponent_policy, spec, args.gamma) for sample in samples]
                )
                row: dict[str, float | str | None | int] = {
                    **base_row,
                    "weighting_scheme": weighting_scheme,
                    "objective_mean": float(np.mean(sample_returns)),
                    "objective_std": float(np.std(sample_returns)),
                    "effective_sample_size": float(posterior.effective_sample_size),
                    "resolved_beta": posterior.beta,
                    "threshold": posterior.threshold,
                    "top_posterior_eigenvalue": float(np.linalg.eigvalsh(posterior.covariance)[-1]),
                    "cooperation_rate": None,
                }
                if env_name == "ipd":
                    row["cooperation_rate"] = float(
                        np.mean([cooperation_rate(sample, opponent_policy, spec, args.gamma) for sample in samples])
                    )
                metric_rows.append(row)

            eigen_rows.extend(
                posterior_snapshot_rows(
                    trajectory.checkpoints,
                    trajectory.evidence_scores,
                    args,
                    env_name=env_name,
                    persona_id=persona_id,
                    seed=seed,
                )
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "kim_iterated_metrics.csv", index=False)

    numeric_cols = [
        "final_objective",
        "best_checkpoint_objective",
        "objective_mean",
        "objective_std",
        "effective_sample_size",
        "resolved_beta",
        "threshold",
        "top_posterior_eigenvalue",
        "cooperation_rate",
    ]
    summary = (
        metrics.groupby(["env_name", "weighting_scheme"], as_index=False)[numeric_cols]
        .mean()
        .sort_values(["env_name", "weighting_scheme"])
    )
    summary.to_csv(output_dir / "kim_iterated_summary.csv", index=False)
    pd.DataFrame(eigen_rows).to_csv(output_dir / "kim_iterated_eigenvalues.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
