from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from meta_swag.configs import DEFAULT_CONFIG, REPORT_EXPERIMENT_1_VARIANCES
from meta_swag.games import default_games
from meta_swag.metrics import evaluate_metrics
from meta_swag.posterior import fit_meta_swag
from meta_swag.simulate import simulate_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Meta-SWAG matrix-game experiments.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seeds", type=int, default=DEFAULT_CONFIG.seeds)
    parser.add_argument("--steps", type=int, default=DEFAULT_CONFIG.steps)
    parser.add_argument("--burn-in", type=int, default=DEFAULT_CONFIG.burn_in)
    parser.add_argument("--posterior-samples", type=int, default=DEFAULT_CONFIG.posterior_samples)
    parser.add_argument("--beta", type=float, default=DEFAULT_CONFIG.beta)
    parser.add_argument("--rank", type=int, default=DEFAULT_CONFIG.rank)
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.lr)
    parser.add_argument(
        "--weighting-schemes",
        nargs="+",
        default=["softmax", "ess", "threshold"],
        choices=["softmax", "ess", "threshold"],
    )
    parser.add_argument("--target-ess-fraction", type=float, default=0.5)
    parser.add_argument("--threshold-quantile", type=float, default=0.75)
    return parser.parse_args()


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    grouped = df.groupby(["game", "variance_setting", "weighting_scheme"], as_index=False).mean(numeric_only=True)
    color_map = {"softmax": "tab:blue", "ess": "tab:green", "threshold": "tab:red"}
    for scheme, sub in grouped.groupby("weighting_scheme"):
        axes[0].scatter(sub["hm_am_ratio"], sub["variance_ratio"], s=60, label=scheme, color=color_map.get(scheme))
    for _, row in grouped.iterrows():
        label = f"{row['game']} {row['variance_setting']} {row['weighting_scheme']}"
        axes[0].annotate(label, (row["hm_am_ratio"], row["variance_ratio"]), fontsize=7)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_xlabel("Predicted HM/AM ratio")
    axes[0].set_ylabel("Empirical posterior/point variance ratio")
    axes[0].set_title("Experiment 1 variance reduction")
    axes[0].legend(fontsize=8)

    for scheme, sub in grouped.groupby("weighting_scheme"):
        axes[1].scatter(
            sub["basin_proxy"],
            sub["posterior_top_eigenvalue"],
            s=60,
            label=scheme,
            color=color_map.get(scheme),
        )
    for _, row in grouped.iterrows():
        label = f"{row['game']} {row['variance_setting']} {row['weighting_scheme']}"
        axes[1].annotate(label, (row["basin_proxy"], row["posterior_top_eigenvalue"]), fontsize=7)
    axes[1].set_xlabel("Basin proxy (mean inverse positive curvature)")
    axes[1].set_ylabel("Top posterior eigenvalue")
    axes[1].set_title("Posterior geometry")

    fig.tight_layout()
    fig.savefig(output_dir / "matrix_games_summary.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    eigen_rows = []
    games = [game for game in default_games() if game.name in {"matching_pennies", "rock_paper_scissors"}]
    for game in games:
        for variance_pair in REPORT_EXPERIMENT_1_VARIANCES:
            variance_label = f"[{variance_pair[0]:.0f},{variance_pair[1]:.0f}]"
            for weighting_scheme in args.weighting_schemes:
                for seed in range(args.seeds):
                    rng = np.random.default_rng(seed)
                    trajectory = simulate_trajectory(
                        game=game,
                        rng=rng,
                        steps=args.steps,
                        burn_in=args.burn_in,
                        lr=args.lr,
                        noise_variance_pair=variance_pair,
                    )
                    posterior = fit_meta_swag(
                        trajectory.checkpoints,
                        trajectory.evidence_scores,
                        beta=args.beta,
                        rank=args.rank,
                        weighting_scheme=weighting_scheme,
                        target_ess_fraction=args.target_ess_fraction,
                        threshold_quantile=args.threshold_quantile,
                    )
                    metrics = evaluate_metrics(
                        game=game,
                        posterior=posterior,
                        checkpoints=trajectory.checkpoints,
                        noise_variance_pair=trajectory.noise_variance_pair,
                        num_samples=min(args.posterior_samples, len(trajectory.checkpoints)),
                        rng=rng,
                    )
                    row = metrics.__dict__.copy()
                    row["seed"] = seed
                    row["variance_setting"] = variance_label
                    row["weighting_scheme"] = weighting_scheme
                    row["effective_sample_size"] = posterior.effective_sample_size
                    row["resolved_beta"] = posterior.beta
                    row["threshold"] = posterior.threshold
                    rows.append(row)

                    eigenvalues = np.linalg.eigvalsh(posterior.deviations.T @ posterior.deviations)
                    top_five = np.sort(eigenvalues)[-5:][::-1]
                    for idx, value in enumerate(top_five, start=1):
                        eigen_rows.append(
                            {
                                "game": game.name,
                                "variance_setting": variance_label,
                                "weighting_scheme": weighting_scheme,
                                "seed": seed,
                                "eigen_rank": idx,
                                "eigenvalue": float(value),
                            }
                        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "matrix_games_metrics.csv", index=False)
    make_plots(df, output_dir)
    pd.DataFrame(eigen_rows).to_csv(output_dir / "matrix_games_eigenvalues.csv", index=False)

    summary = df.groupby(["game", "variance_setting", "weighting_scheme"], as_index=False).mean(numeric_only=True)
    summary.to_csv(output_dir / "matrix_games_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
