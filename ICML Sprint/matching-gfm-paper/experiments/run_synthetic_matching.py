from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from matching_gfm.baselines import fit_pointwise_gbm
from matching_gfm.compact_model import CompactModelConfig, fit_compact_graph_matcher
from matching_gfm.metrics import evaluate_model
from matching_gfm.synthetic_market import generate_synthetic_market


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic matching-GFM validation experiment.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/artifacts"))
    parser.add_argument("--buyers", type=int, default=12)
    parser.add_argument("--sellers", type=int, default=12)
    parser.add_argument("--feature-dim", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--interactions-per-step", type=int, default=6)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--half-life", type=float, default=10.0)
    parser.add_argument("--stability-weight", type=float, default=0.45)
    parser.add_argument("--welfare-weight", type=float, default=0.1)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    market = generate_synthetic_market(
        num_buyers=args.buyers,
        num_sellers=args.sellers,
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        num_steps=args.steps,
        interactions_per_step=args.interactions_per_step,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    baseline = fit_pointwise_gbm(market, half_life=args.half_life, random_state=args.seed)
    compact = fit_compact_graph_matcher(
        market,
        config=CompactModelConfig(
            half_life=args.half_life,
            stability_weight=args.stability_weight,
            welfare_weight=args.welfare_weight,
            maxiter=args.maxiter,
            random_seed=args.seed,
        ),
    )

    baseline_eval = evaluate_model("pointwise_gbm", baseline.buyer_scores, baseline.seller_scores, market)
    compact_eval = evaluate_model("compact_graph_matcher", compact.buyer_scores, compact.seller_scores, market)
    oracle_eval = evaluate_model("oracle_gale_shapley", market.true_buyer_utilities, market.true_seller_utilities, market)

    summary = pd.DataFrame(
        [
            {
                **baseline_eval.summary,
                "train_fit_mse": baseline.train_mse,
                "objective": None,
                "opt_success": None,
            },
            {
                **compact_eval.summary,
                "train_fit_mse": compact.fit_loss,
                "objective": compact.objective,
                "opt_success": compact.success,
                "stability_loss": compact.stability_loss,
                "welfare_term": compact.welfare_term,
                "iterations": compact.iterations,
            },
            {
                **oracle_eval.summary,
                "train_fit_mse": 0.0,
                "objective": 0.0,
                "opt_success": True,
            },
        ]
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    matchings = pd.DataFrame(
        {
            "buyer": list(range(market.num_buyers)),
            "pointwise_gbm": baseline_eval.matching,
            "compact_graph_matcher": compact_eval.matching,
            "oracle_gale_shapley": oracle_eval.matching,
        }
    )
    matchings.to_csv(args.output_dir / "matchings.csv", index=False)

    metadata = {
        "buyers": args.buyers,
        "sellers": args.sellers,
        "feature_dim": args.feature_dim,
        "latent_dim": args.latent_dim,
        "steps": args.steps,
        "interactions_per_step": args.interactions_per_step,
        "train_fraction": args.train_fraction,
        "half_life": args.half_life,
        "stability_weight": args.stability_weight,
        "welfare_weight": args.welfare_weight,
        "maxiter": args.maxiter,
        "seed": args.seed,
        "train_edges": len(market.train_edges),
        "eval_edges": len(market.eval_edges),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
