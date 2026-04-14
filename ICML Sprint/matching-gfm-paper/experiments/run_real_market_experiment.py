from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from matching_gfm.baselines import fit_pointwise_gbm
from matching_gfm.compact_model import CompactModelConfig, fit_compact_graph_matcher
from matching_gfm.dataset_registry import AVAILABLE_SOURCES, load_observed_market
from matching_gfm.real_metrics import evaluate_real_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the matching-GFM scaffold on a real observed market dataset.")
    parser.add_argument("--source", choices=AVAILABLE_SOURCES, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/artifacts/real"))
    parser.add_argument("--half-life", type=float, default=10.0)
    parser.add_argument("--stability-weight", type=float, default=0.45)
    parser.add_argument("--welfare-weight", type=float, default=0.1)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--max-rows", type=int, default=300000)

    parser.add_argument("--max-event-pages", type=int, default=3)
    parser.add_argument("--max-trade-pages", type=int, default=6)
    parser.add_argument("--min-wallet-trades", type=int, default=3)
    parser.add_argument("--min-market-trades", type=int, default=3)
    parser.add_argument("--max-wallets", type=int, default=250)
    parser.add_argument("--max-markets", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_kwargs: dict[str, object]
    if args.source == "polymarket":
        source_kwargs = {
            "max_event_pages": args.max_event_pages,
            "max_trade_pages": args.max_trade_pages,
            "min_wallet_trades": args.min_wallet_trades,
            "min_market_trades": args.min_market_trades,
            "max_wallets": args.max_wallets,
            "max_markets": args.max_markets,
            "snapshot_dir": str(args.output_dir / "snapshot"),
        }
    else:
        if args.data_dir is None:
            raise ValueError("--data-dir is required for --source hm_local")
        source_kwargs = {
            "data_dir": args.data_dir,
            "max_rows": args.max_rows,
        }

    market = load_observed_market(args.source, **source_kwargs)
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

    baseline_eval = evaluate_real_model("pointwise_gbm", baseline.buyer_scores, baseline.seller_scores, market)
    compact_eval = evaluate_real_model("compact_graph_matcher", compact.buyer_scores, compact.seller_scores, market)

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
        ]
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    matchings = pd.DataFrame(
        {
            "buyer_idx": list(range(market.num_buyers)),
            "buyer_id": list(market.buyer_ids),
            "pointwise_gbm": baseline_eval.matching,
            "compact_graph_matcher": compact_eval.matching,
        }
    )
    matchings.to_csv(args.output_dir / "matchings.csv", index=False)

    metadata = {
        "source": args.source,
        "buyers": market.num_buyers,
        "sellers": market.num_sellers,
        "buyer_feature_dim": int(market.buyer_features.shape[1]),
        "seller_feature_dim": int(market.seller_features.shape[1]),
        "train_edges": len(market.train_edges),
        "eval_edges": len(market.eval_edges),
        "market_metadata": market.metadata,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
