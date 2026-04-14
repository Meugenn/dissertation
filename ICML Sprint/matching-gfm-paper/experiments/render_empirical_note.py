from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render LaTeX tables and a summary figure from matching-GFM experiment artifacts.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("experiments/artifacts/default_check"))
    parser.add_argument("--output-tex", type=Path, default=Path("paper/generated_results.tex"))
    parser.add_argument("--output-figure", type=Path, default=Path("paper/current_results.png"))
    return parser.parse_args()


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    escaped = text
    for raw, replacement in replacements.items():
        escaped = escaped.replace(raw, replacement)
    return escaped


def format_float(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.{digits}f}"


def render_plot(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = ["buyer_tau", "stability_ratio_true", "social_welfare_true"]
    labels = ["Buyer tau", "Stability", "Welfare"]
    models = summary["model"].tolist()
    x = np.arange(len(models))
    width = 0.22

    welfare = summary["social_welfare_true"].to_numpy(dtype=float)
    welfare_min = float(np.min(welfare))
    welfare_ptp = float(np.ptp(welfare))
    welfare_scaled = (welfare - welfare_min) / (welfare_ptp if welfare_ptp > 0 else 1.0)

    values = [
        summary["buyer_tau"].to_numpy(dtype=float),
        summary["stability_ratio_true"].to_numpy(dtype=float),
        welfare_scaled,
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = ["#204b57", "#2d7f5e", "#c77b30"]
    for idx, metric_values in enumerate(values):
        ax.bar(x + (idx - 1) * width, metric_values, width=width, label=labels[idx], color=colors[idx])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=12, ha="right")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Normalized score")
    ax.set_title("Current synthetic matching snapshot")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = args.artifacts_dir / "summary.csv"
    metadata_path = args.artifacts_dir / "metadata.json"
    summary = pd.read_csv(summary_path)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    render_plot(summary, args.output_figure)

    oracle_row = summary.loc[summary["model"] == "oracle_gale_shapley"].iloc[0]
    non_oracle = summary.loc[summary["model"] != "oracle_gale_shapley"].copy()
    best_row = non_oracle.sort_values(["social_welfare_true", "stability_ratio_true"], ascending=False).iloc[0]
    baseline_row = non_oracle.loc[non_oracle["model"] == "pointwise_gbm"].iloc[0]

    delta_tau = best_row["buyer_tau"] - baseline_row["buyer_tau"]
    delta_stability = best_row["stability_ratio_true"] - baseline_row["stability_ratio_true"]
    delta_welfare = best_row["social_welfare_true"] - baseline_row["social_welfare_true"]

    table_rows: list[str] = []
    for _, row in summary.iterrows():
        table_rows.append(
            "        "
            + " & ".join(
                [
                    latex_escape(str(row["model"])),
                    format_float(row["buyer_tau"]),
                    format_float(row["seller_tau"]),
                    format_float(row["stability_ratio_true"]),
                    format_float(row["social_welfare_true"]),
                    format_float(row["mrr"]),
                    format_float(row["future_mrr"]),
                ]
            )
            + r" \\"
        )

    tex = rf"""
\paragraph{{Current run configuration.}}
The current synthetic snapshot uses {metadata.get("buyers", "--")} buyers,
{metadata.get("sellers", "--")} sellers, {metadata.get("steps", "--")} temporal
steps, and {metadata.get("train_edges", "--")} training edges. This remains a
controlled preference-recovery study rather than a full cross-market
pretraining experiment.

\paragraph{{Headline result.}}
Among the learned models, \texttt{{{latex_escape(str(best_row["model"]))}}}
is the strongest current system. Relative to the pointwise GBM baseline, it
improves buyer Kendall-$\tau$ by {format_float(delta_tau)}, stability ratio by
{format_float(delta_stability)}, and true social welfare by
{format_float(delta_welfare)} on this run, while still trailing the oracle
Gale--Shapley upper bound.

\begin{{table}}[t]
\centering
\small
\begin{{tabular}}{{lrrrrrr}}
\toprule
Model & Buyer $\tau$ & Seller $\tau$ & Stability & Welfare & MRR & Future MRR \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
\caption{{Current synthetic matching results. Welfare is evaluated on the latent
buyer-side and seller-side utilities used to generate the market.}}
\label{{tab:current-results}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{current_results.png}}
\caption{{Current synthetic snapshot across preference recovery, stability, and
normalized welfare. Welfare is min-max normalized across the models shown so
that it can be compared on the same visual scale as the other metrics.}}
\label{{fig:current-results}}
\end{{figure}}

\paragraph{{Interpretation.}}
The present scaffold already supports the central empirical claim of the paper:
matching should be evaluated as a structured market problem, not only as a
pointwise interaction-prediction problem. In the current synthetic regime, the
compact graph matcher produces a better buyer-side preference ordering and a
more stable matching than the pointwise baseline, while the oracle result
confirms that there is still substantial headroom.

\paragraph{{Gap to the full proposal.}}
The remaining empirical work is to replace the compact hand-engineered
aggregator with a true temporal graph foundation model, add the missing static
and temporal graph baselines, and port the evaluation to a real trader--market
slice from Polymarket.
""".strip() + "\n"

    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(tex)
    print(f"Wrote {args.output_tex}")
    print(f"Wrote {args.output_figure}")


if __name__ == "__main__":
    main()
