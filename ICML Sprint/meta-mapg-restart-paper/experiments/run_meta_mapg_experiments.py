from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["standard_pg", "meta_pg", "lola_style", "meta_mapg"]
METHOD_LABELS = {
    "standard_pg": "PG",
    "meta_pg": "Meta-PG",
    "lola_style": "Peer only",
    "meta_mapg": "Meta-MAPG",
}


@dataclass(frozen=True)
class Game:
    name: str
    payoff_p1: np.ndarray
    payoff_p2: np.ndarray
    horizon: int = 1
    discount: float = 0.96

    @property
    def n_states(self) -> int:
        return 1 if self.horizon == 1 else 5


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def logit(p: float) -> float:
    p = min(max(p, 1e-5), 1.0 - 1e-5)
    return math.log(p / (1.0 - p))


def stag_hunt() -> Game:
    # Action index 0 is cooperate, action index 1 is defect.
    p1 = np.array([[4.0, 0.0], [3.0, 2.0]], dtype=float)
    return Game("stag_hunt", p1, p1.T.copy())


def prisoners_dilemma() -> Game:
    p1 = np.array([[3.0, 0.0], [5.0, 1.0]], dtype=float)
    return Game("ipd", p1, p1.T.copy(), horizon=12, discount=0.96)


def games() -> dict[str, Game]:
    return {"stag_hunt": stag_hunt(), "ipd": prisoners_dilemma()}


def action_index(cooperate: np.ndarray) -> np.ndarray:
    return np.where(cooperate, 0, 1)


def sample_batch(
    theta: np.ndarray,
    game: Game,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample trajectories and return returns, trajectory scores, and log-Hessian diagonals.

    Policies are Bernoulli in every state, with action 1 in the Bernoulli variable
    representing cooperation. The matrix payoffs use action index 0 for cooperation,
    so the sampled actions are converted before rewards are read.
    """

    n_states = game.n_states
    returns = np.zeros((batch_size, 2), dtype=float)
    scores = np.zeros((batch_size, 2, n_states), dtype=float)
    hess_diag = np.zeros((batch_size, 2, n_states), dtype=float)

    if game.horizon == 1:
        p = sigmoid(theta[:, 0])
        coop = rng.random((batch_size, 2)) < p[None, :]
        idx = action_index(coop)
        returns[:, 0] = game.payoff_p1[idx[:, 0], idx[:, 1]]
        returns[:, 1] = game.payoff_p2[idx[:, 0], idx[:, 1]]
        scores[:, 0, 0] = coop[:, 0].astype(float) - p[0]
        scores[:, 1, 0] = coop[:, 1].astype(float) - p[1]
        hess_diag[:, 0, 0] = -p[0] * (1.0 - p[0])
        hess_diag[:, 1, 0] = -p[1] * (1.0 - p[1])
        return returns, scores, hess_diag

    states = np.zeros(batch_size, dtype=int)
    batch_idx = np.arange(batch_size)
    for t in range(game.horizon):
        p0 = sigmoid(theta[0, states])
        p1 = sigmoid(theta[1, states])
        coop0 = rng.random(batch_size) < p0
        coop1 = rng.random(batch_size) < p1
        idx0 = action_index(coop0)
        idx1 = action_index(coop1)
        disc = game.discount**t
        returns[:, 0] += disc * game.payoff_p1[idx0, idx1]
        returns[:, 1] += disc * game.payoff_p2[idx0, idx1]
        np.add.at(scores[:, 0, :], (batch_idx, states), coop0.astype(float) - p0)
        np.add.at(scores[:, 1, :], (batch_idx, states), coop1.astype(float) - p1)
        np.add.at(hess_diag[:, 0, :], (batch_idx, states), -p0 * (1.0 - p0))
        np.add.at(hess_diag[:, 1, :], (batch_idx, states), -p1 * (1.0 - p1))
        states = 1 + 2 * idx0 + idx1

    return returns, scores, hess_diag


@dataclass
class GradientComponents:
    base: np.ndarray
    own: np.ndarray
    peer: np.ndarray
    reward_estimate: np.ndarray


def estimate_components(
    theta: np.ndarray,
    game: Game,
    batch_size: int,
    rng: np.random.Generator,
    inner_lr: float,
) -> GradientComponents:
    returns, scores, hess_diag = sample_batch(theta, game, batch_size, rng)
    dim = game.n_states
    base = np.zeros((2, dim), dtype=float)
    own = np.zeros((2, dim), dtype=float)
    peer = np.zeros((2, dim), dtype=float)

    for player in range(2):
        opp = 1 - player
        r_self = returns[:, player]
        r_opp = returns[:, opp]
        score_self = scores[:, player, :]
        score_opp = scores[:, opp, :]

        g_self = np.mean(r_self[:, None] * score_self, axis=0)
        q_self_wrt_opp = np.mean(r_self[:, None] * score_opp, axis=0)

        h_self = np.einsum("b,bi,bj->ij", r_self, score_self, score_self) / float(batch_size)
        h_self += np.diag(np.mean(r_self[:, None] * hess_diag[:, player, :], axis=0))
        cross_opp_self = np.einsum("b,bi,bj->ij", r_opp, score_opp, score_self) / float(batch_size)

        base[player] = g_self
        own[player] = inner_lr * h_self.T @ g_self
        peer[player] = inner_lr * cross_opp_self.T @ q_self_wrt_opp

    return GradientComponents(
        base=base,
        own=own,
        peer=peer,
        reward_estimate=np.mean(returns, axis=0),
    )


def expected_return(theta: np.ndarray, game: Game) -> np.ndarray:
    if game.horizon == 1:
        p = sigmoid(theta[:, 0])
        probs = np.array([p[0], 1.0 - p[0]])[:, None] * np.array([p[1], 1.0 - p[1]])[None, :]
        return np.array(
            [np.sum(probs * game.payoff_p1), np.sum(probs * game.payoff_p2)],
            dtype=float,
        )

    dist = np.zeros(game.n_states, dtype=float)
    dist[0] = 1.0
    total = np.zeros(2, dtype=float)
    for t in range(game.horizon):
        next_dist = np.zeros_like(dist)
        for state in range(game.n_states):
            if dist[state] == 0.0:
                continue
            p = sigmoid(theta[:, state])
            for c1 in (0, 1):
                for c2 in (0, 1):
                    prob = (p[0] if c1 else 1.0 - p[0]) * (p[1] if c2 else 1.0 - p[1])
                    idx1, idx2 = action_index(np.array([c1, c2], dtype=bool))
                    total[0] += (game.discount**t) * dist[state] * prob * game.payoff_p1[idx1, idx2]
                    total[1] += (game.discount**t) * dist[state] * prob * game.payoff_p2[idx1, idx2]
                    next_state = 1 + 2 * int(idx1) + int(idx2)
                    next_dist[next_state] += dist[state] * prob
        dist = next_dist
    return total


def cooperation_probs(theta: np.ndarray, game: Game) -> np.ndarray:
    probs = sigmoid(theta)
    if game.horizon == 1:
        return probs[:, 0]
    # The initial state is the cleanest finite-horizon proxy for equilibrium selection.
    return probs[:, 0]


def update_from_components(
    comps: GradientComponents,
    method: str,
    peer_coef: float,
    own_coef: float,
) -> np.ndarray:
    update = comps.base.copy()
    if method in {"meta_pg", "meta_mapg"}:
        update += own_coef * comps.own
    if method in {"lola_style", "meta_mapg"}:
        update += peer_coef * comps.peer
    return update


def run_rollout(
    game: Game,
    method: str,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    inner_lr: float,
    peer_coef: float,
    own_coef: float,
    init_theta: np.ndarray | None = None,
    lr_power: float = 0.25,
    lambda_power: float = 0.0,
    log_every: int = 10,
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
    rng = np.random.default_rng(seed)
    theta = (
        init_theta.astype(float).copy()
        if init_theta is not None
        else rng.normal(loc=0.0, scale=1.35, size=(2, game.n_states))
    )
    rows: list[dict[str, float | int | str]] = []

    for step in range(steps):
        comps = estimate_components(theta, game, batch_size, rng, inner_lr)
        lr_step = lr / ((step + 10.0) ** lr_power)
        lambda_step = peer_coef / ((step + 1.0) ** lambda_power)
        update = update_from_components(comps, method, lambda_step, own_coef)
        theta = np.clip(theta + lr_step * update, -8.0, 8.0)

        if step % log_every == 0 or step == steps - 1:
            ret = expected_return(theta, game)
            coop = cooperation_probs(theta, game)
            rows.append(
                {
                    "game": game.name,
                    "method": method,
                    "seed": seed,
                    "step": step,
                    "reward_p1": float(ret[0]),
                    "reward_p2": float(ret[1]),
                    "coop_p1": float(coop[0]),
                    "coop_p2": float(coop[1]),
                    "base_norm": float(np.linalg.norm(comps.base)),
                    "own_norm": float(np.linalg.norm(comps.own)),
                    "peer_norm": float(np.linalg.norm(comps.peer)),
                    "lr_step": float(lr_step),
                    "lambda_step": float(lambda_step),
                }
            )
    return theta, rows


def is_success(theta: np.ndarray, game: Game, threshold: float = 0.82) -> bool:
    coop = cooperation_probs(theta, game)
    return bool(np.min(coop) >= threshold)


def run_ablation(args: argparse.Namespace, outdir: Path) -> pd.DataFrame:
    trace_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for game in games().values():
        for method in METHODS:
            for seed in range(args.seeds):
                theta, rows = run_rollout(
                    game=game,
                    method=method,
                    seed=1000 + 37 * seed,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    inner_lr=args.inner_lr,
                    peer_coef=args.peer_coef,
                    own_coef=args.own_coef,
                    lr_power=args.lr_power,
                    lambda_power=args.lambda_power,
                )
                trace_rows.extend(rows)
                ret = expected_return(theta, game)
                coop = cooperation_probs(theta, game)
                summary_rows.append(
                    {
                        "experiment": "ablation",
                        "game": game.name,
                        "method": method,
                        "seed": seed,
                        "success": int(is_success(theta, game)),
                        "final_reward_mean": float(np.mean(ret)),
                        "final_coop_min": float(np.min(coop)),
                        "final_coop_mean": float(np.mean(coop)),
                    }
                )

    trace = pd.DataFrame(trace_rows)
    summary = pd.DataFrame(summary_rows)
    trace.to_csv(outdir / "ablation_trace.csv", index=False)
    summary.to_csv(outdir / "ablation_summary.csv", index=False)
    return summary


def run_restart(args: argparse.Namespace, outdir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for game in games().values():
        for method in ["standard_pg", "meta_mapg"]:
            for seed in range(args.seeds):
                rng = np.random.default_rng(9000 + 41 * seed)
                success = False
                final_theta = None
                restart_count = args.max_restarts
                for restart in range(args.max_restarts + 1):
                    init_theta = rng.uniform(low=-3.0, high=3.0, size=(2, game.n_states))
                    theta, _ = run_rollout(
                        game=game,
                        method=method,
                        seed=int(rng.integers(0, 2**31 - 1)),
                        steps=args.restart_steps,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        inner_lr=args.inner_lr,
                        peer_coef=args.peer_coef,
                        own_coef=args.own_coef,
                        init_theta=init_theta,
                        lr_power=args.lr_power,
                        lambda_power=0.0,
                    )
                    final_theta = theta
                    if is_success(theta, game, threshold=args.success_threshold):
                        success = True
                        restart_count = restart
                        break
                ret = expected_return(final_theta, game) if final_theta is not None else np.array([np.nan, np.nan])
                rows.append(
                    {
                        "experiment": "restart",
                        "game": game.name,
                        "method": method,
                        "seed": seed,
                        "success": int(success),
                        "restarts": int(restart_count),
                        "final_reward_mean": float(np.mean(ret)),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "restart_summary.csv", index=False)
    return df


def run_basin(args: argparse.Namespace, outdir: Path) -> pd.DataFrame:
    game = stag_hunt()
    rows: list[dict[str, float | int | str]] = []
    grid = np.linspace(0.05, 0.95, args.grid_size)
    for method in ["standard_pg", "meta_mapg"]:
        for i, p1 in enumerate(grid):
            for j, p2 in enumerate(grid):
                init_theta = np.zeros((2, game.n_states), dtype=float)
                init_theta[0, 0] = logit(float(p1))
                init_theta[1, 0] = logit(float(p2))
                theta, _ = run_rollout(
                    game=game,
                    method=method,
                    seed=12000 + 101 * i + 13 * j + (0 if method == "standard_pg" else 1),
                    steps=args.basin_steps,
                    batch_size=args.basin_batch_size,
                    lr=args.lr,
                    inner_lr=args.inner_lr,
                    peer_coef=args.basin_peer_coef,
                    own_coef=args.own_coef,
                    init_theta=init_theta,
                    lr_power=args.lr_power,
                    lambda_power=0.0,
                    log_every=args.basin_steps + 1,
                )
                coop = cooperation_probs(theta, game)
                rows.append(
                    {
                        "experiment": "basin",
                        "game": game.name,
                        "method": method,
                        "init_p1": float(p1),
                        "init_p2": float(p2),
                        "success": int(is_success(theta, game, threshold=args.success_threshold)),
                        "final_coop_min": float(np.min(coop)),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "basin_map.csv", index=False)
    return df


def run_estimator_sanity(args: argparse.Namespace, outdir: Path) -> pd.DataFrame:
    game = stag_hunt()
    theta = np.array([[logit(0.58)], [logit(0.58)]], dtype=float)
    ref_rng = np.random.default_rng(777)
    ref = estimate_components(theta, game, args.reference_batch_size, ref_rng, args.inner_lr)
    ref_update = update_from_components(ref, "meta_mapg", args.peer_coef, args.own_coef).reshape(-1)
    rows: list[dict[str, float | int | str]] = []
    for batch_size in args.sanity_batches:
        estimates = []
        for rep in range(args.sanity_reps):
            rng = np.random.default_rng(16000 + 97 * rep + batch_size)
            comps = estimate_components(theta, game, batch_size, rng, args.inner_lr)
            update = update_from_components(comps, "meta_mapg", args.peer_coef, args.own_coef).reshape(-1)
            estimates.append(update)
        arr = np.stack(estimates, axis=0)
        mean = np.mean(arr, axis=0)
        rows.append(
            {
                "experiment": "estimator_sanity",
                "batch_size": int(batch_size),
                "bias_norm_to_reference": float(np.linalg.norm(mean - ref_update)),
                "variance_trace": float(np.trace(np.cov(arr.T))),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "estimator_sanity.csv", index=False)
    return df


def save_summary_table(ablation: pd.DataFrame, restart: pd.DataFrame, outdir: Path) -> None:
    summary = (
        ablation.groupby(["game", "method"])
        .agg(success_rate=("success", "mean"), reward=("final_reward_mean", "mean"), reward_std=("final_reward_mean", "std"))
        .reset_index()
    )
    restart_summary = (
        restart.groupby(["game", "method"])
        .agg(success_rate=("success", "mean"), restarts=("restarts", "mean"), restarts_std=("restarts", "std"))
        .reset_index()
    )
    with (outdir / "main_results_table.tex").open("w", newline="") as handle:
        handle.write("\\begin{tabular}{llccc}\\toprule\n")
        handle.write("Game & Method & Ablation success & Restart success & Restarts \\\\\\midrule\n")
        for _, row in summary.iterrows():
            game = row["game"].replace("_", " ")
            method = METHOD_LABELS[row["method"]]
            restart_row = restart_summary[
                (restart_summary["game"] == row["game"]) & (restart_summary["method"] == row["method"])
            ]
            if restart_row.empty:
                restart_success = "--"
                restarts = "--"
            else:
                rr = restart_row.iloc[0]
                restart_success = f"{100.0 * rr['success_rate']:.0f}\\%"
                restarts = f"{rr['restarts']:.2f}"
            handle.write(
                f"{game} & {method} & {100.0 * row['success_rate']:.0f}\\% & {restart_success} & {restarts} \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")


def plot_ablation(ablation: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), sharey=True)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for ax, (game_name, game_df) in zip(axes, ablation.groupby("game")):
        grouped = game_df.groupby("method")["success"].mean().reindex(METHODS)
        ax.bar(range(len(METHODS)), grouped.values, color=colors)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(game_name.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Cooperative convergence rate")
    fig.tight_layout()
    fig.savefig(outdir / "ablation_success.pdf")
    fig.savefig(outdir / "ablation_success.png", dpi=180)
    plt.close(fig)


def plot_restart(restart: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    colors = ["#4c78a8", "#b279a2"]
    for ax, (game_name, game_df) in zip(axes, restart.groupby("game")):
        grouped = game_df.groupby("method")["restarts"].mean().reindex(["standard_pg", "meta_mapg"])
        err = game_df.groupby("method")["restarts"].std().reindex(["standard_pg", "meta_mapg"]).fillna(0.0)
        ax.bar(range(2), grouped.values, yerr=err.values, capsize=4, color=colors)
        ax.set_xticks(range(2))
        ax.set_xticklabels(["PG", "Meta-MAPG"])
        ax.set_title(game_name.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Restarts until success")
    fig.tight_layout()
    fig.savefig(outdir / "restart_efficiency.pdf")
    fig.savefig(outdir / "restart_efficiency.png", dpi=180)
    plt.close(fig)


def plot_basin(basin: pd.DataFrame, outdir: Path) -> None:
    methods = ["standard_pg", "meta_mapg"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharex=True, sharey=True)
    for ax, method in zip(axes, methods):
        sub = basin[basin["method"] == method]
        pivot = sub.pivot(index="init_p2", columns="init_p1", values="success").sort_index(ascending=True)
        ax.imshow(
            pivot.values,
            origin="lower",
            extent=[sub["init_p1"].min(), sub["init_p1"].max(), sub["init_p2"].min(), sub["init_p2"].max()],
            vmin=0,
            vmax=1,
            cmap="YlGnBu",
            aspect="equal",
        )
        rate = sub["success"].mean()
        ax.set_title(f"{METHOD_LABELS[method]} ({100.0 * rate:.1f}%)")
        ax.set_xlabel("Initial p1(C)")
        ax.grid(color="white", alpha=0.15)
    axes[0].set_ylabel("Initial p2(C)")
    fig.tight_layout()
    fig.savefig(outdir / "basin_maps.pdf")
    fig.savefig(outdir / "basin_maps.png", dpi=180)
    plt.close(fig)


def plot_sanity(sanity: pd.DataFrame, outdir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(4.8, 3.0))
    ax1.plot(sanity["batch_size"], sanity["bias_norm_to_reference"], marker="o", color="#4c78a8")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Bias to high-batch reference", color="#4c78a8")
    ax1.tick_params(axis="y", labelcolor="#4c78a8")
    ax2 = ax1.twinx()
    ax2.plot(sanity["batch_size"], sanity["variance_trace"], marker="s", color="#f58518")
    ax2.set_ylabel("Estimator variance trace", color="#f58518")
    ax2.tick_params(axis="y", labelcolor="#f58518")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "estimator_sanity.pdf")
    fig.savefig(outdir / "estimator_sanity.png", dpi=180)
    plt.close(fig)


def write_manifest(args: argparse.Namespace, outdir: Path) -> None:
    with (outdir / "run_config.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        for key, value in sorted(vars(args).items()):
            writer.writerow([key, value])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample-based Meta-MAPG restart experiments.")
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/main"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--restart-steps", type=int, default=120)
    parser.add_argument("--max-restarts", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--basin-batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=0.9)
    parser.add_argument("--lr-power", type=float, default=0.24)
    parser.add_argument("--inner-lr", type=float, default=0.55)
    parser.add_argument("--peer-coef", type=float, default=1.2)
    parser.add_argument("--basin-peer-coef", type=float, default=1.75)
    parser.add_argument("--own-coef", type=float, default=0.35)
    parser.add_argument("--lambda-power", type=float, default=0.0)
    parser.add_argument("--success-threshold", type=float, default=0.82)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--basin-steps", type=int, default=140)
    parser.add_argument("--reference-batch-size", type=int, default=120000)
    parser.add_argument("--sanity-reps", type=int, default=80)
    parser.add_argument("--sanity-batches", type=int, nargs="+", default=[48, 96, 192, 384, 768])
    parser.add_argument("--skip-basin", action="store_true")
    parser.add_argument("--skip-sanity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    write_manifest(args, outdir)

    ablation = run_ablation(args, outdir)
    restart = run_restart(args, outdir)
    plot_ablation(ablation, outdir)
    plot_restart(restart, outdir)

    if not args.skip_basin:
        basin = run_basin(args, outdir)
        plot_basin(basin, outdir)
    if not args.skip_sanity:
        sanity = run_estimator_sanity(args, outdir)
        plot_sanity(sanity, outdir)

    save_summary_table(ablation, restart, outdir)


if __name__ == "__main__":
    main()
