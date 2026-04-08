"""
Phase 3: Boundary Specificity — Zodiac vs. Arbitrary Date Partitions

Do the traditional zodiac sign boundaries carry specific predictive information,
or would any 12-way partition of the calendar year perform equally well?

Constructs alternative partitions:
  - calendar months (Jan–Dec, 12 months)
  - school-year-aligned (September-start)
  - 1,000 random contiguous 12-way splits
  - zodiac (traditional boundaries)

For each, runs a reduced Phase 1 (logistic regression CV accuracy).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import BIG5_TRAITS, CHANCE_BASELINE, N_SIGNS, ZODIAC_BOUNDARIES, partition_doy


# --- Partition constructors ---

def zodiac_labels(doy: np.ndarray) -> np.ndarray:
    return partition_doy(doy, ZODIAC_BOUNDARIES)


def calendar_month_labels(doy: np.ndarray) -> np.ndarray:
    # Month start days (approximate, non-leap)
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    return partition_doy(doy, month_starts[1:])  # 11 boundaries


def school_year_labels(doy: np.ndarray) -> np.ndarray:
    # 12 equal 30.4-day segments starting September 1 (day 244)
    start = 244
    boundaries = [(start + i * 30) % 365 + 1 for i in range(1, N_SIGNS)]
    return partition_doy(doy, sorted(boundaries))


def random_contiguous_labels(doy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Pick 11 random cut-points, divide [1,365] into 12 contiguous segments
    cuts = sorted(rng.integers(2, 365, size=N_SIGNS - 1).tolist())
    # Ensure uniqueness
    cuts = sorted(set(cuts))
    while len(cuts) < N_SIGNS - 1:
        cuts.append(rng.integers(2, 365))
        cuts = sorted(set(cuts))
    return partition_doy(doy, cuts[:N_SIGNS - 1])


def partition_accuracy(
    X: np.ndarray,
    labels: np.ndarray,
    cv: int = 5,
    seed: int = 42,
) -> float:
    """CV accuracy of logistic regression predicting partition labels from Big Five."""
    # Skip if degenerate partition (some class < cv members)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < N_SIGNS or counts.min() < cv:
        return np.nan

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, C=1.0, random_state=seed,
                                   solver="lbfgs")),
    ])
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(clf, X, labels, cv=skf)
    return accuracy_score(labels, y_pred)


def run_phase3(
    df: pd.DataFrame,
    n_random: int = 1000,
    cv: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    X = df[BIG5_TRAITS].values.astype(float)
    doy = df["doy"].values.astype(int)
    n = len(df)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase 3: Boundary Specificity")
        print(f"N = {n:,}  |  k = {N_SIGNS}  |  Random partitions = {n_random}")
        print(f"{'='*60}")

    # Fixed partitions
    named_partitions = {
        "zodiac": zodiac_labels(doy),
        "calendar_months": calendar_month_labels(doy),
        "school_year": school_year_labels(doy),
    }

    named_accs = {}
    for name, labels in named_partitions.items():
        acc = partition_accuracy(X, labels, cv=cv, seed=seed)
        named_accs[name] = acc
        if verbose:
            lift = acc / CHANCE_BASELINE if not np.isnan(acc) else float("nan")
            print(f"  {name:<20}: {acc:.4f}  ({lift:.3f}x chance)")

    # Random contiguous partitions
    if verbose:
        print(f"\n→ Running {n_random} random contiguous partitions...")
    rng = np.random.default_rng(seed)
    random_accs = []
    for i in range(n_random):
        labels = random_contiguous_labels(doy, rng)
        acc = partition_accuracy(X, labels, cv=cv, seed=seed)
        if not np.isnan(acc):
            random_accs.append(acc)

    random_accs = np.array(random_accs)
    zodiac_acc = named_accs["zodiac"]

    if len(random_accs) == 0:
        if verbose:
            print("\n→ WARNING: All random partitions produced degenerate classes (insufficient per-class N).")
            print("   This happens when DOY values are discrete (e.g. sign midpoints).")
            print("   Phase 3 is not interpretable for this dataset.")
        return {
            "n": n,
            "named_accuracies": named_accs,
            "random_acc_mean": float("nan"),
            "random_acc_std": float("nan"),
            "random_acc_p5": float("nan"),
            "random_acc_p95": float("nan"),
            "p_zodiac_vs_random": float("nan"),
            "zodiac_percentile": float("nan"),
            "warning": "degenerate_partitions",
        }

    p_zodiac = float(np.mean(random_accs >= zodiac_acc))

    if verbose:
        print(f"\n→ Random partition accuracy: mean={random_accs.mean():.4f}  std={random_accs.std():.4f}")
        print(f"   Zodiac accuracy:           {zodiac_acc:.4f}")
        print(f"   p(random ≥ zodiac):        {p_zodiac:.4f}")

        if p_zodiac < 0.05:
            print("\n→ RESULT: Zodiac boundaries carry SPECIFIC information beyond arbitrary splits.")
        elif p_zodiac < 0.20:
            print("\n→ RESULT: MARGINAL — zodiac slightly outperforms random, but not conclusive.")
        else:
            print("\n→ RESULT: NULL — zodiac performs no better than arbitrary date partitions.")
            print("   Signal (if any) is 'birth timing matters', not 'zodiac boundaries matter'.")

    return {
        "n": n,
        "named_accuracies": named_accs,
        "random_acc_mean": float(random_accs.mean()),
        "random_acc_std": float(random_accs.std()),
        "random_acc_p5": float(np.percentile(random_accs, 5)),
        "random_acc_p95": float(np.percentile(random_accs, 95)),
        "p_zodiac_vs_random": p_zodiac,
        "zodiac_percentile": float(np.mean(random_accs <= zodiac_acc) * 100),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3: boundary specificity")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--n-random", type=int, default=1000)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    summary = run_phase3(df, n_random=args.n_random, cv=args.cv, seed=args.seed)

    out = args.out or args.data.replace(".csv", "_phase3_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
