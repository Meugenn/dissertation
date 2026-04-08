"""
OkCupid → pipeline adapter.

Parses self-reported zodiac sign, concatenates all 10 essay fields,
scores text with nlp_big5.py, and outputs the standard pipeline CSV:
  doy, E, A, C, N, O, zodiac_idx

NOTE on the zodiac variable:
  OkCupid users self-report their sign. We DON'T have birth date.
  - zodiac_idx is derived from the stated sign name
  - doy is set to the midpoint of that sign's date range (for Phase 3/4/5)
  This is a valid proxy for testing whether the sign taxonomy predicts
  personality, but Phase 5 boundary convergence will be circular by
  construction — interpret with caution.

  Phase 1 (classifier accuracy) and Phase 2 (clustering) are fully valid.
  Phase 3 (boundary specificity) is partially valid.
  Phase 4/5 (boundary convergence) are illustrative only on this dataset.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from nlp_big5 import score_texts
from utils import ZODIAC_SIGNS, ZODIAC_NAMES, doy_to_zodiac


# Date range for each sign [start_doy, end_doy] (non-leap year)
# Used to sample a random DOY within the correct sign range
SIGN_DOY_RANGES = {
    "capricorn":   (356, 365),   # Dec 22–31 tail (Jan 1–19 handled separately)
    "aquarius":    (20,  49),
    "pisces":      (50,  79),
    "aries":       (80,  109),
    "taurus":      (110, 140),
    "gemini":      (141, 171),
    "cancer":      (172, 203),
    "leo":         (204, 234),
    "virgo":       (235, 265),
    "libra":       (266, 295),
    "scorpio":     (296, 325),
    "sagittarius": (326, 355),
}
# Capricorn also covers Jan 1–19 (doy 1–19); combine both ranges
CAPRICORN_RANGES = [(1, 19), (356, 365)]

SIGN_NAME_TO_IDX = {name.lower(): i for i, name in enumerate(ZODIAC_NAMES)}


def parse_sign(raw: str) -> tuple[str | None, int | None]:
    """
    Extract sign name and index from OkCupid sign strings like:
      "gemini and it's fun to think about"
      "virgo but it doesn't matter"
      "scorpio and it matters a lot"
      "pisces"
    Returns (sign_name, sign_idx) or (None, None) if unparseable.
    """
    if not isinstance(raw, str):
        return None, None
    # Decode HTML entities
    raw = raw.replace("&rsquo;", "'").replace("&amp;", "&").lower().strip()
    # First word is the sign name
    first_word = raw.split()[0] if raw else ""
    if first_word in SIGN_NAME_TO_IDX:
        return first_word, SIGN_NAME_TO_IDX[first_word]
    return None, None


def parse_attitude(raw: str) -> int:
    """
    Extract attitude toward astrology from sign string.
    Returns: 1=matters a lot, 2=fun to think about, 3=doesn't matter, 0=unknown
    """
    if not isinstance(raw, str):
        return 0
    raw = raw.lower()
    if "matters a lot" in raw:
        return 1
    if "fun to think about" in raw:
        return 2
    if "doesn" in raw and "matter" in raw:
        return 3
    return 0


def load_okcupid(
    path: str = "data/profiles.csv",
    min_essay_words: int = 20,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load OkCupid profiles, score essays for Big Five, return pipeline-ready DataFrame.

    Returns columns: doy, E, A, C, N, O, zodiac_idx, zodiac_name,
                     attitude (1=matters a lot, 2=fun, 3=doesn't matter),
                     age, sex
    """
    if verbose:
        print(f"Loading {path}...")
    df = pd.read_csv(path, low_memory=False)
    if verbose:
        print(f"  Loaded {len(df):,} profiles")

    # Parse sign
    parsed = df["sign"].apply(parse_sign)
    df["zodiac_name"] = [p[0] for p in parsed]
    df["zodiac_idx"]  = [p[1] for p in parsed]
    df["attitude"]    = df["sign"].apply(parse_attitude)

    # Sample a random DOY uniformly within each sign's date range
    rng = np.random.default_rng(42)

    def sample_doy(sign_name):
        if sign_name == "capricorn":
            # Two ranges: Jan 1–19 and Dec 22–31
            r1_len = 19   # days 1–19
            r2_len = 10   # days 356–365
            if rng.random() < r1_len / (r1_len + r2_len):
                return int(rng.integers(1, 20))
            else:
                return int(rng.integers(356, 366))
        elif sign_name in SIGN_DOY_RANGES:
            lo, hi = SIGN_DOY_RANGES[sign_name]
            return int(rng.integers(lo, hi + 1))
        return np.nan

    df["doy"] = df["zodiac_name"].apply(sample_doy)

    # Drop rows without sign or DOY
    before = len(df)
    df = df.dropna(subset=["zodiac_idx", "doy"])
    df["zodiac_idx"] = df["zodiac_idx"].astype(int)
    df["doy"] = df["doy"].astype(int)
    if verbose:
        print(f"  After sign parsing: {len(df):,} (dropped {before - len(df):,} with missing sign)")

    # Concatenate all essay fields
    essay_cols = [f"essay{i}" for i in range(10)]
    df["all_essays"] = df[essay_cols].fillna("").apply(
        lambda row: " ".join(row.values.astype(str)), axis=1
    )

    # Filter minimum word count
    word_counts = df["all_essays"].str.split().str.len().fillna(0)
    df = df[word_counts >= min_essay_words].copy()
    if verbose:
        print(f"  After essay filter (>={min_essay_words} words): {len(df):,}")

    # Score Big Five from text
    big5 = score_texts(df["all_essays"].reset_index(drop=True), verbose=verbose)
    df = df.reset_index(drop=True)
    for col in ["E", "A", "C", "N", "O"]:
        df[col] = big5[col]

    # Drop rows with NaN Big Five scores
    df = df.dropna(subset=["E", "A", "C", "N", "O"])
    if verbose:
        print(f"  Final sample: {len(df):,} profiles with sign + Big Five scores")
        print(f"\n  Sign distribution:")
        sign_counts = df["zodiac_name"].value_counts()
        for sign in ZODIAC_NAMES:
            n = sign_counts.get(sign.lower(), 0)
            print(f"    {sign:<14} {n:5,}")
        print(f"\n  Attitude breakdown:")
        for att, label in [(1, "Matters a lot"), (2, "Fun to think about"), (3, "Doesn't matter"), (0, "Not specified")]:
            n = (df["attitude"] == att).sum()
            print(f"    {label:<22} {n:5,}  ({n/len(df)*100:.1f}%)")

    # Return only pipeline-needed columns + extras
    return df[["doy", "E", "A", "C", "N", "O", "zodiac_idx", "zodiac_name",
               "attitude", "age", "sex"]].copy()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load and score OkCupid data")
    parser.add_argument("--input", default="data/profiles.csv")
    parser.add_argument("--out", default="data/okcupid_big5.csv")
    parser.add_argument("--min-words", type=int, default=20)
    args = parser.parse_args()

    df = load_okcupid(path=args.input, min_essay_words=args.min_words)

    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df):,} rows → {args.out}")
    print("\nBig Five summary:")
    print(df[["E", "A", "C", "N", "O"]].describe().round(3))


if __name__ == "__main__":
    main()
