"""
OpenMed PGC Psychiatric GWAS — Exploration v2
Fixes: column normalization, MAF detection, OR outlier filtering,
       MDD/substance_use per-file loading, cross-disorder correlation.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

SAMPLE_N = 200_000

# ── Column aliases (all lower-case) ───────────────────────────────────────────

SNP_COLS   = {"snp", "snpid", "rsid", "id", "markername", "variant_id", "variantid"}
CHR_COLS   = {"chr", "chrom", "chromosome", "hg18chr"}
BP_COLS    = {"bp", "pos", "position", "basepair"}
P_COLS     = {"p", "pval", "p.value", "p-value", "pvalue", "p_value"}
BETA_COLS  = {"beta", "effect", "b", "effect_size"}
OR_COLS    = {"or", "odds_ratio"}
SE_COLS    = {"se", "se_beta", "stderr", "std_err", "stderror"}
Z_COLS     = {"zscore", "z", "z_score", "stat"}
MAF_COLS   = {"maf", "frq", "a1freq", "eaf", "freq", "freq1", "ceuaf", "ceumaf",
              "eur_frq", "a1_freq", "effect_allele_frequency"}
N_COLS     = {"n", "totaln", "weight", "n_total", "ntotal"}

def find_col(df: pd.DataFrame, aliases: set) -> str | None:
    for c in df.columns:
        if c.strip().lower() in aliases:
            return c
    return None


# ── Normalise each dataset into a common schema ───────────────────────────────

def normalise(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["_condition"] = name

    snp = find_col(df, SNP_COLS)
    if snp:
        out["snp"] = df[snp].astype(str).str.upper().str.strip()

    p = find_col(df, P_COLS)
    if p:
        out["p"] = pd.to_numeric(df[p], errors="coerce")

    beta = find_col(df, BETA_COLS)
    or_  = find_col(df, OR_COLS)
    se   = find_col(df, SE_COLS)
    z    = find_col(df, Z_COLS)

    if beta:
        out["effect"] = pd.to_numeric(df[beta], errors="coerce")
        out["effect_type"] = "BETA"
    elif or_:
        raw = pd.to_numeric(df[or_], errors="coerce")
        # clip extreme ORs (log-scale confusion in some files)
        raw = raw.where((raw > 0) & (raw < 100))
        out["effect"] = raw
        out["effect_type"] = "OR"
    elif z:
        out["effect"] = pd.to_numeric(df[z], errors="coerce")
        out["effect_type"] = "Z"

    if se:
        out["se"] = pd.to_numeric(df[se], errors="coerce")

    maf = find_col(df, MAF_COLS)
    if maf:
        vals = pd.to_numeric(df[maf], errors="coerce")
        out["maf"] = vals.where(vals <= 0.5, 1 - vals)

    n = find_col(df, N_COLS)
    if n:
        out["n"] = pd.to_numeric(df[n], errors="coerce")

    chr_ = find_col(df, CHR_COLS)
    bp_  = find_col(df, BP_COLS)
    if chr_ and bp_:
        out["chr"] = pd.to_numeric(df[chr_], errors="coerce")
        out["bp"]  = pd.to_numeric(df[bp_],  errors="coerce")

    return out


# ── Dataset loaders ───────────────────────────────────────────────────────────

DATASETS = {
    "adhd":           "OpenMed/pgc-adhd",
    "anxiety":        "OpenMed/pgc-anxiety",
    "autism":         "OpenMed/pgc-autism",
    "bipolar":        "OpenMed/pgc-bipolar",
    "cross_disorder": "OpenMed/pgc-cross-disorder",
    "eating":         "OpenMed/pgc-eating-disorders",
    "mdd":            "OpenMed/pgc-mdd",
    "ptsd":           "OpenMed/pgc-ptsd",
    "schizophrenia":  "OpenMed/pgc-schizophrenia",
    "substance_use":  "OpenMed/pgc-substance-use",
}

# Datasets with schema conflicts: load file-by-file, take first parseable shard
SCHEMA_CONFLICT = {"mdd", "substance_use"}


def load_sample(name: str, repo: str, n: int = SAMPLE_N) -> pd.DataFrame:
    print(f"  [{name}] loading...")
    try:
        if name in SCHEMA_CONFLICT:
            # Get file list, try each shard until we get enough rows
            from huggingface_hub import list_repo_tree
            files = [
                f.path for f in list_repo_tree(repo, repo_type="dataset")
                if f.path.endswith(".parquet") and "train" in f.path
            ]
            frames = []
            collected = 0
            for f in sorted(files)[:20]:
                try:
                    url = f"hf://datasets/{repo}@main/{f}"
                    chunk = pd.read_parquet(url)
                    frames.append(chunk)
                    collected += len(chunk)
                    if collected >= n:
                        break
                except Exception:
                    continue
            if not frames:
                print(f"    ERROR: no readable shards")
                return pd.DataFrame()
            df = pd.concat(frames, ignore_index=True).head(n)
        else:
            ds = load_dataset(repo, split="train", streaming=True)
            rows = list(ds.take(n))
            if not rows:
                print(f"    ERROR: no rows returned")
                return pd.DataFrame()
            df = pd.DataFrame(rows)

        # skip VCF-format files
        if df.shape[1] <= 2:
            print(f"    SKIP: looks like raw VCF header ({df.shape})")
            return pd.DataFrame()

        norm = normalise(df, name)
        print(f"    -> {len(norm):,} rows | cols: {[c for c in norm.columns if not c.startswith('_')]}")
        return norm

    except Exception as e:
        print(f"    ERROR: {e}")
        return pd.DataFrame()


# ── MAIN ──────────────────────────────────────────────────────────────────────

print("=" * 65)
print("1. LOADING & NORMALISING")
print("=" * 65)

samples: dict[str, pd.DataFrame] = {}
for name, repo in DATASETS.items():
    df = load_sample(name, repo)
    if not df.empty:
        samples[name] = df

print(f"\nLoaded {len(samples)}/{len(DATASETS)} datasets\n")


# ── 2. Effect-size summary ────────────────────────────────────────────────────

print("=" * 65)
print("2. EFFECT SIZE SUMMARY")
print("=" * 65)
print(f"  {'Condition':<20}  {'Type':<5}  {'Mean':>8}  {'Std':>8}  {'|Max|':>8}  {'SE mean':>8}")
print("  " + "-" * 63)
for name, df in samples.items():
    if "effect" not in df.columns:
        print(f"  {name:<20}  {'N/A'}")
        continue
    vals = df["effect"].dropna()
    etype = df["effect_type"].iloc[0] if "effect_type" in df.columns else "?"
    se_mean = df["se"].dropna().mean() if "se" in df.columns else float("nan")
    print(f"  {name:<20}  {etype:<5}  {vals.mean():>8.4f}  {vals.std():>8.4f}  "
          f"{vals.abs().max():>8.4f}  {se_mean:>8.4f}")


# ── 3. GW-significant hits ────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("3. GENOME-WIDE SIGNIFICANT HITS (p < 5e-8)")
print("=" * 65)
print(f"  {'Condition':<20}  {'GW hits':>8}  {'% of sample':>12}  {'Min p':>12}")
print("  " + "-" * 58)

P_THRESH = 5e-8
for name, df in samples.items():
    if "p" not in df.columns:
        print(f"  {name:<20}  {'no p col'}")
        continue
    p = df["p"].dropna()
    sig = (p < P_THRESH).sum()
    min_p = p.min()
    pct = 100 * sig / len(p)
    print(f"  {name:<20}  {sig:>8,}  {pct:>11.3f}%  {min_p:>12.2e}")


# ── 4. Allele frequency profile ───────────────────────────────────────────────

print("\n" + "=" * 65)
print("4. MAF DISTRIBUTION")
print("=" * 65)
bins = [0, 0.01, 0.05, 0.1, 0.2, 0.5]
labels = ["<1%", "1-5%", "5-10%", "10-20%", "20-50%"]

for name, df in samples.items():
    if "maf" not in df.columns:
        continue
    vals = df["maf"].dropna()
    counts = pd.cut(vals, bins=bins, labels=labels).value_counts().sort_index()
    bar_data = [(lbl, cnt, 100*cnt/len(vals)) for lbl, cnt in zip(labels, counts)]
    print(f"\n  [{name}] — {len(vals):,} variants")
    for lbl, cnt, pct in bar_data:
        bar = "█" * int(pct / 2)
        print(f"    {lbl:>8s}  {cnt:7,}  {pct:5.1f}%  {bar}")


# ── 5. Cross-disorder effect correlation ─────────────────────────────────────

print("\n" + "=" * 65)
print("5. CROSS-DISORDER EFFECT CORRELATION")
print("=" * 65)

# Normalise SNP IDs: keep only rsXXXXX format
beta_series: dict[str, pd.Series] = {}
for name, df in samples.items():
    if "snp" not in df.columns or "effect" not in df.columns:
        continue
    # filter to rs-IDs only for matching
    mask = df["snp"].str.startswith("RS")
    sub = df[mask][["snp", "effect"]].dropna()
    # if OR, log-transform to get BETA-like
    if "effect_type" in df.columns and df["effect_type"].iloc[0] == "OR":
        sub = sub.copy()
        sub["effect"] = np.log(sub["effect"].clip(lower=0.01))
    beta_series[name] = sub.set_index("snp")["effect"]

# Pairwise intersections
print(f"\n  Pairwise shared-SNP counts:")
names = list(beta_series.keys())
counts_mat = pd.DataFrame(index=names, columns=names, dtype=int)
for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        shared = beta_series[n1].index.intersection(beta_series[n2].index)
        counts_mat.loc[n1, n2] = len(shared)
print(counts_mat.to_string())

# Build merged frame for correlation using all pairwise shared SNPs
print(f"\n  Correlation matrix (shared rsIDs, log-OR normalised):")
merged = None
for name, s in beta_series.items():
    col = s.rename(name).to_frame()
    merged = col if merged is None else merged.join(col, how="outer")

if merged is not None:
    # keep SNPs present in at least 3 conditions
    min_conditions = 3
    enough = merged.notna().sum(axis=1) >= min_conditions
    merged_sub = merged[enough]
    print(f"  ({len(merged_sub):,} SNPs present in ≥{min_conditions} conditions)\n")

    if len(merged_sub) > 50:
        corr = merged_sub.corr(min_periods=500)
        print(corr.round(3).to_string())
    else:
        print("  Too few shared SNPs. Increase SAMPLE_N or relax min_conditions.")


# ── 6. Sample size summary ────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("6. SAMPLE SIZE BY CONDITION")
print("=" * 65)
for name, df in samples.items():
    if "n" in df.columns:
        n_vals = df["n"].dropna()
        print(f"  [{name:<20}] N: median={n_vals.median():,.0f}  "
              f"min={n_vals.min():,.0f}  max={n_vals.max():,.0f}")
    else:
        print(f"  [{name:<20}] N: not available in sample")


print("\n" + "=" * 65)
print("DONE — Data saved in `samples` dict for interactive use")
print("=" * 65)
print("\nKey findings to explore next:")
print("  1. Manhattan plots per condition")
print("  2. LD-score regression (ldsc) for genetic correlations")
print("  3. Conjunctive FDR for pleiotropy (ADHD+bipolar, schiz+autism)")
print("  4. Polygenic risk score construction")
