"""
OpenMed PGC Psychiatric GWAS — Exploration Script
Datasets: https://huggingface.co/collections/OpenMed/pgc-psychiatric-gwas-summary-statistics

Loads a small streaming sample from each condition, profiles schema & summary stats,
then runs cross-disorder correlation on effect sizes at shared SNPs.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

# ── 1. Dataset registry ────────────────────────────────────────────────────────

DATASETS = {
    "adhd":            "OpenMed/pgc-adhd",
    "anxiety":         "OpenMed/pgc-anxiety",
    "autism":          "OpenMed/pgc-autism",
    "bipolar":         "OpenMed/pgc-bipolar",
    "cross_disorder":  "OpenMed/pgc-cross-disorder",
    "eating":          "OpenMed/pgc-eating-disorders",
    "mdd":             "OpenMed/pgc-mdd",
    "ocd_tourette":    "OpenMed/pgc-ocd-tourette",
    "ptsd":            "OpenMed/pgc-ptsd",
    "schizophrenia":   "OpenMed/pgc-schizophrenia",
    "substance_use":   "OpenMed/pgc-substance-use",
}

SAMPLE_N = 100_000   # rows to pull per dataset for exploration

# ── 2. Load samples ────────────────────────────────────────────────────────────

def load_sample(name: str, repo: str, n: int = SAMPLE_N) -> pd.DataFrame:
    print(f"  Loading {name} ({repo})...")
    try:
        ds = load_dataset(repo, split="train", streaming=True, trust_remote_code=True)
        rows = list(ds.take(n))
        df = pd.DataFrame(rows)
        df["_condition"] = name
        print(f"    -> {len(df):,} rows, cols: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"    ERROR loading {name}: {e}")
        return pd.DataFrame()


print("=" * 60)
print("1. LOADING SAMPLES")
print("=" * 60)

samples = {}
for name, repo in DATASETS.items():
    df = load_sample(name, repo)
    if not df.empty:
        samples[name] = df

print(f"\nLoaded {len(samples)}/{len(DATASETS)} datasets successfully\n")

# ── 3. Schema profile ──────────────────────────────────────────────────────────

print("=" * 60)
print("2. SCHEMA & COLUMN PROFILES")
print("=" * 60)

for name, df in samples.items():
    print(f"\n[{name.upper()}] — {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")
    for col in df.columns:
        if col.startswith("_"):
            continue
        dtype = df[col].dtype
        n_null = df[col].isna().sum()
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"  {col:20s} {str(dtype):10s}  null={n_null:5d}  "
                  f"min={df[col].min():.4g}  max={df[col].max():.4g}  "
                  f"mean={df[col].mean():.4g}")
        else:
            n_uniq = df[col].nunique()
            sample_vals = df[col].dropna().astype(str).head(3).tolist()
            print(f"  {col:20s} {str(dtype):10s}  null={n_null:5d}  "
                  f"unique={n_uniq}  sample={sample_vals}")

# ── 4. Genome-wide significance hits ──────────────────────────────────────────

print("\n" + "=" * 60)
print("3. GENOME-WIDE SIGNIFICANT HITS (p < 5e-8)")
print("=" * 60)

P_THRESH = 5e-8
sig_counts = {}

for name, df in samples.items():
    # find p-value column (various naming conventions in GWAS files)
    p_col = next((c for c in df.columns if c.upper() in ("P", "PVAL", "P_VALUE", "P.VALUE")), None)
    if p_col is None:
        print(f"  [{name}] No p-value column found")
        continue
    n_sig = (df[p_col].astype(float) < P_THRESH).sum()
    sig_counts[name] = n_sig
    print(f"  [{name:20s}] {n_sig:5d} / {len(df):,} hits  (p < 5e-8)")

# ── 5. Effect size distribution ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. EFFECT SIZE SUMMARY (BETA or OR)")
print("=" * 60)

for name, df in samples.items():
    beta_col = next((c for c in df.columns if c.upper() in ("BETA", "OR", "EFFECT", "B")), None)
    se_col   = next((c for c in df.columns if c.upper() in ("SE", "SE_BETA", "STD_ERR")), None)
    if beta_col is None:
        print(f"  [{name}] No effect size column found")
        continue
    vals = df[beta_col].astype(float).dropna()
    se_vals = df[se_col].astype(float).dropna() if se_col else None
    print(f"  [{name:20s}] {beta_col}: mean={vals.mean():.4f}  "
          f"std={vals.std():.4f}  |max|={vals.abs().max():.4f}"
          + (f"  SE_mean={se_vals.mean():.4f}" if se_vals is not None else ""))

# ── 6. Cross-disorder correlation at shared SNPs ───────────────────────────────

print("\n" + "=" * 60)
print("5. CROSS-DISORDER EFFECT SIZE CORRELATION (shared SNPs)")
print("=" * 60)

# Build one dataframe per condition with SNP -> BETA
beta_dfs = {}
for name, df in samples.items():
    snp_col  = next((c for c in df.columns if c.upper() in ("SNP", "VARIANT_ID", "RSID", "ID", "MARKERNAME")), None)
    beta_col = next((c for c in df.columns if c.upper() in ("BETA", "OR", "EFFECT", "B")), None)
    if snp_col and beta_col:
        sub = df[[snp_col, beta_col]].copy()
        sub.columns = ["snp", name]
        sub[name] = pd.to_numeric(sub[name], errors="coerce")
        beta_dfs[name] = sub.dropna().set_index("snp")

if len(beta_dfs) >= 2:
    merged = None
    for name, bdf in beta_dfs.items():
        merged = bdf if merged is None else merged.join(bdf, how="inner")

    if merged is not None and len(merged) > 100:
        corr = merged.corr()
        print(f"\nCorrelation matrix across {len(merged):,} shared SNPs:\n")
        print(corr.round(3).to_string())
    else:
        n = len(merged) if merged is not None else 0
        print(f"  Only {n} shared SNPs in sample — try larger SAMPLE_N")
else:
    print("  Not enough datasets with SNP+BETA columns to correlate")

# ── 7. Allele frequency distribution by condition ─────────────────────────────

print("\n" + "=" * 60)
print("6. ALLELE FREQUENCY (MAF) BY CONDITION")
print("=" * 60)

for name, df in samples.items():
    maf_col = next((c for c in df.columns if c.upper() in ("MAF", "FRQ", "A1FREQ", "EAF", "FREQ")), None)
    if maf_col is None:
        continue
    vals = pd.to_numeric(df[maf_col], errors="coerce").dropna()
    # MAF is always <= 0.5
    vals = vals.where(vals <= 0.5, 1 - vals)
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.5]
    counts = pd.cut(vals, bins=bins).value_counts().sort_index()
    print(f"\n  [{name}] MAF distribution ({len(vals):,} variants):")
    for interval, cnt in counts.items():
        bar = "#" * int(40 * cnt / len(vals))
        print(f"    {str(interval):15s}  {cnt:6,}  {bar}")

print("\n" + "=" * 60)
print("EXPLORATION COMPLETE")
print("=" * 60)
print("\nNext steps:")
print("  - run_gwas_analysis.py  : LD-score regression for genetic correlations")
print("  - manhattan.py          : Manhattan + QQ plots per condition")
print("  - polygenic_overlap.py  : Conjunctive FDR / pleiotropy analysis")
