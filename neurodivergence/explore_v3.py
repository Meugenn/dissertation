"""
OpenMed PGC Psychiatric GWAS — Exploration v3
All 10 conditions loaded. Schema-conflicted datasets (mdd, schizophrenia,
substance_use) use per-shard hf_hub_download to bypass schema enforcement.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_tree
import warnings
warnings.filterwarnings("ignore")

SAMPLE_N = 200_000

# ── Column aliases ─────────────────────────────────────────────────────────────

SNP_COLS  = {"snp","snpid","rsid","id","markername","variant_id","variantid"}
CHR_COLS  = {"chr","chrom","chromosome","hg18chr","chromosome"}
BP_COLS   = {"bp","pos","position","basepair"}
P_COLS    = {"p","pval","p.value","p-value","pvalue","p_value"}
BETA_COLS = {"beta","effect","b","effect_size","logor"}   # logOR is BETA-equivalent
OR_COLS   = {"or","odds_ratio"}
SE_COLS   = {"se","se_beta","stderr","std_err","stderror"}
Z_COLS    = {"zscore","z","z_score","stat"}
MAF_COLS  = {"maf","frq","a1freq","eaf","freq","freq1","ceuaf","ceumaf",
             "eur_frq","a1_freq","eaf","effect_allele_frequency"}
N_COLS    = {"n","totaln","weight","n_total","ntotal","n_study"}

def find_col(df: pd.DataFrame, aliases: set) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in low:
            return low[a]
    # also try prefix match for FRQ_A_XXXXX style
    if aliases is MAF_COLS:
        for c in df.columns:
            if c.lower().startswith("frq_a_") or c.lower().startswith("frq_u_"):
                return c
    return None


def normalise(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
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


# ── Per-shard loader for schema-conflicted repos ───────────────────────────────

def load_shards_direct(repo: str, sub_study: str, n: int = SAMPLE_N) -> pd.DataFrame:
    """Download parquet shards individually, rename FRQ_A_XXXXX -> FRQ_A etc."""
    try:
        items = list(list_repo_tree(repo, repo_type="dataset",
                                    path_in_repo=f"data/{sub_study}"))
    except Exception:
        return pd.DataFrame()

    files = sorted([x.path for x in items if getattr(x, "path", "").endswith(".parquet")])
    frames = []
    collected = 0

    for fpath in files:
        if collected >= n:
            break
        try:
            local = hf_hub_download(repo_id=repo, filename=fpath, repo_type="dataset")
            chunk = pd.read_parquet(local)
            # normalise variable-suffix frequency columns
            rename = {}
            for c in chunk.columns:
                lc = c.lower()
                if lc.startswith("frq_a_"):
                    rename[c] = "FRQ_A"
                elif lc.startswith("frq_u_"):
                    rename[c] = "FRQ_U"
            chunk = chunk.rename(columns=rename)
            frames.append(chunk)
            collected += len(chunk)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).head(n)


# ── Dataset registry ───────────────────────────────────────────────────────────

# (name, hf_repo, sub_study_for_direct_load_or_None)
DATASETS = [
    ("adhd",            "OpenMed/pgc-adhd",            None),
    ("anxiety",         "OpenMed/pgc-anxiety",          None),
    ("autism",          "OpenMed/pgc-autism",           None),
    ("bipolar",         "OpenMed/pgc-bipolar",          None),
    ("cross_disorder",  "OpenMed/pgc-cross-disorder",   None),
    ("eating",          "OpenMed/pgc-eating-disorders", None),
    ("mdd",             "OpenMed/pgc-mdd",              "mdd2023diverse"),
    ("ptsd",            "OpenMed/pgc-ptsd",             None),
    ("schizophrenia",   "OpenMed/pgc-schizophrenia",    "scz2022"),
    ("substance_use",   "OpenMed/pgc-substance-use",    "SUD2023"),
]


def load_sample(name: str, repo: str, sub: str | None, n: int = SAMPLE_N) -> pd.DataFrame:
    print(f"  [{name}] loading...", end=" ", flush=True)
    try:
        if sub:
            if name == "mdd":
                # mdd2023diverse has 'data_dir' support via streaming
                ds = load_dataset(repo, data_dir=f"data/{sub}", split="train", streaming=True)
                rows = list(ds.take(n))
                df = pd.DataFrame(rows)
            else:
                df = load_shards_direct(repo, sub, n)
        else:
            ds = load_dataset(repo, split="train", streaming=True)
            rows = list(ds.take(n))
            df = pd.DataFrame(rows) if rows else pd.DataFrame()

        if df.empty or df.shape[1] <= 2:
            print("SKIP (VCF or empty)")
            return pd.DataFrame()

        norm = normalise(df, name)
        cols = [c for c in norm.columns if not c.startswith("_")]
        print(f"{len(norm):,} rows | {cols}")
        return norm

    except Exception as e:
        print(f"ERROR: {str(e)[:120]}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("1. LOADING & NORMALISING")
print("=" * 70)

samples: dict[str, pd.DataFrame] = {}
for name, repo, sub in DATASETS:
    df = load_sample(name, repo, sub)
    if not df.empty:
        samples[name] = df

print(f"\nLoaded {len(samples)}/{len(DATASETS)} datasets\n")


# ── 2. Effect-size summary ─────────────────────────────────────────────────────

print("=" * 70)
print("2. EFFECT SIZE SUMMARY")
print("=" * 70)
print(f"  {'Condition':<20} {'Type':<5} {'Mean':>8} {'Std':>8} {'|Max|':>8} {'SE':>8}")
print("  " + "-" * 65)
for name, df in samples.items():
    if "effect" not in df.columns:
        print(f"  {name:<20} {'N/A'}")
        continue
    vals = df["effect"].dropna()
    etype = df["effect_type"].iloc[0] if "effect_type" in df.columns else "?"
    se_m  = df["se"].dropna().mean() if "se" in df.columns else float("nan")
    print(f"  {name:<20} {etype:<5} {vals.mean():>8.4f} {vals.std():>8.4f} "
          f"{vals.abs().max():>8.4f} {se_m:>8.4f}")


# ── 3. GW significant hits ─────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("3. GENOME-WIDE SIGNIFICANT HITS (p < 5e-8)")
print("=" * 70)
print(f"  {'Condition':<20} {'GW hits':>8} {'% sample':>10} {'Min p':>12}")
print("  " + "-" * 55)
P_THRESH = 5e-8
for name, df in samples.items():
    if "p" not in df.columns:
        print(f"  {name:<20} {'no p col'}")
        continue
    p = df["p"].dropna()
    sig   = (p < P_THRESH).sum()
    min_p = p.min()
    pct   = 100 * sig / len(p)
    print(f"  {name:<20} {sig:>8,} {pct:>9.3f}% {min_p:>12.2e}")


# ── 4. MAF distribution ────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("4. MAF DISTRIBUTION")
print("=" * 70)
bins   = [0, 0.01, 0.05, 0.1, 0.2, 0.5]
labels = ["<1%", "1-5%", "5-10%", "10-20%", "20-50%"]

for name, df in samples.items():
    if "maf" not in df.columns:
        continue
    vals   = df["maf"].dropna()
    counts = pd.cut(vals, bins=bins, labels=labels).value_counts().sort_index()
    print(f"\n  [{name}] {len(vals):,} variants")
    for lbl, cnt in zip(labels, counts):
        pct = 100 * cnt / len(vals)
        bar = "█" * int(pct / 2)
        print(f"    {lbl:>8s}  {cnt:7,}  {pct:5.1f}%  {bar}")


# ── 5. Cross-disorder effect correlation ──────────────────────────────────────

print("\n" + "=" * 70)
print("5. CROSS-DISORDER EFFECT CORRELATION (shared rsIDs, log-OR scale)")
print("=" * 70)

beta_series: dict[str, pd.Series] = {}
for name, df in samples.items():
    if "snp" not in df.columns or "effect" not in df.columns:
        continue
    mask = df["snp"].str.startswith("RS")
    sub  = df[mask][["snp", "effect"]].dropna()
    if "effect_type" in df.columns and df["effect_type"].iloc[0] == "OR":
        sub = sub.copy()
        sub["effect"] = np.log(sub["effect"].clip(lower=0.01))
    beta_series[name] = sub.set_index("snp")["effect"]

# Pairwise shared-SNP counts
names = sorted(beta_series.keys())
count_df = pd.DataFrame(index=names, columns=names, dtype=float)
for n1 in names:
    for n2 in names:
        count_df.loc[n1, n2] = len(beta_series[n1].index.intersection(beta_series[n2].index))

print("\nPairwise shared-SNP counts:")
print(count_df.astype(int).to_string())

# Correlation (SNPs present in ≥3 conditions)
merged = None
for name, s in beta_series.items():
    col    = s.rename(name).to_frame()
    merged = col if merged is None else merged.join(col, how="outer")

if merged is not None:
    min_c    = 3
    merged_s = merged[merged.notna().sum(axis=1) >= min_c]
    print(f"\nCorrelation matrix ({len(merged_s):,} SNPs in ≥{min_c} conditions):\n")
    if len(merged_s) > 100:
        corr = merged_s.corr(min_periods=500)
        print(corr.round(3).to_string())


# ── 6. Top genome-wide hits across all conditions ─────────────────────────────

print("\n" + "=" * 70)
print("6. TOP GENOME-WIDE HITS ACROSS ALL CONDITIONS")
print("=" * 70)
top_rows = []
for name, df in samples.items():
    if "p" not in df.columns:
        continue
    avail = [c for c in ["snp", "chr", "bp", "p", "effect", "effect_type"] if c in df.columns]
    top = df.nsmallest(5, "p")[avail].copy()
    top["condition"] = name
    top_rows.append(top)

if top_rows:
    top_all = pd.concat(top_rows).sort_values("p")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(top_all.to_string(index=False))


print("\n" + "=" * 70)
print("EXPLORATION COMPLETE")
print("=" * 70)
