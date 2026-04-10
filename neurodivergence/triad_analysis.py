"""
ADHD × Bipolar × Autism — Pleiotropy & Shared Genetic Architecture
Downloads latest sub-studies, merges on SNP, runs conjunctive FDR,
generates all figures for report section and Streamlit tab.
"""

import glob, os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from huggingface_hub import hf_hub_download, list_repo_tree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)
os.makedirs("data_cache", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Palatino", "Georgia"],
    "font.size": 11, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})

COLORS = {"adhd": "#4C9BE8", "bipolar": "#E76F51", "autism": "#2A9D8F"}
NAMES  = {"adhd": "ADHD (2022)", "bipolar": "Bipolar (2021)", "autism": "Autism (2019)"}

# ── Known shared loci (from literature) ───────────────────────────────────────
# (gene, chr, approx_bp_hg19, conditions_implicated, function_note)
KNOWN_LOCI = [
    ("CACNA1C",  12, 2_394_000,  ["bipolar","adhd","autism"],
     "L-type Ca²⁺ channel; synaptic plasticity, mood regulation"),
    ("ANK3",      10, 61_754_000, ["bipolar","adhd"],
     "Ankyrin G; axon initial segment, neuronal excitability"),
    ("SHANK3",    22, 51_100_000, ["autism","bipolar"],
     "Postsynaptic density scaffold; glutamate signalling"),
    ("RBFOX1",    16, 5_236_000,  ["adhd","autism"],
     "RNA-binding; neuronal splicing regulator"),
    ("DRD4",      11, 636_000,    ["adhd","bipolar"],
     "Dopamine D4 receptor; reward, novelty-seeking"),
    ("NRXN1",      2, 50_145_000, ["autism","bipolar","adhd"],
     "Neurexin-1; synaptic adhesion, excitatory/inhibitory balance"),
    ("KDM5B",      1, 202_740_000,["autism","adhd"],
     "Histone demethylase; neurodevelopmental gene regulation"),
    ("MBD5",       2, 148_900_000,["autism","adhd"],
     "Methyl-CpG binding; chromatin organisation"),
    ("DYRK1A",    21, 38_740_000, ["autism","adhd"],
     "Kinase; neuronal proliferation, Down syndrome pathway"),
]


# ── Download & cache shards ───────────────────────────────────────────────────

def fetch_shards(repo_id: str, sub: str, n_shards: int = 20,
                 cache_file: str = None) -> pd.DataFrame:
    if cache_file and os.path.exists(cache_file):
        print(f"  loading from cache: {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"  downloading {repo_id}/{sub} ({n_shards} shards)...")
    items = list(list_repo_tree(repo_id, repo_type="dataset",
                                path_in_repo=f"data/{sub}"))
    files = sorted([x.path for x in items
                    if getattr(x, "path", "").endswith(".parquet")])[:n_shards]
    frames = []
    for f in files:
        try:
            local = hf_hub_download(repo_id=repo_id, filename=f,
                                    repo_type="dataset")
            chunk = pd.read_parquet(local)
            # normalise FRQ_A_XXXXX columns
            rename = {c: "FRQ_A" for c in chunk.columns
                      if c.lower().startswith("frq_a_")}
            rename.update({c: "FRQ_U" for c in chunk.columns
                           if c.lower().startswith("frq_u_")})
            chunk = chunk.rename(columns=rename)
            frames.append(chunk)
        except Exception as e:
            print(f"    skip {f}: {e}")

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if cache_file:
        df.to_parquet(cache_file)
    return df


def normalise(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Produce standard columns: snp, chr, bp, p, effect (log-OR), se"""
    col = {c.lower(): c for c in df.columns}

    def get(aliases):
        for a in aliases:
            if a in col:
                return col[a]
        return None

    snp_c  = get(["snp","snpid","rsid","id","markername"])
    chr_c  = get(["chr","hg18chr","chrom","chromosome"])
    bp_c   = get(["bp","pos","position"])
    p_c    = get(["p","pval","p.value","p-value","pvalue"])
    or_c   = get(["or","odds_ratio"])
    beta_c = get(["beta","effect","logor","b"])
    se_c   = get(["se","se_beta","stderr"])

    out = pd.DataFrame()
    if snp_c:  out["snp"]  = df[snp_c].astype(str).str.upper().str.strip()
    if chr_c:  out["chr"]  = pd.to_numeric(df[chr_c], errors="coerce")
    if bp_c:   out["bp"]   = pd.to_numeric(df[bp_c],  errors="coerce")
    if p_c:    out["p"]    = pd.to_numeric(df[p_c],   errors="coerce")
    if se_c:   out["se"]   = pd.to_numeric(df[se_c],  errors="coerce")

    if beta_c:
        out["effect"] = pd.to_numeric(df[beta_c], errors="coerce")
    elif or_c:
        raw = pd.to_numeric(df[or_c], errors="coerce")
        out["effect"] = np.log(raw.where((raw > 0) & (raw < 100)))
    else:
        z_c = get(["zscore","z","z_score","stat"])
        if z_c:
            out["effect"] = pd.to_numeric(df[z_c], errors="coerce")

    out = out.dropna(subset=["snp","p"])
    out["condition"] = name
    out["-log10p"] = -np.log10(out["p"].clip(lower=1e-300))
    return out


# ── Load all three ────────────────────────────────────────────────────────────

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

STUDIES = [
    ("adhd",   "OpenMed/pgc-adhd",   "adhd2022",  25),
    ("bipolar","OpenMed/pgc-bipolar", "bip2024",   25),
    ("autism", "OpenMed/pgc-autism",  "asd2019",   25),
]

dfs = {}
for name, repo, sub, n in STUDIES:
    cache = f"data_cache/{name}_{sub}.parquet"
    raw   = fetch_shards(repo, sub, n_shards=n, cache_file=cache)
    if raw.empty:
        print(f"  [{name}] FAILED")
        continue
    norm = normalise(raw, name)
    print(f"  [{name}] {len(norm):,} SNPs  |  "
          f"min p={norm['p'].min():.2e}  |  "
          f"GW hits={(norm['p'] < 5e-8).sum()}")
    dfs[name] = norm

if len(dfs) < 2:
    print("Not enough data loaded. Check network.")
    exit(1)

# ── Merge on rsID ─────────────────────────────────────────────────────────────

print("\nMerging on rsID...")
# Keep only rs-prefixed IDs for clean matching
for name in dfs:
    dfs[name] = dfs[name][dfs[name]["snp"].str.startswith("RS")]
    print(f"  {name}: {len(dfs[name]):,} rs-IDs")

# Triple merge
merged = None
for name, df in dfs.items():
    sub = df[["snp","chr","bp","p","-log10p","effect"]].copy()
    sub.columns = ["snp","chr","bp",
                   f"p_{name}", f"logp_{name}", f"eff_{name}"]
    merged = sub if merged is None else merged.merge(sub, on="snp", how="inner",
                                                     suffixes=("","_dup"))
    # resolve duplicate chr/bp columns
    for col in ["chr","bp"]:
        dups = [c for c in merged.columns if c.startswith(col) and c != col]
        for d in dups:
            merged[col] = merged[col].fillna(merged[d])
            merged.drop(columns=d, inplace=True)

print(f"\nTriple-merged: {len(merged):,} SNPs present in all 3 conditions")


# ── Conjunctive FDR (pleiotropy score) ───────────────────────────────────────
# Fisher: -2 * sum(log(p)) ~ chi2(2k df) where k = number of conditions

print("Computing pleiotropy scores...")
merged["fisher_stat"] = -2 * (
    np.log(merged["p_adhd"].clip(1e-300)) +
    np.log(merged["p_bipolar"].clip(1e-300)) +
    np.log(merged["p_autism"].clip(1e-300))
)
merged["fisher_p"] = 1 - stats.chi2.cdf(merged["fisher_stat"], df=6)
merged["fisher_logp"] = -np.log10(merged["fisher_p"].clip(1e-300))

# Pairwise Fisher for AB, AA, BA
for (a, b) in [("adhd","bipolar"), ("adhd","autism"), ("bipolar","autism")]:
    key = f"fisher_{a[0]}{b[0]}"
    merged[key] = -2 * (np.log(merged[f"p_{a}"].clip(1e-300)) +
                         np.log(merged[f"p_{b}"].clip(1e-300)))
    merged[f"{key}_p"]    = 1 - stats.chi2.cdf(merged[key], df=4)
    merged[f"{key}_logp"] = -np.log10(merged[f"{key}_p"].clip(1e-300))

# GW significance flags
GW = 5e-8
merged["sig_adhd"]    = merged["p_adhd"]    < GW
merged["sig_bipolar"] = merged["p_bipolar"] < GW
merged["sig_autism"]  = merged["p_autism"]  < GW
merged["n_sig"]       = merged[["sig_adhd","sig_bipolar","sig_autism"]].sum(axis=1)
merged["any_sig"]     = merged["n_sig"] > 0
merged["pleiotropic"] = merged["n_sig"] >= 2

print(f"  GW-sig in ADHD:    {merged['sig_adhd'].sum():,}")
print(f"  GW-sig in Bipolar: {merged['sig_bipolar'].sum():,}")
print(f"  GW-sig in Autism:  {merged['sig_autism'].sum():,}")
print(f"  GW-sig in ≥2:      {merged['pleiotropic'].sum():,}")
print(f"  GW-sig in all 3:   {(merged['n_sig']==3).sum():,}")
print(f"  Top Fisher p (triple): {merged['fisher_p'].min():.2e}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 11 — Triple stacked Manhattan
# ══════════════════════════════════════════════════════════════════════════════

print("\nFig 11: Triple Manhattan...")

def compute_offsets(df_merged, chr_col="chr"):
    chrs = sorted(df_merged[chr_col].dropna().astype(int).unique())
    offsets, offset = {}, 0
    for c in chrs:
        offsets[c] = offset
        sub = df_merged[df_merged[chr_col] == c]
        if "bp" in sub.columns:
            offset += int(sub["bp"].max())
    return offsets

offsets = compute_offsets(merged)
merged["bp_cum"] = merged.apply(
    lambda r: int(r["bp"]) + offsets.get(int(r["chr"]) if pd.notna(r["chr"]) else 0, 0)
    if pd.notna(r["bp"]) and pd.notna(r["chr"]) else np.nan, axis=1)

chr_mids = {}
for c in sorted(merged["chr"].dropna().astype(int).unique()):
    sub = merged[merged["chr"] == c]
    chr_mids[c] = sub["bp_cum"].mean()

fig = plt.figure(figsize=(15, 11))
gs  = gridspec.GridSpec(3, 1, hspace=0.08)

conditions_plot = [("adhd","logp_adhd"),("bipolar","logp_bipolar"),("autism","logp_autism")]
chrom_c = ["#4a4e69","#9a8c98"]  # alternating grey tones for non-sig
ymax_global = max(merged[c].max() for _, c in conditions_plot if c in merged.columns) * 1.1

for i, (name, logp_col) in enumerate(conditions_plot):
    ax = fig.add_subplot(gs[i])
    color_main = COLORS[name]

    for chrom in sorted(merged["chr"].dropna().astype(int).unique()):
        sub = merged[merged["chr"] == chrom].copy()
        col = chrom_c[chrom % 2]

        # non-sig
        nsig = sub[sub[logp_col] < 5]
        if len(nsig) > 10_000:
            nsig = nsig.sample(10_000, random_state=42)
        ax.scatter(nsig["bp_cum"], nsig[logp_col],
                   c=col, s=1.5, alpha=0.3, linewidths=0)

        # condition-specific sig (only this condition)
        own_sig = sub[(sub[f"sig_{name}"]) & (~sub["pleiotropic"])]
        ax.scatter(own_sig["bp_cum"], own_sig[logp_col],
                   c=color_main, s=12, zorder=4, linewidths=0,
                   label="Condition-specific" if chrom == 1 else "")

        # pleiotropic hits
        plei = sub[sub["pleiotropic"]]
        ax.scatter(plei["bp_cum"], plei[logp_col],
                   c="#E63946", s=20, zorder=5, marker="D", linewidths=0,
                   label="Pleiotropic (≥2 cond.)" if chrom == 1 else "")

    ax.axhline(-np.log10(GW), color="#E63946", lw=0.9, ls="--", alpha=0.7)
    ax.set_ylim(0, ymax_global)
    ax.set_ylabel(f"–log₁₀(P)\n{NAMES[name]}", fontsize=9)
    ax.set_yticks([0, 5, 10, 15])

    if i < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xticks(list(chr_mids.values()))
        ax.set_xticklabels([str(int(c)) for c in chr_mids], fontsize=7)
        ax.set_xlabel("Chromosome", fontsize=10)

    if i == 0:
        legend_els = [
            Line2D([0],[0], marker="o", color="w", markerfacecolor=COLORS["adhd"],  markersize=7, label="ADHD-specific"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor=COLORS["bipolar"],markersize=7, label="Bipolar-specific"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor=COLORS["autism"], markersize=7, label="Autism-specific"),
            Line2D([0],[0], marker="D", color="w", markerfacecolor="#E63946",        markersize=8, label="Pleiotropic (≥2)"),
        ]
        ax.legend(handles=legend_els, loc="upper right", fontsize=8.5,
                  frameon=True, framealpha=0.9)

fig.suptitle("ADHD × Bipolar × Autism — Stacked Manhattan Plots\n"
             "Red diamonds = loci significant in ≥2 conditions (pleiotropic hits)",
             fontsize=13, fontweight="bold", y=1.01)
plt.savefig("figures/fig11_triple_manhattan.png")
plt.close()
print("  saved fig11")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 12 — Pairwise effect-size scatter plots (3 panels)
# ══════════════════════════════════════════════════════════════════════════════

print("Fig 12: Pairwise scatter plots...")

pairs = [
    ("adhd","bipolar","autism", "ADHD effect (log-OR)", "Bipolar effect (log-OR)"),
    ("adhd","autism","bipolar", "ADHD effect (log-OR)", "Autism effect (log-OR)"),
    ("bipolar","autism","adhd", "Bipolar effect (log-OR)", "Autism effect (log-OR)"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (xa, ya, color_by, xlabel, ylabel) in zip(axes, pairs):
    sub = merged.dropna(subset=[f"eff_{xa}", f"eff_{ya}", f"logp_{color_by}"])
    # sample for speed
    if len(sub) > 30_000:
        sub = sub.sample(30_000, random_state=42)

    scatter = ax.scatter(
        sub[f"eff_{xa}"], sub[f"eff_{ya}"],
        c=sub[f"logp_{color_by}"], cmap="viridis",
        vmin=0, vmax=10, s=3, alpha=0.4, linewidths=0,
    )

    # Highlight pleiotropic hits
    plei = merged[merged["pleiotropic"]].dropna(subset=[f"eff_{xa}", f"eff_{ya}"])
    ax.scatter(plei[f"eff_{xa}"], plei[f"eff_{ya}"],
               c="#E63946", s=40, zorder=5, marker="D",
               linewidths=0.5, edgecolors="white", label="Pleiotropic")

    # Correlation line
    valid = sub[[f"eff_{xa}", f"eff_{ya}"]].dropna()
    if len(valid) > 10:
        m, b, r, p_corr, _ = stats.linregress(valid.iloc[:,0], valid.iloc[:,1])
        x_line = np.array([valid.iloc[:,0].quantile(0.01), valid.iloc[:,0].quantile(0.99)])
        ax.plot(x_line, m*x_line+b, color="#333", lw=1.2, ls="--",
                label=f"r={r:.3f}, p={p_corr:.2e}")

    ax.axhline(0, color="#ccc", lw=0.7)
    ax.axvline(0, color="#ccc", lw=0.7)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{NAMES[xa].split()[0]} vs {NAMES[ya].split()[0]}\n"
                 f"(coloured by {color_by.title()} –log₁₀P)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    cb = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cb.set_label(f"–log₁₀P ({color_by.title()})", fontsize=8)

fig.suptitle("Pairwise Effect-Size Concordance — ADHD, Bipolar, Autism\n"
             "Each point = one SNP present in all 3 studies; colour = –log₁₀P in third condition",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/fig12_pairwise_scatter.png")
plt.close()
print("  saved fig12")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 13 — Pleiotropy score (Fisher) Manhattan
# ══════════════════════════════════════════════════════════════════════════════

print("Fig 13: Pleiotropy Manhattan...")

fig, axes = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={"hspace": 0.08})
pair_info = [
    ("fisher_ab_logp", "ADHD × Bipolar",  "#7B4F9E"),
    ("fisher_aa_logp", "ADHD × Autism",   "#2A9D8F"),
    ("fisher_ba_logp", "Bipolar × Autism","#E76F51"),
]
# remap column names
col_remap = {
    "fisher_ab_logp": "fisher_ab_logp",
    "fisher_aa_logp": "fisher_aa_logp",
    "fisher_ba_logp": "fisher_ba_logp",
}
# ensure columns exist
for k in ["fisher_ab_logp","fisher_aa_logp","fisher_ba_logp"]:
    if k not in merged.columns:
        # try to reconstruct
        mapping = {"ab":("adhd","bipolar"),"aa":("adhd","autism"),"ba":("bipolar","autism")}
        key2 = k.replace("_logp","")
        if key2 in mapping:
            a,b = mapping[key2]
            merged[k] = -np.log10((1 - stats.chi2.cdf(
                -2*(np.log(merged[f"p_{a}"].clip(1e-300)) +
                    np.log(merged[f"p_{b}"].clip(1e-300))), df=4
            )).clip(1e-300))

GW_fisher_2 = -np.log10(stats.chi2.sf(stats.chi2.ppf(1-GW, 1)*2, df=4))

for ax, (col, title, color) in zip(axes, pair_info):
    if col not in merged.columns:
        continue
    for chrom in sorted(merged["chr"].dropna().astype(int).unique()):
        sub = merged[merged["chr"]==chrom].copy()
        c   = chrom_c[chrom % 2]
        nsig = sub[sub[col] < GW_fisher_2]
        sig  = sub[sub[col] >= GW_fisher_2]
        if len(nsig) > 8000:
            nsig = nsig.sample(8000, random_state=42)
        ax.scatter(nsig["bp_cum"], nsig[col], c=c, s=2, alpha=0.3, linewidths=0)
        if not sig.empty:
            ax.scatter(sig["bp_cum"], sig[col], c=color, s=15, zorder=5,
                       linewidths=0, marker="D")

    ax.axhline(GW_fisher_2, color=color, lw=0.9, ls="--", alpha=0.8,
               label=f"Conjunctive GW threshold")
    ax.set_ylabel(f"–log₁₀(Fisher P)\n{title}", fontsize=9)
    ax.set_ylim(0)

axes[-1].set_xticks(list(chr_mids.values()))
axes[-1].set_xticklabels([str(int(c)) for c in chr_mids], fontsize=7)
axes[-1].set_xlabel("Chromosome", fontsize=10)

fig.suptitle("Conjunctive Fisher Pleiotropy Score — Pairwise\n"
             "Diamonds: genome-wide conjunctively significant loci",
             fontsize=12, fontweight="bold", y=1.01)
plt.savefig("figures/fig13_pleiotropy_manhattan.png")
plt.close()
print("  saved fig13")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 14 — Known shared loci + effect concordance heatmap
# ══════════════════════════════════════════════════════════════════════════════

print("Fig 14: Known loci heatmap...")

# Build effect / p-value table for known loci
# Look up nearest SNP in our merged data for each known locus
loci_data = []
for gene, chrom, pos, conds, note in KNOWN_LOCI:
    window = 500_000
    nearby = merged[
        (merged["chr"] == chrom) &
        (merged["bp"] >= pos - window) &
        (merged["bp"] <= pos + window)
    ]
    if nearby.empty:
        row = {"gene": gene, "chr": chrom, "conditions": ", ".join(conds), "note": note,
               "min_p_adhd": np.nan, "min_p_bipolar": np.nan, "min_p_autism": np.nan,
               "eff_adhd": np.nan, "eff_bipolar": np.nan, "eff_autism": np.nan}
    else:
        best = nearby.loc[nearby["fisher_stat"].idxmax()]
        row = {"gene": gene, "chr": chrom, "conditions": ", ".join(conds), "note": note,
               "min_p_adhd":    nearby["p_adhd"].min(),
               "min_p_bipolar": nearby["p_bipolar"].min(),
               "min_p_autism":  nearby["p_autism"].min(),
               "eff_adhd":    best.get("eff_adhd", np.nan),
               "eff_bipolar": best.get("eff_bipolar", np.nan),
               "eff_autism":  best.get("eff_autism", np.nan)}
    loci_data.append(row)

loci_df = pd.DataFrame(loci_data)

# Two-panel figure: effect size heatmap + -log10p heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"width_ratios": [1, 1]})

# Panel A: –log10(p) at known loci
logp_mat = pd.DataFrame({
    "ADHD":    -np.log10(loci_df["min_p_adhd"].clip(1e-300)),
    "Bipolar": -np.log10(loci_df["min_p_bipolar"].clip(1e-300)),
    "Autism":  -np.log10(loci_df["min_p_autism"].clip(1e-300)),
}, index=loci_df["gene"])

sns.heatmap(logp_mat, cmap="YlOrRd", vmin=0, vmax=15,
            annot=True, fmt=".1f", annot_kws={"size": 8.5},
            linewidths=0.5, linecolor="#eee", ax=ax1,
            cbar_kws={"label": "–log₁₀(P)", "shrink": 0.7})
ax1.set_title("–log₁₀(P) at Known Shared Loci\n(window ±500kb)", fontsize=11, fontweight="bold")
ax1.set_xticklabels(["ADHD","Bipolar","Autism"], fontsize=10, rotation=0)
ax1.set_yticklabels(loci_df["gene"], rotation=0, fontsize=9)

# Panel B: effect concordance (direction)
eff_mat = pd.DataFrame({
    "ADHD":    loci_df["eff_adhd"],
    "Bipolar": loci_df["eff_bipolar"],
    "Autism":  loci_df["eff_autism"],
}, index=loci_df["gene"])

sns.heatmap(eff_mat, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
            annot=True, fmt=".3f", annot_kws={"size": 8.5},
            linewidths=0.5, linecolor="#eee", ax=ax2,
            cbar_kws={"label": "Effect size (log-OR)", "shrink": 0.7})
ax2.set_title("Effect Size Direction at Known Shared Loci\n(log-OR; same sign = concordant)", fontsize=11, fontweight="bold")
ax2.set_xticklabels(["ADHD","Bipolar","Autism"], fontsize=10, rotation=0)
ax2.set_yticklabels(loci_df["gene"], rotation=0, fontsize=9)

# Add gene function annotations on right
for i, (_, row) in enumerate(loci_df.iterrows()):
    ax2.text(3.15, i + 0.5, row["note"][:45] + ("…" if len(row["note"]) > 45 else ""),
             va="center", fontsize=6.5, color="#555")

plt.suptitle("Known Pleiotropic Loci — ADHD, Bipolar, Autism",
             fontsize=13, fontweight="bold", x=0.45, y=1.02)
plt.tight_layout()
plt.savefig("figures/fig14_known_loci_heatmap.png", bbox_inches="tight")
plt.close()
print("  saved fig14")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 15 — PRS cross-prediction simulation
# ══════════════════════════════════════════════════════════════════════════════

print("Fig 15: PRS cross-prediction...")

np.random.seed(42)
N = 200_000

# Genetic correlations from literature (LDSC estimates)
# rg(ADHD, Bipolar) ~ 0.19   (Anttila et al. 2018)
# rg(ADHD, Autism)  ~ 0.36   (Anttila et al. 2018)
# rg(Bipolar, Autism) ~ 0.09  (Anttila et al. 2018)
# h2_SNP: ADHD~0.22, Bipolar~0.23, Autism~0.18
rg_AB = 0.19   # ADHD-Bipolar
rg_AA = 0.36   # ADHD-Autism
rg_BA = 0.09   # Bipolar-Autism

h2 = {"adhd": 0.22, "bipolar": 0.23, "autism": 0.18}
prev = {"adhd": 0.05, "bipolar": 0.02, "autism": 0.015}

# Correlated PRS via Cholesky decomposition
Sigma = np.array([
    [h2["adhd"],              rg_AB * np.sqrt(h2["adhd"]*h2["bipolar"]), rg_AA * np.sqrt(h2["adhd"]*h2["autism"])],
    [rg_AB * np.sqrt(h2["adhd"]*h2["bipolar"]), h2["bipolar"],           rg_BA * np.sqrt(h2["bipolar"]*h2["autism"])],
    [rg_AA * np.sqrt(h2["adhd"]*h2["autism"]),  rg_BA * np.sqrt(h2["bipolar"]*h2["autism"]), h2["autism"]],
])
L = np.linalg.cholesky(Sigma)
Z = np.random.normal(0, 1, (N, 3))
PRS = Z @ L.T  # N x 3: columns = ADHD, Bipolar, Autism PRS

# Thresholds
thresh = {c: np.percentile(PRS[:, i], (1 - prev[c]) * 100)
          for i, c in enumerate(["adhd","bipolar","autism"])}

# Case/control status
case = {c: (PRS[:, i] >= thresh[c]) for i, c in enumerate(["adhd","bipolar","autism"])}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("PRS Cross-Prediction Simulation — ADHD × Bipolar × Autism\n"
             "Based on published genetic correlations (Anttila et al. 2018, LDSC)",
             fontsize=13, fontweight="bold")

cond_names = ["adhd","bipolar","autism"]
cond_labels = ["ADHD","Bipolar","Autism"]
cond_cols   = [COLORS[c] for c in cond_names]

# Top row: PRS distributions, coloured by case status in other conditions
for col_i, (target, t_lbl, t_col) in enumerate(zip(cond_names, cond_labels, cond_cols)):
    ax  = axes[0][col_i]
    idx = cond_names.index(target)

    prs_control = PRS[:, idx][~case[target]]
    prs_case    = PRS[:, idx][ case[target]]

    bins = np.linspace(PRS[:, idx].min(), PRS[:, idx].max(), 70)
    ax.hist(prs_control, bins=bins, density=True, alpha=0.5, color="#aaa", label="Controls")
    ax.hist(prs_case,    bins=bins, density=True, alpha=0.7, color=t_col, label=f"{t_lbl} cases")
    ax.axvline(thresh[target], color="#333", lw=1, ls="--",
               label=f"Threshold ({prev[target]*100:.1f}%)")
    ax.set_xlabel(f"{t_lbl} PRS", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{t_lbl} PRS distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.set_axisbelow(True); ax.yaxis.grid(True, alpha=0.3)

# Bottom row: risk by decile (cross-prediction)
pairs_plot = [
    ("adhd", "bipolar", rg_AB),
    ("adhd", "autism",  rg_AA),
    ("bipolar", "autism", rg_BA),
]
for col_i, (predictor, target, rg) in enumerate(pairs_plot):
    ax  = axes[1][col_i]
    p_i = cond_names.index(predictor)

    decile_edges = np.percentile(PRS[:, p_i], np.arange(0, 101, 10))
    decile_risk  = []
    for lo, hi in zip(decile_edges[:-1], decile_edges[1:]):
        mask = (PRS[:, p_i] >= lo) & (PRS[:, p_i] < hi)
        decile_risk.append(case[target][mask].mean() * 100)

    x = np.arange(1, 11)
    bar_cols = [plt.cm.RdBu_r(v / (prev[target]*100*5)) for v in decile_risk]
    bars = ax.bar(x, decile_risk, color=bar_cols, edgecolor="white", lw=0.7)
    ax.axhline(prev[target]*100, color="#555", lw=1, ls="--",
               label=f"Population avg ({prev[target]*100:.1f}%)")
    for bar, v in zip(bars, decile_risk):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.02,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xlabel(f"{cond_labels[cond_names.index(predictor)]} PRS decile", fontsize=10)
    ax.set_ylabel(f"{cond_labels[cond_names.index(target)]} risk (%)", fontsize=10)
    ax.set_title(f"{cond_labels[cond_names.index(predictor)]} PRS → {cond_labels[cond_names.index(target)]} risk\n"
                 f"(r_g = {rg:.2f})", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"D{i}" for i in x], fontsize=8.5)
    ax.set_ylim(0, max(decile_risk)*1.2)
    ax.legend(fontsize=8, frameon=False)
    ax.set_axisbelow(True); ax.yaxis.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/fig15_prs_cross_prediction.png")
plt.close()
print("  saved fig15")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Triple-merged SNPs:  {len(merged):,}")
print(f"Pleiotropic (≥2):    {merged['pleiotropic'].sum():,}")
print(f"All 3 conditions:    {(merged['n_sig']==3).sum():,}")
print(f"Top Fisher (all 3):  {merged['fisher_p'].min():.3e}")
print(f"\nTop 10 pleiotropic SNPs:")
top_plei = merged[merged["pleiotropic"]].nsmallest(10, "fisher_p")[
    ["snp","chr","bp","p_adhd","p_bipolar","p_autism","fisher_p"]
]
print(top_plei.to_string(index=False))
print("\nAll figures saved to ./figures/")
top_plei.to_csv("data_cache/top_pleiotropic_snps.csv", index=False)
merged.to_parquet("data_cache/merged_triad.parquet", index=False)
print("Merged data saved to data_cache/merged_triad.parquet")
