"""
Generate all figures for the PGC Psychiatric GWAS report.
Outputs PNG files to ./figures/

Figures:
  1. Cross-disorder genetic effect correlation heatmap
  2. MAF distribution by condition
  3. Effect size distributions (violin)
  4. Manhattan plot — Schizophrenia (scz2022)
  5. QQ plot — Schizophrenia (scz2022)
  6. Sample size comparison
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from scipy import stats
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_tree
warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)

# ── Styling ────────────────────────────────────────────────────────────────────

PALETTE = {
    "adhd":           "#4C9BE8",
    "anxiety":        "#F4A261",
    "autism":         "#2A9D8F",
    "bipolar":        "#E76F51",
    "cross_disorder": "#8B5CF6",
    "eating":         "#F72585",
    "mdd":            "#457B9D",
    "ptsd":           "#E9C46A",
    "schizophrenia":  "#264653",
    "substance_use":  "#BC6C25",
}

LABELS = {
    "adhd":           "ADHD",
    "anxiety":        "Anxiety",
    "autism":         "Autism",
    "bipolar":        "Bipolar",
    "cross_disorder": "Cross-\nDisorder",
    "eating":         "Eating\nDisorders",
    "mdd":            "MDD",
    "ptsd":           "PTSD",
    "schizophrenia":  "Schizophrenia",
    "substance_use":  "Substance\nUse",
}

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Palatino", "Georgia", "Times New Roman"],
    "font.size":        11,
    "axes.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
})


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Cross-disorder correlation heatmap
# ══════════════════════════════════════════════════════════════════════════════

print("Fig 1: Correlation heatmap...")

# Results from v3 run (489,904 shared SNPs, log-OR scale)
conditions = ["adhd","anxiety","autism","bipolar","cross_disorder",
              "eating","mdd","ptsd","schizophrenia","substance_use"]

corr_vals = np.array([
    [ 1.000, -0.008,  0.038,  0.001,  0.216,  0.009, -0.007,  0.008, -0.002,  0.012],
    [-0.008,  1.000,  0.017,  0.012,  0.090,  0.016, -0.000, -0.007, -0.020,  0.041],
    [ 0.038,  0.017,  1.000, -0.004,  0.386,  0.088,  0.019, -0.075, -0.009,  0.040],
    [ 0.001,  0.012, -0.004,  1.000,  0.034,  0.014, -0.004, -0.012,  0.011,  0.014],
    [ 0.216,  0.090,  0.386,  0.034,  1.000,  0.099, -0.004,  0.014, -0.012,  0.023],
    [ 0.009,  0.016,  0.088,  0.014,  0.099,  1.000,  0.018,  0.051, -0.028,  0.046],
    [-0.007, -0.000,  0.019, -0.004, -0.004,  0.018,  1.000, -0.004,  0.006, -0.043],
    [ 0.008, -0.007, -0.075, -0.012,  0.014,  0.051, -0.004,  1.000, -0.030,  np.nan],
    [-0.002, -0.020, -0.009,  0.011, -0.012, -0.028,  0.006, -0.030,  1.000,  0.190],
    [ 0.012,  0.041,  0.040,  0.014,  0.023,  0.046, -0.043,  np.nan,  0.190,  1.000],
])

corr_df = pd.DataFrame(corr_vals, index=conditions, columns=conditions)
tick_labels = [LABELS[c] for c in conditions]

fig, ax = plt.subplots(figsize=(10, 8.5))

mask = np.zeros_like(corr_vals, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True          # upper triangle masked

cmap = sns.diverging_palette(250, 15, s=75, l=45, n=256, as_cmap=True)
sns.heatmap(
    corr_df, mask=mask, cmap=cmap,
    vmin=-0.45, vmax=0.45, center=0,
    annot=True, fmt=".2f", annot_kws={"size": 8},
    linewidths=0.4, linecolor="#e0e0e0",
    square=True, ax=ax,
    xticklabels=tick_labels, yticklabels=tick_labels,
    cbar_kws={"shrink": 0.7, "label": "Effect-size correlation (log-OR scale)"},
)
ax.set_title("Cross-Disorder Genetic Effect Correlation\nPGC GWAS Meta-Analyses — 489,904 Shared SNPs",
             fontsize=13, fontweight="bold", pad=14)
ax.tick_params(axis="x", rotation=0, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=9)

# Annotate strongest pairs
notable = [(2, 4, "r=0.39"), (0, 4, "r=0.22"), (8, 9, "r=0.19"), (2, 5, "r=0.09")]
for (r, c, txt) in notable:
    if r > c:
        ax.text(c + 0.5, r + 0.65, "▲", fontsize=7, ha="center", color="white", alpha=0.7)

plt.tight_layout()
plt.savefig("figures/fig1_correlation_heatmap.png")
plt.close()
print("  saved figures/fig1_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — MAF distribution stacked bar
# ══════════════════════════════════════════════════════════════════════════════

print("Fig 2: MAF distribution...")

# From v2 run
maf_data = {
    "adhd":           [0.0,  7.8, 13.8, 22.5, 55.9],
    "anxiety":        [0.0,  0.0, 23.1, 24.2, 52.7],
    "autism":         [3.1, 28.4, 14.3, 18.2, 35.9],
    "bipolar":        [0.0, 11.1, 14.5, 22.0, 52.4],
    "cross_disorder": [3.9,  8.2, 12.0, 21.7, 54.2],
    "ptsd":           [0.0,  0.1, 15.2, 16.4, 68.4],
}
maf_labels = ["<1%", "1–5%", "5–10%", "10–20%", "20–50%"]
maf_colors = ["#d62728", "#ff7f0e", "#bcbd22", "#17becf", "#1f77b4"]

conds = list(maf_data.keys())
x = np.arange(len(conds))
bottoms = np.zeros(len(conds))

fig, ax = plt.subplots(figsize=(9, 5))
for i, (lbl, col) in enumerate(zip(maf_labels, maf_colors)):
    vals = [maf_data[c][i] for c in conds]
    bars = ax.bar(x, vals, bottom=bottoms, label=lbl, color=col, width=0.6, edgecolor="white", lw=0.5)
    bottoms += np.array(vals)

ax.set_xticks(x)
ax.set_xticklabels([LABELS[c] for c in conds], fontsize=10)
ax.set_ylabel("Percentage of variants (%)", fontsize=11)
ax.set_title("Minor Allele Frequency Distribution by Psychiatric Condition",
             fontsize=13, fontweight="bold", pad=10)
ax.set_ylim(0, 108)
ax.legend(title="MAF bin", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
ax.set_axisbelow(True)
ax.yaxis.grid(True, alpha=0.35, lw=0.6)

# Note autism rare variant enrichment
ax.annotate("Autism: enriched\nfor rare variants\n(MAF 1–5%: 28%)",
            xy=(2, 55), xytext=(3.5, 85),
            arrowprops=dict(arrowstyle="->", color="#333", lw=0.8),
            fontsize=8.5, color="#333")

plt.tight_layout()
plt.savefig("figures/fig2_maf_distribution.png")
plt.close()
print("  saved figures/fig2_maf_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Effect size violin plots
# (stream ADHD, bipolar, autism, schizophrenia for realistic distributions)
# ══════════════════════════════════════════════════════════════════════════════

print("Fig 3: Effect size violins (streaming data)...")

VIOLIN_DATASETS = {
    "anxiety":      ("OpenMed/pgc-anxiety",      None,       "BETA", "Effect"),
    "autism":       ("OpenMed/pgc-autism",        None,       "OR",   "OR"),
    "bipolar":      ("OpenMed/pgc-bipolar",       None,       "OR",   "or"),
    "schizophrenia":("OpenMed/pgc-schizophrenia", "scz2022",  "OR",   "OR"),
    "cross_disorder":("OpenMed/pgc-cross-disorder",None,      "OR",   "or"),
}

def stream_effect(repo, sub, etype, col, n=80_000):
    try:
        if sub:
            ds = load_dataset(repo, data_dir=f"data/{sub}", split="train", streaming=True)
        else:
            ds = load_dataset(repo, split="train", streaming=True)
        rows = list(ds.take(n))
        df = pd.DataFrame(rows)
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if etype == "OR":
            vals = vals.where((vals > 0.01) & (vals < 20))
            vals = np.log(vals.dropna())
        return vals.values
    except Exception as e:
        print(f"    warn: {e}")
        return np.array([])

violin_data = {}
for name, (repo, sub, etype, col) in VIOLIN_DATASETS.items():
    print(f"  streaming {name}...")
    vals = stream_effect(repo, sub, etype, col)
    if len(vals) > 100:
        violin_data[name] = vals

if violin_data:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names_v = list(violin_data.keys())
    data_v  = [violin_data[n] for n in names_v]
    colors_v = [PALETTE[n] for n in names_v]

    parts = ax.violinplot(data_v, positions=range(len(names_v)),
                          widths=0.7, showmedians=True, showextrema=False)
    for i, (body, col) in enumerate(zip(parts["bodies"], colors_v)):
        body.set_facecolor(col)
        body.set_alpha(0.75)
        body.set_edgecolor("#333")
        body.set_linewidth(0.7)
    parts["cmedians"].set_color("#222")
    parts["cmedians"].set_linewidth(1.5)

    # Overlay IQR box
    for i, d in enumerate(data_v):
        q1, q3 = np.percentile(d, [25, 75])
        ax.plot([i, i], [q1, q3], color="#222", lw=4, solid_capstyle="round", alpha=0.5)

    ax.set_xticks(range(len(names_v)))
    ax.set_xticklabels([LABELS[n] for n in names_v], fontsize=10)
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_ylabel("Effect size (log-OR / BETA)", fontsize=11)
    ax.set_title("Effect Size Distributions Across Psychiatric Conditions\n(Sample of 80,000 SNPs per condition)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3, lw=0.6)

    plt.tight_layout()
    plt.savefig("figures/fig3_effect_distributions.png")
    plt.close()
    print("  saved figures/fig3_effect_distributions.png")
else:
    print("  skipped (no data)")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Manhattan plot (Schizophrenia scz2022, multi-shard)
# ══════════════════════════════════════════════════════════════════════════════

print("Fig 4: Manhattan plot (schizophrenia scz2022)...")

def load_scz_manhattan(n_shards=12):
    repo  = "OpenMed/pgc-schizophrenia"
    items = list(list_repo_tree(repo, repo_type="dataset", path_in_repo="data/scz2022"))
    files = sorted([x.path for x in items if getattr(x,"path","").endswith(".parquet")])[:n_shards]
    frames = []
    for fpath in files:
        try:
            local = hf_hub_download(repo_id=repo, filename=fpath, repo_type="dataset")
            df    = pd.read_parquet(local, columns=["CHR","BP","SNP","P"])
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["CHR"] = pd.to_numeric(df["CHR"], errors="coerce")
    df["BP"]  = pd.to_numeric(df["BP"],  errors="coerce")
    df["P"]   = pd.to_numeric(df["P"],   errors="coerce")
    return df.dropna(subset=["CHR","BP","P"])

scz_df = load_scz_manhattan(n_shards=24)   # all shards
print(f"  loaded {len(scz_df):,} SCZ variants")

if not scz_df.empty:
    # Compute cumulative BP positions
    scz_df["CHR"] = scz_df["CHR"].astype(int)
    scz_df = scz_df.sort_values(["CHR","BP"])
    scz_df["-log10p"] = -np.log10(scz_df["P"].clip(lower=1e-300))

    chr_offsets = {}
    offset = 0
    for chrom in sorted(scz_df["CHR"].unique()):
        chr_offsets[chrom] = offset
        offset += scz_df[scz_df["CHR"] == chrom]["BP"].max()

    scz_df["BP_cum"] = scz_df.apply(lambda r: r["BP"] + chr_offsets.get(int(r["CHR"]), 0), axis=1)

    # Chromosome centres for x-tick labels
    chr_mids = {
        chrom: scz_df[scz_df["CHR"] == chrom]["BP_cum"].mean()
        for chrom in sorted(scz_df["CHR"].unique())
    }

    fig, ax = plt.subplots(figsize=(14, 5))
    chrom_colors = ["#264653", "#a8dadc"]

    for chrom in sorted(scz_df["CHR"].unique()):
        sub = scz_df[scz_df["CHR"] == chrom]
        col = chrom_colors[int(chrom) % 2]
        # thin the non-significant points for speed
        nsig = sub[sub["-log10p"] < 5]
        sig  = sub[sub["-log10p"] >= 5]
        if len(nsig) > 20_000:
            nsig = nsig.sample(20_000, random_state=42)
        ax.scatter(nsig["BP_cum"], nsig["-log10p"], c=col, s=1.5, alpha=0.35, linewidths=0)
        if not sig.empty:
            ax.scatter(sig["BP_cum"], sig["-log10p"], c="#E63946", s=10, zorder=5, linewidths=0)

    # Threshold lines
    ax.axhline(-np.log10(5e-8), color="#E63946", lw=0.9, ls="--", label="GW significance (p=5×10⁻⁸)")
    ax.axhline(-np.log10(1e-5), color="#F4A261", lw=0.7, ls=":",  label="Suggestive (p=1×10⁻⁵)")

    ax.set_xticks(list(chr_mids.values()))
    ax.set_xticklabels([str(int(c)) for c in chr_mids.keys()], fontsize=7.5)
    ax.set_xlabel("Chromosome", fontsize=11)
    ax.set_ylabel("–log₁₀(P)", fontsize=11)
    ax.set_title("Manhattan Plot — Schizophrenia GWAS (scz2022, N≈97,000)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.set_ylim(0, max(scz_df["-log10p"].max() * 1.1, 10))
    ax.set_axisbelow(False)

    plt.tight_layout()
    plt.savefig("figures/fig4_manhattan_schizophrenia.png")
    plt.close()
    print("  saved figures/fig4_manhattan_schizophrenia.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — QQ plot (Schizophrenia)
# ══════════════════════════════════════════════════════════════════════════════

print("Fig 5: QQ plot...")

if not scz_df.empty:
    p_vals = scz_df["P"].dropna().values
    p_vals = p_vals[(p_vals > 0) & (p_vals <= 1)]

    n_pts = len(p_vals)
    observed  = np.sort(-np.log10(p_vals))[::-1]
    expected  = -np.log10(np.arange(1, n_pts + 1) / (n_pts + 1))
    lambda_gc = np.median(stats.chi2.ppf(1 - p_vals, df=1)) / stats.chi2.ppf(0.5, df=1)

    # Thin for plot speed
    idx = np.unique(np.round(np.linspace(0, len(observed) - 1, 5000)).astype(int))
    obs_thin = observed[idx]
    exp_thin = expected[idx]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(exp_thin, obs_thin, s=4, color="#264653", alpha=0.5, linewidths=0)
    max_val = max(obs_thin.max(), exp_thin.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "r--", lw=1, label="Expected under H₀")

    ax.set_xlabel("Expected –log₁₀(P)", fontsize=11)
    ax.set_ylabel("Observed –log₁₀(P)", fontsize=11)
    ax.set_title(f"QQ Plot — Schizophrenia (scz2022)\nλ_GC = {lambda_gc:.3f}",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9, frameon=False)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    plt.tight_layout()
    plt.savefig("figures/fig5_qq_schizophrenia.png")
    plt.close()
    print("  saved figures/fig5_qq_schizophrenia.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Dataset overview: rows + sample sizes
# ══════════════════════════════════════════════════════════════════════════════

print("Fig 6: Dataset overview...")

# From the HF collection page
dataset_sizes_M = {
    "adhd":           31.2,
    "anxiety":        27.5,
    "autism":         18.6,
    "bipolar":        74.4,
    "cross_disorder": 63.3,
    "eating":         10.6,
    "mdd":           215.0,
    "ocd_tourette":   36.5,
    "ptsd":          128.0,
    "schizophrenia":  91.4,
    "substance_use": 214.0,
}

conds_all = list(dataset_sizes_M.keys())
sizes     = [dataset_sizes_M[c] for c in conds_all]
cols_all  = [PALETTE.get(c, "#aaa") for c in conds_all]
sorted_idx = np.argsort(sizes)[::-1]

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(conds_all))
bars = ax.bar(x, [sizes[i] for i in sorted_idx],
              color=[cols_all[i] for i in sorted_idx],
              width=0.65, edgecolor="white", lw=0.7)

for bar, val in zip(bars, [sizes[i] for i in sorted_idx]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f"{val:.0f}M", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_xticks(x)
sorted_conds = [conds_all[i] for i in sorted_idx]
labels_all = {**LABELS, "ocd_tourette": "OCD &\nTourette"}
ax.set_xticklabels([labels_all.get(c, c) for c in sorted_conds], fontsize=9)
ax.set_ylabel("Number of rows (millions)", fontsize=11)
ax.set_title("OpenMed PGC GWAS Dataset Sizes\n1.01 Billion Total Rows Across 11 Conditions",
             fontsize=13, fontweight="bold", pad=10)
ax.set_ylim(0, 260)
ax.set_axisbelow(True)
ax.yaxis.grid(True, alpha=0.35, lw=0.6)
ax.spines["left"].set_visible(True)

total = sum(sizes)
ax.text(0.98, 0.97, f"Total: {total/1000:.2f}B rows\n52 landmark studies\n12 conditions",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9))

plt.tight_layout()
plt.savefig("figures/fig6_dataset_overview.png")
plt.close()
print("  saved figures/fig6_dataset_overview.png")

print("\nAll figures generated in ./figures/")
