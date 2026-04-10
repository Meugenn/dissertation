"""
Data-Driven Genetic Taxonomy of Neurodivergence
================================================
Builds a Condition × SNP effect-size matrix from 5 cached GWAS datasets,
then uses PCA + hierarchical clustering + k-means to find whether the
genome suggests different groupings than the DSM-5.

Figures:
  fig20_pca_conditions.png  — conditions in SNP-effect space (PCA biplot)
  fig21_dendrogram.png      — hierarchical clustering dendrogram on conditions
  fig22_snp_pca.png         — SNPs in condition-effect space, colored by cluster
  fig23_taxonomy_compare.png — data-driven vs DSM-5 groupings side-by-side
  fig24_kmeans_sweep.png    — silhouette/inertia for k=2..8

Runs in ~2-3 min on 25-shard parquets.
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Palatino", "Georgia", "Times New Roman"],
    "font.size":   11,
    "axes.linewidth": 0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":   150,
    "savefig.dpi":  200,
    "savefig.bbox": "tight",
})

# ── DSM-5 taxonomy for comparison ─────────────────────────────────────────────
# fmt: off
DSM_GROUPS = {
    "Neurodevelopmental":       ["adhd", "autism"],
    "Schizophrenia Spectrum":   ["schizophrenia"],
    "Bipolar & Related":        ["bipolar"],
    "Depressive":               ["mdd"],
    "Anxiety":                  ["anxiety"],
    "Trauma/Stressor-Related":  ["ptsd"],
    "Feeding/Eating":           ["eating"],
    "Substance-Related":        ["substance_use"],
}
DSM_COLORS = {
    "adhd":           "#4C9BE8",
    "autism":         "#2A9D8F",
    "schizophrenia":  "#E63946",
    "bipolar":        "#8B5CF6",
    "mdd":            "#457B9D",
    "anxiety":        "#F4A261",
    "ptsd":           "#E9C46A",
    "eating":         "#F72585",
    "substance_use":  "#264653",
    "cross_disorder": "#BC6C25",
}
CONDITION_LABELS = {
    "adhd": "ADHD", "autism": "Autism", "schizophrenia": "SCZ",
    "bipolar": "Bipolar", "substance_use": "SUD",
    "mdd": "MDD", "anxiety": "Anxiety", "ptsd": "PTSD",
    "eating": "Eating", "cross_disorder": "Cross-Dis.",
}
# fmt: on


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD CONDITION PARQUETS → NORMALISED (chr, bp_bin, effect) SERIES
# ══════════════════════════════════════════════════════════════════════════════

CACHE = os.path.expanduser("~/.cache/huggingface/hub")

SHARD_PATTERNS = {
    "adhd":          f"{CACHE}/datasets--OpenMed--pgc-adhd/snapshots/*/data/adhd2022/*.parquet",
    "autism":        f"{CACHE}/datasets--OpenMed--pgc-autism/snapshots/*/data/asd2019/*.parquet",
    "bipolar":       f"{CACHE}/datasets--OpenMed--pgc-bipolar/snapshots/*/data/bip2024/*.parquet",
    "schizophrenia": f"{CACHE}/datasets--OpenMed--pgc-schizophrenia/snapshots/*/data/scz2022/*.parquet",
    "substance_use": f"{CACHE}/datasets--OpenMed--pgc-substance-use/snapshots/*/data/SUD2023/*.parquet",
}

CHR_COLS  = {"chr","chrom","chromosome","hg18chr"}
BP_COLS   = {"bp","pos","position","basepair"}
BETA_COLS = {"beta","effect","b","effect_size","logor"}
OR_COLS   = {"or","odds_ratio"}
SE_COLS   = {"se","se_beta","stderr","std_err","stderror"}
P_COLS    = {"p","pval","p.value","p-value","pvalue","p_value"}

def find_col(df, aliases):
    low = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in low: return low[a]
    return None

def load_condition(name, pattern, max_shards=25):
    files = sorted(glob.glob(pattern))[:max_shards]
    if not files:
        print(f"  [{name}] no files found — skip")
        return None
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # Normalise variable-suffix frequency columns
            rename = {c: "FRQ_A" for c in df.columns if c.lower().startswith("frq_a_")}
            rename.update({c: "FRQ_U" for c in df.columns if c.lower().startswith("frq_u_")})
            df = df.rename(columns=rename)

            chr_ = find_col(df, CHR_COLS)
            bp_  = find_col(df, BP_COLS)
            if chr_ is None or bp_ is None: continue

            out = pd.DataFrame()
            out["chr"] = pd.to_numeric(df[chr_], errors="coerce")
            out["bp"]  = pd.to_numeric(df[bp_],  errors="coerce")

            beta = find_col(df, BETA_COLS)
            or_  = find_col(df, OR_COLS)
            se_  = find_col(df, SE_COLS)
            p_   = find_col(df, P_COLS)

            if beta:
                out["effect"] = pd.to_numeric(df[beta], errors="coerce")
            elif or_:
                raw = pd.to_numeric(df[or_], errors="coerce")
                out["effect"] = np.log(raw.where((raw > 0) & (raw < 100)))
            else:
                continue

            if se_: out["se"]  = pd.to_numeric(df[se_], errors="coerce")
            if p_:  out["p"]   = pd.to_numeric(df[p_],  errors="coerce")

            out = out.dropna(subset=["chr","bp","effect"])
            out["chr"] = out["chr"].astype(int)
            # Position key: chr + 10kb bin
            out["pos_key"] = out["chr"].astype(str) + "_" + (out["bp"] // 10_000).astype(str)
            frames.append(out)
        except Exception as e:
            continue

    if not frames: return None
    df_all = pd.concat(frames, ignore_index=True)

    # Deduplicate by pos_key: keep the SNP with the strongest signal (lowest p) per bin
    if "p" in df_all.columns:
        df_all = df_all.sort_values("p").drop_duplicates("pos_key", keep="first")
    else:
        df_all = df_all.drop_duplicates("pos_key", keep="first")

    print(f"  [{name}] {len(df_all):,} unique 10-kb bins across {df_all['chr'].nunique()} chromosomes")
    keep = ["chr","bp","effect"]
    for col in ["p","se"]:
        if col in df_all.columns:
            keep.append(col)
    return df_all.set_index("pos_key")[keep]


print("=" * 72)
print("1.  LOADING CONDITIONS")
print("=" * 72)
condition_data = {}
for name, pattern in SHARD_PATTERNS.items():
    result = load_condition(name, pattern)
    if result is not None:
        condition_data[name] = result

conditions = sorted(condition_data.keys())
print(f"\nLoaded {len(conditions)} conditions: {conditions}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BUILD CONDITION × SNP EFFECT MATRIX
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("2.  BUILDING CONDITION × SNP MATRIX")
print("=" * 72)

# Merge on pos_key; keep SNPs present in ≥ 3 conditions
effect_cols = {}
for name in conditions:
    effect_cols[name] = condition_data[name]["effect"].rename(name)

merged = pd.concat(effect_cols.values(), axis=1, join="outer")
min_cond = 3   # need data from at least 3/5 conditions (fills missing with 0)
merged = merged[merged.notna().sum(axis=1) >= min_cond]
print(f"  Shared SNPs (≥{min_cond} conditions): {len(merged):,}")

# Fill remaining NaN with 0 (neutral effect)
merged_filled = merged.fillna(0)

# Z-score each condition so different effect scales (OR vs BETA) are comparable
scaler = StandardScaler()
X = scaler.fit_transform(merged_filled)   # shape: n_snps × n_conditions
X_df = pd.DataFrame(X, index=merged.index, columns=conditions)

print(f"  Matrix shape: {X.shape[0]:,} SNPs × {X.shape[1]} conditions")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PCA ON CONDITIONS  (conditions as points in SNP-effect space)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("3.  PCA — CONDITIONS IN SNP-EFFECT SPACE")
print("=" * 72)

# Transpose: conditions × SNPs → PCA in SNP space
Xt = X.T   # n_conditions × n_snps
pca_cond = PCA(n_components=min(len(conditions), 4))
coords_cond = pca_cond.fit_transform(Xt)

print("  Variance explained per PC:")
for i, v in enumerate(pca_cond.explained_variance_ratio_):
    print(f"    PC{i+1}: {v*100:.1f}%")

# Condition genetic correlation matrix (from this SNP matrix)
cond_corr = pd.DataFrame(X.T @ X / X.shape[0], index=conditions, columns=conditions)
print("\n  Condition genetic correlation matrix (from shared SNPs):")
print(cond_corr.round(3).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HIERARCHICAL CLUSTERING ON CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("4.  HIERARCHICAL CLUSTERING")
print("=" * 72)

# Distance = 1 - |correlation|  (conditions close if they share genetic variance)
dist_mat = 1 - np.abs(cond_corr.values)
np.fill_diagonal(dist_mat, 0)
dist_condensed = squareform(dist_mat, checks=False)

link = linkage(dist_condensed, method="ward")
# Optimal cut for k=2..5
for k in range(2, min(len(conditions), 6)):
    labels = fcluster(link, k, criterion="maxclust")
    if len(np.unique(labels)) > 1:
        sil = silhouette_score(dist_mat, labels, metric="precomputed")
        cdict = {c: l for c, l in zip(conditions, labels)}
        groups = {}
        for c, l in cdict.items(): groups.setdefault(l, []).append(c)
        print(f"  k={k}  silhouette={sil:.3f}  groups={dict(sorted(groups.items()))}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  K-MEANS ON CONDITIONS (silhouette sweep)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("5.  K-MEANS SWEEP")
print("=" * 72)

max_k = len(conditions) - 1
inertias, silhouettes = [], []
for k in range(2, max_k + 1):
    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    km.fit(Xt)
    inertias.append(km.inertia_)
    if k < len(conditions):
        sil = silhouette_score(Xt, km.labels_)
        silhouettes.append(sil)
        print(f"  k={k}  inertia={km.inertia_:.1f}  silhouette={sil:.3f}")
    else:
        silhouettes.append(np.nan)

best_k = int(np.argmax(silhouettes) + 2)
print(f"\n  Best k by silhouette: {best_k}")

# Final k-means with best_k
km_best = KMeans(n_clusters=best_k, n_init=100, random_state=42)
km_best.fit(Xt)
cluster_labels = {c: int(l) for c, l in zip(conditions, km_best.labels_)}

# ══════════════════════════════════════════════════════════════════════════════
# 6.  SNP-LEVEL PCA (SNPs as points in condition space)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("6.  SNP-LEVEL PCA")
print("=" * 72)

# Subsample for speed
np.random.seed(42)
N_SUBSAMPLE = min(4_000, len(X))
idx = np.random.choice(len(X), N_SUBSAMPLE, replace=False)
X_sub = X[idx]

pca_snp = PCA(n_components=3, random_state=42)
snp_coords = pca_snp.fit_transform(X_sub)
print(f"  PCA on {N_SUBSAMPLE:,} SNPs; variance explained: "
      + ", ".join([f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(pca_snp.explained_variance_ratio_)]))

# Color SNPs by which condition shows the strongest |effect|
X_sub_df = X_df.iloc[idx]
dominant_cond = X_sub_df.abs().idxmax(axis=1)

# K-means on SNPs (k=5 clusters in condition space)
km_snp = KMeans(n_clusters=5, n_init=20, random_state=42)
snp_cluster = km_snp.fit_predict(X_sub)

# Describe each SNP cluster by its mean effect profile
print("\n  SNP cluster profiles (mean z-scored effect per condition):")
print("  " + " ".join([f"{c:>12}" for c in conditions]))
for cl in sorted(np.unique(snp_cluster)):
    mask = snp_cluster == cl
    means = X_sub[mask].mean(axis=0)
    vals  = " ".join([f"{v:>12.3f}" for v in means])
    n     = mask.sum()
    print(f"  Cluster {cl} (n={n:>5,}): {vals}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

COND_COLS = [DSM_COLORS.get(c, "#888") for c in conditions]
COND_LBLS = [CONDITION_LABELS.get(c, c) for c in conditions]

# ── Fig 20: PCA biplot + condition correlation matrix ────────────────────────
print("\nFig 20: PCA conditions...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: PCA scatter
ax = axes[0]
for i, (c, col, lbl) in enumerate(zip(conditions, COND_COLS, COND_LBLS)):
    ax.scatter(coords_cond[i, 0], coords_cond[i, 1],
               s=200, color=col, zorder=5, edgecolors="white", lw=1.5)
    # Arrow from origin
    ax.annotate("", xy=(coords_cond[i, 0], coords_cond[i, 1]),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, mutation_scale=12))
    # Label with slight offset
    off_x = 0.02 * (1 if coords_cond[i, 0] >= 0 else -1)
    off_y = 0.02 * (1 if coords_cond[i, 1] >= 0 else -1)
    ax.text(coords_cond[i, 0] + off_x, coords_cond[i, 1] + off_y,
            lbl, fontsize=10, fontweight="bold", color=col,
            ha="left" if coords_cond[i, 0] >= 0 else "right")

ax.axhline(0, color="#ccc", lw=0.7); ax.axvline(0, color="#ccc", lw=0.7)
ax.set_xlabel(f"PC1 ({pca_cond.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca_cond.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)
ax.set_title("Conditions in SNP-Effect Space\n(PCA — each condition = vector of SNP effects)",
             fontsize=11, fontweight="bold")
ax.set_axisbelow(True)

# Right: correlation heatmap
ax2 = axes[1]
cmap = plt.cm.RdBu_r
im = ax2.imshow(cond_corr.values, cmap=cmap, vmin=-0.3, vmax=0.3, aspect="auto")
n = len(conditions)
ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
ax2.set_xticklabels(COND_LBLS, rotation=45, ha="right", fontsize=9.5)
ax2.set_yticklabels(COND_LBLS, fontsize=9.5)
for i in range(n):
    for j in range(n):
        v   = cond_corr.values[i, j]
        col = "white" if abs(v) > 0.2 else "#333"
        ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=col)
plt.colorbar(im, ax=ax2, shrink=0.85)
ax2.set_title("Genetic Correlation Matrix\n(from shared SNP effects)", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig20_pca_conditions.png")
plt.close()
print("  → saved fig20")


# ── Fig 21: Hierarchical clustering dendrogram ───────────────────────────────
print("Fig 21: Dendrogram...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: dendrogram
ax = axes[0]
dendro = dendrogram(
    link,
    labels=COND_LBLS,
    color_threshold=link[-best_k + 1, 2],
    leaf_rotation=45,
    leaf_font_size=11,
    ax=ax,
)
ax.set_title(f"Hierarchical Clustering (Ward linkage)\n"
             f"Data-driven grouping suggests k={best_k} clusters",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Distance (1 − |genetic correlation|)", fontsize=10)
ax.axhline(link[-best_k + 1, 2], color="#E63946", ls="--", lw=1.2,
           label=f"Cut for k={best_k}")
ax.legend(fontsize=9, frameon=False)

# Right: DSM vs data-driven comparison table
ax2 = axes[1]
ax2.axis("off")

# Build data-driven group mapping
hc_labels = fcluster(link, best_k, criterion="maxclust")
hc_groups = {}
for c, l in zip(conditions, hc_labels):
    hc_groups.setdefault(int(l), []).append(CONDITION_LABELS.get(c, c))

dsm_groups_here = {}
for g, conds in DSM_GROUPS.items():
    members = [CONDITION_LABELS.get(c, c) for c in conds if c in conditions]
    if members:
        dsm_groups_here[g] = members

# Draw side-by-side boxes
col_x = [0.05, 0.55]
titles = ["DSM-5 Groupings", f"Genetic Clustering (k={best_k})"]
group_lists = [
    list(dsm_groups_here.items()),
    [(f"Cluster {k}", v) for k, v in sorted(hc_groups.items())],
]
colors_scheme = [
    ["#4C9BE8","#E63946","#8B5CF6","#F4A261","#264653"],
    ["#BC6C25","#2A9D8F","#E76F51","#F72585","#457B9D"],
]
for col_i, (cx, title, glist, cols) in enumerate(zip(col_x, titles, group_lists, colors_scheme)):
    ax2.text(cx, 0.97, title, transform=ax2.transAxes,
             fontsize=11, fontweight="bold", va="top")
    y = 0.87
    for gi, (gname, members) in enumerate(glist):
        col = cols[gi % len(cols)]
        box = FancyBboxPatch((cx, y - 0.10), 0.40, 0.10,
                              boxstyle="round,pad=0.005",
                              fc=col + "22", ec=col, lw=1.5,
                              transform=ax2.transAxes)
        ax2.add_patch(box)
        ax2.text(cx + 0.01, y - 0.02, gname, transform=ax2.transAxes,
                 fontsize=8.5, fontweight="bold", color=col, va="top")
        ax2.text(cx + 0.01, y - 0.06, ", ".join(members), transform=ax2.transAxes,
                 fontsize=8, color="#333", va="top")
        y -= 0.14

plt.tight_layout()
plt.savefig("figures/fig21_dendrogram.png")
plt.close()
print("  → saved fig21")


# ── UMAP on SNPs ──────────────────────────────────────────────────────────────
print("  Running UMAP on SNPs...")
try:
    from umap import UMAP
    umap_model = UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                      metric="euclidean", random_state=42)
    snp_umap = umap_model.fit_transform(X_sub)
    has_umap = True
    print("  UMAP done.")
except Exception as e:
    has_umap = False
    print(f"  UMAP skipped: {e}")

# ── Fig 22: SNP PCA — 2D colored by dominant condition ───────────────────────
print("Fig 22: SNP PCA...")
n_cols = 3 if has_umap else 2
fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))

# Left: colored by dominant condition
ax = axes[0]
for c in conditions:
    mask = dominant_cond.values == c
    ax.scatter(snp_coords[mask, 0], snp_coords[mask, 1],
               s=3, alpha=0.35, color=DSM_COLORS.get(c, "#888"),
               label=CONDITION_LABELS.get(c, c), linewidths=0)
# PCA loadings as arrows
loadings = pca_snp.components_[:2].T  # shape: n_conditions × 2
scale = np.abs(snp_coords[:, :2]).max() * 0.6
for i, (c, col, lbl) in enumerate(zip(conditions, COND_COLS, COND_LBLS)):
    ax.annotate("", xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2, mutation_scale=14))
    ax.text(loadings[i, 0] * scale * 1.12, loadings[i, 1] * scale * 1.12,
            lbl, color=col, fontsize=9.5, fontweight="bold", ha="center")

ax.set_xlabel(f"PC1 ({pca_snp.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca_snp.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
ax.set_title("SNPs in Condition-Effect Space\n(color = dominant condition per SNP)",
             fontsize=11, fontweight="bold")
ax.legend(markerscale=4, fontsize=9, frameon=False, loc="upper right",
          ncol=2, columnspacing=0.5, handlelength=1)

# Right: k=5 SNP clusters
ax2 = axes[1]
cluster_colors = ["#E63946","#4C9BE8","#2A9D8F","#F4A261","#8B5CF6","#264653","#F72585"]
for cl in sorted(np.unique(snp_cluster)):
    mask = snp_cluster == cl
    ax2.scatter(snp_coords[mask, 0], snp_coords[mask, 1],
                s=3, alpha=0.35, color=cluster_colors[cl],
                label=f"Cluster {cl+1} (n={mask.sum():,})", linewidths=0)

ax2.set_xlabel(f"PC1 ({pca_snp.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
ax2.set_ylabel(f"PC2 ({pca_snp.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
ax2.set_title("SNP Clusters in Condition-Effect Space\n(k-means k=5; each cluster = genetic module)",
              fontsize=11, fontweight="bold")
ax2.legend(markerscale=4, fontsize=9, frameon=False, loc="upper right",
           ncol=2, columnspacing=0.5, handlelength=1)

# UMAP panel
if has_umap:
    ax3 = axes[2]
    for cl in sorted(np.unique(snp_cluster)):
        mask = snp_cluster == cl
        ax3.scatter(snp_umap[mask, 0], snp_umap[mask, 1],
                    s=6, alpha=0.5, color=cluster_colors[cl],
                    label=f"Module {cl+1}", linewidths=0)
    ax3.set_xlabel("UMAP 1", fontsize=11)
    ax3.set_ylabel("UMAP 2", fontsize=11)
    ax3.set_title("UMAP of SNPs\n(color = genetic module from k-means)",
                  fontsize=11, fontweight="bold")
    ax3.legend(markerscale=3, fontsize=9, frameon=False, ncol=2)

plt.tight_layout()
plt.savefig("figures/fig22_snp_pca.png")
plt.close()
print("  → saved fig22")


# ── Fig 23: Taxonomy comparison radar plot ────────────────────────────────────
print("Fig 23: Taxonomy comparison...")

# Compute inter-condition genetic distances for DSM groups vs data-driven groups
dist_df = pd.DataFrame(dist_mat, index=conditions, columns=conditions)

def intra_group_dist(groups_dict):
    """Mean within-group genetic distance."""
    dists = []
    for g, members in groups_dict.items():
        m = [c for c in members if c in conditions]
        if len(m) >= 2:
            pairs = [(m[i], m[j]) for i in range(len(m)) for j in range(i+1, len(m))]
            dists.extend([dist_df.loc[a, b] for a, b in pairs])
    return np.mean(dists) if dists else np.nan

def inter_group_dist(groups_dict):
    """Mean between-group genetic distance."""
    group_members = [[c for c in mems if c in conditions]
                     for mems in groups_dict.values()]
    dists = []
    for i in range(len(group_members)):
        for j in range(i+1, len(group_members)):
            for a in group_members[i]:
                for b in group_members[j]:
                    dists.append(dist_df.loc[a, b])
    return np.mean(dists) if dists else np.nan

# DSM groups
dsm_intra = intra_group_dist(DSM_GROUPS)
dsm_inter = inter_group_dist(DSM_GROUPS)

# Data-driven groups
hc_groups_cond = {k: v_list for k, v_list in
                  {k: [c for c, l in zip(conditions, hc_labels) if l == k]
                   for k in np.unique(hc_labels)}.items()}
genetic_intra = intra_group_dist(hc_groups_cond)
genetic_inter = inter_group_dist(hc_groups_cond)

print(f"\n  DSM-5 groupings:      intra={dsm_intra:.3f}, inter={dsm_inter:.3f}, "
      f"ratio={dsm_inter/dsm_intra:.2f}x")
print(f"  Genetic clustering:   intra={genetic_intra:.3f}, inter={genetic_inter:.3f}, "
      f"ratio={genetic_inter/genetic_intra:.2f}x")
print("  (Higher inter/intra ratio = better separation)")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Left: silhouette sweep
ax = axes[0]
ks = list(range(2, max_k + 1))
ax.plot(ks, silhouettes, "o-", color="#264653", lw=2, ms=8, label="Silhouette score")
ax_r = ax.twinx()
ax_r.plot(ks, inertias, "s--", color="#E63946", lw=1.5, ms=6, alpha=0.7, label="Inertia (WCSS)")
ax.set_xlabel("Number of clusters k", fontsize=11)
ax.set_ylabel("Silhouette score", fontsize=11, color="#264653")
ax_r.set_ylabel("Inertia", fontsize=11, color="#E63946")
ax.axvline(best_k, color="#aaa", lw=0.9, ls=":")
ax.set_title(f"Optimal k = {best_k}\n(Silhouette criterion)", fontsize=11, fontweight="bold")
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax_r.legend(loc="upper right", fontsize=9, frameon=False)
ax.set_xticks(ks)

# Middle: inter/intra ratio comparison
ax2 = axes[1]
methods = ["DSM-5\n(categorical)", f"Genetic\nClustering\n(k={best_k})"]
ratios  = [dsm_inter / dsm_intra, genetic_inter / genetic_intra]
intras  = [dsm_intra, genetic_intra]
inters  = [dsm_inter, genetic_inter]
cols    = ["#BC6C25", "#2A9D8F"]

x = np.arange(2)
w = 0.3
bars1 = ax2.bar(x - w/2, intras, width=w, color=[c + "99" for c in cols],
                label="Intra-group distance", edgecolor="white")
bars2 = ax2.bar(x + w/2, inters, width=w, color=cols,
                label="Inter-group distance", edgecolor="white")
for i, (m, r) in enumerate(zip(methods, ratios)):
    ax2.text(i, max(inters[i], intras[i]) + 0.01, f"ratio={r:.2f}x",
             ha="center", fontsize=9.5, fontweight="bold")
ax2.set_xticks(x); ax2.set_xticklabels(methods, fontsize=10)
ax2.set_ylabel("Mean genetic distance", fontsize=11)
ax2.set_title("Separation Quality\n(higher inter/intra = better grouping)",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=9, frameon=False)

# Right: pairwise distance matrix colored by DSM vs genetic
ax3 = axes[2]
n = len(conditions)
im = ax3.imshow(dist_mat, cmap="YlOrBr", aspect="auto", vmin=0, vmax=1)
ax3.set_xticks(range(n)); ax3.set_yticks(range(n))
ax3.set_xticklabels(COND_LBLS, rotation=45, ha="right", fontsize=9.5)
ax3.set_yticklabels(COND_LBLS, fontsize=9.5)

# Box the data-driven clusters
for cl in np.unique(hc_labels):
    members_idx = [i for i, l in enumerate(hc_labels) if l == cl]
    if len(members_idx) >= 2:
        lo, hi = min(members_idx), max(members_idx)
        rect = plt.Rectangle((lo - 0.5, lo - 0.5), hi - lo + 1, hi - lo + 1,
                              fill=False, edgecolor="#E63946", lw=2.5, ls="--")
        ax3.add_patch(rect)

plt.colorbar(im, ax=ax3, shrink=0.8, label="Genetic distance")
ax3.set_title(f"Pairwise Genetic Distance\n(dashed = data-driven clusters, k={best_k})",
              fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/fig23_taxonomy_compare.png")
plt.close()
print("  → saved fig23")


# ── Fig 24: SNP cluster profiles heatmap ─────────────────────────────────────
print("Fig 24: SNP cluster profiles...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: cluster mean effect profiles (heatmap)
cluster_profiles = np.zeros((5, len(conditions)))
n_snps_per_cluster = []
for cl in range(5):
    mask = snp_cluster == cl
    cluster_profiles[cl] = X_sub[mask].mean(axis=0)
    n_snps_per_cluster.append(mask.sum())

ax = axes[0]
vmax = np.abs(cluster_profiles).max()
im = ax.imshow(cluster_profiles, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(conditions))); ax.set_yticks(range(5))
ax.set_xticklabels(COND_LBLS, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels([f"Module {i+1}\n(n={n_snps_per_cluster[i]:,})" for i in range(5)], fontsize=9)
for i in range(5):
    for j in range(len(conditions)):
        v   = cluster_profiles[i, j]
        col = "white" if abs(v) > 0.25 else "#333"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color=col)
plt.colorbar(im, ax=ax, shrink=0.85, label="Mean z-scored effect")
ax.set_title("SNP Genetic Modules\n(mean effect per condition — 5 modules)",
             fontsize=11, fontweight="bold")

# Right: module specificity — which conditions define each module
ax2 = axes[1]
module_names = [
    f"M{i+1}: {', '.join([COND_LBLS[j] for j in np.argsort(np.abs(cluster_profiles[i]))[::-1][:2]])}"
    for i in range(5)
]
for i, (name, col) in enumerate(zip(module_names, cluster_colors[:5])):
    bar_vals = np.abs(cluster_profiles[i])
    sorted_idx = np.argsort(bar_vals)[::-1]
    bar_y = np.arange(len(conditions)) + i * (len(conditions) + 1)
    ax2.barh(bar_y, bar_vals[sorted_idx],
             color=col, alpha=0.7, edgecolor="white", height=0.85)
    # Label
    ax2.text(-0.02, bar_y.mean(), f"M{i+1}", ha="right", va="center",
             fontsize=10, fontweight="bold", color=col)
    for bi, cidx in enumerate(sorted_idx):
        ax2.text(bar_vals[cidx] + 0.005, bar_y[bi],
                 COND_LBLS[cidx], va="center", fontsize=7.5)

ax2.set_yticks([])
ax2.set_xlabel("|Mean effect| per condition (z-score)", fontsize=11)
ax2.set_title("Condition Specificity of Each Genetic Module",
              fontsize=11, fontweight="bold")
ax2.set_xlim(0, np.abs(cluster_profiles).max() * 1.4)

plt.tight_layout()
plt.savefig("figures/fig24_snp_modules.png")
plt.close()
print("  → saved fig24")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"""
Matrix:  {X.shape[0]:,} shared SNPs × {X.shape[1]} conditions
         ({', '.join(COND_LBLS)})

PCA (conditions in SNP-effect space):
  PC1 explains {pca_cond.explained_variance_ratio_[0]*100:.1f}% of variance
  PC2 explains {pca_cond.explained_variance_ratio_[1]*100:.1f}% of variance

Optimal number of genetic groups: k = {best_k}  (silhouette-maximising)

Data-driven clusters:""")
for cl, members in sorted(hc_groups_cond.items()):
    labels_m = [CONDITION_LABELS.get(c, c) for c in members]
    print(f"  Cluster {cl}: {', '.join(labels_m)}")

print(f"""
Separation quality:
  DSM-5:             inter/intra distance ratio = {dsm_inter/dsm_intra:.2f}x
  Genetic clustering: inter/intra distance ratio = {genetic_inter/genetic_intra:.2f}x
  {'→ Genetic clustering separates disorders BETTER than DSM-5' if genetic_inter/genetic_intra > dsm_inter/dsm_intra else '→ DSM-5 separates disorders better on genetic criteria (surprising!)'}

5 SNP genetic modules found:""")
for i in range(5):
    top2 = [COND_LBLS[j] for j in np.argsort(np.abs(cluster_profiles[i]))[::-1][:2]]
    direction = "+" if cluster_profiles[i, np.argsort(np.abs(cluster_profiles[i]))[-1]] > 0 else "−"
    print(f"  Module {i+1} ({n_snps_per_cluster[i]:,} SNPs): dominant in {' + '.join(top2)} ({direction})")
