"""
Full 10-Condition Genetic Taxonomy of Psychiatric Disorders
===========================================================
Loads all 10 PGC GWAS conditions:
  - 5 cached locally via parquet glob (adhd, autism, bipolar, schizophrenia, substance_use)
  - 5 streamed from HuggingFace (anxiety, cross_disorder, eating, mdd, ptsd)

Normalises to pos_key = chr_bp//10000, then builds a 10-condition × SNP effect
matrix for PCA + hierarchical clustering + k-means + UMAP.

Figures saved:
  figures/fig27_full_pca.png        — 3-panel: PCA biplot, correlation heatmap, scree
  figures/fig28_full_dendrogram.png — 3-panel: dendrogram, DSM vs genetic, silhouette
  figures/fig29_full_snp_umap.png   — 2-panel: UMAP by dominant condition + k-means
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datasets import load_dataset
warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Palatino", "Georgia", "Times New Roman"],
    "font.size":            11,
    "axes.linewidth":       0.8,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "figure.dpi":           150,
    "savefig.dpi":          200,
    "savefig.bbox":         "tight",
})

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_N = 200_000
CACHE    = os.path.expanduser("~/.cache/huggingface/hub")

# Color palette
COLORS = {
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

LABELS = {
    "adhd":           "ADHD",
    "autism":         "Autism",
    "schizophrenia":  "SCZ",
    "bipolar":        "Bipolar",
    "mdd":            "MDD",
    "anxiety":        "Anxiety",
    "ptsd":           "PTSD",
    "eating":         "Eating",
    "substance_use":  "SUD",
    "cross_disorder": "Cross-Dis.",
}

# DSM-5 groupings (for comparison panel)
DSM_GROUPS = {
    "Neurodevelopmental":      ["adhd", "autism"],
    "Schizophrenia Spectrum":  ["schizophrenia"],
    "Bipolar & Related":       ["bipolar"],
    "Depressive":              ["mdd"],
    "Anxiety":                 ["anxiety", "ptsd"],
    "Feeding/Eating":          ["eating"],
    "Substance-Related":       ["substance_use"],
    "Cross-Disorder":          ["cross_disorder"],
}

# ── Column aliases ─────────────────────────────────────────────────────────────
CHR_COLS  = {"chr", "chrom", "chromosome", "hg18chr"}
BP_COLS   = {"bp", "pos", "position", "basepair"}
BETA_COLS = {"beta", "effect", "b", "effect_size", "logor"}
OR_COLS   = {"or", "odds_ratio"}
SE_COLS   = {"se", "se_beta", "stderr", "std_err", "stderror"}
P_COLS    = {"p", "pval", "p.value", "p-value", "pvalue", "p_value"}


def find_col(df: pd.DataFrame, aliases: set) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in low:
            return low[a]
    return None


def df_to_condition(df: pd.DataFrame, name: str) -> pd.DataFrame | None:
    """Extract pos_key, effect, p from a raw GWAS dataframe."""
    # Normalise variable-suffix freq cols
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("frq_a_"):
            rename[c] = "FRQ_A"
        elif lc.startswith("frq_u_"):
            rename[c] = "FRQ_U"
    if rename:
        df = df.rename(columns=rename)

    chr_ = find_col(df, CHR_COLS)
    bp_  = find_col(df, BP_COLS)
    if chr_ is None or bp_ is None:
        return None

    out = pd.DataFrame()
    out["chr"] = pd.to_numeric(df[chr_], errors="coerce")
    out["bp"]  = pd.to_numeric(df[bp_],  errors="coerce")

    beta = find_col(df, BETA_COLS)
    or_  = find_col(df, OR_COLS)
    p_   = find_col(df, P_COLS)

    if beta:
        out["effect"] = pd.to_numeric(df[beta], errors="coerce")
    elif or_:
        raw = pd.to_numeric(df[or_], errors="coerce")
        out["effect"] = np.log(raw.where((raw > 0) & (raw < 100)))
    else:
        return None

    if p_:
        out["p"] = pd.to_numeric(df[p_], errors="coerce")

    out = out.dropna(subset=["chr", "bp", "effect"])
    if out.empty:
        return None
    out["chr"] = out["chr"].astype(int)
    out["pos_key"] = out["chr"].astype(str) + "_" + (out["bp"] // 10_000).astype(str)

    # Deduplicate: keep lowest-p (or first) per 10-kb bin
    if "p" in out.columns:
        out = out.sort_values("p").drop_duplicates("pos_key", keep="first")
    else:
        out = out.drop_duplicates("pos_key", keep="first")

    return out.set_index("pos_key")[["effect"] + (["p"] if "p" in out.columns else [])]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════

# Cached conditions: glob parquets from HF hub cache
CACHED = {
    "adhd":          f"{CACHE}/datasets--OpenMed--pgc-adhd/snapshots/*/data/adhd2022/*.parquet",
    "autism":        f"{CACHE}/datasets--OpenMed--pgc-autism/snapshots/*/data/asd2019/*.parquet",
    "bipolar":       f"{CACHE}/datasets--OpenMed--pgc-bipolar/snapshots/*/data/bip2024/*.parquet",
    "schizophrenia": f"{CACHE}/datasets--OpenMed--pgc-schizophrenia/snapshots/*/data/scz2022/*.parquet",
    "substance_use": f"{CACHE}/datasets--OpenMed--pgc-substance-use/snapshots/*/data/SUD2023/*.parquet",
}

# Streamed conditions: (repo, data_dir_or_None)
STREAMED = {
    "anxiety":        ("OpenMed/pgc-anxiety",        None),
    "cross_disorder": ("OpenMed/pgc-cross-disorder",  None),
    "eating":         ("OpenMed/pgc-eating-disorders", None),
    "mdd":            ("OpenMed/pgc-mdd",              "data/mdd2023diverse"),
    "ptsd":           ("OpenMed/pgc-ptsd",             None),
}

print("=" * 72)
print("1.  LOADING CONDITIONS")
print("=" * 72)

condition_data: dict[str, pd.DataFrame] = {}

# ── 1a. Cached (parquet glob) ──────────────────────────────────────────────────
for name, pattern in CACHED.items():
    print(f"  [{name}] loading from cache...", end=" ", flush=True)
    files = sorted(glob.glob(pattern))[:25]
    if not files:
        print("no files found — SKIP")
        continue
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            res = df_to_condition(df, name)
            if res is not None:
                frames.append(res)
        except Exception:
            continue
    if not frames:
        print("parse failed — SKIP")
        continue
    combined = pd.concat(frames)
    # Re-deduplicate across shards
    if "p" in combined.columns:
        combined = combined.sort_values("p").groupby(level=0).first()
    else:
        combined = combined[~combined.index.duplicated(keep="first")]
    condition_data[name] = combined
    n_chr = combined.reset_index()["pos_key"].str.split("_").str[0].nunique() if len(combined) else 0
    print(f"{len(combined):,} unique 10-kb bins | {n_chr} chromosomes")

# ── 1b. Streamed (HuggingFace) ────────────────────────────────────────────────
for name, (repo, data_dir) in STREAMED.items():
    print(f"  [{name}] streaming from HF...", end=" ", flush=True)
    try:
        kwargs: dict = {"split": "train", "streaming": True}
        if data_dir:
            kwargs["data_dir"] = data_dir
        ds = load_dataset(repo, **kwargs)
        rows = list(ds.take(SAMPLE_N))
        if not rows:
            print("empty — SKIP")
            continue
        df = pd.DataFrame(rows)
        if df.shape[1] <= 2:
            print("VCF/empty schema — SKIP")
            continue
        res = df_to_condition(df, name)
        if res is None or res.empty:
            print("no chr/bp/effect — SKIP")
            continue
        condition_data[name] = res
        n_chr = res.reset_index()["pos_key"].str.split("_").str[0].nunique()
        print(f"{len(res):,} unique 10-kb bins | {n_chr} chromosomes")
    except Exception as e:
        print(f"ERROR: {str(e)[:120]}")

conditions = sorted(condition_data.keys())
print(f"\nLoaded {len(conditions)}/10 conditions: {conditions}")

if len(conditions) < 4:
    raise RuntimeError(f"Only {len(conditions)} conditions loaded — need ≥4 to proceed.")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BUILD CONDITION × SNP EFFECT MATRIX
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("2.  BUILDING CONDITION × SNP MATRIX")
print("=" * 72)

MIN_COND = 4
effect_cols = {name: condition_data[name]["effect"].rename(name) for name in conditions}
merged = pd.concat(effect_cols.values(), axis=1, join="outer")
merged = merged[merged.notna().sum(axis=1) >= MIN_COND]
print(f"  SNPs present in ≥{MIN_COND} conditions: {len(merged):,}")

# Fill NaN with 0
mat = merged.fillna(0)

# Z-score each condition
mat_z = mat.copy()
for col in mat_z.columns:
    col_data = mat_z[col]
    mu, sd = col_data.mean(), col_data.std()
    mat_z[col] = (col_data - mu) / sd if sd > 0 else col_data - mu

print(f"  Matrix shape: {mat_z.shape}  (SNPs × conditions)")
print(f"  Conditions: {list(mat_z.columns)}")

# ── SNP matrix transposed: conditions × SNPs (for condition-space PCA) ────────
# rows = conditions, cols = SNPs
cond_mat = mat_z.T   # shape: n_conditions × n_snps


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PCA ON CONDITIONS (in SNP-effect space)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("3.  PCA ON CONDITIONS")
print("=" * 72)

n_comp = min(len(conditions), 5)
pca = PCA(n_components=n_comp)
cond_pcs = pca.fit_transform(cond_mat.values)   # shape: n_conditions × n_comp
explained = pca.explained_variance_ratio_

print(f"  PC1 variance: {explained[0]*100:.1f}%")
print(f"  PC2 variance: {explained[1]*100:.1f}%")
for i, ev in enumerate(explained):
    print(f"    PC{i+1}: {ev*100:.1f}%")

# Correlation matrix between conditions
corr_mat = mat_z.corr()
print("\n  Condition correlation matrix:")
print(corr_mat.round(3).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HIERARCHICAL CLUSTERING ON CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("4.  HIERARCHICAL CLUSTERING")
print("=" * 72)

linkage_mat = linkage(cond_mat.values, method="ward")
print("  Ward linkage computed on conditions × SNPs matrix")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  K-MEANS SWEEP k=2..8
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("5.  K-MEANS SWEEP k=2..8")
print("=" * 72)

# k-means on SNP-effect space (SNPs as observations, conditions as features)
# Use mat_z: shape (n_snps, n_conditions)
snp_data = mat_z.values
K_RANGE = range(2, 9)
sil_scores = []
inertias   = []
km_labels  = {}

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labs = km.fit_predict(snp_data)
    km_labels[k] = labs
    inertias.append(km.inertia_)
    if k < snp_data.shape[0]:
        sil = silhouette_score(snp_data, labs, sample_size=min(10_000, len(snp_data)))
        sil_scores.append(sil)
    else:
        sil_scores.append(float("nan"))
    print(f"    k={k}: silhouette={sil_scores[-1]:.4f}  inertia={inertias[-1]:.2e}")

best_k = list(K_RANGE)[np.nanargmax(sil_scores)]
print(f"\n  Best k (highest silhouette): k={best_k}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  UMAP ON SNPs
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("6.  UMAP ON SNPs")
print("=" * 72)

try:
    import umap
    # Subsample for speed if needed
    MAX_UMAP = 50_000
    if len(snp_data) > MAX_UMAP:
        idx = np.random.RandomState(42).choice(len(snp_data), MAX_UMAP, replace=False)
        snp_sub  = snp_data[idx]
        snp_keys = mat_z.index[idx]
    else:
        idx      = np.arange(len(snp_data))
        snp_sub  = snp_data
        snp_keys = mat_z.index

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(snp_sub)
    print(f"  UMAP embedding: {embedding.shape}")

    # Dominant condition per SNP (condition with max |effect|)
    sub_mat = mat_z.iloc[idx]
    dominant_cond = sub_mat.abs().idxmax(axis=1).values
    umap_ok = True
except Exception as e:
    print(f"  UMAP failed: {e}")
    umap_ok = False


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("7.  GENERATING FIGURES")
print("=" * 72)


# ── Helper to remove top/right spines ─────────────────────────────────────────
def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 27 — PCA biplot + correlation heatmap + scree
# ══════════════════════════════════════════════════════════════════════════════

fig27, axes = plt.subplots(1, 3, figsize=(18, 6))
fig27.suptitle("Full 10-Condition Genetic Taxonomy — PCA Analysis",
               fontsize=14, fontweight="bold", y=1.02)

# Panel A: PCA biplot (PC1 vs PC2)
ax = axes[0]
for i, cond in enumerate(cond_mat.index):
    x, y = cond_pcs[i, 0], cond_pcs[i, 1]
    color = COLORS.get(cond, "#888888")
    ax.scatter(x, y, s=180, color=color, zorder=3, edgecolors="white", linewidths=0.8)
    ax.annotate(LABELS.get(cond, cond), (x, y),
                textcoords="offset points", xytext=(6, 4),
                fontsize=9, color=color, fontweight="bold")

ax.axhline(0, color="#cccccc", lw=0.7, zorder=1)
ax.axvline(0, color="#cccccc", lw=0.7, zorder=1)
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var.)", fontsize=10)
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var.)", fontsize=10)
ax.set_title("A. Conditions in SNP-effect space", fontsize=11, fontweight="bold")
clean_ax(ax)

# Panel B: Correlation heatmap
ax = axes[1]
cond_order = list(corr_mat.columns)
corr_arr   = corr_mat.loc[cond_order, cond_order].values
n = len(cond_order)
im = ax.imshow(corr_arr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels([LABELS.get(c, c) for c in cond_order], rotation=45, ha="right", fontsize=8)
ax.set_yticklabels([LABELS.get(c, c) for c in cond_order], fontsize=8)
for i in range(n):
    for j in range(n):
        v = corr_arr[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=6.5, color="white" if abs(v) > 0.4 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
ax.set_title("B. Cross-disorder effect correlation", fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Panel C: Scree plot
ax = axes[2]
pc_nums = np.arange(1, n_comp + 1)
ax.bar(pc_nums, explained * 100, color="#4C9BE8", alpha=0.8, zorder=3)
ax.plot(pc_nums, np.cumsum(explained) * 100, "o-", color="#E63946",
        lw=2, ms=6, zorder=4, label="Cumulative")
ax.set_xlabel("Principal Component", fontsize=10)
ax.set_ylabel("Variance Explained (%)", fontsize=10)
ax.set_title("C. Scree plot", fontsize=11, fontweight="bold")
ax.set_xticks(pc_nums)
ax.legend(fontsize=9, frameon=False)
clean_ax(ax)

plt.tight_layout()
fig27.savefig("figures/fig27_full_pca.png", dpi=200, bbox_inches="tight")
plt.close(fig27)
print("  Saved figures/fig27_full_pca.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 28 — Dendrogram + DSM vs genetic clusters + silhouette sweep
# ══════════════════════════════════════════════════════════════════════════════

fig28 = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 2, figure=fig28, hspace=0.4, wspace=0.35)
fig28.suptitle("Full 10-Condition Genetic Taxonomy — Clustering",
               fontsize=14, fontweight="bold")

# Panel A: Dendrogram
ax_dend = fig28.add_subplot(gs[0, 0])
dend_labels = [LABELS.get(c, c) for c in cond_mat.index]
dend_colors = [COLORS.get(c, "#888888") for c in cond_mat.index]

# Map leaf colors
from scipy.cluster.hierarchy import set_link_color_palette
set_link_color_palette(["#666666"])

dend = dendrogram(
    linkage_mat,
    labels=dend_labels,
    leaf_rotation=45,
    leaf_font_size=10,
    ax=ax_dend,
    color_threshold=linkage_mat[-2, 2] * 0.7,
    above_threshold_color="#888888",
)
# Color leaf labels
for lbl in ax_dend.get_xticklabels():
    txt = lbl.get_text()
    # find matching condition
    for cond, label in LABELS.items():
        if label == txt:
            lbl.set_color(COLORS.get(cond, "black"))
            break

ax_dend.set_title("A. Ward hierarchical clustering", fontsize=11, fontweight="bold")
ax_dend.set_ylabel("Distance", fontsize=9)
clean_ax(ax_dend)

# Panel B: DSM-5 vs Genetic groupings side-by-side
ax_comp = fig28.add_subplot(gs[0, 1])
ax_comp.set_xlim(0, 10)
ax_comp.set_ylim(-0.5, len(conditions) + 1)
ax_comp.axis("off")
ax_comp.set_title("B. DSM-5 vs Genetic cluster groupings", fontsize=11, fontweight="bold")

# Build genetic clusters from dendrogram cut
genetic_labels = fcluster(linkage_mat, best_k, criterion="maxclust")
# Map condition index → cluster
cond_to_cluster = {cond_mat.index[i]: genetic_labels[i] for i in range(len(cond_mat.index))}

# DSM column (left)
ax_comp.text(1.5, len(conditions) + 0.5, "DSM-5 Groups", fontsize=10,
             fontweight="bold", ha="center", va="bottom")
y_pos = len(conditions) - 0.5
for group_name, members in DSM_GROUPS.items():
    members_present = [m for m in members if m in cond_to_cluster]
    if not members_present:
        continue
    box_h = len(members_present) * 1.0
    rect = FancyBboxPatch((0.1, y_pos - box_h + 0.1), 2.8, box_h - 0.2,
                          boxstyle="round,pad=0.05", linewidth=1,
                          edgecolor="#888888", facecolor="#f8f8f8")
    ax_comp.add_patch(rect)
    ax_comp.text(1.5, y_pos - box_h / 2 + 0.4, group_name,
                 fontsize=7.5, ha="center", va="center", fontweight="bold", color="#444444")
    for j, m in enumerate(members_present):
        ax_comp.text(1.5, y_pos - box_h / 2 - 0.1 + (len(members_present) - 1 - j) * 0.5,
                     LABELS.get(m, m),
                     fontsize=8, ha="center", va="center",
                     color=COLORS.get(m, "black"))
    y_pos -= box_h

# Genetic column (right)
ax_comp.text(7.5, len(conditions) + 0.5, f"Genetic Clusters (k={best_k})", fontsize=10,
             fontweight="bold", ha="center", va="bottom")
cluster_ids = sorted(set(cond_to_cluster.values()))
y_pos = len(conditions) - 0.5
for cid in cluster_ids:
    members = [c for c, cl in cond_to_cluster.items() if cl == cid]
    box_h = len(members) * 1.0
    rect = FancyBboxPatch((5.1, y_pos - box_h + 0.1), 4.8, box_h - 0.2,
                          boxstyle="round,pad=0.05", linewidth=1,
                          edgecolor="#888888", facecolor="#f8f8f8")
    ax_comp.add_patch(rect)
    ax_comp.text(7.5, y_pos - box_h / 2 + 0.4, f"Cluster {cid}",
                 fontsize=7.5, ha="center", va="center", fontweight="bold", color="#444444")
    for j, m in enumerate(members):
        ax_comp.text(7.5, y_pos - box_h / 2 - 0.1 + (len(members) - 1 - j) * 0.5,
                     LABELS.get(m, m),
                     fontsize=8, ha="center", va="center",
                     color=COLORS.get(m, "black"))
    y_pos -= box_h

# Panel C: Silhouette scores
ax_sil = fig28.add_subplot(gs[1, 0])
ks = list(K_RANGE)
bar_colors = ["#E63946" if k == best_k else "#4C9BE8" for k in ks]
ax_sil.bar(ks, sil_scores, color=bar_colors, alpha=0.85, zorder=3)
ax_sil.set_xlabel("Number of clusters (k)", fontsize=10)
ax_sil.set_ylabel("Silhouette score", fontsize=10)
ax_sil.set_title("C. Silhouette score by k", fontsize=11, fontweight="bold")
ax_sil.set_xticks(ks)
ax_sil.axvline(best_k, color="#E63946", lw=1.5, ls="--", alpha=0.7, label=f"Best k={best_k}")
ax_sil.legend(fontsize=9, frameon=False)
clean_ax(ax_sil)

# Panel D: Inertia (elbow)
ax_iner = fig28.add_subplot(gs[1, 1])
ax_iner.plot(ks, inertias, "o-", color="#8B5CF6", lw=2, ms=7, zorder=3)
ax_iner.fill_between(ks, inertias, alpha=0.15, color="#8B5CF6")
ax_iner.set_xlabel("Number of clusters (k)", fontsize=10)
ax_iner.set_ylabel("Inertia", fontsize=10)
ax_iner.set_title("D. Elbow plot (inertia)", fontsize=11, fontweight="bold")
ax_iner.set_xticks(ks)
clean_ax(ax_iner)

fig28.savefig("figures/fig28_full_dendrogram.png", dpi=200, bbox_inches="tight")
plt.close(fig28)
print("  Saved figures/fig28_full_dendrogram.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 29 — UMAP of SNPs
# ══════════════════════════════════════════════════════════════════════════════

if umap_ok:
    fig29, axes29 = plt.subplots(1, 2, figsize=(16, 7))
    fig29.suptitle("Full 10-Condition Genetic Taxonomy — SNP UMAP",
                   fontsize=14, fontweight="bold")

    # Panel A: colored by dominant condition
    ax = axes29[0]
    for cond in conditions:
        mask = dominant_cond == cond
        if mask.sum() == 0:
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   s=3, alpha=0.5, color=COLORS.get(cond, "#888888"),
                   label=LABELS.get(cond, cond), rasterized=True)
    ax.set_title("A. SNPs colored by dominant condition", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    legend_handles = [
        Patch(color=COLORS.get(c, "#888888"), label=LABELS.get(c, c))
        for c in conditions if (dominant_cond == c).sum() > 0
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, frameon=False,
              loc="upper right", ncol=2, markerscale=2)
    clean_ax(ax)

    # Panel B: colored by best k-means cluster
    ax = axes29[1]
    km_best = km_labels[best_k][idx] if len(snp_data) > MAX_UMAP else km_labels[best_k]
    cluster_palette = plt.cm.tab10(np.linspace(0, 0.9, best_k))
    for cid in range(best_k):
        mask = km_best == cid
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   s=3, alpha=0.5, color=cluster_palette[cid],
                   label=f"Cluster {cid+1}", rasterized=True)
    ax.set_title(f"B. SNPs colored by k-means cluster (k={best_k})", fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2, markerscale=2)
    clean_ax(ax)

    plt.tight_layout()
    fig29.savefig("figures/fig29_full_snp_umap.png", dpi=200, bbox_inches="tight")
    plt.close(fig29)
    print("  Saved figures/fig29_full_snp_umap.png")
else:
    print("  UMAP not available — fig29 skipped")

print("\n" + "=" * 72)
print("FULL TAXONOMY COMPLETE")
print("=" * 72)
print(f"  Conditions loaded:  {len(conditions)}/10")
print(f"  Shared SNP bins:    {len(mat_z):,}")
print(f"  Best k (silhouette): {best_k}")
print(f"  Figures: fig27_full_pca.png, fig28_full_dendrogram.png, fig29_full_snp_umap.png")
