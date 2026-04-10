"""
Gene Module Mapping
===================
Maps the 5 SNP genetic modules from genetic_taxonomy.py to genes and
biological pathways using a positional gene lookup + curated pathway sets.

Outputs:
  data_cache/module_genes.csv      — top genes per module
  figures/fig_gene_modules.png     — lollipop charts + pathway heatmap
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import glob
warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)
os.makedirs("data_cache", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Palatino","Georgia","Times New Roman"],
    "font.size": 11, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})

# ── Curated gene list: chr/bp positions for known psychiatric loci (hg19) ─────
# Source: GWAS Catalog + PGC flagship papers
KNOWN_GENES = pd.DataFrame([
    # Neurodevelopmental
    ("NRXN1",   2,  50145643,  50710686, "synaptic adhesion, autism/SCZ"),
    ("SHANK3",  22, 51113071,  51187936, "PSD scaffold, autism"),
    ("DYRK1A",  21, 38739189,  38887425, "neurodevelopmental, ID"),
    ("KDM5C",   23, 53224240,  53272954, "histone demethylase, ADHD/ASD"),
    ("RBFOX1",  16, 5239060,   7175424,  "RNA splicing, autism"),
    ("CNTN4",   3,  2222826,   3065078,  "axon guidance, autism"),
    ("CNTNAP2", 7,  146116002, 148420998,"synaptic, autism/ADHD"),
    ("NRXN3",   14, 79100000,  80100000, "synaptic adhesion, SUD/ADHD"),
    ("RAI1",    17, 17584577,  17714322, "chromatin, neurodevelop"),
    # Dopaminergic / reward
    ("DRD2",    11, 113409605, 113475425,"dopamine receptor D2, SUD/SCZ/bipolar"),
    ("DRD4",    11, 637273,    640705,   "dopamine receptor D4, ADHD"),
    ("DAT1",    5,  1392905,   1445114,  "dopamine transporter, ADHD"),
    ("COMT",    22, 19941570,  19969801, "catechol-O-methyltransferase, SCZ"),
    ("MAOA",    23, 43515420,  43600200, "monoamine oxidase A, impulsivity"),
    # Glutamatergic / synaptic
    ("GRIN2A",  16, 9852476,   10228545, "NMDA receptor, epilepsy/SCZ"),
    ("GRIN2B",  12, 13717190,  14116930, "NMDA receptor, NDD/SCZ"),
    ("SHANK2",  11, 70607885,  70917777, "PSD scaffold, autism"),
    ("DLGAP1",  18, 3509688,   3911400,  "PSD, OCD/SCZ"),
    # GABAergic
    ("GABRB3",  15, 26773498,  27190710, "GABA receptor, epilepsy/autism"),
    ("GABRA1",  5,  161311231, 161425064,"GABA receptor, epilepsy"),
    # Bipolar / mood
    ("CACNA1C", 12, 2162000,   2807483,  "L-type VGCC, bipolar/SCZ"),
    ("ANK3",    10, 61501660,  62361890, "ankyrin G, bipolar"),
    ("KCNQ3",   8,  32408484,  32578785, "potassium channel, bipolar"),
    ("CLOCK",   4,  183085553, 183189946,"circadian, bipolar/MDD"),
    # Serotonergic / depression
    ("SLC6A4",  17, 28562750,  28628879, "serotonin transporter, MDD/anxiety"),
    ("HTR2A",   13, 47401609,  47524218, "5-HT2A receptor, MDD/SCZ"),
    ("FKBP5",   6,  35547976,  35733640, "stress response, PTSD/MDD"),
    # Immune / inflammation
    ("MHC",     6,  28477797,  33448354, "major histocompatibility complex, SCZ"),
    ("CRP",     1,  159682233, 159684610,"C-reactive protein, depression"),
    # Chromatin / gene regulation
    ("KDM5B",   1,  202697964, 202832738,"histone demethylase, autism"),
    ("SETD5",   3,  9388697,   9559736,  "histone methyltransferase, NDD"),
    # Transcription / NDD
    ("TCF4",    18, 52888435,  53200000, "transcription factor, Pitt-Hopkins/SCZ"),
    ("FOXP1",   3,  71017682,  71656867, "transcription factor, autism/ID"),
    ("FOXP2",   7,  114055186, 114333827,"language/speech, autism"),
], columns=["gene","chr","start","end","function"])
KNOWN_GENES["mid"] = (KNOWN_GENES["start"] + KNOWN_GENES["end"]) // 2

# ── Curated pathway gene sets ──────────────────────────────────────────────────
PATHWAYS = {
    "Dopamine signalling":     ["DRD2","DRD4","DAT1","COMT","MAOA"],
    "Glutamate/NMDA":          ["GRIN2A","GRIN2B","SHANK2","SHANK3","DLGAP1"],
    "GABA inhibitory":         ["GABRB3","GABRA1"],
    "Synaptic adhesion":       ["NRXN1","NRXN3","CNTNAP2","CNTN4","SHANK3"],
    "Voltage-gated channels":  ["CACNA1C","KCNQ3","GABRB3"],
    "Chromatin/epigenetic":    ["KDM5C","KDM5B","SETD5","DYRK1A","RAI1"],
    "Transcription factors":   ["TCF4","FOXP1","FOXP2"],
    "Circadian/stress":        ["CLOCK","FKBP5","SLC6A4","HTR2A"],
    "Immune/inflammation":     ["MHC","CRP"],
    "Cytoskeletal/scaffold":   ["ANK3","SHANK2","SHANK3","DLGAP1"],
}

COND_COLORS = {
    "adhd": "#4C9BE8", "autism": "#2A9D8F", "bipolar": "#8B5CF6",
    "schizophrenia": "#E63946", "substance_use": "#264653",
}
LABELS = {"adhd": "ADHD", "autism": "Autism", "bipolar": "Bipolar",
          "schizophrenia": "SCZ", "substance_use": "SUD"}
MODULE_COLORS = ["#E63946","#4C9BE8","#2A9D8F","#F4A261","#8B5CF6"]

# ══════════════════════════════════════════════════════════════════════════════
# 1.  REBUILD MODULE ASSIGNMENTS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 68)
print("1.  REBUILDING SNP MODULES FROM GENETIC_TAXONOMY")
print("=" * 68)

CACHE = os.path.expanduser("~/.cache/huggingface/hub")
PATTERNS = {
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
P_COLS    = {"p","pval","p.value","p-value","pvalue","p_value"}

def find_col(df, aliases):
    low = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in low: return low[a]
    return None

def load_cond_full(name, pattern, max_shards=25):
    files = sorted(glob.glob(pattern))[:max_shards]
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            rename = {c: "FRQ_A" for c in df.columns if c.lower().startswith("frq_a_")}
            df = df.rename(columns=rename)
            chr_ = find_col(df, CHR_COLS); bp_ = find_col(df, BP_COLS)
            if chr_ is None or bp_ is None: continue
            out = pd.DataFrame()
            out["chr"] = pd.to_numeric(df[chr_], errors="coerce").astype("Int64")
            out["bp"]  = pd.to_numeric(df[bp_],  errors="coerce")
            beta = find_col(df, BETA_COLS); or_ = find_col(df, OR_COLS)
            p_   = find_col(df, P_COLS)
            if beta:
                out["effect"] = pd.to_numeric(df[beta], errors="coerce")
            elif or_:
                raw = pd.to_numeric(df[or_], errors="coerce")
                out["effect"] = np.log(raw.where((raw > 0) & (raw < 100)))
            else: continue
            if p_: out["p"] = pd.to_numeric(df[p_], errors="coerce")
            out = out.dropna(subset=["chr","bp","effect"])
            out["chr"]     = out["chr"].astype(int)
            out["pos_key"] = out["chr"].astype(str) + "_" + (out["bp"] // 10_000).astype(str)
            frames.append(out)
        except: continue
    if not frames: return None
    df_all = pd.concat(frames, ignore_index=True)
    if "p" in df_all.columns:
        df_all = df_all.sort_values("p").drop_duplicates("pos_key", keep="first")
    else:
        df_all = df_all.drop_duplicates("pos_key", keep="first")
    return df_all.set_index("pos_key")[["chr","bp","effect"] + (["p"] if "p" in df_all.columns else [])]

cond_data = {}
for name, pat in PATTERNS.items():
    r = load_cond_full(name, pat)
    if r is not None:
        cond_data[name] = r
        print(f"  [{name}] {len(r):,} bins")

conditions = sorted(cond_data.keys())
effect_series = {n: cond_data[n]["effect"].rename(n) for n in conditions}
merged = pd.concat(effect_series.values(), axis=1, join="outer")
merged = merged[merged.notna().sum(axis=1) >= 3].fillna(0)

scaler = StandardScaler()
X = scaler.fit_transform(merged.values)
X_df = pd.DataFrame(X, index=merged.index, columns=conditions)

np.random.seed(42)
N_SUB = min(4000, len(X))
idx   = np.random.choice(len(X), N_SUB, replace=False)
X_sub = X[idx]
pos_sub = merged.index[idx]

km = KMeans(n_clusters=5, n_init=20, random_state=42)
snp_cluster = km.fit_predict(X_sub)
cluster_profiles = np.array([X_sub[snp_cluster == cl].mean(axis=0) for cl in range(5)])

# Also get chr/bp for each subsampled SNP
chr_bp = pd.DataFrame({
    "pos_key": pos_sub,
    "module":  snp_cluster,
})
chr_bp["chr"] = [int(k.split("_")[0]) for k in pos_sub]
chr_bp["bp"]  = [int(k.split("_")[1]) * 10_000 for k in pos_sub]

print(f"\nModule assignments: {dict(zip(range(5), [int((snp_cluster==i).sum()) for i in range(5)]))}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MAP SNPs TO NEAREST KNOWN GENE (±500 kb)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 68)
print("2.  MAPPING SNPs TO NEAREST GENE")
print("=" * 68)

FLANK_KB = 500_000

gene_hits = []
for _, snp in chr_bp.iterrows():
    nearby = KNOWN_GENES[
        (KNOWN_GENES["chr"] == snp["chr"]) &
        (KNOWN_GENES["mid"].between(snp["bp"] - FLANK_KB, snp["bp"] + FLANK_KB))
    ].copy()
    if nearby.empty: continue
    nearby["dist"] = (nearby["mid"] - snp["bp"]).abs()
    nearest = nearby.nsmallest(1, "dist").iloc[0]
    gene_hits.append({
        "pos_key": snp["pos_key"], "chr": snp["chr"], "bp": snp["bp"],
        "module": snp["module"], "gene": nearest["gene"],
        "gene_function": nearest["function"], "dist_kb": nearest["dist"] / 1000,
    })

gene_hits_df = pd.DataFrame(gene_hits)
print(f"  {len(gene_hits_df)} SNPs mapped to {gene_hits_df['gene'].nunique()} unique genes")

# Count gene hits per module
gene_counts = (gene_hits_df.groupby(["module","gene","gene_function"])
               .size().reset_index(name="n_snps")
               .sort_values(["module","n_snps"], ascending=[True, False]))
gene_counts.to_csv("data_cache/module_genes.csv", index=False)
print("  → Saved data_cache/module_genes.csv")

print("\n  Top genes per module:")
for mod in range(5):
    top = gene_counts[gene_counts["module"] == mod].head(5)
    lbl = [LABELS[conditions[j]] for j in np.argsort(np.abs(cluster_profiles[mod]))[::-1][:2]]
    print(f"\n  Module {mod+1} (dominant: {'+'.join(lbl)}):")
    for _, row in top.iterrows():
        print(f"    {row['gene']:<12} n={row['n_snps']:>3}  {row['gene_function']}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PATHWAY ENRICHMENT (HYPERGEOMETRIC-LIKE)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 68)
print("3.  PATHWAY ENRICHMENT")
print("=" * 68)

universe_genes = set(gene_hits_df["gene"])
n_universe     = len(universe_genes)

from scipy.stats import hypergeom

pathway_results = []
for pathway, pathway_genes in PATHWAYS.items():
    pathway_in_universe = [g for g in pathway_genes if g in universe_genes]
    K = len(pathway_in_universe)
    if K == 0: continue
    for mod in range(5):
        mod_genes  = set(gene_hits_df[gene_hits_df["module"] == mod]["gene"])
        n_mod      = len(mod_genes)
        k_hit      = len(mod_genes & set(pathway_genes))
        # Hypergeometric: P(X >= k | N, K, n)
        p_val = hypergeom.sf(k_hit - 1, n_universe, K, n_mod) if k_hit > 0 else 1.0
        pathway_results.append({
            "pathway": pathway, "module": mod,
            "n_pathway_genes": K, "n_module_genes": n_mod,
            "n_overlap": k_hit, "p_hypergeom": p_val,
            "fold_enrichment": (k_hit / n_mod) / (K / n_universe) if n_mod > 0 and K > 0 else 0,
        })

pathway_df = pd.DataFrame(pathway_results)
pathway_df["-log10p"] = -np.log10(pathway_df["p_hypergeom"].clip(1e-10))

print("\n  Top enrichments (p<0.2, fold>1):")
sig_pw = pathway_df[(pathway_df["p_hypergeom"] < 0.2) & (pathway_df["fold_enrichment"] > 1)]
for _, r in sig_pw.sort_values("p_hypergeom").head(12).iterrows():
    lbl = [LABELS[conditions[j]] for j in np.argsort(np.abs(cluster_profiles[r["module"]]))[::-1][:2]]
    print(f"  M{r['module']+1} ({'+'.join(lbl)}) — {r['pathway']:<30}  "
          f"fold={r['fold_enrichment']:.1f}  p={r['p_hypergeom']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 68)
print("4.  FIGURES")
print("=" * 68)

fig = plt.figure(figsize=(18, 12))
gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4)

module_names = []
for mod in range(5):
    top2 = [LABELS[conditions[j]] for j in np.argsort(np.abs(cluster_profiles[mod]))[::-1][:2]]
    module_names.append(f"M{mod+1}: {'+'.join(top2)}")

# ── Top row: gene lollipop charts per module ──────────────────────────────────
for mod in range(3):
    ax = fig.add_subplot(gs[0, mod])
    top_genes = gene_counts[gene_counts["module"] == mod].head(8)
    if top_genes.empty:
        ax.text(0.5, 0.5, "No mapped genes", ha="center", transform=ax.transAxes)
        continue
    y = np.arange(len(top_genes))
    col = MODULE_COLORS[mod]
    ax.barh(y, top_genes["n_snps"].values, color=col, alpha=0.75, height=0.6, edgecolor="white")
    ax.scatter(top_genes["n_snps"].values, y, s=80, color=col, zorder=5, edgecolors="white", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(top_genes["gene"].values, fontsize=9.5)
    ax.set_xlabel("SNPs in module", fontsize=9)
    ax.set_title(module_names[mod], fontsize=10, fontweight="bold", color=col)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, alpha=0.3, lw=0.6)

# ── Bottom-left: pathway heatmap ──────────────────────────────────────────────
ax_pw = fig.add_subplot(gs[1, :2])
pw_pivot = pathway_df.pivot(index="pathway", columns="module", values="-log10p").fillna(0)
pw_pivot.columns = [f"M{c+1}" for c in pw_pivot.columns]
cmap = plt.cm.YlOrRd
im = ax_pw.imshow(pw_pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=max(pw_pivot.values.max(), 1))
ax_pw.set_xticks(range(len(pw_pivot.columns)))
ax_pw.set_xticklabels(pw_pivot.columns, fontsize=10)
ax_pw.set_yticks(range(len(pw_pivot.index)))
ax_pw.set_yticklabels(pw_pivot.index, fontsize=9)
for i in range(len(pw_pivot.index)):
    for j in range(len(pw_pivot.columns)):
        v = pw_pivot.values[i, j]
        col = "white" if v > pw_pivot.values.max() * 0.6 else "#333"
        ax_pw.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color=col)
plt.colorbar(im, ax=ax_pw, shrink=0.8, label=r"$-\log_{10}(P)$ hypergeometric")
ax_pw.set_title("Pathway Enrichment per Genetic Module", fontsize=11, fontweight="bold")

# ── Bottom-right: module profile radar / bar ─────────────────────────────────
ax_pr = fig.add_subplot(gs[1, 2])
x = np.arange(len(conditions))
w = 0.14
for i, (mod, col) in enumerate(zip(range(5), MODULE_COLORS)):
    vals = np.abs(cluster_profiles[mod])
    ax_pr.bar(x + i * w - w * 2, vals, width=w, color=col, alpha=0.8,
              label=f"M{mod+1}", edgecolor="white", lw=0.5)
ax_pr.set_xticks(x)
ax_pr.set_xticklabels([LABELS[c] for c in conditions], rotation=20, ha="right", fontsize=9.5)
ax_pr.set_ylabel("|Mean z-effect|", fontsize=10)
ax_pr.set_title("Module Effect Profiles", fontsize=11, fontweight="bold")
ax_pr.legend(fontsize=8.5, frameon=False, ncol=5, loc="upper right",
             columnspacing=0.5, handlelength=1)
ax_pr.set_axisbelow(True)
ax_pr.yaxis.grid(True, alpha=0.3, lw=0.6)

plt.suptitle("Genetic Module Gene Mapping & Pathway Enrichment",
             fontsize=14, fontweight="bold", y=1.01)
plt.savefig("figures/fig_gene_modules.png")
plt.close()
print("  → Saved figures/fig_gene_modules.png")
print("\nDone.")
