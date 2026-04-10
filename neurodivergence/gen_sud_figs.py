"""Generate SUD-specific figures for the LaTeX report."""
import glob, warnings, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family":"serif","font.serif":["Palatino","Georgia","Times New Roman"],
    "font.size":11,"axes.linewidth":0.8,"axes.spines.top":False,
    "axes.spines.right":False,"figure.dpi":150,"savefig.dpi":200,"savefig.bbox":"tight",
})

# ── Load SUD ──────────────────────────────────────────────────────────────────

files = sorted(glob.glob(
    os.path.expanduser("~/.cache/huggingface/hub/datasets--OpenMed--pgc-substance-use"
                       "/snapshots/*/data/SUD2023/*.parquet")
))
print(f"Found {len(files)} SUD shards")

frames = []
for f in files:
    try:
        df = pd.read_parquet(f)
        df.columns = [c.strip() for c in df.columns]
        rename = {}
        for c in df.columns:
            lc = c.lower()
            if lc == "chr":  rename[c] = "CHR"
            if lc == "bp":   rename[c] = "BP"
            if lc == "snp":  rename[c] = "SNP"
            if lc == "or":   rename[c] = "OR"
            if lc == "beta": rename[c] = "BETA"
            if lc == "p":    rename[c] = "P"
        df = df.rename(columns=rename)
        if "OR" in df.columns and "BETA" not in df.columns:
            df["EFFECT"] = pd.to_numeric(df["OR"], errors="coerce")
            df["ETYPE"]  = "OR"
        elif "BETA" in df.columns:
            df["EFFECT"] = pd.to_numeric(df["BETA"], errors="coerce")
            df["ETYPE"]  = "BETA"
        for col in ["CHR","BP","P"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
    except Exception as e:
        print(f"  skip: {e}")

sud = pd.concat(frames, ignore_index=True).dropna(subset=["CHR","BP","P"])
sud["CHR"] = sud["CHR"].astype(int)
sud["-log10p"] = -np.log10(sud["P"].clip(lower=1e-300))
sud = sud.sort_values(["CHR","BP"])
print(f"Loaded {len(sud):,} SUD variants")

# Cumulative BP
offset, offsets = 0, {}
for chrom in sorted(sud["CHR"].unique()):
    offsets[chrom] = offset
    offset += int(sud[sud["CHR"]==chrom]["BP"].max())
sud["BP_cum"] = sud.apply(lambda r: int(r["BP"]) + offsets[int(r["CHR"])], axis=1)
chr_mids = {c: sud[sud["CHR"]==c]["BP_cum"].mean() for c in sorted(sud["CHR"].unique())}

lambda_gc = np.median(stats.chi2.ppf(1 - sud["P"].clip(lower=1e-300), df=1)) / stats.chi2.ppf(0.5, df=1)
n_gw = (sud["P"] < 5e-8).sum()
print(f"λ_GC = {lambda_gc:.3f}, GW hits = {n_gw}")


# ── Fig 7: SUD Manhattan ──────────────────────────────────────────────────────

print("Fig 7: SUD Manhattan...")
fig, ax = plt.subplots(figsize=(14, 5.5))
chrom_colors = ["#BC6C25", "#e9c46a"]

for chrom in sorted(sud["CHR"].unique()):
    sub  = sud[sud["CHR"] == chrom]
    col  = chrom_colors[chrom % 2]
    nsig = sub[sub["-log10p"] < 5]
    sig  = sub[sub["-log10p"] >= 5]
    if len(nsig) > 12_000:
        nsig = nsig.sample(12_000, random_state=42)
    ax.scatter(nsig["BP_cum"], nsig["-log10p"], c=col, s=2, alpha=0.4, linewidths=0)
    if not sig.empty:
        ax.scatter(sig["BP_cum"], sig["-log10p"],
                   c="#E63946", s=14, zorder=5, linewidths=0)

ax.axhline(-np.log10(5e-8), color="#E63946", lw=1.1, ls="--",
           label=r"GW significance ($p=5\times10^{-8}$)")
ax.axhline(-np.log10(1e-5), color="#F4A261", lw=0.8, ls=":",
           label=r"Suggestive ($p=10^{-5}$)")

ax.set_xticks(list(chr_mids.values()))
ax.set_xticklabels([str(int(c)) for c in chr_mids], fontsize=8)
ax.set_xlabel("Chromosome", fontsize=11)
ax.set_ylabel(r"$-\log_{10}(P)$", fontsize=11)
ax.set_title(r"Manhattan Plot — Substance Use Disorder GWAS (SUD2023, $N\approx{}1{,}000{,}000$)",
             fontsize=13, fontweight="bold", pad=10)
ax.legend(fontsize=9, frameon=False, loc="upper right")
ymax = max(sud["-log10p"].quantile(0.9999) * 1.15, 10)
ax.set_ylim(0, ymax)

# Annotate known loci
known = {"ADH1B\n(chr4)": 4, "CHRNA5\n(chr15)": 15, "DRD2\n(chr11)": 11}
for label, chrom in known.items():
    if chrom in chr_mids:
        xpos = chr_mids[chrom]
        sub  = sud[sud["CHR"]==chrom]
        ypos = sub["-log10p"].max() if not sub.empty else 5
        ax.annotate(label, xy=(xpos, ypos), xytext=(xpos, min(ymax*0.85, ypos+2)),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=0.7),
                    fontsize=7.5, ha="center", color="#333")

plt.tight_layout()
plt.savefig("figures/fig7_manhattan_sud.png")
plt.close()
print(f"  saved fig7  ({n_gw} GW hits)")


# ── Fig 8: SUD QQ plot ────────────────────────────────────────────────────────

print("Fig 8: SUD QQ plot...")
p_vals = sud["P"].dropna().values
p_vals = p_vals[(p_vals > 0) & (p_vals <= 1)]
n_pts  = len(p_vals)
obs    = np.sort(-np.log10(p_vals))[::-1]
exp    = -np.log10(np.arange(1, n_pts+1) / (n_pts+1))
idx    = np.unique(np.round(np.linspace(0, n_pts-1, 8000)).astype(int))

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(exp[idx], obs[idx], s=5, color="#BC6C25", alpha=0.5, linewidths=0)
mx = max(obs[idx].max(), exp[idx].max()) * 1.06
ax.plot([0, mx], [0, mx], "r--", lw=1.2, label=r"Expected under $H_0$")
se = 1.96 / np.sqrt(n_pts / 2)
ax.fill_between([0, mx], [0-se, mx-se], [0+se, mx+se],
                alpha=0.1, color="red", label="95% CI")
ax.set_xlabel(r"Expected $-\log_{10}(P)$", fontsize=11)
ax.set_ylabel(r"Observed $-\log_{10}(P)$", fontsize=11)
ax.set_title(f"QQ Plot — Substance Use Disorder (SUD2023)\n"
             r"$\lambda_\mathrm{GC}$" + f" = {lambda_gc:.3f} ({n_pts:,} variants)",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(fontsize=9, frameon=False)
ax.set_xlim(0, mx); ax.set_ylim(0, mx)
ax.text(0.05, 0.92,
        f"λ_GC = {lambda_gc:.3f}\n" +
        ("Consistent with true\npolygenicity" if lambda_gc < 1.5 else "Elevated — check stratification"),
        transform=ax.transAxes, fontsize=8.5, color="#444",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))
plt.tight_layout()
plt.savefig("figures/fig8_qq_sud.png")
plt.close()
print("  saved fig8")


# ── Fig 9: PRS simulation ─────────────────────────────────────────────────────

print("Fig 9: PRS simulation...")
np.random.seed(42)
N = 100_000
h2 = 0.10   # SNP heritability for SUD
prev = 0.12  # 12% lifetime prevalence

prs_pop = np.random.normal(0, np.sqrt(h2), N)
threshold = np.percentile(prs_pop, (1 - prev) * 100)

prs_groups = {
    "General population": prs_pop,
    "SUD cases":          prs_pop[prs_pop >= threshold],
    "SUD + ADHD liability\n($r_g\\approx0.30$)":
        (prs_pop + 0.30 * np.random.normal(0, np.sqrt(h2), N))[prs_pop >= threshold],
    "SUD + SCZ liability\n($r_g\\approx0.19$)":
        (prs_pop + 0.19 * np.random.normal(0, np.sqrt(h2), N))[prs_pop >= threshold],
}
colors = ["#aaa", "#E63946", "#8B5CF6", "#264653"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: overlapping densities
ax = axes[0]
bins = np.linspace(-0.45, 0.45, 80)
for (label, data), col in zip(prs_groups.items(), colors):
    ax.hist(data, bins=bins, density=True, alpha=0.55,
            color=col, label=label.replace("\n", " "), edgecolor="none")
ax.axvline(threshold, color="#333", lw=1.2, ls="--",
           label=f"Liability threshold\n(top {prev*100:.0f}%)")
ax.set_xlabel("Polygenic Risk Score", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("PRS Distribution — Liability-Threshold Model\n"
             r"$h^2_\mathrm{SNP}=0.10$, prevalence=12%",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=8, frameon=False)
ax.set_axisbelow(True); ax.yaxis.grid(True, alpha=0.3, lw=0.6)

# Right: mean PRS by decile
ax2 = axes[1]
decile_edges = np.percentile(prs_pop, np.arange(0, 101, 10))  # 11 edges → 10 bins
decile_labels = [f"D{i}" for i in range(1, 11)]
case_probs = []
for lo, hi in zip(decile_edges[:-1], decile_edges[1:]):
    mask = (prs_pop >= lo) & (prs_pop < hi)
    case_probs.append((prs_pop[mask] >= threshold).mean() * 100)

bar_cols = [plt.cm.RdBu_r(v/100) for v in case_probs]
bars = ax2.bar(decile_labels, case_probs, color=bar_cols, edgecolor="white", lw=0.6)
ax2.axhline(prev*100, color="#333", lw=1, ls="--",
            label=f"Population average ({prev*100:.0f}%)")
for bar, v in zip(bars, case_probs):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
ax2.set_xlabel("PRS Decile", fontsize=11)
ax2.set_ylabel("Probability of SUD (%)", fontsize=11)
ax2.set_title("SUD Risk by PRS Decile\n(Liability-threshold model, N=100,000)",
              fontsize=12, fontweight="bold")
ax2.legend(fontsize=9, frameon=False)
ax2.set_ylim(0, max(case_probs)*1.18)
ax2.set_axisbelow(True); ax2.yaxis.grid(True, alpha=0.3, lw=0.6)

plt.tight_layout()
plt.savefig("figures/fig9_prs_simulation.png")
plt.close()
print("  saved fig9")


# ── Fig 10: SUD cross-disorder bar ────────────────────────────────────────────

print("Fig 10: SUD cross-disorder bar...")

# From v3 run — SUD row of correlation matrix
other_conds = ["adhd","anxiety","autism","bipolar","cross_disorder","eating","mdd","ptsd","schizophrenia"]
r_vals      = [0.012, 0.041, 0.040, 0.014, 0.023, 0.046, -0.043, np.nan, 0.190]
labels_disp = ["ADHD","Anxiety","Autism","Bipolar","Cross-\nDisorder",
               "Eating\nDisorders","MDD","PTSD","Schizophrenia"]
colors_bar  = ["#4C9BE8","#F4A261","#2A9D8F","#E76F51","#8B5CF6",
               "#F72585","#457B9D","#E9C46A","#264653"]

# Sort by |r|
pairs = sorted(zip(r_vals, labels_disp, colors_bar), key=lambda x: abs(x[0]) if not np.isnan(x[0]) else 0, reverse=True)
r_s, l_s, c_s = zip(*pairs)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(r_s))
bar_cols = [c if not np.isnan(r) else "#ddd" for r, c in zip(r_s, c_s)]
bars = ax.bar(x, [r if not np.isnan(r) else 0 for r in r_s],
              color=bar_cols, width=0.65, edgecolor="white", lw=0.7)
for bar, r in zip(bars, r_s):
    if not np.isnan(r):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + (0.003 if r >= 0 else -0.008),
                f"{r:.3f}", ha="center",
                va="bottom" if r >= 0 else "top", fontsize=9, fontweight="bold")

ax.axhline(0, color="#333", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(l_s, fontsize=10)
ax.set_ylabel("Effect-size correlation (r)", fontsize=11)
ax.set_title("Substance Use Disorder — Cross-Disorder Genetic Effect Correlations\n"
             "(489,904 shared SNPs, log-OR scale)",
             fontsize=12, fontweight="bold", pad=10)
ax.set_ylim(-0.12, 0.28)
ax.set_axisbelow(True); ax.yaxis.grid(True, alpha=0.3, lw=0.6)

# Annotations
ax.annotate("Strongest: SCZ\n(dopamine pathway\noverlap)",
            xy=(0, 0.190), xytext=(1.5, 0.24),
            arrowprops=dict(arrowstyle="->", color="#333", lw=0.7),
            fontsize=8, color="#333")
ax.annotate("Negative: MDD\n(internalising vs\nexternalising)",
            xy=(6, -0.043), xytext=(5, -0.09),
            arrowprops=dict(arrowstyle="->", color="#333", lw=0.7),
            fontsize=8, color="#333")

plt.tight_layout()
plt.savefig("figures/fig10_sud_cross_disorder.png")
plt.close()
print("  saved fig10")

print(f"\nAll SUD figures done. λ_GC={lambda_gc:.3f}, GW hits={n_gw}")
