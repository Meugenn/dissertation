"""
All-Pairs Mendelian Randomisation → Psychiatric Causal DAG
===========================================================
Runs IVW-MR for every ordered pair of 5 conditions using the
shared-SNP effect matrix, then infers a causal directed graph.

Outputs:
  data_cache/mr_all_pairs.csv         — full MR results table
  data_cache/mr_causal_matrix.csv     — 5×5 β matrix
  figures/fig25_mr_heatmap.png        — MR estimate matrix + significance
  figures/fig26_causal_dag.png        — static DAG figure
  dag_interactive.html                — standalone interactive DAG (vis.js)
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy import stats
warnings.filterwarnings("ignore")
os.makedirs("figures", exist_ok=True)
os.makedirs("data_cache", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Palatino","Georgia","Times New Roman"],
    "font.size": 11, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})

CACHE = os.path.expanduser("~/.cache/huggingface/hub")
COND_COLORS = {
    "adhd": "#4C9BE8", "autism": "#2A9D8F", "bipolar": "#8B5CF6",
    "schizophrenia": "#E63946", "substance_use": "#264653",
}
LABELS = {
    "adhd": "ADHD", "autism": "Autism", "bipolar": "Bipolar",
    "schizophrenia": "SCZ", "substance_use": "SUD",
}

# ══════════════════════════════════════════════════════════════════════════════
# 1.  BUILD 5-CONDITION MERGED EFFECT MATRIX
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("1.  BUILDING 5-CONDITION MERGED EFFECT MATRIX")
print("=" * 70)

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
SE_COLS   = {"se","se_beta","stderr","std_err","stderror"}
P_COLS    = {"p","pval","p.value","p-value","pvalue","p_value"}

def find_col(df, aliases):
    low = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in low: return low[a]
    return None

def load_cond(name, pattern, max_shards=25):
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
            out["chr"] = pd.to_numeric(df[chr_], errors="coerce")
            out["bp"]  = pd.to_numeric(df[bp_],  errors="coerce")
            beta = find_col(df, BETA_COLS); or_ = find_col(df, OR_COLS)
            se_  = find_col(df, SE_COLS);   p_  = find_col(df, P_COLS)
            if beta:
                out["effect"] = pd.to_numeric(df[beta], errors="coerce")
            elif or_:
                raw = pd.to_numeric(df[or_], errors="coerce")
                out["effect"] = np.log(raw.where((raw > 0) & (raw < 100)))
            else: continue
            if se_: out["se"] = pd.to_numeric(df[se_], errors="coerce")
            if p_:  out["p"]  = pd.to_numeric(df[p_],  errors="coerce")
            out = out.dropna(subset=["chr","bp","effect"])
            out["chr"] = out["chr"].astype(int)
            out["pos_key"] = out["chr"].astype(str) + "_" + (out["bp"] // 10_000).astype(str)
            frames.append(out)
        except: continue
    if not frames: return None
    df_all = pd.concat(frames, ignore_index=True)
    if "p" in df_all.columns:
        df_all = df_all.sort_values("p").drop_duplicates("pos_key", keep="first")
    else:
        df_all = df_all.drop_duplicates("pos_key", keep="first")
    keep = ["chr", "bp", "effect"] + ([c for c in ["p","se"] if c in df_all.columns])
    result = df_all.set_index("pos_key")[keep]
    print(f"  [{name}] {len(result):,} bins, {result['chr'].nunique()} chrs")
    return result

cond_data = {}
for name, pat in PATTERNS.items():
    r = load_cond(name, pat)
    if r is not None: cond_data[name] = r

conditions = sorted(cond_data.keys())
print(f"\nLoaded: {conditions}")

# Merge: keep bins present in ≥3 conditions
effect_series = {n: cond_data[n]["effect"].rename(n) for n in conditions}
p_series      = {n: cond_data[n]["p"].rename(f"p_{n}")
                 for n in conditions if "p" in cond_data[n].columns}

merged_eff = pd.concat(effect_series.values(), axis=1, join="outer")
merged_p   = pd.concat(p_series.values(),      axis=1, join="outer")

min_cond = 3
shared = merged_eff[merged_eff.notna().sum(axis=1) >= min_cond].copy()
shared_p = merged_p.loc[shared.index]
print(f"\nShared bins (≥{min_cond} conditions): {len(shared):,}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ALL-PAIRS MENDELIAN RANDOMISATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("2.  ALL-PAIRS MR")
print("=" * 70)

def se_from_p(effect, p, min_z=0.1):
    z = np.abs(stats.norm.ppf(np.clip(p, 1e-300, 1) / 2))
    return np.abs(effect) / np.maximum(z, min_z)

def ivw_mr(beta_exp, beta_out, se_out, min_instruments=3):
    """IVW fixed-effect MR. Returns (beta, se, z, p, n_instruments)."""
    mask = np.isfinite(beta_exp) & np.isfinite(beta_out) & np.isfinite(se_out)
    mask &= (beta_exp != 0) & (se_out > 0)
    if mask.sum() < min_instruments:
        return np.nan, np.nan, np.nan, np.nan, int(mask.sum())
    bx, by, se = beta_exp[mask], beta_out[mask], se_out[mask]
    w  = 1 / se**2
    beta_ivw = (by / bx * w).sum() / w.sum()
    se_ivw   = np.sqrt(1 / w.sum())
    z = beta_ivw / se_ivw
    p = 2 * stats.norm.sf(abs(z))
    return beta_ivw, se_ivw, z, p, int(mask.sum())

P_INSTRUMENT = 5e-6   # suggestive threshold for instrument selection

results = []
beta_mat = pd.DataFrame(np.nan, index=conditions, columns=conditions)
p_mat    = pd.DataFrame(np.nan, index=conditions, columns=conditions)
se_mat   = pd.DataFrame(np.nan, index=conditions, columns=conditions)

for exp in conditions:
    # Instruments: top SNPs for exposure with low p-value
    if f"p_{exp}" not in shared_p.columns:
        print(f"  [{exp}→*] no p-values — skip")
        continue

    p_exp = shared_p[f"p_{exp}"]
    inst_mask = p_exp < P_INSTRUMENT
    n_inst = inst_mask.sum()

    if n_inst < 3:
        # Relax threshold
        P_INSTRUMENT_RELAX = 1e-4
        inst_mask = p_exp < P_INSTRUMENT_RELAX
        n_inst = inst_mask.sum()
        thresh_used = P_INSTRUMENT_RELAX
    else:
        thresh_used = P_INSTRUMENT

    instruments = shared[inst_mask].copy()
    inst_p_df   = shared_p[inst_mask].copy()

    print(f"\n  Exposure: {LABELS[exp]}  ({n_inst} instruments, p<{thresh_used:.0e})")

    beta_exp_vals = instruments[exp].values

    for out in conditions:
        if out == exp: continue
        beta_out_vals = instruments[out].values

        # SE for outcome: from p-values if available
        out_p_col = f"p_{out}"
        if out_p_col in inst_p_df.columns:
            se_out_vals = se_from_p(beta_out_vals, inst_p_df[out_p_col].values)
        else:
            se_out_vals = np.full(len(beta_out_vals), np.nan)

        beta, se, z, p, n = ivw_mr(beta_exp_vals, beta_out_vals, se_out_vals)

        results.append({
            "exposure": exp, "outcome": out,
            "beta_ivw": beta, "se_ivw": se, "z": z, "p": p,
            "n_instruments": n, "p_threshold": thresh_used,
        })
        beta_mat.loc[exp, out] = beta
        p_mat.loc[exp, out]    = p
        se_mat.loc[exp, out]   = se

        sig = "***" if (p is not None and not np.isnan(p) and p < 0.001) else \
              "**"  if (p is not None and not np.isnan(p) and p < 0.01)  else \
              "*"   if (p is not None and not np.isnan(p) and p < 0.05)  else ""
        print(f"    → {LABELS[out]:<12}  β={beta:+.3f}  p={p:.2e}  n={n}  {sig}"
              if not np.isnan(beta) else f"    → {LABELS[out]:<12}  insufficient instruments")

mr_df = pd.DataFrame(results)
mr_df.to_csv("data_cache/mr_all_pairs.csv", index=False)
beta_mat.to_csv("data_cache/mr_causal_matrix.csv")
print("\n  → Saved data_cache/mr_all_pairs.csv")

# ── Determine most likely causal direction per pair ───────────────────────────
print("\n  Causal direction summary (significant at p<0.05):")
sig_edges = []
for _, row in mr_df.iterrows():
    if np.isnan(row["p"]) or row["p"] >= 0.05: continue
    # Check reverse direction
    rev = mr_df[(mr_df["exposure"] == row["outcome"]) & (mr_df["outcome"] == row["exposure"])]
    if not rev.empty and not np.isnan(rev.iloc[0]["p"]):
        rev_p = rev.iloc[0]["p"]
        rev_b = rev.iloc[0]["beta_ivw"]
    else:
        rev_p, rev_b = 1.0, np.nan

    # Steiger-style: if A→B significant but B→A is not (or weaker), A→B is the direction
    direction = "→" if row["p"] < rev_p else "←"
    sig_edges.append({
        "from": row["exposure"], "to": row["outcome"],
        "beta": row["beta_ivw"], "p": row["p"],
        "direction": direction,
        "bidirectional": row["p"] < 0.05 and rev_p < 0.05,
    })
    print(f"    {LABELS[row['exposure']]} {direction} {LABELS[row['outcome']]}"
          f"  β={row['beta_ivw']:+.3f}  p={row['p']:.2e}"
          + (" [BIDIRECTIONAL]" if row["p"] < 0.05 and rev_p < 0.05 else ""))

sig_edges_df = pd.DataFrame(sig_edges) if sig_edges else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  STATIC FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.  GENERATING FIGURES")
print("=" * 70)

cond_lbls = [LABELS[c] for c in conditions]
n = len(conditions)

# ── Fig 25: MR matrix heatmap ─────────────────────────────────────────────────
print("Fig 25: MR matrix heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

beta_vals = beta_mat.values.astype(float)
p_vals    = p_mat.values.astype(float)

# Left: beta heatmap
ax = axes[0]
vmax = np.nanpercentile(np.abs(beta_vals), 95) * 1.5
im = ax.imshow(beta_vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(cond_lbls, rotation=30, ha="right", fontsize=10)
ax.set_yticklabels(cond_lbls, fontsize=10)
ax.set_xlabel("Outcome", fontsize=11); ax.set_ylabel("Exposure", fontsize=11)
ax.set_title("IVW-MR Causal Effect Estimates\n(β: SD change in outcome per SD of exposure)",
             fontsize=11, fontweight="bold")
for i in range(n):
    for j in range(n):
        if i == j:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color="#eee", zorder=0))
            continue
        bv = beta_vals[i, j]
        pv = p_vals[i, j]
        if np.isnan(bv): continue
        txt_col = "white" if abs(bv) > vmax * 0.5 else "#333"
        stars = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        ax.text(j, i, f"{bv:+.2f}{stars}", ha="center", va="center",
                fontsize=8, color=txt_col, fontweight="bold" if stars else "normal")
plt.colorbar(im, ax=ax, shrink=0.85, label="β (IVW-MR estimate)")

# Right: -log10(p) significance
ax2 = axes[1]
logp = -np.log10(np.where(np.isnan(p_vals), 1, np.clip(p_vals, 1e-10, 1)))
np.fill_diagonal(logp, 0)
im2 = ax2.imshow(logp, cmap="YlOrRd", vmin=0, vmax=max(logp.max(), 2), aspect="auto")
ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
ax2.set_xticklabels(cond_lbls, rotation=30, ha="right", fontsize=10)
ax2.set_yticklabels(cond_lbls, fontsize=10)
ax2.set_xlabel("Outcome", fontsize=11); ax2.set_ylabel("Exposure", fontsize=11)
ax2.set_title(r"Significance $(-\log_{10}P)$" + "\n(row = exposure, col = outcome)",
              fontsize=11, fontweight="bold")
for i in range(n):
    for j in range(n):
        if i == j:
            ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color="#eee", zorder=0))
            continue
        pv = p_vals[i, j]
        if np.isnan(pv): continue
        stars = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        lv = logp[i, j]
        txt_col = "white" if lv > logp.max() * 0.6 else "#333"
        ax2.text(j, i, f"{lv:.1f}{stars}", ha="center", va="center",
                 fontsize=8.5, color=txt_col)
plt.colorbar(im2, ax=ax2, shrink=0.85, label=r"$-\log_{10}(P)$")

plt.tight_layout()
plt.savefig("figures/fig25_mr_heatmap.png")
plt.close()
print("  → saved fig25")

# ── Fig 26: static causal DAG ─────────────────────────────────────────────────
print("Fig 26: static causal DAG...")

# Circular layout
n_c = len(conditions)
angles = np.linspace(0, 2 * np.pi, n_c, endpoint=False) - np.pi / 2
node_pos = {c: (np.cos(a) * 0.75, np.sin(a) * 0.75) for c, a in zip(conditions, angles)}

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

for ax_i, (p_thresh, thresh_label) in enumerate([(0.05, "p < 0.05"), (0.01, "p < 0.01")]):
    ax = axes[ax_i]
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Causal DAG — MR-IVW ({thresh_label})\n"
                 f"Arrow thickness ∝ |β|, colour: positive=red, negative=blue",
                 fontsize=11, fontweight="bold")

    # Draw edges
    drawn_pairs = set()
    for _, row in mr_df.iterrows():
        exp, out = row["exposure"], row["outcome"]
        if np.isnan(row["p"]) or row["p"] >= p_thresh: continue
        pair_key = tuple(sorted([exp, out]))
        bidir = pair_key in drawn_pairs
        drawn_pairs.add(pair_key)

        x0, y0 = node_pos[exp]
        x1, y1 = node_pos[out]
        beta = row["beta_ivw"]
        lw   = max(1.0, min(abs(beta) * 8, 6))
        col  = "#E63946" if beta > 0 else "#4C9BE8"
        alpha = 0.85

        # Curve the arrow slightly to separate bidirectional
        dx, dy = x1 - x0, y1 - y0
        perp   = np.array([-dy, dx]) * (0.12 if bidir else 0)
        mx, my = (x0 + x1) / 2 + perp[0], (y0 + y1) / 2 + perp[1]

        # Bezier-like: draw as annotation with connectionstyle
        ax.annotate(
            "", xy=(x1 * 0.78, y1 * 0.78), xytext=(x0 * 0.78, y0 * 0.78),
            arrowprops=dict(
                arrowstyle=f"-|>,head_width={lw*0.06},head_length={lw*0.05}",
                color=col, lw=lw, alpha=alpha,
                connectionstyle="arc3,rad=0.15" if bidir else "arc3,rad=0.05",
            )
        )
        # Label edge with β
        lx, ly = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(lx + perp[0] * 0.5, ly + perp[1] * 0.5,
                f"{beta:+.2f}", fontsize=7.5, ha="center", va="center",
                color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    # Draw nodes
    for c in conditions:
        x, y = node_pos[c]
        circle = plt.Circle((x, y), 0.13, color=COND_COLORS[c],
                              zorder=5, ec="white", lw=2)
        ax.add_patch(circle)
        ax.text(x, y, LABELS[c], ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white", zorder=6)

    # Legend
    pos_patch = mpatches.Patch(color="#E63946", label="Positive effect (risk-increasing)")
    neg_patch = mpatches.Patch(color="#4C9BE8", label="Negative effect (protective)")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8.5, frameon=False,
              loc="lower right", bbox_to_anchor=(1.3, -0.05))

plt.tight_layout()
plt.savefig("figures/fig26_causal_dag.png")
plt.close()
print("  → saved fig26")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  INTERACTIVE HTML DAG  (vis.js network)
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating interactive DAG HTML...")

# Build nodes + edges JSON for vis.js
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

nodes_js = []
for c in conditions:
    col = COND_COLORS[c]
    r, g, b = hex_to_rgb(col)
    nodes_js.append({
        "id":    c,
        "label": LABELS[c],
        "color": {
            "background": col,
            "border": "#fff",
            "highlight": {"background": col, "border": "#333"},
        },
        "font":  {"color": "#fff", "size": 16, "face": "Georgia"},
        "shape": "ellipse",
        "size":  40,
        "title": f"<b>{LABELS[c]}</b><br>Click to highlight edges",
    })

edges_js = []
edge_id  = 0
for _, row in mr_df.iterrows():
    if np.isnan(row["p"]): continue
    beta, p_val = row["beta_ivw"], row["p"]
    col  = "#E63946" if beta > 0 else "#4C9BE8"
    alpha_hex = "dd" if p_val < 0.05 else "55"
    width = max(1, min(abs(beta) * 10, 8))
    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    title = (f"<b>{LABELS[row['exposure']]} → {LABELS[row['outcome']]}</b><br>"
             f"β = {beta:+.3f}<br>"
             f"SE = {row['se_ivw']:.3f}<br>"
             f"p = {p_val:.2e} ({stars})<br>"
             f"N instruments = {int(row['n_instruments'])}")
    edges_js.append({
        "id":     edge_id,
        "from":   row["exposure"],
        "to":     row["outcome"],
        "arrows": "to",
        "color":  {"color": col + alpha_hex, "highlight": col},
        "width":  width,
        "title":  title,
        "label":  f"{beta:+.2f}" if p_val < 0.05 else "",
        "font":   {"color": col, "size": 11, "face": "Georgia", "strokeWidth": 2, "strokeColor": "#fff"},
        "smooth": {"type": "curvedCW", "roundness": 0.2},
        "hidden": False,
        "p_value": p_val,
        "beta":    float(beta),
    })
    edge_id += 1

import json
nodes_json = json.dumps(nodes_js, indent=2)
edges_json = json.dumps(edges_js, indent=2)

# Compute MR table rows for the info panel
table_rows = ""
for _, row in mr_df.sort_values("p").iterrows():
    if np.isnan(row["p"]): continue
    p_val = row["p"]
    beta  = row["beta_ivw"]
    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    row_class = "sig" if p_val < 0.05 else ""
    table_rows += (
        f'<tr class="{row_class}">'
        f'<td>{LABELS[row["exposure"]]}</td>'
        f'<td>→</td>'
        f'<td>{LABELS[row["outcome"]]}</td>'
        f'<td style="color:{"#E63946" if beta>0 else "#4C9BE8"}">{beta:+.3f}</td>'
        f'<td>{row["se_ivw"]:.3f}</td>'
        f'<td>{p_val:.2e}{stars}</td>'
        f'<td>{int(row["n_instruments"])}</td>'
        f'</tr>\n'
    )

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Psychiatric Causal DAG — MR-IVW</title>
<script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
<link  href="https://unpkg.com/vis-network@9.1.9/dist/dist/vis-network.min.css" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Georgia, serif; background: #0f1117; color: #e0e0e0; }}

  header {{
    padding: 18px 28px; background: #1a1d27;
    border-bottom: 1px solid #2e3146;
    display: flex; align-items: baseline; gap: 16px;
  }}
  header h1 {{ font-size: 1.25rem; color: #fff; font-weight: bold; }}
  header p  {{ font-size: 0.85rem; color: #8b8fa8; }}

  .layout {{ display: grid; grid-template-columns: 1fr 360px; height: calc(100vh - 62px); }}

  #network {{ width: 100%; height: 100%; background: #13161f;
              border-right: 1px solid #2e3146; }}

  .sidebar {{ overflow-y: auto; padding: 16px; background: #1a1d27; display: flex; flex-direction: column; gap: 14px; }}

  .card {{ background: #22263a; border-radius: 8px; padding: 14px; border: 1px solid #2e3146; }}
  .card h3 {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em;
               color: #8b8fa8; margin-bottom: 10px; }}

  .controls label {{ font-size: 0.82rem; color: #b0b4c8; display: block; margin-bottom: 4px; }}
  .controls input[type=range] {{ width: 100%; accent-color: #7c6af5; }}
  .controls input[type=checkbox] {{ accent-color: #7c6af5; margin-right: 6px; }}
  .controls .row {{ display: flex; align-items: center; margin-bottom: 8px; }}

  .legend-item {{ display: flex; align-items: center; gap: 8px;
                   font-size: 0.82rem; color: #b0b4c8; margin-bottom: 6px; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
  .legend-line {{ width: 28px; height: 3px; border-radius: 2px; flex-shrink: 0; }}

  #tooltip {{ font-size: 0.83rem; color: #c8cce0; line-height: 1.55; min-height: 60px; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
  th {{ text-align: left; color: #8b8fa8; font-weight: normal;
        padding: 4px 6px; border-bottom: 1px solid #2e3146; }}
  td {{ padding: 4px 6px; border-bottom: 1px solid #1f2235; color: #c8cce0; }}
  tr.sig td {{ color: #fff; }}
  tr:hover td {{ background: #2a2f4a; }}

  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
             font-size: 0.72rem; font-weight: bold; }}
  .badge-sig {{ background: #E6394622; color: #E63946; border: 1px solid #E6394666; }}
  .badge-ns  {{ background: #44475a; color: #8b8fa8; border: 1px solid #44475a; }}

  button {{ background: #7c6af5; color: #fff; border: none; border-radius: 6px;
             padding: 6px 14px; cursor: pointer; font-size: 0.82rem; font-family: Georgia, serif; }}
  button:hover {{ background: #6557d4; }}
  button.secondary {{ background: #2a2f4a; color: #b0b4c8; }}
  button.secondary:hover {{ background: #363c5a; }}
  .btn-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
</style>
</head>
<body>

<header>
  <h1>Psychiatric Causal DAG — Mendelian Randomisation (IVW)</h1>
  <p>5 conditions · all-pairs MR · hover edges for details · use controls to filter</p>
</header>

<div class="layout">
  <div id="network"></div>

  <div class="sidebar">

    <!-- Tooltip / hover info -->
    <div class="card">
      <h3>Hover Info</h3>
      <div id="tooltip">Hover over a node or edge to see details.</div>
    </div>

    <!-- Controls -->
    <div class="card controls">
      <h3>Controls</h3>

      <label>p-value threshold
        <span id="p-label" style="float:right;color:#fff">0.05</span>
      </label>
      <input type="range" id="p-slider" min="0" max="3" step="0.5" value="1.3"
             oninput="updatePThreshold(this.value)">
      <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#8b8fa8;margin-bottom:10px">
        <span>0.001</span><span>0.01</span><span>0.05</span><span>0.1</span><span>ns</span>
      </div>

      <div class="row">
        <input type="checkbox" id="show-ns" onchange="updatePThreshold(document.getElementById('p-slider').value)">
        <label style="margin:0">Show non-significant edges (dim)</label>
      </div>

      <div class="row">
        <input type="checkbox" id="physics" checked onchange="togglePhysics(this.checked)">
        <label style="margin:0">Physics simulation (drag to pin)</label>
      </div>

      <div class="btn-row" style="margin-top:8px">
        <button onclick="network.fit()">Fit view</button>
        <button class="secondary" onclick="resetLayout()">Reset layout</button>
      </div>
    </div>

    <!-- Legend -->
    <div class="card">
      <h3>Legend</h3>
      <div class="legend-item">
        <div class="legend-line" style="background:#E63946"></div>
        Positive causal effect (risk-increasing)
      </div>
      <div class="legend-item">
        <div class="legend-line" style="background:#4C9BE8"></div>
        Negative causal effect (protective)
      </div>
      <div class="legend-item">
        <div class="legend-line" style="background:#555; border: 1px dashed #666;"></div>
        Non-significant (dimmed)
      </div>
      <div style="height:8px"></div>
      {''.join([f'<div class="legend-item"><div class="legend-dot" style="background:{COND_COLORS[c]}"></div>{LABELS[c]}</div>' for c in conditions])}
    </div>

    <!-- MR table -->
    <div class="card">
      <h3>MR Results Table</h3>
      <table>
        <thead>
          <tr>
            <th>Exp</th><th></th><th>Out</th>
            <th>β</th><th>SE</th><th>p</th><th>N</th>
          </tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
      <div style="margin-top:8px;font-size:0.72rem;color:#8b8fa8">
        *** p&lt;0.001 · ** p&lt;0.01 · * p&lt;0.05 · Bold rows = significant<br>
        β: SD change in outcome per SD of exposure (IVW-MR)
      </div>
    </div>

  </div><!-- sidebar -->
</div><!-- layout -->

<script>
const ALL_NODES = {nodes_json};
const ALL_EDGES = {edges_json};

const nodesDS = new vis.DataSet(ALL_NODES);
const edgesDS = new vis.DataSet(ALL_EDGES);

const container = document.getElementById("network");
const options = {{
  nodes: {{ borderWidth: 2, shadow: {{ enabled: true, size: 8, color: "#00000066" }} }},
  edges: {{
    arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }},
    smooth: {{ type: "curvedCW", roundness: 0.2 }},
    shadow: false,
  }},
  physics: {{
    enabled: true,
    barnesHut: {{ gravitationalConstant: -4000, centralGravity: 0.3,
                  springLength: 200, springConstant: 0.04, damping: 0.3 }},
    stabilization: {{ iterations: 300 }},
  }},
  interaction: {{
    hover: true, tooltipDelay: 100,
    navigationButtons: true, keyboard: true,
  }},
}};

const network = new vis.Network(container, {{ nodes: nodesDS, edges: edgesDS }}, options);

// Hover handlers
network.on("hoverNode", params => {{
  const node = ALL_NODES.find(n => n.id === params.node);
  document.getElementById("tooltip").innerHTML =
    `<b style="color:#fff">${{node.label}}</b><br>
     Click to highlight all connected edges.`;
}});

network.on("hoverEdge", params => {{
  const edge = ALL_EDGES.find(e => e.id === params.edge);
  if (!edge) return;
  const fromN = ALL_NODES.find(n => n.id === edge.from);
  const toN   = ALL_NODES.find(n => n.id === edge.to);
  document.getElementById("tooltip").innerHTML = edge.title;
}});

network.on("blurNode",  () => {{ document.getElementById("tooltip").textContent = "Hover over a node or edge to see details."; }});
network.on("blurEdge",  () => {{ document.getElementById("tooltip").textContent = "Hover over a node or edge to see details."; }});

// Click node → highlight its edges
network.on("click", params => {{
  if (params.nodes.length > 0) {{
    const nodeId = params.nodes[0];
    const connectedEdges = network.getConnectedEdges(nodeId);
    edgesDS.update(ALL_EDGES.map(e => ({{
      id: e.id,
      width: connectedEdges.includes(e.id) ? Math.max(e.width, 3) : e.width,
      color: connectedEdges.includes(e.id) ? e.color : {{ color: e.color.color + "33", highlight: e.color.color }},
    }})));
  }} else {{
    // reset
    updatePThreshold(document.getElementById("p-slider").value);
  }}
}});

// p-value threshold filter
const P_VALUES = [0.001, 0.01, 0.05, 0.1, 1.0];
function updatePThreshold(sliderVal) {{
  const idx    = Math.round(parseFloat(sliderVal) * 2);
  const pThresh = P_VALUES[Math.min(idx, P_VALUES.length - 1)];
  document.getElementById("p-label").textContent =
    pThresh < 1 ? `${{pThresh}}` : "ns (all)";
  const showNS = document.getElementById("show-ns").checked;

  edgesDS.update(ALL_EDGES.map(e => {{
    const sig = e.p_value <= pThresh;
    return {{
      id:     e.id,
      hidden: !sig && !showNS,
      color:  sig
        ? e.color
        : {{ color: "#55555544", highlight: "#aaa" }},
      width:  sig ? Math.max(1, Math.abs(e.beta) * 10) : 1,
      label:  sig ? (e.beta >= 0 ? "+" : "") + e.beta.toFixed(2) : "",
    }};
  }}));
}}

function togglePhysics(on) {{
  network.setOptions({{ physics: {{ enabled: on }} }});
}}

function resetLayout() {{
  network.setOptions({{ physics: {{ enabled: true }} }});
  setTimeout(() => {{
    if (!document.getElementById("physics").checked)
      network.setOptions({{ physics: {{ enabled: false }} }});
  }}, 2000);
  network.fit();
}}

// Init: show p<0.05
updatePThreshold(document.getElementById("p-slider").value);
</script>
</body>
</html>
"""

with open("dag_interactive.html", "w") as f:
    f.write(html)
print("  → Saved dag_interactive.html")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
n_sig   = mr_df[mr_df["p"] < 0.05].shape[0]
n_total = mr_df.dropna(subset=["p"]).shape[0]
print(f"  Estimated {n_total} directed MR pairs; {n_sig} significant at p<0.05")
if not sig_edges_df.empty:
    print("\n  Significant causal edges:")
    for _, r in sig_edges_df.iterrows():
        print(f"    {LABELS[r['from']]} → {LABELS[r['to']]}  β={r['beta']:+.3f}  p={r['p']:.2e}")
print(f"\n  Figures: fig25_mr_heatmap.png, fig26_causal_dag.png")
print(f"  Interactive: dag_interactive.html  (open in browser)")
