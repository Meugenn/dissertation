"""
Substance Use Disorder GWAS — Interactive Explorer
Streamlit app: Manhattan, QQ, cross-disorder PRS simulation, top loci
Run: streamlit run app.py
"""

import glob, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SUD GWAS Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  h1 { color: #264653; }
  h2 { color: #2a9d8f; border-bottom: 1px solid #eee; padding-bottom: 4px; }
  .metric-label { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading SUD GWAS data...")
def load_sud(max_shards: int = 20) -> pd.DataFrame:
    files = sorted(glob.glob(
        "/Users/meuge/.cache/huggingface/hub/datasets--OpenMed--pgc-substance-use"
        "/snapshots/*/data/SUD2023/*.parquet"
    ))[:max_shards]

    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # Normalise column names across the two schema variants
            df.columns = [c.strip() for c in df.columns]
            rename = {}
            for c in df.columns:
                lc = c.lower()
                if lc == "chr":    rename[c] = "CHR"
                if lc == "bp":     rename[c] = "BP"
                if lc == "snp":    rename[c] = "SNP"
                if lc == "or":     rename[c] = "OR"
                if lc == "beta":   rename[c] = "BETA"
                if lc == "p":      rename[c] = "P"
                if lc == "n":      rename[c] = "N"
                if lc == "a1":     rename[c] = "A1"
                if lc == "a2":     rename[c] = "A2"
            df = df.rename(columns=rename)
            # Unify effect column
            if "OR" in df.columns and "BETA" not in df.columns:
                df["EFFECT"]      = pd.to_numeric(df["OR"],   errors="coerce")
                df["EFFECT_TYPE"] = "OR"
            elif "BETA" in df.columns:
                df["EFFECT"]      = pd.to_numeric(df["BETA"], errors="coerce")
                df["EFFECT_TYPE"] = "BETA"
            for col in ["CHR","BP","P"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["CHR","BP","P"])
    df["CHR"] = df["CHR"].astype(int)
    df["-log10p"] = -np.log10(df["P"].clip(lower=1e-300))
    df = df.sort_values(["CHR","BP"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Loading SCZ data for comparison...")
def load_scz(max_shards: int = 10) -> pd.DataFrame:
    files = sorted(glob.glob(
        "/Users/meuge/.cache/huggingface/hub/datasets--OpenMed--pgc-schizophrenia"
        "/snapshots/*/data/scz2022/*.parquet"
    ))[:max_shards]
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            rename = {c: "FRQ_A" for c in df.columns if c.lower().startswith("frq_a_")}
            rename.update({c: "FRQ_U" for c in df.columns if c.lower().startswith("frq_u_")})
            df = df.rename(columns=rename)
            frames.append(df[["CHR","BP","SNP","OR","SE","P"]])
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["P"]   = pd.to_numeric(df["P"],   errors="coerce")
    df["OR"]  = pd.to_numeric(df["OR"],  errors="coerce")
    df["CHR"] = pd.to_numeric(df["CHR"], errors="coerce").astype("Int64")
    return df.dropna(subset=["CHR","BP","P"])


def cumulative_bp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    offset, offsets = 0, {}
    for chrom in sorted(df["CHR"].unique()):
        offsets[chrom] = offset
        offset += int(df[df["CHR"]==chrom]["BP"].max())
    df["BP_cum"] = df.apply(lambda r: int(r["BP"]) + offsets[int(r["CHR"])], axis=1)
    df["chr_mid"] = df["CHR"].map(
        lambda c: df[df["CHR"]==c]["BP_cum"].mean()
    )
    return df, offsets


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧬 SUD GWAS Explorer")
    st.caption("OpenMed · PGC Substance Use Disorder 2023")
    st.divider()

    n_shards = st.slider("Shards to load (×10k rows each)", 5, 20, 15)
    gw_thresh = st.select_slider(
        "GW significance threshold",
        options=[5e-8, 1e-7, 1e-6, 1e-5],
        value=5e-8,
        format_func=lambda x: f"p < {x:.0e}",
    )
    color_scheme = st.selectbox(
        "Manhattan colour scheme",
        ["Teal/Slate", "Blue/Orange", "Purple/Green"],
    )
    show_labels = st.toggle("Label top hits on Manhattan", value=True)
    st.divider()
    st.markdown("""
**About this data**
- 214M rows, SUD2023 sub-study
- 10 sub-conditions (alcohol, cannabis, opioid, tobacco, etc.)
- Ancestry: primarily European
- Effect: OR (alcohol/cannabis) or BETA (tobacco/opioid studies)
    """)

# ── Load data ─────────────────────────────────────────────────────────────────

df = load_sud(max_shards=n_shards)

if df.empty:
    st.error("Could not load SUD data from cache. Run explore_v3.py first.")
    st.stop()

df_cum, chr_offsets = cumulative_bp(df)

# ── Header metrics ────────────────────────────────────────────────────────────

st.title("Substance Use Disorder GWAS Explorer")
st.caption(f"SUD2023 · {len(df):,} variants loaded · {n_shards} shards")

sig_hits = df[df["P"] < gw_thresh]
lambda_gc = np.median(stats.chi2.ppf(1 - df["P"].clip(lower=1e-300), df=1)) / stats.chi2.ppf(0.5, df=1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Variants", f"{len(df):,}")
col2.metric("GW-significant hits", f"{len(sig_hits):,}", help=f"p < {gw_thresh:.0e}")
col3.metric("Min p-value", f"{df['P'].min():.2e}")
col4.metric("λ_GC", f"{lambda_gc:.3f}", help="Genomic inflation factor")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺 Manhattan Plot",
    "📈 QQ Plot",
    "🔬 Top Loci",
    "🔗 Cross-Disorder Comparison",
    "🎲 PRS Simulation",
    "🧬 ADHD × Bipolar × Autism",
])


# ── TAB 1: Manhattan ──────────────────────────────────────────────────────────

with tab1:
    st.subheader("Manhattan Plot — Substance Use Disorder (SUD2023)")

    color_map = {
        "Teal/Slate":     ["#2a9d8f", "#264653"],
        "Blue/Orange":    ["#4C9BE8", "#E76F51"],
        "Purple/Green":   ["#8B5CF6", "#2A9D8F"],
    }[color_scheme]

    # Build traces per chromosome
    chroms = sorted(df_cum["CHR"].unique())
    chr_mids = {c: df_cum[df_cum["CHR"]==c]["BP_cum"].mean() for c in chroms}

    fig = go.Figure()

    for i, chrom in enumerate(chroms):
        sub = df_cum[df_cum["CHR"] == chrom].copy()
        nsig = sub[sub["-log10p"] < -np.log10(gw_thresh)]
        sig  = sub[sub["-log10p"] >= -np.log10(gw_thresh)]

        # Thin non-sig points
        if len(nsig) > 8000:
            nsig = nsig.sample(8000, random_state=42)

        col = color_map[i % 2]
        hover = nsig.apply(
            lambda r: f"<b>{r.get('SNP','?')}</b><br>Chr{int(r['CHR'])}:{int(r['BP']):,}<br>p={r['P']:.2e}",
            axis=1
        )
        fig.add_trace(go.Scatter(
            x=nsig["BP_cum"], y=nsig["-log10p"],
            mode="markers",
            marker=dict(size=3, color=col, opacity=0.5),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False, name=f"Chr{chrom}",
        ))

        if not sig.empty:
            hover_sig = sig.apply(
                lambda r: f"<b>{r.get('SNP','?')}</b><br>Chr{int(r['CHR'])}:{int(r['BP']):,}"
                          f"<br>p={r['P']:.2e}<br>Effect={r['EFFECT']:.3f} ({r.get('EFFECT_TYPE','?')})",
                axis=1
            )
            fig.add_trace(go.Scatter(
                x=sig["BP_cum"], y=sig["-log10p"],
                mode="markers" + ("+text" if show_labels else ""),
                marker=dict(size=10, color="#E63946", symbol="diamond",
                            line=dict(width=1, color="white")),
                text=sig.get("SNP", pd.Series([""] * len(sig))),
                textposition="top center",
                textfont=dict(size=8),
                hovertemplate=hover_sig + "<extra></extra>",
                showlegend=True, name=f"GW-sig (p<{gw_thresh:.0e})",
            ))

    # Threshold lines
    fig.add_hline(y=-np.log10(gw_thresh), line_dash="dash",
                  line_color="#E63946", line_width=1.2,
                  annotation_text=f"p={gw_thresh:.0e}", annotation_position="right")
    fig.add_hline(y=-np.log10(1e-5), line_dash="dot",
                  line_color="#F4A261", line_width=0.9,
                  annotation_text="p=1e-5", annotation_position="right")

    fig.update_layout(
        xaxis=dict(
            tickvals=list(chr_mids.values()),
            ticktext=[str(c) for c in chr_mids],
            title="Chromosome", tickfont=dict(size=10),
        ),
        yaxis=dict(title="–log₁₀(P)", gridcolor="#f0f0f0"),
        height=480, plot_bgcolor="white",
        margin=dict(l=60, r=20, t=20, b=50),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)

    if not sig_hits.empty:
        st.info(f"**{len(sig_hits)} genome-wide significant hits** at p < {gw_thresh:.0e}. "
                "Click/hover on red diamonds for details.")


# ── TAB 2: QQ Plot ────────────────────────────────────────────────────────────

with tab2:
    st.subheader("QQ Plot — Observed vs Expected p-values")

    p_vals = df["P"].dropna().values
    p_vals = p_vals[(p_vals > 0) & (p_vals <= 1)]
    n_pts  = len(p_vals)
    obs    = np.sort(-np.log10(p_vals))[::-1]
    exp    = -np.log10(np.arange(1, n_pts+1) / (n_pts+1))

    # Thin for interactivity
    idx = np.unique(np.round(np.linspace(0, n_pts-1, 6000)).astype(int))
    obs_t, exp_t = obs[idx], exp[idx]
    p_t = p_vals[np.argsort(p_vals)[idx]]

    col_qq, col_info = st.columns([2, 1])
    with col_qq:
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=exp_t, y=obs_t, mode="markers",
            marker=dict(size=4, color="#264653", opacity=0.6),
            hovertemplate="Expected: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>",
            name="Variants",
        ))
        mx = max(obs_t.max(), exp_t.max()) * 1.06
        fig_qq.add_trace(go.Scatter(
            x=[0, mx], y=[0, mx], mode="lines",
            line=dict(color="#E63946", dash="dash", width=1.5),
            name="Expected (H₀)",
        ))
        # 95% CI band
        se = 1.96 / np.sqrt(n_pts / 2)
        fig_qq.add_trace(go.Scatter(
            x=list(exp_t) + list(exp_t[::-1]),
            y=list(exp_t + se) + list((exp_t - se)[::-1]),
            fill="toself", fillcolor="rgba(230,57,70,0.08)",
            line=dict(width=0), name="95% CI", showlegend=True,
        ))
        fig_qq.update_layout(
            xaxis=dict(title="Expected –log₁₀(P)", range=[0, mx]),
            yaxis=dict(title="Observed –log₁₀(P)", range=[0, mx]),
            height=420, plot_bgcolor="white",
            legend=dict(yanchor="top", y=0.3, xanchor="left", x=0.02),
            margin=dict(l=60, r=20, t=20, b=50),
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    with col_info:
        st.markdown("### Interpretation")
        st.metric("λ_GC", f"{lambda_gc:.3f}")
        st.markdown(f"""
**λ_GC = {lambda_gc:.3f}**

{"✅ Within expected range for a polygenic trait — inflation is real signal, not confounding." if lambda_gc < 1.5 else "⚠️ Elevated inflation — check population stratification."}

**What this means:**
- λ_GC measures how much the observed test statistics are inflated vs null
- λ_GC = 1.0 → no inflation (null)
- λ_GC > 1.0 → inflation from either:
  - **True polygenicity** (many small-effect variants) ✓
  - Population stratification ✗
  - Cryptic relatedness ✗

For SUD (λ={lambda_gc:.3f}), the inflation likely reflects genuine polygenic architecture across substance use traits — consistent with SNP-heritability estimates of 8–12% for alcohol use disorder.
        """)


# ── TAB 3: Top Loci ───────────────────────────────────────────────────────────

with tab3:
    st.subheader("Top Genome-Wide Significant Loci")

    n_top = st.slider("Show top N loci", 5, 100, 25)

    if df["P"].min() < 1e-4:
        top_df = df.nsmallest(n_top, "P").copy()
        top_df["–log10(P)"] = top_df["-log10p"].round(2)
        top_df["Effect"]    = top_df["EFFECT"].round(4) if "EFFECT" in top_df.columns else "N/A"
        top_df["Type"]      = top_df.get("EFFECT_TYPE", "?")
        display_cols = ["SNP","CHR","BP","P","–log10(P)","Effect","Type"]
        display_cols = [c for c in display_cols if c in top_df.columns]

        st.dataframe(
            top_df[display_cols].reset_index(drop=True),
            use_container_width=True,
            column_config={
                "P": st.column_config.NumberColumn(format="%.2e"),
                "CHR": st.column_config.NumberColumn(format="%d"),
                "BP": st.column_config.NumberColumn(format="%d"),
                "–log10(P)": st.column_config.ProgressColumn(
                    min_value=0, max_value=float(df["-log10p"].max()),
                    format="%.2f",
                ),
            },
            hide_index=True,
        )

        # Effect size distribution of top hits
        if "EFFECT" in top_df.columns:
            st.subheader("Effect size distribution — top loci")
            fig_eff = px.strip(
                top_df.dropna(subset=["EFFECT"]),
                x="Type", y="EFFECT", color="CHR",
                hover_data=["SNP","P","CHR","BP"],
                color_continuous_scale="Viridis",
                labels={"EFFECT": "Effect size (OR or BETA)", "Type": "Effect type"},
            )
            fig_eff.add_hline(y=1 if "OR" in top_df["Type"].values else 0,
                              line_dash="dash", line_color="#888", line_width=1)
            fig_eff.update_layout(height=350, plot_bgcolor="white",
                                  margin=dict(l=50, r=20, t=20, b=40))
            st.plotly_chart(fig_eff, use_container_width=True)
    else:
        st.info("No genome-wide significant hits in this sample. "
                "Try loading more shards (sidebar slider).")

    st.markdown("""
**Known SUD risk loci** (from the full PGC SUD2023 paper, Hatoum et al. 2023):
- **ADH1B** (chr4) — alcohol metabolism, large effect (OR≈0.7 protective allele)
- **CHRNA5-CHRNA3-CHRNB4** (chr15) — nicotinic receptor cluster, tobacco/opioid
- **DRD2/ANKK1** (chr11) — dopamine receptor, multiple substance dependence
- **FOXP2** (chr7) — general addiction liability factor
- **KSR2** (chr12) — shared with ADHD, BMI
    """)


# ── TAB 4: Cross-Disorder Comparison ─────────────────────────────────────────

with tab4:
    st.subheader("Cross-Disorder Genetic Effect Correlation")

    # Hardcoded from our v3 run (489k SNPs)
    conditions = ["adhd","anxiety","autism","bipolar","cross_disorder",
                  "eating","mdd","ptsd","schizophrenia","substance_use"]
    labels_map = {
        "adhd":"ADHD","anxiety":"Anxiety","autism":"Autism","bipolar":"Bipolar",
        "cross_disorder":"Cross-Disorder","eating":"Eating Disorders","mdd":"MDD",
        "ptsd":"PTSD","schizophrenia":"Schizophrenia","substance_use":"Substance Use",
    }
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

    # Highlight substance_use row
    selected = st.selectbox(
        "Highlight condition",
        options=conditions,
        index=conditions.index("substance_use"),
        format_func=lambda c: labels_map[c],
    )

    col_heat, col_bar = st.columns([3, 2])

    with col_heat:
        mask = np.triu(np.ones_like(corr_vals, dtype=bool), k=1)
        fig_h = go.Figure(go.Heatmap(
            z=np.where(mask, np.nan, corr_vals),
            x=[labels_map[c] for c in conditions],
            y=[labels_map[c] for c in conditions],
            colorscale="RdBu_r",
            zmid=0, zmin=-0.45, zmax=0.45,
            text=np.where(mask, "", corr_df.round(2).astype(str).values),
            texttemplate="%{text}",
            textfont=dict(size=9),
            hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
            colorbar=dict(title="r", thickness=12),
        ))
        # Highlight selected row/col
        sel_idx = conditions.index(selected)
        fig_h.add_shape(type="rect",
            x0=sel_idx-0.5, x1=sel_idx+0.5, y0=-0.5, y1=len(conditions)-0.5,
            line=dict(color="#E63946", width=2), fillcolor="rgba(0,0,0,0)")
        fig_h.add_shape(type="rect",
            x0=-0.5, x1=len(conditions)-0.5, y0=sel_idx-0.5, y1=sel_idx+0.5,
            line=dict(color="#E63946", width=2), fillcolor="rgba(0,0,0,0)")
        fig_h.update_layout(
            height=400, margin=dict(l=120, r=20, t=20, b=120),
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with col_bar:
        row = corr_df.loc[selected].drop(index=selected).sort_values(ascending=False)
        bar_colors = ["#E63946" if v > 0.05 else ("#4C9BE8" if v < -0.05 else "#ccc")
                      for v in row.values]
        fig_bar = go.Figure(go.Bar(
            y=[labels_map[c] for c in row.index],
            x=row.values,
            orientation="h",
            marker_color=bar_colors,
            hovertemplate="%{y}: r=%{x:.3f}<extra></extra>",
        ))
        fig_bar.add_vline(x=0, line_color="#333", line_width=1)
        fig_bar.update_layout(
            title=f"Correlations with {labels_map[selected]}",
            xaxis=dict(title="Effect correlation (r)", range=[-0.5, 0.5]),
            yaxis=dict(title=""),
            height=400, plot_bgcolor="white",
            margin=dict(l=130, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    if selected == "substance_use":
        st.info("""
**Key finding:** Substance use shows the strongest correlation with **Schizophrenia (r=0.190)**,
consistent with shared dopaminergic pathways (DRD2 locus implicated in both).
The autism link (r=0.040) and anxiety link (r=0.041) are modest but non-trivial.
The **negative MDD correlation (r=−0.043)** is intriguing — may reflect diagnostic overlap masking
distinct genetic subtypes (internalising vs externalising disorders).
        """)


# ── TAB 5: PRS Simulation ─────────────────────────────────────────────────────

with tab5:
    st.subheader("Polygenic Risk Score Simulation")
    st.markdown("""
Simulate the distribution of **polygenic risk scores** (PRS) for substance use disorder
under different genetic architectures. This is a forward simulation — not an actual
PRS from genotype data — but it illustrates how genetic liability is distributed in
the population.
    """)

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        h2_snp = st.slider("SNP heritability (h²_SNP)", 0.02, 0.30, 0.10, 0.01,
                           help="Proportion of phenotypic variance explained by common SNPs. "
                                "SUD h² ≈ 8–12% (alcohol), 12–22% (tobacco).")
        n_cases_pct = st.slider("Population prevalence (%)", 1, 30, 12, 1,
                                help="Lifetime prevalence of substance use disorder. ~12% globally.")
        n_sim = st.select_slider("Population size", [10_000, 50_000, 100_000], value=50_000)
        show_comorbid = st.toggle("Show ADHD/SCZ comorbidity shift", value=True)

    with col_plot:
        np.random.seed(42)
        # Liability-threshold model
        # PRS ~ N(0, h²_snp) in population
        prs_pop = np.random.normal(0, np.sqrt(h2_snp), n_sim)
        threshold = np.percentile(prs_pop, 100 - n_cases_pct)

        cases    = prs_pop[prs_pop >= threshold]
        controls = prs_pop[prs_pop <  threshold]

        fig_prs = go.Figure()
        fig_prs.add_trace(go.Histogram(
            x=controls, nbinsx=80, name="Controls",
            marker_color="#4C9BE8", opacity=0.6,
            histnorm="probability density",
        ))
        fig_prs.add_trace(go.Histogram(
            x=cases, nbinsx=60, name="Cases (SUD)",
            marker_color="#E63946", opacity=0.7,
            histnorm="probability density",
        ))

        if show_comorbid:
            # ADHD comorbidity shifts PRS upward (r_g(ADHD,SUD) ~ 0.30 from literature)
            rg_adhd = 0.30
            prs_adhd_shift = prs_pop + rg_adhd * np.random.normal(0, np.sqrt(h2_snp), n_sim)
            adhd_cases = prs_adhd_shift[prs_adhd_shift >= threshold]
            fig_prs.add_trace(go.Histogram(
                x=adhd_cases, nbinsx=50, name="Cases + ADHD liability",
                marker_color="#8B5CF6", opacity=0.6,
                histnorm="probability density",
            ))

            # SCZ comorbidity (r_g(SCZ,SUD) ~ 0.19 from our data)
            rg_scz = 0.19
            prs_scz_shift = prs_pop + rg_scz * np.random.normal(0, np.sqrt(h2_snp), n_sim)
            scz_cases = prs_scz_shift[prs_scz_shift >= threshold]
            fig_prs.add_trace(go.Histogram(
                x=scz_cases, nbinsx=50, name="Cases + SCZ liability",
                marker_color="#264653", opacity=0.6,
                histnorm="probability density",
            ))

        fig_prs.add_vline(x=threshold, line_dash="dash", line_color="#333",
                          annotation_text=f"Threshold ({n_cases_pct}% prev.)",
                          annotation_position="top left")
        fig_prs.update_layout(
            barmode="overlay",
            xaxis=dict(title="Polygenic Risk Score"),
            yaxis=dict(title="Density"),
            height=400, plot_bgcolor="white",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=50, r=20, t=20, b=50),
        )
        st.plotly_chart(fig_prs, use_container_width=True)

    # Metrics
    m1, m2, m3 = st.columns(3)
    mean_case = cases.mean()
    mean_ctrl = controls.mean()
    auc_approx = stats.norm.cdf(np.sqrt(h2_snp / (1 - h2_snp)))
    m1.metric("Mean PRS — Cases", f"{mean_case:.3f}")
    m2.metric("Mean PRS — Controls", f"{mean_ctrl:.3f}")
    m3.metric("Approximate AUC", f"{auc_approx:.3f}",
              help="Area under ROC curve for PRS-based prediction")

    st.markdown(f"""
**Interpretation:**
- With h²_SNP = {h2_snp:.2f}, the PRS alone achieves AUC ≈ {auc_approx:.2f}
  (50% = chance, 100% = perfect)
- The **ADHD comorbidity shift** (purple) reflects the genetic correlation r_g ≈ 0.30
  between ADHD and SUD — people with high ADHD genetic liability also tend toward
  higher SUD genetic liability
- The **SCZ shift** (dark teal) reflects r_g ≈ 0.19 from our cross-disorder analysis,
  consistent with shared dopaminergic risk
- Real-world SUD PRS (from UKBB) achieves AUC ≈ 0.56–0.62 after including
  environmental risk factors alongside genetics
    """)


# ── TAB 6: ADHD × Bipolar × Autism ───────────────────────────────────────────

with tab6:
    st.subheader("ADHD × Bipolar × Autism — Shared Genetic Architecture")

    COLORS_T = {"adhd": "#4C9BE8", "bipolar": "#E76F51", "autism": "#2A9D8F"}
    NAMES_T  = {"adhd": "ADHD (2022)", "bipolar": "Bipolar (2021)", "autism": "Autism (2019)"}

    # ── Load merged triad data if available ──────────────────────────────────
    MERGED_PATH = "data_cache/merged_triad.parquet"
    TOP_PLEI    = "data_cache/top_pleiotropic_snps.csv"

    @st.cache_data(show_spinner="Loading triad data...")
    def load_triad():
        if os.path.exists(MERGED_PATH):
            return pd.read_parquet(MERGED_PATH)
        return pd.DataFrame()

    import os
    triad = load_triad()

    if triad.empty:
        st.warning("Triad data not yet computed. Run `python triad_analysis.py` first — it takes ~5 min to download the data.")
        st.info("While you wait, the figures below show the key results from published LDSC genetic correlations.")
    else:
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Shared SNPs (all 3)", f"{len(triad):,}")
        m2.metric("Pleiotropic hits (≥2 GW-sig)", f"{triad['pleiotropic'].sum():,}")
        m3.metric("Significant in all 3", f"{(triad['n_sig']==3).sum():,}")
        m4.metric("Top Fisher p (all 3)", f"{triad['fisher_p'].min():.2e}")

        st.divider()

    # ── Published genetic correlations (Anttila et al. 2018 LDSC) ───────────
    st.subheader("Genetic Correlations (LDSC, published estimates)")

    rg_data = {
        "Pair": ["ADHD–Autism", "ADHD–Bipolar", "ADHD–MDD", "ADHD–SCZ",
                 "Autism–Bipolar", "Autism–SCZ", "Bipolar–MDD", "Bipolar–SCZ"],
        "rg":   [0.36, 0.19, 0.32, 0.15, 0.09, 0.16, 0.35, 0.68],
        "SE":   [0.03, 0.03, 0.03, 0.03, 0.04, 0.03, 0.03, 0.02],
        "Source": ["Anttila 2018"]*8,
    }
    rg_df = pd.DataFrame(rg_data)
    rg_df["lower"] = rg_df["rg"] - 1.96*rg_df["SE"]
    rg_df["upper"] = rg_df["rg"] + 1.96*rg_df["SE"]

    highlight_pairs = ["ADHD–Autism", "ADHD–Bipolar", "Autism–Bipolar"]
    rg_df["highlight"] = rg_df["Pair"].isin(highlight_pairs)

    col_rg, col_info = st.columns([2, 1])
    with col_rg:
        fig_rg = go.Figure()
        for _, row in rg_df.iterrows():
            col = "#E63946" if row["highlight"] else "#888"
            fig_rg.add_trace(go.Scatter(
                x=[row["rg"]], y=[row["Pair"]],
                error_x=dict(type="data", symmetric=False,
                             array=[row["upper"]-row["rg"]],
                             arrayminus=[row["rg"]-row["lower"]],
                             color=col, thickness=2),
                mode="markers",
                marker=dict(size=12, color=col,
                            symbol="diamond" if row["highlight"] else "circle"),
                hovertemplate=f"<b>{row['Pair']}</b><br>rg={row['rg']:.2f} ± {row['SE']:.2f}<extra></extra>",
                showlegend=False,
            ))
        fig_rg.add_vline(x=0, line_dash="dash", line_color="#999", line_width=1)
        fig_rg.update_layout(
            xaxis=dict(title="Genetic correlation (rg)", range=[-0.1, 0.85]),
            yaxis=dict(title=""),
            height=380, plot_bgcolor="white",
            margin=dict(l=160, r=20, t=20, b=50),
        )
        st.plotly_chart(fig_rg, use_container_width=True)

    with col_info:
        st.markdown("""
**Key correlations:**

🔴 **ADHD–Autism: rg = 0.36**
Strongest link. Shared neurodevelopmental pathways — executive function, sensory processing, social cognition.

🔴 **ADHD–Bipolar: rg = 0.19**
Dopaminergic overlap. Risk alleles at *DRD4*, *ANK3*, *CACNA1C* implicated in both.

🔴 **Autism–Bipolar: rg = 0.09**
Weaker but non-zero. *CACNA1C*, *SHANK3*, *NRXN1* shared.

⚡ **Bipolar–SCZ: rg = 0.68**
The strongest cross-disorder correlation in psychiatry — essentially a continuum.

Source: Anttila et al. 2018, *Science* (LDSC on 265,218 participants)
        """)

    # ── Known pleiotropic loci table ─────────────────────────────────────────
    st.divider()
    st.subheader("Known Pleiotropic Loci — Literature Curated")

    loci_table = [
        {"Gene": "CACNA1C",  "Chr": 12, "Function": "L-type Ca²⁺ channel; synaptic plasticity",
         "ADHD": "✓", "Bipolar": "✓✓", "Autism": "✓"},
        {"Gene": "ANK3",     "Chr": 10, "Function": "Ankyrin G; axon initial segment",
         "ADHD": "✓", "Bipolar": "✓✓", "Autism": ""},
        {"Gene": "SHANK3",   "Chr": 22, "Function": "Postsynaptic scaffold; glutamate signalling",
         "ADHD": "",  "Bipolar": "✓",  "Autism": "✓✓"},
        {"Gene": "RBFOX1",   "Chr": 16, "Function": "RNA-binding; neuronal splicing",
         "ADHD": "✓✓","Bipolar": "",   "Autism": "✓"},
        {"Gene": "DRD4",     "Chr": 11, "Function": "Dopamine D4 receptor; novelty-seeking",
         "ADHD": "✓✓","Bipolar": "✓",  "Autism": ""},
        {"Gene": "NRXN1",    "Chr":  2, "Function": "Neurexin-1; synaptic adhesion E/I balance",
         "ADHD": "✓", "Bipolar": "✓",  "Autism": "✓✓"},
        {"Gene": "KDM5B",    "Chr":  1, "Function": "Histone demethylase; gene regulation",
         "ADHD": "✓", "Bipolar": "",   "Autism": "✓✓"},
        {"Gene": "DYRK1A",   "Chr": 21, "Function": "Kinase; neuronal proliferation",
         "ADHD": "✓", "Bipolar": "",   "Autism": "✓✓"},
    ]
    loci_tdf = pd.DataFrame(loci_table)
    st.dataframe(loci_tdf, use_container_width=True, hide_index=True,
                 column_config={
                     "ADHD":    st.column_config.TextColumn(width="small"),
                     "Bipolar": st.column_config.TextColumn(width="small"),
                     "Autism":  st.column_config.TextColumn(width="small"),
                 })
    st.caption("✓ = implicated; ✓✓ = originally discovered / strongest association")

    # ── PRS cross-prediction simulation ──────────────────────────────────────
    st.divider()
    st.subheader("PRS Cross-Prediction Simulation")
    st.markdown("How much does your ADHD polygenic score predict bipolar and autism risk?")

    cc1, cc2, cc3 = st.columns(3)
    rg_ab = cc1.slider("rg(ADHD, Bipolar)", 0.05, 0.50, 0.19, 0.01)
    rg_aa = cc2.slider("rg(ADHD, Autism)",  0.05, 0.60, 0.36, 0.01)
    rg_ba = cc3.slider("rg(Bipolar, Autism)",0.00, 0.40, 0.09, 0.01)

    import numpy as _np
    _N = 100_000
    _np.random.seed(42)
    h2 = {"adhd": 0.22, "bipolar": 0.23, "autism": 0.18}
    prev = {"adhd": 0.05, "bipolar": 0.02, "autism": 0.015}
    Sigma = _np.array([
        [h2["adhd"],
         rg_ab*_np.sqrt(h2["adhd"]*h2["bipolar"]),
         rg_aa*_np.sqrt(h2["adhd"]*h2["autism"])],
        [rg_ab*_np.sqrt(h2["adhd"]*h2["bipolar"]),
         h2["bipolar"],
         rg_ba*_np.sqrt(h2["bipolar"]*h2["autism"])],
        [rg_aa*_np.sqrt(h2["adhd"]*h2["autism"]),
         rg_ba*_np.sqrt(h2["bipolar"]*h2["autism"]),
         h2["autism"]],
    ])
    try:
        L = _np.linalg.cholesky(Sigma)
    except _np.linalg.LinAlgError:
        st.error("Invalid correlation matrix — adjust sliders.")
        st.stop()
    Z = _np.random.normal(0, 1, (_N, 3))
    PRS = Z @ L.T
    thresh = {c: _np.percentile(PRS[:, i], (1-prev[c])*100)
              for i, c in enumerate(["adhd","bipolar","autism"])}
    case = {c: (PRS[:, i] >= thresh[c])
            for i, c in enumerate(["adhd","bipolar","autism"])}

    # Show risk by ADHD decile for bipolar and autism
    fig_cross = go.Figure()
    decile_edges = _np.percentile(PRS[:, 0], _np.arange(0, 101, 10))
    for target, col, rg_val in [
        ("bipolar","#E76F51", rg_ab),
        ("autism", "#2A9D8F", rg_aa),
    ]:
        risks = []
        for lo, hi in zip(decile_edges[:-1], decile_edges[1:]):
            mask = (PRS[:,0] >= lo) & (PRS[:,0] < hi)
            risks.append(case[target][mask].mean()*100)
        fig_cross.add_trace(go.Scatter(
            x=list(range(1,11)), y=risks,
            mode="lines+markers",
            line=dict(color=col, width=2.5),
            marker=dict(size=8, color=col),
            name=f"{target.title()} risk (rg={rg_val:.2f})",
            hovertemplate="Decile %{x}: %{y:.2f}%<extra></extra>",
        ))
        fig_cross.add_hline(y=prev[target]*100, line_dash="dot",
                            line_color=col, line_width=1, opacity=0.5)

    fig_cross.update_layout(
        xaxis=dict(title="ADHD PRS Decile", tickvals=list(range(1,11)),
                   ticktext=[f"D{i}" for i in range(1,11)]),
        yaxis=dict(title="Risk of condition (%)"),
        height=380, plot_bgcolor="white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=60, r=20, t=20, b=50),
    )
    st.plotly_chart(fig_cross, use_container_width=True)
    st.caption("Dotted lines = population average prevalence. Top ADHD decile carries ~2–4× population risk for bipolar/autism.")

