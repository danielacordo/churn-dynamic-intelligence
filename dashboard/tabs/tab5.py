import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm
from dashboard.tabs.context import DashboardContext
from src.visualization import plot_heatmap_annotated
from src.uncertainty import churn_probability_with_uncertainty


def render(ctx: DashboardContext) -> None:
    """Render the Monte Carlo Analysis tab"""

    st.markdown("#### Monte Carlo Uncertainty Analysis")
    st.markdown("""
    <p class='small'>
    The model runs two independent uncertainty methods per customer and compares them.
    When they agree (Δσ &lt; 0.02) the analytical result is trusted.
    When they diverge, a boundary effect is flagged — the customer sits near the
    edge of the probability space and the MC distribution is more reliable.
    </p>
    """, unsafe_allow_html=True)

    # Section 1: ROI distribution under uncertainty
    st.markdown("---")
    st.markdown("### 1 - ROI distribution under campaign uncertainty")
    st.markdown("<p class='small'>Monte Carlo over the full customer base: sample P(churn) for each customer and compute campaign ROI across N scenarios. Fully vectorized - runs in a single matrix operation.</p>", unsafe_allow_html=True)

    n_mc_roi = st.slider("MC scenarios", 500, 5000, 2000, 500, key="mc_roi_n")

    @st.cache_data(show_spinner=False)
    def run_roi_mc(prob_arr, sigma_arr, n_customers, clv_total, ret_cost_val, n_scenarios, seed=42):
        import numpy as np  # local — st.cache_data closure
        """ Vectorized MC: (n_scenarios x n_customers) matrix in one shot.
        ~50x faster than the per-scenario loop"""

        rng = np.random.default_rng(seed)
        # Shape: (n_scenarios, n_customers)
        p_matrix = np.clip(rng.normal(prob_arr, sigma_arr, size=(n_scenarios, len(prob_arr))), 0, 1,)
        # Expected churners per scenario: sum across customers
        n_churners_vec = p_matrix.sum(axis=1)       
        # Sample success rate per scenario
        sr_vec = np.clip(rng.normal(0.30, 0.05, n_scenarios), 0.05, 0.60)
        revenue_vec = n_churners_vec * sr_vec * clv_total
        campaign_cost = n_customers * ret_cost_val
        return revenue_vec - campaign_cost

    with st.spinner("Running Monte Carlo ROI simulation…"):
        roi_arr = run_roi_mc(
            prob_arr = ctx.df_f["prob_churn"].values,
            sigma_arr = ctx.df_f["sigma_prob"].values,
            n_customers = len(ctx.df_f),
            clv_total = float(ctx.clv * ctx.horizon),
            ret_cost_val= float(ctx.ret_cost),
            n_scenarios = n_mc_roi,)

    roi_p5 = float(np.percentile(roi_arr, 5))
    roi_p50 = float(np.percentile(roi_arr, 50))
    roi_p95 = float(np.percentile(roi_arr, 95))
    roi_prob_positive = float((roi_arr > 0).mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median ROI", f"${roi_p50:+,.0f}")
    c2.metric("5th percentile (worst)", f"${roi_p5:+,.0f}")
    c3.metric("95th percentile (best)", f"${roi_p95:+,.0f}")
    c4.metric("P(ROI > 0)", f"{roi_prob_positive:.0%}")

    # Two separate traces (positive / negative) 
    # Violin + KDE overlay 
    from scipy.stats import gaussian_kde as _gkde
    fig_roi_dist = go.Figure()

    # Violin
    fig_roi_dist.add_trace(go.Violin(
        x=roi_arr,
        orientation="h",
        side="positive",
        fillcolor="rgba(248,150,30,0.15)",
        line=dict(color="#f8961e", width=1.5),
        points=False,
        name="ROI distribution",
        showlegend=False,
        box=dict(visible=True, fillcolor="rgba(248,150,30,0.4)", line=dict(color="#f8961e")),
        meanline=dict(visible=True, color="#ffbe0b", width=2),))

    # KDE smooth curve
    _kde_roi = _gkde(roi_arr, bw_method=0.15)
    _x_range = np.linspace(roi_arr.min(), roi_arr.max(), 300)
    _y_kde = _kde_roi(_x_range)
    _y_scaled = _y_kde / _y_kde.max() * 0.45   
    fig_roi_dist.add_trace(go.Scatter(
        x=_x_range, y=_y_scaled,
        mode="lines", name="KDE density",
        line=dict(color="#f8961e", width=2.5, dash="solid"),
        fill="tozeroy",
        fillcolor="rgba(248,150,30,0.07)",))

    # Annotated percentile lines
    for val, color, label in [
        (0, "#888888", "Break-even"),
        (roi_p5, "#ff4d6d", f"5th pct ${roi_p5:+,.0f}"),
        (roi_p50, "#ffbe0b", f"Median ${roi_p50:+,.0f}"),
        (roi_p95, "#06ffa5", f"95th pct ${roi_p95:+,.0f}"),]:
        fig_roi_dist.add_vline(x=val, line_dash="dot" if val != roi_p50 else "solid",
                                line_color=color, line_width=1.8,
                                annotation_text=label,
                                annotation_font=dict(color=color, size=9),
                                annotation_position="top")

    fig_roi_dist.update_layout(**ctx.plotly_dark(), height=300,
        title=f"ROI distribution - {n_mc_roi:,} Monte Carlo scenarios",
        showlegend=False)
    fig_roi_dist.update_xaxes(**ctx.plotly_axis_style(),
        title_text="Net ROI ($)", tickprefix="$", tickformat=",.0f")
    fig_roi_dist.update_yaxes(visible=False)
    st.plotly_chart(fig_roi_dist, use_container_width=True, key="t5_roi_dist")

    st.markdown(f"""
    <div class='card card-accent'>
    <span class='small'>
    The campaign is ROI-positive in <b style='color:#06ffa5;'>{roi_prob_positive:.0%} of scenarios</b>.
    Even in the worst 5% of cases, ROI is <b style='color:#ff4d6d;'>${roi_p5:+,.0f}</b>.
    The wide range reflects genuine uncertainty in churn probabilities — not a model flaw.
    </span>
    </div>
    """, unsafe_allow_html=True)

    # Section 2: MC vs Analytical per customer 
    st.markdown("---")
    st.markdown("### 2 - Analytical vs Monte Carlo: where do they diverge?")
    st.markdown("<p class='small'>Run both methods on every customer. Divergence (Δσ > 0.02) flags a boundary effect — the analytical Gaussian assumption breaks down near prob = 0 or prob = 1.</p>", unsafe_allow_html=True)

    mc_sample_size = st.slider("Customers to analyze (MC is slow)", 50, 300, 100, 50, key="mc_vs_anal_n")

    @st.cache_data(show_spinner=False)
    def run_mc_comparison(E0_arr, gamma_arr, E_eq_arr, customer_ids, horizon_val, n_samples=2000):
        import pandas as pd  
        results = []
        for i, cid in enumerate(customer_ids):
            comp = ctx.compare_methods(
                E0=float(E0_arr[i]),    sigma_E0=0.05,
                gamma=float(gamma_arr[i]), sigma_gamma=0.02,
                E_eq=float(E_eq_arr[i]), sigma_E_eq=0.03,
                t_horizon=float(horizon_val), n_samples=n_samples,)
            results.append({
                "customerID": cid,
                "prob_analytical": comp["analytical"]["prob"],
                "prob_mc": comp["monte_carlo"]["prob"],
                "sigma_analytical": comp["analytical"]["sigma_prob"],
                "sigma_mc": comp["monte_carlo"]["sigma_prob"],
                "sigma_delta": comp["sigma_delta"],
                "boundary_effect": comp["boundary_effect"],})
        return pd.DataFrame(results)

    with st.spinner(f"Running MC on {mc_sample_size} customers…"):
        sample_df = ctx.df_f.sample(min(mc_sample_size, len(ctx.df_f)), random_state=42)
        mc_df = run_mc_comparison(
            E0_arr = sample_df["E0"].values,
            gamma_arr = sample_df["gamma"].values,
            E_eq_arr = sample_df["E_eq"].values,
            customer_ids= tuple(sample_df["customerID"].tolist()),
            horizon_val = ctx.horizon,)
        # Re-attach risk_level for coloring
        mc_df["risk_level"] = sample_df["risk_level"].values

    n_boundary = mc_df["boundary_effect"].sum()
    pct_boundary = n_boundary / len(mc_df)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Customers analyzed", f"{len(mc_df)}")
    col_b.metric("Boundary effects", f"{n_boundary}", f"{pct_boundary:.0%} of sample")
    col_c.metric("Avg Δσ", f"{mc_df['sigma_delta'].mean():.4f}", "< 0.02 = methods agree")

    fig_comp = go.Figure()
    for boundary, color, name in [(False, "#4cc9f0", "No boundary effect"), (True, "#ff4d6d", "Boundary effect")]:
        sub = mc_df[mc_df["boundary_effect"] == boundary]
        if len(sub) == 0:
            continue
        fig_comp.add_trace(go.Scatter(
            x=sub["prob_analytical"], y=sub["prob_mc"],
            mode="markers", name=name,
            marker=dict(color=color, size=7, opacity=0.7, line=dict(width=0)),
            hovertemplate="<b>%{customdata}</b><br>Analytical=%{x:.3f}<br>MC=%{y:.3f}<extra></extra>",
            customdata=sub["customerID"],))
    # Perfect agreement line
    fig_comp.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#9999c0", dash="dash", width=1),
        name="Perfect agreement", showlegend=True,))
    fig_comp.update_layout(**ctx.plotly_dark(), height=360,
                            xaxis_title="P(churn) — Analytical",
                            yaxis_title="P(churn) — Monte Carlo",
                            xaxis=dict(range=[0, 1], gridcolor="#1e1e3a"),
                            yaxis=dict(range=[0, 1], gridcolor="#1e1e3a"),
                            title="Analytical vs MC probability estimates")
    st.plotly_chart(fig_comp, use_container_width=True, key="t5_comp")

    # Sigma delta distribution
    fig_delta = go.Figure(go.Histogram(
        x=mc_df["sigma_delta"], nbinsx=30,
        marker_color=["#ff4d6d" if v > 0.02 else "#4cc9f0"
                      for v in mc_df["sigma_delta"]],
        marker=dict(line=dict(width=0)), opacity=0.85,))
    fig_delta.add_vline(x=0.02, line_color="#f8961e", line_dash="dash", line_width=2,
                         annotation_text="Boundary ctx.threshold (0.02)",
                         annotation_font_color="#f8961e")
    fig_delta.update_layout(**ctx.plotly_dark(), height=260,
                             xaxis_title="Δσ (|sigma_analytical - sigma_MC|)",
                             yaxis_title="Customers",
                             title="Distribution of sigma divergence")
    st.plotly_chart(fig_delta, use_container_width=True, key="t5_delta")

    if n_boundary > 0:
        st.markdown(f"""
        <div class='card card-churn'>
        <b>{n_boundary} customers ({pct_boundary:.0%}) show boundary effects</b><br>
        <span class='small'>
        These customers have P(churn) near 0 or 1 - the Gaussian approximation used by
        the analytical method distorts their uncertainty intervals. For these customers,
        the MC percentiles (ci_95_lower / ci_95_upper) are more reliable than ±σ.
        </span>
        </div>
        """, unsafe_allow_html=True)

    # Section 3: Sensitivity analysis 
    st.markdown("---")
    st.markdown("### 3 — Sensitivity analysis: what drives the ROI estimate?")
    st.markdown("<p class='small'>How much does the ROI conclusion change if the business assumptions are wrong? The tornado chart ranks each assumption by its individual impact.</p>", unsafe_allow_html=True)

    n_int_sens = (ctx.df_f["risk_level"].isin(["HIGH", "MEDIUM"])).sum()
    n_tp_sens  = int(ctx.df_f[ctx.df_f["risk_level"].isin(["HIGH", "MEDIUM"])]["churn"].sum()) if "churn" in ctx.df_f.columns else int(n_int_sens * 0.30)

    @st.cache_data(show_spinner=False)
    def run_sensitivity(n_int, n_tp, ret_cost_val, horizon_val):
        return ctx.sensitivity_analysis(
            n_intervened=n_int,
            n_true_positives=n_tp,
            clv_monthly_range=(30, 130, 11),
            success_rate_range=(0.05, 0.55, 11),
            retention_cost=ret_cost_val,
            horizon_months=horizon_val,
        )

    sens = run_sensitivity(n_int_sens, n_tp_sens, ctx.ret_cost, ctx.horizon)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Base ROI", f"${sens['base_roi']:+,.0f}")
        st.metric("Campaign cost", f"${sens['campaign_cost']:,.0f}")
        st.metric("Customers flagged", f"{n_int_sens:,}")

    with col_s2:
        # Tornado chart
        tornado = sens["tornado"]
        fig_tornado = go.Figure(go.Bar(
            y=[t["assumption"] for t in tornado],
            x=[t["impact"] for t in tornado],
            orientation="h",
            marker_color="#f8961e", marker_line_width=0,
            text=[f"±${t['impact']:,.0f}" for t in tornado],
            textposition="outside", textfont=dict(color="white", size=11),))
        fig_tornado.update_layout(**ctx.plotly_dark(), height=240,
                                   title="ROI sensitivity by assumption",
                                   xaxis_title="ROI impact ($)", yaxis_title="")
        st.plotly_chart(fig_tornado, use_container_width=True, key="t5_tornado")

    # ROI heatmap over CLV × success_rate grid
    st.markdown("**ROI grid: CLV x success rate**")
    st.markdown("<p class='small'>Each cell is the net ROI at that combination. Green = positive, red = negative.</p>",
                unsafe_allow_html=True)

    clv_vals = sens["clv_values"]
    sr_vals = sens["success_rate_values"]
    grid = sens["roi_grid"]

    # Annotated heatmap, every cell shows its ROI value
    _base_sr_idx = int(np.argmin([abs(sr - 0.30) for sr in sr_vals]))
    fig_heat = plot_heatmap_annotated(
        z = np.array(grid),
        x_labels = [f"{sr:.0%}" for sr in sr_vals],
        y_labels = [f"${c:.0f}" for c in clv_vals],
        title = "Net ROI sensitivity - CLV x Success rate",
        x_title = "Success rate",
        y_title = "CLV / month ($)",
        colorscale= [[0,"#ff4d6d"],[0.5,"#1e1e3a"],[1,"#06ffa5"]],
        fmt = "${:+,.0f}", )
    # Mark the assumed 30% column
    fig_heat.add_vline(
        x=_base_sr_idx - 0.5,
        line_color="#f8961e", line_dash="dash", line_width=2,
        annotation_text="Assumed 30%",
        annotation_font=dict(color="#f8961e", size=11),)
    st.plotly_chart(fig_heat, use_container_width=True, key="t5_heat")

    # What % of grid is ROI-positive
    pct_positive_grid = float((grid > 0).mean())
    _cv1 = "#06ffa5" if grid[0,0] >= 0 else "#ff4d6d"
    st.markdown(f"""
    <div class='card card-green'>
    <span class='small'>
    ROI is positive in <b style='color:#06ffa5;'>{pct_positive_grid:.0%} of the assumption grid</b>.
    The assumption that matters most is <b style='color:white;'>{tornado[0]["assumption"]}</b>
    (±${tornado[0]["impact"]:,.0f} ROI impact).
    Even with conservative assumptions (CLV=${clv_vals[0]:.0f}, success rate={sr_vals[0]:.0%}),
    ROI is <b style='color:{_cv1};'>${grid[0,0]:+,.0f}</b>.
    </span>
    </div>
    """, unsafe_allow_html=True)

    # Section 4: Single customer deep dive 
    st.markdown("---")
    st.markdown("### 4 — Single customer deep dive")
    st.markdown("<p class='small'>Run full MC analysis (10,000 samples) on one customer and inspect the complete probability distribution.</p>", unsafe_allow_html=True)

    customer_ids = ctx.df_f["customerID"].tolist() if "customerID" in ctx.df_f.columns else list(range(len(ctx.df_f)))
    selected_id = st.selectbox("Select customer", customer_ids, key="mc_deep_customer")
    cust_row = ctx.df_f[ctx.df_f["customerID"] == selected_id].iloc[0] if "customerID" in ctx.df_f.columns else ctx.df_f.iloc[selected_id]

    with st.spinner(f"Running 10,000 MC samples for {selected_id}…"):
        @st.cache_data(show_spinner=False)
        def run_deep_mc(E0, gamma, E_eq, horizon_val, n_samples=10_000):
            mc = ctx.monte_carlo_churn(
                E0=E0, sigma_E0=0.05, gamma=gamma, sigma_gamma=0.02,
                E_eq=E_eq, sigma_E_eq=0.03,
                t_horizon=float(horizon_val), n_samples=n_samples,)
            anal = churn_probability_with_uncertainty(
                E0=E0, sigma_E0=0.05, gamma=gamma, sigma_gamma=0.02,
                E_eq=E_eq, sigma_E_eq=0.03, t_horizon=float(horizon_val),)
            return mc, anal

        mc_full, anal_full = run_deep_mc(
            E0 = float(cust_row["E0"]),
            gamma = float(cust_row["gamma"]),
            E_eq = float(cust_row["E_eq"]),
            horizon_val = ctx.horizon,)

    col_d1, col_d2 = st.columns([1, 2])

    with col_d1:
        risk_color = ctx.RISK_COLORS.get(mc_full["risk_level"], "#f8961e")
        sigma_diff = abs(mc_full["sigma_prob"] - anal_full["sigma_prob"])
        bdry_color = "#ff4d6d" if sigma_diff > 0.02 else "#06ffa5"
        bdry_text = "&#9888; boundary effect" if sigma_diff > 0.02 else "&#10003; methods agree"
        mc_pct = f'{mc_full["prob"]:.0%}'
        mc_sig = f'{mc_full["sigma_prob"]:.0%}'
        mc_lo = f'{mc_full["ci_95_lower"]:.0%}'
        mc_hi = f'{mc_full["ci_95_upper"]:.0%}'
        an_pct = f'{anal_full["prob"]:.0%}'
        an_sig = f'{anal_full["sigma_prob"]:.0%}'
        an_lo = f'{anal_full["lower_bound"]:.0%}'
        an_hi = f'{anal_full["upper_bound"]:.0%}'
        rev_stake = f'${mc_full["prob"] * ctx.clv * ctx.horizon:,.0f}'
        e0_v = f'{cust_row["E0"]:.3f}'
        eeq_v = f'{cust_row["E_eq"]:.3f}'
        gam_v = f'{cust_row["gamma"]:.3f}'
        tau_v = f'{cust_row["tau"]:.1f}'
        sd_fmt = f'{sigma_diff:.4f}'

        st.markdown(f"""
        <div class='card' style='border-left: 3px solid {risk_color};'>
          <div class='small'>CUSTOMER</div>
          <div style='font-family:Syne; font-size:18px; font-weight:800; color:white;'>{selected_id}</div>
          <div style='margin-top:16px; display:flex; flex-direction:column; gap:12px;'>
            <div>
              <div class='small'>Monte Carlo (10k samples)</div>
              <div style='font-size:20px; font-weight:700; color:{risk_color};'>{mc_pct} &plusmn; {mc_sig}</div>
              <div class='small'>95% CI [{mc_lo} &ndash; {mc_hi}]</div>
            </div>
            <div>
              <div class='small'>Analytical (closed-form)</div>
              <div style='font-size:18px; font-weight:700; color:#4cc9f0;'>{an_pct} &plusmn; {an_sig}</div>
              <div class='small'>CI [{an_lo} &ndash; {an_hi}]</div>
            </div>
            <div>
              <div class='small'>Delta-sigma (divergence)</div>
              <div style='font-size:16px; font-weight:700; color:{bdry_color};'>
                {sd_fmt} &nbsp; {bdry_text}
              </div>
            </div>
            <div>
              <div class='small'>Physical profile</div>
              <div style='font-size:13px; color:white;'>
                E0={e0_v} &middot; E_eq={eeq_v}<br>
                gamma={gam_v} &middot; tau={tau_v}m
              </div>
            </div>
            <div>
              <div class='small'>Revenue at stake ({ctx.horizon}m)</div>
              <div style='font-size:18px; font-weight:700; color:#f8961e;'>{rev_stake}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        # Rebuild the MC sample distribution for plotting
        rng_plot = np.random.default_rng(42)
        E0_s_p = np.clip(rng_plot.normal(cust_row["E0"], 0.05, 10_000), 0, 1)
        g_s_p = np.clip(rng_plot.normal(cust_row["gamma"], 0.02, 10_000), 1e-6, None)
        Eeq_s_p = np.clip(rng_plot.normal(cust_row["E_eq"], 0.03, 10_000), 0, 1)
        exp_gt_p = np.exp(-g_s_p * ctx.horizon)
        E_t_p = Eeq_s_p + (E0_s_p - Eeq_s_p) * exp_gt_p
        prob_dist  = np.clip(1 - E_t_p, 0, 1)

        # P(churn) distribution: violin + KDE + analytical overlay 
        _kde_p = _gkde(prob_dist, bw_method=0.12)
        _xp = np.linspace(0, 1, 250)
        _yp = _kde_p(_xp)
        anal_pdf = norm.pdf(_xp, anal_full["prob"], anal_full["sigma_prob"])

        fig_dist = go.Figure()
        # Background danger zone
        fig_dist.add_vrect(x0=0.5, x1=1.0, fillcolor="rgba(255,77,109,0.06)", line_width=0)

        # Violin
        fig_dist.add_trace(go.Violin(
            x=prob_dist, orientation="h", side="positive",
            fillcolor="rgba(76,201,240,0.12)",
            line=dict(color="#4cc9f0", width=1.5),
            points=False, showlegend=False,
            box=dict(visible=True, fillcolor="rgba(76,201,240,0.35)", line=dict(color="#4cc9f0")),
            meanline=dict(visible=True, color="#ffbe0b"),))
        # MC KDE
        fig_dist.add_trace(go.Scatter(
            x=_xp, y=_yp / _yp.max() * 0.45,
            mode="lines", name="MC density (KDE)",
            line=dict(color="#4cc9f0", width=2),
            fill="tozeroy", fillcolor="rgba(76,201,240,0.07)",))
        # Analytical overlay
        fig_dist.add_trace(go.Scatter(
            x=_xp, y=anal_pdf / anal_pdf.max() * 0.45,
            mode="lines", name="Analytical (Gaussian)",
            line=dict(color="#f8961e", width=2, dash="dash"),))
        for val, color, label in [
            (mc_full["prob"], "#f8961e", f"MC median {mc_full['prob']:.0%}"),
            (mc_full["ci_95_lower"], "#888888", "95% CI lower"),
            (mc_full["ci_95_upper"], "#888888", "95% CI upper"),
            (0.5, "#ff4d6d", "Churn threshold"),]:
            fig_dist.add_vline(x=val,
                                line_color=color, line_width=1.5,
                                line_dash="solid" if val == mc_full["prob"] else "dot",
                                annotation_text=label,
                                annotation_font=dict(color=color, size=9))
        fig_dist.update_layout(**ctx.plotly_dark(), height=360,
            title=f"P(churn) distribution - {selected_id}",
            showlegend=True,
            legend=dict(x=0.01, y=0.99, **ctx.plotly_legend_style()))
        fig_dist.update_xaxes(**ctx.plotly_axis_style(), title_text="P(churn)", range=[0, 1])
        fig_dist.update_yaxes(visible=False)
        st.plotly_chart(fig_dist, use_container_width=True, key="t5_dist")

    st.markdown("""
    <div class='card card-accent' style='margin-top:8px;'>
    <span class='small'>
    <b>How to read this chart</b><br>
    The histogram is the empirical distribution of P(churn) across 10,000 samples of E0, γ, E_eq
    drawn from their uncertainty ranges. The dashed blue curve is the Gaussian approximation
    used by the analytical method. When they differ significantly (skewed histogram, boundary effect),
    trust the MC percentiles — not the ±σ interval.
    </span>
    </div>
    """, unsafe_allow_html=True)