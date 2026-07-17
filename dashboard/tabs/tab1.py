import streamlit as st
import plotly.graph_objects as go
from dashboard.business import executive_kpis
from dashboard.tabs.context import DashboardContext
from src.visualization import (plot_phase_space_plotly, plot_waterfall_roi, plot_customer_funnel,)


def render(ctx: DashboardContext) -> None:
    """Render the Executive Dashboard tab"""

    kpis = executive_kpis(ctx.df_f, clv=ctx.clv, ret_cost=ctx.ret_cost, horizon=ctx.horizon)
    n_high = kpis["n_high"]
    rev_at_risk = kpis["rev_at_risk"]
    rev_recovered = kpis["rev_recovered"]
    campaign_cost = kpis["campaign_cost"]
    net_roi = kpis["net_roi"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers analyzed", f"{len(ctx.df_f):,}")
    c2.metric("High risk", f"{n_high:,}", f"{n_high/max(len(ctx.df_f),1):.1%} of base")
    c3.metric("Revenue at risk", f"${rev_at_risk:,.0f}", f"{ctx.horizon}m window")
    c4.metric("Revenue recoverable",f"${rev_recovered:,.0f}", "30% retention rate")
    c5.metric("Net ROI", f"${net_roi:+,.0f}", f"${campaign_cost:,.0f} invested")

    st.divider()

    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.markdown("#### Risk distribution")
        risk_counts = ctx.df_f["risk_level"].value_counts().reindex(["HIGH","MEDIUM","UNCERTAIN","LOW"], fill_value=0)
        fig_risk = go.Figure(go.Bar(
            x=risk_counts.index, y=risk_counts.values,
            marker_color=[ctx.RISK_COLORS[r] for r in risk_counts.index],
            marker_line_width=0, text=risk_counts.values, textposition="outside",
            textfont=dict(color="white", size=13),))
        fig_risk.update_layout(**ctx.plotly_dark(), height=280, xaxis_title="", yaxis_title="Customers")

        # Bridge: send specific segment to Campaign Simulator 
        st.plotly_chart(fig_risk, use_container_width=True, key="t1_risk")

        all_segments = sorted(ctx.df_f["segment"].unique().tolist())
        bridge_options = ["All (no filter)"] + all_segments

        current_idx = 0
        if st.session_state.selected_segment in bridge_options:
            current_idx = bridge_options.index(st.session_state.selected_segment)

        bridge_seg = st.selectbox(
            "Send segment to Campaign Simulator",
            bridge_options, index=current_idx, key="bridge_seg_select",
            help="Select any specific segment - the simulator will filter to exactly those customers",)

        if bridge_seg != st.session_state.selected_segment:
            st.session_state.selected_segment = bridge_seg
            # also keep risk_level in sync for Tab 4 physics pre-load
            if bridge_seg == "All (no filter)":
                st.session_state.selected_risk = "All (no filter)"
            else:
                # derive risk level from segment name
                if bridge_seg.startswith("High"):
                    st.session_state.selected_risk = "HIGH"
                elif bridge_seg.startswith("Medium") or bridge_seg == "Structural Risk":
                    st.session_state.selected_risk = "MEDIUM"
                elif bridge_seg == "Uncertain":
                    st.session_state.selected_risk = "UNCERTAIN"
                else:
                    st.session_state.selected_risk = "LOW"

        if bridge_seg != "All (no filter)":
            df_bridge = ctx.df_f[ctx.df_f["segment"] == bridge_seg]
            n_bridge = len(df_bridge)
            rev_bridge = df_bridge["revenue_at_risk"].sum()
            avg_p = df_bridge["prob_churn"].mean()
            avg_s = df_bridge["sigma_prob"].mean()
            risk_tag = st.session_state.selected_risk
            color_tag  = ctx.RISK_COLORS.get(risk_tag, "#f8961e")
            st.markdown(f"""
            <div class='card card-accent' style='padding:12px 16px; margin-top:8px;'>
              <div style='display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;'>
                <div>
                  <span class='small'>Loaded into simulator →</span>
                  <b style='color:white; margin-left:6px;'>{bridge_seg}</b>
                  <span class='badge' style='margin-left:8px; background:rgba(248,150,30,0.15);
                        color:{color_tag}; border:1px solid {color_tag}44;'>{risk_tag}</span>
                </div>
                <div style='text-align:right;'>
                  <span style='color:#06ffa5; font-weight:700;'>{n_bridge} customers</span>
                  <span style='color:#9999c0; margin:0 6px;'>·</span>
                  <span style='color:#f8961e; font-weight:700;'>${rev_bridge:,.0f} at stake</span>
                  <span style='color:#9999c0; margin:0 6px;'>·</span>
                  <span style='color:white;'>avg P={avg_p:.0%} ± {avg_s:.0%}</span>
                </div>
              </div>
              <div class='small' style='margin-top:8px; color:#9999c0;'>
                → Switch to the <b style='color:white;'>Campaign Simulation</b> tab to see ROI, best/worst case, and breakdown
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown("#### Revenue at risk by segment")
        seg_rev = ctx.df_f.groupby("segment")["revenue_at_risk"].sum().sort_values(ascending=True).tail(6)
        fig_seg = go.Figure(go.Bar(
            y=seg_rev.index, x=seg_rev.values, orientation="h",
            marker_color="#f8961e", marker_line_width=0,
            text=[f"${v:,.0f}" for v in seg_rev.values],
            textposition="outside", textfont=dict(color="white", size=11),))
        fig_seg.update_layout(**ctx.plotly_dark(), height=280, xaxis_title="USD", yaxis_title="")
        st.plotly_chart(fig_seg, use_container_width=True, key="t1_seg")

    # ROI Waterfall 
    st.markdown("#### ROI decomposition")
    _high_mask = ctx.df_f["risk_level"] == "HIGH"
    _n_high_wf = int(_high_mask.sum())
    _rev_risk_wf = float((ctx.df_f.loc[_high_mask, "prob_churn"] * ctx.clv * ctx.horizon).sum())
    _saved_wf = _n_high_wf * 0.30
    _rev_rec_wf  = _saved_wf * ctx.clv * ctx.horizon
    _cost_wf = _n_high_wf * ctx.ret_cost
    _roi_wf = _rev_rec_wf - _cost_wf

    fig_wf = plot_waterfall_roi(
        revenue_at_risk = _rev_risk_wf, revenue_recovered = _rev_rec_wf,
        campaign_cost = _cost_wf, net_roi = _roi_wf,
        title = "Campaign ROI Waterfall (HIGH risk segment)",)
    st.plotly_chart(fig_wf, use_container_width=True, key="t1_wf")

    # Customer Funnel 
    st.markdown("#### Retention campaign funnel")

    # Campaign reach: fraction of flagged customers actually contacted.
    _CAMPAIGN_REACH = 0.85
    _n_total_fn = len(ctx.df_f)
    _n_flagged_fn = int((ctx.df_f["risk_level"].isin(["HIGH","MEDIUM"])).sum())
    _n_interv_fn  = int(_n_flagged_fn * _CAMPAIGN_REACH)
    _n_ret_fn = int(_n_flagged_fn * 0.30 * _CAMPAIGN_REACH)
    fig_funnel = plot_customer_funnel(
        n_total = _n_total_fn,
        n_flagged = _n_flagged_fn,
        n_intervened = _n_interv_fn,
        n_retained = _n_ret_fn,)
    st.plotly_chart(fig_funnel, use_container_width=True, key="t1_funnel")

    # Phase space 
    st.markdown("#### Phase space - E₀ vs E_eq")
    st.markdown("<p class='small'>KDE density contours show where customer mass concentrates. "
                "Quadrants show structural risk profile. Red zone = churn territory. "
                "Hover any point for customer details.</p>", unsafe_allow_html=True)

    # Build df with Churn_bin for KDE (use actual churn if available, else prob_churn > 0.5)
    _df_phase = ctx.df_f.copy()
    if "churn" in _df_phase.columns:
        _df_phase["Churn_bin"] = _df_phase["churn"]
    else:
        _df_phase["Churn_bin"] = (_df_phase["prob_churn"] > 0.5).astype(int)

    fig_phase = plot_phase_space_plotly(_df_phase, E_critical=ctx.E_CRITICAL)

    # Overlay per-risk scatter with hover info on top of KDE
    for risk in ["LOW", "MEDIUM", "UNCERTAIN", "HIGH"]:
        sub = ctx.df_f[ctx.df_f["risk_level"] == risk]
        if len(sub) == 0:
            continue
        fig_phase.add_trace(go.Scatter(
            x=sub["E_eq"], y=sub["E0"],
            mode="markers", name=risk,
            marker=dict(color=ctx.RISK_COLORS[risk], size=4, opacity=0.55,
                        line=dict(width=0)),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "E0=%{y:.3f}  E_eq=%{x:.3f}<br>"
                "P(churn)=%{customdata[1]:.0%} ± %{customdata[2]:.0%}<br>"
                "<i>%{customdata[3]}</i><extra></extra>" ),
            customdata=sub[["customerID","prob_churn","sigma_prob","segment"]].values,
            showlegend=True,))
    st.plotly_chart(fig_phase, use_container_width=True, key="t1_phase")

    st.markdown("""
    <div class='card card-accent'>
    <b>How to read this chart</b><br><br>
    <span class='small'>
    → <b>X axis (E_eq)</b>: structural loyalty. Customers left of the red line will churn even without negative events.<br>
    → <b>Y axis (E₀)</b>: current engagement. Customers below the line are already in the churn zone.<br>
    → <b>Bottom-left quadrant</b>: highest priority — both current state and structural tendency point to churn.<br>
    → <b>Select a segment above</b> to pre-load it into the Campaign Simulator.
    </span>
    </div>
    """, unsafe_allow_html=True)
