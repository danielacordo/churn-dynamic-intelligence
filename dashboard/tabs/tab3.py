import numpy as np
import streamlit as st
import plotly.graph_objects as go
from dashboard.tabs.context import DashboardContext
from src.visualization import plot_waterfall_roi


def render(ctx: DashboardContext) -> None:
    """Render the Campaign Simulation tab"""

    # Bridge: inherit exact segment from dashboard 
    loaded_seg = st.session_state.get("selected_segment", "All (no filter)")
    loaded_risk = st.session_state.get("selected_risk", "All (no filter)")

    is_filtered = loaded_seg != "All (no filter)"

    if is_filtered:
        df_sim = ctx.df_f[ctx.df_f["segment"] == loaded_seg].copy()
        risk_tag  = loaded_risk
        color_tag = ctx.RISK_COLORS.get(risk_tag, "#f8961e")
        st.markdown(f"""
        <div class='card card-accent' style='padding: 12px 18px; margin-bottom: 16px;'>
          <div style='display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;'>
            <div>
              <span class='small'>Segment loaded from Dashboard:</span>
              <b style='color:white; margin-left:6px;'>{loaded_seg}</b>
              <span class='badge' style='margin-left:8px; background:rgba(248,150,30,0.15);
                    color:{color_tag}; border:1px solid {color_tag}44;'>{risk_tag}</span>
            </div>
            <div style='text-align:right; font-size:12px;'>
              <span style='color:#06ffa5; font-weight:700;'>{len(df_sim)} customers</span>
              <span style='color:#9999c0; margin:0 6px;'>·</span>
              <span style='color:#f8961e; font-weight:700;'>${df_sim["revenue_at_risk"].sum():,.0f} at stake</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_sim = ctx.df_f.copy()

    seg_label = loaded_seg if is_filtered else "All segments"
    st.markdown(f"#### Campaign Simulation - {seg_label}")
    st.markdown(f"<p class='small'>Parameters: CLV=${ctx.clv}/m · Retention cost=${ctx.ret_cost} · Horizon={ctx.horizon}m · Threshold={ctx.threshold:.0%}</p>", unsafe_allow_html=True)

    if st.button("← Back to all segments", disabled=not is_filtered):
        st.session_state.selected_segment = "All (no filter)"
        st.session_state.selected_risk = "All (no filter)"
        st.rerun()

    sim_df, totals = ctx.compute_campaign_metrics(df_sim, ctx.clv, ctx.ret_cost, ctx.horizon)

    # Top-line summary
    n_target = len(df_sim)
    pct_base = n_target / len(ctx.df_f) * 100 if len(ctx.df_f) > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers saved (est.)", f"{totals['saved']:.0f}")
    c2.metric("Revenue recovered", f"${totals['revenue']:,.0f}")
    c3.metric("Campaign investment", f"${totals['cost']:,.0f}")
    c4.metric("Net ROI", f"${totals['roi']:+,.0f}", f"{totals['roi_pct']:+.0f}%")

    # Target summary box 
    avg_prob = df_sim["prob_churn"].mean() if len(df_sim) > 0 else 0
    avg_sigma = df_sim["sigma_prob"].mean() if len(df_sim) > 0 else 0
    avg_lb = df_sim["lb"].mean() if len(df_sim) > 0 else 0
    avg_ub = df_sim["ub"].mean() if len(df_sim) > 0 else 0

    _cv1 = "#06ffa5" if totals["roi"] >= 0 else "#ff4d6d"
    st.markdown(f"""
    <div class='card card-green' style='margin: 16px 0;'>
      <div style='display: flex; gap: 40px; flex-wrap: wrap; align-items: center;'>
        <div>
          <div class='small'>TARGET</div>
          <div style='font-family:Syne; font-size:19px; font-weight:700; color:white; line-height:1.5; letter-spacing:0.3px;'>
            Top {pct_base:.0f}% highest-risk customers
          </div>
          <div class='small' style='margin-top:4px;'>{n_target:,} customers · {seg_label}</div>
        </div>
        <div>
          <div class='small'>AVG P(CHURN)</div>
          <div style='font-family:Syne; font-size:20px; font-weight:700; color:#f8961e; line-height:1.5; letter-spacing:0.3px;'>
            {avg_prob:.0%} <span style='font-size:14px; color:#9999c0;'>± {avg_sigma:.0%}</span>
          </div>
          <div class='small' style='margin-top:4px;'>95% CI [{avg_lb:.0%} – {avg_ub:.0%}]</div>
        </div>
        <div>
          <div class='small'>EXPECTED RECOVERY</div>
          <div style='font-family:Syne; font-size:20px; font-weight:700; color:#06ffa5; line-height:1.5; letter-spacing:0.3px;'>
            ${totals["revenue"]:,.0f}
          </div>
          <div class='small' style='margin-top:4px;'>${totals["cost"]:,.0f} campaign cost</div>
        </div>
        <div>
          <div class='small'>ROI</div>
          <div style='font-family:Syne; font-size:20px; font-weight:700; color:{_cv1}; line-height:1.5; letter-spacing:0.3px;'>
            {totals["roi_pct"]:+.0f}%
          </div>
          <div class='small' style='margin-top:4px;'>Net ${totals["roi"]:+,.0f}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Best / Base / Worst case scenarios 
    st.markdown("#### Confidence range - best vs. base vs. worst case")
    st.markdown("<p class='small'>Uncertainty in model predictions (± σ) translates directly into a range of campaign outcomes.</p>",
                unsafe_allow_html=True)

    scenarios = [
        ("Worst case", "scenario-worst", "#ff4d6d", avg_lb, 0.15),
        ("Base case", "scenario-base", "#f8961e", avg_prob, 0.30),
        ("Best case", "scenario-best", "#06ffa5", avg_ub, 0.45),]

    s_cols = st.columns(3)
    for col, (label, css_cls, color, prob_s, rate_s) in zip(s_cols, scenarios):
        rev_s  = n_target * prob_s * rate_s * ctx.clv * ctx.horizon
        cost_s = totals["cost"]
        roi_s  = rev_s - cost_s
        roi_pct_s = (roi_s / cost_s * 100) if cost_s > 0 else 0
        saved_s = n_target * prob_s * rate_s
        with col:
            _cv2 = "#06ffa5" if roi_s >= 0 else "#ff4d6d"
            st.markdown(f"""
            <div class='card {css_cls}'>
              <div style='font-family:Syne; font-size:13px; font-weight:700; color:{color};
                          letter-spacing:1px; text-transform:uppercase; margin-bottom:12px;'>
                {label}
              </div>
              <div style='display: flex; flex-direction: column; gap: 10px;'>
                <div>
                  <div class='small'>Assumed P(churn)</div>
                  <div style='font-size:18px; font-weight:700; color:white;'>{prob_s:.0%}</div>
                </div>
                <div>
                  <div class='small'>Retention rate</div>
                  <div style='font-size:18px; font-weight:700; color:white;'>{rate_s:.0%}</div>
                </div>
                <div>
                  <div class='small'>Customers saved</div>
                  <div style='font-size:18px; font-weight:700; color:white;'>{saved_s:.0f}</div>
                </div>
                <div>
                  <div class='small'>Revenue recovered</div>
                  <div style='font-size:18px; font-weight:700; color:{color};'>${rev_s:,.0f}</div>
                </div>
                <div>
                  <div class='small'>Net ROI</div>
                  <div style='font-size:20px; font-weight:800; color:{_cv2};'>
                    {roi_pct_s:+.0f}%
                  </div>
                  <div class='small'>${roi_s:+,.0f}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Range bar visualization
    worst_rev = n_target * avg_lb * 0.15 * ctx.clv * ctx.horizon
    base_rev = n_target * avg_prob * 0.30 * ctx.clv * ctx.horizon
    best_rev = n_target * avg_ub * 0.45 * ctx.clv * ctx.horizon

    fig_range = go.Figure()
    fig_range.add_trace(go.Bar(
        x=["Worst case", "Base case", "Best case"],
        y=[worst_rev - totals["cost"], base_rev - totals["cost"], best_rev - totals["cost"]],
        marker_color=["#ff4d6d", "#f8961e", "#06ffa5"],
        marker_line_width=0,
        text=[f"${v:+,.0f}" for v in
              [worst_rev - totals["cost"], base_rev - totals["cost"], best_rev - totals["cost"]]],
        textposition="outside",
        textfont=dict(color="white", size=12),))
    fig_range.add_hline(y=0, line_color="#9999c0", line_width=1)
    fig_range.update_layout(**ctx.plotly_dark(), height=300, title="Net ROI range across scenarios ($)", yaxis_title="Net ROI ($)", xaxis_title="")
    st.plotly_chart(fig_range, use_container_width=True, key="t3_range")

    st.divider()

    # Segment-level breakdown 
    st.markdown("#### Breakdown by segment")

    #  Waterfall: revenue -> recovered -> cost -> net 
    fig_wf = plot_waterfall_roi(
        revenue_at_risk = totals["revenue"] / 0.30 * ctx.threshold
                            if totals["revenue"] > 0 else totals["cost"],
        revenue_recovered = totals["revenue"],
        campaign_cost = totals["cost"],
        net_roi = totals["roi"],
        title = "Campaign ROI Waterfall - all segments",)
    st.plotly_chart(fig_wf, use_container_width=True, key="t3_wf")

    # Per-segment bar (keep for detail) 
    labels = list(sim_df["segment"]) + ["NET ROI"]
    values = list(sim_df["roi"]) + [totals["roi"]]
    colors = ["#06ffa5" if v >= 0 else "#ff4d6d" for v in values[:-1]] + ["#f8961e"]
    fig_seg_roi = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors, marker_line_width=0,
        text=[f"${v:+,.0f}" for v in values],
        textposition="outside", textfont=dict(color="white", size=11),))
    fig_seg_roi.add_hline(y=0, line_color="#9999c0", line_width=1)
    fig_seg_roi.update_layout(**ctx.plotly_dark(), height=320, title="Net ROI by segment ($)", yaxis_title="USD", xaxis_title="")
    fig_seg_roi.update_xaxes(**ctx.plotly_axis_style(), tickangle=-20)
    st.plotly_chart(fig_seg_roi, use_container_width=True, key="t3_seg_roi")

    # HTML table, st.dataframe renders empty with custom CSS themes in Streamlit 1.32+
    seg_cols = ["Segment", "Customers", "Success rate", "Saved (est.)", "Revenue rec. ($)", "Campaign cost ($)", "Net ROI ($)", "ROI %"]
    header_html = "".join(
        f"<th style='padding:8px 12px; text-align:left; color:#9999c0; font-size:11px; "
        f"letter-spacing:1px; text-transform:uppercase; border-bottom:1px solid #1e1e3a; "
        f"white-space:nowrap;'>{c}</th>"
        for c in seg_cols)
    rows_html = ""
    for _, row in sim_df.iterrows():
        roi_raw = row["roi"]
        roi_color = "#06ffa5" if roi_raw >= 0 else "#ff4d6d"
        roi_pct_raw = row["roi_pct"]
        cells = (
            f"<td style='padding:8px 12px; color:#dcdcf0; font-size:12px; border-bottom:1px solid #1e1e3a;'>{row['segment']}</td>"
            f"<td style='padding:8px 12px; color:#dcdcf0; font-size:12px; border-bottom:1px solid #1e1e3a;'>{int(row['n'])}</td>"
            f"<td style='padding:8px 12px; color:#dcdcf0; font-size:12px; border-bottom:1px solid #1e1e3a;'>{row['success_rate']:.0%}</td>"
            f"<td style='padding:8px 12px; color:#dcdcf0; font-size:12px; border-bottom:1px solid #1e1e3a;'>{row['saved']:.1f}</td>"
            f"<td style='padding:8px 12px; color:#dcdcf0; font-size:12px; border-bottom:1px solid #1e1e3a;'>${row['revenue']:,.0f}</td>"
            f"<td style='padding:8px 12px; color:#dcdcf0; font-size:12px; border-bottom:1px solid #1e1e3a;'>${row['cost']:,.0f}</td>"
            f"<td style='padding:8px 12px; color:{roi_color}; font-size:12px; font-weight:600; border-bottom:1px solid #1e1e3a;'>${roi_raw:+,.0f}</td>"
            f"<td style='padding:8px 12px; color:{roi_color}; font-size:12px; font-weight:600; border-bottom:1px solid #1e1e3a;'>{roi_pct_raw:+.0f}%</td>")
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <div style='overflow-x:auto; border:1px solid #1e1e3a; border-radius:8px; margin-bottom:16px;'>
      <table style='width:100%; border-collapse:collapse; background:#0f0f1e;'>
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # Structural risk warning
    struct_roi = sim_df[sim_df["segment"] == "Structural Risk"]["roi"].sum() if "Structural Risk" in sim_df["segment"].values else 0
    struct_cost = sim_df[sim_df["segment"] == "Structural Risk"]["cost"].sum() if "Structural Risk" in sim_df["segment"].values else 0
    if struct_cost > 0:
        st.markdown(f"""
        <div class='card card-churn' style='margin-top:16px;'>
        <b> Structural Risk customers show negative ROI (${struct_roi:,.0f})</b><br>
        <span class='small'>
        These customers have E_eq &lt; E_critical — their structural equilibrium is below the loyalty ctx.threshold.
        A campaign won't retain them. They need an offer redesign.<br>
        The model prevents spending ${abs(struct_cost):,.0f} on ineffective interventions.
        </span>
        </div>
        """, unsafe_allow_html=True)

    # Sensitivity
    st.divider()
    st.markdown("#### ROI sensitivity - retention rate vs. net ROI")

    rates = np.linspace(0.05, 0.60, 50)
    rois_sens = []
    for r in rates:
        _, t = ctx.compute_campaign_metrics(df_sim, ctx.clv, ctx.ret_cost, ctx.horizon, success_rate_override=r)
        rois_sens.append(t["roi"])

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=rates*100, y=rois_sens, mode="lines", line=dict(color="#f8961e", width=2.5), fill="tozeroy", fillcolor="rgba(248,150,30,0.08)",))
    fig_sens.add_hline(y=0, line_color="#9999c0", line_width=1)
    fig_sens.add_vline(x=30, line_dash="dot", line_color="#06ffa5", annotation_text="Base assumption (30%)", annotation_font_color="#06ffa5")
    breakeven_rate = next((r*100 for r, roi in zip(rates, rois_sens) if roi >= 0), None)
    if breakeven_rate:
        fig_sens.add_vline(x=breakeven_rate, line_dash="dot", line_color="#9b5de5", annotation_text=f"Break-even ({breakeven_rate:.0f}%)", annotation_font_color="#9b5de5")
    fig_sens.update_layout(**ctx.plotly_dark(), height=300, xaxis_title="Retention success rate (%)", yaxis_title="Net ROI ($)", title="")
    st.plotly_chart(fig_sens, use_container_width=True, key="t3_sens")
