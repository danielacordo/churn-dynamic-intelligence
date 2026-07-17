import streamlit as st
import plotly.graph_objects as go
from dashboard.tabs.context import DashboardContext


def render(ctx: DashboardContext) -> None:
    """Render the Customer Prioritization tab """

    st.markdown("# Top customers to intervene")
    st.markdown("<p class='small'>Sorted by churn probability. Each customer comes with a physical diagnosis and a specific recommended action.</p>", unsafe_allow_html=True)

    n_show = st.slider("Customers to display", 5, 50, 15)

    top = (ctx.df_f[ctx.df_f["risk_level"].isin(["HIGH","MEDIUM"])].sort_values("prob_churn", ascending=False).head(n_show).reset_index(drop=True))
    top.index += 1

    for i, row in top.iterrows():
        risk = row["risk_level"]
        prob = row["prob_churn"]
        sigma = row["sigma_prob"]
        lb = row["lb"]
        ub = row["ub"]
        seg = row["segment"]
        act = row["action"]
        tau = row["tau"]
        E_eq = row["E_eq"]
        E0 = row["E0"]
        rev = row["revenue_at_risk"]
        saved = rev * 0.30

        bar = int(prob * 20)
        bar_html = f"{'█' * bar}{'░' * (20-bar)}"
        badge_cls = ctx.RISK_BADGE.get(risk, "badge-medium")
        color = ctx.RISK_COLORS.get(risk, "#f8961e")

        _cv1 = "#ff4d6d" if E_eq < ctx.E_CRITICAL else "#06ffa5"
        _cv2 = "below critical" if E_eq < ctx.E_CRITICAL else "stable"
        st.markdown(f"""
        <div class='card' style='border-left: 3px solid {color}; margin-bottom: 10px;'>
          <div style='display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;'>
            <div>
              <span style='font-size:11px; color:#9999c0;'>#{i:02d}</span>
              <span style='font-family:Syne; font-size:15px; font-weight:700; color:white; margin-left:8px;'>{row["customerID"]}</span>
              <span class='badge {badge_cls}' style='margin-left:10px;'>{risk}</span>
            </div>
            <div style='text-align:right;'>
              <span style='font-size:11px; color:#9999c0;'>Revenue at risk</span>
              <span style='font-family:Syne; font-size:16px; font-weight:700; color:#f8961e; margin-left:8px;'>${rev:,.0f}</span>
              <span style='font-size:11px; color:#06ffa5; margin-left:8px;'>→ save ~${saved:,.0f}</span>
            </div>
          </div>
          <div style='margin: 10px 0 6px 0; font-size:13px; color:{color}; font-family: JetBrains Mono;'>
            {bar_html} {prob:.0%} ± {sigma:.0%} &nbsp;&nbsp;
            <span style='color:#9999c0; font-size:11px;'>95% CI [{lb:.0%} – {ub:.0%}]</span>
          </div>
          <div style='display:flex; gap:32px; flex-wrap:wrap; margin-top:8px;'>
            <div><span class='small'>Segment</span><br><span style='font-size:13px; color:white;'>{seg}</span></div>
            <div><span class='small'>τ (relaxation)</span><br><span style='font-size:13px; color:white;'>{tau:.1f} months</span></div>
            <div><span class='small'>E_eq (structural)</span><br><span style='font-size:13px; color:{_cv1};'>{E_eq:.2f} {_cv2}</span></div>
            <div><span class='small'>E₀ (current)</span><br><span style='font-size:13px; color:white;'>{E0:.2f}</span></div>
          </div>
          <div style='margin-top:10px; padding:8px 12px; background:#080810; border-radius:6px; font-size:12px; color:#f8961e;'>
            ▶ {act}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Uncertainty scatter
    st.divider()
    st.markdown("#### Prediction uncertainty - who to act on vs. who needs more data")
    fig_unc = go.Figure()
    for risk in ["HIGH","MEDIUM","UNCERTAIN","LOW"]:
        sub = ctx.df_f[ctx.df_f["risk_level"] == risk]
        if len(sub) == 0:
            continue
        fig_unc.add_trace(go.Scatter(
            x=sub["prob_churn"], y=sub["sigma_prob"], mode="markers", name=risk,
            marker=dict(color=ctx.RISK_COLORS[risk], size=6, opacity=0.7, line=dict(width=0)),
            hovertemplate="<b>%{customdata}</b><br>P=%{x:.0%} ± %{y:.0%}<extra></extra>",
            customdata=sub["customerID"],))
    fig_unc.add_hline(y=ctx.SIGMA_THRESH, line_dash="dot", line_color="#9b5de5", opacity=0.7,
                       annotation_text="σ ctx.threshold (UNCERTAIN above)", annotation_font_color="#9b5de5")
    fig_unc.add_vline(x=ctx.threshold, line_dash="dot", line_color="#f8961e", opacity=0.7,
                       annotation_text=f"Risk ctx.threshold ({ctx.threshold:.0%})", annotation_font_color="#f8961e")
    fig_unc.update_layout(**ctx.plotly_dark(), height=340,
                           xaxis_title="P(churn)", yaxis_title="σ (uncertainty)",
                           xaxis=dict(tickformat=".0%", gridcolor="#1e1e3a"),
                           yaxis=dict(gridcolor="#1e1e3a"))
    st.plotly_chart(fig_unc, use_container_width=True, key="t2_unc")

    st.markdown("""
    <div class='card card-accent'>
    <span class='small'>
    <b>Bottom-right</b>: high probability + low uncertainty → <b>act now</b><br>
    <b>Top-center</b>: medium probability + high uncertainty → <b>collect more data first</b><br>
    Intervening on UNCERTAIN customers wastes budget. This model separates them explicitly.
    </span>
    </div>
    """, unsafe_allow_html=True)
