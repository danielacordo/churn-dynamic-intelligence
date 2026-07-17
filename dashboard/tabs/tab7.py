import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard.business import ab_test_break_even_analysis
from dashboard.tabs.context import DashboardContext
from src.ab_test import (
    ABTestConfig,
    ab_test_report,
    analyse_experiment,
    required_sample_size,
    sequential_analysis,
    simulate_experiment,
    sweep_true_success_rates,)


def render(ctx: DashboardContext) -> None:
    """Render the A/B Test Validator tab"""

    st.markdown("## A/B Test Validator")
    st.markdown("""
    <div class='card card-accent'>
    The ROI estimate depends critically on one assumption: <strong>30% of at-risk customers
    are retained by the campaign</strong>. This tab designs the experiment that would validate
    (or revise) that number - and simulates what the results would look like.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar-linked controls 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        assumed_sr = st.slider(
            "Assumed success rate",
            min_value=0.05, max_value=0.60, value=0.30, step=0.05,
            help="The campaign effectiveness assumption driving the ROI estimate.",)
    with col2:
        baseline_cr = st.slider(
            "Baseline churn rate (high-risk)",
            min_value=0.30, max_value=0.90,
            value=0.65, step=0.05,
            help="Expected churn rate in the control group (flagged but not contacted).",)
    with col3:
        true_sr_sim = st.slider(
            "True success rate (simulation)",
            min_value=0.00, max_value=0.60, value=0.30, step=0.05,
            help="The real rate in the simulation - vary to test 'what if the campaign underdelivers?'",)
    with col4:
        _ = st.slider(
            "Campaign reach (%)",
            min_value=0.40, max_value=1.00,
            value=0.85, step=0.05,
            help="Fraction of flagged customers that actually receive the campaign "
                 "(email open rate, contact success, etc.).",)

    cfg = ABTestConfig(
        baseline_churn_rate=baseline_cr,
        assumed_success_rate=assumed_sr,
        alpha=0.05,
        power=0.80,
        seed=42,)

    ss = required_sample_size(cfg)

    # Sample size panel 
    st.markdown("### Experiment Design")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers per arm", f"{ss['n_per_group']:,}")
    c2.metric("Total needed", f"{ss['n_total']:,}")
    c3.metric("Runtime estimate", f"{ss['weeks_at_1k_weekly']:.1f} wks", help="At 1,000 high-risk customers flagged per week.")
    c4.metric("Detectable effect", f"{ss['effect_size_abs']:.0%}", help="Minimum absolute reduction in churn rate this test can detect.")

    st.markdown(f"""
    <div class='card' style='font-size:13px; line-height:1.8;'>
    <b>How to read this:</b> With <b>{ss['n_per_group']:,} customers per arm</b>, the test has
    80% power to detect a <b>{ss['effect_size_abs']:.0%} absolute reduction</b> in churn rate
    (from {baseline_cr:.0%} to {cfg.treated_churn_rate():.0%}), using α=0.05.
    At 1,000 high-risk customers per week, this requires
    <b>{ss['weeks_at_1k_weekly']:.1f} weeks</b> of data collection.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Break-even analysis 
    st.markdown("### Break-even: what success rate do we actually need?")
    n_high_risk = int((ctx.df_f["risk_level"] == "HIGH").sum())
    be = ab_test_break_even_analysis(n_high_risk, clv=ctx.clv, ret_cost=ctx.ret_cost, horizon=ctx.horizon)

    c1, c2, c3 = st.columns(3)
    c1.metric("Break-even success rate", f"{be['break_even_rate']:.1%}",
              help="Minimum retention rate for the campaign to be ROI-positive, given the current CLV/cost/horizon assumptions.")
    c2.metric("ROI at assumed 30%", f"${be['roi_at_30_pct']:+,.0f}")
    c3.metric("Assumption viable?", "Yes" if be["is_viable"] else "No",
              help="Viable if the break-even rate is comfortably below the assumed 30% (here: below 50%).")

    st.markdown(f"""
    <div class='card' style='font-size:13px; line-height:1.8;'>
    On the <b>{n_high_risk:,} customers currently flagged HIGH risk</b>, the campaign only needs to
    retain <b>{be['break_even_rate']:.1%}</b> of them to break even — well below the {assumed_sr:.0%}
    assumption driving the headline ROI number, which leaves margin if the real campaign underperforms.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Simulate & analyse 
    st.markdown("### Simulated Experiment")
    st.caption(
        f"True success rate in this simulation: **{true_sr_sim:.0%}** "
        f"(vs assumed {assumed_sr:.0%}). Change the slider to test 'what if the campaign underdelivers?'")

    df_sim = simulate_experiment(cfg, n_per_group=ss['n_per_group'], true_success_rate=true_sr_sim)
    result = analyse_experiment(df_sim, cfg, clv_monthly=ctx.clv, retention_cost=ctx.ret_cost, horizon_months=ctx.horizon)

    # Verdict badge
    verdict_colors = {
        "CONFIRMED": ("OK", "#06ffa5", "rgba(6,255,165,0.1)"),
        "REVISE_DOWN": ("WARN", "#ffbe0b", "rgba(255,190,11,0.1)"),
        "INCONCLUSIVE":("N/A", "#9b5de5", "rgba(155,93,229,0.1)"),
        "NO_EFFECT": ("FAIL", "#ff4d6d", "rgba(255,77,109,0.1)"),}
    icon, color, bg = verdict_colors.get(result.verdict, ("N/A", "#9999c0", "rgba(128,128,128,0.1)"))
    st.markdown(f"""
    <div style='background:{bg}; border:1px solid {color}44; border-left:4px solid {color};
                border-radius:10px; padding:16px 20px; margin-bottom:16px;'>
        <span style='font-family:Syne; font-size:13px; font-weight:800; letter-spacing:1px; color:{color};
                     background:{color}22; padding:3px 10px; border-radius:6px;'>{icon}</span>
        <span style='font-family:Syne; font-size:18px; font-weight:700; color:{color}; margin-left:10px;'>
            {result.verdict}
        </span>
        <p style='color:#dcdcf0; font-size:13px; margin-top:8px; margin-bottom:0;'>
            {result.conclusion}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Control churn", f"{result.control_churn_rate:.1%}")
    m2.metric("Treatment churn", f"{result.treatment_churn_rate:.1%}", delta=f"{result.observed_reduction:+.1%}", delta_color="inverse")
    m3.metric("Observed SR", f"{result.observed_success_rate:.1%}", delta=f"{result.observed_success_rate - assumed_sr:+.1%}", delta_color="normal")
    m4.metric("p-value", f"{result.p_value:.4f}", delta="significant" if result.significant else "not significant", delta_color="normal" if result.significant else "inverse")
    m5.metric("ROI delta",       f"${result.roi_delta:+,.0f}", delta="vs assumed", delta_color="normal" if result.roi_delta >= 0 else "inverse")

    # Distribution chart: churn rate by group
    fig_dist = go.Figure()
    for group, rate, color_bar in [
        ("Control", result.control_churn_rate, "#ff4d6d"),
        ("Treatment", result.treatment_churn_rate, "#4cc9f0"),]:
        fig_dist.add_trace(go.Bar(
            x=[group], y=[rate],
            name=group,
            marker_color=color_bar,
            text=[f"{rate:.1%}"],
            textposition="outside",
            width=0.4,))

    fig_dist.add_hline(y=baseline_cr, line_dash="dot", line_color="#9999c0", annotation_text=f"Assumed baseline {baseline_cr:.0%}", annotation_position="top right")
    fig_dist.update_layout(
        **ctx.plotly_dark(),
        title="Observed Churn Rate by Group",
        yaxis_title="Churn rate",
        yaxis_tickformat=".0%",
        showlegend=False,
        height=320,)
    st.plotly_chart(fig_dist, use_container_width=True, key="t7_dist")

    st.divider()

    # Sensitivity sweep 
    st.markdown("### Sensitivity: What If the True Rate Is Different?")
    st.caption(
        "This sweep runs the test at each possible true success rate. "
        "The key question: **how quickly would this experiment catch an underperforming campaign?**")

    sweep_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    sweep_df = sweep_true_success_rates(cfg, true_rates=sweep_rates, n_per_group=ss['n_per_group'])

    fig_sweep = make_subplots(rows=1, cols=2, subplot_titles=["p-value vs True Success Rate", "Observed ROI vs True Success Rate"])

    # p-value line
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df['true_rate'], y=sweep_df['p_value'],
        mode='lines+markers',
        line=dict(color='#f8961e', width=2),
        marker=dict(color=sweep_df['significant'].map({True: '#06ffa5', False: '#ff4d6d'}),
                    size=10),
        name="p-value",), row=1, col=1)

    fig_sweep.add_hline(y=0.05, line_dash="dot", line_color="#ff4d6d", annotation_text="α = 0.05", row=1, col=1)
    fig_sweep.add_vline(x=assumed_sr, line_dash="dash", line_color="#f8961e", annotation_text=f"Assumed {assumed_sr:.0%}", row=1, col=1)

    # ROI line
    roi_colors = ['#06ffa5' if r >= 0 else '#ff4d6d' for r in sweep_df['roi_observed']]
    fig_sweep.add_trace(go.Bar(
        x=sweep_df['true_rate'], y=sweep_df['roi_observed'],
        marker_color=roi_colors,
        name="Observed ROI",
        text=[f"${r:,.0f}" for r in sweep_df['roi_observed']],
        textposition="outside",), row=1, col=2)

    fig_sweep.add_hline(y=0, line_color="#9999c0", line_width=1, row=1, col=2)
    fig_sweep.add_vline(x=assumed_sr, line_dash="dash", line_color="#f8961e", row=1, col=2)

    fig_sweep.update_layout(**ctx.plotly_dark(), height=380, showlegend=False, )
    fig_sweep.update_xaxes(tickformat=".0%", title_text="True success rate")
    fig_sweep.update_yaxes(title_text="p-value", row=1, col=1)
    fig_sweep.update_yaxes(title_text="ROI ($)", tickprefix="$", row=1, col=2)
    st.plotly_chart(fig_sweep, use_container_width=True, key="t7_sweep")

    # Sweep table
    # HTML table - st.dataframe renders empty with custom CSS themes in Streamlit 1.32+
    sweep_cols = ['True Rate', 'p-value', 'Significant', 'Observed SR', 'ROI ($)', 'Verdict']
    header_html = "".join(
        f"<th style='padding:8px 12px; text-align:left; color:#9999c0; font-size:11px; "
        f"letter-spacing:1px; text-transform:uppercase; border-bottom:1px solid #1e1e3a; "
        f"white-space:nowrap;'>{c}</th>"
        for c in sweep_cols)
    rows_html = ""
    for _, row in sweep_df.iterrows():
        sig = bool(row['significant'])
        roi_val = float(row['roi_observed'])
        p_val = float(row['p_value'])
        sig_color = "#06ffa5" if sig else "#ff4d6d"
        roi_color = "#06ffa5" if roi_val >= 0 else "#ff4d6d"
        p_color = "#06ffa5" if p_val < 0.05 else "#dcdcf0"
        cells = (
            f"<td style='padding:8px 12px; color:#dcdcf0; font-size:12px; border-bottom:1px solid #1e1e3a;'>{row['true_rate']:.0%}</td>"
            f"<td style='padding:8px 12px; color:{p_color}; font-size:12px; border-bottom:1px solid #1e1e3a;'>{p_val:.4f}</td>"
            f"<td style='padding:8px 12px; color:{sig_color}; font-size:12px; font-weight:600; border-bottom:1px solid #1e1e3a;'>{'Yes' if sig else 'No'}</td>"
            f"<td style='padding:8px 12px; color:#dcdcf0; font-size:12px; border-bottom:1px solid #1e1e3a;'>{row['observed_sr']:.1%}</td>"
            f"<td style='padding:8px 12px; color:{roi_color}; font-size:12px; font-weight:600; border-bottom:1px solid #1e1e3a;'>${roi_val:,.0f}</td>"
            f"<td style='padding:8px 12px; color:#dcdcf0; font-size:12px; border-bottom:1px solid #1e1e3a;'>{row['verdict']}</td>")
        rows_html += f"<tr>{cells}</tr>"
    st.markdown(f"""
    <div style='overflow-x:auto; border:1px solid #1e1e3a; border-radius:8px; margin-bottom:16px;'>
      <table style='width:100%; border-collapse:collapse; background:#0f0f1e;'>
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Sequential monitoring 
    st.markdown("### Sequential Monitoring (Bonferroni-corrected)")
    st.caption(
        "In practice, experiments are monitored at interim checkpoints. "
        "Bonferroni correction maintains the family-wise error rate across all looks.")

    seq_df = sequential_analysis(cfg, n_checkpoints=5, true_success_rate=true_sr_sim)

    fig_seq = go.Figure()
    fig_seq.add_trace(go.Scatter(
        x=seq_df['pct_enrolled'], y=seq_df['p_value'],
        mode='lines+markers',
        line=dict(color='#4cc9f0', width=2),
        marker=dict(color=seq_df['significant_at_checkpoint'].map({True: '#06ffa5', False: '#ff4d6d'}), size=12, symbol='circle',),
        name="p-value",
        text=[f"Checkpoint {r['checkpoint']}<br>n={r['n_per_group_so_far']:,}/arm<br>"
              f"p={r['p_value']:.4f}<br>{r['verdict']}"
              for _, r in seq_df.iterrows()],
        hoverinfo='text',))
    fig_seq.add_hline(
        y=seq_df['adjusted_alpha'].iloc[0], line_dash="dash", line_color="#ffbe0b",
        annotation_text=f"Bonferroni α = {seq_df['adjusted_alpha'].iloc[0]:.4f}",
        annotation_position="top right",)
    fig_seq.add_hline(y=0.05, line_dash="dot", line_color="#9999c0", annotation_text="Unadjusted α = 0.05")

    fig_seq.update_layout(
        **ctx.plotly_dark(),
        title="p-value at Each Interim Analysis",
        xaxis_title="% of target enrollment",
        yaxis_title="p-value",
        xaxis_ticksuffix="%",
        height=340,
        showlegend=False,)
    st.plotly_chart(fig_seq, use_container_width=True, key="t7_seq")

    #  Full text report 
    st.divider()
    with st.expander("Full statistical report"):
        report_text = ab_test_report(result, ss)
        st.code(report_text, language=None)

    #  Business framing 
    st.markdown("### How to use this in production")
    st.markdown(f"""
    <div class='card card-accent' style='font-size:13px; line-height:1.9;'>

    <b>1. Set up the experiment</b><br>
    Split high-risk customers (risk_level = HIGH) randomly into two arms.
    Control receives no campaign. Treatment receives your retention offer.
    Required sample: <b>{ss['n_per_group']:,} per arm</b>
    (~{ss['weeks_at_1k_weekly']:.1f} weeks at current flagging rate).

    <br><br>
    <b>2. Collect outcomes</b><br>
    After {ctx.horizon} months, record which customers churned.
    Feed the results into <code>analyse_experiment(real_df, cfg)</code>.

    <br><br>
    <b>3. Act on the verdict</b><br>
    • <span style='color:#06ffa5'><b>CONFIRMED</b></span> - ROI estimate is reliable. Scale the campaign.<br>
    • <span style='color:#ffbe0b'><b>REVISE_DOWN</b></span> - Update success_rate in ASSUMPTIONS and rerun ROI model.<br>
    • <span style='color:#9b5de5'><b>INCONCLUSIVE</b></span> - Extend the experiment. Current power: {result.achieved_power:.0%}.<br>
    • <span style='color:#ff4d6d'><b>NO_EFFECT</b></span> - Redesign the campaign offer before investing at scale.<br>

    <br>
    <b>Break-even rate: {result.break_even_rate:.1%}</b> - the minimum success rate for the campaign to be ROI-positive.
    As long as the true rate exceeds this threshold, the campaign pays for itself.

    </div>
    """, unsafe_allow_html=True)
