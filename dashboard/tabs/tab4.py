import numpy as np
import streamlit as st
import plotly.graph_objects as go

from dashboard.business import physics_snapshot
from dashboard.tabs.context import DashboardContext
from src.visualization import plot_trajectory_plotly


def render(ctx: DashboardContext) -> None:
    """Render the Physics Simulator tab """

    st.markdown("#### Interactive physics simulator")

    # Bridge from dashboard 
    loaded_seg_phys = st.session_state.get("selected_segment", "All (no filter)")
    _ = st.session_state.get("selected_risk", "All (no filter)")

    is_filtered_phys = loaded_seg_phys != "All (no filter)"

    if is_filtered_phys:
        # Filter to exact segment, not just risk level
        bridge_customers = (
            ctx.df_f[ctx.df_f["segment"] == loaded_seg_phys]
            .sort_values("prob_churn", ascending=False))
        if len(bridge_customers) > 0:
            top_cust = bridge_customers.iloc[0]
            st.markdown(f"""
            <div class='card card-accent' style='padding: 12px 18px; margin-bottom: 16px;'>
            <span class='small'>Pre-loaded from Dashboard — top customer in
              <b style='color:white;'>{loaded_seg_phys}</b>:</span>
            <b style='color:#f8961e; margin-left:6px;'>{top_cust["customerID"]}</b>
            <span style='color:#9999c0; font-size:12px; margin-left:10px;'>
            P(churn)={top_cust["prob_churn"]:.0%} ± {top_cust["sigma_prob"]:.0%} · τ={top_cust["tau"]:.1f}m · E_eq={top_cust["E_eq"]:.2f}
            </span>
            <br><span style='color:#9999c0; font-size:11px; margin-top:4px; display:block;'>
            Sliders pre-set to match this customer's physical profile.
            </span>
            </div>
            """, unsafe_allow_html=True)
            default_contract = top_cust["contract"]
            default_tenure = int(top_cust["tenure"])
            default_monthly = int(top_cust["monthly"])
            default_services_protective = int(top_cust.get("n_services_protective", 2))
            default_services_entertainment = int(top_cust.get("n_services_entertainment", 1))
            default_internet_type = top_cust.get("internet_type", "DSL")
            default_autopay = bool(top_cust["auto_pay"])
        else:
            default_contract, default_tenure, default_monthly = "Month-to-month", 12, 65
            default_services_protective, default_services_entertainment, default_autopay = 2, 1, False
            default_internet_type = "DSL"
    else:
        st.markdown("<p class='small'>Build a customer profile and see their energy trajectory. Simulate what happens when a negative event hits.</p>", unsafe_allow_html=True)
        default_contract, default_tenure, default_monthly = "Month-to-month", 12, 65
        default_services_protective, default_services_entertainment, default_autopay = 2, 1, False
        default_internet_type = "DSL"

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        st.markdown("**Customer profile**")
        sim_contract = st.selectbox("Contract",
            ["Month-to-month","One year","Two year"],
            index=["Month-to-month","One year","Two year"].index(default_contract))
        sim_tenure = st.slider("Tenure (months)", 0, 72, default_tenure)
        sim_monthly  = st.slider("Monthly charge ($)", 20, 110, min(max(default_monthly, 20), 110))
        sim_services_protective = st.slider(
            "Protective services", 0, 4, default_services_protective,
            help="OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport — raise switching cost meaningfully.")
        sim_services_entertainment = st.slider(
            "Entertainment services", 0, 2, default_services_entertainment,
            help="StreamingTV, StreamingMovies — easily substituted; barely move churn risk.")
        internet_options = ["No", "DSL", "Fiber optic"]
        sim_internet_type = st.selectbox(
            "Internet type", internet_options,
            index=internet_options.index(default_internet_type) if default_internet_type in internet_options else 1,
            help="Largest single churn-rate gap in the dataset (7.4% No vs 19.0% DSL vs 41.9% Fiber optic) — Fiber optic customers churn most, despite paying for the fastest connection.")
        payment_options = ["Auto (bank transfer / credit card)", "Mailed check", "Electronic check"]
        default_payment_idx = 0 if default_autopay else 1
        sim_payment_label = st.selectbox(
            "Payment method", payment_options, index=default_payment_idx,
            help="Electronic check churns far more (45.3%) than mailed check (19.1%) or the two automatic methods (~15-17%) — a gap that survives controlling for contract type.")
        sim_payment_method = {"Auto (bank transfer / credit card)": "auto", "Mailed check": "mailed_check", "Electronic check": "electronic_check"}[sim_payment_label]
        sim_auto_pay = sim_payment_method == "auto"

        st.divider()
        st.markdown("**Simulate an event**")
        sim_event = st.toggle("Add negative event", value=True)
        sim_t_event = st.slider("Event timing (month)", 1, 30, 8, disabled=not sim_event)
        sim_mag = st.slider("Event magnitude", 0.05, 0.60, 0.25, 0.05, disabled=not sim_event)

        st.divider()
        sim_horizon_s = st.slider("Prediction ctx.horizon", 1, 12, ctx.horizon)

    with col_plot:
        total_s = sim_monthly * max(sim_tenure, 1)
        long_c  = sim_contract in ("One year","Two year")
        E0_s = ctx.compute_E0(sim_tenure, sim_monthly, total_s, sim_services_protective, sim_services_entertainment, long_c, sim_internet_type)
        E_eq_s  = ctx.compute_E_eq(sim_contract, sim_services_protective, sim_services_entertainment, sim_auto_pay, sim_internet_type, payment_method=sim_payment_method)
        gamma_s = ctx.compute_gamma(sim_contract, sim_services_protective, sim_services_entertainment, sim_auto_pay, sim_internet_type, payment_method=sim_payment_method)
        tau_s = 1 / (gamma_s + 1e-9)

        def F_free(t):
            return 0.0

        def F_event(t):
            return -sim_mag if (sim_event and t >= sim_t_event) else 0.0

        t_free, E_free  = ctx.solve_trajectory(E0_s, gamma_s, E_eq_s, F_free,  (0, 36))
        t_event_, E_event = ctx.solve_trajectory(E0_s, gamma_s, E_eq_s, F_event, (0, 36))

        prob_s, sigma_s, lb_s, ub_s, risk_s = ctx.churn_prob_with_uncertainty(E0_s, 0.05, gamma_s, 0.02, E_eq_s, 0.03, sim_horizon_s)
        seg_s, act_s = ctx.segment_customer(risk_s, tau_s, E_eq_s)

        # Advanced trajectory with colour-coded risk zones 
        sigma_band = 0.05 * np.exp(-gamma_s * t_free)

        # Free trajectory via plot_trajectory_plotly
        fig_sim = plot_trajectory_plotly(
            t = t_free,
            E = E_free,
            E_eq = E_eq_s,
            E_critical = ctx.E_CRITICAL,
            sigma_upper  = np.clip(E_free + sigma_band, 0, 1),
            sigma_lower = np.clip(E_free - sigma_band, 0, 1),
            title = "dE/dt = −γ(E − E_eq) + F(t)  |  Energy trajectory",)

        # Overlay event trajectory if perturbation active
        if sim_event:
            sigma_band_ev = 0.05 * np.exp(-gamma_s * t_event_)
            # Uncertainty band for event trajectory
            fig_sim.add_trace(go.Scatter(
                x=np.concatenate([t_event_, t_event_[::-1]]),
                y=np.concatenate([
                    np.clip(E_event + sigma_band_ev, 0, 1),
                    np.clip(E_event - sigma_band_ev, 0, 1)[::-1],]),
                fill="toself", fillcolor="rgba(255,77,109,0.10)",
                line=dict(width=0), showlegend=True,
                name="±σ band (with event)", hoverinfo="skip",))
            # Event trajectory 
            churn_ev = E_event < ctx.E_CRITICAL
            if (~churn_ev).any():
                idx = np.where(~churn_ev)[0]
                fig_sim.add_trace(go.Scatter(
                    x=t_event_[idx], y=E_event[idx],
                    mode="lines", name="With event (safe)",
                    line=dict(color="#ff4d6d", width=2.5, dash="longdash"),))
            if churn_ev.any():
                idx2 = np.where(churn_ev)[0]
                fig_sim.add_trace(go.Scatter(
                    x=t_event_[idx2], y=E_event[idx2],
                    mode="lines", name="With event (at risk)",
                    line=dict(color="#ff2244", width=2.5),))
            fig_sim.add_vline(x=sim_t_event, line_dash="dot",
                               line_color="#ffbe0b", line_width=1.5,
                               annotation_text=f"Perturbation t={sim_t_event}m",
                               annotation_font=dict(color="#ffbe0b", size=10))

        # Prediction horizon shading
        fig_sim.add_vrect(x0=0, x1=sim_horizon_s,
                           fillcolor="#f8961e", opacity=0.04, line_width=0,
                           annotation_text=f"{sim_horizon_s}m horizon",
                           annotation_position="top left", annotation_font=dict(color="#f8961e", size=9))

        st.plotly_chart(fig_sim, use_container_width=True, key="t4_sim")

        # Diagnosis card 
        color_risk = ctx.RISK_COLORS.get(risk_s, "#f8961e")
        intrinsic  = "YES - will churn without offer change" if E_eq_s < ctx.E_CRITICAL else "NO — structurally stable"

        _cv1 = "#ff4d6d" if E_eq_s < ctx.E_CRITICAL else "#06ffa5"
        _cv2 = "#ff4d6d" if E_eq_s < ctx.E_CRITICAL else "#06ffa5"
        st.markdown(f"""
        <div class='card' style='border-left: 3px solid {color_risk};'>
          <div style='display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px;'>
            <div>
              <div class='small'>Risk level</div>
              <div style='font-family:Syne; font-size:22px; font-weight:700; color:{color_risk}; line-height:1.5; letter-spacing:0.3px;'>{risk_s}</div>
            </div>
            <div>
              <div class='small'>P(churn) in {sim_horizon_s}m</div>
              <div style='font-family:Syne; font-size:22px; font-weight:700; color:white; line-height:1.5; letter-spacing:0.3px;'>{prob_s:.0%} ± {sigma_s:.0%}</div>
            </div>
            <div>
              <div class='small'>95% CI [{lb_s:.0%} – {ub_s:.0%}]</div>
              <div style='font-size:13px; color:#9999c0; margin-top:6px;'>τ = {tau_s:.1f}m &nbsp;|&nbsp; γ = {gamma_s:.3f}</div>
            </div>
          </div>
          <div style='margin-top:14px; display:flex; gap:24px; flex-wrap:wrap;'>
            <div><span class='small'>E₀ (current)</span><br><b>{E0_s:.3f}</b></div>
            <div><span class='small'>E_eq (structural)</span><br><b style='color:{_cv1};'>{E_eq_s:.3f}</b></div>
            <div><span class='small'>Intrinsic risk</span><br><b style='color:{_cv2};'>{intrinsic}</b></div>
          </div>
          <div style='margin-top:12px; padding:10px 14px; background:#080810; border-radius:6px;'>
            <span class='small'>Segment: <b style='color:white;'>{seg_s}</b></span><br>
            <span style='font-size:13px; color:#f8961e;'>▶ {act_s}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Deterministic snapshot (point estimate, no-event trajectory) — complements the
        # uncertainty-propagated P(churn) above with the raw ODE point estimate at this horizon
        snap = physics_snapshot(E0_s, gamma_s, E_eq_s, sim_horizon_s, E_critical=ctx.E_CRITICAL)
        with st.expander("Deterministic snapshot (point estimate, no perturbation)"):
            sn1, sn2, sn3, sn4 = st.columns(4)
            sn1.metric(f"E(t={sim_horizon_s}m)", f"{snap['E_t']:.3f}")
            sn2.metric("Δ from E_eq", f"{snap['delta_E']:+.3f}")
            sn3.metric("P(churn) ≈ 1−E(t)", f"{snap['p_churn_approx']:.0%}")
            sn4.metric("Crosses E_critical?", "Yes" if snap["will_churn"] else "No")

        # Best / Base / Worst for this single customer 
        st.markdown("#### Campaign outcome - confidence range for this customer")
        rev_risk_s = prob_s * ctx.clv * sim_horizon_s

        scenarios_cust = [
            ("Worst", "#ff4d6d", lb_s, 0.15),
            ("Base", "#f8961e", prob_s, 0.30),
            ("Best", "#06ffa5", ub_s, 0.45),]
        sc_cols = st.columns(3)
        for scol, (slabel, scolor, sprob, srate) in zip(sc_cols, scenarios_cust):
            srev = sprob * srate * ctx.clv * sim_horizon_s
            sroi = srev - ctx.ret_cost
            with scol:
                _cv3 = "#06ffa5" if sroi >= 0 else "#ff4d6d"
                st.markdown(f"""
                <div class='card' style='border-left:3px solid {scolor}; padding:14px 16px;'>
                  <div style='font-size:11px; color:{scolor}; font-weight:700; letter-spacing:1px;'>{slabel}</div>
                  <div style='font-size:15px; font-weight:700; color:white; margin:6px 0 2px 0;'>P = {sprob:.0%}</div>
                  <div class='small'>Retention rate: {srate:.0%}</div>
                  <div style='margin-top:8px;'>
                    <div class='small'>Revenue saved</div>
                    <div style='font-size:16px; font-weight:700; color:{scolor};'>${srev:,.0f}</div>
                  </div>
                  <div style='margin-top:6px;'>
                    <div class='small'>Net ROI (after ${ctx.ret_cost} cost)</div>
                    <div style='font-size:15px; font-weight:700; color:{_cv3};'>${sroi:+,.0f}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # Physical parameters explanation 
    st.divider()
    col_eq1, col_eq2, col_eq3 = st.columns(3)
    with col_eq1:
        _cv4 = "High resilience — responds fast" if tau_s < 4 else "Medium resilience" if tau_s < 7 else "Low resilience — slow to respond"
        st.markdown(f"""
        <div class='card card-safe'>
        <div class='small'>DAMPING γ</div>
        <div style='font-size:20px; font-weight:700; color:white;'>{gamma_s:.3f}</div>
        <div class='small' style='margin-top:6px;'>
        τ = {tau_s:.1f} months<br>
        {_cv4}
        </div>
        </div>
        """, unsafe_allow_html=True)
    with col_eq2:
        _cv5 = "#ff4d6d" if E_eq_s < ctx.E_CRITICAL else "#06ffa5"
        _cv6 = "Below critical threshold — intrinsic risk" if E_eq_s < ctx.E_CRITICAL else "Above critical threshold — structurally loyal"
        st.markdown(f"""
        <div class='card card-green'>
        <div class='small'>EQUILIBRIUM E_eq</div>
        <div style='font-size:20px; font-weight:700; color:{_cv5};'>{E_eq_s:.3f}</div>
        <div class='small' style='margin-top:6px;'>
        Critical threshold: {ctx.E_CRITICAL}<br>
        {_cv6}
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col_eq3:
        rev_risk_s = prob_s * ctx.clv * sim_horizon_s
        rev_save_s = rev_risk_s * 0.30
        st.markdown(f"""
        <div class='card card-accent'>
        <div class='small'>ECONOMIC IMPACT</div>
        <div style='font-size:20px; font-weight:700; color:#f8961e;'>${rev_risk_s:,.0f}</div>
        <div class='small' style='margin-top:6px;'>
        Revenue at risk ({sim_horizon_s}m)<br>
        Recoverable: <b style='color:#06ffa5;'>${rev_save_s:,.0f}</b> (base rate 30%)<br>
        Range: <b style='color:#ff4d6d;'>${prob_s * 0.15 * ctx.clv * sim_horizon_s:,.0f}</b>
        – <b style='color:#06ffa5;'>${prob_s * 0.45 * ctx.clv * sim_horizon_s:,.0f}</b>
        </div>
        </div>
        """, unsafe_allow_html=True)
