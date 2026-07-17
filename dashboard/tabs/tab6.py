import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard.tabs.context import DashboardContext


def render(ctx: DashboardContext) -> None:
    """Render the Model Validation tab"""

    st.markdown("#### Model Validation - Physical Model vs ML Baselines")
    st.markdown("""
    <p class='small'>
    Executable comparison: physical model vs Logistic Regression, Random Forest, and GBM.
    All metrics computed with 5-fold stratified cross-validation on the same data.
    The physical model has lower AUC — that gap is real and documented.
    The table below explains what each model can and cannot do.
    </p>
    """, unsafe_allow_html=True)

    if not ctx.using_real_data:
        st.markdown("""
        <div class='card card-churn'>
        <b>Synthetic data - results are illustrative, not real</b><br>
        <span class='small'>
        Load the real IBM Telco dataset (<code>data/telco_churn.csv</code>) to see actual AUC,
        recall, and ROI numbers from cross-validation.<br>
        The comparison below runs on synthetic data to demonstrate the structure.
        Numbers will change, and become credible, with real data.
        </span>
        </div>
        """, unsafe_allow_html=True)

    # Prepare features 
    @st.cache_data(show_spinner=False)
    def run_baseline_comparison(df_hash, n_folds=5, phys_threshold=0.20):
        import pandas as pd  
        """Trains LR, RF, GBM and the physical model score on the same data"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

        y = df_hash["churn"].values

        # Physical model features
        phys_cols = ["E0", "E_eq", "gamma", "tau", "n_services_protective", "n_services_entertainment"]  # internet_type contributes via cat_df one-hot below
        phys_cols = [c for c in phys_cols if c in df_hash.columns]

        # Business features for baselines
        base_cols = ["tenure", "monthly", "n_services_protective", "n_services_entertainment"]
        base_cols = [c for c in base_cols if c in df_hash.columns]

        # Contract / payment one-hot
        cat_cols_present = [c for c in ["contract", "internet_type"] if c in df_hash.columns]
        cat_df = pd.get_dummies(df_hash[cat_cols_present], drop_first=True) if cat_cols_present else pd.DataFrame()
        full_X = pd.concat([df_hash[base_cols].fillna(0), cat_df], axis=1)
        phys_X = df_hash[phys_cols].fillna(0)

        scaler = StandardScaler()

        models = {
            "LR (raw features)": (LogisticRegression(max_iter=1000, random_state=42), full_X),
            "LR (physical features)": (LogisticRegression(max_iter=1000, random_state=42), phys_X),
            "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), full_X),
            "GBM": (GradientBoostingClassifier(n_estimators=100, random_state=42), full_X), }

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        results = {}
        for name, (model, X) in models.items():
            Xs = scaler.fit_transform(X)
            auc_scores = cross_val_score(model, Xs, y, cv=cv, scoring="roc_auc")
            recall_scores = cross_val_score(model, Xs, y, cv=cv, scoring="recall")
            prec_scores = cross_val_score(model, Xs, y, cv=cv, scoring="precision")
            f1_scores = cross_val_score(model, Xs, y, cv=cv, scoring="f1")
            results[name] = {
                "AUC": auc_scores.mean(),
                "AUC_std": auc_scores.std(),
                "Recall": recall_scores.mean(),
                "Recall_std": recall_scores.std(),
                "Precision": prec_scores.mean(),
                "F1": f1_scores.mean(),}

        # Physical model: use prob_churn as the score
        if "prob_churn" in df_hash.columns and "churn" in df_hash.columns:
            phys_proba = df_hash["prob_churn"].values
            phys_pred  = (phys_proba > phys_threshold).astype(int)
            results["Physical model"] = {
                "AUC": roc_auc_score(y, phys_proba),
                "AUC_std": 0.0,
                "Recall": recall_score(y, phys_pred, zero_division=0),
                "Recall_std": 0.0,
                "Precision": precision_score(y, phys_pred, zero_division=0),
                "F1": f1_score(y, phys_pred, zero_division=0), }

        return results

    with st.spinner("Running 5-fold cross-validation on all models…"):
        # Hash the dataframe by shape + churn rate for cache key
        _ = tuple([len(ctx.df_f), round(ctx.df_f["churn"].mean(), 4)] + ctx.df_f["prob_churn"].head(10).round(3).tolist())
        cv_results = run_baseline_comparison(
            ctx.df_f[["churn", "E0", "E_eq", "gamma", "tau", "n_services_protective", "n_services_entertainment", "tenure", "monthly", "contract", "internet_type", "prob_churn"]].copy())

    # Metrics table 
    st.markdown("### Technical metrics (5-fold CV)")

    table_rows = []
    model_order = ["LR (raw features)", "LR (physical features)", "Random Forest", "GBM", "Physical model"]
    for m in model_order:
        if m not in cv_results:
            continue
        r = cv_results[m]
        is_physical = m == "Physical model"
        table_rows.append({
            "Model": m,
            "AUC": f"{r['AUC']:.4f}" + (f" ± {r['AUC_std']:.4f}" if r["AUC_std"] > 0 else ""),
            "Recall": f"{r['Recall']:.4f}" + (f" ± {r['Recall_std']:.4f}" if r["Recall_std"] > 0 else ""),
            "Precision": f"{r['Precision']:.4f}",
            "F1": f"{r['F1']:.4f}",
            "Uncertainty": "Yes (± σ per customer)" if is_physical else "No",
            "Diagnosis": "Yes (E_eq, τ)" if is_physical else "No",
            "Simulation": "Yes (ODE + MC)" if is_physical else "No",
            "_is_physical": is_physical,})

    # HTML table - st.dataframe renders empty with custom CSS themes in Streamlit 1.32+
    if table_rows:
        cols = ["Model", "AUC", "Recall", "Precision", "F1", "Uncertainty", "Diagnosis", "Simulation"]
        header_html = "".join(
            f"<th style='padding:8px 12px; text-align:left; color:#9999c0; font-size:11px; "
            f"letter-spacing:1px; text-transform:uppercase; border-bottom:1px solid #1e1e3a; "
            f"white-space:nowrap;'>{c}</th>"
            for c in cols)
        rows_html = ""
        for row in table_rows:
            is_phys = row["_is_physical"]
            row_style = "background:#141428;" if is_phys else ""
            cells = ""
            for col in cols:
                val = row.get(col, "")
                cell_color = "#f8961e" if is_phys and col == "Model" else "#dcdcf0"
                check_color = "#06ffa5" if val.startswith("Yes") else "#555"
                if val.startswith("Yes") or val == "No":
                    cell_color = check_color
                cells += (
                    f"<td style='padding:8px 12px; color:{cell_color}; font-size:12px; "
                    f"border-bottom:1px solid #1e1e3a; white-space:nowrap;'>{val}</td>")
            rows_html += f"<tr style='{row_style}'>{cells}</tr>"

        st.markdown(f"""
        <div style='overflow-x:auto; border:1px solid #1e1e3a; border-radius:8px; margin-bottom:16px;'>
          <table style='width:100%; border-collapse:collapse; background:#0f0f1e;'>
            <thead><tr>{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

    # ROC curves 
    st.markdown("### ROC curves")
    st.markdown("<p class='small'>Area under curve - higher is better for pure ranking. The physical model trades AUC for interpretability and uncertainty.</p>", unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def compute_roc_curves(df_roc, phys_threshold=0.20):
        import pandas as pd  
        import numpy as np  
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_curve, roc_auc_score

        y = df_roc["churn"].values
        base_cols = [c for c in ["tenure", "monthly", "n_services_protective", "n_services_entertainment"] if c in df_roc.columns]
        cat_cols_present_roc = [c for c in ["contract", "internet_type"] if c in df_roc.columns]
        cat_df = pd.get_dummies(df_roc[cat_cols_present_roc], drop_first=True) if cat_cols_present_roc else pd.DataFrame()
        X = pd.concat([df_roc[base_cols].fillna(0), cat_df], axis=1)
        Xs = StandardScaler().fit_transform(X)

        idx = np.arange(len(y))
        idx_tr, idx_te, y_tr, y_te = train_test_split(
            idx, y, test_size=0.25, random_state=42, stratify=y )
        X_tr, X_te = Xs[idx_tr], Xs[idx_te]

        curves = {}
        for name, model in [
            ("LR (raw features)", LogisticRegression(max_iter=1000, random_state=42)),
            ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("GBM", GradientBoostingClassifier(n_estimators=100, random_state=42)),]:
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            fpr, tpr, _ = roc_curve(y_te, proba)
            curves[name] = (fpr, tpr, roc_auc_score(y_te, proba))

        # Physical model ROC — use idx_te to correctly index prob_churn
        phys_proba_te = df_roc["prob_churn"].values[idx_te]
        try:
            fpr_p, tpr_p, _ = roc_curve(y_te, phys_proba_te)
            curves["Physical model"] = (fpr_p, tpr_p, roc_auc_score(y_te, phys_proba_te))
        except Exception:
            pass

        return curves

    with st.spinner("Computing ROC curves…"):
        roc_curves = compute_roc_curves(
            ctx.df_f[["churn", "tenure", "monthly", "n_services_protective", "n_services_entertainment", "contract", "internet_type", "prob_churn"]].copy(),
            phys_threshold=ctx.threshold,)

    roc_colors = {
        "LR (raw features)": "#8d8dbd",
        "Random Forest": "#4cc9f0",
        "GBM": "#06ffa5",
        "Physical model": "#f8961e",}
    fig_roc = go.Figure()
    for name, (fpr, tpr, auc) in roc_curves.items():
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})",
            line=dict(color=roc_colors.get(name, "#9999c0"), width=3 if name == "Physical model" else 1.5),))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#333355", dash="dash", width=1),
        name="Random (AUC=0.5)", showlegend=True,))
    fig_roc.update_layout(**ctx.plotly_dark(), height=400, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    fig_roc.update_xaxes(**ctx.plotly_axis_style(), range=[0, 1])
    fig_roc.update_yaxes(**ctx.plotly_axis_style(), range=[0, 1])
    st.plotly_chart(fig_roc, use_container_width=True, key="t6_roc")

    # Lift curve 
    st.markdown("---")
    st.markdown("### Lift curve - is the model better than random targeting?")
    st.markdown(
        "<p class='small'>The lift curve answers the most important question in retention analytics: "
        "if we can only contact the top X% of customers, how much better is the model vs random selection? "
        "Lift = 1.0 means the model adds no value over random. "
        "Lift = 2.0 at 20% means targeting the top 20% catches 2x more churners than random.</p>",
        unsafe_allow_html=True,)

    if "churn" in ctx.df_f.columns and "prob_churn" in ctx.df_f.columns:
        @st.cache_data(show_spinner=False)
        def compute_lift(probs_tuple, actuals_tuple):
            import numpy as np
            probs = np.array(probs_tuple)
            actuals = np.array(actuals_tuple)
            order = np.argsort(probs)[::-1]
            _ = probs[order]
            acts_s  = actuals[order]
            n = len(acts_s)
            base_rate = actuals.mean()
            pcts, lifts, cum_gains, cum_random = [], [], [], []
            for k in range(1, n + 1):
                pct = k / n
                rate = acts_s[:k].mean()
                lift = rate / base_rate if base_rate > 0 else 1.0
                pcts.append(round(pct * 100, 2))
                lifts.append(round(lift, 4))
                cum_gains.append(round(acts_s[:k].sum() / actuals.sum() * 100, 2))
                cum_random.append(round(pct * 100, 2))
            return pcts, lifts, cum_gains, cum_random

        _probs = tuple(ctx.df_f["prob_churn"].values)
        _actuals = tuple(ctx.df_f["churn"].values)
        pcts, lifts, cum_gains, cum_random = compute_lift(_probs, _actuals)

        import numpy as np
        # Subsample to 200 points for plotting
        idx = np.linspace(0, len(pcts) - 1, min(200, len(pcts)), dtype=int)

        fig_lift = go.Figure()

        # Lift chart 
        fig_lift = make_subplots(rows=1, cols=2, subplot_titles=["Lift curve", "Cumulative gain curve"],)

        # Lift curve
        fig_lift.add_trace(go.Scatter(
            x=[pcts[i] for i in idx],
            y=[lifts[i] for i in idx],
            mode="lines", name="Physical model lift",
            line=dict(color="#f8961e", width=2.5),
            fill="tozeroy", fillcolor="rgba(248,150,30,0.07)",
            hovertemplate="Top %{x:.0f}% -> lift = %{y:.2f}x<extra></extra>",), row=1, col=1)

        # Baseline (lift = 1)
        fig_lift.add_hline(y=1.0, line_dash="dash", line_color="#9999c0", line_width=1.2, row=1, col=1,
                            annotation_text="Random (lift = 1x)", annotation_font=dict(color="#9999c0", size=9))

        # Annotate key points
        for pct_target in [10, 20, 30, 50]:
            close_idx = min(range(len(pcts)), key=lambda i: abs(pcts[i] - pct_target))
            lift_val = lifts[close_idx]
            fig_lift.add_annotation(
                x=pcts[close_idx], y=lift_val,
                text=f"Top {pct_target}%<br>{lift_val:.1f}×",
                showarrow=True, arrowhead=2, arrowwidth=1,
                arrowcolor="#f8961e", ax=25, ay=-30,
                font=dict(color="#f8961e", size=9),row=1, col=1, )

        # Cumulative gain curve
        fig_lift.add_trace(go.Scatter(
            x=[pcts[i] for i in idx],
            y=[cum_gains[i] for i in idx],
            mode="lines", name="Model",
            line=dict(color="#06ffa5", width=2.5),
            hovertemplate="Top %{x:.0f}% → captured %{y:.1f}% of churners<extra></extra>",), row=1, col=2)

        fig_lift.add_trace(go.Scatter(
            x=[pcts[i] for i in idx],
            y=[cum_random[i] for i in idx],
            mode="lines", name="Random baseline",
            line=dict(color="#9999c0", width=1.5, dash="dash"),
            hovertemplate="Random: %{x:.0f}%<extra></extra>",), row=1, col=2)

        # Shade the gain area between model and random
        fig_lift.add_trace(go.Scatter(
            x=[pcts[i] for i in idx] + [pcts[i] for i in idx][::-1],
            y=[cum_gains[i] for i in idx] + [cum_random[i] for i in idx][::-1],
            fill="toself", fillcolor="rgba(6,255,165,0.08)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",), row=1, col=2)

        fig_lift.update_layout(
            **ctx.plotly_dark(),
            height=400,
            showlegend=True,
            legend=dict(x=0.55, y=0.15, **ctx.plotly_legend_style()), )
        fig_lift.update_xaxes(title_text="% of customers contacted", ticksuffix="%")
        fig_lift.update_yaxes(title_text="Lift (x)", row=1, col=1)
        fig_lift.update_yaxes(title_text="% of churners captured", ticksuffix="%", row=1, col=2)
        st.plotly_chart(fig_lift, use_container_width=True, key="t6_lift")

        # Key number callout
        lift_20 = lifts[min(range(len(pcts)), key=lambda i: abs(pcts[i] - 20))]
        gain_20 = cum_gains[min(range(len(pcts)), key=lambda i: abs(pcts[i] - 20))]
        st.markdown(f"""
        <div class='card card-green' style='font-size:13px;'>
        <b>Key number:</b> Contacting the <b>top 20%</b> of customers (ranked by model score)
        captures <b>{gain_20:.0f}% of all churners</b> - a <b>{lift_20:.1f}x lift</b> over random selection.
        That means the same campaign budget reaches {lift_20:.1f}× more actual churners when guided by the model.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Lift curve requires actual churn labels. Load the real dataset to see this chart.")

    #  Capability matrix 
    st.markdown("### What each model can do")

    # NOTE (bug fix): the AUC row below used to be hardcoded ("~0.84", "~0.87", "~0.89", "~0.83")
    # and didn't match the actual cross-validated results computed above in this same tab
    # (cv_results) -- classic case of a decorative summary drifting out of sync with the real
    # numbers. Pulled from cv_results directly so it can't diverge again.
    _auc_lr  = cv_results.get("LR (raw features)", {}).get("AUC")
    _auc_rf  = cv_results.get("Random Forest", {}).get("AUC")
    _auc_gbm = cv_results.get("GBM", {}).get("AUC")
    _auc_phys = cv_results.get("Physical model", {}).get("AUC")
    def _fmt_auc(v):
        return f"{v:.2f}" if v is not None else "n/a"

    st.markdown(f"""
    <div style='overflow-x: auto;'>
    <table style='width:100%; border-collapse:collapse; font-family: JetBrains Mono; font-size:12px;'>
    <thead>
    <tr style='border-bottom: 1px solid #1e1e3a;'>
      <th style='text-align:left; padding:10px 16px; color:#9999c0;'>CAPABILITY</th>
      <th style='text-align:center; padding:10px 16px; color:#8d8dbd;'>LR</th>
      <th style='text-align:center; padding:10px 16px; color:#4cc9f0;'>Random Forest</th>
      <th style='text-align:center; padding:10px 16px; color:#06ffa5;'>GBM</th>
      <th style='text-align:center; padding:10px 16px; color:#f8961e;'>Physical model</th>
    </tr>
    </thead>
    <tbody>
    <tr style='border-bottom:1px solid #1e1e3a;'>
      <td style='padding:9px 16px; color:#dcdcf0;'>Churn probability score</td>
      <td style='text-align:center; color:#06ffa5;'>Yes</td>
      <td style='text-align:center; color:#06ffa5;'>Yes</td>
      <td style='text-align:center; color:#06ffa5;'>Yes</td>
      <td style='text-align:center; color:#06ffa5;'>Yes</td>
    </tr>
    <tr style='border-bottom:1px solid #1e1e3a; background:#0f0f1e;'>
      <td style='padding:9px 16px; color:#dcdcf0;'>Uncertainty per customer (± σ)</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#06ffa5;'>Yes</td>
    </tr>
    <tr style='border-bottom:1px solid #1e1e3a;'>
      <td style='padding:9px 16px; color:#dcdcf0;'>Structural risk (E_eq &lt; E_critical)</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#06ffa5;'>Yes</td>
    </tr>
    <tr style='border-bottom:1px solid #1e1e3a; background:#0f0f1e;'>
      <td style='padding:9px 16px; color:#dcdcf0;'>Recovery time (τ) per customer</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#06ffa5;'>Yes</td>
    </tr>
    <tr style='border-bottom:1px solid #1e1e3a;'>
      <td style='padding:9px 16px; color:#dcdcf0;'>Forward simulation (what-if events)</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#06ffa5;'>Yes</td>
    </tr>
    <tr style='border-bottom:1px solid #1e1e3a; background:#0f0f1e;'>
      <td style='padding:9px 16px; color:#dcdcf0;'>ROI-direct threshold optimization</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#06ffa5;'>Yes</td>
    </tr>
    <tr style='border-bottom:1px solid #1e1e3a;'>
      <td style='padding:9px 16px; color:#dcdcf0;'>Monte Carlo boundary detection</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#ff4d6d;'>No</td>
      <td style='text-align:center; color:#06ffa5;'>Yes</td>
    </tr>
    <tr style='background:#0f0f1e;'>
      <td style='padding:9px 16px; color:#9999c0;'>Max AUC</td>
      <td style='text-align:center; color:#8d8dbd;'>{_fmt_auc(_auc_lr)}</td>
      <td style='text-align:center; color:#4cc9f0;'>{_fmt_auc(_auc_rf)}</td>
      <td style='text-align:center; color:#06ffa5;'>{_fmt_auc(_auc_gbm)}</td>
      <td style='text-align:center; color:#f8961e;'>{_fmt_auc(_auc_phys)}</td>
    </tr>
    </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card card-accent' style='margin-top:16px;'>
    <b>When to use GBM instead</b><br>
    <span class='small'>
    If the only question is "rank these customers by churn risk" and you don't need
    to explain why or simulate what happens next - use GBM. It wins on AUC and trains in seconds.<br><br>
    <b>When to use this model</b><br>
    When the retention team asks "should we call this customer with a discount or redesign their plan?"
    - the physical model is the only one that answers that question without a human in the loop.
    </span>
    </div>
    """, unsafe_allow_html=True)

    # Calibration 
    st.markdown("---")
    st.markdown("### Calibration: are the probabilities trustworthy?")
    st.markdown("<p class='small'>A well-calibrated model where P(churn)=0.7 means 70% of those customers actually churned. Poorly calibrated models are confidently wrong.</p>", unsafe_allow_html=True)

    if "churn" in ctx.df_f.columns and "prob_churn" in ctx.df_f.columns:
        @st.cache_data(show_spinner=False)
        def compute_calibration(probs_tuple, actuals_tuple, n_bins=10):
            import numpy as np  
            from src.business import calibration_analysis
            return calibration_analysis(np.array(actuals_tuple), np.array(probs_tuple), n_bins=n_bins)

        cal = compute_calibration(
            tuple(ctx.df_f["prob_churn"].round(4).tolist()),
            tuple(ctx.df_f["churn"].tolist()),)

        c1, c2, c3 = st.columns(3)
        c1.metric("Brier Score",   f"{cal['brier_score']:.4f}", help="Lower = better. Perfect = 0. Random = churn_rate*(1-churn_rate).")
        c2.metric("Brier baseline",f"{cal['brier_baseline']:.4f}", help="What you'd get predicting the base rate for every customer.")
        c3.metric("Brier skill",   f"{cal.get('brier_skill', 1 - cal['brier_score']/cal['brier_baseline']):.3f}",
                  help="> 0 means the model beats the naive baseline.")

        # Calibration plot: predicted vs actual
        bins = cal.get("bin_centers", np.linspace(0.05, 0.95, 10))
        frac_pos  = cal.get("fraction_positive", np.zeros(10))
        bin_counts= cal.get("bin_counts", np.ones(10))

        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#333355", dash="dash", width=1),
            name="Perfect calibration",))
        fig_cal.add_trace(go.Scatter(
            x=bins, y=frac_pos, mode="lines+markers", name="Physical model",
            line=dict(color="#f8961e", width=2.5),
            marker=dict(size=[max(4, min(18, c / max(bin_counts) * 18)) for c in bin_counts],
                        color="#f8961e"),
            hovertemplate="Predicted: %{x:.2f}<br>Actual: %{y:.2f}<extra></extra>",))
        fig_cal.update_layout(**ctx.plotly_dark(), height=320,
                               xaxis_title="Mean predicted P(churn)",
                               yaxis_title="Fraction actually churned",
                               title="Calibration curve - marker size ∝ number of customers in bin")
        fig_cal.update_xaxes(**ctx.plotly_axis_style(), range=[0, 1])
        fig_cal.update_yaxes(**ctx.plotly_axis_style(), range=[0, 1])
        st.plotly_chart(fig_cal, use_container_width=True, key="t6_cal")

        st.markdown("""
        <div class='card card-accent'>
        <span class='small'>
        Dots above the diagonal = model is underconfident (predicts lower than reality).<br>
        Dots below the diagonal = model is overconfident (predicts higher than reality).<br>
        Perfect calibration = all dots on the diagonal. The physical model's calibration
        reflects the structure of the energy mapping P = 1 - E(t), not tuned for calibration
        specifically, so some deviation is expected.
        </span>
        </div>
        """, unsafe_allow_html=True)
