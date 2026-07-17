import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

# Page config 
st.set_page_config(
    page_title="Churn as a Dynamic System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",)

# Theme 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg: #080810;
    --panel: #0f0f1e;
    --border: #1e1e3a;
    --text: #dcdcf0;
    --accent: #f8961e;
    --churn: #ff4d6d;
    --safe: #4cc9f0;
    --green: #06ffa5;
    --yellow: #ffbe0b;
    --purple: #9b5de5;}

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg);
    color: var(--text);}

h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    color: white !important;
    letter-spacing: -0.5px;}

.stApp { background-color: var(--bg); }

section[data-testid="stSidebar"] {
    background-color: var(--panel);
    border-right: 1px solid var(--border);}

/* st.metric() widgets default to Streamlit's light-theme text color (a dark navy, meant for
   a white background) -- on this dark theme that renders as near-invisible dark-on-dark text.
   Used 38 times across the dashboard (tab1, tab3-7). The `metric-container`-scoped rules below
   are from an earlier attempt at this same fix and don't match current Streamlit's DOM (verified
   via computed style: the values were still rendering at rgb(49,51,63), not white) -- kept for
   the container background/border, but the actual text-color fix needs the direct selectors. */
[data-testid="stMetricValue"] {
    color: white !important;
    font-family: 'Syne', sans-serif !important;}
[data-testid="stMetricLabel"] p {
    color: #a8a8d0 !important;
    font-size: 12px !important;}

div[data-testid="metric-container"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;}
            
div[data-testid="metric-container"] label {
    color: var(--text) !important;
    font-size: 11px !important;
    letter-spacing: 1px;
    text-transform: uppercase;}
            
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    color: white !important;
}

button[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: var(--text) !important;
    border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

div[data-testid="stSlider"] > div > div > div {
    background: var(--accent) !important;
}

/*  Slider labels  */
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] label p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label p {
    color: #e0e0f8 !important;
    font-size: 13px !important;
}

.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px; }

/*  Dark theme for dataframes / tables  */
div[data-testid="stDataFrame"] iframe,
div[data-testid="stDataFrame"] > div {
    background: var(--panel) !important;
    border-radius: 8px;
}
.stDataFrame [class*="dvn-scroller"] { background: var(--panel) !important; }

/*  Dark theme for selectbox / multiselect / text inputs  */
div[data-baseweb="select"] > div,
div[data-baseweb="select"] ul,
div[data-baseweb="popover"] {
    background-color: #1a1a2e !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: var(--text) !important;
}
li[role="option"] {
    background-color: #1a1a2e !important;
    color: var(--text) !important;
}
li[role="option"]:hover,
li[aria-selected="true"] {
    background-color: #2a2a4e !important;
    color: white !important;
}
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background-color: #1a1a2e !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/*  Dark theme for sliders and number inputs  */
div[data-testid="stNumberInput"] input {
    background-color: #1a1a2e !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

/*  st.dataframe internal dark styling  */
[data-testid="stDataFrame"] th {
    background: #0f0f1e !important;
    color: var(--text) !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] td {
    background: #0d0d1a !important;
    color: var(--text) !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] tr:hover td {
    background: #1a1a2e !important;
}

hr { border-color: var(--border) !important; }

.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.card-accent { border-left: 3px solid var(--accent); }
.card-churn  { border-left: 3px solid var(--churn); }
.card-safe { border-left: 3px solid var(--safe); }
.card-green  { border-left: 3px solid var(--green); }

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-high { background: rgba(255,77,109,0.15); color: #ff4d6d; border: 1px solid #ff4d6d44; }
.badge-medium { background: rgba(248,150,30,0.15); color: #f8961e; border: 1px solid #f8961e44; }
.badge-low { background: rgba(76,201,240,0.15); color: #4cc9f0; border: 1px solid #4cc9f044; }
.badge-uncertain{ background: rgba(155,93,229,0.15); color: #9b5de5; border: 1px solid #9b5de544; }

.scenario-best { border-left: 3px solid #06ffa5; }
.scenario-base { border-left: 3px solid #f8961e; }
.scenario-worst  { border-left: 3px solid #ff4d6d; }

.mono { font-family: 'JetBrains Mono', monospace; }
.small { font-size: 12px; color: #a8a8d0; }

/* Streamlit wraps injected HTML (cards, paragraphs) in these containers; without this,
   Streamlit's own default paragraph color can win over the broader html/body rule above,
   leaving card body text (e.g. tab7's intro cards) too dim to read. */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] div,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] b {
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)


# Physics / uncertainty / business imports
# NOTE: these have to come after st.set_page_config()/st.markdown() above (Streamlit requires
# set_page_config to be the first Streamlit call), hence the noqa: E402 on each.

from src.business import (  # noqa: E402
    calibration_analysis as _calibration_analysis,
    optimize_threshold as _optimize_threshold,
    segment_customer as _segment_customer_row,
    sensitivity_analysis as _sensitivity_analysis,
)
from src.physics import (  # noqa: E402
    E_CRITICAL,
    combined_perturbation,
    no_perturbation,
    periodic_perturbation,
    step_perturbation,
    compute_damping as _compute_gamma,
    compute_E_eq as _compute_E_eq,
    compute_energy as _compute_E0,
    solve_trajectory as _solve_trajectory,
)
from src.uncertainty import (  # noqa: E402
    SIGMA_UNCERTAIN_THRESHOLD as SIGMA_THRESH,
    churn_probability_with_uncertainty as _churn_prob_core,
    compare_uncertainty_methods as _compare_methods,
    monte_carlo_churn as _monte_carlo_churn,
)

# ── App-level constants ────────────────────────────────────────────────────────

SUCCESS_RATES = {
    "High Risk / Resilient":         0.45,
    "High Risk / Fragile":           0.20,
    "High Risk / Medium resilience": 0.30,
    "Medium Risk / Resilient":       0.15,
    "Medium Risk / Fragile":         0.08,
    "Structural Risk":               0.10,
    "Stable":                        0.00,
    "Uncertain":                     0.05,
}
COST_MULTS = {
    "High Risk / Resilient":         1.0,
    "High Risk / Fragile":           2.0,
    "High Risk / Medium resilience": 1.5,
    "Medium Risk / Resilient":       0.8,
    "Medium Risk / Fragile":         1.2,
    "Structural Risk":               2.5,
    "Stable":                        0.0,
    "Uncertain":                     0.5,
}

RISK_COLORS = {"HIGH": "#ff4d6d", "MEDIUM": "#f8961e", "LOW": "#4cc9f0", "UNCERTAIN": "#9b5de5"}
RISK_BADGE  = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low", "UNCERTAIN": "badge-uncertain"}


# ── Thin adapters ──────────────────────────────────────────────────────────────

def compute_E_eq(contract, n_services_protective, n_services_entertainment, auto_pay, internet_type="No", payment_method=None):
    return _compute_E_eq(contract, n_services_protective, n_services_entertainment, auto_pay, internet_type=internet_type, payment_method=payment_method)

def compute_gamma(contract, n_services_protective, n_services_entertainment, auto_pay, internet_type="No", payment_method=None):
    return _compute_gamma(contract, n_services_protective, n_services_entertainment, auto_pay, internet_type=internet_type, payment_method=payment_method)

def compute_E0(tenure, monthly, total, n_services_protective, n_services_entertainment, long_contract, internet_type="No"):
    return _compute_E0(
        tenure=tenure, monthly_charges=monthly, total_charges=total,
        num_services_protective=n_services_protective, num_services_entertainment=n_services_entertainment,
        has_long_contract=long_contract, internet_type=internet_type,
    )

def solve_trajectory(E0, gamma, E_eq, F_func, t_span=(0, 36), n=300):
    t_eval = np.linspace(*t_span, n)
    sol = _solve_trajectory(E0, gamma, F_func, t_span, t_eval=t_eval, E_eq=E_eq)
    return sol.t, sol.y[0]

def churn_prob_with_uncertainty(E0, sigma_E0, gamma, sigma_gamma, E_eq, sigma_Eeq, t_h):
    result = _churn_prob_core(
        E0=E0, sigma_E0=sigma_E0, gamma=gamma, sigma_gamma=sigma_gamma,
        t_horizon=t_h, E_eq=E_eq, sigma_E_eq=sigma_Eeq,
    )
    return result["prob"], result["sigma_prob"], result["lower_bound"], result["upper_bound"], result["risk_level"]

def segment_customer(risk, tau, E_eq_val):
    row = pd.Series({"risk_level": risk, "tau": tau, "E_eq": E_eq_val, "sigma_prob": 0.025})
    result = _segment_customer_row(row)
    return result["segment"], result["action"]


# Plotly theme 
def plotly_dark():
    """Base dark theme for update_layout(). Does NOT include xaxis/yaxis/legend/margin
    so callers can pass those without duplicate-keyword errors."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f0f1e",
        font=dict(color="#dcdcf0", family="JetBrains Mono"),)


def plotly_axis_style():
    """Grid/zeroline styling to apply via fig.update_xaxes(**ctx.plotly_axis_style())
    and fig.update_yaxes(**ctx.plotly_axis_style()) after update_layout()."""
    return dict(gridcolor="#1e1e3a", zerolinecolor="#1e1e3a")


def plotly_legend_style():
    """Legend styling: fig.update_layout(legend=dict(**ctx.plotly_legend_style(), x=0.01, y=0.99))."""
    return dict(bgcolor="#0f0f1e", bordercolor="#1e1e3a")


# Campaign metrics (shared by Tab 3 and Tab 4) 
def compute_campaign_metrics(df_seg, clv, ret_cost, horizon, success_rate_override=None):
    seg_counts = df_seg["segment"].value_counts()
    rows = []
    for seg, n_seg in seg_counts.items():
        sr  = success_rate_override if success_rate_override is not None else SUCCESS_RATES.get(seg, 0.0)
        cm  = COST_MULTS.get(seg, 1.0)
        saved, revenue = n_seg * sr, n_seg * sr * clv * horizon
        cost = n_seg * ret_cost * cm
        roi  = revenue - cost
        rows.append({"segment": seg, "n": int(n_seg), "success_rate": sr,
                     "saved": saved, "revenue": revenue, "cost": cost,
                     "roi": roi, "roi_pct": (roi / cost * 100) if cost > 0 else 0})
    sim = pd.DataFrame(rows)
    totals = {"saved": sim["saved"].sum(), "revenue": sim["revenue"].sum(),
              "cost": sim["cost"].sum(), "roi": sim["roi"].sum(),
              "roi_pct": (sim["roi"].sum() / sim["cost"].sum() * 100) if sim["cost"].sum() > 0 else 0}
    return sim, totals


# Data loading 
@st.cache_data
def load_real_dataset(path: str) -> pd.DataFrame:
    from src.business import apply_segmentation
    from src.features import build_physical_features
    from src.uncertainty import calibrate_dataframe_probabilities, score_dataframe_with_uncertainty

    raw = pd.read_csv(path)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    raw["TotalCharges"].fillna(raw["MonthlyCharges"], inplace=True)
    raw["Churn_bin"] = (raw["Churn"] == "Yes").astype(int)
    raw = raw.drop_duplicates(subset="customerID")
    df = build_physical_features(raw)
    df = score_dataframe_with_uncertainty(df, t_horizon=6.0)

    # NOTE (bug fix): raw prob_churn is systematically overconfident on this dataset (mean
    # ~0.50 vs an actual churn rate of ~0.265) -- isotonic-calibrate it before anything downstream
    # uses it, same as pipeline/steps/score.py and update_scores.py. Without this, the dashboard
    # was flagging the large majority of customers as HIGH risk with poor precision.
    df, _calibrator = calibrate_dataframe_probabilities(df)
    df["prob_churn"] = df["prob_churn_calibrated"]
    df["prob_inf"] = df["prob_inf_calibrated"]
    df["prob_sup"] = df["prob_sup_calibrated"]

    df = apply_segmentation(df)
    df = df.rename(columns={
        "Contract": "contract", "tenure": "tenure", "MonthlyCharges": "monthly",
        "num_services": "n_services", "num_services_protective": "n_services_protective",
        "num_services_entertainment": "n_services_entertainment",
        "InternetService": "internet_type",
        "auto_payment": "auto_pay", "Churn_bin": "churn",
        "prob_churn": "prob_churn", "sigma_prob": "sigma_prob",
        # NOTE (bug fix): this used to map "lower_bound"/"upper_bound", which don't exist --
        # score_dataframe_with_uncertainty produces "prob_inf"/"prob_sup" -- so "lb"/"ub" were
        # silently never created here (rename() ignores missing keys) until _recompute() ran.
        "prob_inf": "lb", "prob_sup": "ub",})
    if "customerID" not in df.columns:
        df["customerID"] = [f"CUST-{i+1:04d}" for i in range(len(df))]
    if "revenue_at_risk" not in df.columns:
        df["revenue_at_risk"] = df["prob_churn"] * 65.0 * 6
    return df


@st.cache_data
def generate_synthetic_dataset(n=500, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    contracts = np.random.choice(["Month-to-month","One year","Two year"], n, p=[0.55,0.25,0.20])
    payments  = np.random.choice(
        ["Bank transfer (automatic)","Credit card (automatic)","Electronic check","Mailed check"],
        n, p=[0.22,0.22,0.33,0.23])
    tenure = np.random.randint(0, 73, n)
    monthly  = np.random.uniform(20, 110, n)
    n_srv_protective = np.random.randint(0, 5, n)   # 0-4: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
    n_srv_entertainment = np.random.randint(0, 3, n)  # 0-2: StreamingTV, StreamingMovies
    internet_types = np.random.choice(["No", "DSL", "Fiber optic"], n, p=[0.22, 0.34, 0.44])
    auto_pay = np.isin(payments, ["Bank transfer (automatic)","Credit card (automatic)"])
    # payment_method gives compute_E_eq/compute_gamma the full 3-way distinction (auto / mailed
    # check / electronic check) instead of collapsing to the auto_pay binary -- see
    # src/physics_calibration.py for why electronic check is a materially different signal from
    # mailed check, not just "non-automatic".
    payment_method = np.where(
        auto_pay, "auto", np.where(payments == "Mailed check", "mailed_check", "electronic_check"))
    long_c = np.isin(contracts, ["One year","Two year"])
    total = monthly * np.maximum(tenure, 1) * np.random.uniform(0.9, 1.1, n)
    rows = []
    for i in range(n):
        E0 = compute_E0(tenure[i], monthly[i], total[i], n_srv_protective[i], n_srv_entertainment[i], long_c[i], internet_types[i])
        E_eq  = compute_E_eq(contracts[i], n_srv_protective[i], n_srv_entertainment[i], auto_pay[i], internet_types[i], payment_method=payment_method[i])
        gamma = compute_gamma(contracts[i], n_srv_protective[i], n_srv_entertainment[i], auto_pay[i], internet_types[i], payment_method=payment_method[i])
        tau = 1 / (gamma + 1e-9)
        prob, sigma, lb, ub, risk = churn_prob_with_uncertainty(E0, 0.05, gamma, 0.02, E_eq, 0.03, 6.0)
        seg, action = segment_customer(risk, tau, E_eq)
        rows.append({
            "customerID": f"CUST-{i+1:04d}", "contract": contracts[i],
            "tenure": int(tenure[i]), "monthly": round(float(monthly[i]), 2),
            "n_services": int(n_srv_protective[i] + n_srv_entertainment[i]),
            "n_services_protective": int(n_srv_protective[i]), "n_services_entertainment": int(n_srv_entertainment[i]),
            "internet_type": internet_types[i],
            "auto_pay": bool(auto_pay[i]),
            "E0": round(E0,3), "E_eq": round(E_eq,3), "gamma": round(gamma,3), "tau": round(tau,1),
            "prob_churn": round(prob,3), "sigma_prob": round(sigma,3),
            "lb": round(lb,3), "ub": round(ub,3), "risk_level": risk, "segment": seg, "action": action,
            "churn": int(np.random.random() < np.clip(prob + np.random.normal(0,0.1), 0, 1)),
            "revenue_at_risk": round(prob * 65.0 * 6, 2),})
    return pd.DataFrame(rows)


# Session state 
for _key in ("selected_segment", "selected_risk"):
    if _key not in st.session_state:
        st.session_state[_key] = "All (no filter)"


# Load data 
_REAL_DATA_PATH = "data/telco_churn.csv"
_using_real_data = False

if os.path.exists(_REAL_DATA_PATH):
    try:
        df = load_real_dataset(_REAL_DATA_PATH)
        _using_real_data = True
    except Exception:
        df = generate_synthetic_dataset(500)
else:
    df = generate_synthetic_dataset(500)

if not _using_real_data:
    st.warning(
        "**Demo mode - synthetic data.** "
        "Download the [IBM Telco dataset from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) "
        "and save it as `data/telco_churn.csv` to run on real data.")
else:
    st.success(f"Real dataset loaded — {len(df):,} customers · {df['churn'].mean():.1%} churn rate")


# Sidebar 
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px 0;'>
        <div style='font-family: Syne; font-size: 18px; font-weight: 800; color: white;'>Churn</div>
        <div style='font-family: Syne; font-size: 11px; color: #f8961e; letter-spacing: 2px; text-transform: uppercase;'>Dynamic System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Business parameters**")
    clv = st.slider("CLV monthly ($)", 20, 200, 65, 5)
    ret_cost  = st.slider("Retention cost ($)", 5, 100, 25, 5)
    horizon = st.slider("Prediction horizon (months)", 1, 12, 6)
    threshold = st.slider("Risk threshold", 0.20, 0.70, 0.20, 0.01)

    st.divider()
    st.markdown("**Filter dataset**")
    contracts_filter = st.multiselect("Contract type",
        ["Month-to-month","One year","Two year"],
        default=["Month-to-month","One year","Two year"])
    risk_filter = st.multiselect("Risk level",
        ["HIGH","MEDIUM","LOW","UNCERTAIN"],
        default=["HIGH","MEDIUM","LOW","UNCERTAIN"])

    st.divider()
    st.markdown("""
    <div class='small'>
    Dataset: IBM Telco<br>
    Model: dE/dt = -γ(E - E_eq) + F(t)<br>
    Tests: 356/356 passing<br><br>
    <a href='https://github.com/danielacordo/churn-dynamic-intelligence' style='color:#f8961e;'>→ GitHub repo</a>
    </div>
    """, unsafe_allow_html=True)


# Filter + recompute 
df_f = df[df["contract"].isin(contracts_filter) & df["risk_level"].isin(risk_filter)].copy()

def _recompute(row):
    prob, sigma, lb, ub, risk = churn_prob_with_uncertainty(
        row["E0"], 0.05, row["gamma"], 0.02, row["E_eq"], 0.03, horizon)
    seg, action = segment_customer(risk, row["tau"], row["E_eq"])
    return pd.Series({"prob_churn": round(prob,3), "sigma_prob": round(sigma,3),
                      "lb": round(lb,3), "ub": round(ub,3), "risk_level": risk,
                      "segment": seg, "action": action, "revenue_at_risk": round(prob * clv * horizon, 2)})

df_f[["prob_churn","sigma_prob","lb","ub","risk_level","segment","action","revenue_at_risk"]] =     df_f.apply(_recompute, axis=1)

# NOTE (bug fix): _recompute() calls the raw physics formula directly (with fixed placeholder
# sigmas), bypassing calibration the same way load_real_dataset() used to. Re-calibrate in-sample
# against the currently filtered slice whenever there's enough label variation to fit against --
# small or single-class slices (e.g. filtered to one risk level) fall back to the uncalibrated
# score rather than fitting isotonic regression on too little signal.
if "churn" in df_f.columns and df_f["churn"].nunique() > 1 and len(df_f) >= 30:
    from src.uncertainty import calibrate_dataframe_probabilities as _calibrate_df_f
    df_f, _ = _calibrate_df_f(df_f, prob_col="prob_churn", lower_col="lb", upper_col="ub", target_col="churn")
    df_f["prob_churn"] = df_f["prob_churn_calibrated"]
    df_f["lb"] = df_f["prob_inf_calibrated"]
    df_f["ub"] = df_f["prob_sup_calibrated"]


# Header 
st.markdown("""
<div style='padding: 12px 0 24px 0;'>
    <h1 style='font-size: 36px; margin-bottom: 4px;'>Churn as a Dynamic System</h1>
    <p style='color: #f8961e; font-size: 13px; letter-spacing: 2px; text-transform: uppercase; margin: 0;'>
        Physics-Informed Customer Retention Analysis
    </p>
</div>
""", unsafe_allow_html=True)


#  Build shared context 
from dashboard.tabs.context import DashboardContext  # noqa: E402

ctx = DashboardContext(
    df=df, df_f=df_f, using_real_data=_using_real_data,
    clv=clv, ret_cost=ret_cost, horizon=horizon, threshold=threshold,
    E_CRITICAL=E_CRITICAL, SIGMA_THRESH=SIGMA_THRESH,
    RISK_COLORS=RISK_COLORS, RISK_BADGE=RISK_BADGE,
    SUCCESS_RATES=SUCCESS_RATES, COST_MULTS=COST_MULTS,
    plotly_dark=plotly_dark,
    plotly_axis_style=plotly_axis_style,
    plotly_legend_style=plotly_legend_style,
    compute_campaign_metrics=compute_campaign_metrics,
    solve_trajectory=solve_trajectory,
    compute_E0=compute_E0,
    compute_E_eq=compute_E_eq,
    compute_gamma=compute_gamma,
    churn_prob_with_uncertainty=churn_prob_with_uncertainty,
    segment_customer=segment_customer,
    no_perturbation=no_perturbation,
    step_perturbation=step_perturbation,
    periodic_perturbation=periodic_perturbation,
    combined_perturbation=combined_perturbation,
    monte_carlo_churn=_monte_carlo_churn,
    compare_methods=_compare_methods,
    sensitivity_analysis=_sensitivity_analysis,
    optimize_threshold=_optimize_threshold,
    calibration_analysis=_calibration_analysis,)


# Tabs 
from dashboard.tabs import tab1, tab2, tab3, tab4, tab5, tab6, tab7  # noqa: E402

w1, w2, w3, w4, w5, w6, w7 = st.tabs([
    "Executive Dashboard",
    "Customer Prioritization",
    "Campaign Simulation",
    "Physics Simulator",
    "Monte Carlo Analysis",
    "Model Validation",
    "A/B Test Validator",])

with w1:
    tab1.render(ctx)
with w2:
    tab2.render(ctx)
with w3:
    tab3.render(ctx)
with w4:
    tab4.render(ctx)
with w5:
    tab5.render(ctx)
with w6:
    tab6.render(ctx)
with w7:
    tab7.render(ctx)
