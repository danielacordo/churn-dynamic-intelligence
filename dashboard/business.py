from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

# Executive Dashboard 
def executive_kpis(df: pd.DataFrame, clv: float, ret_cost: float, horizon: int, success_rate: float = 0.30,) -> dict[str, Any]:
    """ Computes the five headline KPIs shown at the top of the Executive Dashboard"""
    n_total = max(len(df), 1)
    n_high = int((df["risk_level"] == "HIGH").sum())
    n_uncertain = int((df["risk_level"] == "UNCERTAIN").sum())

    high_mask = df["risk_level"] == "HIGH"
    rev_at_risk = float((df.loc[high_mask, "prob_churn"] * clv * horizon).sum())

    est_saved = n_high * success_rate
    rev_recovered = est_saved * clv * horizon
    campaign_cost = n_high * ret_cost
    net_roi = rev_recovered - campaign_cost

    return {
        "n_total": n_total,
        "n_high": n_high,
        "n_uncertain": n_uncertain,
        "pct_high": round(n_high / n_total, 4),
        "rev_at_risk": round(rev_at_risk, 2),
        "est_saved": round(est_saved, 2),
        "rev_recovered": round(rev_recovered, 2),
        "campaign_cost": round(campaign_cost, 2),
        "net_roi": round(net_roi, 2),}


# Campaign Simulation 
def campaign_scenario_table(totals_best: dict[str, float], totals_base: dict[str, float], totals_worst: dict[str, float],) -> pd.DataFrame:
    """Builds the scenario comparison table (best / base / worst case)"""
    rows = []
    for label, t in [("Best case", totals_best), ("Base case", totals_base), ("Worst case", totals_worst)]:
        rows.append({
            "scenario": label, "customers_saved": round(t["saved"], 1), "revenue": round(t["revenue"], 2),
            "cost": round(t["cost"], 2), "roi": round(t["roi"], 2), "roi_pct": round(t["roi_pct"], 2),})
    return pd.DataFrame(rows)


def threshold_roi_curve(df: pd.DataFrame, clv: float, ret_cost: float, horizon: int, success_rate: float = 0.30,n_thresholds: int = 20, ) -> pd.DataFrame:
    """ Computes ROI at each probability threshold for the ROI-vs-threshold chart"""
    clv_total  = clv * horizon
    thresholds = np.linspace(0.20, 0.80, n_thresholds)
    rows = []
    for thr in thresholds:
        flagged = df[df["prob_churn"] >= thr]
        n_f = len(flagged)
        tp_est = float((flagged["prob_churn"] * success_rate).sum())
        revenue = tp_est * clv_total
        cost = n_f * ret_cost
        roi = revenue - cost
        rows.append({
            "threshold": round(float(thr), 3), "n_flagged": n_f,
            "tp_estimate": round(tp_est, 2), "revenue": round(revenue, 2),
            "cost": round(cost, 2), "roi": round(roi, 2),
            "roi_pct": round((roi / cost * 100) if cost > 0 else 0, 2),})
    return pd.DataFrame(rows)


def optimal_threshold(roi_curve: pd.DataFrame) -> dict[str, Any]:
    """Finds the threshold that maximises ROI from a threshold_roi_curve() result"""
    if roi_curve.empty:
        return {"threshold": 0.5, "n_flagged": 0, "roi": 0.0, "roi_pct": 0.0}
    best = roi_curve.loc[roi_curve["roi"].idxmax()]
    return {
        "threshold": float(best["threshold"]),
        "n_flagged": int(best["n_flagged"]),
        "roi": float(best["roi"]),
        "roi_pct": float(best["roi_pct"]),}


# Physics Simulator 
def physics_snapshot(E0: float, gamma: float, E_eq: float, t_horizon: float, E_critical: float = 0.25,) -> dict[str, Any]:
    """ Computes derived physics quantities for a single customer snapshot"""
    tau = 1.0 / (gamma + 1e-15)
    E_t = E_eq + (E0 - E_eq) * np.exp(-gamma * t_horizon)
    E_t = float(np.clip(E_t, 0.0, 1.0))
    delta = E_t - E_eq

    return {
        "tau": round(tau, 2),
        "E_t": round(E_t, 4),
        "delta_E": round(delta, 4),
        "structural_risk": bool(E_eq < E_critical),
        "will_churn": bool(E_t < E_critical),
        "p_churn_approx": round(float(np.clip(1.0 - E_t, 0.0, 1.0)), 4),}


# A/B Test Validator 
def ab_test_break_even_analysis(n_high_risk: int, clv: float, ret_cost: float, horizon: int,) -> dict[str, Any]:
    """ Computes the break-even success rate for the retention campaign

    The break-even is the minimum success_rate such that:
        revenue_recovered >= campaign_cost
        n_high * sr * clv * horizon >= n_high * ret_cost
        sr >= ret_cost / (clv * horizon)"""
    clv_total = clv * horizon
    be = ret_cost / clv_total if clv_total > 0 else float("nan")
    be = float(np.clip(be, 0.0, 1.0)) if not np.isnan(be) else float("nan")

    def _roi(sr: float) -> float:
        return n_high_risk * sr * clv_total - n_high_risk * ret_cost

    roi_30 = _roi(0.30)
    roi_be = _roi(be) if not np.isnan(be) else float("nan")

    return {
        "break_even_rate": round(be, 4) if not np.isnan(be) else float("nan"),
        "roi_at_30_pct": round(roi_30, 2),
        "roi_at_break_even": round(roi_be, 2) if not np.isnan(roi_be) else float("nan"),
        "is_viable": bool(be < 0.50) if not np.isnan(be) else False,}
