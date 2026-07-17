from typing import Any
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

ASSUMPTIONS = {
    'CLV_MONTHLY': 65.0,  # USD, average monthly charge in Telco dataset
    'RETENTION_COST': 25.0, # USD, campaign cost per customer
    'SUCCESS_RATE': 0.30,  # 30% of churners retained with intervention
    'HORIZON_MONTHS': 6,  # months of prediction horizon
}


# Strategic segmentation by physical parameters 
def segment_customer(row: pd.Series) -> dict[str, Any]:
    """ Translates a customer's physical parameters into a strategic segment
    and a concrete action recommendation.

    Note: risk_level must be pre-computed by score_dataframe_with_uncertainty() in src/uncertainty.py before calling
    this function.

    - HIGH Risk: 
               -> if HIGH resilience (tau < 4): Immediate discount, reacts within days
               -> if LOW resilience (tau > 7): Personalized plan, human contact needed
    
    - MEDIUM Risk:
               -> if HIGH resilience (tau < 4): Upsell / upgrade
               -> if LOW resilience (tau > 7): Monitor + alert
    
    - LOW Risk:
               -> if HIGH resilience (tau < 4): Do not intervene
               -> if LOW resilience (tau > 7): Do not intervene

    Additional signal E_eq:
        E_eq < E_critical: structural risk -> customer needs offer change, not just campaign """
    
    from src.physics import E_CRITICAL, classify_resilience
    from src.uncertainty import SIGMA_UNCERTAIN_THRESHOLD

    risk_level = row.get('risk_level', 'MEDIUM')
    tau = float(row.get('tau', 10.0))
    E_eq = float(row.get('E_eq', 0.5))
    sigma_prob = float(row.get('sigma_prob', 0.025))

    # NOTE: previously duplicated its own (tau<5, tau<15) cutoffs here instead of calling classify_resilience, and had silently drifted out of sync with it - 
    # both used stale thresholds under which "Low" resilience (and therefore "High Risk / Fragile" below) was mathematically unreachable given gamma's actual range in this dataset. 
    # Now calls the single source of truth (src.physics.classify_resilience) so the two can't diverge again.
    resilience = classify_resilience(tau)

    if risk_level == 'HIGH':
        priority = 1
        if resilience == 'High':
            segment = 'High Risk / Resilient'
            action = 'Immediate discount or upgrade offer, customer responds quickly'
            reason = f'High churn risk, fast recovery (tau={tau:.1f}m) -> high and fast ROI'
        elif resilience == 'Low':
            segment = 'High Risk / Fragile'
            action = 'Personalized plan + human contact, slow response to incentives'
            reason = f'High churn risk, slow recovery (tau={tau:.1f}m) -> requires patience'
        else:
            segment = 'High Risk / Medium resilience'
            action = 'Targeted offer based on usage pattern'
            reason = f'High churn risk, moderate recovery (tau={tau:.1f}m)'

    elif risk_level == 'MEDIUM':
        priority = 2
        if E_eq < E_CRITICAL:
            segment = 'Structural Risk'
            action = 'Review offer structure, equilibrium below critical threshold'
            reason = f'E_eq={E_eq:.2f} < E_critical={E_CRITICAL} -> will churn without offer change'
        elif resilience == 'High':
            segment = 'Medium Risk / Resilient'
            action = 'Upsell or service upgrade — optimal timing window'
            reason = f'Moderate risk, resilient (tau={tau:.1f}m) -> good upsell candidate'
        else:
            segment = 'Medium Risk / Fragile'
            action = 'Monitor closely + trigger alert if E0 drops'
            reason = f'Moderate risk, low resilience (tau={tau:.1f}m) -> watch carefully'

    elif risk_level == 'UNCERTAIN':
        priority = 3
        segment = 'Uncertain'
        action = 'Collect more data before intervening, high prediction uncertainty'
        reason = f'sigma_prob={sigma_prob:.3f} > {SIGMA_UNCERTAIN_THRESHOLD} (SIGMA_UNCERTAIN_THRESHOLD) -> prediction near decision boundary'

    else:  # LOW
        priority = 4
        segment = 'Stable'
        action = 'Do not invest in retention, loyal customer'
        reason = 'Low churn risk -> spending here is waste'

    return {
        'segment': segment,
        'action': action,
        'priority': priority,
        'reason': reason,}


def apply_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """ Applies strategic segmentation to the full DataFrame

    Requires that df already contains a 'risk_level' column (produced by
    score_dataframe_with_uncertainty)"""
    df = df.copy()
    segmentation = df.apply(segment_customer, axis=1)
    seg_df = pd.DataFrame(segmentation.tolist(), index=df.index)
    df['segment'] = seg_df['segment']
    df['action'] = seg_df['action']
    df['priority'] = seg_df['priority']
    df['reason'] = seg_df['reason']

    return df


# Economic impact calculation 
def compute_economic_impact(df: pd.DataFrame, assumptions: dict[str, Any] | None = None, prob_col: str = 'prob_churn_calibrated', threshold: float = 0.20) -> dict[str, Any]:
    """ Calculates the economic impact of the model on the customer base.

    NOTE: defaults changed from ('prob_churn', 0.31) to ('prob_churn_calibrated', 0.20) after a
    review found the raw prob_churn is systematically overconfident on this dataset (mean ~0.50
    vs an actual churn rate of ~0.265). Every new call site that relied on the old defaults
    instead of passing prob_col/threshold explicitly reproduced the same "intervene on ~everyone,
    ~28% precision" bug (main.py, the Streamlit dashboard, pipeline/steps/score.py) - so the
    defaults themselves are now the calibrated, ROI-optimal ones. 
    Pass prob_col='prob_churn' explicitly if you specifically want the raw, uncalibrated score.

    -> Real-world ROI depends on actual success rates per segment and campaign quality. 
    These are illustrative estimates

    CLV: uses each customer's actual `MonthlyCharges` as their individual CLV when that column
    is present in df, instead of the flat ASSUMPTIONS['CLV_MONTHLY'] average. 
    This was checked, not assumed: MonthlyCharges varies far more *within* a contract type (std ~27) than *between*
    contract-type averages (std ~2.9 across Month-to-month/One year/Two year), so segmenting CLV
    by contract type would have captured almost none of the real variance in what customers
    actually pay - using the per-customer value directly captures all of it, with no new
    assumption, since MonthlyCharges is already an observed column. 
    Falls back to the flat assumption if MonthlyCharges isn't in df (e.g. synthetic test fixtures), so this is backward-compatible with every existing call site."""
    params = {**ASSUMPTIONS, **(assumptions or {})}

    cost = params['RETENTION_COST']
    success_rate = params['SUCCESS_RATE']
    horizon = params['HORIZON_MONTHS']

    total = len(df)
    n_churners = int(df['Churn_bin'].sum()) if 'Churn_bin' in df.columns else 0
    churn_rate = n_churners / total if total > 0 else 0.0

    # Per-customer CLV: real MonthlyCharges when available, flat assumption otherwise.
    if 'MonthlyCharges' in df.columns:
        clv_series = df['MonthlyCharges'].astype(float)
    else:
        clv_series = pd.Series(params['CLV_MONTHLY'], index=df.index)

    # Customers predicted as high-risk
    # BUG FIX: this used to silently fall back to `df` (i.e. treat *every* customer as flagged) when `prob_col` was missing from the dataframe, instead of raising. 
    # That was happening unnoticed in notebooks/11's ROI-sensitivity cell (called with the features-only dataframe,
    # which has no prob_churn column) and quietly produced a "recall=100%, intervene on everyone" ROI curve instead of the intended sensitivity analysis. Fail loudly instead.
    if prob_col not in df.columns:
        raise KeyError(
            f"compute_economic_impact: column '{prob_col}' not found in df. "
            f"Did you mean to pass the scored dataframe (with uncertainty/calibration columns), "
            f"not the features-only one? Available columns: {list(df.columns)[:10]}...")
    intervened = df[df[prob_col] > threshold]
    n_intervened = len(intervened)

    # True positives: actual churners among those flagged
    if 'Churn_bin' in intervened.columns and n_intervened > 0:
        tp_mask = intervened['Churn_bin'] == 1
        tp = int(tp_mask.sum())
        fp = n_intervened - tp
        fn = n_churners - tp
        tn = total - tp - fp - fn
        precision = tp / n_intervened if n_intervened > 0 else 0.0
        recall = tp / n_churners if n_churners > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        # Revenue recovered/at-risk use each true-positive customer's own MonthlyCharges,
        # not a flat average, so a base of high-paying customers contributes proportionally more to the ROI figure than a base of low-paying ones would.
        tp_clv_sum = float(clv_series.loc[intervened.index[tp_mask]].sum()) if tp > 0 else 0.0
    else:
        tp = fp = fn = tn = 0
        precision = recall = accuracy = f1 = 0.0
        tp_clv_sum = 0.0

    # Revenue and cost
    revenue_at_risk = float(clv_series.loc[df['Churn_bin'] == 1].sum()) * horizon if 'Churn_bin' in df.columns else 0.0
    customers_saved = tp * success_rate
    revenue_recovered = tp_clv_sum * success_rate * horizon
    campaign_cost = n_intervened * cost
    net_roi = revenue_recovered - campaign_cost
    roi_pct = (net_roi / campaign_cost * 100) if campaign_cost > 0 else 0.0

    return {
        'total_customers': total,
        'actual_churners': n_churners,
        'churn_rate': churn_rate,
        'n_intervened': n_intervened,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'accuracy': round(accuracy, 4),
        'f1': round(f1, 4),
        'revenue_at_risk_usd': round(revenue_at_risk, 2),
        'revenue_recovered_usd': round(revenue_recovered, 2),
        'campaign_cost_usd': round(campaign_cost, 2),
        'net_roi_usd': round(net_roi, 2),
        'roi_pct': round(roi_pct, 1),
        'customers_saved_est': round(customers_saved, 1),
        'avg_clv_monthly_used': round(float(clv_series.mean()), 2),
        'assumptions': params,}


def impact_table(impact: dict) -> pd.DataFrame:
    """Formats the impact dict into a readable DataFrame for display """
    rows = [
        ('Total customers', impact['total_customers']),
        ('Actual churners', impact['actual_churners']),
        ('Churn rate', f"{impact['churn_rate']:.1%}"),
        ('Customers flagged', impact['n_intervened']),
        ('True positives (TP)', impact['true_positives']),
        ('Precision', f"{impact['precision']:.1%}"),
        ('Recall', f"{impact['recall']:.1%}"),
        ('Revenue at risk ($)', f"${impact['revenue_at_risk_usd']:,.0f}"),
        ('Revenue recovered ($)', f"${impact['revenue_recovered_usd']:,.0f}"),
        ('Campaign cost ($)', f"${impact['campaign_cost_usd']:,.0f}"),
        ('Net ROI ($)', f"${impact['net_roi_usd']:,.0f}"),
        ('ROI (%)', f"{impact['roi_pct']:+.0f}%"),]
    
    return pd.DataFrame(rows, columns=['Metric', 'Value'])


# Model evaluation for baseline comparison 
def evaluate_model_business(y_true: ArrayLike, y_pred_proba: ArrayLike, threshold: float = 0.31, clv_monthly: float = ASSUMPTIONS['CLV_MONTHLY'], retention_cost: float = ASSUMPTIONS['RETENTION_COST'],
                              success_rate: float = ASSUMPTIONS['SUCCESS_RATE'], horizon: int = ASSUMPTIONS['HORIZON_MONTHS'],) -> dict[str, Any]:
    """ Evaluates a model on both technical metrics (AUC, recall, F1) and business metrics (recovered revenue, ROI)"""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,)

    y_pred = (y_pred_proba >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    n_intervened = tp + fp
    customers_saved = tp * success_rate
    clv_total = clv_monthly * horizon
    revenue_recovered = customers_saved * clv_total
    campaign_cost = n_intervened * retention_cost
    net_roi = revenue_recovered - campaign_cost
    revenue_lost = fn * clv_total

    return {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'n_intervened': n_intervened,
        'customers_saved_est': customers_saved,
        'revenue_recovered_usd': revenue_recovered,
        'campaign_cost_usd': campaign_cost,
        'net_roi_usd': net_roi,
        'roi_pct': (net_roi / campaign_cost * 100) if campaign_cost > 0 else 0.0,
        'revenue_lost_usd': revenue_lost,}


# Legacy alias
evaluate_model_business_v1 = evaluate_model_business

def model_comparison_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Builds a comparison table from a dict of model results """
    rows = {}
    for name, r in results.items():
        rows[name] = {
            'AUC': r.get('auc', None),
            'Accuracy': r.get('accuracy', None),
            'Recall churn': r.get('recall', None),
            'Precision': r.get('precision', None),
            'F1': r.get('f1', None),
            'TP': r.get('tp', None),
            'FP': r.get('fp', None),
            'Customers flagged': r.get('n_intervened', None),
            'Customers saved': r.get('customers_saved_est', None),
            'Revenue rec. $': r.get('revenue_recovered_usd', None),
            'Campaign cost $': r.get('campaign_cost_usd', None),
            'ROI net $': r.get('net_roi_usd', None),}
        
    df = pd.DataFrame(rows).T
    df.index.name = 'Model'
    return df.round(4)


# Early detection advantage 
def compute_detection_advantage(df: pd.DataFrame, horizon_months: int = 6) -> dict[str, Any]:
    """ Estimates the early detection advantage of the physical model vs reactive detection.

    Simulates trajectories for actual churners and measures when they cross the critical threshold, then compares to reactive detection timing"""
    from src.physics import detect_threshold_crossing, no_perturbation, solve_trajectory

    churners = df[df['Churn_bin'] == 1].copy()
    crossing_times = []

    for _, row in churners.iterrows():
        sol = solve_trajectory(
            E0=row.get('E0', 0.5),
            gamma=row.get('gamma', 0.1),
            F_func=no_perturbation,
            t_span=(0, 36),
            E_eq=row.get('E_eq', 0.4),)
        result = detect_threshold_crossing(sol.y[0], sol.t)
        crossing_times.append(result.get('t_critical'))

    crossing_times = [t for t in crossing_times if t is not None]
    median_crossing = float(np.median(crossing_times)) if crossing_times else None

    return {
        'horizon_months': horizon_months,
        'advantage_days': horizon_months * 30,
        'n_churners_analyzed': len(churners),
        'n_with_crossing': len(crossing_times),
        'median_crossing_months': median_crossing,
        'description': (
            f"The model detects churn risk {horizon_months * 30} days "
            f"({horizon_months} months) before the critical threshold crossing, "
            f"vs. reactive detection which occurs after churn has happened."),}


# Calibration analysis 
def calibration_analysis(y_true: ArrayLike, y_pred_proba: ArrayLike, n_bins: int = 10) -> dict[str, Any]:
    """ Evaluates probability calibration of the model against true churn labels.

    A well-calibrated model predicts 0.7 probability for customers where ~70% actually churn. 
    Poor calibration means the probabilities are directionally correct but numerically unreliable,
    you can rank customers but not trust the absolute values for budget decisions"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred_proba, dtype=float)
    n = len(y_true)

    # Brier Score
    brier_score = float(np.mean((y_pred - y_true) ** 2))
    churn_rate = float(y_true.mean())
    brier_baseline = float(churn_rate * (1 - churn_rate))
    brier_skill = float(brier_baseline - brier_score)   

    # Reliability diagram bins
    bin_edges  = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[1:-1])   

    bins = []
    weighted_error = 0.0

    for b in range(n_bins):
        mask = bin_indices == b
        n_bin = int(mask.sum())
        if n_bin == 0:
            bins.append({
                'bin_lower': round(float(bin_edges[b]), 3),
                'bin_upper': round(float(bin_edges[b+1]), 3),
                'mean_pred': None,
                'actual_rate': None,
                'n': 0,})
            continue

        mean_pred = float(y_pred[mask].mean())
        actual_rate = float(y_true[mask].mean())
        bin_error = abs(mean_pred - actual_rate)
        weighted_error += (n_bin / n) * bin_error

        # Store unrounded so ECE is exactly verifiable from bins: sum((b['n'] / n) * abs(b['mean_pred'] - b['actual_rate'])) == ece
        bins.append({
            'bin_lower': round(float(bin_edges[b]), 3),
            'bin_upper': round(float(bin_edges[b+1]), 3),
            'mean_pred': mean_pred,
            'actual_rate': actual_rate,
            'n': n_bin,
            'gap': mean_pred - actual_rate,})

    ece = float(weighted_error)
    n_populated = sum(1 for b in bins if b['n'] > 0)

    if ece < 0.05:
        assessment = 'Well calibrated'
    elif ece < 0.10:
        assessment = 'Moderate'
    else:
        assessment = 'Poorly calibrated'

    return {
        'brier_score': round(brier_score, 4),
        'brier_baseline': round(brier_baseline, 4),
        'brier_skill': round(brier_skill, 4),
        'ece': float(ece),  
        'bins': bins,
        'n_bins_populated': n_populated,
        'assessment': assessment,
        'n_samples': n,
        'churn_rate': round(churn_rate, 4),}


def calibration_summary(cal: dict[str, Any]) -> str:
    """ Returns a human-readable summary of calibration results"""
    lines = [
        "Calibration Analysis",
        f" Assessment : {cal['assessment']}",
        f" Brier Score : {cal['brier_score']:.4f}  (baseline: {cal['brier_baseline']:.4f})",
        f" Brier Skill : {cal['brier_skill']:+.4f}  ({'better' if cal['brier_skill'] > 0 else 'worse'} than naive baseline)",
        f" ECE : {cal['ece']:.4f}  (expected calibration error)",
        f" Bins populated : {cal['n_bins_populated']} / {len(cal['bins'])}",
        f" Samples : {cal['n_samples']:,}  |  churn rate: {cal['churn_rate']:.1%}", "",]
    return "\n".join(lines)


# Sensitivity analysis 
def sensitivity_analysis(n_intervened: int, n_true_positives: int, clv_monthly_range: tuple = (40.0, 120.0, 9), success_rate_range: tuple = (0.10, 0.50,  9),
    retention_cost: float = ASSUMPTIONS['RETENTION_COST'], horizon_months: int = ASSUMPTIONS['HORIZON_MONTHS'],) -> dict[str, Any]:
    """Computes net ROI across a grid of CLV and success_rate values.

    Answers the question: "How sensitive is our ROI estimate to the
    business assumptions we've hardcoded?"

    The tornado chart variant ranks each assumption by its individual impact on ROI, 
    useful for identifying which assumption drives the result most"""
    if n_intervened < 0:
        raise ValueError(f"n_intervened must be >= 0, got {n_intervened}")
    if n_true_positives < 0:
        raise ValueError(f"n_true_positives must be >= 0, got {n_true_positives}")
    if n_true_positives > n_intervened:
        raise ValueError(
            f"n_true_positives ({n_true_positives}) cannot exceed "
            f"n_intervened ({n_intervened}): true positives are a subset of intervened customers" )
    if retention_cost < 0:
        raise ValueError(f"retention_cost must be >= 0, got {retention_cost}")
    if horizon_months < 1:
        raise ValueError(f"horizon_months must be >= 1, got {horizon_months}")

    clv_values = np.linspace(*clv_monthly_range)
    sr_values = np.linspace(*success_rate_range)
    campaign_cost = float(n_intervened * retention_cost)

    # 2D ROI grid: rows = CLV, cols = success_rate
    roi_grid = np.zeros((len(clv_values), len(sr_values)))
    for i, clv in enumerate(clv_values):
        for j, sr in enumerate(sr_values):
            revenue = n_true_positives * sr * clv * horizon_months
            roi_grid[i, j] = revenue - campaign_cost

    # Base ROI at default assumptions
    base_clv = ASSUMPTIONS['CLV_MONTHLY']
    base_sr = ASSUMPTIONS['SUCCESS_RATE']
    base_roi = float(
        n_true_positives * base_sr * base_clv * horizon_months - campaign_cost)

    # Tornado: vary each assumption one at a time, holding others at base
    def _roi(clv, sr, cost, h):
        return n_true_positives * sr * clv * h - n_intervened * cost

    tornado = []

    for assumption, low, high in [
        ('CLV monthly ($)', clv_monthly_range[0], clv_monthly_range[1]),
        ('Success rate', success_rate_range[0], success_rate_range[1]),
        ('Retention cost ($)', retention_cost * 0.5, retention_cost * 2.0),
        ('Horizon (months)', max(1, horizon_months-3), horizon_months + 6),
    ]:
        if assumption == 'CLV monthly ($)':
            roi_low = _roi(low,  base_sr, retention_cost, horizon_months)
            roi_high = _roi(high, base_sr, retention_cost, horizon_months)
        elif assumption == 'Success rate':
            roi_low = _roi(base_clv, low, retention_cost, horizon_months)
            roi_high = _roi(base_clv, high, retention_cost, horizon_months)
        elif assumption == 'Retention cost ($)':
            roi_low = _roi(base_clv, base_sr, high, horizon_months)   # high cost -> low ROI
            roi_high = _roi(base_clv, base_sr, low,  horizon_months)
        else:  # Horizon
            roi_low  = _roi(base_clv, base_sr, retention_cost, low)
            roi_high = _roi(base_clv, base_sr, retention_cost, high)

        tornado.append({
            'assumption': assumption,
            'roi_low': round(float(roi_low),  2),
            'roi_high': round(float(roi_high), 2),
            'impact': round(float(abs(roi_high - roi_low)), 2),})

    # Sort tornado by impact (descending)
    tornado.sort(key=lambda x: x['impact'], reverse=True)

    return {
        'roi_grid': roi_grid,
        'clv_values':clv_values,
        'success_rate_values': sr_values,
        'campaign_cost': round(campaign_cost, 2),
        'base_roi': round(base_roi, 2),
        'tornado': tornado,
        'n_intervened': n_intervened,
        'n_true_positives': n_true_positives,}


def sensitivity_summary(sens: dict[str, Any]) -> str:
    """ Returns a human-readable tornado summary from sensitivity_analysis()"""
    lines = [
        " Sensitivity Analysis (Tornado)",
        f" Base ROI : ${sens['base_roi']:>+,.0f}",
        f" Campaign cost : ${sens['campaign_cost']:>,.0f} (fixed)",
        f" Customers flagged: {sens['n_intervened']:,}  |  TP: {sens['n_true_positives']:,}",
        "",
        f" {'Assumption':<25} {'Low':>10} {'High':>10} {'Swing':>10}", "",]
    
    for row in sens['tornado']:
        lines.append(
            f" {row['assumption']:<25} "
            f"${row['roi_low']:>+9,.0f} "
            f"${row['roi_high']:>+9,.0f} "
            f"${row['impact']:>9,.0f}")
    lines.append("")
    return "\n".join(lines)


# Threshold optimization 
def optimize_threshold(y_true, y_pred_proba, clv_monthly: float = ASSUMPTIONS['CLV_MONTHLY'],
    retention_cost: float = ASSUMPTIONS['RETENTION_COST'], success_rate: float = ASSUMPTIONS['SUCCESS_RATE'],
    horizon_months: int = ASSUMPTIONS['HORIZON_MONTHS'], n_thresholds: int = 200,
    clv_per_customer: ArrayLike | None = None,) -> dict[str, Any]:
    """ Finds the classification threshold that maximizes net ROI.

    Standard ML practice optimizes F1 or recall at a fixed threshold.
    This function optimizes for the business objective directly: net_roi(t) = TP(t) * success_rate * CLV * horizon - (TP+FP)(t) * cost

    The F1-optimal threshold is included for comparison. The difference quantifies the cost of optimizing the wrong metric.

    clv_per_customer: optional array aligned with y_true/y_pred_proba giving each customer's own CLV (e.g. their MonthlyCharges), mirroring compute_economic_impact's per-customer CLV.
    When omitted, falls back to the flat clv_monthly scalar for every customer - this keeps every existing call site (and all pre-existing tests) working unchanged."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred_proba, dtype=float)
    if clv_per_customer is not None:
        clv_arr = np.asarray(clv_per_customer, dtype=float) * horizon_months
    else:
        clv_arr = np.full_like(y_true, clv_monthly * horizon_months)

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    curve = []
    best_roi = -np.inf
    best_t = 0.5
    best_f1 = -np.inf
    best_f1_t = 0.5

    for t in thresholds:
        y_pred_bin = (y_pred >= t).astype(int)
        tp_mask = (y_pred_bin == 1) & (y_true == 1)
        fp_mask = (y_pred_bin == 1) & (y_true == 0)
        tp = int(tp_mask.sum())
        fp = int(fp_mask.sum())
        fn = int(((y_pred_bin == 0) & (y_true == 1)).sum())
        n_intervened = tp + fp

        revenue = float(clv_arr[tp_mask].sum()) * success_rate
        cost = n_intervened * retention_cost
        roi = revenue - cost

        precision = tp / n_intervened if n_intervened > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        curve.append({
            'threshold': round(float(t), 4),
            'roi': round(float(roi), 2),
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1': round(float(f1), 4),
            'tp': tp, 'fp': fp, 'n_intervened': n_intervened,})

        if roi > best_roi:
            best_roi = roi
            best_t = float(t)
            best_tp, best_fp, _ = tp, fp, fn
            best_precision, best_recall, best_f1_val = precision, recall, f1
            best_n_intervened = n_intervened

        if f1 > best_f1:
            best_f1 = f1
            best_f1_t = float(t)

    # ROI at F1-optimal threshold
    f1_bin = (y_pred >= best_f1_t).astype(int)
    f1_tp_mask = (f1_bin == 1) & (y_true == 1)
    f1_fp = int(((f1_bin == 1) & (y_true == 0)).sum())
    f1_roi = float(clv_arr[f1_tp_mask].sum()) * success_rate - (int(f1_tp_mask.sum()) + f1_fp) * retention_cost

    # ROI at default threshold 0.50
    default_bin = (y_pred >= 0.50).astype(int)
    d_tp_mask = (default_bin == 1) & (y_true == 1)
    d_fp = int(((default_bin == 1) & (y_true == 0)).sum())
    default_roi = float(clv_arr[d_tp_mask].sum()) * success_rate - (int(d_tp_mask.sum()) + d_fp) * retention_cost

    return {
        'optimal_threshold': round(best_t, 4),
        'optimal_roi': round(float(best_roi), 2),
        'f1_threshold': round(best_f1_t, 4),
        'f1_roi': round(f1_roi, 2),
        'roi_gain': round(float(best_roi) - f1_roi, 2),
        'default_threshold_roi': round(default_roi, 2),
        'optimal_metrics': {
            'precision': round(best_precision, 4),
            'recall': round(best_recall, 4),
            'f1': round(best_f1_val, 4),
            'tp': best_tp,
            'fp': best_fp,
            'n_intervened': best_n_intervened,},
        'curve': curve,
        'assumptions': {
            'clv_monthly': clv_monthly,
            'retention_cost': retention_cost,
            'success_rate': success_rate,
            'horizon_months': horizon_months,
            # Makes explicit whether clv_monthly (the flat scalar above) was actually used for this calculation, or whether clv_per_customer took priority - 
            # reading clv_monthly alone here could otherwise mislead a caller into thinking the flat assumption drove the result even when a per-customer CLV array was passed and used instead 
            # (see DECISIONS.md #12/#13 for why this distinction matters).
            'clv_source': 'clv_per_customer' if clv_per_customer is not None else 'clv_monthly',
            'avg_clv_used': round(float(clv_arr.mean() / horizon_months), 2),},}


def threshold_summary(opt: dict[str, Any]) -> str:
    """ Returns a human-readable summary of optimize_threshold() results"""
    m = opt['optimal_metrics']
    a = opt['assumptions']
    clv_note = (f"${a['avg_clv_used']:,.2f}/mo (per-customer avg)" if a['clv_source'] == 'clv_per_customer'
                else f"${a['avg_clv_used']:,.2f}/mo (flat assumption)")
    lines = [
        " Threshold Optimization",
        f" CLV used : {clv_note}",
        f" Optimal threshold : {opt['optimal_threshold']:.2f}  (maximizes net ROI)",
        f" Optimal ROI : ${opt['optimal_roi']:>+,.0f}",
        f" F1-optimal threshold: {opt['f1_threshold']:.2f}",
        f" F1-optimal ROI : ${opt['f1_roi']:>+,.0f}",
        f" ROI gain vs F1 : ${opt['roi_gain']:>+,.0f}  ({'better' if opt['roi_gain'] > 0 else 'worse'} by optimizing ROI directly)",
        f" Default (0.50) ROI : ${opt['default_threshold_roi']:>+,.0f}",
        "",
        "  At optimal threshold:",
        f" Precision : {m['precision']:.1%}",
        f" Recall : {m['recall']:.1%}",
        f" F1 : {m['f1']:.4f}",
        f" Customers flagged: {m['n_intervened']:,}  (TP={m['tp']:,}  FP={m['fp']:,})",
        "",]
    return "\n".join(lines)