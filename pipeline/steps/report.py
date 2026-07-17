import argparse
import json
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

def report(input_path: str, output_dir: str = "report", assumptions: dict | None = None, threshold: float = 0.20,
           prob_col: str = "prob_churn_calibrated",) -> dict:
    """Generates the executive impact report for the model.

    NOTE: defaults to the isotonic-calibrated probability column (prob_churn_calibrated),
    produced by pipeline.steps.score since the raw prob_churn is systematically overconfident
    on this dataset (see pipeline/steps/score.py and src/uncertainty.py)."""
    from src.ab_test import (
        ABTestConfig, analyse_experiment,
        required_sample_size, simulate_experiment,)
    from src.business import ASSUMPTIONS, compute_economic_impact

    effective = {**ASSUMPTIONS, **(assumptions or {})}

    logger.info(f"[report] Reading {input_path}")
    df = pd.read_csv(input_path)

    impact = compute_economic_impact(df, assumptions=effective, threshold=threshold, prob_col=prob_col)

    risk_dist = df['risk_level'].value_counts().to_dict() if 'risk_level' in df.columns else {}
    segment_dist = df['segment'].value_counts().to_dict() if 'segment' in df.columns else {}
    resilience_dist = (
        df['resilience'].value_counts(normalize=True).round(4).to_dict()
        if 'resilience' in df.columns else {})

    physical_by_churn = {}
    if 'Churn_bin' in df.columns:
        for col in ['E0', 'E_eq', 'gamma', 'tau', 'prob_churn', 'prob_churn_calibrated']:
            if col in df.columns:
                physical_by_churn[col] = {
                    'churners': round(df[df['Churn_bin'] == 1][col].mean(), 4),
                    'non_churners': round(df[df['Churn_bin'] == 0][col].mean(), 4),}

    report_data = {
        'dataset': {
            'total_customers': impact['total_customers'],
            'actual_churners': impact['actual_churners'],
            'churn_rate': impact['churn_rate'],},
        'economic_impact': {
            'revenue_at_risk_usd': impact['revenue_at_risk_usd'],
            'revenue_recovered_usd': impact['revenue_recovered_usd'],
            'campaign_cost_usd': impact['campaign_cost_usd'],
            'net_roi_usd': impact['net_roi_usd'],
            'roi_pct': impact['roi_pct'],},
        'model_precision': {
            'true_positives': impact['true_positives'],
            'precision': impact['precision'],
            'recall': impact['recall'],
            'n_intervened': impact['n_intervened'],},
        'early_detection': {
            'horizon_months': effective['HORIZON_MONTHS'],
            'advantage_days': effective['HORIZON_MONTHS'] * 30,
            'description': (
                f"The model detects churn risk {effective['HORIZON_MONTHS'] * 30} days "
                f"({effective['HORIZON_MONTHS']} months) before the critical threshold "
                f"crossing, vs. reactive detection which occurs after churn has happened."),},
        'segmentation': {
            'risk_level_distribution': risk_dist,
            'segment_distribution': segment_dist,
            'resilience_distribution': resilience_dist,},
        'physical_by_churn_group': physical_by_churn,
        'assumptions': effective,}

    # A/B test design
    # Simulates what the validation experiment would look like if run on the high-risk cohort from this scored dataset.  
    # Uses the same assumed success_rate that drives the ROI estimate so any discrepancy is visible
    ab_cfg = ABTestConfig(
        baseline_churn_rate = impact.get('precision', min(0.95, impact.get('churn_rate', 0.65) * 1.5)),
        assumed_success_rate = effective.get('SUCCESS_RATE', 0.30),
        alpha = 0.05, power = 0.80, seed = 42,)
    ab_ss = required_sample_size(ab_cfg)
    ab_sim_df = simulate_experiment(ab_cfg, n_per_group=ab_ss['n_per_group'])
    ab_result = analyse_experiment(ab_sim_df, ab_cfg,
                                    clv_monthly = effective.get('CLV_MONTHLY', 65),
                                    retention_cost = effective.get('RETENTION_COST', 25),
                                    horizon_months = effective.get('HORIZON_MONTHS', 6))

    report_data['ab_test_validation'] = {
        'design': {
            'n_per_group': ab_ss['n_per_group'],
            'n_total': ab_ss['n_total'],
            'baseline_churn_rate': ab_cfg.baseline_churn_rate,
            'assumed_success_rate': ab_cfg.assumed_success_rate,
            'minimum_detectable_effect': ab_ss['effect_size_abs'],
            'alpha': ab_cfg.alpha,
            'power': ab_cfg.power,
            'weeks_at_1k_weekly': ab_ss['weeks_at_1k_weekly'],},
        'simulation_result': {
            'n_control': ab_result.n_control,
            'n_treatment': ab_result.n_treatment,
            'control_churn_rate': ab_result.control_churn_rate,
            'treatment_churn_rate': ab_result.treatment_churn_rate,
            'observed_success_rate': ab_result.observed_success_rate,
            'z_statistic': ab_result.z_statistic,
            'p_value': ab_result.p_value,
            'significant': ab_result.significant,
            'ci_lower': ab_result.ci_lower,
            'ci_upper': ab_result.ci_upper,
            'roi_at_observed_rate': ab_result.roi_at_observed_rate,
            'roi_at_assumed_rate': ab_result.roi_at_assumed_rate,
            'break_even_rate': ab_result.break_even_rate,
            'achieved_power': ab_result.achieved_power,
            'verdict': ab_result.verdict,
            'conclusion': ab_result.conclusion,},
        'note': (
            'Simulated experiment - runs the A/B test methodology on synthetic data with the same assumed success rate used in the ROI model. '
            'In production, replace simulate_experiment() with real experiment data by calling analyse_experiment(real_df, ab_cfg).'),}

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "impact_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    logger.info(f"[report] Report saved to {output_path}")

    _print_summary(report_data)
    return report_data


def _print_summary(r: dict) -> None:
    """Prints the executive summary to stdout."""
    d = r['dataset']
    eco = r['economic_impact']
    pre = r['model_precision']
    det = r['early_detection']

    print("\n" + "")
    print(" EXECUTIVE SUMMARY - Churn as a Dynamic System")
    print("")
    print(f"  Dataset : {d['total_customers']:,} customers | "
          f"Churn rate: {d['churn_rate']:.1%}")
    print("")
    print(" EARLY DETECTION")
    print(f"  * Anticipation: {det['advantage_days']} days before threshold crossing")
    print(f"  * Customers in window: {pre['n_intervened']:,} "
          f"({pre['n_intervened']/d['total_customers']:.1%} of total)")
    print(f"  * Recall: {pre['recall']:.1%}  |  Precision: {pre['precision']:.1%}")
    print("")
    print("  ECONOMIC IMPACT  (assumptions: CLV=$65/m, cost=$25, rate=30%)")
    print(f" * Revenue at risk: ${eco['revenue_at_risk_usd']:>10,.0f}")
    print(f" * Revenue recovered: ${eco['revenue_recovered_usd']:>10,.0f}")
    print(f" * Campaign cost: ${eco['campaign_cost_usd']:>10,.0f}")
    print(f" * Net ROI: ${eco['net_roi_usd']:>10,.0f}  "
          f"({eco['roi_pct']:+.0f}%)")
    print("")
    ab = r.get('ab_test_validation', {})
    if ab:
        sr = ab['design']['assumed_success_rate']
        n = ab['design']['n_per_group']
        wks = ab['design']['weeks_at_1k_weekly']
        res = ab['simulation_result']
        print(" A/B TEST DESIGN (validates the 30% success rate assumption)")
        print(f" * Required n/group : {n:,} customers per arm")
        print(f" * Runtime estimate : {wks:.1f} weeks at 1,000 high-risk customers/week")
        print(f" * Simulated verdict : {res['verdict']}")
        print(f" * Simulated p-value : {res['p_value']:.4f}  "  f"(observed SR: {res['observed_success_rate']:.1%} vs assumed {sr:.1%})")
        print(f" * Break-even rate : {res['break_even_rate']:.1%}  "  "(minimum SR for ROI > 0)")
    print("" + "\n")


def main():
    parser = argparse.ArgumentParser(description="Churn pipeline - report step")
    parser.add_argument("--input", default="data/telco_final.csv", help="Final scored CSV")
    parser.add_argument("--output-dir", default="report", help="Output directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report(args.input, args.output_dir)


if __name__ == "__main__":
    main()
