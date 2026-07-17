import argparse
import logging
import time
from pathlib import Path
from pipeline.config import BusinessParams, Config, ModelParams
from pipeline.run import run_pipeline

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",)
logger = logging.getLogger(__name__)

BANNER = """
          CHURN AS A DYNAMIC SYSTEM                  
  Physics-Informed Customer Retention Analysis       
"""


# Actionable outputs 
def print_top_customers(n: int, cfg: Config) -> None:
    import pandas as pd

    df = pd.read_csv(cfg.paths.final)

    cols = ["customerID", "prob_churn_calibrated", "sigma_prob", "prob_inf_calibrated", "prob_sup_calibrated",
            "risk_level", "segment", "action", "E0", "E_eq", "tau"]
    available = [c for c in cols if c in df.columns]

    top = (df[df["risk_level"] == "HIGH"].sort_values("prob_churn_calibrated", ascending=False).head(n)[available].reset_index(drop=True))
    top.index += 1

    print("\n" + "")
    print(f" TOP {n} CUSTOMERS TO INTERVENE - sorted by churn probability")
    print("")

    for i, row in top.iterrows():
        cid = row.get("customerID", f"Customer {i}")
        prob = row.get("prob_churn_calibrated", 0)
        sigma = row.get("sigma_prob", 0)
        lb = row.get("prob_inf_calibrated", prob - sigma)
        ub = row.get("prob_sup_calibrated", prob + sigma)
        seg = row.get("segment", "-")
        act = row.get("action", "-")
        tau = row.get("tau", 0)
        E_eq = row.get("E_eq", 0)

        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        print(f"\n  #{i:02d}  {cid}")
        print(f" Risk [{bar}] {prob:.0%} ± {sigma:.0%}  (IC: {lb:.0%}–{ub:.0%})")
        print(f" Segment {seg}")
        print(f" Action {act}")
        print(f" Physics  τ={tau:.1f}m  |  E_eq={E_eq:.2f}")

    print("\n" + "")


def print_campaign_simulation(cfg: Config) -> None:
    """Campaign simulation by risk level - matches the table in README.md.
    Uses compute_economic_impact() for consistency with the main ROI report."""
    import pandas as pd
    from src.business import compute_economic_impact

    df = pd.read_csv(cfg.paths.final)
    clv  = cfg.business.clv_monthly
    cost = cfg.business.retention_cost
    h = cfg.business.horizon_months
    sr = cfg.business.success_rate
    thr  = cfg.model.prob_threshold

    if "risk_level" not in df.columns:
        print(" [risk_level column not found - run full pipeline first]")
        return

    print("\n" + "")
    print(" CAMPAIGN SIMULATION - by risk level")
    print(f" Assumptions: avg CLV=${clv}/m (per-customer MonthlyCharges used internally) - Campaign cost=${cost} - SR={sr:.0%} - Horizon={h}m")
    print("")
    print(f" {'Segment':<28} {'N':>6} {'Retained':>9} {'Revenue':>11} {'Cost':>9} {'ROI':>11}")
    print("  ")

    order = ["HIGH", "UNCERTAIN", "MEDIUM", "LOW"]
    skip_note = {"LOW": "<- skip"}
    grand_rev = grand_cost = grand_roi = grand_ret = 0
    low_rev = low_cost = low_roi = low_ret = 0  # tracked separately for TOTAL excl. LOW

    for lvl in order:
        sub = df[df["risk_level"] == lvl]
        n = len(sub)
        if n == 0:
            continue

        n_churners = int(sub["Churn_bin"].sum()) if "Churn_bin" in sub.columns else int(n * 0.265)
        retained = n_churners * sr
        revenue = retained * clv * h
        camp_cost  = n * cost
        roi = revenue - camp_cost
        note = skip_note.get(lvl, "")

        grand_rev += revenue
        grand_cost += camp_cost
        grand_roi += roi
        grand_ret += retained

        if lvl == "LOW":
            low_rev, low_cost, low_roi, low_ret = revenue, camp_cost, roi, retained

        label = {"HIGH": "HIGH risk (lb > 0.35)", "LOW": "Stable (LOW)",
                 "UNCERTAIN": "UNCERTAIN", "MEDIUM": "MEDIUM risk"}.get(lvl, lvl)

        print(f" {label:<28} {n:>6,} {retained:>9.0f} ${revenue:>10,.0f} ${camp_cost:>8,.0f}"
              f" ${roi:>+10,.0f}  {note}")

    # Recommended: at ROI-optimal threshold
    impact_opt = compute_economic_impact(df, threshold=thr, assumptions=cfg.business.to_assumptions(), prob_col='prob_churn_calibrated')
    print("  ")
    print(f" {'TOTAL (excl. LOW)':<28} {len(df[df.risk_level != 'LOW']):>6,}"
          f" {grand_ret - low_ret:>9.0f}"
          f" ${grand_rev - low_rev:>10,.0f}"
          f" ${grand_cost - low_cost:>8,.0f}"
          f" ${grand_roi - low_roi:>+10,.0f}")
    print(f"  {'-> At threshold '+str(thr):<28} {impact_opt['n_intervened']:>6,}"
          f" {impact_opt['customers_saved_est']:>9.0f}"
          f" ${impact_opt['revenue_recovered_usd']:>10,.0f}"
          f" ${impact_opt['campaign_cost_usd']:>8,.0f}"
          f" ${impact_opt['net_roi_usd']:>+10,.0f}  <- recommended")
    print("")
    print(f"""
  KEY METRICS  (at threshold {thr})
  
  Customers flagged : {impact_opt['n_intervened']:,}  ({impact_opt['n_intervened']/len(df):.1%} of base)
  Recall : {impact_opt['recall']:.1%}  (churners caught)
  Precision : {impact_opt['precision']:.1%}  (flagged that actually churn)
  Revenue recovered : ${impact_opt['revenue_recovered_usd']:,.0f}
  Campaign investment : ${impact_opt['campaign_cost_usd']:,.0f}
  Net ROI : ${impact_opt['net_roi_usd']:+,.0f}  ({impact_opt['roi_pct']:+.0f}%)
  Detection advantage : {h * 30} days before churn event
""")


def print_economic_impact(cfg: Config) -> None:
    import json

    report_path = Path(cfg.paths.report) / "impact_report.json"
    if not report_path.exists():
        logger.warning("Report not found - run full pipeline first.")
        return

    with open(report_path) as f:
        r = json.load(f)

    eco = r["economic_impact"]
    det = r["early_detection"]
    seg = r.get("segmentation", {})

    print("\n" + "")
    print(" ECONOMIC IMPACT REPORT")
    print("")
    print(f" Revenue at risk : ${eco['revenue_at_risk_usd']:>10,.0f}")
    print(f" Revenue recovered : ${eco['revenue_recovered_usd']:>10,.0f}")
    print(f" Campaign investment : ${eco['campaign_cost_usd']:>10,.0f}")
    print(f" Net ROI : ${eco['net_roi_usd']:>+10,.0f}  ({eco['roi_pct']:+.0f}%)")
    print(f"\n  Early detection window : {det['advantage_days']} days")
    print(f" Precision : {r['model_precision']['precision']:.1%}")
    print(f" Recall : {r['model_precision']['recall']:.1%}")

    if seg.get("risk_level_distribution"):
        print("\n  Risk level distribution:")
        for level, count in sorted(seg["risk_level_distribution"].items()):
            print(f" {level:<12}: {count:,} customers")
    print("" + "\n")


# Main 
def main():
    parser = argparse.ArgumentParser(
        description="Churn as a Dynamic System -- Physics-Informed Retention Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py   # full pipeline
  python main.py --top 10   # top 10 customers to intervene
  python main.py --simulate  # campaign ROI simulation by segment
  python main.py --skip-ingest  # skip data loading step
  python main.py --horizon 12  # 12-month prediction window
        """,)
    parser.add_argument("--top", type=int, default=10, help="Show top N customers to intervene (default: 10)")
    parser.add_argument("--simulate", action="store_true", help="Show campaign simulation by segment")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip data loading (reuse existing clean data)")
    parser.add_argument("--horizon", type=float, default=None, help="Prediction horizon in months (default: 6)")
    parser.add_argument("--threshold", type=float, default=None, help="Segmentation threshold on the calibrated probability (default: 0.20, ROI-optimal)")
    parser.add_argument("--clv", type=float, default=None, help="Monthly CLV per customer in USD (default: 65)")
    parser.add_argument("--only-output", action="store_true", help="Skip pipeline, show outputs only (data must exist)")
    args = parser.parse_args()

    print(BANNER)

    # Build Config - only override what the user explicitly passed
    business = BusinessParams()
    model = ModelParams()

    if args.horizon is not None:
        business.horizon_months = int(args.horizon)
    if args.clv is not None:
        business.clv_monthly = args.clv
    if args.threshold is not None:
        model.prob_threshold = args.threshold

    cfg = Config(
        business=business,
        model=model,
        skip_ingest=args.skip_ingest,)

    t0 = time.time()

    if not args.only_output:
        run_pipeline(cfg)
        logger.info(f"Pipeline completed in {time.time() - t0:.1f}s")

    # Actionable outputs 
    print_top_customers(args.top, cfg)
    print_economic_impact(cfg)

    if args.simulate:
        print_campaign_simulation(cfg)

    print("\n Run 'streamlit run app.py' to open the interactive dashboard.\n")


if __name__ == "__main__":
    main()
