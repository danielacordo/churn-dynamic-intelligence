import os
import pandas as pd
from src.business import apply_segmentation
from src.features import build_physical_features
from src.uncertainty import calibrate_dataframe_probabilities, score_dataframe_with_uncertainty

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')


def main() -> None:
    clean_path = os.path.join(DATA_DIR, 'telco_clean.csv')
    if not os.path.exists(clean_path):
        raise FileNotFoundError(
            f"{clean_path} not found. Run notebooks/01_eda.ipynb first to produce it "
            f"(or download_data.py + 01_eda.ipynb if starting from scratch).")

    print(f"Loading {clean_path} ...")
    df = pd.read_csv(clean_path)
    print(f"  {len(df):,} customers | churn rate {df['Churn_bin'].mean():.1%}")

    print("Building physical features (E0, E_eq, gamma, tau, resilience, physical_state) ...")
    df = build_physical_features(df)

    print("Scoring churn probability with uncertainty propagation ...")
    df = score_dataframe_with_uncertainty(df, t_horizon=6.0)

    print("Calibrating probability against observed churn (isotonic regression) ...")
    df, calibrator = calibrate_dataframe_probabilities(df)
    print(f" Mean raw prob_churn: {df['prob_churn'].mean():.3f}")
    print(f" Mean calibrated prob_churn: {df['prob_churn_calibrated'].mean():.3f}")
    print(f" Actual churn rate: {df['Churn_bin'].mean():.3f}")

    print("Applying strategic segmentation ...")
    df = apply_segmentation(df)

    out_path = os.path.join(DATA_DIR, 'telco_final.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} scored customers to {out_path}")

    n_high = (df['risk_level'] == 'HIGH').sum()
    print(f"\n{n_high:,} customers ({n_high/len(df):.1%}) flagged risk_level == 'HIGH' "
          f"for this week's outreach queue.")


if __name__ == '__main__':
    main()
