import argparse
import logging
from pathlib import Path
import pandas as pd
from pipeline.schema import validate
from src.business import apply_segmentation
from src.uncertainty import calibrate_dataframe_probabilities, score_dataframe_with_uncertainty

logger = logging.getLogger(__name__)


def score(input_path: str, output_path: str, t_horizon: float = 6.0, prob_threshold: float = 0.20) -> pd.DataFrame:
    """ Computes churn probabilities with formal error propagation, calibrates them against
    observed churn (raw prob_churn is systematically overconfident -- mean ~0.50 vs an actual
    churn rate of ~0.265 on this dataset, see notebooks/06), and applies strategic segmentation.

    NOTE: prob_threshold is applied to the *calibrated* probability (prob_churn_calibrated).
    0.20 is the ROI-optimal cutoff as of the last recalibration (matches pipeline/config.py's
    default) -- re-run src.business.optimize_threshold periodically as the calibrator is refit
    on new data."""

    logger.info(f"[score] Reading {input_path}")
    df = pd.read_csv(input_path)
    validate(df, 'features')
    logger.info(f"[score] Shape: {df.shape} | Horizon: {t_horizon}m")

    df = score_dataframe_with_uncertainty(df, t_horizon=t_horizon)

    logger.info(f"[score] prob_churn (raw) - mean={df['prob_churn'].mean():.3f}  std={df['prob_churn'].std():.3f}")

    df, _calibrator = calibrate_dataframe_probabilities(df)
    logger.info(f"[score] prob_churn_calibrated - mean={df['prob_churn_calibrated'].mean():.3f}  "
                f"(actual churn rate={df['Churn_bin'].mean():.3f})")
    logger.info(f"[score] sigma_prob - mean={df['sigma_prob'].mean():.3f}")
    logger.info(f"[score] Risk level distribution:\n{df['risk_level'].value_counts()}")

    df = apply_segmentation(df)
    logger.info(f"[score] Segment distribution:\n{df['segment'].value_counts()}")

    validate(df, 'final')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"[score] Saved to {output_path} - {len(df):,} rows, {df.shape[1]} columns")
    return df


def main():
    parser = argparse.ArgumentParser(description="Churn pipeline — scoring step")
    parser.add_argument("--input", default="data/telco_features.csv", help="Features CSV")
    parser.add_argument("--output", default="data/telco_final.csv", help="Scored CSV")
    parser.add_argument("--horizon", type=float, default=6.0, help="Horizon in months")
    parser.add_argument("--threshold", type=float, default=0.20,
                         help="Classification threshold on the CALIBRATED probability "
                              "(ROI-optimal default; re-run src.business.optimize_threshold "
                              "periodically as the calibrator is refit)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    score(args.input, args.output, args.horizon, args.threshold)


if __name__ == "__main__":
    main()
