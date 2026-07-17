import argparse
import logging
from pathlib import Path
import pandas as pd
from pipeline.schema import validate
from src.features import build_physical_features

logger = logging.getLogger(__name__)


def featurize(input_path: str, output_path: str) -> pd.DataFrame:
    """ Builds all physical state variables on the cleaned dataset"""

    logger.info(f"[featurize] Reading {input_path}")
    df = pd.read_csv(input_path)
    validate(df, 'clean')
    logger.info(f"[featurize] Shape: {df.shape}")

    df = build_physical_features(df)

    logger.info(f"[featurize] E0 - mean={df['E0'].mean():.3f} std={df['E0'].std():.3f}")
    logger.info(f"[featurize] E_eq - mean={df['E_eq'].mean():.3f} std={df['E_eq'].std():.3f}")
    logger.info(f"[featurize] gamma - mean={df['gamma'].mean():.3f} std={df['gamma'].std():.3f}")
    logger.info(f"[featurize] tau - median={df['tau'].median():.1f} months")

    if 'Churn_bin' in df.columns:
        churn_by_state = df.groupby('physical_state')['Churn_bin'].mean().round(3)
        logger.info(f"[featurize] Churn rate by physical state:\n{churn_by_state}")

    validate(df, 'features')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"[featurize] Saved to {output_path} - {len(df):,} rows, {df.shape[1]} columns")
    return df


def main():
    parser = argparse.ArgumentParser(description="Churn pipeline - featurization step")
    parser.add_argument("--input", default="data/telco_clean.csv", help="Cleaned CSV input")
    parser.add_argument("--output", default="data/telco_features.csv", help="Features CSV output")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    featurize(args.input, args.output)


if __name__ == "__main__":
    main()
