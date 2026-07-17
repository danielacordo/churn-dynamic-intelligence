import argparse
import logging
from pathlib import Path
import pandas as pd
from pipeline.schema import validate

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn',]


def validate_columns(df: pd.DataFrame) -> None:
    """Raises ValueError if any required column is missing. """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """TotalCharges comes as a string with whitespace for new customers (tenure=0).
    Converts to float; missing values are imputed with MonthlyCharges """
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    null_mask = df['TotalCharges'].isna()
    n_nulls = null_mask.sum()
    if n_nulls > 0:
        logger.info(f"TotalCharges: imputing {n_nulls} null values with MonthlyCharges")
        df.loc[null_mask, 'TotalCharges'] = df.loc[null_mask, 'MonthlyCharges']
    return df


def encode_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes Churn ('Yes'/'No') as Churn_bin (1/0) """
    df = df.copy()
    df['Churn_bin'] = (df['Churn'] == 'Yes').astype(int)
    return df


def ingest(input_path: str, output_path: str) -> pd.DataFrame:
    """Full ingestion pipeline"""
    logger.info(f"[ingest] Reading {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"[ingest] Initial shape: {df.shape}")

    validate_columns(df)
    validate(df, 'raw')
    df = clean_total_charges(df)
    df = encode_churn(df)

    # Remove duplicates by customerID
    n_before = len(df)
    df = df.drop_duplicates(subset='customerID')
    if len(df) < n_before:
        logger.warning(f"[ingest] Removed {n_before - len(df)} duplicate rows")

    validate(df, 'clean')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"[ingest] Saved to {output_path} - {len(df):,} rows")
    return df


def main():
    parser = argparse.ArgumentParser(description="Churn pipeline - ingestion step")
    parser.add_argument("--input", default="data/telco_churn.csv", help="Raw CSV input")
    parser.add_argument("--output", default="data/telco_clean.csv", help="Cleaned CSV output")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingest(args.input, args.output)


if __name__ == "__main__":
    main()
