import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.steps.featurize import featurize
from pipeline.steps.ingest import clean_total_charges, encode_churn, ingest, validate_columns
from pipeline.steps.report import report
from pipeline.steps.score import score

TELCO_COLUMNS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn',]


def make_synthetic_dataset(n=50, seed=42):
    np.random.seed(seed)
    contracts = np.random.choice(['Month-to-month', 'One year', 'Two year'], n)
    payments = np.random.choice(['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'], n)
    internet  = np.random.choice(['Fiber optic', 'DSL', 'No'], n)
    tenure = np.random.randint(0, 72, n)
    monthly = np.random.uniform(20, 110, n)
    total_c = np.where(tenure > 0, (monthly * tenure).astype(str), ' ')

    rows = []
    for i in range(n):
        rows.append({
            'customerID': f'SYNTH-{i:04d}',
            'gender': np.random.choice(['Male', 'Female']),
            'SeniorCitizen': np.random.randint(0, 2),
            'Partner': np.random.choice(['Yes', 'No']),
            'Dependents': np.random.choice(['Yes', 'No']),
            'tenure': int(tenure[i]),
            'PhoneService': np.random.choice(['Yes', 'No']),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service']),
            'InternetService': internet[i],
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service']),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service']),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service']),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service']),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service']),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service']),
            'Contract': contracts[i],
            'PaperlessBilling': np.random.choice(['Yes', 'No']),
            'PaymentMethod': payments[i],
            'MonthlyCharges': round(float(monthly[i]), 2),
            'TotalCharges': total_c[i],
            'Churn': np.random.choice(['Yes', 'No'], p=[0.27, 0.73]),})
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_df():
    return make_synthetic_dataset(n=60)


@pytest.fixture
def paths(tmp_path):
    return {
        'raw': str(tmp_path / 'raw.csv'),
        'clean': str(tmp_path / 'clean.csv'),
        'features': str(tmp_path / 'features.csv'),
        'final': str(tmp_path / 'final.csv'),
        'report': str(tmp_path / 'report'),}


class TestIngest:
    def test_validate_columns_ok(self, synthetic_df):
        validate_columns(synthetic_df)

    def test_validate_columns_missing(self, synthetic_df):
        df = synthetic_df.drop(columns=['Churn', 'tenure'])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_columns(df)

    def test_clean_total_charges_numeric(self, synthetic_df):
        df = clean_total_charges(synthetic_df)
        assert df['TotalCharges'].dtype in [np.float64, float]
        assert df['TotalCharges'].isna().sum() == 0

    def test_clean_total_charges_imputes_zero_tenure(self, synthetic_df):
        df = clean_total_charges(synthetic_df)
        for idx in synthetic_df[synthetic_df['tenure'] == 0].index:
            assert df.loc[idx, 'TotalCharges'] == pytest.approx(df.loc[idx, 'MonthlyCharges'], abs=0.01)

    def test_encode_churn(self, synthetic_df):
        df = encode_churn(synthetic_df)
        assert 'Churn_bin' in df.columns
        assert set(df['Churn_bin'].unique()).issubset({0, 1})
        assert (df[df['Churn'] == 'Yes']['Churn_bin'] == 1).all()
        assert (df[df['Churn'] == 'No']['Churn_bin'] == 0).all()

    def test_ingest_end_to_end(self, synthetic_df, paths):
        synthetic_df.to_csv(paths['raw'], index=False)
        df = ingest(paths['raw'], paths['clean'])
        assert Path(paths['clean']).exists()
        assert len(df) > 0
        assert 'Churn_bin' in df.columns
        assert df['TotalCharges'].isna().sum() == 0

    def test_ingest_removes_duplicates(self, synthetic_df, paths):
        pd.concat([synthetic_df, synthetic_df.head(5)], ignore_index=True).to_csv(paths['raw'], index=False)
        df = ingest(paths['raw'], paths['clean'])
        assert len(df) == len(synthetic_df)


class TestFeaturizePipeline:

    def test_featurize_end_to_end(self, synthetic_df, paths):
        synthetic_df.to_csv(paths['raw'], index=False)
        ingest(paths['raw'], paths['clean'])
        df = featurize(paths['clean'], paths['features'])
        assert Path(paths['features']).exists()
        for col in ['E0', 'E_eq', 'gamma', 'tau', 'resilience', 'physical_state']:
            assert col in df.columns, f"Missing column: {col}"

    def test_featurize_no_nans(self, synthetic_df, paths):
        synthetic_df.to_csv(paths['raw'], index=False)
        ingest(paths['raw'], paths['clean'])
        df = featurize(paths['clean'], paths['features'])
        for col in ['E0', 'E_eq', 'gamma', 'tau']:
            assert df[col].isna().sum() == 0, f"NaN in {col}"


class TestScorePipeline:

    def test_score_end_to_end(self, synthetic_df, paths):
        synthetic_df.to_csv(paths['raw'], index=False)
        ingest(paths['raw'], paths['clean'])
        featurize(paths['clean'], paths['features'])
        df = score(paths['features'], paths['final'])
        assert Path(paths['final']).exists()
        for col in ['prob_churn', 'sigma_prob', 'risk_level', 'segment', 'action']:
            assert col in df.columns, f"Missing column: {col}"

    def test_prob_churn_in_range(self, synthetic_df, paths):
        synthetic_df.to_csv(paths['raw'], index=False)
        ingest(paths['raw'], paths['clean'])
        featurize(paths['clean'], paths['features'])
        df = score(paths['features'], paths['final'])
        assert df['prob_churn'].between(0, 1).all()

    def test_risk_level_valid_values(self, synthetic_df, paths):
        synthetic_df.to_csv(paths['raw'], index=False)
        ingest(paths['raw'], paths['clean'])
        featurize(paths['clean'], paths['features'])
        df = score(paths['features'], paths['final'])
        assert set(df['risk_level'].unique()).issubset({'HIGH', 'MEDIUM', 'LOW', 'UNCERTAIN'})


class TestEndToEnd:

    def test_full_pipeline(self, synthetic_df, paths):
        synthetic_df.to_csv(paths['raw'], index=False)
        df1 = ingest(paths['raw'], paths['clean'])
        df2 = featurize(paths['clean'], paths['features'])
        df3 = score(paths['features'], paths['final'])
        rep = report(paths['final'], paths['report'])
        assert len(df1) > 0
        assert len(df2) == len(df1)
        assert len(df3) == len(df2)
        assert 'economic_impact' in rep
        assert Path(paths['report'], 'impact_report.json').exists()

    def test_report_json_valid(self, synthetic_df, paths):
        synthetic_df.to_csv(paths['raw'], index=False)
        ingest(paths['raw'], paths['clean'])
        featurize(paths['clean'], paths['features'])
        score(paths['features'], paths['final'])
        report(paths['final'], paths['report'])
        with open(Path(paths['report']) / 'impact_report.json') as f:
            data = json.load(f)
        assert 'dataset' in data
        assert 'economic_impact' in data
        assert 'early_detection' in data
        assert data['dataset']['total_customers'] > 0
