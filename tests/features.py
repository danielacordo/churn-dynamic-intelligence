import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import pytest

from src.features import (
    build_physical_features,
    count_services, is_auto_payment,
    is_long_contract, physical_summary,)
from src.physics import E_CRITICAL

# Fixture: minimal realistic dataset 
@pytest.fixture
def minimal_df():
    """Minimal DataFrame with the columns required by build_physical_features"""
    return pd.DataFrame([
        {
            'customerID': 'A001', 'tenure': 24, 'MonthlyCharges': 65.0,
            'TotalCharges': 1560.0, 'Contract': 'Two year',
            'PaymentMethod': 'Bank transfer (automatic)',
            'PhoneService': 'Yes', 'MultipleLines': 'Yes',
            'InternetService': 'Fiber optic', 'OnlineSecurity': 'Yes',
            'OnlineBackup': 'No', 'DeviceProtection': 'Yes',
            'TechSupport': 'Yes', 'StreamingTV': 'No', 'StreamingMovies': 'No',
            'Churn': 'No', 'Churn_bin': 0,},
        {
            'customerID': 'A002', 'tenure': 2, 'MonthlyCharges': 89.0,
            'TotalCharges': 178.0, 'Contract': 'Month-to-month',
            'PaymentMethod': 'Electronic check',
            'PhoneService': 'No', 'MultipleLines': 'No phone service',
            'InternetService': 'Fiber optic', 'OnlineSecurity': 'No',
            'OnlineBackup': 'No', 'DeviceProtection': 'No',
            'TechSupport': 'No', 'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
            'Churn': 'Yes', 'Churn_bin': 1,},
        {
            'customerID': 'A003', 'tenure': 0, 'MonthlyCharges': 45.0,
            'TotalCharges': 45.0, 'Contract': 'One year',
            'PaymentMethod': 'Credit card (automatic)',
            'PhoneService': 'Yes', 'MultipleLines': 'No',
            'InternetService': 'DSL', 'OnlineSecurity': 'No',
            'OnlineBackup': 'Yes', 'DeviceProtection': 'No',
            'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
            'Churn': 'No', 'Churn_bin': 0,},])


# count_services() 
class TestCountServices:

    def test_customer_with_no_services(self, minimal_df):
        """Customer with all services as 'No' must count 0 """
        row = minimal_df.iloc[1].copy()
        for col in ['PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']:
            row[col] = 'No'
        assert count_services(row) == 0

    def test_customer_with_all_services(self, minimal_df):
        """All services active -> count = 9 """
        row = minimal_df.iloc[0].copy()
        for col in ['PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']:
            row[col] = 'Yes'
        assert count_services(row) == 9

    def test_count_positive_for_normal_customer(self, minimal_df):
        """First fixture customer must have > 0 services """
        assert count_services(minimal_df.iloc[0]) > 0

    def test_count_in_range(self, minimal_df):
        """Count must be between 0 and 9 for all customers."""
        for _, row in minimal_df.iterrows():
            n = count_services(row)
            assert 0 <= n <= 9


# is_auto_payment() 
class TestIsAutoPayment:

    def test_bank_transfer_automatic(self):
        assert is_auto_payment('Bank transfer (automatic)') is True

    def test_credit_card_automatic(self):
        assert is_auto_payment('Credit card (automatic)') is True

    def test_electronic_check_not_automatic(self):
        assert is_auto_payment('Electronic check') is False

    def test_mailed_check_not_automatic(self):
        assert is_auto_payment('Mailed check') is False

    def test_unknown_value_not_automatic(self):
        assert is_auto_payment('PayPal') is False


# is_long_contract() }
class TestIsLongContract:

    def test_two_year(self):
        assert is_long_contract('Two year') is True

    def test_one_year(self):
        assert is_long_contract('One year') is True

    def test_month_to_month_not_long(self):
        assert is_long_contract('Month-to-month') is False


# build_physical_features() 
class TestBuildPhysicalFeatures:

    def test_generated_columns_present(self, minimal_df):
        """Output DataFrame must contain all physical columns."""
        df_out = build_physical_features(minimal_df)
        expected = [
            'E0', 'E_eq', 'gamma', 'tau', 'resilience',
            'estimated_perturbation', 'physical_state', 'intrinsic_risk',
            'num_services', 'auto_payment', 'long_contract',]
        for col in expected:
            assert col in df_out.columns, f"Missing column: {col}"

    def test_E0_in_range(self, minimal_df):
        """E0 must be in [0, 1] for all customers."""
        df_out = build_physical_features(minimal_df)
        assert df_out['E0'].between(0, 1).all(), f"E0 out of range: {df_out['E0'].describe()}"

    def test_E_eq_in_range(self, minimal_df):
        """E_eq must be in [0, 1]"""
        df_out = build_physical_features(minimal_df)
        assert df_out['E_eq'].between(0, 1).all()

    def test_gamma_in_range(self, minimal_df):
        """gamma must be in (0, 1] """
        df_out = build_physical_features(minimal_df)
        assert (df_out['gamma'] > 0).all()
        assert (df_out['gamma'] <= 1).all()

    def test_tau_is_inverse_gamma(self, minimal_df):
        """tau = 1/gamma within numerical tolerance """
        df_out = build_physical_features(minimal_df)
        diff = (df_out['tau'] - 1.0 / df_out['gamma']).abs()
        assert (diff < 1e-6).all()

    def test_two_year_has_higher_E_eq(self, minimal_df):
        """Two year contract must have higher E_eq than month-to-month."""
        df_out = build_physical_features(minimal_df)
        e_two = df_out[df_out['Contract'] == 'Two year']['E_eq'].values[0]
        e_mon = df_out[df_out['Contract'] == 'Month-to-month']['E_eq'].values[0]
        assert e_two > e_mon

    def test_two_year_auto_payment_high_resilience(self, minimal_df):
        """Two year contract + auto payment -> High resilience"""
        df_out = build_physical_features(minimal_df)
        customer = df_out[df_out['Contract'] == 'Two year'].iloc[0]
        assert customer['resilience'] == 'High'

    def test_zero_tenure_no_nan(self, minimal_df):
        """tenure=0 must not produce NaN in any physical column"""
        df_out = build_physical_features(minimal_df)
        new_customers = df_out[df_out['tenure'] == 0]
        assert len(new_customers) > 0, "Fixture should have a customer with tenure=0"
        for col in ['E0', 'E_eq', 'gamma', 'tau']:
            assert new_customers[col].notna().all(), f"NaN found in {col} for tenure=0"

    def test_physical_state_valid_categories(self, minimal_df):
        """ physical_state must only take valid values."""
        df_out = build_physical_features(minimal_df)
        valid  = {'Stable', 'At risk', 'Critical'}
        assert set(df_out['physical_state'].unique()).issubset(valid)

    def test_intrinsic_risk_logic(self, minimal_df):
        """intrinsic_risk must be 'Yes' when E_eq < E_CRITICAL """
        df_out = build_physical_features(minimal_df)
        mask_yes = df_out['intrinsic_risk'] == 'Yes'
        mask_no = df_out['intrinsic_risk'] == 'No'
        assert (df_out.loc[mask_yes, 'E_eq'] < E_CRITICAL).all()
        assert (df_out.loc[mask_no, 'E_eq'] >= E_CRITICAL).all()

    def test_does_not_modify_input(self, minimal_df):
        """The function must not modify the input DataFrame."""
        original = minimal_df.copy()
        build_physical_features(minimal_df)
        pd.testing.assert_frame_equal(minimal_df, original)

    def test_idempotent(self, minimal_df):
        """Applying the function twice must give the same result as once."""
        df_once  = build_physical_features(minimal_df)
        df_twice = build_physical_features(minimal_df)
        pd.testing.assert_frame_equal(df_once, df_twice)


# physical_summary() 
class TestPhysicalSummary:
    def test_returns_dataframe(self, minimal_df):
        df_out = build_physical_features(minimal_df)
        summary = physical_summary(df_out)
        assert isinstance(summary, pd.DataFrame)

    def test_has_churn_groups(self, minimal_df):
        """Summary must have at least one churn group """
        df_out = build_physical_features(minimal_df)
        summary = physical_summary(df_out)
        assert len(summary) >= 1
