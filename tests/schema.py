import numpy as np
import pandas as pd
import pytest

from pipeline.schema import (
    SchemaError, schema_report,
    validate, validate_or_warn,)

# Fixtures: minimal valid DataFrames 
@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "customerID": ["C001", "C002"],
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "tenure": [12, 24],
        "MonthlyCharges": [55.0, 80.0],
        "TotalCharges": ["660.00", "1920.00"],   
        "Contract": ["Month-to-month", "One year"],
        "PaymentMethod": ["Electronic check", "Bank transfer (automatic)"],
        "Churn": ["Yes", "No"],})


@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "customerID": ["C001", "C002"],
        "tenure": [12, 24],
        "MonthlyCharges": [55.0, 80.0],
        "TotalCharges": [660.0, 1920.0],
        "Contract": ["Month-to-month", "One year"],
        "PaymentMethod": ["Electronic check", "Bank transfer (automatic)"],
        "Churn": ["Yes", "No"],
        "Churn_bin": [1, 0],})


@pytest.fixture
def features_df():
    return pd.DataFrame({
        "customerID": ["C001", "C002"],
        "tenure": [12, 24],
        "MonthlyCharges": [55.0, 80.0],
        "TotalCharges": [660.0, 1920.0],
        "Contract": ["Month-to-month", "One year"],
        "PaymentMethod": ["Electronic check", "Bank transfer (automatic)"],
        "num_services": [3, 5],
        "num_services_protective": [2, 3],
        "num_services_entertainment": [1, 2],
        "is_internet_no": [False, False],
        "is_internet_dsl": [True, False],
        "is_internet_fiber": [False, True],
        "auto_payment": [False, True],
        "is_mailed_check": [False, False],
        "is_electronic_check": [True, False],
        "long_contract": [False, True],
        "E0": [0.45, 0.72],
        "E_eq": [0.40, 0.62],
        "gamma": [0.09, 0.20],
        "tau": [11.1, 5.0],
        "resilience": ["Medium", "High"],
        "physical_state": ["At risk", "Stable"],
        "intrinsic_risk": ["No", "No"],
        "estimated_perturbation": [0.05, -0.10], })


@pytest.fixture
def final_df():
    return pd.DataFrame({
        "customerID": ["C001", "C002"],
        "E0": [0.45, 0.72],
        "E_eq": [0.40, 0.62],
        "gamma": [0.09, 0.20],
        "tau": [11.1, 5.0],
        "prob_churn": [0.62, 0.18],
        "prob_churn_calibrated": [0.55, 0.10],
        "sigma_prob": [0.08, 0.05],
        "risk_level": ["HIGH", "LOW"],
        "segment": ["High Risk / Resilient", "Stable"],
        "action": ["Immediate discount", "Do not invest"],
        "priority": [1, 4],})


# Happy path 
class TestValidateHappyPath:

    def test_raw_valid(self, raw_df):
        validate(raw_df, "raw")  # should not raise

    def test_clean_valid(self, clean_df):
        validate(clean_df, "clean")

    def test_features_valid(self, features_df):
        validate(features_df, "features")

    def test_final_valid(self, final_df):
        validate(final_df, "final")

    def test_extra_columns_are_ignored(self, clean_df):
        """Columns not in the schema should not cause failures"""
        df = clean_df.copy()
        df["unexpected_col"] = "whatever"
        validate(df, "clean")  # should not raise

    def test_large_df_passes(self, features_df):
        """Validation should work on a larger repeated DataFrame"""
        big = pd.concat([features_df] * 500, ignore_index=True)
        validate(big, "features")


# Missing columns 
class TestMissingColumns:

    def test_single_missing_column(self, clean_df):
        df = clean_df.drop(columns=["Churn_bin"])
        with pytest.raises(SchemaError, match="Missing columns"):
            validate(df, "clean")

    def test_multiple_missing_columns_reported_together(self, features_df):
        df = features_df.drop(columns=["E0", "E_eq", "gamma"])
        with pytest.raises(SchemaError) as exc_info:
            validate(df, "features")
        msg = str(exc_info.value)
        assert "E0" in msg or "E_eq" in msg or "gamma" in msg

    def test_empty_dataframe_missing_columns(self):
        with pytest.raises(SchemaError):
            validate(pd.DataFrame(), "clean")


# Dtype violations 
class TestDtypeViolations:

    def test_numeric_column_as_string(self, clean_df):
        df = clean_df.copy()
        df["tenure"] = df["tenure"].astype(str)
        with pytest.raises(SchemaError, match="tenure"):
            validate(df, "clean")

    def test_prob_churn_as_string(self, final_df):
        df = final_df.copy()
        df["prob_churn"] = df["prob_churn"].astype(str)
        with pytest.raises(SchemaError, match="prob_churn"):
            validate(df, "final")


# Null violations 
class TestNullViolations:

    def test_null_in_non_nullable_column(self, clean_df):
        df = clean_df.copy()
        df.loc[0, "tenure"] = np.nan
        with pytest.raises(SchemaError, match="null value"):
            validate(df, "clean")

    def test_null_in_E0(self, features_df):
        df = features_df.copy()
        df.loc[0, "E0"] = np.nan
        with pytest.raises(SchemaError, match="E0"):
            validate(df, "features")

    def test_null_in_prob_churn(self, final_df):
        df = final_df.copy()
        df.loc[0, "prob_churn"] = np.nan
        with pytest.raises(SchemaError, match="prob_churn"):
            validate(df, "final")


# Range violations 
class TestRangeViolations:

    def test_E0_above_1(self, features_df):
        df = features_df.copy()
        df.loc[0, "E0"] = 1.5
        with pytest.raises(SchemaError, match="E0"):
            validate(df, "features")

    def test_E0_below_0(self, features_df):
        df = features_df.copy()
        df.loc[0, "E0"] = -0.1
        with pytest.raises(SchemaError, match="E0"):
            validate(df, "features")

    def test_prob_churn_above_1(self, final_df):
        df = final_df.copy()
        df.loc[0, "prob_churn"] = 1.01
        with pytest.raises(SchemaError, match="prob_churn"):
            validate(df, "final")

    def test_negative_monthly_charges(self, clean_df):
        df = clean_df.copy()
        df.loc[0, "MonthlyCharges"] = -10.0
        with pytest.raises(SchemaError, match="MonthlyCharges"):
            validate(df, "clean")

    def test_priority_out_of_range(self, final_df):
        df = final_df.copy()
        df.loc[0, "priority"] = 5
        with pytest.raises(SchemaError, match="priority"):
            validate(df, "final")

    def test_range_check_disabled_with_extra_checks_false(self, features_df):
        """extra_checks=False should skip range validation """
        df = features_df.copy()
        df.loc[0, "E0"] = 2.0  # invalid range
        # Should NOT raise when extra_checks=False
        validate(df, "features", extra_checks=False)


# Categorical violations 
class TestCategoricalViolations:

    def test_unexpected_contract_value(self, clean_df):
        df = clean_df.copy()
        df.loc[0, "Contract"] = "Weekly"
        with pytest.raises(SchemaError, match="Contract"):
            validate(df, "clean")

    def test_unexpected_risk_level(self, final_df):
        df = final_df.copy()
        df.loc[0, "risk_level"] = "VERY_HIGH"
        with pytest.raises(SchemaError, match="risk_level"):
            validate(df, "final")

    def test_unexpected_resilience_value(self, features_df):
        df = features_df.copy()
        df.loc[0, "resilience"] = "Very High"
        with pytest.raises(SchemaError, match="resilience"):
            validate(df, "features")

    def test_categorical_check_disabled_with_extra_checks_false(self, final_df):
        """extra_checks=False should skip categorical validation"""
        df = final_df.copy()
        df.loc[0, "risk_level"] = "CRITICAL"
        validate(df, "final", extra_checks=False)  # should not raise


# Error accumulation 
class TestErrorAccumulation:

    def test_multiple_errors_reported_at_once(self, features_df):
        """All violations should appear in a single SchemaError."""
        df = features_df.copy()
        df.loc[0, "E0"] = -0.5   # range violation
        df.loc[0, "gamma"] = 5.0    # range violation
        df.loc[0, "resilience"] = "VeryHigh"  # categorical violation

        with pytest.raises(SchemaError) as exc_info:
            validate(df, "features")

        msg = str(exc_info.value)
        assert "[1]" in msg  
        assert "[2]" in msg  


# validate_or_warn 
class TestValidateOrWarn:

    def test_returns_true_on_valid(self, clean_df):
        result = validate_or_warn(clean_df, "clean")
        assert result is True

    def test_returns_false_on_invalid(self, clean_df):
        df = clean_df.drop(columns=["Churn_bin"])
        result = validate_or_warn(df, "clean")
        assert result is False

    def test_does_not_raise(self, clean_df):
        df = clean_df.drop(columns=["Churn_bin"])
        try:
            validate_or_warn(df, "clean")
        except SchemaError:
            pytest.fail("validate_or_warn() raised SchemaError unexpectedly")

    def test_calls_logger_warning(self, clean_df):
        """If a logger is supplied it should receive the warning"""
        messages = []

        class CapturingLogger:
            def warning(self, msg):
                messages.append(msg)

        df = clean_df.drop(columns=["Churn_bin"])
        validate_or_warn(df, "clean", logger=CapturingLogger())
        assert len(messages) == 1
        assert "Schema validation failed" in messages[0]


# schema_report 
class TestSchemaReport:

    @pytest.mark.parametrize("stage", ["raw", "clean", "features", "final"])
    def test_returns_non_empty_string(self, stage):
        report = schema_report(stage)
        assert isinstance(report, str)
        assert len(report) > 50

    def test_report_contains_column_names(self):
        report = schema_report("final")
        assert "prob_churn" in report
        assert "risk_level" in report
        assert "segment" in report

    def test_unknown_stage_raises_key_error(self):
        with pytest.raises(KeyError):
            schema_report("nonexistent_stage")


# Unknown stage 
class TestUnknownStage:

    def test_validate_unknown_stage_raises_key_error(self, clean_df):
        with pytest.raises(KeyError, match="Unknown schema stage"):
            validate(clean_df, "doesnt_exist")

    def test_error_message_lists_valid_stages(self, clean_df):
        with pytest.raises(KeyError) as exc_info:
            validate(clean_df, "bad_stage")
        assert "clean" in str(exc_info.value)
        assert "features" in str(exc_info.value)


# min_rows 
class TestMinRows:

    def test_empty_df_fails_min_rows(self, clean_df):
        empty = clean_df.iloc[:0]
        with pytest.raises(SchemaError, match="rows"):
            validate(empty, "clean", min_rows=1)

    def test_custom_min_rows_passes(self, clean_df):
        validate(clean_df, "clean", min_rows=2)  # has exactly 2 rows

    def test_custom_min_rows_fails(self, clean_df):
        with pytest.raises(SchemaError, match="rows"):
            validate(clean_df, "clean", min_rows=10)
