import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.business import (
    ASSUMPTIONS, calibration_analysis,
    calibration_summary, optimize_threshold, sensitivity_analysis,
    sensitivity_summary, threshold_summary,)

# Fixtures 
def _synthetic_predictions(n=500, seed=42):
    """Generates synthetic y_true and y_pred_proba for testing """

    rng = np.random.default_rng(seed)
    y_true = rng.binomial(1, 0.27, n).astype(float)
    # Moderately useful predictions, correlated with truth but noisy
    noise = rng.normal(0, 0.2, n)
    y_pred = np.clip(y_true * 0.5 + 0.2 + noise, 0.01, 0.99)
    return y_true, y_pred


def _perfect_predictions(n=200, seed=0):
    """Perfect model: pred = true"""
    rng = np.random.default_rng(seed)
    y_true = rng.binomial(1, 0.27, n).astype(float)
    y_pred = y_true.copy()
    return y_true, y_pred


# calibration_analysis()  
class TestCalibrationAnalysis:
    def test_returns_required_keys(self):
        y_true, y_pred = _synthetic_predictions()
        result = calibration_analysis(y_true, y_pred)
        required = {
            'brier_score', 'brier_baseline', 'brier_skill',
            'ece', 'bins', 'n_bins_populated', 'assessment', 'n_samples', 'churn_rate',}
        assert required == set(result.keys())

    def test_brier_score_in_valid_range(self):
        y_true, y_pred = _synthetic_predictions()
        result = calibration_analysis(y_true, y_pred)
        assert 0.0 <= result['brier_score'] <= 1.0

    def test_brier_skill_positive_for_useful_model(self):
        """A model better than naive baseline should have brier_skill > 0 """

        y_true, y_pred = _synthetic_predictions()
        result = calibration_analysis(y_true, y_pred)
        assert result['brier_skill'] > 0, (
            f"Expected positive skill, got {result['brier_skill']:.4f}")

    def test_perfect_model_brier_score_zero(self):
        """A perfect model (pred == true) should have Brier Score = 0 """

        y_true, y_pred = _perfect_predictions()
        result = calibration_analysis(y_true, y_pred)
        assert result['brier_score'] < 1e-6, (
            f"Expected ~0 Brier Score for perfect model, got {result['brier_score']:.6f}")

    def test_brier_baseline_equals_variance(self):
        """brier_baseline should equal churn_rate * (1 - churn_rate)"""

        y_true, y_pred = _synthetic_predictions()
        result = calibration_analysis(y_true, y_pred)
        expected = result['churn_rate'] * (1 - result['churn_rate'])
        assert abs(result['brier_baseline'] - expected) < 1e-4

    def test_ece_non_negative(self):
        y_true, y_pred = _synthetic_predictions()
        result = calibration_analysis(y_true, y_pred)
        assert result['ece'] >= 0.0

    def test_bins_count_matches_n_bins(self):
        y_true, y_pred = _synthetic_predictions()
        result = calibration_analysis(y_true, y_pred, n_bins=8)
        assert len(result['bins']) == 8

    def test_bins_cover_unit_interval(self):
        """First bin starts at 0, last bin ends at 1 """

        y_true, y_pred = _synthetic_predictions()
        result = calibration_analysis(y_true, y_pred)
        assert result['bins'][0]['bin_lower'] == 0.0
        assert result['bins'][-1]['bin_upper'] == 1.0

    def test_assessment_is_valid_string(self):
        y_true, y_pred = _synthetic_predictions()
        result = calibration_analysis(y_true, y_pred)
        assert result['assessment'] in {'Well calibrated', 'Moderate', 'Poorly calibrated'}

    def test_n_samples_matches_input(self):
        y_true, y_pred = _synthetic_predictions(n=300)
        result = calibration_analysis(y_true, y_pred)
        assert result['n_samples'] == 300

    def test_churn_rate_matches_y_true_mean(self):
        y_true, y_pred = _synthetic_predictions()
        result = calibration_analysis(y_true, y_pred)
        assert abs(result['churn_rate'] - float(y_true.mean())) < 1e-4

    def test_populated_bins_have_valid_fields(self):
        """All populated bins must have mean_pred and actual_rate in [0, 1] """
        y_true, y_pred = _synthetic_predictions()
        result = calibration_analysis(y_true, y_pred)
        for b in result['bins']:
            if b['n'] > 0:
                assert 0.0 <= b['mean_pred'] <= 1.0
                assert 0.0 <= b['actual_rate'] <= 1.0


class TestCalibrationSummary:

    def test_returns_string(self):
        y_true, y_pred = _synthetic_predictions()
        cal = calibration_analysis(y_true, y_pred)
        summary = calibration_summary(cal)
        assert isinstance(summary, str)

    def test_summary_contains_assessment(self):
        y_true, y_pred = _synthetic_predictions()
        cal = calibration_analysis(y_true, y_pred)
        summary = calibration_summary(cal)
        assert cal['assessment'] in summary

    def test_summary_contains_brier_score(self):
        y_true, y_pred = _synthetic_predictions()
        cal = calibration_analysis(y_true, y_pred)
        summary = calibration_summary(cal)
        assert str(cal['brier_score']) in summary


# sensitivity_analysis() 
class TestSensitivityAnalysis:

    def test_returns_required_keys(self):
        result = sensitivity_analysis(n_intervened=500, n_true_positives=120)
        required = {
            'roi_grid', 'clv_values', 'success_rate_values',
            'campaign_cost', 'base_roi', 'tornado',
            'n_intervened', 'n_true_positives',}
        assert required == set(result.keys())

    def test_roi_grid_shape_matches_ranges(self):
        result = sensitivity_analysis(
            n_intervened=500, n_true_positives=120,
            clv_monthly_range=(40, 120, 7),
            success_rate_range=(0.10, 0.50, 5),)
        assert result['roi_grid'].shape == (7, 5)

    def test_campaign_cost_is_correct(self):
        """campaign_cost must equal n_intervened * retention_cost"""
        result = sensitivity_analysis(n_intervened=400, n_true_positives=100, retention_cost=25.0,)
        assert abs(result['campaign_cost'] - 400 * 25.0) < 0.01

    def test_base_roi_consistent_with_assumptions(self):
        """base_roi must match manual computation at ASSUMPTIONS defaults"""
        n_tp = 100
        n_int = 400
        clv = ASSUMPTIONS['CLV_MONTHLY']
        sr = ASSUMPTIONS['SUCCESS_RATE']
        cost = ASSUMPTIONS['RETENTION_COST']
        h = ASSUMPTIONS['HORIZON_MONTHS']
        expected_roi = n_tp * sr * clv * h - n_int * cost
        result = sensitivity_analysis(n_intervened=n_int, n_true_positives=n_tp)
        assert abs(result['base_roi'] - expected_roi) < 0.01

    def test_tornado_is_sorted_by_impact_descending(self):
        result = sensitivity_analysis(n_intervened=500, n_true_positives=120)
        impacts = [row['impact'] for row in result['tornado']]
        assert impacts == sorted(impacts, reverse=True), "Tornado must be sorted by impact"

    def test_tornado_has_four_assumptions(self):
        result = sensitivity_analysis(n_intervened=500, n_true_positives=120)
        assert len(result['tornado']) == 4

    def test_tornado_impact_is_non_negative(self):
        result = sensitivity_analysis(n_intervened=500, n_true_positives=120)
        for row in result['tornado']:
            assert row['impact'] >= 0.0, f"Negative impact: {row}"

    def test_higher_clv_increases_roi(self):
        """Higher CLV should produce higher ROI in the grid """
        result = sensitivity_analysis(
            n_intervened=500, n_true_positives=120,
            clv_monthly_range=(40, 120, 5),
            success_rate_range=(0.30, 0.30, 1),)
        grid = result['roi_grid'][:, 0]
        assert all(grid[i] <= grid[i+1] for i in range(len(grid)-1)), ("ROI should increase monotonically with CLV")

    def test_higher_success_rate_increases_roi(self):
        """Higher success rate should produce higher ROI in the grid."""
        result = sensitivity_analysis(n_intervened=500, n_true_positives=120,
            clv_monthly_range=(65, 65, 1), success_rate_range=(0.10, 0.50, 5),)
        grid = result['roi_grid'][0, :]
        assert all(grid[i] <= grid[i+1] for i in range(len(grid)-1)), ("ROI should increase monotonically with success_rate")

    def test_zero_true_positives_gives_negative_roi(self):
        """With no true positives, ROI must be negative (cost only) """
        result = sensitivity_analysis(n_intervened=200, n_true_positives=0)
        assert result['base_roi'] < 0.0


class TestSensitivitySummary:
    def test_returns_string(self):
        sens = sensitivity_analysis(n_intervened=500, n_true_positives=120)
        assert isinstance(sensitivity_summary(sens), str)

    def test_summary_contains_base_roi(self):
        sens = sensitivity_analysis(n_intervened=500, n_true_positives=120)
        summary = sensitivity_summary(sens)
        assert 'Base ROI' in summary


# optimize_threshold() 
class TestOptimizeThreshold:

    def test_returns_required_keys(self):
        y_true, y_pred = _synthetic_predictions()
        result = optimize_threshold(y_true, y_pred)
        required = {
            'optimal_threshold', 'optimal_roi', 'f1_threshold', 'f1_roi',
            'roi_gain', 'default_threshold_roi', 'optimal_metrics', 'curve', 'assumptions',}
        assert required == set(result.keys())

    def test_uniform_clv_per_customer_matches_flat_scalar(self):
        # A clv_per_customer array that's constant everywhere should reproduce the exact same ROI as the flat clv_monthly scalar it's standing in for - 
        # confirms the per-customer path is a strict generalization, not a different calculation.
        y_true, y_pred = _synthetic_predictions()
        flat = optimize_threshold(y_true, y_pred, clv_monthly=80.0)
        uniform_array = optimize_threshold(y_true, y_pred, clv_per_customer=np.full_like(y_true, 80.0))
        assert flat['optimal_roi'] == pytest.approx(uniform_array['optimal_roi'])
        assert flat['optimal_threshold'] == pytest.approx(uniform_array['optimal_threshold'])

    def test_clv_per_customer_reflects_who_gets_flagged(self):
        # Concentrate high CLV on the customers most likely to be true positives (highest y_pred among actual churners) and confirm the ROI curve responds - 
        # the optimizer should be sensitive to *whose* CLV is high, not just the average CLV.
        y_true, y_pred = _synthetic_predictions()
        clv_flat = np.full_like(y_true, 65.0)
        clv_concentrated = np.where(y_pred > np.median(y_pred), 150.0, 20.0)
        result_flat = optimize_threshold(y_true, y_pred, clv_per_customer=clv_flat)
        result_concentrated = optimize_threshold(y_true, y_pred, clv_per_customer=clv_concentrated)
        assert result_concentrated['optimal_roi'] != pytest.approx(result_flat['optimal_roi'])

    def test_optimal_threshold_in_unit_interval(self):
        y_true, y_pred = _synthetic_predictions()
        result = optimize_threshold(y_true, y_pred)
        assert 0.0 < result['optimal_threshold'] < 1.0

    def test_optimal_roi_geq_default_roi(self):
        """Optimized threshold must yield ROI >= default threshold (0.50)."""
        y_true, y_pred = _synthetic_predictions()
        result = optimize_threshold(y_true, y_pred)
        assert result['optimal_roi'] >= result['default_threshold_roi'] - 0.01, (
            f"Optimal ROI {result['optimal_roi']:.0f} < default {result['default_threshold_roi']:.0f}")

    def test_curve_length_matches_n_thresholds(self):
        y_true, y_pred = _synthetic_predictions()
        result = optimize_threshold(y_true, y_pred, n_thresholds=50)
        assert len(result['curve']) == 50

    def test_curve_thresholds_monotonically_increasing(self):
        y_true, y_pred = _synthetic_predictions()
        result = optimize_threshold(y_true, y_pred)
        thresholds = [p['threshold'] for p in result['curve']]
        assert all(thresholds[i] <= thresholds[i+1] for i in range(len(thresholds)-1))

    def test_curve_recall_monotonically_decreasing(self):
        """As threshold increases, recall should be non-increasing """
        y_true, y_pred = _synthetic_predictions()
        result = optimize_threshold(y_true, y_pred)
        recalls = [p['recall'] for p in result['curve']]
        # Allow small ties but no increases
        assert all(recalls[i] >= recalls[i+1] - 1e-9 for i in range(len(recalls)-1))

    def test_optimal_metrics_has_required_keys(self):
        y_true, y_pred = _synthetic_predictions()
        result = optimize_threshold(y_true, y_pred)
        required = {'precision', 'recall', 'f1', 'tp', 'fp', 'n_intervened'}
        assert required == set(result['optimal_metrics'].keys())

    def test_optimal_metrics_precision_recall_in_unit_interval(self):
        y_true, y_pred = _synthetic_predictions()
        result = optimize_threshold(y_true, y_pred)
        m = result['optimal_metrics']
        assert 0.0 <= m['precision'] <= 1.0
        assert 0.0 <= m['recall'] <= 1.0

    def test_roi_gain_equals_optimal_minus_f1_roi(self):
        """roi_gain must equal optimal_roi - f1_roi"""
        y_true, y_pred = _synthetic_predictions()
        result = optimize_threshold(y_true, y_pred)
        expected_gain = result['optimal_roi'] - result['f1_roi']
        assert abs(result['roi_gain'] - expected_gain) < 0.01

    def test_assumptions_returned_correctly(self):
        y_true, y_pred = _synthetic_predictions()
        result = optimize_threshold(y_true, y_pred, clv_monthly=80.0, retention_cost=30.0, success_rate=0.25, horizon_months=9,)
        assert result['assumptions']['clv_monthly'] == 80.0
        assert result['assumptions']['retention_cost'] == 30.0
        assert result['assumptions']['success_rate'] == 0.25
        assert result['assumptions']['horizon_months'] == 9

    def test_assumptions_clv_source_is_flat_when_no_per_customer_array_given(self):
        # Without clv_per_customer, the flat clv_monthly scalar drives the result - 
        # the assumptions dict should say so explicitly (see DECISIONS.md #12/#13).
        y_true, y_pred = _synthetic_predictions()
        result = optimize_threshold(y_true, y_pred, clv_monthly=80.0)
        assert result['assumptions']['clv_source'] == 'clv_monthly'
        assert result['assumptions']['avg_clv_used'] == pytest.approx(80.0)

    def test_assumptions_clv_source_is_per_customer_when_array_given(self):
        # With clv_per_customer passed, it takes priority over clv_monthly - 
        # the assumptions dict should reflect that clv_monthly was NOT what drove the calculation, and report the actual average CLV used instead of the (unused) flat scalar.
        y_true, y_pred = _synthetic_predictions()
        clv_array = np.linspace(20.0, 150.0, len(y_true))
        result = optimize_threshold(y_true, y_pred, clv_monthly=80.0, clv_per_customer=clv_array)
        assert result['assumptions']['clv_source'] == 'clv_per_customer'
        assert result['assumptions']['avg_clv_used'] == pytest.approx(clv_array.mean())
        assert result['assumptions']['avg_clv_used'] != pytest.approx(80.0)

    def test_high_cost_shifts_threshold_higher(self):
        """ Very high retention cost makes intervening expensive, so the optimal threshold should be higher (intervene only on near-certain churners) """
        y_true, y_pred = _synthetic_predictions()
        result_low_cost = optimize_threshold(y_true, y_pred, retention_cost=5.0)
        result_high_cost = optimize_threshold(y_true, y_pred, retention_cost=200.0)
        assert result_high_cost['optimal_threshold'] >= result_low_cost['optimal_threshold'] - 0.05, (
            "Higher cost should shift optimal threshold up or hold it constant")

    def test_zero_cost_selects_low_threshold(self):
        """ With zero retention cost, all interventions are free, so the model should flag as many churners as possible (low threshold)"""
        y_true, y_pred = _synthetic_predictions()
        result = optimize_threshold(y_true, y_pred, retention_cost=0.0)
        assert result['optimal_threshold'] < 0.5, (
            f"With zero cost, threshold should be low, got {result['optimal_threshold']:.2f}")


class TestThresholdSummary:

    def test_returns_string(self):
        y_true, y_pred = _synthetic_predictions()
        opt = optimize_threshold(y_true, y_pred)
        assert isinstance(threshold_summary(opt), str)

    def test_summary_contains_optimal_threshold(self):
        y_true, y_pred = _synthetic_predictions()
        opt = optimize_threshold(y_true, y_pred)
        summary = threshold_summary(opt)
        assert f"{opt['optimal_threshold']:.2f}" in summary

    def test_summary_contains_roi_gain(self):
        y_true, y_pred = _synthetic_predictions()
        opt = optimize_threshold(y_true, y_pred)
        summary = threshold_summary(opt)
        assert 'ROI gain' in summary

    def test_summary_distinguishes_clv_source(self):
        # A reader of the summary text alone (not the raw dict) should be able to tell whether a flat CLV assumption or per-customer CLV drove the reported ROI figures.
        y_true, y_pred = _synthetic_predictions()
        opt_flat = optimize_threshold(y_true, y_pred, clv_monthly=80.0)
        opt_per_customer = optimize_threshold(y_true, y_pred, clv_per_customer=np.full_like(y_true, 80.0))
        assert 'flat assumption' in threshold_summary(opt_flat)
        assert 'per-customer avg' in threshold_summary(opt_per_customer)
