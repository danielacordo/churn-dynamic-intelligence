import os
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.business import (
    compute_economic_impact, evaluate_model_business,
    model_comparison_table, segment_customer,)


def make_row(**kwargs):
    defaults = {'risk_level': 'MEDIUM', 'tau': 10.0, 'E_eq': 0.50, 'sigma_prob': 0.025, 'prob_churn': 0.50, 'resilience': 'Medium',}
    return pd.Series({**defaults, **kwargs})


@pytest.fixture
def business_df():
    n, churners = 200, 50
    np.random.seed(42)
    prob_churn = np.concatenate([
        np.random.uniform(0.55, 0.95, churners),
        np.random.uniform(0.05, 0.45, n-churners),])
    return pd.DataFrame({
        'Churn_bin': [1]*churners + [0]*(n-churners),
        'prob_churn': prob_churn,
        # compute_economic_impact() defaults to this column (see src/business.py); 
        # this synthetic fixture doesn't need real isotonic calibration, just the same well-separated values under the name the function expects by default.
        'prob_churn_calibrated': prob_churn,
        'risk_level': ['HIGH']*churners + ['LOW']*(n-churners),
        'tau': np.random.uniform(2, 25, n),
        'E_eq': np.random.uniform(0.30, 0.80, n),
        'sigma_prob': np.random.uniform(0.03, 0.15, n),
        'resilience': np.random.choice(['High', 'Medium', 'Low'], n),
        'segment': ['High Risk / Resilient']*churners + ['Stable']*(n-churners),})


class TestSegmentCustomer:

    def test_high_risk_resilient(self):
        seg = segment_customer(make_row(risk_level='HIGH', tau=3.0))
        assert seg['priority'] == 1
        assert 'Resilient' in seg['segment']

    def test_high_risk_fragile(self):
        seg = segment_customer(make_row(risk_level='HIGH', tau=20.0))
        assert seg['priority'] == 1
        assert 'Fragile' in seg['segment']

    def test_low_risk_stable(self):
        seg = segment_customer(make_row(risk_level='LOW'))
        assert seg['priority'] == 4
        assert seg['segment'] == 'Stable'

    def test_uncertain(self):
        seg = segment_customer(make_row(risk_level='UNCERTAIN', sigma_prob=0.25))
        assert seg['segment'] == 'Uncertain'
        assert seg['priority'] == 3

    def test_structural_risk_low_E_eq(self):
        seg = segment_customer(make_row(risk_level='MEDIUM', E_eq=0.15))
        assert seg['segment'] == 'Structural Risk'

    def test_returns_all_keys(self):
        for level in ['HIGH', 'MEDIUM', 'LOW', 'UNCERTAIN']:
            seg = segment_customer(make_row(risk_level=level))
            assert set(seg.keys()) == {'segment', 'action', 'priority', 'reason'}

    def test_priority_ordering(self):
        p_high = segment_customer(make_row(risk_level='HIGH'))['priority']
        p_medium = segment_customer(make_row(risk_level='MEDIUM', E_eq=0.50))['priority']
        p_uncertain= segment_customer(make_row(risk_level='UNCERTAIN'))['priority']
        p_low = segment_customer(make_row(risk_level='LOW'))['priority']
        assert p_high < p_medium <= p_uncertain < p_low

    def test_action_not_empty(self):
        for level in ['HIGH', 'MEDIUM', 'LOW', 'UNCERTAIN']:
            seg = segment_customer(make_row(risk_level=level))
            assert len(seg['action']) > 0


class TestEvaluateModelBusiness:

    @pytest.fixture
    def perfect_data(self):
        y_true = np.array([1]*50 + [0]*150)
        y_proba = np.array([0.95]*50 + [0.05]*150, dtype=float)
        return y_true, y_proba

    @pytest.fixture
    def random_data(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_proba = np.random.uniform(0, 1, 200)
        return y_true, y_proba

    def test_perfect_model_recall_one(self, perfect_data):
        y_true, y_proba = perfect_data
        r = evaluate_model_business(y_true, y_proba)
        assert r['recall'] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_model_precision_one(self, perfect_data):
        y_true, y_proba = perfect_data
        r = evaluate_model_business(y_true, y_proba)
        assert r['precision'] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_model_auc_one(self, perfect_data):
        y_true, y_proba = perfect_data
        r = evaluate_model_business(y_true, y_proba)
        assert r['auc'] == pytest.approx(1.0, abs=1e-6)

    def test_roi_invariant(self, perfect_data):
        y_true, y_proba = perfect_data
        r = evaluate_model_business(y_true, y_proba)
        assert r['net_roi_usd'] == pytest.approx(r['revenue_recovered_usd'] - r['campaign_cost_usd'], abs=1.0)

    def test_tp_plus_fn_equals_churners(self, perfect_data):
        y_true, y_proba = perfect_data
        r = evaluate_model_business(y_true, y_proba)
        assert r['tp'] + r['fn'] == y_true.sum()

    def test_metrics_in_range(self, random_data):
        y_true, y_proba = random_data
        r = evaluate_model_business(y_true, y_proba)
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            assert 0 <= r[metric] <= 1, f"{metric}={r[metric]} out of [0,1]"

    def test_no_churners_detected_roi_non_positive(self):
        y_true  = np.array([0]*190 + [1]*10)
        y_proba = np.zeros(200)
        r = evaluate_model_business(y_true, y_proba, threshold=0.9)
        assert r['net_roi_usd'] <= 0

    def test_higher_clv_increases_revenue(self):
        y_true = np.array([1]*50 + [0]*150)
        y_proba = np.array([0.9]*50 + [0.1]*150, dtype=float)
        r_low = evaluate_model_business(y_true, y_proba, clv_monthly=30.0)
        r_high = evaluate_model_business(y_true, y_proba, clv_monthly=100.0)
        assert r_high['revenue_recovered_usd'] > r_low['revenue_recovered_usd']


class TestComputeEconomicImpact:

    def test_returns_all_keys(self, business_df):
        impact = compute_economic_impact(business_df)
        for key in ['total_customers', 'actual_churners', 'churn_rate',
                    'n_intervened', 'true_positives', 'precision', 'recall',
                    'revenue_at_risk_usd', 'revenue_recovered_usd',
                    'campaign_cost_usd', 'net_roi_usd', 'roi_pct', 'assumptions']:
            assert key in impact, f"Missing key: {key}"

    def test_total_customers_correct(self, business_df):
        assert compute_economic_impact(business_df)['total_customers'] == len(business_df)

    def test_actual_churners_correct(self, business_df):
        assert compute_economic_impact(business_df)['actual_churners'] == business_df['Churn_bin'].sum()

    def test_churn_rate_in_range(self, business_df):
        assert 0 <= compute_economic_impact(business_df)['churn_rate'] <= 1

    def test_roi_invariant(self, business_df):
        impact = compute_economic_impact(business_df)
        assert impact['net_roi_usd'] == pytest.approx(
            impact['revenue_recovered_usd'] - impact['campaign_cost_usd'], abs=1.0)

    def test_assumptions_override(self, business_df):
        base = compute_economic_impact(business_df)
        alt  = compute_economic_impact(business_df, assumptions={'CLV_MONTHLY': 200.0})
        assert alt['revenue_recovered_usd'] > base['revenue_recovered_usd']

    def test_falls_back_to_flat_clv_without_monthly_charges(self, business_df):
        # business_df has no MonthlyCharges column - this is the pre-existing behavior every other test in this class already relies on, made explicit here.
        assert 'MonthlyCharges' not in business_df.columns
        impact = compute_economic_impact(business_df)
        assert impact['avg_clv_monthly_used'] == pytest.approx(65.0)

    def test_uses_per_customer_monthly_charges_when_present(self, business_df):
        # Give flagged (HIGH risk) customers a materially higher MonthlyCharges than
        # everyone else, and confirm the reported average CLV reflects that -- not the
        # flat ASSUMPTIONS default -- once the column is present.
        df = business_df.copy()
        df['MonthlyCharges'] = np.where(df['risk_level'] == 'HIGH', 100.0, 30.0)
        impact = compute_economic_impact(df)
        assert impact['avg_clv_monthly_used'] == pytest.approx(df['MonthlyCharges'].mean())
        assert impact['avg_clv_monthly_used'] != pytest.approx(65.0)

    def test_high_paying_true_positives_increase_revenue_recovered(self, business_df):
        # Same flag/precision/recall either way; only the $ each true positive is worth changes.
        # Revenue recovered must scale with what the *flagged, actually-churning* customers pay, not with a flat dataset-wide average.
        df_low = business_df.copy()
        df_low['MonthlyCharges'] = 30.0
        df_high = business_df.copy()
        df_high['MonthlyCharges'] = 130.0

        impact_low = compute_economic_impact(df_low)
        impact_high = compute_economic_impact(df_high)
        assert impact_high['revenue_recovered_usd'] > impact_low['revenue_recovered_usd']
        assert impact_high['true_positives'] == impact_low['true_positives']

    def test_clv_monthly_assumption_ignored_when_monthly_charges_present(self, business_df):
        # Documented, intentional behavior (see compute_economic_impact docstring): 
        # once MonthlyCharges is available, it is the source of truth for CLV and the flat
        # ASSUMPTIONS['CLV_MONTHLY'] override has no effect - this is not a bug.
        df = business_df.copy()
        df['MonthlyCharges'] = 50.0
        base = compute_economic_impact(df)
        overridden = compute_economic_impact(df, assumptions={'CLV_MONTHLY': 500.0})
        assert base['revenue_recovered_usd'] == pytest.approx(overridden['revenue_recovered_usd'])


class TestModelComparisonTable:

    def _result(self, **kw):
        d = dict(auc=0.80, accuracy=0.82, recall=0.65, precision=0.70, f1=0.67,
                 tp=40, fp=17, fn=10, tn=133, n_intervened=57, customers_saved_est=12.0, revenue_recovered_usd=15600,
                 campaign_cost_usd=1425, net_roi_usd=14175, roi_pct=995.0, revenue_lost_usd=3900)
        d.update(kw)
        return d

    def test_returns_dataframe(self):
        assert isinstance(model_comparison_table({'A': self._result()}), pd.DataFrame)

    def test_one_row_per_model(self):
        t = model_comparison_table({'LR': self._result(), 'RF': self._result(auc=0.83)})
        assert len(t) == 2 and 'LR' in t.index

    def test_key_columns_present(self):
        t = model_comparison_table({'M': self._result()})
        for col in ['AUC', 'Recall churn', 'ROI net $', 'Customers saved']:
            assert col in t.columns, f"Missing column: {col}"
