import pandas as pd
import pytest

from dashboard.business import (
    ab_test_break_even_analysis, campaign_scenario_table,
    executive_kpis, optimal_threshold,
    physics_snapshot,threshold_roi_curve,)

# Fixtures 
@pytest.fixture
def small_df():
    """ 10-row DataFrame with all columns used by the dashboard business functions """
    return pd.DataFrame({
        "risk_level": ["HIGH", "HIGH", "HIGH", "MEDIUM", "MEDIUM", "LOW",  "LOW",  "UNCERTAIN", "LOW", "HIGH"],
        "prob_churn": [0.80, 0.75, 0.70, 0.45, 0.40, 0.20, 0.18, 0.55, 0.22, 0.82],
        "sigma_prob": [0.08] * 10,
        "segment": ["High Risk / Resilient"] * 4 + ["Medium Risk / Resilient"] * 3
                       + ["Uncertain"] * 2 + ["Stable"],
        "E0": [0.2, 0.3, 0.25, 0.5, 0.55, 0.7, 0.75, 0.45, 0.72, 0.18],
        "E_eq": [0.3, 0.35, 0.28, 0.55, 0.52, 0.72, 0.68, 0.50, 0.70, 0.25],
        "gamma": [0.08, 0.10, 0.09, 0.15, 0.18, 0.25, 0.22, 0.12, 0.20, 0.07],
        "tau": [12.5, 10.0, 11.1, 6.7, 5.6, 4.0, 4.5, 8.3, 5.0, 14.3],})


@pytest.fixture
def empty_df():
    return pd.DataFrame({"risk_level": pd.Series([], dtype=str), "prob_churn": pd.Series([], dtype=float),})


# executive_kpis
class TestExecutiveKpis:

    def test_returns_all_keys(self, small_df):
        kpis = executive_kpis(small_df, clv=65, ret_cost=25, horizon=6)
        expected = {
            "n_total", "n_high", "n_uncertain", "pct_high",
            "rev_at_risk", "est_saved", "rev_recovered",
            "campaign_cost", "net_roi",}
        assert expected == set(kpis.keys())

    def test_n_high_correct(self, small_df):
        kpis = executive_kpis(small_df, clv=65, ret_cost=25, horizon=6)
        assert kpis["n_high"] == 4   # 4 HIGH rows in fixture

    def test_n_uncertain_correct(self, small_df):
        kpis = executive_kpis(small_df, clv=65, ret_cost=25, horizon=6)
        assert kpis["n_uncertain"] == 1

    def test_pct_high_is_fraction(self, small_df):
        kpis = executive_kpis(small_df, clv=65, ret_cost=25, horizon=6)
        assert 0 <= kpis["pct_high"] <= 1
        assert kpis["pct_high"] == pytest.approx(kpis["n_high"] / kpis["n_total"])

    def test_rev_at_risk_positive(self, small_df):
        kpis = executive_kpis(small_df, clv=65, ret_cost=25, horizon=6)
        assert kpis["rev_at_risk"] > 0

    def test_campaign_cost_equals_n_high_times_ret_cost(self, small_df):
        kpis = executive_kpis(small_df, clv=65, ret_cost=25, horizon=6)
        assert kpis["campaign_cost"] == pytest.approx(kpis["n_high"] * 25)

    def test_net_roi_equals_recovered_minus_cost(self, small_df):
        kpis = executive_kpis(small_df, clv=65, ret_cost=25, horizon=6)
        assert kpis["net_roi"] == pytest.approx(kpis["rev_recovered"] - kpis["campaign_cost"], abs=0.01)

    def test_zero_high_risk_zero_roi(self, small_df):
        df = small_df.copy()
        df["risk_level"] = "LOW"
        kpis = executive_kpis(df, clv=65, ret_cost=25, horizon=6)
        assert kpis["n_high"] == 0
        assert kpis["campaign_cost"] == 0
        assert kpis["net_roi"] == 0

    def test_success_rate_scales_recovered_revenue(self, small_df):
        kpis_30 = executive_kpis(small_df, clv=65, ret_cost=25, horizon=6, success_rate=0.30)
        kpis_60 = executive_kpis(small_df, clv=65, ret_cost=25, horizon=6, success_rate=0.60)
        assert kpis_60["rev_recovered"] == pytest.approx(2 * kpis_30["rev_recovered"], rel=1e-5)

    def test_empty_df_does_not_crash(self, empty_df):
        kpis = executive_kpis(empty_df, clv=65, ret_cost=25, horizon=6)
        assert kpis["n_total"] >= 1   # clamped to 1
        assert kpis["n_high"] == 0
        assert kpis["net_roi"] == 0

    def test_higher_clv_increases_roi(self, small_df):
        kpis_low = executive_kpis(small_df, clv=40,  ret_cost=25, horizon=6)
        kpis_high = executive_kpis(small_df, clv=120, ret_cost=25, horizon=6)
        assert kpis_high["net_roi"] > kpis_low["net_roi"]

    def test_longer_horizon_increases_revenue(self, small_df):
        kpis_3 = executive_kpis(small_df, clv=65, ret_cost=25, horizon=3)
        kpis_12 = executive_kpis(small_df, clv=65, ret_cost=25, horizon=12)
        assert kpis_12["rev_recovered"] > kpis_3["rev_recovered"]


# campaign_scenario_table 
class TestCampaignScenarioTable:

    @pytest.fixture
    def three_totals(self):
        best = {"saved": 30, "revenue": 5_850, "cost": 1_000, "roi": 4_850, "roi_pct": 485}
        base = {"saved": 20, "revenue": 3_900, "cost": 1_000, "roi": 2_900, "roi_pct": 290}
        worst = {"saved": 10, "revenue": 1_950, "cost": 1_000, "roi":   950, "roi_pct":  95}
        return best, base, worst

    def test_returns_dataframe(self, three_totals):
        df = campaign_scenario_table(*three_totals)
        assert isinstance(df, pd.DataFrame)

    def test_has_three_rows(self, three_totals):
        df = campaign_scenario_table(*three_totals)
        assert len(df) == 3

    def test_scenario_labels(self, three_totals):
        df = campaign_scenario_table(*three_totals)
        assert list(df["scenario"]) == ["Best case", "Base case", "Worst case"]

    def test_roi_decreasing_from_best_to_worst(self, three_totals):
        df = campaign_scenario_table(*three_totals)
        rois = df["roi"].tolist()
        assert rois[0] > rois[1] > rois[2]

    def test_required_columns_present(self, three_totals):
        df = campaign_scenario_table(*three_totals)
        for col in ("scenario", "customers_saved", "revenue", "cost", "roi", "roi_pct"):
            assert col in df.columns


# threshold_roi_curve 
class TestThresholdRoiCurve:

    def test_returns_dataframe(self, small_df):
        df = threshold_roi_curve(small_df, clv=65, ret_cost=25, horizon=6)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self, small_df):
        df = threshold_roi_curve(small_df, clv=65, ret_cost=25, horizon=6, n_thresholds=10)
        assert len(df) == 10

    def test_thresholds_increasing(self, small_df):
        df = threshold_roi_curve(small_df, clv=65, ret_cost=25, horizon=6)
        assert df["threshold"].is_monotonic_increasing

    def test_high_threshold_fewer_flagged(self, small_df):
        df = threshold_roi_curve(small_df, clv=65, ret_cost=25, horizon=6, n_thresholds=5)
        assert df["n_flagged"].iloc[-1] <= df["n_flagged"].iloc[0]

    def test_required_columns(self, small_df):
        df = threshold_roi_curve(small_df, clv=65, ret_cost=25, horizon=6)
        for col in ("threshold", "n_flagged", "tp_estimate", "revenue", "cost", "roi", "roi_pct"):
            assert col in df.columns

    def test_cost_equals_n_flagged_times_ret_cost(self, small_df):
        df = threshold_roi_curve(small_df, clv=65, ret_cost=25, horizon=6)
        expected_cost = df["n_flagged"] * 25
        pd.testing.assert_series_equal(df["cost"].round(2), expected_cost.round(2), check_names=False)

    def test_roi_equals_revenue_minus_cost(self, small_df):
        df = threshold_roi_curve(small_df, clv=65, ret_cost=25, horizon=6)
        expected_roi = (df["revenue"] - df["cost"]).round(2)
        pd.testing.assert_series_equal(df["roi"].round(2), expected_roi, check_names=False)


# optimal_threshold 
class TestOptimalThreshold:

    def test_returns_dict_with_required_keys(self, small_df):
        curve = threshold_roi_curve(small_df, clv=65, ret_cost=25, horizon=6)
        opt = optimal_threshold(curve)
        for key in ("threshold", "n_flagged", "roi", "roi_pct"):
            assert key in opt

    def test_threshold_is_actual_maximum(self, small_df):
        curve = threshold_roi_curve(small_df, clv=65, ret_cost=25, horizon=6)
        opt = optimal_threshold(curve)
        assert opt["roi"] == pytest.approx(curve["roi"].max(), abs=0.01)

    def test_empty_curve_returns_defaults(self):
        opt = optimal_threshold(pd.DataFrame())
        assert opt["threshold"] == 0.5
        assert opt["n_flagged"] == 0
        assert opt["roi"] == 0.0

    def test_threshold_within_range(self, small_df):
        curve = threshold_roi_curve(small_df, clv=65, ret_cost=25, horizon=6)
        opt = optimal_threshold(curve)
        assert 0 <= opt["threshold"] <= 1


# physics_snapshot 
class TestPhysicsSnapshot:

    def test_returns_all_keys(self):
        snap = physics_snapshot(E0=0.4, gamma=0.1, E_eq=0.55, t_horizon=6.0)
        expected = {"tau", "E_t", "delta_E", "structural_risk", "will_churn", "p_churn_approx"}
        assert expected == set(snap.keys())

    def test_tau_is_reciprocal_of_gamma(self):
        snap = physics_snapshot(E0=0.5, gamma=0.2, E_eq=0.5, t_horizon=6.0)
        assert snap["tau"] == pytest.approx(1 / 0.2, rel=1e-4)

    def test_E_t_between_0_and_1(self):
        snap = physics_snapshot(E0=0.4, gamma=0.1, E_eq=0.55, t_horizon=6.0)
        assert 0 <= snap["E_t"] <= 1

    def test_high_E0_above_E_eq_converges_down(self):
        """E0 > E_eq → E(t) should be between E_eq and E0."""
        snap = physics_snapshot(E0=0.9, gamma=0.15, E_eq=0.4, t_horizon=6.0)
        assert snap["E_t"] >= 0.4    # at least E_eq
        assert snap["E_t"] <= 0.9    # at most E0

    def test_long_horizon_approaches_E_eq(self):
        """Very long horizon -> E(t) ≈ E_eq"""
        snap = physics_snapshot(E0=0.2, gamma=0.5, E_eq=0.7, t_horizon=100.0)
        assert abs(snap["E_t"] - 0.7) < 0.01

    def test_structural_risk_when_E_eq_low(self):
        snap = physics_snapshot(E0=0.5, gamma=0.1, E_eq=0.10, t_horizon=6.0)
        assert snap["structural_risk"] is True

    def test_no_structural_risk_when_E_eq_high(self):
        snap = physics_snapshot(E0=0.5, gamma=0.1, E_eq=0.70, t_horizon=6.0)
        assert snap["structural_risk"] is False

    def test_will_churn_flag_consistent_with_E_t(self):
        snap = physics_snapshot(E0=0.1, gamma=0.3, E_eq=0.1, t_horizon=6.0)
        assert snap["will_churn"] == (snap["E_t"] < 0.25)

    def test_p_churn_approx_is_complement_of_E_t(self):
        snap = physics_snapshot(E0=0.4, gamma=0.1, E_eq=0.55, t_horizon=6.0)
        assert snap["p_churn_approx"] == pytest.approx(1.0 - snap["E_t"], abs=1e-4)

    def test_delta_E_sign(self):
        """ delta_E = E_t - E_eq; sign indicates direction of energy change """
        snap_above = physics_snapshot(E0=0.9, gamma=0.1, E_eq=0.4, t_horizon=6.0)
        snap_below = physics_snapshot(E0=0.1, gamma=0.1, E_eq=0.8, t_horizon=6.0)
        assert snap_above["delta_E"] > 0    # E_t above E_eq
        assert snap_below["delta_E"] < 0    # E_t below E_eq


# ab_test_break_even_analysis 
class TestABTestBreakEvenAnalysis:

    def test_returns_all_keys(self):
        result = ab_test_break_even_analysis(100, clv=65, ret_cost=25, horizon=6)
        expected = {"break_even_rate", "roi_at_30_pct", "roi_at_break_even", "is_viable"}
        assert expected == set(result.keys())

    def test_break_even_formula(self):
        """break_even = ret_cost / (clv * horizon)"""
        result = ab_test_break_even_analysis(100, clv=65, ret_cost=25, horizon=6)
        expected_be = 25 / (65 * 6)
        assert result["break_even_rate"] == pytest.approx(expected_be, rel=1e-4)

    def test_roi_at_break_even_is_near_zero(self):
        result = ab_test_break_even_analysis(100, clv=65, ret_cost=25, horizon=6)
        # ROI at break-even should be approximately 0
        assert abs(result["roi_at_break_even"]) < 10   

    def test_roi_at_30_pct_positive_when_viable(self):
        """With standard params, 30% SR should be ROI-positive."""
        result = ab_test_break_even_analysis(100, clv=65, ret_cost=25, horizon=6)
        assert result["roi_at_30_pct"] > 0

    def test_is_viable_true_for_low_break_even(self):
        """Low cost relative to CLV -> break-even is low -> viable """
        result = ab_test_break_even_analysis(100, clv=200, ret_cost=10, horizon=6)
        assert result["is_viable"] is True

    def test_is_viable_false_for_high_break_even(self):
        """Very high cost relative to CLV -> break-even > 50% -> not viable"""
        result = ab_test_break_even_analysis(100, clv=20, ret_cost=50, horizon=1)
        assert result["is_viable"] is False

    def test_zero_n_gives_zero_roi(self):
        result = ab_test_break_even_analysis(0, clv=65, ret_cost=25, horizon=6)
        assert result["roi_at_30_pct"] == 0.0

    def test_break_even_clipped_to_1(self):
        """Impossible scenario: ret_cost > clv * horizon -> clamped to 1"""
        result = ab_test_break_even_analysis(100, clv=1, ret_cost=100, horizon=1)
        assert result["break_even_rate"] <= 1.0

    def test_higher_clv_lowers_break_even(self):
        low_clv = ab_test_break_even_analysis(100, clv=30, ret_cost=25, horizon=6)
        high_clv = ab_test_break_even_analysis(100, clv=150, ret_cost=25, horizon=6)
        assert high_clv["break_even_rate"] < low_clv["break_even_rate"]
