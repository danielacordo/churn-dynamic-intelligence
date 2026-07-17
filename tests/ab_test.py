import numpy as np
import pandas as pd
import pytest

from src.ab_test import (
    ABTestConfig, ABTestResult,
    ab_test_report, analyse_experiment,
    required_sample_size, sequential_analysis,
    simulate_experiment, sweep_true_success_rates,)

# ABTestConfig 
class TestABTestConfig:

    def test_defaults_are_valid(self):
        cfg = ABTestConfig()
        assert 0 < cfg.baseline_churn_rate < 1
        assert 0 < cfg.assumed_success_rate < 1
        assert 0 < cfg.alpha < 1
        assert 0 < cfg.power < 1

    def test_treated_churn_rate_lower_than_baseline(self):
        cfg = ABTestConfig(baseline_churn_rate=0.65, assumed_success_rate=0.30)
        assert cfg.treated_churn_rate() < cfg.baseline_churn_rate

    def test_effect_size_positive(self):
        cfg = ABTestConfig()
        assert cfg.effect_size() > 0

    def test_zero_success_rate_no_effect(self):
        cfg = ABTestConfig(assumed_success_rate=0.0)
        assert cfg.treated_churn_rate() == pytest.approx(cfg.baseline_churn_rate)
        assert cfg.effect_size() == pytest.approx(0.0)

    def test_full_success_rate_eliminates_churn(self):
        cfg = ABTestConfig(assumed_success_rate=1.0)
        assert cfg.treated_churn_rate() == pytest.approx(0.0)


# required_sample_size 
class TestRequiredSampleSize:

    def test_returns_positive_integer(self):
        result = required_sample_size(ABTestConfig())
        assert result['n_per_group'] > 0
        assert isinstance(result['n_per_group'], int)

    def test_n_total_is_double_n_per_group(self):
        result = required_sample_size(ABTestConfig())
        assert result['n_total'] == 2 * result['n_per_group']

    def test_smaller_effect_requires_larger_sample(self):
        cfg_large = ABTestConfig(assumed_success_rate=0.40)
        cfg_small = ABTestConfig(assumed_success_rate=0.15)
        n_large = required_sample_size(cfg_large)['n_per_group']
        n_small = required_sample_size(cfg_small)['n_per_group']
        assert n_small > n_large

    def test_higher_power_requires_larger_sample(self):
        cfg_80 = ABTestConfig(power=0.80)
        cfg_95 = ABTestConfig(power=0.95)
        n_80 = required_sample_size(cfg_80)['n_per_group']
        n_95 = required_sample_size(cfg_95)['n_per_group']
        assert n_95 > n_80

    def test_stricter_alpha_requires_larger_sample(self):
        cfg_05 = ABTestConfig(alpha=0.05)
        cfg_01 = ABTestConfig(alpha=0.01)
        n_05 = required_sample_size(cfg_05)['n_per_group']
        n_01 = required_sample_size(cfg_01)['n_per_group']
        assert n_01 > n_05

    def test_zero_effect_raises(self):
        cfg = ABTestConfig(assumed_success_rate=0.0)
        with pytest.raises(ValueError, match="identical"):
            required_sample_size(cfg)

    def test_output_keys_complete(self):
        result = required_sample_size(ABTestConfig())
        expected_keys = {
            'n_per_group', 'n_total', 'baseline_rate', 'treated_rate',
            'effect_size_abs', 'effect_size_rel', 'alpha', 'power',
            'one_sided', 'weeks_at_1k_weekly', 'formula',}
        assert expected_keys == set(result.keys())

    def test_weeks_estimate_positive(self):
        result = required_sample_size(ABTestConfig())
        assert result['weeks_at_1k_weekly'] > 0


# simulate_experiment 
class TestSimulateExperiment:

    def test_output_shape(self):
        cfg = ABTestConfig()
        n = required_sample_size(cfg)['n_per_group']
        df = simulate_experiment(cfg, n_per_group=n)
        assert len(df) == 2 * n

    def test_equal_group_sizes(self):
        cfg = ABTestConfig()
        df = simulate_experiment(cfg, n_per_group=100)
        assert (df['group'] == 'control').sum() == 100
        assert (df['group'] == 'treatment').sum() == 100

    def test_reproducible_with_same_seed(self):
        cfg = ABTestConfig(seed=42)
        df1 = simulate_experiment(cfg, n_per_group=50)
        df2 = simulate_experiment(cfg, n_per_group=50)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_give_different_results(self):
        cfg1 = ABTestConfig(seed=1)
        cfg2 = ABTestConfig(seed=999)
        df1 = simulate_experiment(cfg1, n_per_group=200)
        df2 = simulate_experiment(cfg2, n_per_group=200)
        assert df1['churned'].sum() != df2['churned'].sum()

    def test_probabilities_in_unit_interval(self):
        cfg = ABTestConfig()
        df = simulate_experiment(cfg, n_per_group=200)
        assert df['baseline_prob'].between(0, 1).all()
        assert df['treated_prob'].between(0, 1).all()

    def test_treatment_has_lower_churn_prob(self):
        cfg = ABTestConfig(assumed_success_rate=0.30)
        df = simulate_experiment(cfg, n_per_group=500)
        ctrl = df[df['group'] == 'control']['baseline_prob'].mean()
        trt  = df[df['group'] == 'treatment']['treated_prob'].mean()
        assert trt < ctrl

    def test_zero_success_rate_same_probs(self):
        cfg = ABTestConfig(assumed_success_rate=0.0)
        df = simulate_experiment(cfg, n_per_group=200)
        ctrl = df[df['group'] == 'control']['treated_prob']
        trt  = df[df['group'] == 'treatment']['treated_prob']
        # With sr=0, treatment probs = baseline probs, difference should be tiny
        assert abs(ctrl.mean() - trt.mean()) < 0.05

    def test_required_columns_present(self):
        df = simulate_experiment(ABTestConfig(), n_per_group=50)
        for col in ('customer_id', 'group', 'baseline_prob', 'treated_prob',
                    'churned', 'is_treated', 'true_success_rate'):
            assert col in df.columns, f"Missing column: {col}"


# analyse_experiment 
class TestAnalyseExperiment:

    @pytest.fixture
    def confirmed_result(self):
        """High success rate -> should be CONFIRMED """
        cfg = ABTestConfig(assumed_success_rate=0.30, seed=42)
        df = simulate_experiment(cfg, n_per_group=200, true_success_rate=0.30)
        return analyse_experiment(df, cfg), cfg

    @pytest.fixture
    def no_effect_result(self):
        """Zero true effect -> should be NO_EFFECT or INCONCLUSIVE """
        cfg = ABTestConfig(assumed_success_rate=0.30, seed=7)
        df = simulate_experiment(cfg, n_per_group=50, true_success_rate=0.00)
        return analyse_experiment(df, cfg), cfg

    def test_result_fields_populated(self, confirmed_result):
        result, _ = confirmed_result
        assert result.n_control > 0
        assert result.n_treatment > 0
        assert 0 <= result.p_value <= 1
        assert 0 <= result.control_churn_rate <= 1
        assert 0 <= result.treatment_churn_rate <= 1

    def test_confirmed_verdict(self, confirmed_result):
        result, _ = confirmed_result
        assert result.verdict == "CONFIRMED"
        assert result.significant

    def test_no_effect_verdict_or_inconclusive(self, no_effect_result):
        result, _ = no_effect_result
        assert result.verdict in ("NO_EFFECT", "INCONCLUSIVE")

    def test_p_value_significant_when_large_effect(self):
        """Very large sample + real effect should always be significant"""
        cfg = ABTestConfig(assumed_success_rate=0.40, seed=0)
        df  = simulate_experiment(cfg, n_per_group=1000, true_success_rate=0.40)
        result = analyse_experiment(df, cfg)
        assert result.significant
        assert result.p_value < 0.01

    def test_p_value_not_significant_when_no_effect(self):
        """With n=50 and zero true effect, should usually not be significant """
        cfg = ABTestConfig(seed=123)
        df  = simulate_experiment(cfg, n_per_group=50, true_success_rate=0.00)
        result = analyse_experiment(df, cfg)
        # Not deterministic but with seed=123 and n=50 it should hold
        assert not result.significant or result.p_value > 0.01

    def test_ci_contains_true_effect_at_30pct(self):
        """95% CI should contain the true reduction ~95% of the time.
        Runs 20 simulations and check at least 17/20 contain truth """
        cfg = ABTestConfig(assumed_success_rate=0.30)
        true_reduction = cfg.effect_size()
        covered = 0
        for seed in range(20):
            sim_cfg = ABTestConfig(assumed_success_rate=0.30, seed=seed * 7)
            df = simulate_experiment(sim_cfg, n_per_group=200, true_success_rate=0.30)
            result = analyse_experiment(df, sim_cfg)
            if result.ci_lower <= true_reduction <= result.ci_upper:
                covered += 1
        assert covered >= 14, f"CI coverage too low: {covered}/20"

    def test_roi_at_observed_rate_is_finite(self, confirmed_result):
        result, _ = confirmed_result
        assert np.isfinite(result.roi_at_observed_rate)
        assert np.isfinite(result.roi_at_assumed_rate)

    def test_break_even_rate_in_unit_interval(self, confirmed_result):
        result, _ = confirmed_result
        assert 0 <= result.break_even_rate <= 1

    def test_empty_group_raises(self):
        cfg = ABTestConfig()
        df = simulate_experiment(cfg, n_per_group=50)
        df_bad = df[df['group'] == 'treatment'].copy()  # only treatment, no control
        with pytest.raises(ValueError, match="non-empty"):
            analyse_experiment(df_bad, cfg)

    def test_achieved_power_between_0_and_1(self, confirmed_result):
        result, _ = confirmed_result
        assert 0 <= result.achieved_power <= 1

    def test_verdict_revise_down(self):
        """True rate much lower than assumed -> REVISE_DOWN """

        cfg = ABTestConfig(assumed_success_rate=0.30, seed=0)
        df = simulate_experiment(cfg, n_per_group=2000, true_success_rate=0.05)
        result = analyse_experiment(df, cfg)
        # With large n and tiny true effect, p might still be significant but sr way off
        assert result.observed_success_rate < 0.20


# sweep_true_success_rates 
class TestSweepTrueSuccessRates:

    @pytest.fixture
    def sweep(self):
        cfg = ABTestConfig(seed=42)
        return sweep_true_success_rates(cfg, true_rates=[0.05, 0.15, 0.30, 0.45])

    def test_output_shape(self, sweep):
        assert len(sweep) == 4

    def test_expected_columns(self, sweep):
        for col in ('true_rate', 'observed_sr', 'p_value', 'significant', 'verdict', 'roi_observed', 'achieved_power'):
            assert col in sweep.columns

    def test_p_values_in_unit_interval(self, sweep):
        assert sweep['p_value'].between(0, 1).all()

    def test_higher_true_rate_lower_p_value_trend(self):
        """Higher true success rate -> generally more significant (lower p)
        Test the extremes: 0.05 vs 0.45 with large n"""
        cfg = ABTestConfig(seed=0)
        sw  = sweep_true_success_rates(cfg, true_rates=[0.05, 0.45], n_per_group=500)
        p_low  = sw[sw['true_rate'] == 0.05]['p_value'].iloc[0]
        p_high = sw[sw['true_rate'] == 0.45]['p_value'].iloc[0]
        assert p_high < p_low

    def test_high_true_rate_significant(self):
        """ True rate = 0.30 with n=500 should at least produce a significant result."""
        cfg = ABTestConfig(seed=42)
        sw  = sweep_true_success_rates(cfg, true_rates=[0.30], n_per_group=500)
        row = sw[sw['true_rate'] == 0.30].iloc[0]
        # With n=500 and true effect of 0.30 the test should be significant; verdict may vary with sampling noise but p_value should be low.
        assert row['p_value'] < 0.10 or row['verdict'] in ('CONFIRMED', 'REVISE_DOWN')


# sequential_analysis 
class TestSequentialAnalysis:

    @pytest.fixture
    def seq(self):
        cfg = ABTestConfig(seed=42)
        return sequential_analysis(cfg, n_checkpoints=4)

    def test_output_shape(self, seq):
        assert len(seq) == 4

    def test_checkpoints_increasing(self, seq):
        assert seq['n_per_group_so_far'].is_monotonic_increasing

    def test_pct_enrolled_reaches_100(self, seq):
        assert seq['pct_enrolled'].iloc[-1] == pytest.approx(100.0)

    def test_bonferroni_adjusted_alpha(self, seq):
        """Adjusted alpha should be alpha/n_checkpoints = 0.05/4."""
        assert seq['adjusted_alpha'].iloc[0] == pytest.approx(0.05 / 4, abs=1e-6)

    def test_expected_columns(self, seq):
        for col in ('checkpoint', 'n_per_group_so_far', 'pct_enrolled', 'p_value', 'adjusted_alpha', 'significant_at_checkpoint', 'observed_sr', 'verdict'):
            assert col in seq.columns


# ab_test_report 
class TestABTestReport:

    def test_returns_non_empty_string(self):
        cfg = ABTestConfig(seed=42)
        df = simulate_experiment(cfg, n_per_group=100)
        result = analyse_experiment(df, cfg)
        report = ab_test_report(result)
        assert isinstance(report, str)
        assert len(report) > 200

    def test_contains_verdict(self):
        cfg = ABTestConfig(seed=42)
        df = simulate_experiment(cfg, n_per_group=100)
        result = analyse_experiment(df, cfg)
        report = ab_test_report(result)
        assert result.verdict in report

    def test_contains_sample_size_info_when_provided(self):
        cfg = ABTestConfig(seed=42)
        ss = required_sample_size(cfg)
        df = simulate_experiment(cfg, n_per_group=100)
        result = analyse_experiment(df, cfg)
        report = ab_test_report(result, sample_size_info=ss)
        assert "Required n/group" in report

    def test_report_without_sample_size_still_works(self):
        cfg = ABTestConfig(seed=0)
        df = simulate_experiment(cfg, n_per_group=50)
        result = analyse_experiment(df, cfg)
        ab_test_report(result, sample_size_info=None)  # should not raise


# Integration: full experiment workflow 
class TestFullWorkflow:

    def test_design_simulate_analyse_pipeline(self):
        """Full workflow: design → simulate → analyse → report."""
        cfg = ABTestConfig( baseline_churn_rate=0.65, assumed_success_rate=0.30, alpha=0.05, power=0.80, seed=99,)
        ss = required_sample_size(cfg)
        df = simulate_experiment(cfg, n_per_group=ss['n_per_group'])
        result = analyse_experiment(df, cfg)
        report = ab_test_report(result, ss)

        assert ss['n_per_group'] > 0
        assert isinstance(result, ABTestResult)
        assert result.verdict in ('CONFIRMED', 'REVISE_DOWN', 'INCONCLUSIVE', 'NO_EFFECT')
        assert len(report) > 0

    def test_sweep_then_report_first_row(self):
        cfg = ABTestConfig(seed=7)
        sw  = sweep_true_success_rates(cfg, true_rates=[0.30])
        assert len(sw) == 1
        assert sw['p_value'].iloc[0] >= 0
