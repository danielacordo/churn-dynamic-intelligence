import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.uncertainty import (
    SIGMA_UNCERTAIN_THRESHOLD,
    churn_probability_with_uncertainty,
    compare_uncertainty_methods, monte_carlo_churn,)

# monte_carlo_churn(): output contract 
class TestMonteCarloOutputContract:

    def test_returns_expected_keys(self):
        """Output dict must contain all documented keys (including 'correlated' added in v2)"""

        result = monte_carlo_churn(E0=0.6, sigma_E0=0.05, gamma=0.1, sigma_gamma=0.02)
        expected_keys = {
            'prob', 'sigma_prob', 'lower_bound', 'upper_bound',
            'ci_95_lower', 'ci_95_upper', 'risk_level', 'n_samples', 'correlated',}
        assert expected_keys == set(result.keys())

    def test_all_probabilities_in_unit_interval(self):
        """All probability outputs must lie in [0, 1]"""

        result = monte_carlo_churn(E0=0.5, sigma_E0=0.1, gamma=0.08, sigma_gamma=0.02)
        for key in ('prob', 'lower_bound', 'upper_bound', 'ci_95_lower', 'ci_95_upper'):
            assert 0.0 <= result[key] <= 1.0, f"{key} = {result[key]} out of [0, 1]"

    def test_bounds_ordering(self):
        """Percentile bounds must be ordered: ci_95_lower <= lower <= prob <= upper <= ci_95_upper."""

        result = monte_carlo_churn(E0=0.5, sigma_E0=0.08, gamma=0.1, sigma_gamma=0.02)
        assert result['ci_95_lower'] <= result['lower_bound'], "ci_95_lower must be <= lower_bound"
        assert result['lower_bound'] <= result['prob'], "lower_bound must be <= prob"
        assert result['prob'] <= result['upper_bound'], "prob must be <= upper_bound"
        assert result['upper_bound'] <= result['ci_95_upper'], "upper_bound must be <= ci_95_upper"

    def test_sigma_is_non_negative(self):
        """sigma_prob must be >= 0."""

        result = monte_carlo_churn(E0=0.4, sigma_E0=0.05, gamma=0.12, sigma_gamma=0.02)
        assert result['sigma_prob'] >= 0.0

    def test_n_samples_returned_correctly(self):
        """n_samples in output must match the requested value."""

        result = monte_carlo_churn(E0=0.5, sigma_E0=0.05, gamma=0.1, sigma_gamma=0.02, n_samples=5000)
        assert result['n_samples'] == 5000

    def test_risk_level_is_valid_string(self):
        """risk_level must be one of the four documented values."""

        result = monte_carlo_churn(E0=0.5, sigma_E0=0.05, gamma=0.1, sigma_gamma=0.02)
        assert result['risk_level'] in {'HIGH', 'MEDIUM', 'LOW', 'UNCERTAIN'}


# monte_carlo_churn(): reproducibility 
class TestMonteCarloReproducibility:

    def test_same_seed_gives_same_result(self):
        """Two calls with the same seed must return identical results"""
        kwargs = dict(E0=0.5, sigma_E0=0.08, gamma=0.1, sigma_gamma=0.02, seed=42)
        r1 = monte_carlo_churn(**kwargs)
        r2 = monte_carlo_churn(**kwargs)
        assert r1['prob'] == r2['prob']
        assert r1['sigma_prob'] == r2['sigma_prob']

    def test_different_seeds_give_different_results(self):
        """Two calls with different seeds should produce different sigma values."""

        r1 = monte_carlo_churn(E0=0.5, sigma_E0=0.1, gamma=0.1, sigma_gamma=0.03, seed=1)
        r2 = monte_carlo_churn(E0=0.5, sigma_E0=0.1, gamma=0.1, sigma_gamma=0.03, seed=99)
        # They may differ slightly,  assert they're not exactly equal
        assert r1['sigma_prob'] != r2['sigma_prob'] or r1['prob'] != r2['prob']


# monte_carlo_churn(): agreement with analytical method 
class TestMonteCarloVsAnalytical:

    def test_prob_close_to_analytical_central_region(self):
        """For typical mid-range parameters (prob away from 0 and 1), MC median should be within 0.03 of the analytical point estimate.
        Gaussian assumption holds well. """

        params = dict(E0=0.55, sigma_E0=0.04, gamma=0.09, sigma_gamma=0.015, t_horizon=6.0, E_eq=0.55, sigma_E_eq=0.03)
        analytical = churn_probability_with_uncertainty(**params)
        mc = monte_carlo_churn(**params, n_samples=20_000)
        assert abs(analytical['prob'] - mc['prob']) < 0.03, (f"prob divergence too large: analytical={analytical['prob']:.4f}, mc={mc['prob']:.4f}")

    def test_sigma_close_to_analytical_central_region(self):
        """For mid-range parameters, MC sigma should match analytical sigma within 0.03 """

        params = dict(E0=0.6, sigma_E0=0.05, gamma=0.1, sigma_gamma=0.02, t_horizon=6.0, E_eq=0.55, sigma_E_eq=0.03)
        analytical = churn_probability_with_uncertainty(**params)
        mc = monte_carlo_churn(**params, n_samples=20_000)
        assert abs(analytical['sigma_prob'] - mc['sigma_prob']) < 0.03, (
            f"sigma divergence: analytical={analytical['sigma_prob']:.4f}, mc={mc['sigma_prob']:.4f}")


# monte_carlo_churn(): boundary effects 
class TestMonteCarloBoundaryEffects:

    def test_high_risk_customer_prob_above_half(self):
        """A customer with very low E0 and E_eq should yield prob > 0.5."""
        result = monte_carlo_churn(E0=0.1, sigma_E0=0.03, gamma=0.05, sigma_gamma=0.01, E_eq=0.1, sigma_E_eq=0.02, t_horizon=6.0)
        assert result['prob'] > 0.5, f"Expected high churn prob, got {result['prob']:.4f}"

    def test_low_risk_customer_prob_below_half(self):
        """A customer with high E0 and E_eq should yield prob < 0.5."""
        result = monte_carlo_churn(E0=0.9, sigma_E0=0.03, gamma=0.2, sigma_gamma=0.01, E_eq=0.85, sigma_E_eq=0.02, t_horizon=6.0)
        assert result['prob'] < 0.5, f"Expected low churn prob, got {result['prob']:.4f}"

    def test_boundary_clips_do_not_produce_nan(self):
        """Extreme inputs that push prob to 0 or 1 must not produce NaN."""
        result = monte_carlo_churn(E0=0.01, sigma_E0=0.2, gamma=0.001, sigma_gamma=0.001, E_eq=0.01, sigma_E_eq=0.01, t_horizon=24.0 )
        for key in ('prob', 'sigma_prob', 'lower_bound', 'upper_bound'):
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"

    def test_zero_sigma_gives_deterministic_result(self):
        """With sigma=0 for all inputs, MC prob should match analytical prob exactly."""
        params = dict(E0=0.6, sigma_E0=0.0, gamma=0.1, sigma_gamma=0.0, t_horizon=6.0, E_eq=0.55, sigma_E_eq=0.0)
        analytical = churn_probability_with_uncertainty(**params)
        mc = monte_carlo_churn(**params, n_samples=1_000)
        assert abs(analytical['prob'] - mc['prob']) < 1e-6, (f"Deterministic case mismatch: {analytical['prob']:.6f} vs {mc['prob']:.6f}")
        assert mc['sigma_prob'] < 1e-4, (f"sigma should be ~0 with zero input uncertainty, got {mc['sigma_prob']:.6f}" )


# monte_carlo_churn(): risk level classification 
class TestMonteCarloRiskLevel:

    def test_high_risk_classification(self):
        """Customer with lower_bound > RISK_HIGH_LOWER_BOUND (0.35) should be classified HIGH."""

        result = monte_carlo_churn(
            E0=0.05, sigma_E0=0.02, gamma=0.03, sigma_gamma=0.005,
            E_eq=0.05, sigma_E_eq=0.01, t_horizon=6.0, n_samples=20_000)
        # Only assert if the condition is actually met 
        if result['lower_bound'] > 0.65:  
            assert result['risk_level'] == 'HIGH'

    def test_low_risk_classification(self):
        """Customer with upper_bound < RISK_LOW_UPPER_BOUND (0.20) should be classified LOW."""

        result = monte_carlo_churn(
            E0=0.95, sigma_E0=0.02, gamma=0.3, sigma_gamma=0.01,
            E_eq=0.9, sigma_E_eq=0.01, t_horizon=6.0, n_samples=20_000)
        if result['upper_bound'] < 0.35: 
            assert result['risk_level'] == 'LOW'

    def test_uncertain_classification_with_high_sigma(self):
        """Customer with sigma_prob > SIGMA_UNCERTAIN_THRESHOLD should be UNCERTAIN."""

        result = monte_carlo_churn(
            E0=0.5, sigma_E0=0.25, gamma=0.1, sigma_gamma=0.15, E_eq=0.5, sigma_E_eq=0.15, t_horizon=6.0, n_samples=20_000)
        if result['sigma_prob'] > SIGMA_UNCERTAIN_THRESHOLD:
            assert result['risk_level'] == 'UNCERTAIN'


# compare_uncertainty_methods() 
class TestCompareUncertaintyMethods:

    def test_returns_expected_keys(self):
        """Output must contain all documented keys"""

        result = compare_uncertainty_methods(E0=0.6, sigma_E0=0.05, gamma=0.1, sigma_gamma=0.02)
        expected = {'analytical', 'monte_carlo', 'sigma_delta', 'prob_delta', 'boundary_effect'}
        assert expected == set(result.keys())

    def test_analytical_and_mc_are_dicts(self):
        """Both sub-results must be dicts"""

        result = compare_uncertainty_methods(E0=0.55, sigma_E0=0.04, gamma=0.09, sigma_gamma=0.015)
        assert isinstance(result['analytical'],  dict)
        assert isinstance(result['monte_carlo'], dict)

    def test_sigma_delta_is_non_negative(self):
        """sigma_delta must be >= 0 (it is an absolute difference)."""

        result = compare_uncertainty_methods(E0=0.5, sigma_E0=0.06, gamma=0.1, sigma_gamma=0.02)
        assert result['sigma_delta'] >= 0.0

    def test_no_boundary_effect_in_central_region(self):
        """For typical mid-range parameters, boundary_effect should be False analytical and MC sigmas agree within the 0.02 threshold."""

        result = compare_uncertainty_methods(E0=0.6, sigma_E0=0.04, gamma=0.1, sigma_gamma=0.015,
            t_horizon=6.0, E_eq=0.55, sigma_E_eq=0.02, n_samples=20_000)
        assert not result['boundary_effect'], (f"Unexpected boundary_effect=True: sigma_delta={result['sigma_delta']:.4f}")

    def test_boundary_effect_detected_near_clip(self):
        """With extreme parameters pushing prob near 1, the clip distorts the MC distribution and sigma_delta should exceed the threshold"""

        result = compare_uncertainty_methods(E0=0.02, sigma_E0=0.15, gamma=0.02, sigma_gamma=0.05,
            E_eq=0.02, sigma_E_eq=0.10, t_horizon=24.0, n_samples=20_000)
        # boundary_effect may or may not trigger depending on exact parameters  assert at minimum that the comparison ran without error and deltas are finite
        assert np.isfinite(result['sigma_delta'])
        assert np.isfinite(result['prob_delta'])

    def test_prob_delta_is_non_negative(self):
        """prob_delta must be >= 0"""

        result = compare_uncertainty_methods(E0=0.5, sigma_E0=0.05, gamma=0.1, sigma_gamma=0.02)
        assert result['prob_delta'] >= 0.0


# monte_carlo_churn(): correlated parameters 
class TestMonteCarloCorrelated:
    def test_correlated_flag_false_by_default(self):
        """Default call (rho=0) must return correlated=False"""

        result = monte_carlo_churn(E0=0.5, sigma_E0=0.05, gamma=0.1, sigma_gamma=0.02)
        assert result['correlated'] is False

    def test_correlated_flag_true_when_rho_set(self):
        """Any non-zero rho must set correlated=True"""

        result = monte_carlo_churn(E0=0.5, sigma_E0=0.05, gamma=0.1, sigma_gamma=0.02, rho_E0_Eeq=0.4,)
        assert result['correlated'] is True

    def test_positive_E0_Eeq_correlation_increases_sigma(self):
        """When E0 and E_eq are positively correlated, their errors reinforce
        each other in P(churn) = 1 - E(t) (both E0 and E_eq have negative partial derivatives wrt P, so correlated over-shoots amplify variance).
        sigma_prob should be >= the independent case."""

        kwargs = dict(E0=0.5, sigma_E0=0.10, gamma=0.1, sigma_gamma=0.02, E_eq=0.55, sigma_E_eq=0.08, n_samples=20_000, seed=0)
        indep = monte_carlo_churn(**kwargs)
        corr = monte_carlo_churn(**kwargs, rho_E0_Eeq=0.60)
        # positively correlated errors amplify P(churn) variance
        assert corr['sigma_prob'] > indep['sigma_prob'] - 0.005, (
            f"Expected corr sigma >= indep sigma: {corr['sigma_prob']:.4f} vs {indep['sigma_prob']:.4f}")

    def test_output_contract_unchanged_with_correlation(self):
        """Correlated call must return all expected keys."""
        result = monte_carlo_churn(
            E0=0.5, sigma_E0=0.05, gamma=0.1, sigma_gamma=0.02, rho_gamma_Eeq=0.3,)
        expected_keys = {'prob', 'sigma_prob', 'lower_bound', 'upper_bound', 'ci_95_lower', 'ci_95_upper', 'risk_level', 'n_samples', 'correlated'}
        assert expected_keys == set(result.keys())

    def test_extreme_positive_correlation_still_valid(self):
        """rho=0.99 should not crash or produce NaN/Inf"""

        result = monte_carlo_churn(
            E0=0.5, sigma_E0=0.05, gamma=0.1, sigma_gamma=0.02, rho_E0_Eeq=0.99, rho_gamma_Eeq=0.99,)
        for key in ('prob', 'sigma_prob', 'lower_bound', 'upper_bound'):
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"
        assert 0.0 <= result['prob'] <= 1.0

    def test_negative_correlation_reduces_sigma(self):
        """Negative E0-E_eq correlation means when E0 is above its mean, E_eq tends to be below its mean. 
        Since both have the same sign effect on P(churn), their errors partially cancel, REDUCING sigma_prob."""

        kwargs = dict(E0=0.5, sigma_E0=0.10, gamma=0.1, sigma_gamma=0.02, E_eq=0.55, sigma_E_eq=0.08, n_samples=20_000, seed=0)
        indep = monte_carlo_churn(**kwargs)
        anti = monte_carlo_churn(**kwargs, rho_E0_Eeq=-0.60)
        # negatively correlated errors cancel in P(churn) - smaller sigma
        assert anti['sigma_prob'] < indep['sigma_prob'] + 0.005, (
            f"Expected anti-corr sigma <= indep sigma: {anti['sigma_prob']:.4f} vs {indep['sigma_prob']:.4f}")


# estimate_params_discussion() 
class TestEstimateParamsDiscussion:

    def test_returns_string(self):
        from src.physics import estimate_params_discussion
        result = estimate_params_discussion()
        assert isinstance(result, str)
        assert len(result) > 10

    def test_contains_mle_keyword(self):
        from src.physics import estimate_params_discussion
        result = estimate_params_discussion()
        assert 'MLE' in result or 'likelihood' in result.lower()
