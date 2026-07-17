import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest

from src.physics import (
    E_CRITICAL, classify_resilience, compute_damping,
    compute_E_eq, compute_energy,
    detect_threshold_crossing, equilibrium_model,
    relaxation_time, solve_trajectory,)
from src.uncertainty import churn_probability_with_uncertainty

# compute_energy() 
class TestComputeEnergy:

    def test_zero_tenure_does_not_explode(self):
        """tenure=0 must not produce NaN or raise an error"""

        e = compute_energy(tenure=0, monthly_charges=50, total_charges=0, num_services_protective=2, num_services_entertainment=0, has_long_contract=False)
        assert np.isfinite(e), "Energy must be finite for tenure=0"

    def test_energy_in_range(self):
        """Energy must always be in [0, 1]"""

        cases = [
            (0, 0, 0, 0, 0, False),
            (72, 20, 1440, 4, 2, True),
            (1, 200, 200, 0, 0, False),]
        for args in cases:
            e = compute_energy(*args)
            assert 0.0 <= e <= 1.0, f"Energy out of range for args={args}: {e}"

    def test_high_tenure_gives_higher_energy(self):
        """Higher tenure with equal billing should produce higher energy """

        e_low = compute_energy(1,  50, 50, 2, 1, False)
        e_high = compute_energy(72, 50, 3600, 2, 1, False)
        assert e_high > e_low

    def test_long_contract_increases_energy(self):
        """A long contract must increase energy vs month-to-month"""

        e_monthly = compute_energy(12, 50, 600, 2, 1, False)
        e_two_year = compute_energy(12, 50, 600, 2, 1, True)
        assert e_two_year > e_monthly

    def test_protective_services_increase_energy_more_than_entertainment(self):
        """Protective services (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport)
        must raise E0 more per service than entertainment services (StreamingTV, StreamingMovies) - 
        fit via src.physics_calibration (e0_services_w_protective=0.8036), matching the observed churn-rate evidence (see features.py::_SERVICE_COLS_PROTECTIVE)."""

        e_base = compute_energy(12, 50, 600, 0, 0, False)
        e_one_protective = compute_energy(12, 50, 600, 1, 0, False)
        e_one_entertainment = compute_energy(12, 50, 600, 0, 1, False)
        assert e_one_protective > e_base
        assert (e_one_protective - e_base) > (e_one_entertainment - e_base)

    def test_internet_type_hierarchy_no_gt_dsl_gt_fiber(self):
        """InternetService is its own E0 sub-score, independent of protective/entertainment
        services -- 'No' internet fits the highest sub-score (0.1914), DSL the middle (0.1005),
        and Fiber optic converges to exactly 0.0 (src/physics_calibration.py), matching the raw
        churn-rate ordering (7.4% No < 19.0% DSL < 41.9% Fiber optic) which persists even
        holding protective service count fixed (see features.py::build_physical_features)."""
        e_no = compute_energy(12, 50, 600, 0, 0, False, internet_type='No')
        e_dsl = compute_energy(12, 50, 600, 0, 0, False, internet_type='DSL')
        e_fiber = compute_energy(12, 50, 600, 0, 0, False, internet_type='Fiber optic')
        assert e_no > e_dsl > e_fiber


# compute_damping() 
class TestComputeDamping:

    def test_gamma_in_range(self):
        """gamma must be in (0.01, 1) """
        for contract in ['Month-to-month', 'One year', 'Two year']:
            for n_prot in [0, 2, 4]:
                for n_ent in [0, 1, 2]:
                    for auto in [True, False]:
                        g = compute_damping(contract, n_prot, n_ent, auto)
                        assert 0.01 <= g <= 1.0, f"gamma={g} out of range"

    def test_two_year_contract_gives_higher_gamma(self):
        """Two year contract must have higher gamma than month-to-month"""
        g_monthly = compute_damping('Month-to-month', 2, 1, False)
        g_two_year = compute_damping('Two year', 2, 1, False)
        assert g_two_year > g_monthly

    def test_auto_payment_gamma_close_to_mailed_check_fallback(self):
        """Payment method's effect on damping is small either way: with the 3-way split
        (src/physics_calibration.py), auto (0.0452) and mailed_check (0.0252, the boolean-arg
        fallback for auto_payment=False - see compute_damping's docstring) are close together,
        both well below the ordering constraint's minimum gap that separates them from
        electronic_check (0.0052). The old binary version fit this bonus to exactly 0 for both;
        the 3-way split recovers a small but real ordering instead of erasing it entirely."""
        g_manual = compute_damping('One year', 2, 1, False)
        g_auto = compute_damping('One year', 2, 1, True)
        assert g_auto > g_manual
        assert (g_auto - g_manual) == pytest.approx(0.02, abs=0.005)

    def test_payment_method_three_way_ordering_gamma(self):
        """With payment_method passed explicitly, gamma should follow the fitted
        auto >= mailed_check >= electronic_check ordering (see the
        gamma_payment_auto_ge_mailed / gamma_payment_mailed_ge_electronic constraints in src/physics_calibration.py)."""
        g_auto = compute_damping('One year', 2, 1, False, payment_method='auto')
        g_mailed = compute_damping('One year', 2, 1, False, payment_method='mailed_check')
        g_electronic = compute_damping('One year', 2, 1, False, payment_method='electronic_check')
        assert g_auto >= g_mailed >= g_electronic

    def test_protective_services_increase_gamma_more_than_entertainment(self):
        """Protective services must contribute more to gamma per service than entertainment services - 
        fit via src.physics_calibration under the gamma_protective_ge_entertainment inequality constraint (binding at the optimum, given
        gamma's intentionally narrow bounds; see PARAM_BOUNDS note in physics_calibration.py)."""
        g_base = compute_damping('Month-to-month', 0, 0, False)
        g_one_protective = compute_damping('Month-to-month', 1, 0, False)
        g_one_entertainment = compute_damping('Month-to-month', 0, 1, False)
        assert (g_one_protective - g_base) > (g_one_entertainment - g_base)

    def test_internet_type_hierarchy_no_gt_dsl_gt_fiber_gamma(self):
        """internet_no >= internet_dsl >= internet_fiber in gamma too, same economically-
        motivated ordering enforced in src/physics_calibration.py's constraint functions."""
        g_no = compute_damping('Month-to-month', 0, 0, False, internet_type='No')
        g_dsl = compute_damping('Month-to-month', 0, 0, False, internet_type='DSL')
        g_fiber = compute_damping('Month-to-month', 0, 0, False, internet_type='Fiber optic')
        assert g_no > g_dsl > g_fiber


# compute_E_eq() 
class TestComputeEEq:

    def test_E_eq_in_range(self):
        """E_eq must be in [0, 1] """
        for contract in ['Month-to-month', 'One year', 'Two year']:
            e = compute_E_eq(contract, 4, 2, False)
            assert 0.0 <= e <= 1.0

    def test_auto_payment_increases_E_eq(self):
        """Auto payment shows up as a meaningful bump to E_eq (auto=0.20 vs. mailed_check=0.18), reading primarily as a structural-loyalty signal rather than a recovery-speed one on gamma. 
        Fiber optic is used in this test as E_eq's lowest tier so the comparison isn't washed out by E_eq's [0,1] clip - 
        other combinations (like "No" internet plus a long contract) could already saturate E_eq at 1.0 regardless of auto-payment."""
        e_manual = compute_E_eq('One year', 2, 1, False, internet_type='Fiber optic')
        e_auto = compute_E_eq('One year', 2, 1, True, internet_type='Fiber optic')
        assert e_auto > e_manual

    def test_payment_method_three_way_ordering_E_eq(self):
        """With payment_method passed explicitly, E_eq should follow the fitted auto >= mailed_check >= electronic_check ordering (see the
        eeq_payment_auto_ge_mailed / eeq_payment_mailed_ge_electronic constraints in src/physics_calibration.py) - 
        electronic check is the real, uncollapsed signal the binary auto_payment flag was averaging away (see module docstring for the churn-rate evidence)."""
        e_auto = compute_E_eq('One year', 2, 1, False, internet_type='Fiber optic', payment_method='auto')
        e_mailed = compute_E_eq('One year', 2, 1, False, internet_type='Fiber optic', payment_method='mailed_check')
        e_electronic = compute_E_eq('One year', 2, 1, False, internet_type='Fiber optic', payment_method='electronic_check')
        assert e_auto >= e_mailed >= e_electronic

    def test_two_year_contract_gives_higher_E_eq(self):
        """Two year contract must have higher E_eq."""

        e_monthly = compute_E_eq('Month-to-month', 2, 1, False)
        e_two_year = compute_E_eq('Two year', 2, 1, False)
        assert e_two_year > e_monthly

    def test_hierarchy_maintained(self):
        """ E_eq hierarchy: month-to-month < one year < two year."""

        e_m = compute_E_eq('Month-to-month', 0, 0, False)
        e_2 = compute_E_eq('Two year', 4, 2, True)
        assert e_m < e_2

    def test_protective_services_increase_E_eq_entertainment_does_not(self):
        """Protective services fit a meaningfully positive E_eq coefficient (0.0680); entertainment services' coefficient fits to exactly 0.0 - the optimizer independently
        confirms entertainment carries no equilibrium-energy signal once protective services are modeled separately (src/physics_calibration.py). 
        This matches the raw churn-rate evidence: ~15% churn with protective services present vs ~30% for entertainment-only."""

        e_base = compute_E_eq('One year', 0, 0, False)
        e_one_protective = compute_E_eq('One year', 1, 0, False)
        e_one_entertainment = compute_E_eq('One year', 0, 1, False)
        assert e_one_protective > e_base
        assert e_one_entertainment == e_base

    def test_internet_type_hierarchy_no_gt_dsl_gt_fiber_eeq(self):
        """InternetService fits its own three-way E_eq coefficient: 'No' internet highest
        (0.30), DSL middle (0.2384), Fiber optic converges to exactly 0.0 - 
        the same "converges to zero" pattern seen with entertainment services, and a larger raw
        churn-rate gap (7.4% vs 19.0% vs 41.9%) than the one that motivated the services split in the first place (see src/features.py::build_physical_features)."""

        e_no = compute_E_eq('Month-to-month', 0, 0, False, internet_type='No')
        e_dsl = compute_E_eq('Month-to-month', 0, 0, False, internet_type='DSL')
        e_fiber = compute_E_eq('Month-to-month', 0, 0, False, internet_type='Fiber optic')
        assert e_no > e_dsl > e_fiber


# relaxation_time() and classify_resilience() 
class TestRelaxationTime:

    def test_tau_equals_inverse_gamma(self):
        """tau = 1/gamma within numerical tolerance."""

        gamma = 0.15
        tau = relaxation_time(gamma)
        assert abs(tau - 1.0 / gamma) < 1e-6

    def test_high_gamma_gives_small_tau(self):
        """gamma high -> tau small (high resilience)."""

        tau_high_gamma = relaxation_time(0.50)
        tau_low_gamma = relaxation_time(0.05)
        assert tau_high_gamma < tau_low_gamma

    def test_resilience_classification_correct(self):
        """tau < 4 -> 'High' resilience
        tau 4-6 -> 'Medium'
        tau > 6 -> 'Low' resilience

        Recalibrated for the fitted gamma's tau range (~2.5-7.3 months) after adding InternetService as a third categorical component to gamma - 
        see src/physics_calibration.py and src/features.py::build_physical_features for why this compressed the range from the previous (~4.4-17.0 months, thresholds 9/12)."""

        assert classify_resilience(3.0) == 'High'
        assert classify_resilience(5.0) == 'Medium'
        assert classify_resilience(7.0) == 'Low'


# ODE: analytical vs numerical solution +
class TestODE:

    def test_free_solution_matches_analytical(self):
        """Without perturbations, E(t) must match the analytical solution
            E(t) = E_eq + (E0 - E_eq) * exp(-gamma * t)"""
        
        E0, gamma, E_eq = 0.8, 0.15, 0.55
        t_test = np.array([0, 5, 10, 20, 30])

        sol = solve_trajectory(E0, gamma, lambda t: 0, (0, 30), t_eval=t_test, E_eq=E_eq)
        E_analytical = equilibrium_model(t_test, E0, gamma, E_eq)

        np.testing.assert_allclose(sol.y[0], E_analytical, rtol=1e-4, err_msg="Numerical solution differs from analytical")

    def test_initial_state(self):
        """E(t=0) must equal E0 """

        E0 = 0.72
        sol = solve_trajectory(E0, 0.1, lambda t: 0, (0, 10))
        assert abs(sol.y[0][0] - E0) < 1e-6

    def test_converges_to_E_eq(self):
        """Without perturbations, E(t->inf) must converge to E_eq."""

        E0, gamma, E_eq = 0.9, 0.3, 0.55
        sol = solve_trajectory(E0, gamma, lambda t: 0, (0, 60), E_eq=E_eq)
        assert abs(sol.y[0][-1] - E_eq) < 0.02, \
            f"Did not converge to E_eq={E_eq}: ended at {sol.y[0][-1]:.4f}"

    def test_perturbation_reduces_energy(self):
        """A negative perturbation must reduce energy vs the free case."""

        E0, gamma, E_eq = 0.7, 0.15, 0.55
        t_eval = np.linspace(0, 24, 200)

        sol_free = solve_trajectory(E0, gamma, lambda t: 0, (0, 24), t_eval, E_eq)
        sol_pert = solve_trajectory(E0, gamma, lambda t: -0.3 if t >= 6 else 0, (0, 24), t_eval, E_eq,)
        E_free_post = sol_free.y[0][t_eval > 6].mean()
        E_pert_post = sol_pert.y[0][t_eval > 6].mean()
        assert E_pert_post < E_free_post


# detect_threshold_crossing() 
class TestDetectThresholdCrossing:

    def test_stable_customer_does_not_cross(self):
        """A stable customer (E always > E_critical) must not trigger a crossing."""

        t_eval = np.linspace(0, 30, 200)
        E = np.ones_like(t_eval) * 0.6
        result = detect_threshold_crossing(E, t_eval)
        assert result['crossing'] is False

    def test_at_risk_customer_crosses(self):
        """A customer decaying below the threshold must trigger a crossing."""

        t_eval = np.linspace(0, 30, 200)
        E = np.linspace(0.8, 0.1, 200)
        result = detect_threshold_crossing(E, t_eval)
        assert result['crossing'] is True
        assert result['t_critical'] is not None
        assert E_CRITICAL * 0.5 < result['E_at_critical'] < E_CRITICAL * 1.5

    def test_crossing_time_is_correct(self):
        """t_critical must be in the range where energy crosses the threshold"""
        
        t_eval = np.linspace(0, 30, 1000)
        E = np.linspace(0.6, 0.1, 1000)
        result = detect_threshold_crossing(E, t_eval)
        assert result['crossing'] is True
        assert 18 < result['t_critical'] < 24


# churn_probability_with_uncertainty() 
class TestUncertainty:

    def test_prob_in_range(self):
        """Central probability must be in [0, 1] """
        r = churn_probability_with_uncertainty(0.7, 0.05, 0.15, 0.02, 6.0, 0.55, 0.03)
        assert 0.0 <= r['prob'] <= 1.0

    def test_sigma_positive(self):
        """Uncertainty must be positive"""
        r = churn_probability_with_uncertainty(0.7, 0.05, 0.15, 0.02, 6.0, 0.55, 0.03)
        assert r['sigma_prob'] > 0

    def test_interval_contains_prob(self):
        """The interval [lower, upper] must contain prob"""
        r = churn_probability_with_uncertainty(0.5, 0.05, 0.10, 0.02, 6.0, 0.55, 0.03)
        assert r['lower_bound'] <= r['prob'] <= r['upper_bound']

    def test_high_E0_gives_lower_churn_prob(self):
        """Higher E0 -> lower churn probability """
        r_low = churn_probability_with_uncertainty(0.3, 0.05, 0.10, 0.02, 6.0, 0.40, 0.03)
        r_high = churn_probability_with_uncertainty(0.9, 0.05, 0.10, 0.02, 6.0, 0.70, 0.03)
        assert r_high['prob'] < r_low['prob']

    def test_zero_uncertainty_gives_zero_sigma(self):
        """Zero input uncertainty -> sigma = 0"""
        r = churn_probability_with_uncertainty(0.6, 0.0, 0.15, 0.0, 6.0, 0.55, 0.0)
        assert r['sigma_prob'] < 1e-10

    def test_high_risk_level_for_low_E0(self):
        """Very low E0 and E_eq must produce HIGH risk level."""
        r = churn_probability_with_uncertainty(0.05, 0.02, 0.05, 0.01, 6.0, 0.10, 0.01)
        assert r['risk_level'] == 'HIGH'

    def test_low_risk_level_for_high_E0(self):
        """Very high E0 and E_eq must produce LOW risk level."""
        r = churn_probability_with_uncertainty(0.95, 0.02, 0.30, 0.01, 6.0, 0.90, 0.01)
        assert r['risk_level'] == 'LOW'
