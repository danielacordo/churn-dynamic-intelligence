import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

E_CRITICAL = 0.25   # Energy threshold for churn zone crossing (E_eq scale, see note below)
E_EQ_BASE = 0.55  # Base equilibrium energy 
TAU_REFERENCE = 12.0   # Reference relaxation time 

#E0 and E_eq aren't on the same scale here - reusing E_CRITICAL (calibrated for E_eq) as the cutoff for E0-based classification left the "Critical" bucket almost empty. 
# These two thresholds are calibrated directly on E0's actual distribution instead, so each band reflects a real churn-rate difference. 
# They were recalibrated (0.8/0.96 → 0.45/0.65) after adding InternetService compressed E0's range and left the old "Stable" cutoff empty.
E0_AT_RISK_THRESHOLD = 0.45  # below this, observed churn rate is ~55%
E0_STABLE_THRESHOLD = 0.65  # above this, observed churn rate is ~17%

E_CRITICO = E_CRITICAL
TAU_REFERENCIA = TAU_REFERENCE


# Main differential equation 
def dynamic_system(t, E, gamma, E_eq, F_func):
    """Differential equation of the customer system with non-zero equilibrium.
        dE/dt = -gamma * (E - E_eq) + F(t)"""
    dE_dt = -gamma * (E[0] - E_eq) + F_func(t)
    return [dE_dt]


# Legacy alias
sistema_dinamico = dynamic_system


def solve_trajectory(E0, gamma, F_func, t_span, t_eval=None, E_eq=E_EQ_BASE):
    """Solves the customer system trajectory over time"""
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 200)

    sol = solve_ivp(fun=lambda t, E: dynamic_system(t, E, gamma, E_eq, F_func), t_span=t_span, y0=[E0], t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8,)
    
    return sol


# Legacy alias
solve_trajectory_v1 = solve_trajectory


# Perturbation functions 
def no_perturbation(t):
    """No external perturbations, free system"""
    return 0.0


def step_perturbation(t, t0, magnitude):
    """Step perturbation: single negative event at t0. magnitude > 0 -> negative effect (removes energy from the system)"""
    return -magnitude if t >= t0 else 0.0


def periodic_perturbation(t, amplitude, frequency, phase=0):
    """Periodic perturbation: usage cycles, seasonality."""
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def combined_perturbation(t, t_event, mag_event, amp_cycle, freq_cycle):
    """Combined perturbation: single event + periodic component."""
    return (step_perturbation(t, t_event, mag_event) + periodic_perturbation(t, amp_cycle, freq_cycle))


# Legacy aliases
no_perturbation_v1 = no_perturbation
step_perturbation_v1 = step_perturbation
periodic_perturbation_v1 = periodic_perturbation
combined_perturbation_v1 = combined_perturbation


# Customer energy 
def compute_energy(tenure, monthly_charges, total_charges, num_services_protective, num_services_entertainment, has_long_contract, internet_type='No'):
    """Computes the initial energy E0 of a customer from business variables.

    Energy represents the current engagement/adhesion level of the customer.
    Combines tenure, relative billing load, service mix, contract type, and internet connection type.

    Services are split into two groups: protective (security/support add-ons that raise switching cost) and entertainment (easily substituted by other platforms, so they don't). 
    PhoneService/MultipleLines are excluded from both since they're base connectivity prerequisites, not discretionary services. 
    Internet type gets its own sub-score rather than being folded into services, since it's a structural prerequisite with a larger, independent churn-rate gap that holds even controlling for protective service count."""
    # Normalized tenure: saturates at ~1 for tenure >> TAU_REFERENCE
    e_tenure = np.tanh(tenure / TAU_REFERENCE)

    # Relative billing: if monthly charge is much higher than historical average, it reduces energy (customer may feel overcharged)
    historical_avg = total_charges / max(tenure, 1)
    relative_load  = monthly_charges / (historical_avg + 1e-9)
    e_billing = np.exp(-0.1 * max(relative_load - 1, 0))

    # Service mix: protective services raise switching cost; entertainment services barely do (see docstring above). 
    # Each sub-count is normalized by its own max (4 protective, 2 entertainment) before combining, so neither group is structurally capped by the other's scale. 
    # Weights fit via src.physics_calibration - see PARAM_NAMES note there.
    e_services_protective = min(num_services_protective / 4.0, 1.0)
    e_services_entertainment = min(num_services_entertainment / 2.0, 1.0)
    e_services = np.clip(0.8036 * e_services_protective + 0.1964 * e_services_entertainment, 0, 1)

    # Long contract: a genuine [0,1] sub-score per contract length (not an additive bonus).
    e_contract = 0.1322 if has_long_contract else 0.0939

    # Internet connection type: a genuine [0,1] sub-score, same structure as e_contract above - see docstring for the rationale. 
    # Fiber optic converges to exactly 0.0 (fit via src.physics_calibration), i.e. it carries no positive energy signal beyond the baseline.
    e_internet = {'No': 0.1914, 'DSL': 0.1005, 'Fiber optic': 0.0000}.get(internet_type, 0.0000)

    # E0 is a convex combination (weights normalized to sum to 1) of five sub-scores each already in [0,1] 
    energy = (0.5456 * e_tenure +
              0.2258 * e_billing +
              0.0121 * e_services +
              0.0661 * e_contract +
              0.1504 * e_internet)
    return float(np.clip(energy, 0, 1))

# Legacy alias
calcular_energia = compute_energy


# Equilibrium energy 
def compute_E_eq(contract_type, num_services_protective, num_services_entertainment, auto_payment, internet_type='No', payment_method=None):
    """Computes the equilibrium energy E_eq of a customer

    E_eq is the stationary state the system converges to in the absence of perturbations. 
    Customers with E_eq > E_critical are intrinsically stable; those with E_eq < E_critical will tend toward churn even without negative events

    Protective and entertainment services get separate coefficients - protective meaningfully raises the equilibrium, entertainment barely moves it, consistent with observed churn rates. 
    Internet type gets its own three-way coefficient on top of that, and payment method follows the same pattern via an optional payment_method argument; 
    the legacy boolean auto_payment is kept for backward compatibility but falls back to the less extreme 'mailed_check' coefficient, undercounting electronic check's effect - pass payment_method explicitly for the full distinction. 
    These constants must stay in sync with features.py::build_physical_features."""
    base = {'Month-to-month': 0.2809, 'One year': 0.5000, 'Two year': 0.5640,}.get(contract_type, 0.2809)

    bonus_protective = 0.0656 * min(num_services_protective, 4)
    bonus_entertainment = 0.0000 * min(num_services_entertainment, 2)
    payment_coefs = {'auto': 0.2000, 'mailed_check': 0.1800, 'electronic_check': 0.0283}
    if payment_method is not None:
        bonus_payment = payment_coefs.get(payment_method, 0.1800)
    else:
        bonus_payment = payment_coefs['auto'] if auto_payment else payment_coefs['mailed_check']
    bonus_internet = {'No': 0.3000, 'DSL': 0.2198, 'Fiber optic': 0.0000}.get(internet_type, 0.0000)

    return float(np.clip(base + bonus_protective + bonus_entertainment + bonus_payment + bonus_internet, 0, 1))

# Legacy alias
calcular_E_eq = compute_E_eq


# Damping coefficient
def compute_damping(contract_type, num_services_protective, num_services_entertainment, auto_payment, internet_type='No', payment_method=None):
    """Computes the damping coefficient gamma for a customer.

    IMPORTANT: gamma high does NOT imply higher churn risk. 
    It means the system returns faster to the equilibrium E_eq. Churn risk depends on whether E_eq > E_critical, not on gamma alone.

    Protective and entertainment services get separate coefficients - see compute_E_eq's docstring for the rationale. 
    internet_type gets its own three-way coefficient, same pattern.
    payment_method follows the same auto/mailed_check/electronic_check split and fallback behavior as compute_E_eq - see its docstring for details.

    Fit via src.physics_calibration with narrower bounds than E_eq/E0"""
    gamma_base = {'Month-to-month': 0.1206, 'One year': 0.1406, 'Two year': 0.3141,}.get(contract_type, 0.1206)

    gamma_protective = 0.0200 * min(num_services_protective, 4)
    gamma_entertainment = 0.0000 * min(num_services_entertainment, 2)
    gamma_payment_coefs = {'auto': 0.0452, 'mailed_check': 0.0252, 'electronic_check': 0.0052}
    if payment_method is not None:
        gamma_payment = gamma_payment_coefs.get(payment_method, 0.0252)
    else:
        gamma_payment = gamma_payment_coefs['auto'] if auto_payment else gamma_payment_coefs['mailed_check']
    gamma_internet = {'No': 0.1000, 'DSL': 0.0200, 'Fiber optic': 0.0000}.get(internet_type, 0.0000)

    return float(np.clip(gamma_base + gamma_protective + gamma_entertainment + gamma_payment + gamma_internet, 0.01, 1))

# Legacy alias
compute_damping_alias = compute_damping


# Relaxation time 
def relaxation_time(gamma):
    """Relaxation time tau = 1/gamma in months """
    return 1.0 / (gamma + 1e-9)


# Legacy alias
tiempo_de_relajacion = relaxation_time


def classify_resilience(tau):
    """Classifies a customer by their relaxation time tau.

    Note on naming convention: in this model, gamma represents loyalty friction (not energy decay rate). 
    High gamma = high friction = customer is strongly anchored = less likely to churn = HIGH resilience. Since tau = 1/gamma,
    HIGH resilience corresponds to LOW tau (e.g. two-year contract customers).

    tau < 4  -> 'High' resilience
    tau 4-6  -> 'Medium' resilience
    tau > 6  -< 'Low' resilience

    NOTE: recalibrated for the fitted gamma's tau range (~2.5-7.3 months here, down from
    ~4.4-17.0 before InternetService was added to gamma.
    Thresholds chosen against the actual tau/churn-rate relationship  in this dataset (see src/physics_calibration.py), matching the vectorized version in
    src/features.py::build_physical_features. 
    Previously (9, 12); before that (7, 11); before that (4, 7); before that (5, 15) which made 'Low' mathematically unreachable regardless of gamma. """
    if tau < 4:
        return 'High'
    elif tau < 6:
        return 'Medium'
    else:
        return 'Low'


# Legacy alias
clasificar_resiliencia = classify_resilience
clasificar_resilencia = classify_resilience  


# Critical threshold crossing detection
def detect_threshold_crossing(energy_trajectory, t_eval, threshold=E_CRITICAL):
    """ Detects the first crossing of the critical threshold """
    below = energy_trajectory < threshold
    indices  = np.where(np.diff(below.astype(int)) == 1)[0]

    if len(indices) > 0:
        idx = indices[0]
        return {
            'crossing': True,
            't_critical': float(t_eval[idx]),
            'E_at_critical': float(energy_trajectory[idx]),}
    return {'crossing': False, 't_critical': None, 'E_at_critical': None}

# Legacy alias
detectar_transicion = detect_threshold_crossing


# Parameter fitting from data 
def equilibrium_model(t, E0, gamma, E_eq):
    """Analytical solution with non-zero equilibrium: E(t) = E_eq + (E0 - E_eq) * exp(-gamma * t)

    Preferred over exponential_model() for all new analyses"""
    return E_eq + (E0 - E_eq) * np.exp(-gamma * t)


# Legacy alias
modelo_con_equilibrio = equilibrium_model


def fit_parameters(t_obs, E_obs, E0_init=0.8, E_eq_init=0.55):
    """Estimates gamma and E_eq by fitting observed data to the analytical solution.

    Methodological note: fitting uses retention rate per tenure cohort as a proxy for energy. 
    This proxy is a coarse approximation; sigma values should be interpreted with caution"""
    try:
        popt, pcov = curve_fit(
            equilibrium_model,
            t_obs, E_obs,
            p0=[E0_init, 0.1, E_eq_init],
            bounds=([0, 0.001, 0], [1, 1, 1]),
            maxfev=5000,)
        perr = np.sqrt(np.diag(pcov))
        return {
            'E0': popt[0], 'sigma_E0': perr[0],
            'gamma': popt[1], 'sigma_gamma': perr[1],
            'E_eq': popt[2], 'sigma_E_eq':  perr[2],}
    except RuntimeError:
        return None


# Legacy aliases 
adjust_parameters = fit_parameters


def estimate_params_discussion() -> str:
    """ Documents the parameter estimation approach and what remains out of reach without longitudinal data. 
    This is a documentation function.

   - E0, E_eq, and gamma are computed from fixed formulas in features.py, whose coefficients were originally hand-picked, 
   then replaced with values fit via SLSQP to minimize log-loss against actual churn, subject to economically-motivated ordering constraints. 
   Honest 5-fold CV shows out-of-fold AUC rising from 0.699 (hand-picked) to 0.791 after redesigning E0 (fixing a 60.7% saturation issue) 
   and splitting services into protective vs entertainment - the optimizer found protective services meaningfully raise E_eq while entertainment converges to exactly 0.0, matching the observed churn-rate gap. 
   Adding InternetService as its own categorical component pushed AUC to 0.832: its churn-rate gap is larger than the services split's and persists even controlling for protective service count, and Fiber optic's coefficient likewise converges to 0.0.

    This is exactly the "practical path forward" sketched in an earlier version of this
    docstring (learn the coefficients via Bernoulli MLE / scipy.optimize, compare AUC against the heuristic on held-out data) - now implemented
    
    - What's still out of reach: full ODE dynamics fitting
    What calibration here does NOT do: fit the actual time-dynamics of E(t) (i.e. does gamma
    correctly describe how fast *this specific customer's* energy decays over time). 
    That would require:
    1. Longitudinal data: multiple energy observations per customer at different time points.
       Telco provides one snapshot per customer, so gamma and E_eq are only jointly (not
       separately) identifiable - many (gamma, E_eq) pairs explain the same E(t_horizon) for a
       given E0. This is why src/physics_calibration.py keeps gamma's bounds narrower than
       E_eq/E0's: left unconstrained, the optimizer pushes gamma to degenerate values for some
       contract tiers (better log-loss, worse identifiability and worse consistency with everything else in this project that assumes a realistic tau range).
    2. Repeated-measures likelihood: y_it ~ Bernoulli(1 - E(t)) observed at multiple t per customer i, not a single cross-sectional y_i.
    3. Joint estimation of the full ODE trajectory, not just its value at one horizon.

    - Practical path forward
    If/when longitudinal data becomes available (e.g. monthly engagement scores per customer):
    - Refit gamma per customer (or per segment) against the actual trajectory, not just one point
    - Validate that the fitted gamma ordering across contract tiers is stable over time
    - Compare against the current single-horizon calibration to quantify what was lost by only
      having cross-sectional data"""
    return (
        "Parameter estimation: coefficients fit via SLSQP-constrained log-loss minimization "
        "(equivalent to maximizing Bernoulli log-likelihood subject to economically-motivated "
        "inequality constraints) against observed churn (src/physics_calibration.py), 5-fold CV "
        "validated (AUC 0.699 -> 0.791 -> 0.832). Protective services (security/support/backup) "
        "fit a materially larger E_eq/gamma contribution than entertainment services (streaming), "
        "matching the raw ~15% vs ~30% churn-rate gap between those groups. InternetService "
        "(No/DSL/Fiber optic) fits its own three-way E_eq/gamma contribution on top of that "
        "split, matching an even larger raw churn-rate gap (~7% vs ~19% vs ~42%) that persists "
        "even holding protective service count fixed. Full ODE dynamics fitting still requires "
        "longitudinal data this dataset doesn't have; "
        "see estimate_params_discussion.__doc__ for what that would take.")


# Backward compatibility (simplified model, E_eq=0) 
def exponential_model(t, E0, gamma):
    """Simplified model with E_eq=0. Kept for notebook 03 compatibility"""
    return E0 * np.exp(-gamma * t)


def fit_gamma(t_obs, E_obs, E0_init=0.8):
    """Simplified fitting (E_eq=0). Kept for notebook 03 compatibility"""
    try:
        popt, pcov = curve_fit(
            exponential_model, t_obs, E_obs,
            p0=[E0_init, 0.1], bounds=([0, 0], [1, 1]), maxfev=5000,)
        perr = np.sqrt(np.diag(pcov))
        return {'E0': popt[0], 'gamma': popt[1],
                'sigma_E0': perr[0], 'sigma_gamma': perr[1]}
    except RuntimeError:
        return None


modelo_exponencial = exponential_model
ajustar_gamma = fit_gamma

periodic_perturbation_v1 = periodic_perturbation
