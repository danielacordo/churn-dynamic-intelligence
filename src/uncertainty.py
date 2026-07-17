import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

BASE_UNCERTAINTIES = {
    'tenure': 0.5, # +-0.5 months (dataset discretization)
    'MonthlyCharges': 0.5, # +-$0.5 (rounding)
    'TotalCharges': 5.0, # +-$5 (accumulated rounding errors)
    'num_services': 0.0, # exact (discrete count) -- legacy, kept for backward compat
    'num_services_protective': 0.0, # exact (discrete count)
    'num_services_entertainment': 0.0, # exact (discrete count)
    'E0': 0.05, # +-0.05 (energy model uncertainty)
    'gamma': 0.02, # +-0.02 (damping model uncertainty)
    'E_eq': 0.03, # +-0.03 (equilibrium estimate uncertainty)
}

# Threshold above which a prediction is classified as UNCERTAIN.
SIGMA_UNCERTAIN_THRESHOLD = 0.025

# Risk level probability bounds (calibrated on IBM Telco, 7,043 customers)
RISK_HIGH_LOWER_BOUND = 0.35
RISK_LOW_UPPER_BOUND = 0.20


# Numerical partial derivatives 
def numerical_partial(func, params: dict, variable: str, h: float = 1e-5) -> float:
    """ Computes df/dx numerically using central differences"""
    params_plus = {**params, variable: params[variable] + h}
    params_minus = {**params, variable: params[variable] - h}
    return (func(**params_plus) - func(**params_minus)) / (2 * h)


def propagate_errors(func, params: dict, uncertainties: dict[str, float]) -> tuple[float, float]:
    """Propagates uncertainties through a function f(params).
     sigma_f = sqrt( sum( (df/dxi * sigma_xi)^2 ) )"""
    central_value = func(**params)
    sum_of_squares = 0.0

    for var, sigma in uncertainties.items():
        if var in params and sigma > 0:
            df_dxi = numerical_partial(func, params, var)
            sum_of_squares += (df_dxi * sigma) ** 2

    return float(central_value), float(np.sqrt(sum_of_squares))


# Energy uncertainty 
def energy_uncertainty(tenure: float, monthly_charges: float, total_charges: float, num_services_protective: int, num_services_entertainment: int, has_long_contract: bool, internet_type: str = 'No') -> tuple[float, float]:
    """ Computes E0 +- sigma_E for a given customer.
    
    num_services_protective/num_services_entertainment are exact discrete counts (zero uncertainty), 
    so they don't contribute to propagated uncertainty - they're only needed to compute E0's central value. 
    internet_type works the same way: a zero-uncertainty categorical input used only for the central E0 calculation, kept as its own separate sub-score."""
    from src.physics import compute_energy

    params = {
        'tenure': float(tenure),
        'monthly_charges': float(monthly_charges),
        'total_charges': float(total_charges),
        'num_services_protective': float(num_services_protective),
        'num_services_entertainment': float(num_services_entertainment),
        'has_long_contract': has_long_contract,
        'internet_type': internet_type,}
    
    uncertainties = {
        'tenure': BASE_UNCERTAINTIES['tenure'],
        'monthly_charges': BASE_UNCERTAINTIES['MonthlyCharges'],
        'total_charges': BASE_UNCERTAINTIES['TotalCharges'],}
    return propagate_errors(compute_energy, params, uncertainties)


# Churn probability with uncertainty 
def churn_probability_with_uncertainty(E0: float, sigma_E0: float, gamma: float, sigma_gamma: float,
    t_horizon: float = 6.0, E_eq: float = 0.55, sigma_E_eq: float = 0.03,) -> dict[str, float]:
    """ Estimates churn probability over a time horizon, with a formally propagated uncertainty interval from E0, gamma, and E_eq.

    Model:
        E(t) = E_eq + (E0 - E_eq) * exp(-gamma * t)
        P(churn) = 1 - E(t_horizon) """
    
    t = t_horizon
    exp_gt = np.exp(-gamma * t)

    # Central energy and probability
    E_t  = E_eq + (E0 - E_eq) * exp_gt
    prob = float(np.clip(1 - E_t, 0, 1))

    # Analytical partial derivatives
    dP_dE0 = -exp_gt
    dP_dgamma = (E0 - E_eq) * t * exp_gt
    dP_dE_eq = exp_gt - 1.0

    sigma_prob = float(np.clip(np.sqrt(
        (dP_dE0 * sigma_E0) ** 2 +
        (dP_dgamma * sigma_gamma) ** 2 +
        (dP_dE_eq * sigma_E_eq) ** 2), 0, 0.5))

    lower_bound = float(np.clip(prob - sigma_prob, 0, 1))
    upper_bound = float(np.clip(prob + sigma_prob, 0, 1))

    # Risk level (considers the full interval)
    if lower_bound > RISK_HIGH_LOWER_BOUND:
        risk_level = 'HIGH'
    elif upper_bound < RISK_LOW_UPPER_BOUND:
        risk_level = 'LOW'
    elif sigma_prob > SIGMA_UNCERTAIN_THRESHOLD:
        risk_level = 'UNCERTAIN'
    else:
        risk_level = 'MEDIUM'

    return {
        'prob': prob,
        'sigma_prob': sigma_prob,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'risk_level': risk_level,}


# Legacy aliases
probabilidad_churn_con_incertidumbre = churn_probability_with_uncertainty
derivada_parcial_numerica = numerical_partial
propagar_errores = propagate_errors
incertidumbre_energia = energy_uncertainty


# Vectorized DataFrame scoring 
def score_dataframe_with_uncertainty(df: pd.DataFrame, t_horizon: float = 6.0) -> pd.DataFrame:
    """ Applies error propagation to the full DataFrame.

    Uses vectorized numpy operations where possible; applies row-wise only for sigma_E0  """
    df = df.copy()
    t = t_horizon

    # sigma_E0: propagated from original variables (row-wise, unavoidable)
    df['sigma_E0'] = df.apply(
        lambda r: energy_uncertainty(
            r['tenure'], r['MonthlyCharges'], r['TotalCharges'],
            r.get('num_services_protective', 0), r.get('num_services_entertainment', 0),
            r.get('long_contract', r.get('contrato_largo', False)),
            r.get('InternetService', 'No'))[1], axis=1,)

    # sigma_gamma and sigma_E_eq: uniform base uncertainties
    df['sigma_gamma'] = BASE_UNCERTAINTIES['gamma']
    df['sigma_E_eq'] = BASE_UNCERTAINTIES['E_eq']

    # Vectorized probability computation 
    E0 = df['E0'].values
    gamma = df['gamma'].values
    E_eq = df['E_eq'].values
    s_E0 = df['sigma_E0'].values
    s_g = df['sigma_gamma'].values
    s_eq = df['sigma_E_eq'].values

    exp_gt = np.exp(-gamma * t)
    E_t = E_eq + (E0 - E_eq) * exp_gt
    prob = np.clip(1 - E_t, 0, 1)

    # Analytical partial derivatives (vectorized)
    dP_dE0 = -exp_gt
    dP_dgamma = (E0 - E_eq) * t * exp_gt
    dP_dE_eq = exp_gt - 1.0

    sigma_prob = np.clip(np.sqrt((dP_dE0 * s_E0)**2 + (dP_dgamma * s_g)**2 + (dP_dE_eq * s_eq)**2), 0, 0.5)
    lower_bound = np.clip(prob - sigma_prob, 0, 1)
    upper_bound = np.clip(prob + sigma_prob, 0, 1)

    df['prob_churn'] = prob
    df['sigma_prob'] = sigma_prob
    df['prob_inf'] = lower_bound
    df['prob_sup'] = upper_bound

    # Risk level (vectorized via np.select)
    risk_level = np.select(
        [lower_bound > RISK_HIGH_LOWER_BOUND,
         upper_bound < RISK_LOW_UPPER_BOUND,
         sigma_prob  > SIGMA_UNCERTAIN_THRESHOLD],
        ['HIGH', 'LOW', 'UNCERTAIN'],
        default='MEDIUM',)
    df['risk_level'] = risk_level

    return df

# Legacy alias
calcular_incertidumbre_dataframe = score_dataframe_with_uncertainty


# Probability calibration
# BUG FOUND IN REVIEW:The raw physics-based churn probability was badly overconfident (mean ~0.50 vs. an actual churn rate of ~0.265), 
# pushing ~85% of customers into HIGH risk and dropping precision to ~28%. The formula is a good ranking signal, just poorly calibrated in absolute terms. 
# Isotonic regression fixes that average level while preserving rank order, applied in-sample - 
# consistent with how the rest of the pipeline (E0/E_eq/gamma) is already calibrated on the full population rather than a held-out set.
def fit_probability_calibrator(raw_prob: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """ Fits an isotonic map from raw physics-based prob_churn to a calibrated probability
    whose average matches the observed churn rate, preserving the model's rank ordering."""
    calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    calibrator.fit(np.asarray(raw_prob, dtype=float), np.asarray(y_true, dtype=float))
    return calibrator


def apply_probability_calibrator(calibrator: IsotonicRegression, raw_prob: np.ndarray) -> np.ndarray:
    """ Applies a fitted calibrator to (possibly out-of-sample) raw probabilities."""
    return np.clip(calibrator.predict(np.asarray(raw_prob, dtype=float)), 0.0, 1.0)


def calibrate_dataframe_probabilities(df: pd.DataFrame, prob_col: str = 'prob_churn', lower_col: str = 'prob_inf',
    upper_col: str = 'prob_sup', target_col: str = 'Churn_bin',) -> tuple[pd.DataFrame, IsotonicRegression]:
    """ Calibrates prob_churn (and its uncertainty bounds) against actual churn labels.

    Adds 'prob_churn_calibrated', 'prob_inf_calibrated', 'prob_sup_calibrated' columns, and
    recomputes 'risk_level' from the calibrated bounds so downstream risk segmentation reflects real-world churn rates instead of the raw model's overconfidence.

    Note: applying the same monotone map to the lower/upper uncertainty bounds preserves their
    ordering (calibrated_lower <= calibrated_prob <= calibrated_upper) but is an approximation - it does not re-derive the interval width from first principles.
    Adequate for risk tiering, not a substitute for re-propagating uncertainty through the calibration map analytically."""
    df = df.copy()
    calibrator = fit_probability_calibrator(df[prob_col].values, df[target_col].values)

    df['prob_churn_calibrated'] = apply_probability_calibrator(calibrator, df[prob_col].values)
    df['prob_inf_calibrated'] = apply_probability_calibrator(calibrator, df[lower_col].values)
    df['prob_sup_calibrated'] = apply_probability_calibrator(calibrator, df[upper_col].values)

    df['risk_level'] = np.select(
        [df['prob_inf_calibrated'] > RISK_HIGH_LOWER_BOUND,
         df['prob_sup_calibrated'] < RISK_LOW_UPPER_BOUND,
         df['sigma_prob'] > SIGMA_UNCERTAIN_THRESHOLD],
        ['HIGH', 'LOW', 'UNCERTAIN'],
        default='MEDIUM',)

    return df, calibrator


# Legacy alias
calibrar_probabilidades_dataframe = calibrate_dataframe_probabilities


# Uncertainty report 
def uncertainty_report(df: pd.DataFrame) -> pd.DataFrame:
    """ Generates a summary report of uncertainty by risk level.

    Uses the calibrated probability ('prob_churn_calibrated') when available, since risk_level
    is derived from the calibrated bounds; falls back to the raw 'prob_churn' otherwise."""
    col = 'risk_level'
    prob_col = 'prob_churn_calibrated' if 'prob_churn_calibrated' in df.columns else 'prob_churn'
    return (
        df.groupby(col)
        .agg(
            total =(prob_col, 'count'),
            mean_prob  =(prob_col, 'mean'),
            mean_sigma =('sigma_prob', 'mean'),
            churn_rate =('Churn_bin', 'mean'),).round(4).sort_values('mean_prob', ascending=False))


# Legacy alias
reporte_incertidumbre = uncertainty_report


#─ Monte Carlo simulation 
def monte_carlo_churn(E0: float, sigma_E0: float, gamma: float, sigma_gamma: float, t_horizon: float = 6.0,
    E_eq: float = 0.55, sigma_E_eq: float = 0.03, n_samples: int = 10_000, seed: int = 42, rho_E0_gamma: float = 0.0,
    rho_E0_Eeq: float = 0.0, rho_gamma_Eeq: float = 0.0,) -> dict[str, float]:
    """ Estimates churn probability and uncertainty via Monte Carlo simulation.

    Samples (E0, gamma, E_eq) from a multivariate normal distribution and computes P(churn) = 1 - E(t_horizon) for each sample.

    Unlike analytical error propagation, this method:
    - Makes no linearity assumption on the model
    - Captures asymmetric uncertainty near the [0, 1] clip boundaries
    - Provides the full empirical distribution for inspection
    - Correctly propagates correlated parameter errors (rho != 0 cases)"""
    rng = np.random.default_rng(seed)

    # Build 3x3 correlation matrix
    # Clamp rho values to (-1, 1) to avoid degenerate covariance matrices.
    def _clamp(r: float) -> float:
        return float(np.clip(r, -0.99, 0.99))

    r_EG = _clamp(rho_E0_gamma)
    r_EEq = _clamp(rho_E0_Eeq)
    r_GEq = _clamp(rho_gamma_Eeq)

    corr = np.array([
        [1.0, r_EG, r_EEq],
        [r_EG,  1.0, r_GEq],
        [r_EEq, r_GEq, 1.0],])

    # Scale by individual sigmas -> covariance matrix
    sigmas = np.array([sigma_E0, sigma_gamma, sigma_E_eq])
    cov = np.outer(sigmas, sigmas) * corr

    # Ensure positive-definiteness (floating-point issues for extreme rho values)
    min_eig = np.linalg.eigvalsh(cov).min()
    if min_eig <= 0:
        cov += np.eye(3) * (abs(min_eig) + 1e-10)

    # Draw correlated samples
    mean = np.array([E0, gamma, E_eq])
    samples = rng.multivariate_normal(mean, cov, size=n_samples)

    E0_samples = np.clip(samples[:, 0], 0.0, 1.0)
    gamma_samples = np.clip(samples[:, 1], 1e-6, None)
    E_eq_samples = np.clip(samples[:, 2], 0.0, 1.0)

    # Compute P(churn) for each sample
    exp_gt  = np.exp(-gamma_samples * t_horizon)
    E_t = E_eq_samples + (E0_samples - E_eq_samples) * exp_gt
    p_churn = np.clip(1.0 - E_t, 0.0, 1.0)

    prob = float(np.median(p_churn))
    sigma_prob = float(np.std(p_churn))
    lower_bound = float(np.percentile(p_churn, 16))
    upper_bound = float(np.percentile(p_churn, 84))
    ci_95_lower = float(np.percentile(p_churn, 2.5))
    ci_95_upper = float(np.percentile(p_churn, 97.5))

    if lower_bound > RISK_HIGH_LOWER_BOUND:
        risk_level = 'HIGH'
    elif upper_bound < RISK_LOW_UPPER_BOUND:
        risk_level = 'LOW'
    elif sigma_prob > SIGMA_UNCERTAIN_THRESHOLD:
        risk_level = 'UNCERTAIN'
    else:
        risk_level = 'MEDIUM'

    correlated = any(abs(r) > 1e-9 for r in [rho_E0_gamma, rho_E0_Eeq, rho_gamma_Eeq])

    return {
        'prob': prob,
        'sigma_prob': sigma_prob,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'ci_95_lower': ci_95_lower,
        'ci_95_upper': ci_95_upper,
        'risk_level': risk_level,
        'n_samples': n_samples,
        'correlated': correlated,}


def compare_uncertainty_methods(E0: float, sigma_E0: float, gamma: float, sigma_gamma: float, t_horizon: float = 6.0,
    E_eq: float = 0.55,sigma_E_eq: float = 0.03,n_samples: int = 10_000,) -> dict[str, object]:
    """ Runs both analytical and Monte Carlo methods and compares their results"""
    analytical = churn_probability_with_uncertainty(
        E0=E0, sigma_E0=sigma_E0,
        gamma=gamma, sigma_gamma=sigma_gamma,
        t_horizon=t_horizon,
        E_eq=E_eq, sigma_E_eq=sigma_E_eq,)
    
    mc = monte_carlo_churn(
        E0=E0, sigma_E0=sigma_E0,
        gamma=gamma, sigma_gamma=sigma_gamma,
        t_horizon=t_horizon,
        E_eq=E_eq, sigma_E_eq=sigma_E_eq,
        n_samples=n_samples,)

    sigma_delta = float(abs(analytical['sigma_prob'] - mc['sigma_prob']))
    prob_delta = float(abs(analytical['prob'] - mc['prob']))
    boundary_effect = sigma_delta > 0.02

    return {
        'analytical': analytical,
        'monte_carlo': mc,
        'sigma_delta': sigma_delta,
        'prob_delta': prob_delta,
        'boundary_effect': boundary_effect,}
