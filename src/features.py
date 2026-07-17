import numpy as np
import pandas as pd
from src.physics import (
    E_CRITICAL,
    E0_AT_RISK_THRESHOLD,
    E0_STABLE_THRESHOLD,
    TAU_REFERENCE,)


_SERVICE_COLS = [
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',]

# Service groups with distinct physical roles, replacing the old homogeneous num_services count.
_SERVICE_COLS_PROTECTIVE = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
_SERVICE_COLS_ENTERTAINMENT = ['StreamingTV', 'StreamingMovies']

_NO_SERVICE_VALUES = frozenset({'No', 'No internet service', 'No phone service'})


# Row-level helpers (used only in tests or single-row scoring) 
def count_services(row) -> int:
    """Counts how many additional services the customer has (single-row version).

    Kept for backward compatibility (e.g. reporting total add-on count); physics scoring uses count_services_protective/count_services_entertainment instead 
    See module docstring above _SERVICE_COLS_PROTECTIVE for why the two groups are no longer combined into one signal."""
    return sum(
        1 for col in _SERVICE_COLS
        if str(row.get(col, 'No')) not in _NO_SERVICE_VALUES)


def count_services_protective(row) -> int:
    """Counts protective/support services (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport) the customer has (single-row version)"""
    return sum(
        1 for col in _SERVICE_COLS_PROTECTIVE
        if str(row.get(col, 'No')) not in _NO_SERVICE_VALUES)


def count_services_entertainment(row) -> int:
    """Counts entertainment services (StreamingTV, StreamingMovies) the customer has (single-row version)"""
    return sum(
        1 for col in _SERVICE_COLS_ENTERTAINMENT
        if str(row.get(col, 'No')) not in _NO_SERVICE_VALUES)


def is_auto_payment(payment_method: str) -> bool:
    """Returns True if the payment method is automatic """
    return payment_method in ('Bank transfer (automatic)', 'Credit card (automatic)')


def is_mailed_check(payment_method: str) -> bool:
    """Returns True if the payment method is a mailed check (single-row version)"""
    return payment_method == 'Mailed check'


def is_electronic_check(payment_method: str) -> bool:
    """Returns True if the payment method is electronic check (single-row version).

    Kept as its own category rather than folded into a generic "not auto" bucket: electronic check churns at 45.3% vs mailed check's 19.1% -
    a ~26pp gap that survives controlling for Contract (see src/physics_calibration.py module docstring for the full evidence and the controlled comparison)"""
    return payment_method == 'Electronic check'


def is_long_contract(contract: str) -> bool:
    """Returns True if the contract is 1 or 2 years"""
    return contract in ('One year', 'Two year')


# Legacy aliases with original Spanish names (kept for backward compat)
count_services_v1 = count_services
is_auto_payment_v1 = is_auto_payment
is_long_contract_v1 = is_long_contract


# Vectorized feature pipeline 
def build_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Builds all physical state variables on the DataFrame"""
    df = df.copy()

    # Service counts (vectorized) - split into protective vs entertainment groups rather than one homogeneous count. 
    # See _SERVICE_COLS_PROTECTIVE's module-level note for the churn-rate evidence behind the split (protective services show a strong, ~monotonic dose-response with
    # lower churn; entertainment services show essentially none). 
    # num_services is kept as the simple total (backward-compatible column, used for reporting/display) but no longer drives any E0/E_eq/gamma formula below.
    service_flags = pd.DataFrame(index=df.index)
    for col in _SERVICE_COLS:
        if col in df.columns:
            service_flags[col] = ~df[col].isin(_NO_SERVICE_VALUES)
    df['num_services'] = service_flags.sum(axis=1)
    df['num_services_protective'] = service_flags[[c for c in _SERVICE_COLS_PROTECTIVE if c in service_flags.columns]].sum(axis=1)
    df['num_services_entertainment'] = service_flags[[c for c in _SERVICE_COLS_ENTERTAINMENT if c in service_flags.columns]].sum(axis=1)

    # Internet connection type flags (vectorized). InternetService is modeled as its own
    # categorical connection-type component - kept separate from the protective/entertainment service groups above, not folded into either. 
    df['is_internet_no'] = df['InternetService'] == 'No'
    df['is_internet_dsl'] = df['InternetService'] == 'DSL'
    df['is_internet_fiber'] = df['InternetService'] == 'Fiber optic'

    # Payment and contract flags (vectorized). 
    # PaymentMethod is modeled as three categories (auto / mailed check / electronic check) rather than one auto_payment binary 
    df['auto_payment'] = df['PaymentMethod'].isin(('Bank transfer (automatic)', 'Credit card (automatic)'))
    df['is_mailed_check'] = df['PaymentMethod'] == 'Mailed check'
    df['is_electronic_check'] = df['PaymentMethod'] == 'Electronic check'
    df['long_contract'] = df['Contract'].isin(('One year', 'Two year'))

    # The E_eq coefficients were fit via SLSQP against actual churn and validated with 5-fold CV, with out-of-fold AUC improving in steps (0.699 → 0.791 → 0.832 → 0.837) 
    # as services, internet type, and payment method were each split into their own components. 
    # Protective services and "no internet" add real signal, while entertainment and fiber optic contribute nothing (converge to 0.0); 
    # electronic check payment sits far below auto/mailed check, confirming its higher churn risk even controlling for contract. 
    # Gamma uses tighter bounds than E_eq/E0 since it's weakly identifiable from this cross-sectional dataset. 
    # See src/physics_calibration.py for full methodology and CV numbers.
    contract_base = df['Contract'].map({'Month-to-month': 0.2809, 'One year': 0.5000, 'Two year': 0.5640,}).fillna(0.2809)
    bonus_protective_eq = 0.0656 * df['num_services_protective'].clip(upper=4)
    bonus_entertainment_eq = 0.0000 * df['num_services_entertainment'].clip(upper=2)
    bonus_payment_eq = (
        0.2000 * df['auto_payment'] + 0.1800 * df['is_mailed_check'] + 0.0283 * df['is_electronic_check'])
    bonus_internet_eq = (
        0.3000 * df['is_internet_no'] + 0.2198 * df['is_internet_dsl'] + 0.0000 * df['is_internet_fiber'])
    df['E_eq'] = np.clip(contract_base + bonus_protective_eq + bonus_entertainment_eq + bonus_payment_eq + bonus_internet_eq, 0, 1)

    # Damping coefficient gamma (vectorized) 
    # Here protective (0.0200) also clearly exceeds entertainment (0.0000, this time an independently-discovered zero), 
    # internet_no (0.10) > internet_dsl (0.02) > internet_fiber (0.0, constraint-boundary)
    gamma_base = df['Contract'].map({'Month-to-month': 0.1206, 'One year': 0.1406, 'Two year': 0.3141,}).fillna(0.1206)
    gamma_protective = 0.0200 * df['num_services_protective'].clip(upper=4)
    gamma_entertainment = 0.0000 * df['num_services_entertainment'].clip(upper=2)
    gamma_payment = (0.0452 * df['auto_payment'] + 0.0252 * df['is_mailed_check'] + 0.0052 * df['is_electronic_check'])
    gamma_internet = (0.1000 * df['is_internet_no'] + 0.0200 * df['is_internet_dsl'] + 0.0000 * df['is_internet_fiber'])
    df['gamma'] = np.clip(gamma_base + gamma_protective + gamma_entertainment + gamma_payment + gamma_internet, 0.01, 1)

    # Relaxation time tau (vectorized) 
    df['tau'] = 1.0 / (df['gamma'] + 1e-9)

    # Resilience classification (vectorized via pd.cut)
    # NOTE: The resilience thresholds were recalibrated because tau's (1/gamma) range compressed from ~4.4-17.0 to ~2.5-7.3 months once InternetService was added as a third component of gamma - 
    # with more variables splitting the variance, it's no longer concentrated in contract alone. 
    # The new numeric cutoffs were chosen against the actual tau/churn-rate relationship in the data. 
    # The convention itself is unchanged: low tau still means high gamma (strongly anchored, HIGH resilience) - only the cut points moved.
    df['resilience'] = pd.cut(df['tau'], bins=[-np.inf, 4, 6, np.inf], labels=['High', 'Medium', 'Low'],).astype(str)

    # Initial energy E0 (vectorized)
    # e_tenure: tanh normalization saturating at TAU_REFERENCE months
    e_tenure = np.tanh(df['tenure'] / TAU_REFERENCE)

    # e_billing: penalize when monthly charge exceeds historical average
    hist_avg = df['TotalCharges'] / df['tenure'].clip(lower=1)
    relative_load = df['MonthlyCharges'] / (hist_avg + 1e-9)
    e_billing = np.exp(-0.1 * (relative_load - 1).clip(lower=0))

    # e_services: service mix sub-score, in [0,1] by construction. 
    # Protective and entertainment counts are each normalized by their own max (4 and 2 respectively), then combined with a
    # fitted internal weight (0.8036 protective, matching e0_services_w_protective in physics_calibration.py) - 
    # switching cost is driven mostly, but not exclusively, by protective services. 
    # See _SERVICE_COLS_PROTECTIVE's module-level note for the churn-rate evidence behind this weighting.
    e_services_protective = (df['num_services_protective'] / 4.0).clip(upper=1.0)
    e_services_entertainment = (df['num_services_entertainment'] / 2.0).clip(upper=1.0)
    e_services = np.clip(0.8036 * e_services_protective + 0.1964 * e_services_entertainment, 0, 1)

    # e_contract: a genuine [0,1] sub-score per contract length (not an additive bonus)
    e_contract = np.where(df['long_contract'], 0.1322, 0.0939)

    # e_internet: a genuine [0,1] sub-score per connection type, structurally the same shape as e_contract above but for InternetService - 
    # see the E_eq/gamma comment block above for the churn-rate evidence and why this is a separate sub-score rather than folded into e_services.
    e_internet = (0.1914 * df['is_internet_no'] + 0.1005 * df['is_internet_dsl'] + 0.0000 * df['is_internet_fiber'])

    # E0 as a convex combination (weights normalized to sum to 1) of five sub-scores, each already in [0,1]. 
    # This replaced a straight sum-then-clip formula that put 60.7% of customers at exactly E0=1.0 (clip saturation) -
    # any customer with decent tenure + stable billing + a long contract already summed past 1.0 before the clip, losing all
    # discriminative power between them. A weighted *average* is in [0,1] by construction, and only saturates at exactly 1.0 if every sub-score is simultaneously 1 -
    # fit against actual churn, this dropped saturation to 0% with no meaningful AUC cost. See
    # src/physics_calibration.py's PARAM_NAMES note for the full explanation.
    e0_w_tenure, e0_w_billing, e0_w_services, e0_w_contract, e0_w_internet = 0.5456, 0.2258, 0.0121, 0.0661, 0.1504
    df['E0'] = np.clip(
        e0_w_tenure * e_tenure +
        e0_w_billing * e_billing +
        e0_w_services * e_services +
        e0_w_contract * e_contract +
        e0_w_internet * e_internet, 0, 1,)

    # Estimated perturbation (vectorized) 
    df['historical_avg_charge'] = hist_avg
    df['estimated_perturbation'] = ((df['MonthlyCharges'] - hist_avg) / (hist_avg + 1e-9)).clip(-1, 2)

    # Physical state (vectorized via pd.cut on E0)
    # NOTE: uses E0-specific thresholds (calibrated on the observed E0 distribution), not E_CRITICAL  
    # E0 and E_eq are on different scales here, and reusing E_CRITICAL (0.25) left "Critical" almost empty (E0 rarely drops below ~0.45 for tenure > 0).
    df['physical_state'] = pd.cut(
        df['E0'], bins=[-np.inf, E0_AT_RISK_THRESHOLD, E0_STABLE_THRESHOLD, np.inf],
        labels=['Critical', 'At risk', 'Stable'],).astype(str)

    # Intrinsic risk (vectorized)
    df['intrinsic_risk'] = np.where(df['E_eq'] < E_CRITICAL, 'Yes', 'No')

    return df


# Legacy alias
build_physical_features_v1 = build_physical_features


# Summary statistics 
def physical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """ Generates a comparative summary of physical variables between churners and non-churners"""

    physical_cols = ['E0', 'E_eq', 'gamma', 'tau', 'num_services', 'num_services_protective', 'num_services_entertainment', 'estimated_perturbation']
    available = [c for c in physical_cols if c in df.columns]
    return (df.groupby('Churn')[available].agg(['mean', 'std', 'median']).round(4))


# Legacy alias
physical_summary_v1 = physical_summary


def state_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of physical states by churn group."""
    return (df.groupby(['Churn', 'physical_state']).size().unstack(fill_value=0).apply(lambda x: x / x.sum() * 100, axis=1).round(2))


def E_eq_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of E_eq by contract type.
    Verifies that customers with longer contracts have higher equilibrium energies"""
    return (
        df.groupby('Contract').agg(E_eq_mean =('E_eq', 'mean'), E_eq_median =('E_eq', 'median'),
            churn_rate  =('Churn_bin', 'mean'),n =('E_eq', 'count'),).round(4))


# Notebook compatibility aliases 
state_distribution_v1 = state_distribution
E_eq_summary_v1 = E_eq_summary
physical_summary_v1 = physical_summary
