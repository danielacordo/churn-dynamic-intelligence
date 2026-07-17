"""Empirically calibrates the free constants in the E0/E_eq/gamma formulas against observed churn, instead of leaving them hand-picked.

This doesn't fit the ODE dynamics themselves (that needs longitudinal data IBM Telco doesn't have). 
What it does fit: the 33 coefficients that turn raw customer attributes into E0/E_eq/gamma, replacing "chosen to look reasonable by hand" with "chosen to minimize log-loss against actual churn, 
subject to economically-motivated constraints, validated out-of-fold" - a strictly stronger claim, while keeping each parameter's physical meaning unchanged.

Services are split into two groups instead of one homogeneous count: PROTECTIVE (security/support add-ons) and ENTERTAINMENT (streaming). 
Protective services show a strong, near-monotonic dose-response with lower churn, while entertainment shows essentially none - 
averaging them into one count washed out a real signal. Each formula now has independent, jointly-fit coefficients for both, 
with a constraint enforcing protective ≥ entertainment rather than just hoping the data respects that ordering.

InternetService is modeled as its own categorical component, separate from the protective/entertainment split. 
Its churn-rate gap (7.4% No vs. 41.9% Fiber optic) is larger than the one that motivated the services split, and persists even controlling for zero protective services - so it's a separate, real signal, not redundant with it. 
It's deliberately kept out of the service groups since it's a structural prerequisite for having those services at all, not an equivalent discretionary add-on, 
and is instead modeled as a third categorical term in E_eq/gamma and a fourth E0 sub-score.

PaymentMethod was previously collapsed into a single auto/non-auto binary, which captures most of the gap but was checked - not assumed - to still hide a real difference: 
Electronic check churns at 45.3% vs. Mailed check's 19.1%, a gap that survives controlling for Contract. 
It's now modeled as three categories (auto / mailed check / electronic check), each with its own coefficient, subject to an auto ≥ mailed ≥ electronic ordering constraint - the same evidence standard applied to the earlier splits.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Parameter order matters - this list is the single source of truth for packing/unpacking the flat vector scipy.optimize works with.
#
# E0 used to be a straight sum of weighted terms hard-clipped to [0,1] - a design where individually reasonable weights could sum past 1 for anyone with decent tenure, stable billing, and a long contract, 
# which turned out to be 60.7% of customers, all pinned at E0=1.0 with zero discriminative power between them. 
# It's now redesigned as a convex combination: 4 non-negative weights normalized to sum to 1, applied to 4 sub-scores each already in [0,1]. 
# This makes E0 bounded in [0,1] by construction - saturating at exactly 1.0 now requires all four sub-scores to be 1 simultaneously, not just their unnormalized sum exceeding 1.
#
# Services parametrization note: What used to be a single *_services coefficient per formula (driven by a homogeneous service count) is now a pair - *_protective and *_entertainment - per formula. 
# Within E0's service sub-score specifically, the two are combined via their own internal weight rather than adding a 4th top-level weight,
# which keeps E0's "4 weights normalized to sum to 1" structure intact while still letting the service mix be protective-dominated.
PARAM_NAMES = [
    'e0_rw_tenure', 'e0_rw_billing', 'e0_rw_services', 'e0_rw_contract', 'e0_rw_internet',
    'e0_services_w_protective',
    'e0_s_contract_short', 'e0_s_contract_long',
    'e0_s_internet_no', 'e0_s_internet_dsl', 'e0_s_internet_fiber',
    'eeq_mtm', 'eeq_1y', 'eeq_2y', 'eeq_protective_coef', 'eeq_entertainment_coef',
    'eeq_payment_auto', 'eeq_payment_mailed', 'eeq_payment_electronic',
    'eeq_internet_no', 'eeq_internet_dsl', 'eeq_internet_fiber',
    'gamma_mtm', 'gamma_1y', 'gamma_2y', 'gamma_protective_coef', 'gamma_entertainment_coef',
    'gamma_payment_auto', 'gamma_payment_mailed', 'gamma_payment_electronic',
    'gamma_internet_no', 'gamma_internet_dsl', 'gamma_internet_fiber',]

#These are physically-motivated starting defaults - used both as the optimizer's initial point (x0) and as the "before" baseline in evaluate_calibration's honest comparison. 
# Protective coefficients start well above entertainment's, informed by the raw churn-rate evidence, but this isn't hard-coded as a foregone conclusion - 
# the constraints below make the optimizer justify keeping that ordering rather than assuming it.
HAND_PICKED_PARAMS = {
    'e0_rw_tenure': 0.4, 'e0_rw_billing': 0.3, 'e0_rw_services': 0.2, 'e0_rw_contract': 0.1, 'e0_rw_internet': 0.1,
    'e0_services_w_protective': 0.8,
    'e0_s_contract_short': 0.1, 'e0_s_contract_long': 0.13,
    'e0_s_internet_no': 0.15, 'e0_s_internet_dsl': 0.10, 'e0_s_internet_fiber': 0.05,
    'eeq_mtm': 0.17, 'eeq_1y': 0.22, 'eeq_2y': 0.50, 'eeq_protective_coef': 0.05, 'eeq_entertainment_coef': 0.01,
    'eeq_payment_auto': 0.04, 'eeq_payment_mailed': 0.02, 'eeq_payment_electronic': 0.00,
    'eeq_internet_no': 0.15, 'eeq_internet_dsl': 0.08, 'eeq_internet_fiber': 0.0,
    'gamma_mtm': 0.08, 'gamma_1y': 0.18, 'gamma_2y': 0.30, 'gamma_protective_coef': 0.02, 'gamma_entertainment_coef': 0.005,
    'gamma_payment_auto': 0.04, 'gamma_payment_mailed': 0.02, 'gamma_payment_electronic': 0.00,
    'gamma_internet_no': 0.05, 'gamma_internet_dsl': 0.03, 'gamma_internet_fiber': 0.0,}

# Bounds keep every coefficient in a physically sensible range  without forcing them to their hand-picked values - the optimizer is free to move within the box.
PARAM_BOUNDS = {
    # Raw weights just need to be positive - normalization takes care of keeping E0 in range regardless of scale, so bounds here only rule out zero/ negative weights, not cap any one component's share.
    'e0_rw_tenure': (0.01, 5.0), 'e0_rw_billing': (0.01, 5.0), 'e0_rw_services': (0.01, 5.0), 'e0_rw_contract': (0.01, 5.0), 'e0_rw_internet': (0.01, 5.0),
    # Internal split of E0's service sub-score between protective and entertainment. 
    # Bounded away from the edges (not [0,1]) so entertainment always retains at least a token influence and
    # protective can't be driven to exactly 1.0 - keeps the sub-score a genuine mix rather than degenerating into "protective count only", which would just be renaming the coefficient.
    'e0_services_w_protective': (0.5, 0.98),
    'e0_s_contract_short': (0.0, 1.0), 'e0_s_contract_long': (0.0, 1.0),
    'e0_s_internet_no': (0.0, 1.0), 'e0_s_internet_dsl': (0.0, 1.0), 'e0_s_internet_fiber': (0.0, 1.0),
    'eeq_mtm': (0.01, 0.8), 'eeq_1y': (0.01, 0.9), 'eeq_2y': (0.01, 1.0),
    'eeq_protective_coef': (0.0, 0.15), 'eeq_entertainment_coef': (0.0, 0.15),
    'eeq_payment_auto': (0.0, 0.2), 'eeq_payment_mailed': (0.0, 0.2), 'eeq_payment_electronic': (0.0, 0.2),
    'eeq_internet_no': (0.0, 0.3), 'eeq_internet_dsl': (0.0, 0.3), 'eeq_internet_fiber': (0.0, 0.3),
    # NOTE: Gamma's bounds are kept narrower than the other groups because cross-sectional data (one snapshot per customer) can't jointly identify gamma and E_eq - 
    # many pairs explain the same outcome equally well. Left unbounded, the optimizer pushes gamma to its floor for most contract tiers (tau -> 50 months), 
    # which barely improves log-loss but would silently break every downstream tau-based threshold calibrated elsewhere against the dataset's real tau range (~2-10.5 months). 
    # Keeping gamma near its original envelope lets it still improve without invalidating that calibration.
    'gamma_mtm': (0.05, 0.35), 'gamma_1y': (0.05, 0.35), 'gamma_2y': (0.05, 0.35),
    'gamma_protective_coef': (0.0, 0.05), 'gamma_entertainment_coef': (0.0, 0.05),
    'gamma_payment_auto': (0.0, 0.08), 'gamma_payment_mailed': (0.0, 0.08), 'gamma_payment_electronic': (0.0, 0.08),
    'gamma_internet_no': (0.0, 0.1), 'gamma_internet_dsl': (0.0, 0.1), 'gamma_internet_fiber': (0.0, 0.1),}

#These inequality constraints encode domain knowledge the raw log-loss objective can't know on its own -
#  without them, nothing stops the optimizer from picking an ordering that fits this sample better but makes no business sense (e.g. valuing Month-to-month over Two-year contracts). 
# Each constraint was checked for whether it binds at the unconstrained optimum - a non-binding one costs nothing to impose, 
# but it's kept explicit as a guarantee for future re-fits where the optimum might drift, not just a note about this one run.
# A minimum gap of 0.02 (not a bare >=) keeps tiers meaningfully distinguishable, so collapsing two tiers to the same value doesn't count as "satisfying" the ordering.
_MIN_GAP = 0.02


def _constraint_funcs():
    """Returns the list of constraint dicts for scipy.optimize.minimize(method='SLSQP').
    Each function receives the flat theta vector and must return >= 0 at a feasible point."""
    idx = {name: i for i, name in enumerate(PARAM_NAMES)}

    def eeq_mtm_lt_1y(theta):
        return theta[idx['eeq_1y']] - theta[idx['eeq_mtm']] - _MIN_GAP

    def eeq_1y_lt_2y(theta):
        return theta[idx['eeq_2y']] - theta[idx['eeq_1y']] - _MIN_GAP

    def gamma_mtm_lt_1y(theta):
        return theta[idx['gamma_1y']] - theta[idx['gamma_mtm']] - _MIN_GAP

    def gamma_1y_lt_2y(theta):
        return theta[idx['gamma_2y']] - theta[idx['gamma_1y']] - _MIN_GAP

    def eeq_protective_ge_entertainment(theta):
        # Protective services must contribute at least as much equilibrium energy as entertainment services - 
        # the economically-sensible ordering the raw churn rates already support (~15% vs ~30%), enforced rather than left to chance.
        return theta[idx['eeq_protective_coef']] - theta[idx['eeq_entertainment_coef']] - _MIN_GAP

    def gamma_protective_ge_entertainment(theta):
        return theta[idx['gamma_protective_coef']] - theta[idx['gamma_entertainment_coef']] - _MIN_GAP

    def eeq_payment_auto_ge_mailed(theta):
        # Auto-pay -> highest E_eq (most stable); mailed check -> middle; electronic check -> lowest. 
        # Matches the raw churn-rate ordering (15-17% auto < 19.1% mailed < 45.3% electronic), which also holds controlling for Contract (see module docstring).
        return theta[idx['eeq_payment_auto']] - theta[idx['eeq_payment_mailed']] - _MIN_GAP

    def eeq_payment_mailed_ge_electronic(theta):
        return theta[idx['eeq_payment_mailed']] - theta[idx['eeq_payment_electronic']] - _MIN_GAP

    def gamma_payment_auto_ge_mailed(theta):
        return theta[idx['gamma_payment_auto']] - theta[idx['gamma_payment_mailed']] - _MIN_GAP

    def gamma_payment_mailed_ge_electronic(theta):
        return theta[idx['gamma_payment_mailed']] - theta[idx['gamma_payment_electronic']] - _MIN_GAP

    def eeq_internet_no_ge_dsl(theta):
        # No internet at all -> highest E_eq (most stable); DSL -> middle; Fiber -> lowest.
        # Matches the raw churn-rate ordering (7.4% No < 19.0% DSL < 41.9% Fiber), which also
        # holds controlling for zero protective services (see module docstring) - 
        # this isn't redundant with the protective/entertainment split, it's an independent connection-type signal.
        return theta[idx['eeq_internet_no']] - theta[idx['eeq_internet_dsl']] - _MIN_GAP

    def eeq_internet_dsl_ge_fiber(theta):
        return theta[idx['eeq_internet_dsl']] - theta[idx['eeq_internet_fiber']] - _MIN_GAP

    def gamma_internet_no_ge_dsl(theta):
        return theta[idx['gamma_internet_no']] - theta[idx['gamma_internet_dsl']] - _MIN_GAP

    def gamma_internet_dsl_ge_fiber(theta):
        return theta[idx['gamma_internet_dsl']] - theta[idx['gamma_internet_fiber']] - _MIN_GAP

    return [
        {'type': 'ineq', 'fun': eeq_mtm_lt_1y},
        {'type': 'ineq', 'fun': eeq_1y_lt_2y},
        {'type': 'ineq', 'fun': gamma_mtm_lt_1y},
        {'type': 'ineq', 'fun': gamma_1y_lt_2y},
        {'type': 'ineq', 'fun': eeq_protective_ge_entertainment},
        {'type': 'ineq', 'fun': gamma_protective_ge_entertainment},
        {'type': 'ineq', 'fun': eeq_payment_auto_ge_mailed},
        {'type': 'ineq', 'fun': eeq_payment_mailed_ge_electronic},
        {'type': 'ineq', 'fun': gamma_payment_auto_ge_mailed},
        {'type': 'ineq', 'fun': gamma_payment_mailed_ge_electronic},
        {'type': 'ineq', 'fun': eeq_internet_no_ge_dsl},
        {'type': 'ineq', 'fun': eeq_internet_dsl_ge_fiber},
        {'type': 'ineq', 'fun': gamma_internet_no_ge_dsl},
        {'type': 'ineq', 'fun': gamma_internet_dsl_ge_fiber},]


def _pack(params: dict) -> np.ndarray:
    return np.array([params[name] for name in PARAM_NAMES], dtype=float)


def _unpack(theta: np.ndarray) -> dict:
    return dict(zip(PARAM_NAMES, theta))


@dataclass
class RawInputs:
    """Pre-extracted arrays needed to recompute E0/E_eq/gamma for arbitrary parameters, so the
    optimizer's inner loop doesn't repeat string/categorical work on every call"""
    is_mtm: np.ndarray
    is_1y: np.ndarray
    is_2y: np.ndarray
    is_internet_no: np.ndarray
    is_internet_dsl: np.ndarray
    is_internet_fiber: np.ndarray
    num_services_protective_clipped: np.ndarray
    num_services_entertainment_clipped: np.ndarray
    is_auto_payment: np.ndarray
    is_mailed_check: np.ndarray
    is_electronic_check: np.ndarray
    long_contract: np.ndarray
    e_tenure: np.ndarray
    e_billing: np.ndarray
    e_services_protective: np.ndarray
    e_services_entertainment: np.ndarray

    @classmethod
    def from_df(cls, df: pd.DataFrame, tau_reference: float = 12.0) -> 'RawInputs':
        # NOTE (bug fix, kept from the pre-split version): this used to assume services/payment
        # columns were already present and fell back to all-zeros via df.get(..., default) when
        # they weren't - which is exactly the case when fitting directly on telco_clean.csv (pre-build_physical_features). 
        # That silently zeroed out the services signal and every services/payment coefficient, so the optimizer had zero gradient on them. 
        # Compute them from the raw columns here, identical to features.py::build_physical_features.
        from src.features import _NO_SERVICE_VALUES, _SERVICE_COLS_ENTERTAINMENT, _SERVICE_COLS_PROTECTIVE

        if 'num_services_protective' in df.columns:
            num_protective = df['num_services_protective']
        else:
            flags = pd.DataFrame(index=df.index)
            for col in _SERVICE_COLS_PROTECTIVE:
                if col in df.columns:
                    flags[col] = ~df[col].isin(_NO_SERVICE_VALUES)
            num_protective = flags.sum(axis=1) if len(flags.columns) else pd.Series(0, index=df.index)

        if 'num_services_entertainment' in df.columns:
            num_entertainment = df['num_services_entertainment']
        else:
            flags = pd.DataFrame(index=df.index)
            for col in _SERVICE_COLS_ENTERTAINMENT:
                if col in df.columns:
                    flags[col] = ~df[col].isin(_NO_SERVICE_VALUES)
            num_entertainment = flags.sum(axis=1) if len(flags.columns) else pd.Series(0, index=df.index)

        # PaymentMethod: modeled as three categories (auto / mailed check / electronic check) rather than one auto_payment binary. 
        # See module docstring for the churn-rate evidence: electronic check (45.3%) is far worse than mailed check (19.1%), which is itself close
        # to the two automatic methods (15-17%) - a gap the binary was averaging away, the same mistake made with services and InternetService before their splits.
        if 'PaymentMethod' in df.columns:
            payment = df['PaymentMethod']
            is_auto_payment = payment.isin(('Bank transfer (automatic)', 'Credit card (automatic)'))
            is_mailed_check = payment == 'Mailed check'
            is_electronic_check = payment == 'Electronic check'
        elif 'auto_payment' in df.columns:
            # Backward-compat fallback for callers that only have the old binary column (no raw
            # PaymentMethod to recover the 3-way split from) - treat non-auto as mailed check, since that's the more common and less extreme of the two non-auto categories. 
            # This path undercounts electronic check's effect and should only be hit by legacy callers.
            is_auto_payment = df['auto_payment']
            is_mailed_check = ~df['auto_payment']
            is_electronic_check = pd.Series(False, index=df.index)
        else:
            is_auto_payment = pd.Series(False, index=df.index)
            is_mailed_check = pd.Series(False, index=df.index)
            is_electronic_check = pd.Series(False, index=df.index)

        contract = df['Contract']
        internet = df['InternetService'] if 'InternetService' in df.columns else pd.Series('No', index=df.index)
        tenure = df['tenure']
        monthly = df['MonthlyCharges'] if 'MonthlyCharges' in df.columns else df.get('monthly')
        total = df['TotalCharges'] if 'TotalCharges' in df.columns else tenure * monthly

        hist_avg = total / tenure.clip(lower=1)
        relative_load = monthly / (hist_avg + 1e-9)

        return cls(
            is_mtm=(contract == 'Month-to-month').values.astype(float),
            is_1y=(contract == 'One year').values.astype(float),
            is_2y=(contract == 'Two year').values.astype(float),
            is_internet_no=(internet == 'No').values.astype(float),
            is_internet_dsl=(internet == 'DSL').values.astype(float),
            is_internet_fiber=(internet == 'Fiber optic').values.astype(float),
            num_services_protective_clipped=num_protective.clip(upper=4).values.astype(float),
            num_services_entertainment_clipped=num_entertainment.clip(upper=2).values.astype(float),
            is_auto_payment=is_auto_payment.values.astype(float),
            is_mailed_check=is_mailed_check.values.astype(float),
            is_electronic_check=is_electronic_check.values.astype(float),
            long_contract=contract.isin(('One year', 'Two year')).values.astype(float),
            e_tenure=np.tanh(tenure.values / tau_reference),
            e_billing=np.exp(-0.1 * (relative_load - 1).clip(lower=0).values),
            e_services_protective=(num_protective.values / 4.0).clip(max=1.0),
            e_services_entertainment=(num_entertainment.values / 2.0).clip(max=1.0),)


def compute_prob_churn_from_params(raw: RawInputs, params: dict, t_horizon: float = 6.0) -> np.ndarray:
    """Recomputes E0, E_eq, gamma and prob_churn = 1 - E(t_horizon) for an arbitrary parameter set, without touching a DataFrame - 
    this is the function the optimizer calls repeatedly."""
    # E0 is a convex combination of 5 sub-scores (tenure, billing, services, contract, internet type), each already in [0,1] with weights normalized to sum to 1 - 
    # bounded by construction, replacing the old sum-then-clip formula. 
    # The "services" sub-score is itself a convex combination of protective vs. entertainment (not a single homogeneous count), 
    # and internet type is kept as its own separate sub-score rather than folded into services, since it's a structural prerequisite for having those services, not an equivalent discretionary add-on -
    #  and its Fiber-vs-DSL gap holds even controlling for protective service count.
    raw_w = np.array([params['e0_rw_tenure'], params['e0_rw_billing'], params['e0_rw_services'], params['e0_rw_contract'], params['e0_rw_internet']])
    raw_w = np.clip(raw_w, 1e-6, None)
    w = raw_w / raw_w.sum()
    w_prot = params['e0_services_w_protective']
    e_services = w_prot * raw.e_services_protective + (1 - w_prot) * raw.e_services_entertainment
    s_contract = np.where(raw.long_contract > 0.5, params['e0_s_contract_long'], params['e0_s_contract_short'])
    s_internet = (
        params['e0_s_internet_no'] * raw.is_internet_no
        + params['e0_s_internet_dsl'] * raw.is_internet_dsl
        + params['e0_s_internet_fiber'] * raw.is_internet_fiber)
    e0 = np.clip(
        w[0] * raw.e_tenure + w[1] * raw.e_billing + w[2] * e_services + w[3] * s_contract + w[4] * s_internet, 0, 1,)

    eeq_base = (params['eeq_mtm'] * raw.is_mtm + params['eeq_1y'] * raw.is_1y + params['eeq_2y'] * raw.is_2y)
    eeq_internet = (
        params['eeq_internet_no'] * raw.is_internet_no
        + params['eeq_internet_dsl'] * raw.is_internet_dsl
        + params['eeq_internet_fiber'] * raw.is_internet_fiber)
    eeq_payment = (
        params['eeq_payment_auto'] * raw.is_auto_payment
        + params['eeq_payment_mailed'] * raw.is_mailed_check
        + params['eeq_payment_electronic'] * raw.is_electronic_check)
    e_eq = np.clip(
        eeq_base
        + params['eeq_protective_coef'] * raw.num_services_protective_clipped
        + params['eeq_entertainment_coef'] * raw.num_services_entertainment_clipped
        + eeq_payment
        + eeq_internet, 0, 1,)

    gamma_base = (params['gamma_mtm'] * raw.is_mtm + params['gamma_1y'] * raw.is_1y + params['gamma_2y'] * raw.is_2y)
    gamma_internet = (
        params['gamma_internet_no'] * raw.is_internet_no
        + params['gamma_internet_dsl'] * raw.is_internet_dsl
        + params['gamma_internet_fiber'] * raw.is_internet_fiber)
    gamma_payment = (
        params['gamma_payment_auto'] * raw.is_auto_payment
        + params['gamma_payment_mailed'] * raw.is_mailed_check
        + params['gamma_payment_electronic'] * raw.is_electronic_check)
    gamma = np.clip(
        gamma_base
        + params['gamma_protective_coef'] * raw.num_services_protective_clipped
        + params['gamma_entertainment_coef'] * raw.num_services_entertainment_clipped
        + gamma_payment
        + gamma_internet, 0.01, 1,)

    e_t = e_eq + (e0 - e_eq) * np.exp(-gamma * t_horizon)
    return np.clip(1 - e_t, 1e-6, 1 - 1e-6)


def _log_loss_objective(theta: np.ndarray, raw: RawInputs, y: np.ndarray, t_horizon: float, alpha: float, theta_prior: np.ndarray) -> float:
    params = _unpack(theta)
    p = compute_prob_churn_from_params(raw, params, t_horizon)
    loss = log_loss(y, p)
    # An L2 penalty pulls coefficients back toward the hand-picked prior, guarding against edge-of-bounds degeneracies -
    #  especially for the smaller "One year" group (n=1,473 vs 3,875), where an unconstrained fit could technically improve log-loss while breaking the intuitive month-to-month < one-year < two-year ordering. 
    # The inequality constraints already enforce that ordering directly, so this penalty is a secondary safeguard, not the main mechanism. alpha=0 recovers the unregularized (but still constrained) fit.
    penalty = alpha * np.sum((theta - theta_prior) ** 2)
    return loss + penalty


def fit_physics_parameters(df: pd.DataFrame, y, t_horizon: float = 6.0, x0: dict | None = None, alpha: float = 0.0) -> dict:
    """Fits the E0/E_eq/gamma coefficients to minimize log-loss against actual churn (plus an L2 penalty back toward the hand-picked values), 
    subject to economically-motivated ordering constraints - starting from the hand-picked values and using SLSQP, since it supports nonlinear inequality constraints that L-BFGS-B doesn't. 
    Returns a plain dict of fitted parameters, drop-in compatible with compute_prob_churn_from_params."""
    raw = RawInputs.from_df(df)
    y = np.asarray(y, dtype=float)
    x0_dict = x0 or HAND_PICKED_PARAMS
    theta0 = _pack(x0_dict)
    theta_prior = _pack(HAND_PICKED_PARAMS)
    bounds = [PARAM_BOUNDS[name] for name in PARAM_NAMES]

    result = minimize(
        _log_loss_objective, theta0, args=(raw, y, t_horizon, alpha, theta_prior),
        method='SLSQP', bounds=bounds, constraints=_constraint_funcs(),
        options={'maxiter': 500, 'ftol': 1e-9},)

    return _unpack(result.x)


def evaluate_calibration(df: pd.DataFrame, y, t_horizon: float = 6.0, n_splits: int = 5, random_state: int = 42, alpha: float = 0.0) -> pd.DataFrame:
    """Honest out-of-fold comparison: hand-picked params vs. fitted params, both evaluated on held-out folds the fitting never saw (fitting happens fresh inside each training fold, so
    this is not in-sample - the AUC/log-loss reported here is what you'd actually get on new customers, not an inflated same-data-fit number)."""
    y = np.asarray(y, dtype=int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(df, y), start=1):
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        raw_test = RawInputs.from_df(df_test)

        p_hand = compute_prob_churn_from_params(raw_test, HAND_PICKED_PARAMS, t_horizon)
        fitted = fit_physics_parameters(df_train, y_train, t_horizon, alpha=alpha)
        p_fitted = compute_prob_churn_from_params(raw_test, fitted, t_horizon)

        rows.append({
            'fold': fold,
            'auc_hand_picked': roc_auc_score(y_test, p_hand),
            'auc_fitted': roc_auc_score(y_test, p_fitted),
            'logloss_hand_picked': log_loss(y_test, p_hand),
            'logloss_fitted': log_loss(y_test, p_fitted),})

    return pd.DataFrame(rows)


def check_constraints_binding(theta: np.ndarray, tol: float = 1e-6) -> pd.DataFrame:
    """Reports, per constraint, whether it is binding (active, g(theta) ~= 0) or slack
    (g(theta) > 0, i.e. the unconstrained optimum would already satisfy it) at a given solution.
    Useful for documenting that a constraint costs nothing (slack) vs actually shapes the fit(binding)"""
    names = ['eeq_mtm_lt_1y', 'eeq_1y_lt_2y', 'gamma_mtm_lt_1y', 'gamma_1y_lt_2y',
             'eeq_protective_ge_entertainment', 'gamma_protective_ge_entertainment',
             'eeq_payment_auto_ge_mailed', 'eeq_payment_mailed_ge_electronic',
             'gamma_payment_auto_ge_mailed', 'gamma_payment_mailed_ge_electronic',
             'eeq_internet_no_ge_dsl', 'eeq_internet_dsl_ge_fiber',
             'gamma_internet_no_ge_dsl', 'gamma_internet_dsl_ge_fiber']
    values = [c['fun'](theta) for c in _constraint_funcs()]
    return pd.DataFrame({'constraint': names, 'g_theta': values, 'binding': [abs(v) < tol for v in values],})


def fit_final_params(df: pd.DataFrame, y, t_horizon: float = 6.0, alpha: float = 0.0) -> dict:
    """Fits on the full dataset for deployment, after evaluate_calibration() has already established (via held-out folds) that fitting generalizes. """
    return fit_physics_parameters(df, y, t_horizon, alpha=alpha)
