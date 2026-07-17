from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class DashboardContext:
    """All shared state passed to tab render functions"""

    # Data 
    df: pd.DataFrame
    df_f: pd.DataFrame
    using_real_data: bool

    # Slider values 
    clv: float
    ret_cost: float
    horizon: int
    threshold: float

    # Physics / uncertainty constants 
    E_CRITICAL: float
    SIGMA_THRESH: float

    #  Colour helpers 
    RISK_COLORS: dict[str, str]
    RISK_BADGE:  dict[str, str]

    # Business constants 
    SUCCESS_RATES: dict[str, float]
    COST_MULTS: dict[str, float]

    # Shared callables 
    # () -> Dict[str, Any]   -  Plotly layout kwargs for dark theme
    plotly_dark: Callable[[], dict[str, Any]]
    plotly_axis_style: Callable[[], dict[str, Any]]
    plotly_legend_style: Callable[[], dict[str, Any]]

    # (df, clv, ret_cost, horizon, override?) -> Tuple[pd.DataFrame, Dict]
    compute_campaign_metrics: Callable[
        [pd.DataFrame, float, float, int, float | None],
        tuple[pd.DataFrame, dict[str, float]], ]

    # (E0, gamma, E_eq, F_func, t_span, n) -> Tuple[np.ndarray, np.ndarray]
    solve_trajectory: Callable[
        [float, float, float, Callable[[float], float]],
        tuple[np.ndarray, np.ndarray],]

    # (tenure, monthly, total, n_services_protective, n_services_entertainment, long_contract, internet_type) -> float
    compute_E0: Callable[[float, float, float, int, int, bool, str], float]

    # (contract, n_services_protective, n_services_entertainment, auto_pay, internet_type, payment_method=None) -> float
    compute_E_eq: Callable[[str, int, int, bool, str], float]

    # (contract, n_services_protective, n_services_entertainment, auto_pay, internet_type, payment_method=None) -> float
    compute_gamma: Callable[[str, int, int, bool, str], float]

    # (E0, σE0, gamma, σgamma, E_eq, σEeq, t_h) -> Tuple[float,float,float,float,str]
    churn_prob_with_uncertainty: Callable[
        [float, float, float, float, float, float, float],
        tuple[float, float, float, float, str],]

    # (risk, tau, E_eq) -> Tuple[str, str]
    segment_customer: Callable[
        [str, float, float],
        tuple[str, str],]

    # Perturbation functions: (t, ...) -> float 
    no_perturbation: Callable[[float], float]
    step_perturbation: Callable[[float, float, float], float]
    periodic_perturbation: Callable[[float, float, float, float], float]
    combined_perturbation: Callable[[float, float, float, float, float], float]

    # Analysis functions (E0, σE0, gamma, σgamma, t_h, E_eq, σEeq, n_samples) -> Dict[str, Any]
    monte_carlo_churn: Callable[..., dict[str, Any]]

    # (E0, σE0, gamma, σgamma, t_h, E_eq, σEeq, n_samples) -> Dict[str, Any]
    compare_methods: Callable[..., dict[str, Any]]

    # (n_intervened, n_tp, clv_range, sr_range, cost, horizon) -> Dict[str, Any]
    sensitivity_analysis: Callable[..., dict[str, Any]]

    # (y_true, y_pred_proba, clv, cost, sr, horizon, n_thresholds) -> Dict[str, Any]
    optimize_threshold: Callable[..., dict[str, Any]]

    # (y_true, y_pred_proba, n_bins) -> Dict[str, Any]
    calibration_analysis: Callable[..., dict[str, Any]]
