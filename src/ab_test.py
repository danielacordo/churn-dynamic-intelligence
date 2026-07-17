from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_ALPHA = 0.05 # type I error rate (false positive)
DEFAULT_POWER = 0.80 # 1 - type II error rate (true positive rate)
DEFAULT_BASELINE = 0.49   # expected churn rate in high-risk group without intervention
                          # calibrated on IBM Telco dataset: HIGH risk segment (lb>0.35) = 48.6%
DEFAULT_MDE = 0.10 # minimum detectable effect on churn rate (absolute)


# Data structures
@dataclass
class ABTestConfig:
    """Configuration for an A/B test on retention campaign effectiveness """
    baseline_churn_rate: float = DEFAULT_BASELINE
    assumed_success_rate: float = 0.30
    minimum_detectable_effect: float = DEFAULT_MDE
    alpha: float = DEFAULT_ALPHA
    power: float = DEFAULT_POWER
    one_sided: bool  = True
    seed: int = 42

    def treated_churn_rate(self) -> float:
        """Expected churn rate in treatment group under H1"""
        return self.baseline_churn_rate * (1.0 - self.assumed_success_rate)

    def effect_size(self) -> float:
        """Absolute reduction in churn rate under the assumed success rate """
        return self.baseline_churn_rate - self.treated_churn_rate()


@dataclass
class ABTestResult:
    """Results of an A/B test analysis

    All fields are set by analyse_experiment(); use the factory function rather than constructing directly"""
    # Raw counts
    n_control: int = 0
    n_treatment: int = 0
    churned_control: int = 0
    churned_treatment: int = 0

    # Observed rates
    control_churn_rate: float = 0.0
    treatment_churn_rate: float = 0.0
    observed_reduction: float = 0.0     
    observed_success_rate: float = 0.0    
    assumed_success_rate: float = 0.0   # from config, kept alongside the observed rate for reporting

    # Statistical test
    z_statistic: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    alpha: float = DEFAULT_ALPHA

    # Confidence interval on reduction 
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    ci_level: float = 0.95

    # Business translation
    roi_at_observed_rate: float = 0.0
    roi_at_assumed_rate: float = 0.0
    roi_delta: float = 0.0   
    break_even_rate: float = 0.0   

    # Power achieved (post-hoc)
    achieved_power: float = 0.0

    # Verdict
    verdict: str = ""
    conclusion: str = ""


# Sample size calculation 
def required_sample_size(config: ABTestConfig,) -> dict[str, Any]:
    """Calculates the minimum sample size per group to detect the assumed effect

    Uses the standard two-proportion z-test power formula:
    n = (z_alpha + z_beta)^2 * (p1*(1-p1) + p2*(1-p2)) / (p1 - p2)^2"""
    p1 = config.baseline_churn_rate
    p2 = config.treated_churn_rate()
    delta = abs(p1 - p2)

    if delta < 1e-9:
        raise ValueError(
            "baseline_churn_rate and treated_churn_rate are identical - "
            "no detectable effect. Increase assumed_success_rate.")

    z_alpha = stats.norm.ppf(1 - config.alpha if config.one_sided else 1 - config.alpha / 2)
    z_beta = stats.norm.ppf(config.power)

    # Pooled estimate under H0 and H1 variance
    n = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (delta ** 2)
    n_per_group = int(np.ceil(n))

    return {
        'n_per_group': n_per_group,
        'n_total': 2 * n_per_group,
        'baseline_rate': p1,
        'treated_rate': p2,
        'effect_size_abs': round(delta, 4),
        'effect_size_rel': round(delta / p1, 4),
        'alpha': config.alpha,
        'power': config.power,
        'one_sided': config.one_sided,
        'weeks_at_1k_weekly': round(2 * n_per_group / 1000, 1),
        'formula': (
            f"n = (z_α={z_alpha:.3f} + z_β={z_beta:.3f})² "
            f"x (p1·(1-p1) + p2·(1-p2)) / (p1-p2)² "
            f"= {n_per_group} per group"),}


# Simulation
def simulate_experiment(config: ABTestConfig, n_per_group: int | None = None, true_success_rate: float | None = None,) -> pd.DataFrame:
    """ Simulates an A/B test on retention campaign effectiveness

    Each customer in the treatment group has their churn probability reduced by the campaign effect. Churn outcomes are Bernoulli draws."""
    if n_per_group is None:
        n_per_group = required_sample_size(config)['n_per_group']

    true_sr = true_success_rate if true_success_rate is not None else config.assumed_success_rate

    rng = np.random.default_rng(config.seed)

    n_total = 2 * n_per_group
    groups = np.array(['control'] * n_per_group + ['treatment'] * n_per_group)
    is_treated = groups == 'treatment'

    # Baseline churn probability 
    baseline_probs = np.clip(rng.normal(config.baseline_churn_rate, 0.08, n_total), 0.05, 0.98)

    # Treatment reduces churn probability by success_rate * baseline
    treated_probs = baseline_probs.copy()
    treated_probs[is_treated] = baseline_probs[is_treated] * (1.0 - true_sr)

    # Realise churn outcomes
    churned = rng.binomial(1, treated_probs).astype(bool)

    df = pd.DataFrame({
        'customer_id': np.arange(n_total),
        'group': groups,
        'baseline_prob': baseline_probs.round(4),
        'treated_prob': treated_probs.round(4),
        'churned': churned,
        'is_treated': is_treated,
        'true_success_rate': true_sr,})
    return df


# Analysis 
def analyse_experiment(df: pd.DataFrame, config: ABTestConfig, clv_monthly: float = 65.0, retention_cost: float = 25.0, horizon_months: int   = 6,) -> ABTestResult:
    """Analyses A/B test results and returns a structured ABTestResult"""
    control = df[df['group'] == 'control']
    treatment = df[df['group'] == 'treatment']

    n_c = len(control)
    n_t = len(treatment)
    k_c = int(control['churned'].sum())
    k_t = int(treatment['churned'].sum())

    if n_c == 0 or n_t == 0:
        raise ValueError("Both control and treatment groups must be non-empty.")

    p_c = k_c / n_c # control churn rate
    p_t = k_t / n_t  # treatment churn rate

    reduction = p_c - p_t  
    implied_sr = reduction / p_c if p_c > 0 else 0.0

    # Two-proportion z-test 
    # H0: p_c = p_t  (no effect)
    # H1: p_c > p_t   (campaign reduces churn, one-sided)
    p_pool = (k_c + k_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))
    z = (p_c - p_t) / (se_pool + 1e-15)

    if config.one_sided:
        p_value = 1.0 - stats.norm.cdf(z)
    else:
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    significant = bool(p_value < config.alpha)

    # Newcombe CI for difference of proportions 
    ci_level = 1 - config.alpha
    z_ci = stats.norm.ppf(1 - config.alpha / 2)   

    # Wilson score intervals for each proportion
    def _wilson(k, n, z):
        centre = (k + z**2/2) / (n + z**2)
        margin = z * np.sqrt(k*(n-k)/n + z**2/4) / (n + z**2)
        return max(0, centre - margin), min(1, centre + margin)

    l_c, u_c = _wilson(k_c, n_c, z_ci)
    l_t, u_t = _wilson(k_t, n_t, z_ci)

    ci_lower = float((p_c - p_t) - z_ci * np.sqrt(p_c*(1-p_c)/n_c + p_t*(1-p_t)/n_t))
    ci_upper = float((p_c - p_t) + z_ci * np.sqrt(p_c*(1-p_c)/n_c + p_t*(1-p_t)/n_t))

    # Business translation 
    clv_total = clv_monthly * horizon_months

    def _roi(sr, n_flagged, n_tp, cost_per_customer):
        saved = n_tp * sr
        revenue  = saved * clv_total
        cost = n_flagged * cost_per_customer
        return revenue - cost

    # Estimate TP count: treatment group actual churners 
    n_tp_estimate = k_t + int(round(reduction * n_t))   # counterfactual churners
    n_tp_estimate = max(0, n_tp_estimate)

    roi_observed = _roi(implied_sr, n_t, n_tp_estimate, retention_cost)
    roi_assumed = _roi(config.assumed_success_rate, n_t, n_tp_estimate, retention_cost)

    # Break-even: minimum success_rate such that ROI ≥ 0
    # revenue = n_tp * sr * clv_total >= n_t * cost
    # sr >= n_t * cost / (n_tp * clv_total)
    if n_tp_estimate > 0 and clv_total > 0:
        break_even = (n_t * retention_cost) / (n_tp_estimate * clv_total)
        break_even = float(np.clip(break_even, 0, 1))
    else:
        break_even = float('nan')

    # Post-hoc power 
    # Power to detect the observed effect with the actual sample size
    if abs(p_c - p_t) > 1e-9:
        se_h1 = np.sqrt(p_c*(1-p_c)/n_c + p_t*(1-p_t)/n_t)
        z_alpha  = stats.norm.ppf(1 - config.alpha)
        achieved_power = float(1 - stats.norm.cdf(z_alpha - abs(p_c - p_t) / (se_h1 + 1e-15)))
    else:
        achieved_power = 0.0

    # Verdict 
    if significant and implied_sr >= config.assumed_success_rate * 0.80:
        verdict = "CONFIRMED"
        conclusion = (
            f"Campaign effect is statistically significant (p={p_value:.4f} < α={config.alpha}). "
            f"Observed success rate {implied_sr:.1%} is within 20% of assumed {config.assumed_success_rate:.1%}. "
            f"The 30% assumption is validated, ROI estimate is reliable.")
    elif significant and implied_sr < config.assumed_success_rate * 0.80:
        verdict = "REVISE_DOWN"
        conclusion = (
            f"Campaign effect is significant (p={p_value:.4f}) but observed rate "
            f"{implied_sr:.1%} is materially below assumed {config.assumed_success_rate:.1%}. "
            f"Revise ROI estimates downward. New ROI estimate: ${roi_observed:,.0f} vs assumed ${roi_assumed:,.0f}."
        )
    elif not significant and p_value < 0.15:
        verdict = "INCONCLUSIVE"
        conclusion = (
            f"Trend in expected direction but not significant (p={p_value:.4f})."
            f"Increase sample size or run longer."
            f"Current power: {achieved_power:.0%}.")
    else:
        verdict = "NO_EFFECT"
        conclusion = (
            f"No statistically significant campaign effect detected (p={p_value:.4f})."
            f"Either the campaign is ineffective or the sample size is insufficient."
            f"Break-even success rate: {break_even:.1%}.")

    return ABTestResult(
        n_control=n_c, n_treatment=n_t,
        churned_control=k_c, churned_treatment=k_t,
        control_churn_rate=round(p_c, 4),
        treatment_churn_rate=round(p_t, 4),
        observed_reduction=round(reduction, 4),
        observed_success_rate=round(implied_sr, 4),
        assumed_success_rate=config.assumed_success_rate,
        z_statistic=round(float(z), 4),
        p_value=round(float(p_value), 6),
        significant=significant,
        alpha=config.alpha,
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        ci_level=ci_level,
        roi_at_observed_rate=round(roi_observed, 2),
        roi_at_assumed_rate=round(roi_assumed, 2),
        roi_delta=round(roi_observed - roi_assumed, 2),
        break_even_rate=round(break_even, 4) if not np.isnan(break_even) else float('nan'),
        achieved_power=round(achieved_power, 4),
        verdict=verdict,
        conclusion=conclusion,)


# Sensitivity sweep 
def sweep_true_success_rates(config: ABTestConfig, true_rates: list[float] | None = None, n_per_group: int | None = None,) -> pd.DataFrame:
    """ Runs the A/B test simulation across a range of true success rates.

    Answers: "Would the test detect a problem if the true rate is only X%?" """
    if true_rates is None:
        true_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    if n_per_group is None:
        n_per_group = required_sample_size(config)['n_per_group']

    rows = []
    for rate in true_rates:
        sim_config = ABTestConfig(
            baseline_churn_rate=config.baseline_churn_rate,
            assumed_success_rate=config.assumed_success_rate,
            minimum_detectable_effect=config.minimum_detectable_effect,
            alpha=config.alpha,
            power=config.power,
            one_sided=config.one_sided,
            seed=config.seed + int(rate * 1000),)
        df_sim = simulate_experiment(sim_config, n_per_group, true_success_rate=rate)
        result = analyse_experiment(df_sim, sim_config)

        rows.append({
            'true_rate': rate,
            'observed_sr': result.observed_success_rate,
            'p_value': result.p_value,
            'significant': result.significant,
            'verdict': result.verdict,
            'roi_observed': result.roi_at_observed_rate,
            'roi_assumed': result.roi_at_assumed_rate,
            'roi_delta': result.roi_delta,
            'achieved_power': result.achieved_power,
            'control_rate': result.control_churn_rate,
            'treatment_rate': result.treatment_churn_rate,})

    return pd.DataFrame(rows)


# Sequential monitoring 
def sequential_analysis(config: ABTestConfig, n_checkpoints: int = 5, true_success_rate: float | None = None,) -> pd.DataFrame:
    """Simulates sequential (interim) monitoring of the experiment
    This function shows how the p-value and effect estimate evolve as more data accumulates.

    Uses a conservative Bonferroni correction for multiple looks:
    if you plan k checkpoints, use alpha/k at each checkpoint to maintain the overall type I error rate"""
    n_final = required_sample_size(config)['n_per_group']

    # Simulate the full experiment once
    df_full = simulate_experiment(config, n_final, true_success_rate)

    checkpoints = np.linspace(int(n_final / n_checkpoints), n_final, n_checkpoints, dtype=int)

    adjusted_alpha = config.alpha / n_checkpoints # Bonferroni

    rows = []
    for i, n_check in enumerate(checkpoints):
        df_check = pd.concat([
            df_full[df_full['group'] == 'control'].head(n_check),
            df_full[df_full['group'] == 'treatment'].head(n_check),], ignore_index=True)

        result = analyse_experiment(df_check, config)

        rows.append({
            'checkpoint': i + 1,
            'n_per_group_so_far': n_check,
            'pct_enrolled': round(n_check / n_final * 100, 1),
            'p_value': result.p_value,
            'adjusted_alpha': round(adjusted_alpha, 4),
            'significant_at_checkpoint': bool(result.p_value < adjusted_alpha),
            'observed_sr': result.observed_success_rate,
            'control_rate': result.control_churn_rate,
            'treatment_rate': result.treatment_churn_rate,
            'verdict': result.verdict,})

    return pd.DataFrame(rows)


# Human-readable report 
def ab_test_report(result: ABTestResult, sample_size_info: dict[str, Any] | None = None,) -> str:
    """ Returns a human-readable report of A/B test results"""
    lines = [
        "" ,
        "A/B TEST REPORT - Retention Campaign Effectiveness",
        "",
        "",
        " SAMPLE",
        f" Control : {result.n_control:,} customers  "
        f"-> {result.churned_control:,} churned ({result.control_churn_rate:.1%})",
        f" Treatment : {result.n_treatment:,} customers  "
        f"-> {result.churned_treatment:,} churned ({result.treatment_churn_rate:.1%})",
        "",
        " EFFECT",
        f" Churn reduction : {result.observed_reduction:+.1%}  "
        f"({result.control_churn_rate:.1%} -> {result.treatment_churn_rate:.1%})",
        f" Implied success rate : {result.observed_success_rate:.1%}  "
        f"(assumed: {result.assumed_success_rate:.1%} - see α below)",
        "",
        " STATISTICS",
        f" z-statistic : {result.z_statistic:+.3f}",
        f" p-value : {result.p_value:.4f}  "
        f"({'< α -> SIGNIFICANT' if result.significant else '≥ α -> not significant'})",
        f" α (threshold) : {result.alpha:.3f}",
        f" {result.ci_level:.0%} CI (reduction): "
        f"[{result.ci_lower:+.3f}, {result.ci_upper:+.3f}]",
        f" Achieved power: {result.achieved_power:.0%}",
        "",
        " BUSINESS IMPACT",
        f" ROI at observed rate : ${result.roi_at_observed_rate:>+,.0f}",
        f" ROI at assumed 30% : ${result.roi_at_assumed_rate:>+,.0f}",
        f" Delta : ${result.roi_delta:>+,.0f}",
        f" Break-even rate : {result.break_even_rate:.1%}  "
        f"(minimum success rate for ROI > 0)",
        "",
        " VERDICT",
        f" {result.verdict}",
        "",]
    
    # Wrap conclusion
    words = result.conclusion.split()
    line, wrapped = "    ", []
    for w in words:
        if len(line) + len(w) + 1 > 60:
            wrapped.append(line)
            line = "    " + w + " "
        else:
            line += w + " "
    wrapped.append(line)
    lines.extend(wrapped)

    if sample_size_info:
        lines += [
            "",
            " DESIGN (pre-experiment)",
            f" Required n/group : {sample_size_info['n_per_group']:,}",
            f" Detectable effect: {sample_size_info['effect_size_abs']:.1%} absolute",
            f" Power at design : {sample_size_info['power']:.0%}",
            f" Runtime estimate : {sample_size_info['weeks_at_1k_weekly']:.1f} weeks "
            f"(at 1,000 flagged/week)",]

    lines += ["", ""]
    return "\n".join(lines)
