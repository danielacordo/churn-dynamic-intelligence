import time
import numpy as np
import pandas as pd
import pytest
from pipeline.schema import validate
from src.ab_test import ABTestConfig, required_sample_size, simulate_experiment
from src.business import apply_segmentation
from src.features import build_physical_features
from src.uncertainty import monte_carlo_churn, score_dataframe_with_uncertainty

# Synthetic dataset factory 
def _make_raw_df(n: int = 7_000, seed: int = 0) -> pd.DataFrame:
    """Generates a synthetic Telco-like raw DataFrame of size n """
    rng = np.random.default_rng(seed)
    contracts = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    payments = rng.choice(
        ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],n, p=[0.22, 0.22, 0.33, 0.23],)
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(20, 110, n)
    total = monthly * np.maximum(tenure, 1) * rng.uniform(0.9, 1.1, n)

    df = pd.DataFrame({
        "customerID": [f"C{i:06d}" for i in range(n)],
        "tenure": tenure.astype(int),
        "MonthlyCharges": monthly.round(2),
        "TotalCharges": total.round(2),
        "Contract": contracts,
        "PaymentMethod": payments,
        "Churn": rng.choice(["Yes", "No"], n, p=[0.265, 0.735]),
        "Churn_bin": (rng.random(n) < 0.265).astype(int),
        # Service columns (minimal, enough for num_services)
        "PhoneService": rng.choice(["Yes", "No"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n), })
    return df


# Performance fixtures 
@pytest.fixture(scope="module")
def raw_7k():
    return _make_raw_df(7_000)


@pytest.fixture(scope="module")
def features_7k(raw_7k):
    return build_physical_features(raw_7k)


@pytest.fixture(scope="module")
def scored_7k(features_7k):
    return score_dataframe_with_uncertainty(features_7k, t_horizon=6.0)


# build_physical_features 
class TestBuildPhysicalFeaturesPerformance:
    THRESHOLD_S = 2.0

    def test_7k_rows_within_threshold(self, raw_7k):
        t0 = time.perf_counter()
        result = build_physical_features(raw_7k)
        elapsed = time.perf_counter() - t0
        print(f"\n  build_physical_features(7k): {elapsed:.3f}s  (limit {self.THRESHOLD_S}s)")
        assert elapsed < self.THRESHOLD_S, (
            f"build_physical_features took {elapsed:.2f}s > {self.THRESHOLD_S}s limit. "
            "Likely cause: apply(axis=1) regression.")
        assert len(result) == 7_000

    def test_scales_linearly_not_quadratically(self, raw_7k):
        """2x rows should take < 3x the time (linear, not quadratic) """
        df_small = raw_7k.iloc[:1_000].copy()
        df_large = raw_7k.iloc[:2_000].copy()

        t0 = time.perf_counter()
        build_physical_features(df_small)
        t_small = time.perf_counter() - t0

        t0 = time.perf_counter()
        build_physical_features(df_large)
        t_large = time.perf_counter() - t0

        ratio = t_large / (t_small + 1e-9)
        print(f"\n  1k→2k scaling ratio: {ratio:.2f}x  (limit 3.0x)")
        assert ratio < 3.0, (f"2x rows took {ratio:.1f}x longer - possible O(n²) regression.")


# score_dataframe_with_uncertainty 
class TestScoringPerformance:
    THRESHOLD_S = 5.0

    def test_7k_rows_within_threshold(self, features_7k):
        t0 = time.perf_counter()
        result = score_dataframe_with_uncertainty(features_7k, t_horizon=6.0)
        elapsed = time.perf_counter() - t0
        print(f"\n  score_dataframe_with_uncertainty(7k): {elapsed:.3f}s  (limit {self.THRESHOLD_S}s)")
        assert elapsed < self.THRESHOLD_S, (f"Scoring took {elapsed:.2f}s > {self.THRESHOLD_S}s limit.")
        assert len(result) == 7_000
        assert "prob_churn" in result.columns


# apply_segmentation 
class TestSegmentationPerformance:
    THRESHOLD_S = 1.0

    def test_7k_rows_within_threshold(self, scored_7k):
        t0 = time.perf_counter()
        result = apply_segmentation(scored_7k)
        elapsed = time.perf_counter() - t0
        print(f"\n apply_segmentation(7k): {elapsed:.3f}s  (limit {self.THRESHOLD_S}s)")
        assert elapsed < self.THRESHOLD_S, (f"Segmentation took {elapsed:.2f}s > {self.THRESHOLD_S}s limit.")
        assert "segment" in result.columns


# monte_carlo_churn 
class TestMonteCarloPerfomance:
    THRESHOLD_S = 1.0

    def test_10k_samples_within_threshold(self):
        t0 = time.perf_counter()
        monte_carlo_churn(E0=0.5, sigma_E0=0.05, gamma=0.1, sigma_gamma=0.02, n_samples=10_000, seed=42,)
        elapsed = time.perf_counter() - t0
        print(f"\n monte_carlo_churn(n=10k): {elapsed:.3f}s (limit {self.THRESHOLD_S}s)")
        assert elapsed < self.THRESHOLD_S

    def test_correlated_not_significantly_slower(self):
        """Correlated MC (multivariate_normal) should be < 2x  independent"""
        kwargs = dict(E0=0.5, sigma_E0=0.05, gamma=0.1, sigma_gamma=0.02, n_samples=10_000, seed=0)

        t0 = time.perf_counter()
        monte_carlo_churn(**kwargs)
        t_indep = time.perf_counter() - t0

        t0 = time.perf_counter()
        monte_carlo_churn(**kwargs, rho_E0_Eeq=0.40, rho_gamma_Eeq=0.30)
        t_corr = time.perf_counter() - t0

        ratio = t_corr / (t_indep + 1e-9)
        print(f"\n correlated/independent time ratio: {ratio:.2f}x (limit 2.0x)")
        assert ratio < 2.0, (
            f"Correlated MC is {ratio:.1f}x slower than independent - "
            "multivariate_normal overhead too high.")


# required_sample_size 
class TestABTestPerformance:

    def test_sample_size_calc_fast(self):
        """Closed-form formula, should be effectively instant"""
        t0 = time.perf_counter()
        for _ in range(100):
            required_sample_size(ABTestConfig())
        elapsed = time.perf_counter() - t0
        print(f"\n required_sample_size (x100): {elapsed:.4f}s (limit 0.1s)")
        assert elapsed < 0.1

    def test_simulation_7k_within_threshold(self):
        cfg = ABTestConfig(seed=42)
        n = required_sample_size(cfg)['n_per_group']
        t0 = time.perf_counter()
        simulate_experiment(cfg, n_per_group=n)
        elapsed = time.perf_counter() - t0
        print(f"\n simulate_experiment(n={n}): {elapsed:.3f}s (limit 0.5s)")
        assert elapsed < 0.5


# schema validation 
class TestSchemaValidationPerformance:
    THRESHOLD_S = 0.5

    def test_final_schema_7k_within_threshold(self, scored_7k):
        seg = apply_segmentation(scored_7k)
        # Build minimal final-schema-compatible df
        final = seg.rename(columns={"prob_churn": "prob_churn", "sigma_prob": "sigma_prob",})
        if "priority" not in final.columns:
            final["priority"] = 1

        t0 = time.perf_counter()
        try:
            validate(final, "final")
        except Exception:
            pass   
        elapsed = time.perf_counter() - t0
        print(f"\n  validate(final, 7k): {elapsed:.3f}s  (limit {self.THRESHOLD_S}s)")
        assert elapsed < self.THRESHOLD_S
