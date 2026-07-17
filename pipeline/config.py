from __future__ import annotations
import os
from dataclasses import dataclass, field


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


# Paths 
@dataclass
class Paths:
    """All file paths used by the pipeline"""
    raw: str = field(default_factory=lambda: _env_str("CHURN_INPUT", "data/telco_churn.csv"))
    clean: str = "data/telco_clean.csv"
    features: str = "data/telco_features.csv"
    final: str = "data/telco_final.csv"
    report: str = "report"


# Business assumptions 
@dataclass
class BusinessParams:
    """Economic parameters used for segmentation and ROI reporting"""
    clv_monthly: float = field(default_factory=lambda: _env_float("CHURN_CLV",           65.0))
    retention_cost: float = field(default_factory=lambda: _env_float("CHURN_RET_COST",      25.0))
    success_rate: float = field(default_factory=lambda: _env_float("CHURN_SUCCESS_RATE",   0.30))
    horizon_months: int   = field(default_factory=lambda: int(_env_float("CHURN_HORIZON",    6)))

    def to_assumptions(self) -> dict:
        """Returns the dict format expected by src/business.py"""
        return {
            "CLV_MONTHLY": self.clv_monthly,
            "RETENTION_COST": self.retention_cost,
            "SUCCESS_RATE": self.success_rate,
            "HORIZON_MONTHS": self.horizon_months,}


# Scoring / model parameters 
@dataclass
class ModelParams:
    """Parameters controlling the scoring and segmentation steps """
    # ROI-optimal threshold on the *calibrated* probability (prob_churn_calibrated).
    # The raw physics-based prob_churn is systematically overconfident (mean ~0.50 vs an
    # actual churn rate of ~0.265 on this dataset), so score.py now isotonic-calibrates it
    # before thresholding (see src.uncertainty.calibrate_dataframe_probabilities). 0.20 is the
    # ROI-optimal cutoff on the calibrated score as of the last recalibration -- re-run
    # src.business.optimize_threshold whenever the calibrator is refit on new data.
    prob_threshold: float = field(default_factory=lambda: _env_float("CHURN_THRESHOLD", 0.20))


# Config 
@dataclass
class Config:
    """ Aggregates config object passed to run_pipeline()"""
    paths: Paths = field(default_factory=Paths)
    business: BusinessParams = field(default_factory=BusinessParams)
    model: ModelParams = field(default_factory=ModelParams)
    log_level: str = field(default_factory=lambda: _env_str("CHURN_LOG_LEVEL", "INFO"))

    # Pipeline flow flags (set by CLI, not stored as env vars)
    skip_ingest: bool = False
    only_score: bool = False

    def summary(self) -> str:
        """Human-readable config summary for logging"""
        p = self.paths
        b = self.business
        m = self.model
        return (
            f"\n Paths raw={p.raw} clean={p.clean}"
            f"\n features={p.features} final={p.final} report={p.report}"
            f"\n Business  CLV=${b.clv_monthly}/m  cost=${b.retention_cost}"
            f" success={b.success_rate:.0%}  horizon={b.horizon_months}m"
            f"\n Model threshold={m.prob_threshold}"
            f"\n Flow skip_ingest={self.skip_ingest} only_score={self.only_score}")
