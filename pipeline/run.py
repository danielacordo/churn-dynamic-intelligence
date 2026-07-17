from __future__ import annotations
import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pipeline.config import BusinessParams, Config, ModelParams, Paths

logger = logging.getLogger(__name__)


# Step definition 
@dataclass
class PipelineStep:
    """Lightweight descriptor for one pipeline stage"""
    name: str
    number: int
    fn: Callable[[Config], None]
    enabled: bool = True


def _elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


def _run_step(step: PipelineStep, cfg: Config) -> str:
    """Executes one step and returns its status string for the final summary.
    Raises the original exception on failure so the orchestrator can abort"""
    header = f"STEP {step.number}/4 — {step.name}"

    if not step.enabled:
        logger.info("")
        logger.info(f"{header}  [SKIPPED]")
        logger.info("")
        return f"{step.name} [skip]"

    t0 = time.time()
    logger.info("")
    logger.info(header)
    logger.info("")

    try:
        step.fn(cfg)
        elapsed = _elapsed(t0)
        logger.info(f"OK {header} done ({elapsed})")
        return f"{step.name} ({elapsed})"
    except Exception as exc:
        logger.error(f"X {header} failed: {exc}")
        raise


# Step implementations 
def _step_ingest(cfg: Config) -> None:
    from pipeline.steps.ingest import ingest
    ingest(cfg.paths.raw, cfg.paths.clean)


def _step_featurize(cfg: Config) -> None:
    from pipeline.steps.featurize import featurize
    featurize(cfg.paths.clean, cfg.paths.features)


def _step_score(cfg: Config) -> None:
    from pipeline.steps.score import score
    score(cfg.paths.features, cfg.paths.final, t_horizon=cfg.business.horizon_months, prob_threshold=cfg.model.prob_threshold,)


def _step_report(cfg: Config) -> None:
    from pipeline.steps.report import report
    report(cfg.paths.final, cfg.paths.report, assumptions=cfg.business.to_assumptions(), threshold=cfg.model.prob_threshold)


# Orchestrator 
def build_steps(cfg: Config) -> list:
    """ Returns the ordered list of PipelineStep objects with their enabled flags resolved from the Config's flow flags"""
    return [
        PipelineStep(name="INGEST", number=1, fn=_step_ingest, enabled=not cfg.skip_ingest and not cfg.only_score,),
        PipelineStep(name="FEATURIZE", number=2, fn=_step_featurize, enabled=not cfg.only_score,),
        PipelineStep(name="SCORE", number=3, fn=_step_score, enabled=True,),
        PipelineStep(name="REPORT", number=4, fn=_step_report, enabled=True,),]


def run_pipeline(cfg: Config | None = None) -> None:
    """ Orchestrates all pipeline steps in sequence"""
    if cfg is None:
        cfg = Config()

    logger.info("Pipeline config:%s", cfg.summary())

    steps = build_steps(cfg)
    t_total = time.time()
    results = []

    for step in steps:
        result = _run_step(step, cfg)
        results.append(result)

    logger.info("=" * 52)
    logger.info(f"Pipeline complete in {_elapsed(t_total)}")
    logger.info("Steps: %s", " -> ".join(results))
    logger.info("=" * 52)


# CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="Churn Dynamic System — full pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
Examples:
  python -m pipeline.run
  python -m pipeline.run --input data/my_dataset.csv
  python -m pipeline.run --skip-ingest
  python -m pipeline.run --only-score --horizon 12 --threshold 0.20
  python -m pipeline.run --clv 80 --retention-cost 20 --success-rate 0.35""", )

    # Paths
    parser.add_argument("--input", default=None, help="Raw CSV input (overrides CHURN_INPUT env var)")

    # Model params
    parser.add_argument("--horizon", type=float, default=None, help="Prediction horizon in months (default: 6)")
    parser.add_argument("--threshold", type=float, default=None, help="Segmentation probability threshold on the calibrated probability (default: 0.20, ROI-optimal)")

    # Business params
    parser.add_argument("--clv", type=float, default=None, help="Monthly CLV per customer in USD (default: 65)")
    parser.add_argument("--retention-cost", type=float, default=None, help="Retention campaign cost per customer in USD (default: 25)")
    parser.add_argument("--success-rate", type=float, default=None, help="Fraction of churners retained with intervention (default: 0.30)")

    # Flow flags
    parser.add_argument("--skip-ingest", action="store_true", help="Skip step 1 (use existing telco_clean.csv)")
    parser.add_argument("--only-score",  action="store_true", help="Run only steps 3 and 4 (score + report)")

    # Logging
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S",)

    # Config, only overrides values the user explicitly passed
    paths = Paths()
    business = BusinessParams()
    model = ModelParams()

    if args.input is not None:
        paths.raw = args.input
    if args.horizon is not None:
        business.horizon_months = int(args.horizon)
    if args.clv is not None:
        business.clv_monthly = args.clv
    if args.retention_cost is not None:
        business.retention_cost = args.retention_cost
    if args.success_rate is not None:
        business.success_rate = args.success_rate
    if args.threshold is not None:
        model.prob_threshold = args.threshold

    cfg = Config(paths=paths, business=business,
        model=model, log_level=args.log_level,
        skip_ingest=args.skip_ingest, only_score=args.only_score,)

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
