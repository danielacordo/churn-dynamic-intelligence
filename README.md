# Churn Dynamic Intelligence
### Physics-Informed Predictive Analytics | IBM Telco Dataset

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![Tests](https://github.com/danielacordo/churn-dynamic-intelligence/actions/workflows/tests.yml/badge.svg)](https://github.com/danielacordo/churn-dynamic-intelligence/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red?style=flat-square)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-IBM%20Telco-lightgrey?style=flat-square)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**[-> Executive Summary](EXECUTIVE_SUMMARY.md)** | **[-> Notebooks](notebooks/)** | **[-> Source Code](src/)**

---

## What this project is

Most churn models answer one question: *who will leave?* The campaigns built from them treat every flagged customer the same (same offer, same timing, same channel) and wonder why conversion is low.

The question that actually drives campaign design is different: *why is this customer at risk, and when will an intervention work?* A customer who's structurally misaligned (wrong product, wrong contract) won't respond to a discount. A customer who had a bad billing month probably will, but only if you reach them soon, not months later.

I modeled customer loyalty as a damped harmonic oscillator: each customer has an engagement level, a structural equilibrium they drift toward, and a recovery speed after external shocks. That gives three outputs a standard classifier doesn't produce: a confidence interval per prediction, an intervention-timing estimate (τ), and a diagnosis of *structural* vs *transient* risk. The model scores 7,043 customers in a few seconds and tells the CRM team not just who to call, but what to offer and why.

The ROI result: thresholding the calibrated probability at 0.20 delivers **$132,055 net ROI per campaign**. The industry-default threshold (0.50) delivers $108,836 on the same model, same data. That ~$23K difference comes from one number in a config file, on top of another ~$20K unlocked by calibrating the physics formula's coefficients, which itself came from fixing a structural flaw in how E0 combined its terms and then giving three features their own physically-distinct coefficients instead of one shared or absent one (see below).

---

## What I did, and what was hard

**The physics formula's coefficients were hand-picked, and fitting them properly surfaced a deeper structural bug, then surfaced three more real signals the formula was averaging away.** E0, E_eq, and gamma are computed from contract type, tenure, services, payment method, and internet connection type via 33 constants, originally chosen to "look reasonable," not estimated from data. I built a calibration module (`src/physics_calibration.py`) that fits them by minimizing log-loss against actual churn, via `scipy.optimize.minimize` (SLSQP, which supports nonlinear inequality constraints; L-BFGS-B, used in an earlier iteration, only supports box bounds) subject to economically-motivated orderings (contract-tier ordering for E_eq/gamma; protective services' coefficient ≥ entertainment services'; "no internet" ≥ DSL ≥ fiber optic; auto-pay ≥ mailed check ≥ electronic check), plus a mild L2 penalty back toward reasonable defaults, validated with 5-fold cross-validation (fitting fresh inside each training fold, evaluated on held-out data it never saw, not an in-sample number). The first fit revealed the real problem, not just an opportunity: E0 was a straight sum of terms hard-clipped to `[0,1]`, and any customer with decent tenure, stable billing, and a long contract already summed past 1.0 *before* the clip. **60.7% of customers landed at exactly E0=1.0**, indistinguishable from each other. I redesigned E0 as a convex combination (weights normalized to sum to 1, applied to sub-scores each already in `[0,1]`) so it's bounded *by construction*, not by an after-the-fact clip. Saturation dropped to 0%, at effectively no AUC cost.

Two more findings came from refusing to treat every discretionary service as economically equivalent. `num_services` used to be one homogeneous count of six add-ons. Splitting it into **protective** (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport) and **entertainment** (StreamingTV, StreamingMovies), motivated by a raw churn-rate gap of ~15% vs ~30% between the groups, let the constrained optimizer independently discover that entertainment's E_eq coefficient converges to *exactly* 0.0: it carries no equilibrium-energy signal once protective services are modeled separately. That alone moved out-of-fold AUC from 0.699 to 0.791. Digging into what was still being left out of both groups turned up an even bigger gap. `InternetService` was excluded entirely, on the reasonable-sounding grounds that it's a structural prerequisite for having those services at all (you can't have OnlineSecurity without Internet), not a discretionary choice with the same economic meaning; but the gap itself (7.4% churn with no internet vs 19.0% DSL vs 41.9% Fiber optic) is larger than the protective/entertainment gap, and persists even holding protective-service count fixed (42.7% vs 63.6% among customers with zero protective add-ons). The exclusion from the two service groups was still the right call, since it's genuinely a different kind of variable, but it needed its own coefficient, not no coefficient. Modeled as a third categorical connection-type term, symmetric with the existing contract-tier terms, it moved out-of-fold AUC again, from 0.791 to 0.832, and Fiber optic's fitted E_eq coefficient converges to exactly 0.0, the same pattern as entertainment services, an independent confirmation that the highest-churn segment carries no positive equilibrium signal beyond the baseline. The same audit applied to `PaymentMethod`: its binary `auto_payment` flag (automatic vs. not) already caught most of the churn-rate gap, but within "not automatic," Electronic check (45.3%) and Mailed check (19.1%) turned out to be far from equivalent, a ~26-point gap that survives controlling for Contract. Splitting into three payment categories instead of one binary moved out-of-fold AUC once more, from 0.832 to 0.837. Gamma needed the opposite kind of restraint throughout: left unconstrained, the optimizer collapses it to a degenerate value for some contract tiers, because a single cross-sectional time horizon can't separately identify gamma and E_eq; many (gamma, E_eq) pairs explain the same outcome equally well. I narrowed gamma's search bounds to keep it in a physically realistic range instead of chasing the last bit of log-loss at the cost of a nonsensical fit; both trade-offs, and why, are documented directly in the module.

**The raw model score was overconfident, and I had to catch it before it reached a business decision.** Even after coefficient calibration, thresholding a raw probability directly is risky, so I added isotonic calibration (fit against observed churn) as a second, independent safety net before any threshold or ROI number is computed. It's applied consistently everywhere the score is used, in the pipeline, the CLI, and the dashboard, precisely because I initially fixed it in one place and then found the same uncalibrated score quietly reappearing in other code paths that had been added afterward. That's not a hypothetical risk; it's what happened during development, and it's why calibration is now the *default* behavior of the scoring function rather than an opt-in step.

**Threshold optimization isn't in any tutorial, so I derived the objective directly from the business.** Every ML resource either fixes the threshold at 0.50 or tunes it for F1. Neither makes sense once you know a retention campaign costs $25/customer and each customer's CLV is their own actual `MonthlyCharges`, not a flat average. I wrote the ROI objective directly, `n_retained x CLV x horizon − n_intervened x cost`, and maximized it over the threshold grid on the *calibrated* probability. At 0.50, the model flags 1,732 customers and catches 58% of actual churners. At 0.20, it flags 3,253 and catches 84%. Same model, same data, ~$23K difference in net ROI.

**The physical model no longer concedes much AUC to gradient boosting; the protective/entertainment and InternetService fixes changed that.** Gradient boosting on the combined physical + raw features reaches AUC 0.845 (`notebooks/09`) to 0.844 (`notebooks/07`). A logistic regression trained on physics features alone now reaches 0.834 (`notebooks/09`, on `[E0, E_eq, gamma, tau]` only) to 0.844 (`notebooks/07`, on the fuller physical feature set including the `is_internet_*` flags directly), both ahead of the 0.812-0.822 baseline LR on raw features, not just close to it. The two notebooks measure slightly different "Physical LR" feature sets by design (`07` includes the new `is_internet_*` flags explicitly; `09` deliberately keeps to the four core ODE outputs to isolate what the dynamics themselves, without any raw categorical flags, can do), and the gap between them shrank once InternetService got its own coefficient in the formula that produces E0/E_eq/gamma in the first place, since more of its signal now flows through those four numbers even in the narrower feature set. The reason for the physical model isn't AUC anyway; it's that XGBoost gives you a single number, and this model gives you a number plus a τ that tells you whether a targeted offer will convert in weeks or won't move the customer at all. The right production architecture uses both: the GBM blend for ranking, the physical model's τ/segment for triage.

**367 tests that run without the dataset, and it caught real bugs, including one in the calibration code itself.** The Kaggle CSV isn't committed to the repo, which forced every physical claim to be expressed as a testable assertion on synthetic inputs rather than "run it on real data and eyeball the output." That same synthetic-fixture design caught a bug in the coefficient-fitting module during development: it silently zeroed out the services/payment signal when run directly on pre-feature-engineering data, because it assumed columns that hadn't been computed yet existed, a wrong assumption that (before it was caught) made a chunk of the physics coefficients meaningless. It also caught a pytest-discovery bug: most test files are named after the module they cover (`physics.py`, `business.py`, ...) rather than pytest's default `test_*.py` pattern, so a plain `pytest` was silently collecting only 18 of 349 tests. A `pytest.ini` now makes discovery explicit.

---

## TL;DR: The Findings

| # | Finding | Implication |
|---|---------|-------------|
| 1 | E0 was structurally saturated at exactly 1.0 for 60.7% of customers (sum-then-clip formula) | Redesigning it as a bounded weighted average dropped saturation to 0% at no AUC cost |
| 2 | `num_services` treated protective (security/support) and entertainment (streaming) services as economically equivalent | Splitting them let the optimizer discover entertainment's E_eq coefficient is exactly 0.0: no equilibrium signal at all |
| 3 | `InternetService` was excluded from the model entirely; its churn-rate gap (7.4% No vs 41.9% Fiber optic) is *larger* than the one that motivated finding #2, and wasn't being used anywhere | Adding it as its own categorical term moved out-of-fold AUC from 0.791 to 0.832 |
| 4 | `PaymentMethod` was collapsed into a binary `auto_payment` flag, still averaging away Electronic check's 45.3% churn vs. Mailed check's 19.1%, a gap that survives controlling for Contract | Splitting into three categories moved out-of-fold AUC from 0.832 to 0.837 |
| 5 | Empirically calibrating the physics formula's coefficients (SLSQP, constrained) raised its AUC from 0.699 to 0.837 across all four fixes above | ~$20K net ROI per cycle from the formula fixes alone, before touching the threshold |
| 6 | CLV was a flat $65/month average; using each customer's own `MonthlyCharges` instead captures real variance a contract-type segmentation would have missed (see `DECISIONS.md`) | Customers the model actually flags skew toward higher-paying segments than the dataset average |
| 7 | Default threshold 0.50 flags 1,732 customers and catches 58% of actual churners | $23K left on the table per campaign cycle |
| 8 | ROI-optimal threshold on the *calibrated* probability is 0.20, not 0.50 | One config change, no retraining required |
| 9 | Two customers at 58.1% churn probability with τ=6.0m vs τ=5.4m need different interventions | A standard offer works for one, does nothing for the other |
| 10 | The LOW-risk (Stable) segment costs $89,675 to contact and recovers $31,005 | Skip it, it's negative ROI |
| 11 | The model identifies at-risk customers 180 days before the churn event | Multiple intervention cycles before the customer is lost |

---

## Honest Calibration

> **The $132,055 ROI number depends on one assumption: `SUCCESS_RATE = 30%`.** That's the fraction of flagged churners who actually stay after intervention. It's the single assumption most worth validating with a real experiment; everything else in the pipeline (physics coefficients, probability calibration, threshold optimization, segmentation, and now each customer's own CLV via their actual `MonthlyCharges`) is now derived from the data itself, cross-validated out-of-fold where relevant. The full design for that validation (sample size calculation, sequential monitoring, break-even analysis) is already in `src/ab_test.py`. Running it on one campaign cycle would make the number defensible to finance.

> **Gamma is only weakly identifiable from this dataset, and the calibration is upfront about it.** IBM Telco is cross-sectional (one snapshot per customer, no time series), so a single time horizon can't separately pin down recovery speed (gamma) and structural equilibrium (E_eq); many combinations explain the same outcome equally well. Left unconstrained, the optimizer pushes gamma to a degenerate value for some contract tiers to shave a little more log-loss. `src/physics_calibration.py` keeps gamma inside a narrower, physically realistic search range for exactly this reason, and documents the trade-off directly rather than reporting the unconstrained (better-looking, less trustworthy) fit. The roadmap for what full ODE dynamics fitting would need (longitudinal data, a repeated-measures likelihood) is in `src/physics.py::estimate_params_discussion()`.

> **The protective/entertainment, InternetService, and PaymentMethod splits were feature-engineering decisions guided by the data, not free wins with no cost.** Each split adds parameters (18 to 23 for InternetService, 29 to 33 for PaymentMethod) and constraints to keep the fit interpretable; the justification for each one is a real, sizeable churn-rate gap that persists under a control (protective services still matter after splitting out entertainment; InternetService's gap survives holding protective-service count fixed; PaymentMethod's Electronic-check-vs-Mailed-check gap survives holding Contract fixed), not just "more parameters improve in-sample fit," which is true of nearly any split and proves nothing on its own. The 5-fold out-of-fold AUC gain at each step (0.699 to 0.791 to 0.832 to 0.837) is the evidence that these particular splits captured real structure rather than noise.

> **Telecom only.** The coefficients (contract-type base rates, service-group bonuses, internet-type effects, payment-method effects) are fit on this sector's data. The framework applies conceptually to any subscription business with tenure, monthly charge, and contract-type data, but the numbers don't transfer without refitting on the target sector's own data (which is now a documented, repeatable procedure, not a one-off hand-calibration).

---

## Campaign Output

```
$ python main.py --simulate

 CAMPAIGN SIMULATION - by risk level
 Assumptions: avg CLV=$65.0/m (per-customer MonthlyCharges used internally) - Campaign cost=$25.0 - SR=30% - Horizon=6m

 Segment                           N  Retained     Revenue      Cost         ROI
 HIGH risk (lb > 0.35)          1,967       355  $   138,528  $  49,175  $   +89,353
 UNCERTAIN                        360        32  $    12,636  $   9,000  $    +3,636
 MEDIUM risk                    1,129        94  $    36,504  $  28,225  $    +8,279
 Stable (LOW)                   3,587        80  $    31,005  $  89,675  $   -58,670  <- skip
 -> At threshold 0.20           3,253       473  $   213,380  $  81,325  $  +132,055  <- recommended

What each customer looks like (python main.py):

  #01  7590-VHVEG
       Risk     [███████████░░░░░░░░░] 58% ± 2%   CI [58%–58%]
       Segment  High Risk / Fragile
       Action   Personalized plan + human contact, slow response to incentives (τ=6.0m)

  #02  0235-KGSLC
       Risk     [███████████░░░░░░░░░] 58% ± 2%   CI [58%–67%]
       Segment  High Risk / Medium resilience
       Action   Targeted offer based on usage pattern (τ=5.4m)
```

Two customers with the *identical* churn probability. Completely different interventions. The model surfaces this automatically, with no rules written by hand.

---

## How it works

**Physical model core:**
```
dE/dt = -γ · (E - E_eq) + F(t)

E(t)        -> customer engagement at time t
E_eq        -> structural loyalty (where they settle without perturbation)
γ           -> damping coefficient  |  τ = 1/γ -> recovery time in months
F(t)        -> external shocks (price increase, service failure, competitor offer)
E_critical  = 0.25 -> below this, churn is likely regardless of intervention
```

The 23 coefficients that turn contract/tenure/services/payment/internet-type into E0/E_eq/γ are fit empirically (`src/physics_calibration.py`), not hand-picked. This includes a redesign of how E0's sub-scores combine (weighted average, not sum-then-clip) that eliminated a 60.7%-of-customers saturation bug, a split of the services signal into protective (security/support) vs entertainment (streaming) groups that let the optimizer discover entertainment's equilibrium-energy coefficient is exactly 0.0, and a third categorical component for internet connection type (No/DSL/Fiber optic) whose churn-rate gap turned out to be larger than the services split's. See "What I actually did" above.

**Pipeline:**
```
data/telco_churn.csv
        │
        ▼
pipeline/steps/ingest.py     <- raw data cleaning
        │
        ▼
pipeline/steps/featurize.py  <- physical features: E0, γ, E_eq per customer
        │
        ▼
pipeline/steps/score.py      <- ODE -> churn probability -> calibration -> segment + action
        │
        ▼
pipeline/steps/report.py     <- ranked list + campaign simulation + sensitivity + A/B design
        │
        ▼
app.py (Streamlit)           <- 7-tab interactive dashboard
```

**Schema contracts at every boundary:**
```
raw CSV -> [validate "raw"] -> ingest -> [validate "clean"] -> featurize
       -> [validate "features"] -> score -> [validate "final"] -> report
```
`pipeline/schema.py` validates required columns, types, and value ranges at each of the four stages with a small hand-written checker (not a third-party library): if a column is renamed or a type changes upstream, the pipeline fails loudly at the exact step where it broke, not silently downstream.

**Test coverage: 367 tests, no dataset required:**

| Test file | What it covers | Tests |
|---|---|---|
| `business_advanced.py` | Threshold optimization (incl. per-customer CLV, CLV-source transparency), sensitivity tornado | 47 |
| `ab_test.py` | Sample size, CI coverage, sequential analysis | 48 |
| `dashboard.py` | Dashboard KPIs, scenarios, threshold curve, break-even | 47 |
| `schema.py` | Schema validation across all 4 pipeline stages | 40 |
| `visualization.py` | Chart functions, axes, data integrity (headless) | 40 |
| `physics.py` | ODE correctness, equilibrium, perturbations, service/internet-type/payment-type coefficients | 36 |
| `monte_carlo.py` | MC vs analytical, boundary effects, correlations | 31 |
| `features.py` | Vectorized feature engineering | 26 |
| `business.py` | Segmentation, ROI (incl. per-customer CLV), calibration | 29 |
| `pipeline.py` | End-to-end integration (synthetic data) | 14 |
| `performance.py` | Throughput benchmarks: 7k customers in a few seconds | 9 |

---

## Project structure

```
churn-dynamic-intelligence/
├── main.py                    <- entry point: python main.py
├── app.py                     <- Streamlit dashboard (7 tabs)
├── update_scores.py           <- re-score + recalibrate the full base end to end
├── Makefile                   <- make run / simulate / test / lint / streamlit
├── pipeline/
│   ├── run.py                 <- orchestrator (python -m pipeline.run)
│   ├── schema.py               <- hand-written column/type/range contracts, all 4 stages
│   ├── config.py               <- CLV, cost, horizon, threshold (overridable via env vars)
│   └── steps/                 <- ingest · featurize · score · report
├── src/
│   ├── physics.py              <- ODE, energy, damping, equilibrium
│   ├── physics_calibration.py  <- empirical coefficient fitting (SLSQP, constrained, 5-fold CV) + E0 redesign + service/internet-type splits
│   ├── uncertainty.py          <- analytical propagation, Monte Carlo, probability calibration
│   ├── business.py             <- ROI optimization, segmentation, sensitivity
│   ├── features.py             <- vectorized physical feature pipeline
│   └── ab_test.py              <- A/B test design + simulation (wired into pipeline/steps/report.py and the dashboard)
├── tests/                     <- 367 tests, no dataset needed
├── notebooks/                 <- 12 notebooks (01_eda -> 11_impact_metrics, + a DuckDB/SQL analysis)
└── dashboard/
    ├── business.py             <- shared KPI/break-even logic, used by app.py's tabs
    └── tabs/                   <- the 7 Streamlit tabs
```

> Notebooks are the exploratory/demo layer, and they're honest ones. See `notebooks/09_baseline_comparison.ipynb` for where the physical model concedes AUC to gradient boosting. Production logic lives in `src/` and `pipeline/`.

---

## How to run

```bash
git clone https://github.com/danielacordo/churn-dynamic-intelligence
cd churn-dynamic-intelligence
pip install -r requirements.txt

# Download dataset -> save as data/telco_churn.csv
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# (or: pip install kagglehub && python download_data.py)

python main.py                 # full pipeline + top 10 customers
python main.py --simulate      # + campaign ROI simulation by segment
python main.py --top 20        # top 20 customers
python main.py --skip-ingest   # re-score only (data already loaded)

pytest tests/ -v               # 367 tests, no dataset needed
streamlit run app.py           # interactive dashboard

make simulate                  # same as `python main.py --simulate`
make test                      # same as `pytest tests/ -v`
```

To refit the physics coefficients yourself (e.g. after a period of new data):
```python
from src.physics_calibration import evaluate_calibration, fit_final_params
import pandas as pd

df = pd.read_csv('data/telco_clean.csv')
report = evaluate_calibration(df, df['Churn_bin'])   # honest 5-fold CV, hand-picked vs fitted
final = fit_final_params(df, df['Churn_bin'], alpha=0.05)  # production coefficients
```

---

## Next steps

**1. A/B test to validate the success-rate assumption.** `SUCCESS_RATE = 30%` is the single assumption that moves the ROI result most now that the formula itself is calibrated. A treatment-vs-control experiment on HIGH-risk customers, measured at 6 months, would make the ROI number defensible to finance. The design (sample size, sequential monitoring, break-even rate) is already in `src/ab_test.py`.

**2. ~~Segment CLV by contract type.~~ Done, using each customer's own `MonthlyCharges` instead.** The model used to apply a uniform CLV of $65/month to every customer. Before implementing a contract-type segmentation, I checked whether contract type actually explains much of the variance in what customers pay: it doesn't. `MonthlyCharges` varies far more *within* a contract type (std ≈ 27) than *between* contract-type averages (std ≈ 2.9 across Month-to-month/One year/Two year). Segmenting by contract would have captured almost none of the real signal. Using each customer's actual `MonthlyCharges` directly, already an observed column with no new assumption needed, captures all of it. See `DECISIONS.md` for the full before/after.

**3. ~~Audit `PaymentMethod` for the same signal InternetService carried.~~ Done, split into three categories.** `PaymentMethod` used to collapse into a binary `auto_payment` flag (automatic vs. not), which already caught most of the gap (34.7% churn non-auto vs 16.0% auto). Checked whether that binary was still averaging away a real difference before touching anything: it was. Electronic check churns at 45.3% vs. Mailed check's 19.1%, a ~26pp gap that survives controlling for Contract. Modeled as three categories (auto / mailed check / electronic check) instead of one binary, with the same SLSQP + inequality-constraint + 5-fold-CV discipline as the other splits. Combined with the CLV change above, this moved the recommended-threshold ROI from $101,483 to **$132,055** (+162%) and the raw formula's out-of-fold AUC from 0.832 to 0.837. See `DECISIONS.md` for the full analysis, including why the constraint bound (not just the raw gap) is what confirms this is a real, non-forced ordering.

**4. Recalibrate on a cadence, not once.** `update_scores.py` refreshes scores weekly; `src/physics_calibration.py`'s coefficient fit should be re-run periodically (monthly/quarterly) as the customer base and churn patterns drift, with the honest CV comparison (`evaluate_calibration`) checked each time before adopting a new fit, not just the in-sample number.

**5. ~~Audit Contract x InternetService for an interaction effect.~~ Checked, not worth adding.** A logistic regression with the interaction term (9 parameters) was compared against one without it (5 parameters) via AIC and a likelihood ratio test: the interaction model's AIC is worse (6361.11 vs. 6358.99) and the LR test gives p=0.209, not significant. Checking additivity directly confirms it: the churn log-odds gap between Month-to-month and Two-year contracts is close to constant across all three internet types (3.19 / 2.74 / 3.38), which is what an additive model predicts. The current formula, with contract and internet-type bonuses added independently, already captures this correctly; adding an interaction term would have been 4 more parameters for a change that fails its own significance test. See `notebooks/07_final_model.ipynb` (re-runnable check, no new dependency) and `DECISIONS.md` entry #14 for the full analysis.

---

*Analysis, code and methodology by [Daniela Cordo](https://www.linkedin.com/in/daniela-cordo-1708203bb) · [Email](mailto:danielacordo24@gmail.com)*

> Dataset: IBM Telco Customer Churn (public, Kaggle). The methodology (physical variables empirically calibrated against outcomes, probability calibration, threshold optimization, uncertainty propagation, segment-level ROI) applies conceptually to any subscription business with tenure, monthly charge, and contract-type data.
