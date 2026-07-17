# Executive Summary
## Physics-Informed Churn Prediction: Retention ROI Analysis

**Dataset:** IBM Telco Customer Churn (7,043 customers, 26.5% churn rate)
**Business:** Telecom subscription service
**Model:** Physics-informed ODE (empirically calibrated via SLSQP, saturation-free, protective/entertainment service split, InternetService and PaymentMethod each as their own component) + calibrated probability + ROI-optimized threshold

---

## The Problem

At 26.5% churn rate, **$728,910** in monthly recurring revenue is at risk per campaign cycle. Standard retention approaches have three failure modes: they flag customers too late, optimize for the wrong metric (accuracy instead of ROI), and treat every at-risk customer the same, burning budget on interventions that don't match the actual risk profile.

---

## What the Data Shows

**The physics formula's own coefficients were empirically calibrated, not hand-picked, and three structural gaps in how they were built were closed along the way.**

The 33 constants that turn contract type, tenure, services, payment method, and internet connection type into E0/E_eq/gamma were originally chosen by hand, and E0 itself was a straight sum of terms hard-clipped to [0,1], a design that let **60.7% of customers saturate at exactly E0=1.0**, with zero discriminative power between them. Fitting the coefficients to minimize log-loss against actual churn (via SLSQP, subject to economically-motivated orderings, 5-fold cross-validated, see `src/physics_calibration.py`), redesigning E0 as a proper weighted average, splitting a single homogeneous "number of services" count into **protective** (security/support add-ons) and **entertainment** (streaming) groups with independent coefficients, adding **InternetService** (No/DSL/Fiber optic) as its own categorical term where before it had none at all, and splitting **PaymentMethod** into three categories (auto / mailed check / electronic check) instead of one binary flag, took the raw formula's own AUC from **0.699 to 0.837** across the four fixes combined, while dropping E0 saturation to **0%**. The ROI-optimal threshold (0.20) now recovers **$132,055** per campaign cycle, up from $108,836 at the industry-default threshold of 0.50. CLV also moved from a flat $65/month average to each customer's own `MonthlyCharges`, checked, to capture more real variance than a contract-type segmentation would have (see `DECISIONS.md`).

At the same time, not all high-risk customers respond the same way. Two customers at the *identical* churn probability can require completely different interventions depending on how fast they recover from negative events (τ).

---

## Three Key Findings

**1. Calibration, a formula redesign, and closing three real feature gaps, not just a coefficient tweak, is what moved the number**

| Stage | Raw formula AUC | E0 saturation |
|---|---|---|
| Hand-picked coefficients, sum-then-clip E0 | 0.699 | 60.7% of customers |
| + Fitted coefficients, weighted-average E0, protective/entertainment services split | 0.791 | 0% |
| + InternetService as its own categorical term | 0.832 | 0% |
| **+ PaymentMethod split into three categories** | **0.837** | **0%** |

With all four fixes plus the ROI-optimal threshold and per-customer CLV in place, the recommended campaign now recovers **$132,055** net ROI per cycle.

Isotonic calibration (fixing the *average level* of the prediction) was necessary but not sufficient: it can't fix a formula that structurally can't distinguish 6 in 10 customers from each other, nor one that treats economically different services (or payment methods) as identical, nor one that ignores a variable entirely. Each fix closed a specific, verifiable gap: E0 bounded by construction instead of an after-the-fact clip; a services split that let the optimizer independently confirm entertainment add-ons carry zero equilibrium-energy signal; an InternetService term that captured a churn-rate gap (7.4% No vs 41.9% Fiber optic) larger than the one the services split was built to address; and a PaymentMethod split that found Electronic check (45.3% churn) and Mailed check (19.1%) were being averaged together despite a ~26-point gap that survives controlling for contract type.

**2. Threshold selection still matters on top of that**

| Threshold | Customers flagged | Recall | Precision | Net ROI |
|-----------|-------------------|--------|-----------|---------|
| 0.50 (industry default) | 1,732 (24.6%) | 58.0% | 62.6% | $108,836 |
| **0.20 (ROI-optimal)** | **3,253 (46.2%)** | **84.3%** | **48.4%** | **$132,055** |

Optimal here means *maximum absolute dollars recovered*, not the highest ROI percentage: 0.50 has a better ROI *rate* on a smaller, more targeted spend.

**3. Intervention timing matters as much as targeting**

Two customers flagged at the exact same calibrated probability (58.1%) can need opposite interventions. In the scored dataset, customer `0235-KGSLC` (τ=5.4 months, "Medium resilience") is recommended for a targeted offer based on usage pattern. Customer `7590-VHVEG` (τ=6.0 months, "Fragile") is recommended for a personalized plan change with human contact, since a standard offer alone won't move them fast enough. Same score, different playbook.

---

## Cost of Doing Nothing

| Inaction | Estimated cost |
|---|---|
| Using default threshold (0.50) instead of ROI-optimal (0.20) | −$23,219 per campaign cycle |
| Intervening on the LOW-risk (Stable) segment anyway | −$58,670 (negative-ROI spend avoided by skipping it) |
| Using the original hand-picked, saturation-prone physics formula with services/InternetService/PaymentMethod unmodeled | AUC stuck at 0.699 instead of 0.837 (from the calibration + E0 redesign + service/internet-type/payment-type splits combined) |
| Using a flat $65/month CLV instead of each customer's actual `MonthlyCharges` | Understates ROI, since flagged customers skew toward higher-paying segments than the dataset average |
| No early warning system (reactive approach) | 180-day detection disadvantage |

---

## Three Actions, In Order of Priority

**Action 1: Set the classification threshold to 0.20 on the calibrated probability**
Expected outcome: +$132,055 net ROI per cycle (vs. $108,836 at the 0.50 default)
Risk: Low. Backed by 367 passing tests, a documented calibration methodology, and honest 5-fold cross-validation (not an in-sample fit).
How to validate: A/B test, splitting HIGH-risk customers into treatment vs. control, measured at 6 months. Design already in `src/ab_test.py`.

**Action 2: Differentiate intervention by recovery time (τ)**
Expected outcome: higher conversion on fast responders (τ < 4m), better-targeted spend on slow ones (τ > 6m)
Risk: Low. The Streamlit dashboard surfaces τ and a recommended action per customer automatically.
What to do: fast responders (τ < 4m) -> targeted offer; slow responders (τ > 6m) -> plan change + human contact.

**Action 3: Recalibrate the physics coefficients periodically, not just the probability**
Expected outcome: the formula stays aligned with actual churn patterns as the customer base drifts, instead of only correcting the output after the fact
Risk: Low. `update_scores.py` re-scores and recalibrates the full base in a few seconds; refitting the 33 physics coefficients (`src/physics_calibration.py`) is a heavier but still fast (~seconds) operation worth running monthly or quarterly.
How to implement: `python update_scores.py` weekly for scoring; re-run `src.physics_calibration.fit_final_params` periodically and update `src/features.py`/`src/physics.py` if the fit meaningfully shifts.

---

## Expected Impact

| Action | Per-campaign estimate |
|---|---|
| Empirically calibrated + saturation-free physics formula, with service, internet-type, and payment-type splits, vs. original hand-picked | AUC 0.699 -> 0.837 |
| Per-customer CLV (`MonthlyCharges`) vs. flat $65/month average | Captures real variance a contract-type segmentation would have missed |
| ROI-optimal threshold (0.20) vs. default (0.50) | +$23,219 |
| Avoided by skipping the LOW-risk segment | +$58,670 (loss avoided) |
| **Combined effect (all fixes above)** | **~$132,055 net ROI on the recommended campaign** |

The ROI estimate is still sensitive to `SUCCESS_RATE` (the assumed 30% retention rate on a contacted churner); this remains the single assumption most worth validating experimentally. The A/B test design in `src/ab_test.py` (sample size, sequential monitoring, break-even analysis) exists specifically to replace that assumption with a measured number.

---

*Full technical analysis: see `src/physics_calibration.py` (coefficient fitting methodology and the E0 redesign), `notebooks/09_baseline_comparison.ipynb` (model comparison), and `notebooks/11_impact_metrics.ipynb` (this ROI analysis).*
