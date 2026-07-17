# Decision Log

This log records the significant technical and methodological decisions made on this
project, the alternatives considered, and why each call was made. It exists so that a
technical reviewer, or a future maintainer, doesn't have to reverse-engineer intent
from diffs. Each entry states the decision, the trade-off, and where to verify it in
the code.

---

### 1. Model the physical formula's coefficients as *fitted*, not hand-picked

**Decision:** Replace the original hand-picked constants in the E0 / E_eq / γ formulas
with coefficients estimated from data (`src/physics_calibration.py`), validated with
5-fold cross-validation.

**Why:** A physical framing is only a genuine analytical asset if its parameters are
falsifiable against outcomes. Hand-picked constants that "look reasonable" are a
narrative, not a model: they can't be wrong, which means they can't be right either.
Fitting them turned the formula into something that can actually be checked, and
immediately surfaced a real structural bug (see #3) that hand-picking had been hiding.

**Trade-off accepted:** Fitted coefficients need a validation discipline (out-of-fold,
not in-sample) to be trustworthy, which hand-picked constants don't require. That
discipline is now load-bearing; see #2.

**Verify:** `src/physics_calibration.py::fit_final_params`, `evaluate_calibration`.

---

### 2. Cross-validate out-of-fold, refit inside each fold

**Decision:** `evaluate_calibration` refits the coefficients from scratch inside each
of 5 training folds and scores only on the held-out fold; it never reports an
in-sample number.

**Why:** Reporting AUC on the same data used to fit 23 coefficients is close to
meaningless; the optimizer will always find a fit that flatters itself on data it has
seen. The out-of-fold number (0.699 to 0.791 to 0.832 across the three fixes below) is
the only version of "the calibration improved things" that a skeptical reviewer should
accept, and it's the number this project reports everywhere.

**Trade-off accepted:** 5x the compute cost of a single fit, and the reported metric is
necessarily a few points lower than the (more flattering, less meaningful) in-sample
number would be.

**Verify:** `src/physics_calibration.py::evaluate_calibration`; the CV loop is explicit,
not hidden behind a library call.

---

### 3. Redesign E0 as a bounded convex combination, not a sum-then-clip

**Decision:** E0 was originally a straight sum of five sub-scores, hard-clipped to
`[0, 1]`. It's now a weighted average of sub-scores that are each already in `[0, 1]`,
with weights normalized to sum to 1, bounded by construction, not by an after-the-fact
clip.

**Why:** The sum-then-clip version put **60.7% of customers at exactly E0 = 1.0**.
Any customer with decent tenure, stable billing, and a long contract summed past 1.0
before the clip, making them indistinguishable from each other regardless of how much
better one actually was. This wasn't a cosmetic issue; it was destroying real signal
for the majority-good-standing segment of the customer base, which is exactly the
segment where distinguishing "very stable" from "extremely stable" determines whether
a borderline case gets flagged early. The fitting exercise in #1 is what surfaced it:
hand-picked constants had been quietly living with a saturated formula.

**Trade-off accepted:** None material. Saturation dropped from 60.7% to 0% at
effectively no AUC cost, because the information the sum-then-clip version was
discarding had no business being discarded in the first place.

**Verify:** `src/physics.py` (E0 computation), `tests/physics.py` (saturation
regression tests).

---

### 4. Split `num_services` into protective vs. entertainment sub-scores

**Decision:** Stop treating the six discretionary add-on services as one homogeneous
count. Split them into **protective** (OnlineSecurity, OnlineBackup, DeviceProtection,
TechSupport) and **entertainment** (StreamingTV, StreamingMovies), each with its own
fitted coefficient in E_eq and γ.

**Why:** The raw churn-rate gap between the two groups (~15% vs. ~30%) suggested they
weren't economically equivalent. Splitting them let the constrained optimizer
*independently discover* that entertainment's E_eq coefficient converges to exactly
0.0: it carries no equilibrium-energy signal once protective services are modeled
separately. That's a substantive finding, not an assumption baked into the split: the
optimizer wasn't told to zero it out, the constrained fit found that on its own. This
one change moved out-of-fold AUC from 0.699 to 0.791.

**Trade-off accepted:** Five more parameters to fit and constrain (protective ≥
entertainment, enforced via SLSQP inequality; see #6), and a services taxonomy that
now needs to be re-examined any time IBM Telco-style categorical services are added to
a future dataset (see the "Open items" section below on `PaymentMethod`).

**Verify:** `src/features.py` (feature construction), `src/physics_calibration.py`
(constraint definitions), `notebooks/07_final_model.ipynb` (the dose-response evidence
that motivated the split, re-derived directly from data).

---

### 5. Give `InternetService` its own categorical term, instead of leaving it implicit

**Decision:** Add internet connection type (No / DSL / Fiber optic) as a third
categorical component in E0 / E_eq / γ, symmetric with the existing contract-tier
terms.

**Why:** `InternetService` had been excluded from the formula entirely, on the
reasonable-sounding logic that it's a structural prerequisite for having protective or
entertainment services at all, not a discretionary choice with the same economic
meaning. That logic was still correct, but "excluded because it's a different kind of
variable" had silently become "excluded, full stop." The raw gap (7.4% churn with no
internet vs. 19.0% DSL vs. 41.9% Fiber optic) is **larger** than the gap that motivated
the protective/entertainment split, and survives holding protective-service count
fixed, meaning it's not redundant with that split. Adding it moved out-of-fold AUC from
0.791 to 0.832, and Fiber optic's fitted E_eq coefficient converges to exactly 0.0,
independently confirming the same pattern as entertainment services: the highest-churn
segment carries no positive equilibrium signal beyond baseline.

**Trade-off accepted:** 18 to 23 parameters. More parameters to keep interpretable and
constrained, for a gain that the out-of-fold AUC shows is real rather than an
in-sample artifact.

**Verify:** `src/physics_calibration.py`, `notebooks/07_final_model.ipynb` (cells
deriving the connection-type churn gap and the controlled comparison against
protective-service count).

---

### 6. Constrained optimization (SLSQP) over an unconstrained fit, and why L-BFGS-B was rejected

**Decision:** Fit all coefficients jointly via `scipy.optimize.minimize(method='SLSQP')`
subject to economically-motivated inequality constraints (contract-tier ordering for
E_eq/γ; protective ≥ entertainment; "no internet" ≥ DSL ≥ Fiber optic on E_eq), plus a
mild L2 penalty back toward the original hand-picked defaults.

**Why SLSQP specifically:** An earlier iteration used L-BFGS-B, which only supports
box bounds; it has no mechanism for the inequality constraints above. Once those
constraints became necessary (see below), SLSQP was the correct choice because it
natively supports nonlinear inequality constraints alongside box bounds.

**Why constraints at all, instead of a free fit:** An unconstrained log-loss
minimization has no notion of what makes economic sense; nothing stops it from
deciding, on this particular sample, that Month-to-month customers are structurally
more loyal than Two-year customers, because that ordering happens to fit the noise
marginally better. The constraints encode domain knowledge the objective function
doesn't have access to on its own, at the cost of a very slightly worse (but
trustworthy) log-loss compared to the unconstrained optimum.

**Verify:** `src/physics_calibration.py`, lines documenting `_constraint_funcs()` and
the SLSQP vs. L-BFGS-B comment inline.

---

### 7. Keep γ's search bounds narrow, and report why instead of hiding it

**Decision:** γ (damping / recovery speed) is bounded to a narrower, physically
realistic range per contract tier, rather than the wide box used for other groups.

**Why:** IBM Telco is cross-sectional: one snapshot per customer, no repeated
observations over time. A single time horizon can't separately identify γ and E_eq;
many (γ, E_eq) pairs explain the same six-month outcome equally well. Left
unconstrained, the optimizer pushes γ to its floor for two of three contract tiers
(τ approaching ~50 months), a fit that technically improves log-loss but is barely
distinguishable from having no γ term at all for those tiers, and would silently
invalidate every downstream τ-based threshold calibrated elsewhere in the project
(resilience tiers, timing simulations) against the dataset's actual τ range
(~2–10.5 months).

**Trade-off accepted:** This is a real, acknowledged limitation, not a solved problem:
γ is only weakly identifiable from this dataset, full stop. The narrower bounds are a
deliberate choice to keep the fit *usable* rather than chase the last fraction of a
log-loss point into a degenerate, business-nonsensical result. The roadmap for what
would actually resolve the identifiability problem (longitudinal data, a
repeated-measures likelihood) is documented in
`src/physics.py::estimate_params_discussion()`, not silently deferred.

**Verify:** `src/physics_calibration.py` (`PARAM_BOUNDS`, inline comment on γ),
`src/physics.py::estimate_params_discussion`.

---

### 8. Optimize the campaign threshold for ROI, not F1 or the 0.50 default

**Decision:** Derive the decision threshold directly from a business objective,
`n_retained x CLV x horizon − n_intervened x cost`, maximized over the threshold grid
on the *calibrated* probability, rather than defaulting to 0.50 or tuning for F1.

**Why:** Every standard ML workflow either fixes the threshold at 0.50 or tunes it for
a statistical metric that has no connection to what the business actually pays for a
false positive ($25 wasted) versus what it loses on a false negative (a customer's
CLV). Once those costs are known, F1 and 0.50 are both arbitrary. At 0.50, the model
flags 1,732 customers and catches 58% of actual churners; at the ROI-optimal 0.20, it
flags 3,253 and catches 84%, a ~$23K difference in net ROI, from a single config value,
with no retraining required.

**Trade-off accepted:** The ROI number is only as good as the cost/CLV assumptions
feeding it (see #9). Optimizing for a business objective makes the result more
useful, but also more exposed to whichever assumption is weakest.

**Verify:** `src/business.py` (threshold optimization), `pipeline/config.py`
(overridable CLV / cost / horizon constants), `src/ab_test.py` (the validation design
for the weakest assumption).

---

### 9. Treat `SUCCESS_RATE = 30%` as the single assumption most worth validating

**Decision:** Explicitly flag the campaign success-rate assumption (the fraction of
flagged churners who actually stay after intervention) as the one number in the entire
pipeline that is *not* derived from or validated against this dataset, and design (but
not fabricate the results of) an A/B test to validate it.

**Why:** Every other number in the ROI calculation, physics coefficients, probability
calibration, threshold optimization, segmentation, is now either fit or cross-validated
against actual outcomes in the data. `SUCCESS_RATE` can't be, because the dataset has
no record of customers who were actually targeted with a retention campaign. Presenting
it with the same confidence as the validated numbers would misrepresent how much of the
$132,055 headline figure is evidence-backed versus assumed.

**Trade-off accepted:** This means the single most important number for a go/no-go
budget decision is explicitly the least certain one in the report, an uncomfortable
thing to lead with, but the honest one. The A/B test design in `src/ab_test.py`
(sample size, sequential monitoring, break-even analysis) is what would close that gap;
the report includes a simulated run of that design on synthetic data to demonstrate the
methodology, clearly labeled as simulated, not as evidence the 30% figure is correct.

**Verify:** `src/ab_test.py`, `report/impact_report.json` field `ab_test_validation.note`.

---

### 10. Make probability calibration the default, not an opt-in step

**Decision:** Isotonic calibration is applied to the raw model score by default, inside
the scoring function itself, not as a separate step a caller can forget to invoke.

**Why:** During development, calibration was first added in one code path, and the
uncalibrated raw score was later found reappearing in other code paths (CLI, dashboard)
that had been added afterward without anyone realizing the calibration step needed to
be duplicated there. That's not a hypothetical risk; it happened, and the fix was to
make correctness the default rather than something every future call site has to
remember to opt into.

**Trade-off accepted:** None, this is a straightforward "make the safe path also the
easy path" fix, once the bug that motivated it had been found and understood.

**Verify:** `src/uncertainty.py` (calibration function); grep for calls to the raw
scoring function across `pipeline/`, `dashboard/`, and `main.py`. There should be none
that bypass calibration.

---

### 11. Give test files module-matching names, and make discovery explicit via `pytest.ini`

**Decision:** Test files are named after the module they cover (`physics.py`,
`business.py`, ...) rather than pytest's default `test_*.py` convention, with
`pytest.ini` making discovery explicit so this doesn't silently break.

**Why it needed a fix:** The module-matching naming was already in place for
readability: it's obvious which test file covers which module without opening either.
But a plain `pytest` invocation was silently collecting only 18 of 349 tests under that
naming scheme, because pytest's default discovery pattern doesn't match it. That's a
worse failure mode than an import error: the test suite reports green while covering a
fraction of itself.

**Trade-off accepted:** Anyone extending the test suite needs to know the naming
convention is deliberate (module name, not `test_` prefix) and that `pytest.ini` is
what makes it discoverable, which isn't obvious from a first glance at the directory.

**Verify:** `pytest.ini`, and `pytest tests/ -v` reporting 367 collected tests.

---

### 12. Use each customer's own `MonthlyCharges` as CLV, instead of segmenting by contract type

**Decision:** Replace the flat `CLV_MONTHLY = $65/month` assumption in
`compute_economic_impact` and `optimize_threshold` with each customer's actual
`MonthlyCharges`, rather than a CLV segmented by contract type (the originally-planned
fix; see the old "Open items" entry this replaces).

**Why the originally-planned fix (segment by contract type) was rejected first:**
Before implementing it, I checked how much of the real variance in what customers pay
is actually explained by contract type: very little. `MonthlyCharges` varies far more
*within* a contract type (std ≈ 27) than *between* contract-type averages (std ≈ 2.9
across Month-to-month/One year/Two year). Segmenting by contract would have captured
almost none of the real signal, the same mistake as reporting a flat average, just
with three flat averages instead of one.

**Why per-customer `MonthlyCharges` instead:** It's already an observed column, so
using it directly requires no new assumption and captures all the real variance a
contract-type segmentation would have mostly missed. `compute_economic_impact` now uses
`df['MonthlyCharges']` per row when present, falling back to the flat assumption
otherwise (so every existing call site and test fixture without that column keeps
working unchanged, this is a strict generalization, not a breaking change).
`optimize_threshold` gained an optional `clv_per_customer` array parameter with the same
fallback behavior.

**What this changed:** The customers the model actually flags for HIGH risk skew
toward higher-paying segments than the dataset average, so using their real CLV instead
of the flat average raised the recommended-threshold net ROI. Combined with the
PaymentMethod split below, the recommended-threshold net ROI moved from $101,483 to
$132,055.

**Trade-off accepted:** `optimize_threshold`'s reported "assumptions" dict still shows
`clv_monthly` as a scalar even when `clv_per_customer` was actually used for the
calculation. A caller reading only that dict without checking whether
`clv_per_customer` was passed could be misled about what CLV was actually applied. Worth
fixing if this function's output is ever surfaced directly to a non-technical audience.

**Verify:** `src/business.py::compute_economic_impact` and `::optimize_threshold`,
`tests/business.py::test_uses_per_customer_monthly_charges_when_present` and
`::test_high_paying_true_positives_increase_revenue_recovered`,
`tests/business_advanced.py::test_uniform_clv_per_customer_matches_flat_scalar`.

---

### 13. Split `PaymentMethod` into three categories, not just the existing `auto_payment` binary

**Decision:** Replace the single `auto_payment` bonus in E_eq/gamma with three
categories, auto, mailed check, electronic check, each with its own fitted
coefficient, subject to an `auto >= mailed >= electronic` ordering constraint. This is
the item flagged (not assumed) in the original "Open items" list, resolved after
checking the evidence first.

**Why the binary wasn't good enough:** `auto_payment` (automatic vs. not) already
captures most of the churn-rate gap (34.7% non-auto vs. 16.0% auto). But within
"non-auto," Electronic check (45.3% churn) and Mailed check (19.1%) aren't close to
each other, a ~26-point gap the binary was averaging away, the same mistake made with
services (protective vs. entertainment) and InternetService before their splits. The
gap survives controlling for Contract (e.g. within Month-to-month alone: 53.7%
Electronic check vs. 31.6% Mailed check vs. 32-34% for the two automatic methods), so
it isn't just a Month-to-month proxy, the same evidence bar the other two splits had
to clear.

**What the fit found:** `eeq_payment_electronic` converges to 0.0283, close to zero,
the same pattern as `eeq_entertainment_coef` and `eeq_internet_fiber` before it,
independently confirming Electronic check carries almost no equilibrium-energy signal
beyond baseline. `eeq_payment_auto` (0.20) and `eeq_payment_mailed` (0.18) fit close
together, consistent with the raw churn-rate gap between those two being much smaller
than either is from Electronic check. Checking `check_constraints_binding` after the
fit confirms `eeq_payment_mailed_ge_electronic` is *not* binding (slack of ~0.15): the
gap between mailed and electronic is a real finding the optimizer discovered, not an
artifact of the constraint forcing them apart at the boundary.

**What this moved:** Out-of-fold AUC of the raw formula, 0.832 to 0.837 (a smaller jump
than the two earlier splits, consistent with the binary already having captured most of
the signal). Combined with the CLV change above, recommended-threshold net ROI moved
from $101,483 to $132,055.

**A side effect worth naming rather than leaving unexplained:** the count of customers
classified `UNCERTAIN` (sigma_prob above the fixed 0.025 threshold) rose from 87 to 360
after this change. This is not a new bug: `BASE_UNCERTAINTIES` for gamma/E_eq are
fixed constants, unrelated to the payment coefficients. It's a side effect of the
probability distribution shifting slightly as AUC improved: more customers ended up
with `sigma_prob` just above the fixed 0.025 cutoff. Worth knowing if this number is
reported without context, since on its own it can look like a regression.

**Trade-off accepted:** 29 to 33 parameters. `compute_E_eq`/`compute_damping` in
`src/physics.py` kept their original `auto_payment: bool` positional argument for
backward compatibility with existing call sites (`app.py`, `dashboard/tabs/tab4.py`),
adding an optional `payment_method` argument that takes priority when provided; when
only the boolean is passed, `False` falls back to the `mailed_check` coefficient (the
more common and less extreme non-auto category), which undercounts Electronic check's
effect for any caller that hasn't been updated to pass `payment_method` explicitly.
Both production call sites were updated to pass it; the fallback exists only for
robustness, not as the intended path.

**Verify:** `src/physics_calibration.py` (`PARAM_NAMES`, `_constraint_funcs`,
`RawInputs.from_df`), `src/features.py::build_physical_features`, `src/physics.py`
(`compute_E_eq`, `compute_damping`), `notebooks/07_final_model.ipynb` (the new
PaymentMethod section, added after the InternetService section), `tests/physics.py`
(`test_payment_method_three_way_ordering_gamma`, `test_payment_method_three_way_ordering_E_eq`).

---

---

### 14. Checked the Contract x InternetService interaction, and didn't add it

**Decision:** Do not add a Contract x InternetService interaction term to the physics
formula. This was the largest item left in "Open items," and it's resolved by checking
the evidence and finding it doesn't support the change, not by running out of time to
look at it.

**Why this needed a different evidence bar than the splits in entries #4, #5, and #13:**
Those splits replaced one coefficient with several *independent* categorical terms
(services, internet type, payment method), each one adds parameters but keeps the
model additive: `E_eq = base(contract) + bonus(services) + bonus(internet) +
bonus(payment)`. An interaction term is a different kind of complexity: it asks
whether the *combination* of contract and internet type behaves differently than the
sum of their individual effects, and with only 9 cells in this dataset's Contract x
InternetService grid, some genuinely small (One year x Fiber optic: n=539; Two year x
No internet: n=638), that's a much easier hypothesis to fit noise to.

**What the check found:** A logistic regression with `Contract * InternetService`
(9 parameters) was compared against one with `Contract + InternetService` (5
parameters, no interaction) via a likelihood ratio test. The interaction model's AIC is
*worse* (6361.11 vs. 6358.99 for the additive model) and the likelihood ratio test gives
p=0.209, not significant at any conventional threshold. Checking additivity directly
on the log-odds scale confirms it: the gap between Month-to-month and Two-year churn
log-odds is 3.19 for DSL, 2.74 for Fiber optic, and 3.38 for "No internet", close to
constant across all three internet types, which is exactly what an additive model
(no interaction) predicts. The existing formula's design, `bonus(contract)` and
`bonus(internet)` added independently in E_eq/gamma, already captures this
correctly.

**Trade-off accepted:** None, this is the version without a trade-off. Adding 4
more parameters for a change that fails its own significance test would have been
complexity without benefit, exactly the failure mode entries #4-#6 and #13 were
careful to avoid.

**Verify:** `notebooks/07_final_model.ipynb`, the "A fourth check: does Contract need
to interact with InternetService?" section. A re-runnable comparison using
`sklearn.linear_model.LogisticRegression` (unregularized, `C=1e10`) and
`scipy.stats.chi2` for the likelihood ratio test, both already project dependencies (no
new package added just for this check).

---

## Open items (deliberately not resolved yet, see README -> Next steps)

There are currently no open items being deliberately deferred. The two carried over
from the previous revision are resolved:

- CLV segmentation (see entry #12) and the PaymentMethod split (see entry #13) are
  both implemented and validated.
- `optimize_threshold`'s `assumptions` dict now reports `clv_source` and
  `avg_clv_used`, making explicit whether the flat `clv_monthly` scalar or a
  per-customer CLV array actually drove the result; `threshold_summary` surfaces the
  same distinction in its human-readable output.
- The Contract x InternetService interaction (see entry #14) was checked and found not
  to earn its complexity. This is a closed question with a negative result, not an
  open one.

If and when a genuinely new open item arises (e.g. from the "Audit the remaining
categorical variables" note in the README), it belongs here, stated plainly, the same
way the items above were before they were resolved.
