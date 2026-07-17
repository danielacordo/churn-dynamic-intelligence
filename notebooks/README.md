# Notebooks

This directory is the analytical narrative behind `src/` and `pipeline/`. Each notebook
is a checkpoint in how the model was actually built, in the order it was built. They are
meant to be read, not just run: some contain a "bug found in review" or "scope note"
callout where an earlier assumption turned out to be wrong, and the notebook documents
what was found and what changed, rather than silently presenting the corrected version.

Run `bash run_all_notebooks.sh` to re-execute all of them in order against a fresh copy
of the dataset (see `../download_data.py`). Outputs are already saved in each `.ipynb`,
so opening them directly in Jupyter or GitHub's preview does not require rerunning anything.

## Reading order

| # | Notebook | What it does |
|---|----------|---------------|
| 01 | `01_eda.ipynb` | Exploratory analysis and cleaning of the raw IBM Telco dataset (7,043 customers): the starting point everything downstream depends on. |
| 02 | `02_physical_variables.ipynb` | Translates business variables (tenure, contract, billing, services) into the physical state space the dynamic model operates in: where E0, E_eq, and γ come from before any fitting happens. |
| 03 | `03_dynamic_model.ipynb` | Sets up the governing differential equation `dE/dt = -γ·E + F(t)` and its analytical solution `E(t) = E0 · e^(-γt)`, and fits the damping coefficient γ against real data. |
| 04 | `04_phase_transitions.ipynb` | Tests the "customers don't gradually decide to leave, they cross a threshold" hypothesis via a simulated perturbation. Includes an explicit scope note: the simulation's 9.7% threshold-crossing rate is well below the actual 26.5% churn rate, because it applies one fixed synthetic shock to everyone rather than modeling each customer's real perturbation history, a limitation stated plainly, not glossed over. |
| 05 | `05_relaxation_time.ipynb` | Breaks down τ = 1/γ by customer segment: small τ (fast recovery or fast collapse) vs large τ (slow-responding, either loyal or quietly accumulating risk). |
| 06 | `06_error_propagation.ipynb` | Applies formal uncertainty propagation to the model's predictions, so risk scores come with real confidence intervals rather than a bare point estimate. This is also where a real calibration bug was caught and fixed: the raw physics-based `prob_churn = 1 - E(t=6m)` was systematically overconfident (roughly 2x the actual churn rate), which had been pushing most customers into `risk_level='HIGH'`. The notebook documents the bug, not just the fix. |
| 07 | `07_final_model.ipynb` | Integrates every component (E0, E_eq, γ, calibration) into the final pipeline and benchmarks it against a Logistic Regression baseline. This is also where the SLSQP coefficient fitting lives, including the three model-improving splits (protective vs. entertainment services, internet connection type, and payment method) derived directly from the data in-notebook before being adopted into `src/physics_calibration.py`. It also includes a fourth check, with a negative result: whether Contract needs to interact with InternetService (it doesn't; an AIC and likelihood-ratio comparison rejects the interaction). |
| 08 | `08_survival_analysis.ipynb` | Cross-checks the physical model's τ = 1/γ against Kaplan-Meier mean survival time: an independent, non-physics estimate of the same underlying quantity, used as a sanity check rather than taken on faith. |
| 09 | `09_baseline_comparison.ipynb` | The "is the physical complexity worth it?" notebook. Compares four models (raw-feature Logistic Regression, physical-parameter Logistic Regression, a GBM on physical + raw features, and XGBoost on raw features) on both technical metrics (AUC-ROC, recall, 5-fold CV) and business metrics (net ROI per campaign cycle on a 25% holdout). |
| 10 | `10_business_decisions.ipynb` | Translates E0/E_eq/γ/τ into concrete, interpretable actions for a retention team, e.g. distinguishing a structurally loyal customer from a structurally at-risk one, which calls for different interventions even at the same churn probability. |
| 11 | `11_impact_metrics.ipynb` | Quantifies what the model is actually worth: early-detection window (days gained vs. reactive detection), retention gain, and the ROI figures that feed `report/impact_report.json`. |
| — | `sql_analysis_executed.ipynb` | A DuckDB-based SQL layer answering the upstream business question the physics model doesn't: not who will churn, but where the problem is concentrated and which segments to prioritize first. Runs in-process against the same data, no server required. |

## A note on how to read these

The narrative deliberately keeps its own false starts visible. Notebook 04's scope-limited simulation, notebook 06's calibration bug, and notebook 07's mid-analysis discovery that the original E0 formula was saturated for 60.7% of customers (see `../DECISIONS.md` #3) are all left in, with the reasoning that led to each fix. If you're reviewing this project for technical rigor, those are the sections worth reading closely:
they show the actual analytical process, not a retroactively cleaned-up version of it.
