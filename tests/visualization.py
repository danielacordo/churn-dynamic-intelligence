import matplotlib
matplotlib.use("Agg") 
import os
import sys
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.visualization import (
    PALETTE, plot_phase_space, plot_phase_space_v1,
    plot_physical_states, plot_prediction_uncertainty,
    plot_relaxation, plot_trajectory,
    plot_trajectory_v1, set_style,
    state_distribution, state_distribution_v1,)

# Fixtures
@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after every test to avoid memory leaks"""

    yield
    plt.close("all")


def _mock_ode_result(n=40):
    """Returns a SimpleNamespace mimicking scipy OdeResult with y[0] and t """

    t = np.linspace(0, 12, n)
    E = 0.3 + (0.7 - 0.3) * np.exp(-0.1 * t)
    sol = SimpleNamespace(t=t, y=np.array([E]))
    return sol


def _phase_df(n=60):
    """Minimal DataFrame for plot_phase_space"""

    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "E0": rng.uniform(0.1, 0.9, n),
        "E_eq": rng.uniform(0.1, 0.9, n),
        "Churn_bin": rng.integers(0, 2, n),})


def _state_df(n=80):
    """DataFrame with physical_state, risk_level, tau, Contract, prob_churn, sigma_prob."""
    
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "physical_state": rng.choice(["Stable", "At risk", "Critical"], n),
        "risk_level": rng.choice(["HIGH", "MEDIUM", "LOW", "UNCERTAIN"], n),
        "tau": rng.uniform(1, 30, n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "prob_churn": rng.uniform(0, 1, n),
        "sigma_prob": rng.uniform(0.02, 0.15, n),})


# PALETTE 
class TestPalette:

    def test_palette_has_required_keys(self):
        for key in ("churn", "no_churn", "accent", "bg", "panel", "text", "border", "neutral"):
            assert key in PALETTE, f"Missing palette key: {key}"

    def test_all_values_are_hex_strings(self):
        for key, value in PALETTE.items():
            assert isinstance(value, str), f"{key} is not a string"
            assert value.startswith("#"), f"{key} value {value!r} is not a hex colour"


# set_style
class TestSetStyle:

    def test_set_style_runs_without_error(self):
        set_style()   

    def test_set_style_updates_facecolor(self):
        set_style()
        assert plt.rcParams["figure.facecolor"] == PALETTE["bg"]

    def test_set_style_enables_grid(self):
        set_style()
        assert plt.rcParams["axes.grid"] is True


# plot_trajectory
class TestPlotTrajectory:

    def test_returns_axes_object(self):
        sol = _mock_ode_result()
        ax = plot_trajectory(sol)
        assert isinstance(ax, plt.Axes)

    def test_uses_provided_axes(self):
        sol = _mock_ode_result()
        fig, ax_in = plt.subplots()
        ax_out = plot_trajectory(sol, ax=ax_in)
        assert ax_out is ax_in

    def test_creates_new_axes_when_none(self):
        sol = _mock_ode_result()
        ax = plot_trajectory(sol, ax=None)
        assert ax is not None

    def test_y_axis_label(self):
        sol = _mock_ode_result()
        ax = plot_trajectory(sol)
        assert "Energy" in ax.get_ylabel() or "E" in ax.get_ylabel()

    def test_x_axis_label(self):
        sol = _mock_ode_result()
        ax = plot_trajectory(sol)
        assert "month" in ax.get_xlabel().lower() or "time" in ax.get_xlabel().lower()

    def test_y_limits_are_0_to_1(self):
        sol = _mock_ode_result()
        ax = plot_trajectory(sol)
        ylo, yhi = ax.get_ylim()
        assert ylo <= 0.0
        assert yhi >= 0.95

    def test_E_critical_line_drawn(self):
        sol = _mock_ode_result()
        ax = plot_trajectory(sol, E_critical=0.25)
        hlines = [ln for ln in ax.get_lines() if len(set(ln.get_ydata())) == 1]
        yvals = {round(ln.get_ydata()[0], 4) for ln in hlines}
        assert any(abs(y - 0.25) < 1e-6 for y in yvals), \
            f"E_critical=0.25 not found in hlines; found: {yvals}"

    def test_E_eq_line_drawn_when_provided(self):
        sol = _mock_ode_result()
        ax  = plot_trajectory(sol, E_eq=0.55)
        hlines = [ln for ln in ax.get_lines() if len(set(ln.get_ydata())) == 1]
        yvals  = {round(ln.get_ydata()[0], 4) for ln in hlines}
        assert any(abs(y - 0.55) < 1e-4 for y in yvals), \
            f"E_eq=0.55 not found in hlines; found: {yvals}"

    def test_trajectory_line_data_matches_sol(self):
        sol = _mock_ode_result()
        ax = plot_trajectory(sol)
        # First line with multiple y values is the trajectory
        traj_lines = [ln for ln in ax.get_lines() if len(set(ln.get_ydata())) > 1]
        assert len(traj_lines) >= 1
        np.testing.assert_array_almost_equal(traj_lines[0].get_ydata(), sol.y[0])

    def test_alias_plot_trajectory_v1_same_result(self):
        sol = _mock_ode_result()
        ax1 = plot_trajectory(sol)
        plt.close("all")
        ax2 = plot_trajectory_v1(sol)
        assert type(ax1) is type(ax2)


# plot_phase_space 
class TestPlotPhaseSpace:
    def test_returns_axes(self):
        df = _phase_df()
        ax = plot_phase_space(df)
        assert isinstance(ax, plt.Axes)

    def test_uses_provided_axes(self):
        df = _phase_df()
        fig, ax_in = plt.subplots()
        ax_out = plot_phase_space(df, ax=ax_in)
        assert ax_out is ax_in

    def test_x_axis_label_contains_E_eq(self):
        df = _phase_df()
        ax = plot_phase_space(df)
        assert "E_eq" in ax.get_xlabel() or "eq" in ax.get_xlabel().lower()

    def test_y_axis_label_contains_E0(self):
        df = _phase_df()
        ax = plot_phase_space(df)
        assert "E0" in ax.get_ylabel() or "current" in ax.get_ylabel().lower()

    def test_xlim_and_ylim_0_to_1(self):
        df = _phase_df()
        ax = plot_phase_space(df)
        assert ax.get_xlim()[0] <= 0 and ax.get_xlim()[1] >= 1
        assert ax.get_ylim()[0] <= 0 and ax.get_ylim()[1] >= 1

    def test_two_scatter_collections(self):
        """One collection for churners, one for non-churners"""
        df = _phase_df(n=80)
        ax = plot_phase_space(df)
        # scatter creates PathCollection objects
        collections = ax.collections
        assert len(collections) >= 2

    def test_alias_plot_phase_space_v1(self):
        df  = _phase_df()
        _ = plot_phase_space(df)
        plt.close("all")
        ax2 = plot_phase_space_v1(df)
        assert isinstance(ax2, plt.Axes)


# state_distribution 
class TestStateDistribution:

    def test_returns_axes_with_physical_state(self):
        df = _state_df()
        ax = state_distribution(df)
        assert isinstance(ax, plt.Axes)

    def test_returns_axes_with_risk_level(self):
        df = _state_df().drop(columns=["physical_state"])
        ax = state_distribution(df)
        assert isinstance(ax, plt.Axes)

    def test_falls_back_gracefully_with_no_state_col(self):
        df = pd.DataFrame({"prob_churn": [0.3, 0.5, 0.7]})
        ax = state_distribution(df)
        assert isinstance(ax, plt.Axes)   # should not raise

    def test_uses_provided_axes(self):
        df = _state_df()
        fig, ax_in = plt.subplots()
        ax_out = state_distribution(df, ax=ax_in)
        assert ax_out is ax_in

    def test_alias_state_distribution_v1(self):
        df = _state_df()
        _ = state_distribution(df)
        plt.close("all")
        ax2 = state_distribution_v1(df)
        assert isinstance(ax2, plt.Axes)

    def test_bar_patches_present(self):
        df = _state_df()
        ax = state_distribution(df)
        assert len(ax.patches) > 0


# plot_relaxation 
class TestPlotRelajacion:

    def test_returns_axes(self):
        df = _state_df()
        ax = plot_relaxation(df)
        assert isinstance(ax, plt.Axes)

    def test_x_label_contains_tau(self):
        df = _state_df()
        ax = plot_relaxation(df)
        assert "τ" in ax.get_xlabel() or "tau" in ax.get_xlabel().lower()

    def test_kde_lines_present(self):
        """plot_relaxation now uses KDE curves (lines) not histogram patches """
        df = _state_df()
        ax = plot_relaxation(df)
        # KDE version draws Line2D objects, not patches
        data_lines = [ln for ln in ax.get_lines() if len(set(ln.get_ydata())) > 1]
        assert len(data_lines) > 0, "Expected at least one KDE curve line"

    def test_fill_between_present(self):
        """KDE fill_between creates a PolyCollection """
        df = _state_df()
        ax = plot_relaxation(df)
        from matplotlib.collections import PolyCollection
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(polys) > 0, "Expected fill_between PolyCollection under KDE curve"

    def test_works_without_contract_column(self):
        df = _state_df().drop(columns=["Contract"])
        ax = plot_relaxation(df)   # should not raise
        assert isinstance(ax, plt.Axes)


# plot_physical_states 
class TestPlotPhysicalStates:

    def test_returns_axes(self):
        df = _state_df()
        ax = plot_physical_states(df)
        assert isinstance(ax, plt.Axes)

    def test_patches_present(self):
        df = _state_df()
        ax = plot_physical_states(df)
        assert len(ax.patches) > 0

    def test_y_label(self):
        df = _state_df()
        ax = plot_physical_states(df)
        assert "customer" in ax.get_ylabel().lower()


# plot_prediction_uncertainty 
class TestPlotPredictionUncertainty:

    def test_returns_axes(self):
        df = _state_df()
        ax = plot_prediction_uncertainty(df)
        assert isinstance(ax, plt.Axes)

    def test_x_label_contains_prob(self):
        df = _state_df()
        ax = plot_prediction_uncertainty(df)
        assert "P" in ax.get_xlabel() or "prob" in ax.get_xlabel().lower() or "churn" in ax.get_xlabel().lower()

    def test_y_label_contains_sigma(self):
        df = _state_df()
        ax = plot_prediction_uncertainty(df)
        assert "σ" in ax.get_ylabel() or "sigma" in ax.get_ylabel().lower() or "uncertainty" in ax.get_ylabel().lower()

    def test_scatter_collections_present(self):
        df = _state_df()
        ax = plot_prediction_uncertainty(df)
        assert len(ax.collections) >= 1
