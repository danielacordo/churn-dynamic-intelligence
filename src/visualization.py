"""Re-export facade for the plotting layer.

Callers (dashboard tabs, notebooks, tests) import from here without needing
to know whether a given plot function lives in visualization_plotly.py
(interactive/Plotly) or visualization_static.py (Matplotlib/Seaborn). The
names below are this module's public API, not unused imports.
"""

from src.visualization_plotly import (  
    plot_customer_funnel,
    plot_heatmap_annotated,
    plot_phase_space_plotly,
    plot_trajectory_plotly,
    plot_waterfall_roi,)

from src.visualization_static import (  
    _DARK_LAYOUT,
    PALETTE,
    plot_phase_space,
    plot_phase_space_v1,
    plot_physical_states,
    plot_prediction_uncertainty,
    plot_relaxation,
    plot_trajectory,
    plot_trajectory_v1,
    set_style,
    state_distribution,
    state_distribution_v1,)
